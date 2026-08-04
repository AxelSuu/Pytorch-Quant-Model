"""``train()``: fit a TFT and persist it as a bundle.

Split out of ``models/tft.py`` (PYQ-269). Depends one-directionally on
``models/backtest.py`` for window geometry (``_selection_split``) and the
shared checkpoint/evaluation helpers -- see that module's docstring for why
they live there rather than here. See ``models/tft.py`` for the compatibility
re-export surface.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet

from pyquant.analysis.calibrate import fit_conformal_offset
from pyquant.config import Settings
from pyquant.data.dataset import (
    align_time_index,
    build_panel,
    feature_columns,
    make_dataset,
    panel_to_long,
    target_column,
)
from pyquant.models.backtest import (
    _evaluate_validation,
    _load_best_checkpoint,
    _raw_validation_arrays,
    _selection_split,
    build_model,
)
from pyquant.models.bundle import TrainResult, _bundle_dir, _provenance

logger = logging.getLogger(__name__)


def _build_pooled_long_df(
    symbols: list[str],
    settings: Settings,
    start: str | None,
    end: str | None,
    pin: str | None = None,
) -> pd.DataFrame:
    """Fetch + join each symbol's panel, then pool into one long df.

    TimeSeriesDataSet is built with group_ids=["symbol"], so a single dataset
    can span multiple tickers. Their time_idx is re-mapped onto a single shared
    calendar by align_time_index() so pooled groups line up by *date* rather
    than by row position (PYQ-116). A source that's flaky for only some symbols
    (see PYQ-302) would otherwise inject NaNs into the pooled frame; instead we
    keep only the columns common to every symbol's panel and log what got
    dropped.
    """
    frames = [
        panel_to_long(build_panel(symbol, settings, start, end, pin=pin), symbol)
        for symbol in symbols
    ]
    if len(frames) == 1:
        return align_time_index(frames[0])

    # A sector-ETF symbol has its own SEC_<symbol> column deliberately dropped to
    # avoid self-leakage (PYQ-102). Left as-is, the column-intersection below
    # would then strip that (perfectly valid) feature from *every other* pooled
    # symbol too (PYQ-111). Re-add it as neutral 0.0 for just the symbol that
    # leaks, so it survives the intersection for the others -- while genuinely
    # missing columns (a source flaky for some symbols, PYQ-302) still get
    # dropped as intended.
    for frame, symbol in zip(frames, symbols, strict=True):
        self_col = f"SEC_{symbol}"
        if self_col not in frame.columns and any(self_col in f.columns for f in frames):
            frame[self_col] = 0.0

    common_cols = set.intersection(*(set(f.columns) for f in frames))
    all_cols = set.union(*(set(f.columns) for f in frames))
    dropped = sorted(all_cols - common_cols)
    if dropped:
        logger.warning(
            "Pooling %s: dropping columns not common to every symbol's panel: %s",
            symbols,
            dropped,
        )
    ordered_cols = [c for c in frames[0].columns if c in common_cols]
    return align_time_index(pd.concat([f[ordered_cols] for f in frames], ignore_index=True))


def _warn_on_stale_symbols(df: pd.DataFrame, cutoff: int) -> list[str]:
    """Warn about symbols with no data after the training cutoff.

    align_time_index() puts every pooled symbol on one calendar, which fixes the
    late-*start* case that made a short-history symbol's validation window fall
    inside the training slice (PYQ-116). A symbol whose data *stops* early -- a
    delisting, or a stale feed -- still has that problem, and nothing in the
    dataset machinery notices. Name it rather than quietly reporting an
    optimistic val_loss. Returns the offending symbols.
    """
    last_per_symbol = df.groupby("symbol")["time_idx"].max()
    stale = sorted(str(s) for s in last_per_symbol[last_per_symbol <= cutoff].index)
    if stale:
        logger.warning(
            "Symbol(s) %s have no data after the training cutoff (time_idx %d): their "
            "validation window overlaps the training slice, so the reported val_loss "
            "and metrics are optimistic for them. Drop them from the pool or narrow "
            "the date range so every symbol ends together.",
            ", ".join(stale),
            cutoff,
        )
    return stale


def train(
    symbols: str | list[str],
    settings: Settings,
    *,
    bundle_name: str | None = None,
    start: str | None = None,
    end: str | None = None,
    max_epochs: int | None = None,
    progress: bool = True,
    pin: str | None = None,
) -> TrainResult:
    """Train a TFT for ``symbols`` and persist the bundle.

    A single symbol trains a per-ticker model as before. Multiple symbols are
    pooled into one TimeSeriesDataSet/model (group_ids=["symbol"]), giving the
    same architecture meaningfully more training data; pass ``bundle_name`` to
    control the resulting checkpoint directory name (defaults to the symbol,
    or the joined symbol list when pooling). ``pin`` names a reproducible
    dataset snapshot (see pyquant.data.cache) so the same experiment can be
    re-run later against identical data.
    """
    symbols = [symbols] if isinstance(symbols, str) else list(symbols)
    symbols = [s.upper() for s in symbols]
    bundle_name = (bundle_name or "_".join(symbols)).upper()

    # Seed before any data loading / weight init so a run is reproducible and
    # the recorded seed (below) actually reconstructs it (PYQ-210).
    seed_everything(settings.training.seed, workers=True)

    df = _build_pooled_long_df(symbols, settings, start, end, pin=pin)

    horizon = settings.training.max_prediction_length
    encoder_len = settings.training.max_encoder_length
    # The holdout spans validation_days, not a single horizon, so many windows are
    # scored instead of one (PYQ-117). Never shorter than one horizon, or there
    # would be no complete validation window at all.
    validation_days = max(settings.training.validation_days, horizon)
    calibration_days = max(0, settings.training.calibration_days)
    selection_days = max(1, settings.training.selection_days)
    max_idx = int(df["time_idx"].max())
    # Geometry, left to right:
    #   [ training .. train_cutoff ][ purge+embargo ][ selection ]
    #   [ purge+embargo ][ calibration ][ validation ]
    # `cutoff` is the last index before the *held-out* region; the calibration
    # slice (PYQ-248) sits between it and the scored validation ("test") window
    # so the conformal offset is fitted on data that is out-of-sample for
    # training and disjoint from what it is later judged on. `selection` sits
    # earlier still, purged from both training and the calibration+validation
    # region: EarlyStopping/ModelCheckpoint select the checkpoint against it,
    # never against the window `EvaluationMetrics` is reported from (PYQ-143).
    validation_start = max_idx - validation_days + 1
    cutoff = validation_start - calibration_days - 1
    train_cutoff, selection_start, selection_end = _selection_split(cutoff, settings)
    if train_cutoff <= encoder_len:
        raise ValueError(
            f"Not enough history for {bundle_name}: need more than "
            f"{encoder_len + validation_days + calibration_days + selection_days + 2 * (cutoff - selection_end)} "
            f"rows (a {encoder_len}-day encoder, a {selection_days}-day selection window, a "
            f"{validation_days}-day validation holdout, {calibration_days} calibration day(s) and "
            f"two {cutoff - selection_end}-day purged/embargoed gaps -- one around selection, one "
            f"around training), got {len(df)}."
        )

    # Staleness is about whether a symbol has any data left to *validate* on, so
    # it is measured against the start of the scored window -- not against the
    # purged training cutoff, which sits earlier and would let a symbol whose
    # data stops inside the purge gap pass unflagged.
    _warn_on_stale_symbols(df, validation_start - 1)

    training = make_dataset(df, settings, training_cutoff=train_cutoff)
    # Every window whose decoder starts after the cutoff, rather than predict=True's
    # single last window -- this is what gives the metrics and early stopping a
    # usable sample size (PYQ-117).
    validation = TimeSeriesDataSet.from_dataset(
        training, df, min_prediction_idx=validation_start, stop_randomization=True
    )
    # Bounded above at selection_end so no selection window can reach into the
    # purge gap or the calibration/test region beyond it (PYQ-143).
    selection_df = df[df["time_idx"] <= selection_end + horizon - 1]
    selection = TimeSeriesDataSet.from_dataset(
        training, selection_df, min_prediction_idx=selection_start, stop_randomization=True
    )

    batch_size = settings.training.batch_size
    num_workers = settings.training.num_workers
    train_loader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=num_workers
    )
    val_loader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=num_workers
    )
    selection_loader = selection.to_dataloader(
        train=False, batch_size=batch_size, num_workers=num_workers
    )

    model = build_model(training, settings)

    bundle_dir = _bundle_dir(settings, bundle_name)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    epochs = max_epochs if max_epochs is not None else settings.training.max_epochs
    ckpt_cb = ModelCheckpoint(
        dirpath=bundle_dir, filename="model", monitor="val_loss", save_top_k=1, mode="min"
    )
    early_stop = EarlyStopping(
        monitor="val_loss", patience=settings.training.early_stopping_patience, mode="min"
    )

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        precision=settings.training.precision,
        gradient_clip_val=settings.training.gradient_clip_val,
        callbacks=[ckpt_cb, early_stop],
        logger=False,
        enable_progress_bar=progress,
        enable_model_summary=False,
    )
    # val_dataloaders drives EarlyStopping/ModelCheckpoint -- selection_loader,
    # never val_loader, so checkpoint choice is not a selection event scored
    # against the same window the reported metrics come from (PYQ-143).
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=selection_loader)

    # ModelCheckpoint may version the filename; load the actual best path.
    best_path = ckpt_cb.best_model_path or str(bundle_dir / "model.ckpt")
    best_path = Path(best_path)
    if best_path != bundle_dir / "model.ckpt" and best_path.exists():
        best_path.replace(bundle_dir / "model.ckpt")

    torch.save(training.get_parameters(), bundle_dir / "dataset_params.pt")

    val_loss = (
        float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else float("nan")
    )
    # Evaluate the *best* checkpoint (the one saved to model.ckpt and actually
    # loaded by forecast/explain), not the live post-fit model -- EarlyStopping
    # stops training several epochs past the best one without rewinding the live
    # weights, so reporting on `model` measures a worse, already-discarded
    # checkpoint than the one that gets deployed (PYQ-109).
    best_model = _load_best_checkpoint(bundle_dir / "model.ckpt", model)
    target = target_column(settings)

    # Fit the conformal offset on the calibration slice -- disjoint from both the
    # training data and the validation window the metrics come from, so the
    # widening/narrowing it implies is not read off the same points it is later
    # judged on (PYQ-248).
    conformal = None
    if calibration_days > 0:
        calibration = TimeSeriesDataSet.from_dataset(
            training,
            df[df["time_idx"] < validation_start],
            min_prediction_idx=cutoff + 1,
            stop_randomization=True,
        )
        cal_loader = calibration.to_dataloader(
            train=False, batch_size=batch_size, num_workers=num_workers
        )
        cal_pred, cal_actual, _, _ = _raw_validation_arrays(best_model, cal_loader)
        conformal = fit_conformal_offset(cal_actual, cal_pred, settings.tft.quantiles)
        logger.info(
            "Conformal offset %s fitted on %d calibration window(s) (%d effective)",
            [f"{o:.6g}" for o in conformal.offset],
            cal_actual.shape[0],
            conformal.n_calibration,
        )

    evaluation = _evaluate_validation(
        best_model, val_loader, settings.tft.quantiles, target, conformal=conformal
    )
    meta = {
        "symbol": bundle_name,
        "symbols": symbols,
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "features": feature_columns(df),
        "target": target_column(settings),
        "val_loss": val_loss,
        "epochs_run": trainer.current_epoch,
        "seed": settings.training.seed,
        "quantiles": settings.tft.quantiles,
        "max_encoder_length": settings.training.max_encoder_length,
        "max_prediction_length": horizon,
        # The feature schema is a function of the data toggles, and nothing else
        # can recover them once the run is over -- so the bundle records the
        # resolved config it was actually trained with (PYQ-119). Secrets are not
        # included: DataConfig/TrainingConfig/TFTConfig hold no keys.
        "config": {
            "data": settings.data.model_dump(mode="json"),
            "training": settings.training.model_dump(mode="json"),
            "tft": settings.tft.model_dump(mode="json"),
        },
        "provenance": _provenance(pin),
        # vars() doesn't recurse into the nested PerHorizonMetrics dataclasses
        # (PYQ-267); flatten those to plain dicts so json.dumps below doesn't
        # choke, consistent with vars() itself already omitting computed
        # properties like skill_vs_baseline (derivable from model_mae/baseline_mae).
        "evaluation": {
            **vars(evaluation),
            "per_horizon": [vars(step) for step in evaluation.per_horizon],
        },
        # Explicitly null, not omitted (PYQ-270): this evaluation is one
        # held-out validation split, not a multi-window walk-forward backtest,
        # so there is no per-window series to bootstrap a skill interval from.
        # `pyquant backtest --format json`'s `skill_ci` carries a real one.
        "skill_ci": None,
        # Persisted so `forecast` applies the same band correction the metrics
        # above were computed under, without refitting it (PYQ-248).
        "conformal": conformal.to_dict() if conformal else None,
    }
    (bundle_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    # meta.json reflects only the deployable (latest) bundle; runs.jsonl is an
    # append-only history so retraining doesn't silently erase past runs.
    with (bundle_dir / "runs.jsonl").open("a") as f:
        f.write(json.dumps(meta) + "\n")

    return TrainResult(
        symbols=symbols,
        bundle_dir=bundle_dir,
        val_loss=val_loss,
        n_features=len(meta["features"]),
        epochs_run=trainer.current_epoch,
        evaluation=evaluation,
    )
