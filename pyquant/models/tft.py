"""Temporal Fusion Transformer wrapper.

All pytorch-forecasting / Lightning calls are isolated here so the rest of the
codebase stays library-agnostic. If the upstream stack changes, only this file
needs to adapt.

A trained model is persisted as a bundle directory under ``checkpoints/<bundle_name>/``
(``<bundle_name>`` is the symbol, or the joined symbol list for pooled training):
    model.ckpt          Lightning checkpoint (architecture + weights)
    dataset_params.pt   TimeSeriesDataSet parameters (encoders/normalizers)
    meta.json           symbol, feature names, metrics, training timestamp
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

from pyquant import provenance
from pyquant.analysis.calibrate import (
    ConformalOffset,
    apply_conformal_offset,
    fit_conformal_offset,
)
from pyquant.analysis.metrics import EvaluationMetrics, aggregate_metrics, evaluate_predictions
from pyquant.config import Settings
from pyquant.data.dataset import (
    SCHEMA_DATA_FIELDS,
    align_time_index,
    build_panel,
    extend_for_prediction,
    feature_columns,
    make_dataset,
    panel_to_long,
    target_column,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    symbols: list[str]
    bundle_dir: Path
    val_loss: float
    n_features: int
    epochs_run: int
    evaluation: EvaluationMetrics


@dataclass
class BacktestResult:
    symbol: str
    n_windows: int
    per_window: list[EvaluationMetrics]
    aggregated: EvaluationMetrics
    # Populated only when walk_forward_backtest(..., compute_signals=True): the
    # BUY/SELL/HOLD scan() would have emitted at each origin, and the realized
    # percent move over that origin's horizon (PYQ-255). Kept optional rather
    # than always computed -- it costs one extra forward pass per window.
    signals: list[str] = field(default_factory=list)
    signal_returns_pct: list[float] = field(default_factory=list)


@dataclass
class TuneResult:
    """An Optuna hyperparameter search (PYQ-253), plus its winner's honest score.

    ``held_out_evaluation`` comes from data the search never trained or selected
    on -- every trial is a selection event, so the in-search score
    (``best_value``, the pruned/selected trial's own validation loss) is
    optimistically biased and must not be reported as the model's real
    performance.
    """

    symbol: str
    n_trials: int
    best_params: dict
    best_value: float
    held_out_evaluation: EvaluationMetrics
    bundle_dir: Path
    config_path: Path


class FeatureSchemaMismatch(RuntimeError):
    """A bundle's trained features are not all present in the freshly built panel.

    Raised instead of letting pytorch-forecasting fail with a bare ``KeyError``
    naming one column and nothing else (PYQ-118).
    """


# Which source each feature family comes from, so a missing column can say what
# went away rather than just which key was absent.
_FEATURE_SOURCE_HINTS: tuple[tuple[str, str], ...] = (
    ("SEC_", "sector ETF returns (DataConfig.use_sectors; Yahoo Finance)"),
    ("VIX", "macro context (DataConfig.use_macro; Yahoo Finance, no key needed)"),
    ("FedFunds", "FRED macro series (DataConfig.use_macro + FRED_API_KEY)"),
    ("YieldSpread", "FRED macro series (DataConfig.use_macro + FRED_API_KEY)"),
    ("CPI", "FRED macro series (DataConfig.use_macro + FRED_API_KEY)"),
    ("Sentiment", "news sentiment (DataConfig.use_sentiment + FINNHUB_API_KEY + 'sentiment' extra)"),
    ("HeadlineCount", "news sentiment (DataConfig.use_sentiment + FINNHUB_API_KEY + 'sentiment' extra)"),
)


def _source_hint(column: str) -> str:
    for prefix, hint in _FEATURE_SOURCE_HINTS:
        if column.startswith(prefix):
            return hint
    return "price data / technical indicators"


@dataclass
class ModelBundle:
    """A loaded model plus everything needed to forecast/interpret with it."""

    model: TemporalFusionTransformer
    dataset_params: dict
    meta: dict


def build_model(training_dataset: TimeSeriesDataSet, settings: Settings) -> TemporalFusionTransformer:
    """Construct a TFT sized to the dataset and config."""
    return TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=settings.training.learning_rate,
        hidden_size=settings.tft.hidden_size,
        attention_head_size=settings.tft.attention_head_size,
        dropout=settings.tft.dropout,
        hidden_continuous_size=settings.tft.hidden_continuous_size,
        loss=QuantileLoss(quantiles=settings.tft.quantiles),
        log_interval=-1,
        optimizer="adam",
    )


def _bundle_dir(settings: Settings, name: str) -> Path:
    """The on-disk directory for bundle ``name``, guaranteed inside ``checkpoint_dir``.

    Belt and braces beneath the API layer's request-schema validation
    (PYQ-145): a ``name`` like ``"../../etc"`` reaches this function directly
    from a POST body (unlike a GET route's ``{symbol}``, a JSON body field
    isn't subject to Starlette's `/`-rejecting path-parameter matching), and
    from here flows straight into ``mkdir``/``torch.save``. Checked here too so
    every caller is covered, not only the ones that remembered to validate
    first.
    """
    bundle_dir = settings.checkpoint_dir / name.upper()
    checkpoint_root = settings.checkpoint_dir.resolve()
    resolved = bundle_dir.resolve()
    if resolved != checkpoint_root and checkpoint_root not in resolved.parents:
        raise ValueError(f"Invalid bundle name {name!r}: resolves outside checkpoint_dir")
    return bundle_dir


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
        panel_to_long(build_panel(symbol, settings, start, end, pin=pin), symbol) for symbol in symbols
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


# These were duplicated here so data/cache.py could stamp a pin without importing
# the ML stack. Two copies meant PYQ-134's "an unrelated repo's sha is recorded as
# PyQuant's provenance" had to be fixed twice; delegate instead so it cannot drift.
_package_version = provenance.package_version
_git_sha = provenance.git_sha


def _provenance(pin: str | None) -> dict:
    """What is needed to reproduce this run, beyond the seed and the data.

    PYQ-210 recorded the seed and PYQ-205 added pinned datasets, but neither
    captured *which code* ran -- and feature definitions do change (PYQ-121
    redefined RSI_14). Version + sha + pin is the set that actually reproduces a
    bundle (PYQ-225).
    """
    return {"pyquant_version": _package_version(), "git_sha": _git_sha(), "pin": pin}


def purged_training_cutoff(cutoff: int, settings: Settings) -> int:
    """Last ``time_idx`` a *training* sample may decode, given purge + embargo.

    ``cutoff`` is where the held-out period begins minus one. Left alone, the
    last training samples decode the days immediately before it -- and a
    validation sample starting at ``cutoff + 1`` reads exactly those days
    through its own encoder. Training and evaluation therefore share target
    days across the boundary, which biases reported out-of-sample performance
    optimistically. The financial-ML standard treatment (López de Prado,
    *Advances in Financial Machine Learning*) is to **purge** one label horizon
    either side of the split and then **embargo** a further buffer, because
    serial correlation carries information across the boundary even where no
    literal overlap remains (PYQ-250).

    This is the last known member of the leak family PYQ-101/103/115/116/123/127
    belong to, and the one the literature considers table stakes.

    Returns the reduced cutoff; callers keep using the original for the
    validation window, which must not move.
    """
    horizon = settings.training.max_prediction_length
    purge = settings.training.purge_horizon
    purge = horizon if purge is None else int(purge)
    return cutoff - max(0, purge) - max(0, settings.training.embargo_days)


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
    max_idx = int(df["time_idx"].max())
    # Geometry, left to right:
    #   [ training .. train_cutoff ][ purge+embargo ][ calibration ][ validation ]
    # `cutoff` is the last index before the *held-out* region; the calibration
    # slice (PYQ-248) sits between it and the scored validation window so the
    # conformal offset is fitted on data that is out-of-sample for training and
    # disjoint from what it is later judged on.
    validation_start = max_idx - validation_days + 1
    cutoff = validation_start - calibration_days - 1
    train_cutoff = purged_training_cutoff(cutoff, settings)
    if train_cutoff <= encoder_len:
        raise ValueError(
            f"Not enough history for {bundle_name}: need more than "
            f"{encoder_len + validation_days + calibration_days + (cutoff - train_cutoff)} rows "
            f"(a {encoder_len}-day encoder, a {validation_days}-day validation holdout, "
            f"{calibration_days} calibration day(s) and {cutoff - train_cutoff} purged/embargoed "
            f"day(s)), got {len(df)}."
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

    batch_size = settings.training.batch_size
    num_workers = settings.training.num_workers
    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)

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
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # ModelCheckpoint may version the filename; load the actual best path.
    best_path = ckpt_cb.best_model_path or str(bundle_dir / "model.ckpt")
    best_path = Path(best_path)
    if best_path != bundle_dir / "model.ckpt" and best_path.exists():
        best_path.replace(bundle_dir / "model.ckpt")

    torch.save(training.get_parameters(), bundle_dir / "dataset_params.pt")

    val_loss = float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else float("nan")
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
        cal_pred, cal_actual, _ = _raw_validation_arrays(best_model, cal_loader)
        conformal = fit_conformal_offset(cal_actual, cal_pred, settings.tft.quantiles)
        logger.info(
            "Conformal offset %.6g fitted on %d calibration point(s)",
            conformal.offset,
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
        "evaluation": vars(evaluation),
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


def _window_validation_dataset(
    training: TimeSeriesDataSet, df: pd.DataFrame, cutoff: int, horizon: int
) -> TimeSeriesDataSet:
    """Validation set for one walk-forward origin: exactly the horizon after ``cutoff``.

    ``predict=True`` anchors the decoder to the last ``horizon`` timesteps of
    whatever frame it is handed -- so passing the *full* df made every rolling
    origin evaluate the identical final window, and a 5-window backtest was five
    differently-trained models scored on the same five days (PYQ-127). Truncating
    the frame at ``cutoff + horizon`` puts the decoder on that origin's own
    out-of-sample window instead.
    """
    window = df[df["time_idx"] <= cutoff + horizon]
    return TimeSeriesDataSet.from_dataset(
        training, window, predict=True, stop_randomization=True
    )


def _window_signal(
    predictions: np.ndarray,
    actuals: np.ndarray,
    last_observed: np.ndarray,
    quantiles: list[float],
    target: str,
) -> tuple[str, float]:
    """The (signal, realized_return_pct) scan() would have shown for one window.

    Derived from that walk-forward window's raw prediction/actual arrays
    (PYQ-255). walk_forward_backtest is single-symbol, so each window's arrays
    hold exactly one sample.
    """
    from pyquant.analysis.forecast import log_returns_to_prices
    from pyquant.analysis.signals import classify_signal

    median_idx = quantiles.index(0.5)
    pred, actual, last_obs = predictions[0], actuals[0], float(last_observed[0])

    if target == "LogReturn":
        median_path = log_returns_to_prices(pred[:, median_idx], last_obs)
        lower_path = log_returns_to_prices(pred[:, 0], last_obs)
        upper_path = log_returns_to_prices(pred[:, -1], last_obs)
        actual_path = log_returns_to_prices(actual, last_obs)
    else:
        median_path, lower_path, upper_path = pred[:, median_idx], pred[:, 0], pred[:, -1]
        actual_path = actual

    expected_pct = float((median_path[-1] - last_obs) / last_obs * 100)
    lower_pct = float((lower_path[-1] - last_obs) / last_obs * 100)
    upper_pct = float((upper_path[-1] - last_obs) / last_obs * 100)
    realized_pct = float((actual_path[-1] - last_obs) / last_obs * 100)

    return classify_signal(expected_pct, lower_pct, upper_pct), realized_pct


def walk_forward_backtest(
    symbol: str,
    settings: Settings,
    *,
    n_windows: int = 5,
    step: int | None = None,
    start: str | None = None,
    end: str | None = None,
    max_epochs: int | None = None,
    progress: bool = False,
    compute_signals: bool = False,
) -> BacktestResult:
    """Train/evaluate across many rolling origins (walk-forward validation).

    Unlike train(), each window's model is discarded after evaluation -- this
    measures how stable the metrics are across time (see PYQ-303), it does not
    produce a deployable bundle. ``compute_signals`` additionally records, per
    window, the BUY/SELL/HOLD signal scan() would have shown and the realized
    return -- one extra forward pass per window, so it defaults off (PYQ-255).
    """
    symbol = symbol.upper()
    seed_everything(settings.training.seed, workers=True)
    panel = build_panel(symbol, settings, start, end)
    df = align_time_index(panel_to_long(panel, symbol))

    horizon = settings.training.max_prediction_length
    encoder_len = settings.training.max_encoder_length
    step = step if step is not None else horizon
    max_idx = int(df["time_idx"].max())

    latest_cutoff = max_idx - horizon
    earliest_cutoff = latest_cutoff - (n_windows - 1) * step
    # Purge + embargo eat into the *earliest* origin's training slice, so the
    # history check has to account for them or the first window fails mid-run
    # rather than up front (PYQ-250).
    if purged_training_cutoff(earliest_cutoff, settings) <= encoder_len:
        gap = earliest_cutoff - purged_training_cutoff(earliest_cutoff, settings)
        raise ValueError(
            f"Not enough history for {n_windows} walk-forward window(s) of {symbol}: "
            f"need more than {encoder_len + horizon + (n_windows - 1) * step + gap} rows "
            f"({gap} of them purged/embargoed before each origin), got {len(df)}."
        )

    cutoffs = sorted(latest_cutoff - i * step for i in range(n_windows))
    epochs = max_epochs if max_epochs is not None else settings.training.max_epochs
    batch_size = settings.training.batch_size

    num_workers = settings.training.num_workers
    per_window: list[EvaluationMetrics] = []
    signals: list[str] = []
    signal_returns_pct: list[float] = []
    for cutoff in cutoffs:
        training = make_dataset(df, settings, training_cutoff=purged_training_cutoff(cutoff, settings))
        validation = _window_validation_dataset(training, df, cutoff, horizon)
        train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
        val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)

        model = build_model(training, settings)
        # Checkpoint into a throwaway dir: the backtest discards each window's
        # model, but it must still evaluate that window's *best* epoch rather
        # than its final (post-EarlyStopping) one, same as train() (PYQ-109).
        with tempfile.TemporaryDirectory() as tmp_ckpt_dir:
            ckpt_cb = ModelCheckpoint(
                dirpath=tmp_ckpt_dir, filename="best", monitor="val_loss", save_top_k=1, mode="min"
            )
            trainer = Trainer(
                max_epochs=epochs,
                accelerator="auto",
                precision=settings.training.precision,
                gradient_clip_val=settings.training.gradient_clip_val,
                callbacks=[
                    ckpt_cb,
                    EarlyStopping(
                        monitor="val_loss",
                        patience=settings.training.early_stopping_patience,
                        mode="min",
                    ),
                ],
                logger=False,
                enable_progress_bar=progress,
                enable_model_summary=False,
            )
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            per_window.append(
                _evaluate_best_checkpoint(
                    ckpt_cb.best_model_path,
                    model,
                    val_loader,
                    settings.tft.quantiles,
                    target_column(settings),
                )
            )
            if compute_signals:
                # A second forward pass rather than reusing _evaluate_best_checkpoint's:
                # that helper returns only the aggregated EvaluationMetrics, not the raw
                # arrays a signal needs, and this stays opt-in specifically to keep the
                # default backtest path (no signals) at its current one-pass cost.
                best_model = _load_best_checkpoint(ckpt_cb.best_model_path, model)
                predictions, actuals, last_observed = _raw_validation_arrays(best_model, val_loader)
                signal, realized_pct = _window_signal(
                    predictions, actuals, last_observed, settings.tft.quantiles, target_column(settings)
                )
                signals.append(signal)
                signal_returns_pct.append(realized_pct)

    return BacktestResult(
        symbol=symbol,
        n_windows=len(cutoffs),
        per_window=per_window,
        aggregated=aggregate_metrics(per_window),
        signals=signals,
        signal_returns_pct=signal_returns_pct,
    )


def tune(
    symbol: str,
    settings: Settings,
    *,
    n_trials: int = 15,
    held_out_days: int | None = None,
    max_epochs: int = 5,
    progress: bool = False,
) -> TuneResult:
    """Optuna hyperparameter search over the coupled TFT/training knobs (PYQ-253).

    Absorbs PYQ-211's narrower scope (learning-rate-only tuning via
    ``Tuner.lr_find``): learning rate is one of at least six coupled knobs
    (``hidden_size``, ``attention_head_size``, ``dropout``,
    ``hidden_continuous_size``, ``learning_rate``, ``gradient_clip_val``), and
    tuning one in isolation is close to uninformative.

    The search trains and selects entirely within ``df[time_idx < held_out_start]``
    -- the final ``held_out_days`` of the panel are never seen by any trial. The
    winning configuration is then retrained via :func:`train` with
    ``validation_days=held_out_days`` on the *full* panel, so
    ``TuneResult.held_out_evaluation`` is scored on data the search never touched.
    Report that number, not ``best_value`` (the winning trial's own in-search
    validation loss) -- every trial is a selection event, so the in-search score is
    optimistically biased by construction.
    """
    try:
        import optuna
        from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
            optimize_hyperparameters,
        )
    except ImportError as exc:
        raise ImportError(
            "pyquant tune needs the 'tuning' extra: uv sync --extra tuning"
        ) from exc

    symbol = symbol.upper()
    seed_everything(settings.training.seed, workers=True)
    panel = build_panel(symbol, settings)
    df = align_time_index(panel_to_long(panel, symbol))

    horizon = settings.training.max_prediction_length
    encoder_len = settings.training.max_encoder_length
    held_out_days = max(
        held_out_days if held_out_days is not None else settings.training.validation_days, horizon
    )
    max_idx = int(df["time_idx"].max())
    held_out_start = max_idx - held_out_days + 1

    search_df = df[df["time_idx"] < held_out_start]
    search_validation_days = max(settings.training.validation_days, horizon)
    search_max_idx = int(search_df["time_idx"].max()) if len(search_df) else -1
    search_validation_start = search_max_idx - search_validation_days + 1
    search_train_cutoff = purged_training_cutoff(search_validation_start - 1, settings)
    if search_train_cutoff <= encoder_len:
        raise ValueError(
            f"Not enough history for {symbol} to reserve {held_out_days} held-out day(s) "
            f"AND run a search-region validation split of {search_validation_days} day(s): "
            f"need more history, or a smaller held_out_days/validation_days."
        )

    training = make_dataset(search_df, settings, training_cutoff=search_train_cutoff)
    validation = TimeSeriesDataSet.from_dataset(
        training, search_df, min_prediction_idx=search_validation_start, stop_randomization=True
    )
    batch_size = settings.training.batch_size
    num_workers = settings.training.num_workers
    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)

    bundle_name = f"{symbol}_TUNED"
    bundle_dir = _bundle_dir(settings, bundle_name)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    study_path = bundle_dir / "optuna_study.db"
    study = optuna.create_study(
        study_name=bundle_name,
        storage=f"sqlite:///{study_path}",
        direction="minimize",
        load_if_exists=True,
    )

    with tempfile.TemporaryDirectory() as tmp_model_dir:
        # optimize_hyperparameters() unconditionally adds a LearningRateMonitor
        # callback, which raises unless the Trainer has a logger -- so unlike every
        # other Trainer in this file, `logger` must NOT be forced off here. Its
        # own TensorBoardLogger writes under log_dir, contained to the same
        # temporary directory as the throwaway per-trial checkpoints.
        optimize_hyperparameters(
            train_loader,
            val_loader,
            model_path=tmp_model_dir,
            max_epochs=max_epochs,
            n_trials=n_trials,
            use_learning_rate_finder=False,
            trainer_kwargs={
                "enable_progress_bar": progress,
                "accelerator": "auto",
                "gradient_clip_val": settings.training.gradient_clip_val,
            },
            log_dir=str(Path(tmp_model_dir) / "tb_logs"),
            study=study,
            verbose=1 if progress else 0,
        )

    best_params = dict(study.best_trial.params)
    logger.info("Optuna search for %s: %d trials, best value %.6g", symbol, n_trials, study.best_value)

    tuned = settings.model_copy(deep=True)
    for tft_field in ("hidden_size", "hidden_continuous_size", "attention_head_size", "dropout"):
        if tft_field in best_params:
            setattr(tuned.tft, tft_field, best_params[tft_field])
    if "learning_rate" in best_params:
        tuned.training.learning_rate = best_params["learning_rate"]
    if "gradient_clip_val" in best_params:
        tuned.training.gradient_clip_val = best_params["gradient_clip_val"]
    # The final retrain's validation slice IS the honest held-out evaluation: same
    # size as what the search excluded, so train()'s own validation_days-from-the-
    # end logic lands on exactly the region no trial ever saw.
    tuned.training.validation_days = held_out_days
    tuned.training.max_epochs = max_epochs

    result = train(symbol, tuned, bundle_name=bundle_name, progress=progress)

    config_path = _write_tuned_config(bundle_name, tuned, best_params)

    return TuneResult(
        symbol=symbol,
        n_trials=n_trials,
        best_params=best_params,
        best_value=float(study.best_value),
        held_out_evaluation=result.evaluation,
        bundle_dir=bundle_dir,
        config_path=config_path,
    )


def _write_tuned_config(bundle_name: str, tuned: Settings, best_params: dict) -> Path:
    """Write the winning configuration as a YAML file in configs/ (PYQ-209's format)."""
    import yaml

    from pyquant.config import project_root

    configs_dir = project_root() / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    config_path = configs_dir / f"{bundle_name.lower()}_tuned.yaml"
    payload = {
        "tft": {
            "hidden_size": tuned.tft.hidden_size,
            "hidden_continuous_size": tuned.tft.hidden_continuous_size,
            "attention_head_size": tuned.tft.attention_head_size,
            "dropout": tuned.tft.dropout,
        },
        "training": {
            "learning_rate": tuned.training.learning_rate,
            "gradient_clip_val": tuned.training.gradient_clip_val,
        },
    }
    header = (
        f"# PyQuant experiment config -- Optuna search winner for {bundle_name} (PYQ-253).\n"
        f"# Trial params: {best_params}\n"
        "# The in-search validation loss that selected this configuration is optimistically\n"
        "# biased (every trial is a selection event) -- see the bundle's meta.json for its\n"
        "# score on data the search never saw, which is the number to trust.\n"
    )
    config_path.write_text(header + yaml.dump(payload, sort_keys=False))
    return config_path


def _load_best_checkpoint(
    best_model_path: str | Path | None, live_model: TemporalFusionTransformer
) -> TemporalFusionTransformer:
    """The best-epoch checkpoint, falling back to the live model.

    EarlyStopping does not rewind the live model's weights to the best epoch --
    ModelCheckpoint does, and it must be reloaded explicitly (as tft.load() does
    for forecast/explain). Reload the saved best checkpoint so reported metrics
    reflect the model that actually gets deployed, not the worse final one
    (PYQ-109). Falls back to the live model only if no checkpoint was written.
    """
    best_model_path = Path(best_model_path) if best_model_path else None
    if best_model_path and best_model_path.exists():
        return TemporalFusionTransformer.load_from_checkpoint(
            str(best_model_path), map_location="cpu"
        )
    logger.warning("No best checkpoint found; evaluating the live post-fit model instead.")
    return live_model


def _evaluate_best_checkpoint(
    best_model_path: str | Path | None,
    live_model: TemporalFusionTransformer,
    val_loader,
    quantiles: list[float],
    target: str = "Close",
    conformal: ConformalOffset | None = None,
) -> EvaluationMetrics:
    """Evaluate the best-epoch checkpoint (PYQ-109), applying any conformal offset."""
    model = _load_best_checkpoint(best_model_path, live_model)
    return _evaluate_validation(model, val_loader, quantiles, target, conformal=conformal)


def _raw_validation_arrays(model: TemporalFusionTransformer, loader):
    """(predictions, actuals, last_observed) for a loader, all in target units.

    PYQ-313 verified against pytorch-forecasting 1.7.0 that these three come back
    in the target's own space rather than the normalizer's; PYQ-240's test pins
    it. Extracted so the calibration slice and the validation window are read
    exactly the same way.
    """
    result = model.predict(
        loader,
        mode="quantiles",
        return_x=True,
        return_y=True,
        trainer_kwargs={"enable_progress_bar": False, "logger": False},
    )
    return (
        result.output.cpu().numpy(),
        result.y[0].cpu().numpy(),
        result.x["encoder_target"][:, -1].cpu().numpy(),
    )


def _evaluate_validation(
    model: TemporalFusionTransformer,
    val_loader,
    quantiles: list[float],
    target: str = "Close",
    conformal: ConformalOffset | None = None,
) -> EvaluationMetrics:
    """Score the held-out validation window vs. a persistence baseline."""
    predictions, actuals, last_observed = _raw_validation_arrays(model, val_loader)
    # Score the band the user will actually be shown. Reporting coverage for an
    # uncalibrated band while `forecast` prints a calibrated one would make the
    # published number describe something nobody sees (PYQ-248).
    predictions = apply_conformal_offset(predictions, conformal)
    return evaluate_predictions(
        predictions,
        actuals,
        last_observed,
        quantiles,
        target="log_return" if target == "LogReturn" else "close",
    )


def _check_feature_schema(bundle: ModelBundle, df: pd.DataFrame) -> None:
    """Fail clearly if the panel is missing features the bundle was trained on.

    build_panel()'s per-source graceful degradation means the column set depends
    on which optional sources succeeded on that particular call. Extra columns are
    harmless (from_parameters ignores them), but a *missing* one used to surface as
    a bare ``KeyError`` from deep inside pytorch-forecasting, with no hint that a
    data source had gone away (PYQ-118).
    """
    expected = list(bundle.meta.get("features") or [])
    if not expected:
        return
    available = set(feature_columns(df))
    missing = [c for c in expected if c not in available]
    if not missing:
        return
    details = "\n".join(f"  - {c}  (from: {_source_hint(c)})" for c in missing)
    raise FeatureSchemaMismatch(
        f"The data panel is missing {len(missing)} of the {len(expected)} feature(s) "
        f"this bundle was trained on:\n{details}\n"
        "The model cannot be used without them. Either restore the source (check the "
        "API key / network, or the matching DataConfig toggle) or retrain the bundle "
        "against the feature set you have now."
    )


def _prediction_dataset(bundle: ModelBundle, df) -> TimeSeriesDataSet:
    """Rebuild a prediction TimeSeriesDataSet from saved params + fresh data.

    The frame is extended with the horizon's future rows first, so ``predict=True``
    decodes genuinely future timesteps instead of re-predicting the last observed
    window (PYQ-115). This also leaves the encoder sitting on the most recent real
    observations, which is what makes interpret()'s attention line up with the
    last panel dates.
    """
    symbols = sorted(str(symbol) for symbol in df["symbol"].unique())
    if len(symbols) != 1:
        raise ValueError(
            "predict_quantiles and interpret currently require exactly one symbol; "
            f"received {symbols}. Build one panel per symbol before predicting."
        )
    _check_feature_schema(bundle, df)
    horizon = int(bundle.dataset_params["max_prediction_length"])
    return TimeSeriesDataSet.from_parameters(
        bundle.dataset_params,
        extend_for_prediction(df, horizon),
        predict=True,
        stop_randomization=True,
    )


def bundle_conformal_offset(bundle: ModelBundle) -> ConformalOffset | None:
    """The conformal band correction recorded at train time, if any (PYQ-248)."""
    recorded = bundle.meta.get("conformal")
    return ConformalOffset.from_dict(recorded) if recorded else None


def predict_quantiles(bundle: ModelBundle, df):
    """Return a (horizon, n_quantiles) array of quantile forecasts.

    The band carries whatever conformal correction the bundle was calibrated
    with, so what a user is shown is the same band the bundle's reported
    coverage describes (PYQ-248).
    """
    ds = _prediction_dataset(bundle, df)
    dl = ds.to_dataloader(train=False, batch_size=1, num_workers=0)
    out = bundle.model.predict(dl, mode="quantiles")
    return apply_conformal_offset(out[0].cpu().numpy(), bundle_conformal_offset(bundle))


def interpret(bundle: ModelBundle, df) -> dict:
    """Return labelled feature importances + temporal attention for one forecast.

    Keys:
        encoder_importance: {feature: weight} over the lookback window
        attention:          1-D array of attention weight per past time step
    """
    ds = _prediction_dataset(bundle, df)
    dl = ds.to_dataloader(train=False, batch_size=1, num_workers=0)
    raw = bundle.model.predict(dl, mode="raw", return_x=True)
    interpretation = bundle.model.interpret_output(raw.output, reduction="sum")

    enc_names = bundle.model.encoder_variables
    enc_weights = interpretation["encoder_variables"].detach().cpu().numpy().reshape(-1)
    importance = dict(zip(enc_names, enc_weights.tolist(), strict=True))
    # Normalise to fractions for readability.
    total = sum(importance.values()) or 1.0
    importance = {k: v / total for k, v in importance.items()}

    attention = interpretation["attention"].detach().cpu().numpy().reshape(-1)
    return {"encoder_importance": importance, "attention": attention}


def permutation_importance(
    bundle: ModelBundle, df: pd.DataFrame, settings: Settings, *, seed: int = 42
) -> dict[str, float]:
    """Model-agnostic feature importance: MAE degradation from shuffling one column.

    A check on interpret()'s TFT variable-selection weights, which are a property of
    the model's internals and only as trustworthy as the literature on attention-based
    explanations allows (investigations.md#pyq-314). This assumes nothing about the
    model except that it can predict() -- so agreement between the two methods is real
    evidence the weights mean something, and disagreement is worth knowing before
    either is trusted.

    Evaluated over the bundle's own validation slice (the last
    ``TrainingConfig.validation_days`` of ``df``), which has real held-out actuals to
    score against -- unlike a single live forecast, which has none. Costs one forward
    pass over the validation set per feature, so this is an offline analysis step, not
    something to run on every ``explain`` call.
    """
    horizon = int(bundle.dataset_params["max_prediction_length"])
    validation_days = max(settings.training.validation_days, horizon)
    max_idx = int(df["time_idx"].max())
    validation_start = max_idx - validation_days + 1
    quantiles = list(bundle.meta["quantiles"])
    median_idx = quantiles.index(0.5)

    def _mae(frame: pd.DataFrame) -> float:
        ds = TimeSeriesDataSet.from_parameters(
            bundle.dataset_params,
            frame,
            predict=False,
            stop_randomization=True,
            min_prediction_idx=validation_start,
        )
        dl = ds.to_dataloader(train=False, batch_size=64, num_workers=0)
        result = bundle.model.predict(
            dl,
            mode="quantiles",
            return_y=True,
            trainer_kwargs={"enable_progress_bar": False, "logger": False},
        )
        predictions = result.output.cpu().numpy()
        actuals = result.y[0].cpu().numpy()
        return float(np.mean(np.abs(actuals - predictions[:, :, median_idx])))

    baseline = _mae(df)

    rng = np.random.default_rng(seed)
    feature_names = [c for c in (bundle.meta.get("features") or []) if c in df.columns]
    degradation: dict[str, float] = {}
    for col in feature_names:
        shuffled = df.copy()
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())
        degradation[col] = _mae(shuffled) - baseline

    # Floor at zero: a feature whose shuffling *improves* MAE is noise, not negative
    # importance, and interpret()'s weights are never negative either -- flooring
    # keeps the two comparable fraction-for-fraction.
    total = sum(max(0.0, v) for v in degradation.values()) or 1.0
    return {k: max(0.0, v) / total for k, v in degradation.items()}


def settings_for_bundle(bundle: ModelBundle, settings: Settings) -> Settings:
    """Return a copy of ``settings`` using the data toggles ``bundle`` was trained with.

    A bundle's feature schema is decided by which sources were enabled at train
    time. Rebuilding the prediction panel from whatever the current defaults happen
    to be is precisely how the PYQ-118 mismatch gets triggered: train with
    ``--no-sectors`` and forecast without it, and the schemas differ by
    construction rather than by bad luck (PYQ-119).

    Bundles trained before this was recorded simply keep the caller's settings.
    """
    recorded = (bundle.meta.get("config") or {}).get("data") or {}
    if not recorded:
        return settings
    restored = settings.model_copy(deep=True)
    for field_name in SCHEMA_DATA_FIELDS:
        if field_name in recorded:
            setattr(restored.data, field_name, recorded[field_name])
    return restored


def load(symbol: str, settings: Settings) -> ModelBundle:
    """Load a trained bundle for ``symbol``."""
    bundle_dir = _bundle_dir(settings, symbol)
    ckpt = bundle_dir / "model.ckpt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No trained model for {symbol.upper()} at {ckpt}. Run `pyquant train` first."
        )
    model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt), map_location="cpu")
    # weights_only=False is required: get_parameters() serializes
    # pytorch-forecasting normalizers/encoders that are not on PyTorch's
    # safe-unpickling allowlist, so weights_only=True raises UnpicklingError
    # (verified: PYQ-306). This deserialization can execute arbitrary code, so
    # only ever load bundles from your own trusted training runs -- the same
    # trust boundary relied on by the pickle panel cache in pyquant.data.cache.
    dataset_params = torch.load(bundle_dir / "dataset_params.pt", weights_only=False)
    meta = json.loads((bundle_dir / "meta.json").read_text())
    return ModelBundle(model=model, dataset_params=dataset_params, meta=meta)
