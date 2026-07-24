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
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

from pyquant.analysis.metrics import EvaluationMetrics, aggregate_metrics, evaluate_predictions
from pyquant.config import Settings
from pyquant.data.dataset import build_panel, feature_columns, make_dataset, panel_to_long

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
    return settings.checkpoint_dir / name.upper()


def _build_pooled_long_df(
    symbols: list[str],
    settings: Settings,
    start: str | None,
    end: str | None,
    pin: str | None = None,
) -> pd.DataFrame:
    """Fetch + join each symbol's panel, then pool into one long df.

    TimeSeriesDataSet is built with group_ids=["symbol"], so a single dataset
    can span multiple tickers -- each keeps its own zero-based time_idx. A
    source that's flaky for only some symbols (see PYQ-302) would otherwise
    inject NaNs into the pooled frame; instead we keep only the columns common
    to every symbol's panel and log what got dropped.
    """
    frames = [
        panel_to_long(build_panel(symbol, settings, start, end, pin=pin), symbol) for symbol in symbols
    ]
    if len(frames) == 1:
        return frames[0]

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
    return pd.concat([f[ordered_cols] for f in frames], ignore_index=True)


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
    max_idx = int(df["time_idx"].max())
    cutoff = max_idx - horizon
    if cutoff <= settings.training.max_encoder_length:
        raise ValueError(
            f"Not enough history for {bundle_name}: need more than "
            f"{settings.training.max_encoder_length + horizon} rows, got {len(df)}."
        )

    training = make_dataset(df, settings, training_cutoff=cutoff)
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

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
    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")

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
    evaluation = _evaluate_best_checkpoint(
        bundle_dir / "model.ckpt", model, val_loader, settings.tft.quantiles
    )
    meta = {
        "symbol": bundle_name,
        "symbols": symbols,
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "features": feature_columns(df),
        "val_loss": val_loss,
        "epochs_run": trainer.current_epoch,
        "seed": settings.training.seed,
        "quantiles": settings.tft.quantiles,
        "max_encoder_length": settings.training.max_encoder_length,
        "max_prediction_length": horizon,
        "evaluation": vars(evaluation),
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
) -> BacktestResult:
    """Train/evaluate across many rolling origins (walk-forward validation).

    Unlike train(), each window's model is discarded after evaluation -- this
    measures how stable the metrics are across time (see PYQ-303), it does not
    produce a deployable bundle.
    """
    symbol = symbol.upper()
    seed_everything(settings.training.seed, workers=True)
    panel = build_panel(symbol, settings, start, end)
    df = panel_to_long(panel, symbol)

    horizon = settings.training.max_prediction_length
    encoder_len = settings.training.max_encoder_length
    step = step if step is not None else horizon
    max_idx = int(df["time_idx"].max())

    latest_cutoff = max_idx - horizon
    earliest_cutoff = latest_cutoff - (n_windows - 1) * step
    if earliest_cutoff <= encoder_len:
        raise ValueError(
            f"Not enough history for {n_windows} walk-forward window(s) of {symbol}: "
            f"need more than {encoder_len + horizon + (n_windows - 1) * step} rows, got {len(df)}."
        )

    cutoffs = sorted(latest_cutoff - i * step for i in range(n_windows))
    epochs = max_epochs if max_epochs is not None else settings.training.max_epochs
    batch_size = settings.training.batch_size

    num_workers = settings.training.num_workers
    per_window: list[EvaluationMetrics] = []
    for cutoff in cutoffs:
        training = make_dataset(df, settings, training_cutoff=cutoff)
        validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
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
                callbacks=[ckpt_cb, EarlyStopping(monitor="val_loss", patience=5, mode="min")],
                logger=False,
                enable_progress_bar=progress,
                enable_model_summary=False,
            )
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            per_window.append(
                _evaluate_best_checkpoint(
                    ckpt_cb.best_model_path, model, val_loader, settings.tft.quantiles
                )
            )

    return BacktestResult(
        symbol=symbol,
        n_windows=len(cutoffs),
        per_window=per_window,
        aggregated=aggregate_metrics(per_window),
    )


def _evaluate_best_checkpoint(
    best_model_path: str | Path | None,
    live_model: TemporalFusionTransformer,
    val_loader,
    quantiles: list[float],
) -> EvaluationMetrics:
    """Evaluate the best-epoch checkpoint, falling back to the live model.

    EarlyStopping does not rewind the live model's weights to the best epoch --
    ModelCheckpoint does, and it must be reloaded explicitly (as tft.load() does
    for forecast/explain). Reload the saved best checkpoint so reported metrics
    reflect the model that actually gets deployed, not the worse final one
    (PYQ-109). Falls back to the live model only if no checkpoint was written.
    """
    best_model_path = Path(best_model_path) if best_model_path else None
    if best_model_path and best_model_path.exists():
        model = TemporalFusionTransformer.load_from_checkpoint(
            str(best_model_path), map_location="cpu"
        )
    else:
        logger.warning("No best checkpoint found; evaluating the live post-fit model instead.")
        model = live_model
    return _evaluate_validation(model, val_loader, quantiles)


def _evaluate_validation(
    model: TemporalFusionTransformer, val_loader, quantiles: list[float]
) -> EvaluationMetrics:
    """Score the held-out validation window vs. a persistence baseline."""
    result = model.predict(
        val_loader,
        mode="quantiles",
        return_x=True,
        return_y=True,
        trainer_kwargs={"enable_progress_bar": False, "logger": False},
    )
    predictions = result.output.cpu().numpy()
    actuals = result.y[0].cpu().numpy()
    last_observed = result.x["encoder_target"][:, -1].cpu().numpy()
    return evaluate_predictions(predictions, actuals, last_observed, quantiles)


def _prediction_dataset(bundle: ModelBundle, df) -> TimeSeriesDataSet:
    """Rebuild a prediction TimeSeriesDataSet from saved params + fresh data."""
    return TimeSeriesDataSet.from_parameters(
        bundle.dataset_params, df, predict=True, stop_randomization=True
    )


def predict_quantiles(bundle: ModelBundle, df):
    """Return a (horizon, n_quantiles) array of quantile forecasts."""
    ds = _prediction_dataset(bundle, df)
    dl = ds.to_dataloader(train=False, batch_size=1, num_workers=0)
    out = bundle.model.predict(dl, mode="quantiles")
    return out[0].cpu().numpy()


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
    importance = dict(zip(enc_names, enc_weights.tolist(), strict=False))
    # Normalise to fractions for readability.
    total = sum(importance.values()) or 1.0
    importance = {k: v / total for k, v in importance.items()}

    attention = interpretation["attention"].detach().cpu().numpy().reshape(-1)
    return {"encoder_importance": importance, "attention": attention}


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
