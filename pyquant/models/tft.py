"""Temporal Fusion Transformer wrapper.

All pytorch-forecasting / Lightning calls are isolated here so the rest of the
codebase stays library-agnostic. If the upstream stack changes, only this file
needs to adapt.

A trained model is persisted as a bundle directory under ``checkpoints/<symbol>/``:
    model.ckpt          Lightning checkpoint (architecture + weights)
    dataset_params.pt   TimeSeriesDataSet parameters (encoders/normalizers)
    meta.json           symbol, feature names, metrics, training timestamp
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

from pyquant.config import Settings
from pyquant.data.dataset import build_panel, feature_columns, make_dataset, panel_to_long

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    symbol: str
    bundle_dir: Path
    val_loss: float
    n_features: int
    epochs_run: int


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


def _bundle_dir(settings: Settings, symbol: str) -> Path:
    return settings.checkpoint_dir / symbol.upper()


def train(
    symbol: str,
    settings: Settings,
    *,
    start: str | None = None,
    end: str | None = None,
    max_epochs: int | None = None,
    progress: bool = True,
) -> TrainResult:
    """Train a TFT for ``symbol`` and persist the bundle."""
    symbol = symbol.upper()
    panel = build_panel(symbol, settings, start, end)
    df = panel_to_long(panel, symbol)

    horizon = settings.training.max_prediction_length
    max_idx = int(df["time_idx"].max())
    cutoff = max_idx - horizon
    if cutoff <= settings.training.max_encoder_length:
        raise ValueError(
            f"Not enough history for {symbol}: need more than "
            f"{settings.training.max_encoder_length + horizon} rows, got {len(df)}."
        )

    training = make_dataset(df, settings, training_cutoff=cutoff)
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

    batch_size = settings.training.batch_size
    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    model = build_model(training, settings)

    bundle_dir = _bundle_dir(settings, symbol)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    epochs = max_epochs if max_epochs is not None else settings.training.max_epochs
    ckpt_cb = ModelCheckpoint(
        dirpath=bundle_dir, filename="model", monitor="val_loss", save_top_k=1, mode="min"
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
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
    meta = {
        "symbol": symbol,
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "features": feature_columns(df),
        "val_loss": val_loss,
        "epochs_run": trainer.current_epoch,
        "quantiles": settings.tft.quantiles,
        "max_encoder_length": settings.training.max_encoder_length,
        "max_prediction_length": horizon,
    }
    (bundle_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return TrainResult(
        symbol=symbol,
        bundle_dir=bundle_dir,
        val_loss=val_loss,
        n_features=len(meta["features"]),
        epochs_run=trainer.current_epoch,
    )


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
    dataset_params = torch.load(bundle_dir / "dataset_params.pt", weights_only=False)
    meta = json.loads((bundle_dir / "meta.json").read_text())
    return ModelBundle(model=model, dataset_params=dataset_params, meta=meta)
