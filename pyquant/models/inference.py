"""Prediction, interpretation, and permutation importance for a loaded bundle.

Split out of ``models/tft.py`` (PYQ-269). Depends only on ``models/bundle.py``
(``ModelBundle``, feature-schema checking, the conformal offset a bundle was
calibrated with). See ``models/tft.py`` for the compatibility re-export
surface.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

from pyquant.analysis.calibrate import apply_conformal_offset
from pyquant.config import Settings
from pyquant.data.dataset import extend_for_prediction
from pyquant.models.bundle import ModelBundle, _check_feature_schema, bundle_conformal_offset

logger = logging.getLogger(__name__)


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

    baseline_start = time.monotonic()
    baseline = _mae(df)
    pass_seconds = time.monotonic() - baseline_start

    rng = np.random.default_rng(seed)
    feature_names = [c for c in (bundle.meta.get("features") or []) if c in df.columns]
    # One more _mae() pass per feature, each costing about as long as the baseline
    # pass just measured -- macro + sectors + sentiment + technicals + options can
    # total 20-30+ columns, and this had no feedback at all before the loop started
    # (bugs.md#pyq-329). Logging the estimate is cheap; a max_features/subsampling
    # knob is not added here since nothing calls this function outside a test yet
    # (see PYQ-329's resolution note) -- add one against a real caller's actual
    # constraint, not a guessed one.
    logger.info(
        "permutation_importance: %d features, ~%.1fs/pass (baseline) -> ~%.0fs estimated total",
        len(feature_names),
        pass_seconds,
        pass_seconds * len(feature_names),
    )
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
