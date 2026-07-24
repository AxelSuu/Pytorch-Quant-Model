"""Backtest evaluation metrics: persistence baseline, directional accuracy, calibration.

All functions take plain numpy arrays so they can be reused both for a single
held-out validation window (see pyquant.models.tft.train) and across many
rolling origins in a walk-forward backtest.

Shapes: ``actuals``/``median``/``lower``/``upper`` are (n_samples, horizon);
``last_observed`` is (n_samples,) -- the last known close before each window,
broadcast across the horizon.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


def warn_on_quantile_crossing(predictions: np.ndarray, quantiles: list[float]) -> int:
    """Warn if quantiles cross (a higher quantile below a lower one).

    ``predictions`` is (..., n_quantiles), ordered ascending to match
    ``quantiles``. QuantileLoss does not enforce monotonicity pointwise
    (PYQ-216), so surface any crossing rather than silently scoring/rendering a
    band whose lower bound exceeds its upper. Returns the number of crossed
    points.
    """
    preds = np.asarray(predictions, dtype=float)
    if preds.shape[-1] < 2:
        return 0
    n_crossed = int(np.count_nonzero(np.diff(preds, axis=-1) < 0))
    if n_crossed:
        logger.warning(
            "Quantile crossing detected: %d point(s) where a higher quantile is "
            "below a lower one (quantiles=%s). Predictions are not monotonic.",
            n_crossed,
            quantiles,
        )
    return n_crossed


def persistence_baseline_mae(actuals: np.ndarray, last_observed: np.ndarray) -> float:
    """MAE of naively predicting "no change" from the last observed close."""
    baseline = np.broadcast_to(np.asarray(last_observed)[:, None], actuals.shape)
    return float(np.mean(np.abs(np.asarray(actuals) - baseline)))


def model_mae(actuals: np.ndarray, median: np.ndarray) -> float:
    """MAE of the model's median forecast."""
    return float(np.mean(np.abs(np.asarray(actuals) - np.asarray(median))))


def directional_hit_rate(actuals: np.ndarray, median: np.ndarray, last_observed: np.ndarray) -> float:
    """Fraction of forecasts whose direction (vs. last observed) matches the realized direction."""
    baseline = np.broadcast_to(np.asarray(last_observed)[:, None], actuals.shape)
    predicted_dir = np.sign(np.asarray(median) - baseline)
    actual_dir = np.sign(np.asarray(actuals) - baseline)
    return float(np.mean(predicted_dir == actual_dir))


def calibration_coverage(actuals: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of actuals falling within [lower, upper] (e.g. the p10-p90 band)."""
    actuals, lower, upper = np.asarray(actuals), np.asarray(lower), np.asarray(upper)
    inside = (actuals >= lower) & (actuals <= upper)
    return float(np.mean(inside))


@dataclass
class EvaluationMetrics:
    """Model quality vs. a naive baseline, direction, and quantile calibration."""

    model_mae: float
    baseline_mae: float
    directional_accuracy: float
    calibration_coverage: float  # empirical coverage of the outermost quantile band

    @property
    def skill_vs_baseline(self) -> float:
        """Relative MAE improvement over the persistence baseline (positive = better)."""
        if self.baseline_mae == 0:
            return 0.0
        return (self.baseline_mae - self.model_mae) / self.baseline_mae


def evaluate_predictions(
    predictions: np.ndarray,
    actuals: np.ndarray,
    last_observed: np.ndarray,
    quantiles: list[float],
) -> EvaluationMetrics:
    """Compute all evaluation metrics for one batch of quantile forecasts.

    ``predictions`` is (n_samples, horizon, n_quantiles), ordered to match
    ``quantiles``; the first/last columns are treated as the calibration band.
    """
    if 0.5 not in quantiles:
        raise ValueError(
            f"0.5 is not among the configured quantiles {quantiles}; "
            "TFTConfig.quantiles must include 0.5 to evaluate a median forecast."
        )
    predictions = np.asarray(predictions)
    warn_on_quantile_crossing(predictions, quantiles)
    median_idx = quantiles.index(0.5)
    median = predictions[:, :, median_idx]
    lower = predictions[:, :, 0]
    upper = predictions[:, :, -1]

    return EvaluationMetrics(
        model_mae=model_mae(actuals, median),
        baseline_mae=persistence_baseline_mae(actuals, last_observed),
        directional_accuracy=directional_hit_rate(actuals, median, last_observed),
        calibration_coverage=calibration_coverage(actuals, lower, upper),
    )


def aggregate_metrics(results: list[EvaluationMetrics]) -> EvaluationMetrics:
    """Average metrics across multiple windows (e.g. a walk-forward backtest)."""
    return EvaluationMetrics(
        model_mae=float(np.mean([r.model_mae for r in results])),
        baseline_mae=float(np.mean([r.baseline_mae for r in results])),
        directional_accuracy=float(np.mean([r.directional_accuracy for r in results])),
        calibration_coverage=float(np.mean([r.calibration_coverage for r in results])),
    )
