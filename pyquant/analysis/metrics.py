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
    """Model quality vs. a naive baseline, direction, and quantile calibration.

    ``n_samples``/``n_points`` travel with the metrics deliberately (PYQ-117): a
    directional accuracy of 100.0% means something very different from 5 points
    than from 500, and every consumer -- the Rich tables, ``--format json``,
    meta.json -- should be able to say which it got.
    """

    model_mae: float
    baseline_mae: float
    directional_accuracy: float
    calibration_coverage: float  # empirical coverage of the outermost quantile band
    n_samples: int = 0  # forecast windows scored
    n_points: int = 0  # n_samples * horizon -- individual predictions scored

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

    n_samples, horizon = median.shape
    return EvaluationMetrics(
        model_mae=model_mae(actuals, median),
        baseline_mae=persistence_baseline_mae(actuals, last_observed),
        directional_accuracy=directional_hit_rate(actuals, median, last_observed),
        calibration_coverage=calibration_coverage(actuals, lower, upper),
        n_samples=int(n_samples),
        n_points=int(n_samples * horizon),
    )


def aggregate_metrics(results: list[EvaluationMetrics]) -> EvaluationMetrics:
    """Pool metrics across multiple windows (e.g. a walk-forward backtest).

    The sample counts *sum* -- five windows of five points is 25 points of
    evidence, and averaging them back down to 5 would throw away the only thing
    that makes an aggregate worth more than a single window (PYQ-117).

    Every rate and error metric is therefore weighted by its window's
    ``n_points``, so the aggregate is the true rate *over the reported
    denominator*. An unweighted mean paired with a summed denominator computes
    numerator and denominator two different ways, and reads as a pooled figure
    it is not (PYQ-136). The two coincide only when every window has the same
    point count -- which is true of today's ``predict=True`` backtest and stops
    being true the moment windows differ in size (PYQ-250's embargo, a pooled
    multi-symbol backtest).

    Windows carrying no point count at all (metrics built without PYQ-117's
    counts) cannot be weighted, so they fall back to an unweighted mean.
    """
    weights = np.array([r.n_points for r in results], dtype=float)
    if weights.sum() == 0:
        weights = np.ones(len(results), dtype=float)

    def pooled(values: list[float]) -> float:
        return float(np.average(np.asarray(values, dtype=float), weights=weights))

    return EvaluationMetrics(
        model_mae=pooled([r.model_mae for r in results]),
        baseline_mae=pooled([r.baseline_mae for r in results]),
        directional_accuracy=pooled([r.directional_accuracy for r in results]),
        calibration_coverage=pooled([r.calibration_coverage for r in results]),
        n_samples=int(sum(r.n_samples for r in results)),
        n_points=int(sum(r.n_points for r in results)),
    )
