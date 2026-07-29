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
import math
from dataclasses import dataclass, field

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


def quantile_exceedance(actuals: np.ndarray, predictions: np.ndarray, quantiles: list[float]) -> dict[float, float]:
    """Empirical fraction of outcomes at or below each predicted quantile."""
    actuals = np.asarray(actuals)
    return {
        q: float(np.mean(actuals <= np.asarray(predictions)[:, :, i]))
        for i, q in enumerate(quantiles)
    }


def pinball_loss(actuals: np.ndarray, predictions: np.ndarray, quantiles: list[float]) -> dict[float, float]:
    """Mean quantile-regression loss for each requested quantile."""
    actuals = np.asarray(actuals)
    losses: dict[float, float] = {}
    for i, q in enumerate(quantiles):
        error = actuals - np.asarray(predictions)[:, :, i]
        losses[q] = float(np.mean(np.maximum(q * error, (q - 1) * error)))
    return losses


def crps_from_quantiles(
    actuals: np.ndarray, predictions: np.ndarray, quantiles: list[float]
) -> float:
    """CRPS approximated as the mean pinball loss across the quantile set.

    The continuous ranked probability score is the standard *strictly proper*
    scoring rule for a full predictive distribution: unlike coverage it cannot be
    gamed by widening the band, and unlike MAE it scores the whole distribution
    rather than the median. That is what makes it the right number for comparing
    this model against a differently-shaped baseline (PYQ-249's foundation model),
    and it collapses PYQ-227's per-quantile pinball table into one figure
    (PYQ-252).

    Approximating it by averaging pinball loss over a finite quantile grid is the
    standard discrete estimator; with only three quantiles it is coarse, so it is
    comparable *between models scored on the same grid* rather than against
    published CRPS figures.
    """
    losses = pinball_loss(actuals, predictions, quantiles)
    return float(np.mean(list(losses.values()))) if losses else 0.0


def winkler_score(
    actuals: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> float:
    """Interval score: band width plus a ``2/alpha`` penalty for each miss.

    Scores an interval on coverage *and* width in one number, which is exactly
    the diagnosis plain coverage cannot give: PYQ-117's 99.3%-on-a-nominal-80%
    band looks excellent by coverage alone and is in fact nearly uninformative,
    because it achieves that coverage by being enormous. Winkler penalises the
    width directly, so the pathology shows up as a bad score rather than a good
    one (PYQ-252). Lower is better.

    ``alpha`` is the nominal miss rate of the band -- 0.2 for a p10-p90 interval.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between zero and one, got {alpha}")
    actuals = np.asarray(actuals, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = upper - lower
    below = np.where(actuals < lower, (2.0 / alpha) * (lower - actuals), 0.0)
    above = np.where(actuals > upper, (2.0 / alpha) * (actuals - upper), 0.0)
    return float(np.mean(width + below + above))


def pit_values(
    actuals: np.ndarray, predictions: np.ndarray, quantiles: list[float]
) -> np.ndarray:
    """Probability-integral transform of each actual through the predictive CDF.

    The predicted quantiles give the CDF at a handful of points; each actual is
    mapped to its own CDF value by interpolating between them. A calibrated
    forecaster produces uniform PIT values; U-shaped means overconfident
    (too many outcomes in the tails), hump-shaped means underconfident -- the
    expected shape here given a band covering 99.3% of a nominal 80% (PYQ-252).

    One histogram replaces several numbers, which is why this is worth rendering
    rather than only tabulating.
    """
    actuals = np.asarray(actuals, dtype=float).reshape(-1)
    preds = np.asarray(predictions, dtype=float).reshape(-1, len(quantiles))
    # Sorting guards the non-monotonic bands QuantileLoss can emit (PYQ-216/124);
    # np.interp requires an increasing x.
    preds = np.sort(preds, axis=-1)
    return np.array(
        [float(np.interp(a, row, quantiles)) for a, row in zip(actuals, preds, strict=True)]
    )


def effective_sample_size(n_samples: int, horizon: int) -> int:
    """Conservative independent-window estimate for overlapping horizon windows."""
    if n_samples < 0 or horizon <= 0:
        raise ValueError("n_samples must be non-negative and horizon must be positive")
    return math.ceil(n_samples / horizon)


def moving_block_bootstrap_interval(
    values: np.ndarray | list[float],
    block_size: int,
    *,
    n_resamples: int = 1_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap a mean with contiguous blocks to preserve serial dependence."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("values must be a non-empty one-dimensional array")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    block_size = min(block_size, len(values))
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(len(values) / block_size)
    starts = rng.integers(0, len(values) - block_size + 1, size=(n_resamples, n_blocks))
    samples = np.concatenate([values[start : start + block_size] for start in starts.flat]).reshape(
        n_resamples, -1
    )[:, : len(values)]
    alpha = (1 - confidence) / 2
    return tuple(float(x) for x in np.quantile(samples.mean(axis=1), [alpha, 1 - alpha]))


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
    quantile_exceedance: dict[float, float] = field(default_factory=dict)
    pinball_losses: dict[float, float] = field(default_factory=dict)
    # Proper scoring rule over the whole predictive distribution, and an interval
    # score that charges for width as well as coverage (PYQ-252). Both are
    # "lower is better", unlike every rate above.
    crps: float = 0.0
    winkler_score: float = 0.0
    # PIT values, one per scored point, for the calibration histogram. Kept off
    # the Rich tables (it is a distribution, not a number) but carried here so
    # `pyquant calibration` and --format json can both reach it.
    pit: list[float] = field(default_factory=list)

    @property
    def skill_vs_baseline(self) -> float:
        """Relative MAE improvement over the persistence baseline (positive = better)."""
        if self.baseline_mae == 0:
            return 0.0
        return (self.baseline_mae - self.model_mae) / self.baseline_mae

    @property
    def effective_n_samples(self) -> int:
        """Approximate non-overlapping windows behind the reported rates (PYQ-251)."""
        if not self.n_samples or not self.n_points:
            return 0
        horizon = max(1, round(self.n_points / self.n_samples))
        return effective_sample_size(self.n_samples, horizon)


def evaluate_predictions(
    predictions: np.ndarray,
    actuals: np.ndarray,
    last_observed: np.ndarray,
    quantiles: list[float],
    *,
    target: str = "close",
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
    # The persistence baseline for log returns is zero return. In close space
    # it remains the final observed close, preserving legacy bundle semantics.
    baseline = np.zeros(n_samples) if target == "log_return" else last_observed
    return EvaluationMetrics(
        model_mae=model_mae(actuals, median),
        baseline_mae=persistence_baseline_mae(actuals, baseline),
        directional_accuracy=directional_hit_rate(actuals, median, baseline),
        calibration_coverage=calibration_coverage(actuals, lower, upper),
        n_samples=int(n_samples),
        n_points=int(n_samples * horizon),
        quantile_exceedance=quantile_exceedance(actuals, predictions, quantiles),
        pinball_losses=pinball_loss(actuals, predictions, quantiles),
        crps=crps_from_quantiles(actuals, predictions, quantiles),
        # The band's nominal miss rate follows from the configured quantiles --
        # p10..p90 is alpha=0.2 -- rather than being assumed to be 0.2.
        winkler_score=winkler_score(
            actuals, lower, upper, alpha=max(1e-9, quantiles[0] + (1.0 - quantiles[-1]))
        ),
        pit=pit_values(actuals, predictions, quantiles).tolist(),
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
    if not results:
        # np.average([], weights=[]) raises ZeroDivisionError, not something a
        # caller would expect from an aggregation function (PYQ-156) -- reachable
        # via `pyquant backtest SYMBOL --windows 0`, which produces an empty
        # `cutoffs` list. ValueError is in cli/app.py's EXPECTED_FAILURES, so this
        # renders as a clean CLI error instead of a raw traceback.
        raise ValueError("aggregate_metrics() requires at least one window's results")
    weights = np.array([r.n_points for r in results], dtype=float)
    if weights.sum() == 0:
        weights = np.ones(len(results), dtype=float)

    def pooled(values: list[float]) -> float:
        return float(np.average(np.asarray(values, dtype=float), weights=weights))

    def pooled_dict(values: list[dict[float, float]]) -> dict[float, float]:
        keys = set.intersection(*(set(value) for value in values)) if values else set()
        return {key: pooled([value[key] for value in values]) for key in sorted(keys)}

    return EvaluationMetrics(
        model_mae=pooled([r.model_mae for r in results]),
        baseline_mae=pooled([r.baseline_mae for r in results]),
        directional_accuracy=pooled([r.directional_accuracy for r in results]),
        calibration_coverage=pooled([r.calibration_coverage for r in results]),
        n_samples=int(sum(r.n_samples for r in results)),
        n_points=int(sum(r.n_points for r in results)),
        quantile_exceedance=pooled_dict([r.quantile_exceedance for r in results]),
        pinball_losses=pooled_dict([r.pinball_losses for r in results]),
        crps=pooled([r.crps for r in results]),
        winkler_score=pooled([r.winkler_score for r in results]),
        # PIT values are per-point, so they concatenate rather than average --
        # the pooled histogram is the whole point of collecting them.
        pit=[value for r in results for value in r.pit],
    )
