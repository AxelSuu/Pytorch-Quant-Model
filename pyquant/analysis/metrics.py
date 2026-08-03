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

from pyquant.analysis import baselines

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


def directional_hit_rate(
    actuals: np.ndarray, median: np.ndarray, last_observed: np.ndarray
) -> float:
    """Fraction of forecasts whose direction (vs. last observed) matches the realized direction.

    >>> actuals = np.array([[101.0], [99.0]])
    >>> median = np.array([[100.5], [99.5]])
    >>> last_observed = np.array([100.0, 100.0])
    >>> directional_hit_rate(actuals, median, last_observed)
    1.0
    """
    baseline = np.broadcast_to(np.asarray(last_observed)[:, None], actuals.shape)
    predicted_dir = np.sign(np.asarray(median) - baseline)
    actual_dir = np.sign(np.asarray(actuals) - baseline)
    return float(np.mean(predicted_dir == actual_dir))


def calibration_coverage(actuals: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of actuals falling within [lower, upper] (e.g. the p10-p90 band).

    >>> actuals = np.array([100.0, 105.0, 90.0])
    >>> lower = np.array([95.0, 95.0, 95.0])
    >>> upper = np.array([110.0, 110.0, 110.0])
    >>> round(calibration_coverage(actuals, lower, upper), 4)
    0.6667
    """
    actuals, lower, upper = np.asarray(actuals), np.asarray(lower), np.asarray(upper)
    inside = (actuals >= lower) & (actuals <= upper)
    return float(np.mean(inside))


def quantile_exceedance(
    actuals: np.ndarray, predictions: np.ndarray, quantiles: list[float]
) -> dict[float, float]:
    """Empirical fraction of outcomes at or below each predicted quantile."""
    actuals = np.asarray(actuals)
    return {
        q: float(np.mean(actuals <= np.asarray(predictions)[:, :, i]))
        for i, q in enumerate(quantiles)
    }


def pinball_loss(
    actuals: np.ndarray, predictions: np.ndarray, quantiles: list[float]
) -> dict[float, float]:
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


def winkler_score(actuals: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
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


def pit_values(actuals: np.ndarray, predictions: np.ndarray, quantiles: list[float]) -> np.ndarray:
    """Probability-integral transform of each actual through the predictive CDF.

    The predicted quantiles give the CDF at a handful of points; each actual is
    mapped to its own CDF value by interpolating between them. A calibrated
    forecaster produces uniform PIT values; U-shaped means overconfident
    (too many outcomes in the tails), hump-shaped means underconfident -- the
    expected shape here given a band covering 99.3% of a nominal 80% (PYQ-252).

    One histogram replaces several numbers, which is why this is worth rendering
    rather than only tabulating.

    Edge-clamped, not extrapolated (PYQ-153): ``np.interp`` saturates outside
    its knot range by construction, so with the default three quantiles
    (p10/p50/p90) every actual below the predicted p10 maps to exactly 0.1 and
    every actual above p90 maps to exactly 0.9 -- a point mass at each band
    edge rather than a spread into the tails. A resulting histogram can show
    *how many* points landed outside the band, not *how far past it* they
    landed; treat mass piled at the two edge bins as a lower bound on
    miscalibration, not the whole picture.
    """
    actuals = np.asarray(actuals, dtype=float).reshape(-1)
    preds = np.asarray(predictions, dtype=float).reshape(-1, len(quantiles))
    # Sorting guards the non-monotonic bands QuantileLoss can emit (PYQ-216/124);
    # np.interp requires an increasing x.
    preds = np.sort(preds, axis=-1)
    return np.array(
        [float(np.interp(a, row, quantiles)) for a, row in zip(actuals, preds, strict=True)]
    )


def skill_vs_baseline_from_maes(
    baseline_mae: float | None, model_mae: float | None
) -> float | None:
    """`EvaluationMetrics.skill_vs_baseline`'s formula, computable from just the two MAEs.

    A bundle's persisted ``meta.json`` records ``vars(evaluation)`` -- dataclass
    fields only, not the ``@property`` itself -- so anything reading skill back
    off a bundle's own meta.json (rather than a live ``EvaluationMetrics``
    instance) needs to recompute it the same way every other reader does
    (`analysis.interpret._bundle_skill`, the API's ``GET /symbols``/``GET
    /metrics/{symbol}``, PYQ-283). ``None`` when ``baseline_mae`` is missing or
    zero, distinguishing "not recorded" from a real zero-skill result.
    """
    if not baseline_mae:
        return None
    return (baseline_mae - model_mae) / baseline_mae


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


def skill_confidence_interval(
    per_window: list[EvaluationMetrics], **kwargs
) -> tuple[float, float] | None:
    """Moving-block bootstrap CI for skill across walk-forward windows (PYQ-270).

    Bootstraps the per-window ``skill_vs_baseline`` values themselves -- the
    mean-of-ratios estimator the per-window table already reports (PYQ-141),
    not the pooled-ratio headline `aggregate_metrics` computes. The two are
    different statistics; this is an interval on the former, reused because
    it is the series a window-level bootstrap can actually be built from.

    ``block_size=1``, deliberately: each entry in ``per_window`` is already a
    whole walk-forward window's pooled metrics, and consecutive origins are
    disjoint (invariant 7 / PYQ-127) -- there is no overlapping-data
    correlation between windows for a wider block to preserve, unlike the
    within-window overlap a point-level bootstrap corrects for. This is the
    same "windows are the independent unit" assumption `effective_n_samples`
    already makes elsewhere in this module.

    Returns ``None`` with fewer than two windows: a bootstrap over a single
    point has nothing to resample and would misreport a zero-width interval
    as if it were informative.
    """
    if len(per_window) < 2:
        return None
    skills = [w.skill_vs_baseline for w in per_window]
    return moving_block_bootstrap_interval(skills, block_size=1, **kwargs)


def directional_accuracy_confidence_interval(
    per_window: list[EvaluationMetrics], **kwargs
) -> tuple[float, float] | None:
    """Moving-block bootstrap CI for directional accuracy across walk-forward windows.

    Same block-size reasoning as `skill_confidence_interval` (PYQ-270): this
    used to be called with ``block_size = max(1, horizon)``, a point-level
    overlap correction mistakenly applied to a series whose elements are
    already whole windows -- at the default 5 windows / 5-day horizon that
    made ``block_size == len(values)``, collapsing the "95% CI" to a single
    possible resample and reporting a zero-width interval as real precision.
    Fixed to ``block_size=1`` for the same "windows are independent" reason.
    """
    if len(per_window) < 2:
        return None
    accuracies = [w.directional_accuracy for w in per_window]
    return moving_block_bootstrap_interval(accuracies, block_size=1, **kwargs)


@dataclass
class ScoredWindows:
    """The per-window results + window identity `compare_backtests` needs (PYQ-266).

    Deliberately not `models.tft.BacktestResult` itself: that module imports
    Lightning/pytorch-forecasting, and `analysis/` must stay free of both (the
    layering rule CLAUDE.md states and PYQ-267's own resolution ran into when
    `serialize.py` tried the reverse import). Any caller holding a
    `BacktestResult` builds one with
    ``ScoredWindows(result.per_window, result.origins)``.
    """

    per_window: list[EvaluationMetrics]
    origins: list[int]


@dataclass
class PairedComparison:
    """A moving-block-bootstrapped, window-paired comparison of two backtests' skill (PYQ-266).

    The two configurations are scored on the *same* walk-forward windows, so
    their difference is paired, not two independent marginal intervals --
    overlapping marginal intervals do not imply no difference, which is
    precisely the error this shape of reporting exists to avoid.
    """

    mean_diff: float  # mean(skill_a - skill_b) across paired windows
    ci_low: float
    ci_high: float
    n_windows: int
    block_size: int

    @property
    def excludes_zero(self) -> bool:
        """True when the interval does not straddle zero.

        The pre-registrable form of "is this difference real" that
        investigations.md#pyq-322 asks for: "flip the default when the paired
        interval excludes zero" can be written down before the run, unlike
        "when the number looks better."
        """
        return self.ci_low > 0.0 or self.ci_high < 0.0


def compare_backtests(
    a: ScoredWindows,
    b: ScoredWindows,
    *,
    block_size: int | None = None,
    n_resamples: int = 1_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> PairedComparison:
    """Paired moving-block-bootstrap comparison of two backtests' per-window skill.

    Refuses to compare results whose windows do not verifiably align -- that
    guard is the whole value of the paired framing over two eyeballed marginal
    intervals. Both sides need a non-empty, equal, elementwise-identical
    ``origins`` list (``models.tft.walk_forward_backtest`` populates it); a
    `BacktestResult` built before PYQ-266, or from a different symbol/window
    count/step, fails this and raises rather than being silently compared.

    ``block_size`` defaults to the horizon recorded on ``a``'s first window
    (``n_points / n_samples``, the same derivation `EvaluationMetrics.
    effective_n_samples` uses) so overlapping windows don't inflate
    significance, per PYQ-251's own reasoning; pass it explicitly to override.
    """
    if not a.origins or not b.origins:
        raise ValueError(
            "compare_backtests() requires both sides to carry recorded window origins "
            "(walk_forward_backtest() populates BacktestResult.origins) -- refusing to "
            "treat an unverifiable comparison as paired."
        )
    if len(a.per_window) != len(a.origins) or len(b.per_window) != len(b.origins):
        raise ValueError("per_window and origins must be the same length on each side")
    if a.origins != b.origins:
        raise ValueError(
            f"window origins do not align: a has {a.origins}, b has {b.origins}. "
            "compare_backtests() only compares two configurations scored on identical "
            "walk-forward windows -- rerun both with the same symbol, n_windows, step "
            "and start/end."
        )

    skill_a = np.array([w.skill_vs_baseline for w in a.per_window])
    skill_b = np.array([w.skill_vs_baseline for w in b.per_window])
    diffs = skill_a - skill_b

    if block_size is None:
        first = a.per_window[0]
        block_size = max(1, round(first.n_points / first.n_samples)) if first.n_samples else 1

    ci_low, ci_high = moving_block_bootstrap_interval(
        diffs, block_size, n_resamples=n_resamples, seed=seed, confidence=confidence
    )
    return PairedComparison(
        mean_diff=float(np.mean(diffs)),
        ci_low=ci_low,
        ci_high=ci_high,
        n_windows=len(diffs),
        block_size=block_size,
    )


@dataclass
class PerHorizonMetrics:
    """One decoder step's metrics, isolated from the horizon-wide mean (PYQ-267).

    Every field on `EvaluationMetrics` above averages over h=1..horizon, which
    discards exactly the structure most likely to distinguish a model that is
    genuinely learning something (skill rising with horizon, since persistence
    is hardest to beat at h=1) from one that is only tracking the last close
    (skill falling). `model_mae`/`baseline_mae` are kept rather than a bare
    `skill` float so a caller can derive it (`skill_vs_baseline` below, same
    formula as `EvaluationMetrics`) without hand-copying the formula, and so
    position-wise pooling in `aggregate_metrics` can weight-average the same
    way the top-level metrics do.
    """

    step: int  # 1-indexed decoder position
    model_mae: float
    baseline_mae: float
    directional_accuracy: float
    calibration_coverage: float

    @property
    def skill_vs_baseline(self) -> float:
        """Relative MAE improvement over the persistence baseline, at this step."""
        if self.baseline_mae == 0:
            return 0.0
        return (self.baseline_mae - self.model_mae) / self.baseline_mae


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
    # The per-horizon-step profile the fields above collapse into a mean over
    # (PYQ-267). Empty for metrics built without it (e.g. hand-constructed in
    # older tests/scripts) rather than raising -- this is additive detail, not
    # a required input.
    per_horizon: list[PerHorizonMetrics] = field(default_factory=list)
    # MAE against every comparator in analysis/baselines.py, not only persistence
    # (PYQ-275) -- persistence is uniquely favourable to the null on a
    # near-random-walk level series, so failing to beat it alone is weak
    # evidence. Always has a "persistence" key equal to `baseline_mae` above;
    # populated with the rest only when `evaluate_predictions` is given
    # encoder history to compute them from.
    baseline_maes: dict[str, float] = field(default_factory=dict)

    @property
    def skill_vs_baseline(self) -> float:
        """Relative MAE improvement over the persistence baseline (positive = better).

        A dimensionless fraction: 0.25 means the model's MAE is 25% below the
        persistence baseline's, regardless of whether the underlying MAE is in
        dollars or log-return units.

        >>> metrics = EvaluationMetrics(
        ...     model_mae=1.5, baseline_mae=2.0, directional_accuracy=0.6, calibration_coverage=0.8
        ... )
        >>> round(metrics.skill_vs_baseline, 2)
        0.25
        """
        if self.baseline_mae == 0:
            return 0.0
        return (self.baseline_mae - self.model_mae) / self.baseline_mae

    @property
    def strongest_baseline(self) -> tuple[str, float] | None:
        """The (name, mae) of whichever baseline has the *lowest* MAE (PYQ-275).

        The lowest-MAE baseline is the hardest one for the model to beat, and
        therefore the honest one to headline skill against -- reporting skill
        against the weakest available comparator is the failure mode this
        exists to prevent. None if no baselines were recorded.
        """
        if not self.baseline_maes:
            return None
        return min(self.baseline_maes.items(), key=lambda kv: kv[1])

    @property
    def skill_vs_strongest_baseline(self) -> float | None:
        """`skill_vs_baseline`, but against `strongest_baseline` instead of persistence alone."""
        strongest = self.strongest_baseline
        if strongest is None:
            return None
        _, mae = strongest
        if mae == 0:
            return 0.0
        return (mae - self.model_mae) / mae

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
    history: np.ndarray | None = None,
) -> EvaluationMetrics:
    """Compute all evaluation metrics for one batch of quantile forecasts.

    ``predictions`` is (n_samples, horizon, n_quantiles), ordered to match
    ``quantiles``; the first/last columns are treated as the calibration band.

    ``history`` is each sample's full encoder window, ``(n_samples,
    encoder_length)``, in the same units as ``actuals`` -- pass it to also
    score against analysis/baselines.py's comparators beyond persistence
    (PYQ-275). Optional and additive: without it, ``EvaluationMetrics.
    baseline_maes`` still carries a "persistence" entry (the same value as
    ``baseline_mae``), just none of the others.
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
    # Isolate each decoder step before the horizon axis gets averaged away
    # below (PYQ-267). `[:, h : h + 1]` rather than `[:, h]` keeps the 2D shape
    # persistence_baseline_mae/directional_hit_rate broadcast `baseline`
    # against.
    per_horizon = [
        PerHorizonMetrics(
            step=h + 1,
            model_mae=model_mae(actuals[:, h : h + 1], median[:, h : h + 1]),
            baseline_mae=persistence_baseline_mae(actuals[:, h : h + 1], baseline),
            directional_accuracy=directional_hit_rate(
                actuals[:, h : h + 1], median[:, h : h + 1], baseline
            ),
            calibration_coverage=calibration_coverage(
                actuals[:, h : h + 1], lower[:, h : h + 1], upper[:, h : h + 1]
            ),
        )
        for h in range(horizon)
    ]
    # "persistence" always matches `baseline_mae` below -- same target-aware
    # computation (zero return for log_return, last close otherwise) -- so the
    # two never silently disagree. The other comparators don't have that
    # target-specific convention; they read `history` directly (PYQ-275).
    baseline_mae_by_name = {"persistence": persistence_baseline_mae(actuals, baseline)}
    if history is not None:
        baseline_mae_by_name.update(
            baselines.baseline_maes(
                actuals,
                history,
                [b for b in baselines.DEFAULT_BASELINES if b.name != "persistence"],
            )
        )
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
        per_horizon=per_horizon,
        baseline_maes=baseline_mae_by_name,
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
        per_horizon=_pool_per_horizon(results),
        baseline_maes=pooled_dict([r.baseline_maes for r in results]),
    )


def _pool_per_horizon(results: list[EvaluationMetrics]) -> list[PerHorizonMetrics]:
    """Position-wise pool of `per_horizon` across windows (PYQ-267).

    Weighted by each window's `n_samples`, not `n_points` like the top-level
    `pooled()` above: a `PerHorizonMetrics` entry already isolates one step, so
    the count backing it is that window's sample count at that step, not
    samples-times-horizon. Windows missing the breakdown (or shorter than the
    longest one) are skipped position-wise rather than raising -- additive
    detail degrading gracefully, same as an empty `per_horizon` does above.
    """
    with_profile = [r for r in results if r.per_horizon]
    if not with_profile:
        return []
    horizon = max(len(r.per_horizon) for r in with_profile)
    pooled_steps = []
    for h in range(horizon):
        cells = [(r.per_horizon[h], r.n_samples) for r in with_profile if len(r.per_horizon) > h]
        if not cells:
            continue
        weights = np.array([n for _, n in cells], dtype=float)
        if weights.sum() == 0:
            weights = np.ones(len(cells), dtype=float)

        def pooled_attr(attr: str, cells=cells, weights=weights) -> float:
            return float(np.average([getattr(c, attr) for c, _ in cells], weights=weights))

        pooled_steps.append(
            PerHorizonMetrics(
                step=h + 1,
                model_mae=pooled_attr("model_mae"),
                baseline_mae=pooled_attr("baseline_mae"),
                directional_accuracy=pooled_attr("directional_accuracy"),
                calibration_coverage=pooled_attr("calibration_coverage"),
            )
        )
    return pooled_steps
