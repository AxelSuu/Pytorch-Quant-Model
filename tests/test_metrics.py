"""Tests for evaluation metrics: persistence baseline, directional accuracy, calibration."""

import numpy as np
import pytest

from pyquant.analysis import baselines as default_baselines
from pyquant.analysis import metrics


def test_persistence_baseline_mae_is_zero_when_price_never_moves():
    actuals = np.array([[100.0, 100.0, 100.0]])
    last_observed = np.array([100.0])
    assert metrics.persistence_baseline_mae(actuals, last_observed) == 0.0


def test_persistence_baseline_mae_matches_manual_calculation():
    actuals = np.array([[101.0, 103.0], [98.0, 97.0]])
    last_observed = np.array([100.0, 100.0])
    # |1|,|3|,|2|,|3| -> mean = 9/4
    assert abs(metrics.persistence_baseline_mae(actuals, last_observed) - 2.25) < 1e-9


def test_directional_hit_rate_all_correct():
    actuals = np.array([[105.0], [95.0]])  # up, down
    median = np.array([[102.0], [98.0]])  # predicted up, predicted down
    last_observed = np.array([100.0, 100.0])
    assert metrics.directional_hit_rate(actuals, median, last_observed) == 1.0


def test_directional_hit_rate_all_wrong():
    actuals = np.array([[105.0], [95.0]])  # up, down
    median = np.array([[98.0], [102.0]])  # predicted down, predicted up
    last_observed = np.array([100.0, 100.0])
    assert metrics.directional_hit_rate(actuals, median, last_observed) == 0.0


def test_calibration_coverage_all_inside_band():
    actuals = np.array([[100.0, 101.0]])
    lower = np.array([[95.0, 95.0]])
    upper = np.array([[105.0, 105.0]])
    assert metrics.calibration_coverage(actuals, lower, upper) == 1.0


def test_calibration_coverage_half_outside_band():
    actuals = np.array([[100.0, 110.0]])
    lower = np.array([[95.0, 95.0]])
    upper = np.array([[105.0, 105.0]])
    assert metrics.calibration_coverage(actuals, lower, upper) == 0.5


def test_calibration_coverage_band_is_closed_at_both_ends():
    """PYQ-245's mutation survey: neither existing coverage test has an actual
    landing exactly on a bound, so `>=`/`<=` surviving as `>`/`<` went unnoticed.
    The docstring's own `[lower, upper]` notation means closed -- an actual
    exactly at either edge counts as inside the band, not outside it."""
    actuals = np.array([[95.0, 105.0]])
    lower = np.array([[95.0, 95.0]])
    upper = np.array([[105.0, 105.0]])
    assert metrics.calibration_coverage(actuals, lower, upper) == 1.0


def test_quantile_exceedance_and_pinball_loss_match_hand_calculation():
    actuals = np.array([[1.0, 3.0]])
    predictions = np.array([[[0.0, 0.0, 2.0], [2.0, 4.0, 4.0]]])
    quantiles = [0.1, 0.5, 0.9]

    assert metrics.quantile_exceedance(actuals, predictions, quantiles) == {
        0.1: 0.0,
        0.5: 0.5,
        0.9: 1.0,
    }
    losses = metrics.pinball_loss(actuals, predictions, quantiles)
    assert losses[0.1] == pytest.approx(0.1)
    assert losses[0.5] == pytest.approx(0.5)
    assert losses[0.9] == pytest.approx(0.1)


def test_log_return_metrics_use_zero_return_persistence_baseline():
    predictions = np.array([[[0.0, 0.01, 0.02]]])
    actuals = np.array([[0.01]])
    result = metrics.evaluate_predictions(
        predictions, actuals, np.array([123.0]), [0.1, 0.5, 0.9], target="log_return"
    )
    assert result.baseline_mae == pytest.approx(0.01)
    assert result.model_mae == pytest.approx(0.0)


def test_skill_vs_baseline_from_maes_matches_the_dataclass_property():
    ev = metrics.EvaluationMetrics(
        model_mae=1.5, baseline_mae=2.0, directional_accuracy=0.6, calibration_coverage=0.8
    )
    assert metrics.skill_vs_baseline_from_maes(ev.baseline_mae, ev.model_mae) == pytest.approx(
        ev.skill_vs_baseline
    )


def test_skill_vs_baseline_from_maes_is_none_without_a_recorded_baseline():
    assert metrics.skill_vs_baseline_from_maes(None, 1.0) is None
    assert metrics.skill_vs_baseline_from_maes(0.0, 1.0) is None


def test_effective_sample_size_accounts_for_overlapping_horizons():
    assert metrics.effective_sample_size(56, 5) == 12
    assert (
        metrics.EvaluationMetrics(1, 2, 0.5, 0.8, n_samples=56, n_points=280).effective_n_samples
        == 12
    )


def test_aggregated_walk_forward_windows_are_a_different_estimator_than_train_validation():
    """docs/methodology.md's decision-rule criterion 2 (GitHub Issue #193).

    `train()`'s ~12-effective-window figure comes from one fitted model scored
    on 56 overlapping decode origins in a single validation holdout. Each
    `walk_forward_backtest` window is instead exactly one sample (n_samples=1)
    from one independently retrained model (PYQ-127), so aggregating windows
    sums n_samples directly rather than reproducing that ~12. This pins the
    two numbers methodology.md cites so a future change to either estimator
    can't silently invalidate the documented claim without a test noticing.
    """
    horizon = 5

    def one_window() -> metrics.EvaluationMetrics:
        # walk_forward_backtest's per-window result: exactly one sample,
        # `horizon` points (PYQ-127's docstring on _window_signal).
        return metrics.EvaluationMetrics(
            model_mae=1.0,
            baseline_mae=1.0,
            directional_accuracy=0.5,
            calibration_coverage=0.8,
            n_samples=1,
            n_points=horizon,
        )

    # `backtest --windows 5` (methodology.md's own worked example): 5 independently
    # retrained models, one sample each -- effective_n_samples = 1, not ~12.
    five_windows = metrics.aggregate_metrics([one_window() for _ in range(5)])
    assert five_windows.effective_n_samples == 1

    # Clearing the decision rule's own effective_n >= 10 floor at horizon=5 needs
    # windows >= 50, exactly as the corrected criterion 2 text now states.
    fifty_windows = metrics.aggregate_metrics([one_window() for _ in range(50)])
    assert fifty_windows.effective_n_samples == 10

    # train()'s validation split is the other estimator: one fit, ~56 overlapping
    # decode origins -- the ~12 figure criterion 2 now attributes to train() only.
    train_validation = metrics.EvaluationMetrics(
        model_mae=1.0,
        baseline_mae=1.0,
        directional_accuracy=0.5,
        calibration_coverage=0.8,
        n_samples=56,
        n_points=56 * horizon,
    )
    assert train_validation.effective_n_samples == 12


def test_moving_block_bootstrap_interval_uses_contiguous_blocks_deterministically():
    values = [0.0, 0.0, 1.0, 1.0]
    interval = metrics.moving_block_bootstrap_interval(
        values, block_size=2, n_resamples=100, seed=7
    )
    assert interval == metrics.moving_block_bootstrap_interval(values, 2, n_resamples=100, seed=7)
    assert 0.0 <= interval[0] <= interval[1] <= 1.0
    with pytest.raises(ValueError, match="positive"):
        metrics.moving_block_bootstrap_interval(values, block_size=0)


# --- PYQ-275: baselines beyond persistence ------------------------------------


def _quantile_band(median: np.ndarray) -> np.ndarray:
    """(n_samples, horizon, 3) predictions with an arbitrary p10/p90 around median."""
    return np.stack([median - 5.0, median, median + 5.0], axis=-1)


def test_evaluate_predictions_without_history_only_records_persistence():
    median = np.array([[101.0, 103.0], [98.0, 96.0]])
    predictions = _quantile_band(median)
    actuals = np.array([[102.0, 104.0], [97.0, 95.0]])
    last_observed = np.array([100.0, 100.0])

    result = metrics.evaluate_predictions(predictions, actuals, last_observed, [0.1, 0.5, 0.9])

    assert set(result.baseline_maes) == {"persistence"}
    assert result.baseline_maes["persistence"] == pytest.approx(result.baseline_mae)


def test_evaluate_predictions_with_history_records_every_default_baseline():
    median = np.array([[101.0, 103.0], [98.0, 96.0]])
    predictions = _quantile_band(median)
    actuals = np.array([[102.0, 104.0], [97.0, 95.0]])
    last_observed = np.array([100.0, 100.0])
    rng = np.random.default_rng(3)
    history = rng.normal(100, 2, size=(2, 15))

    result = metrics.evaluate_predictions(
        predictions, actuals, last_observed, [0.1, 0.5, 0.9], history=history
    )

    expected_names = {b.name for b in default_baselines.DEFAULT_BASELINES}
    assert set(result.baseline_maes) == expected_names
    # The target-aware persistence entry must still match `baseline_mae`
    # exactly, not the generic history[:, -1] carried-forward baseline.
    assert result.baseline_maes["persistence"] == pytest.approx(result.baseline_mae)


def test_evaluation_metrics_strongest_baseline_picks_the_lowest_mae():
    ev = metrics.EvaluationMetrics(
        model_mae=5.0,
        baseline_mae=10.0,
        directional_accuracy=0.5,
        calibration_coverage=0.8,
        baseline_maes={"persistence": 10.0, "seasonal_naive": 3.0, "ar1": 7.0},
    )
    assert ev.strongest_baseline == ("seasonal_naive", 3.0)
    # skill vs. the strongest (3.0) is much worse than skill vs. persistence (10.0).
    assert ev.skill_vs_strongest_baseline == pytest.approx((3.0 - 5.0) / 3.0)
    assert ev.skill_vs_strongest_baseline < ev.skill_vs_baseline


def test_evaluation_metrics_strongest_baseline_is_none_without_any_recorded():
    ev = metrics.EvaluationMetrics(
        model_mae=1.0, baseline_mae=2.0, directional_accuracy=0.5, calibration_coverage=0.8
    )
    assert ev.strongest_baseline is None
    assert ev.skill_vs_strongest_baseline is None


def test_aggregate_metrics_pools_baseline_maes_across_windows():
    a = metrics.EvaluationMetrics(
        model_mae=1.0,
        baseline_mae=2.0,
        directional_accuracy=0.5,
        calibration_coverage=0.8,
        n_points=4,
        baseline_maes={"persistence": 2.0, "ar1": 4.0},
    )
    b = metrics.EvaluationMetrics(
        model_mae=1.0,
        baseline_mae=6.0,
        directional_accuracy=0.5,
        calibration_coverage=0.8,
        n_points=4,
        baseline_maes={"persistence": 6.0, "ar1": 8.0},
    )
    agg = metrics.aggregate_metrics([a, b])
    assert agg.baseline_maes["persistence"] == pytest.approx(4.0)  # equal weight -> midpoint
    assert agg.baseline_maes["ar1"] == pytest.approx(6.0)


def test_evaluate_predictions_combines_all_metrics():
    # 2 samples, horizon 2, quantiles [0.1, 0.5, 0.9]
    predictions = np.array(
        [
            [[95.0, 101.0, 106.0], [96.0, 103.0, 109.0]],  # sample 1
            [[90.0, 98.0, 104.0], [88.0, 96.0, 102.0]],  # sample 2
        ]
    )
    actuals = np.array([[102.0, 104.0], [97.0, 95.0]])
    last_observed = np.array([100.0, 100.0])

    result = metrics.evaluate_predictions(predictions, actuals, last_observed, [0.1, 0.5, 0.9])

    assert abs(result.model_mae - np.mean(np.abs(actuals - predictions[:, :, 1]))) < 1e-9
    assert (
        abs(result.baseline_mae - metrics.persistence_baseline_mae(actuals, last_observed)) < 1e-9
    )
    assert 0.0 <= result.directional_accuracy <= 1.0
    assert 0.0 <= result.calibration_coverage <= 1.0


def test_aggregate_metrics_averages_across_windows():
    a = metrics.EvaluationMetrics(
        model_mae=1.0, baseline_mae=2.0, directional_accuracy=0.5, calibration_coverage=0.7
    )
    b = metrics.EvaluationMetrics(
        model_mae=3.0, baseline_mae=4.0, directional_accuracy=0.9, calibration_coverage=0.9
    )
    agg = metrics.aggregate_metrics([a, b])
    assert agg.model_mae == 2.0
    assert agg.baseline_mae == 3.0
    assert agg.directional_accuracy == 0.7
    assert abs(agg.calibration_coverage - 0.8) < 1e-9


# --- PYQ-266: paired comparison of two backtests -----------------------------


def _windows(model_maes, baseline_maes, origins):
    per_window = [
        metrics.EvaluationMetrics(
            model_mae=m,
            baseline_mae=b,
            directional_accuracy=0.5,
            calibration_coverage=0.8,
            n_samples=1,
            n_points=5,
        )
        for m, b in zip(model_maes, baseline_maes, strict=True)
    ]
    return metrics.ScoredWindows(per_window=per_window, origins=list(origins))


def test_compare_backtests_identical_inputs_give_zero_difference():
    a = _windows([1.0, 2.0, 3.0], [2.0, 2.0, 2.0], origins=[100, 105, 110])
    result = metrics.compare_backtests(a, a)
    assert result.mean_diff == pytest.approx(0.0)
    assert result.ci_low <= 0.0 <= result.ci_high
    assert result.n_windows == 3
    assert result.excludes_zero is False


def test_compare_backtests_recovers_a_constant_offset():
    # a's skill is exactly 1.0 at every window; b's is exactly 0.2 -- a noiseless
    # +0.8 difference, so the bootstrap interval should collapse to a point.
    a = _windows([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], origins=[1, 2, 3, 4])
    b = _windows([0.8, 0.8, 0.8, 0.8], [1.0, 1.0, 1.0, 1.0], origins=[1, 2, 3, 4])

    result = metrics.compare_backtests(a, b)

    assert result.mean_diff == pytest.approx(0.8)
    assert result.ci_low == pytest.approx(0.8)
    assert result.ci_high == pytest.approx(0.8)
    assert result.excludes_zero is True


def test_compare_backtests_defaults_block_size_from_recorded_horizon():
    """The block length is keyed to the horizon (n_points / n_samples) recorded
    on each window, so overlapping windows don't inflate significance."""
    a = _windows([1.0, 1.0], [2.0, 2.0], origins=[1, 2])  # n_points=5, n_samples=1
    result = metrics.compare_backtests(a, a)
    assert result.block_size == 5


def test_compare_backtests_raises_on_misaligned_origins():
    a = _windows([1.0], [2.0], origins=[100])
    b = _windows([1.0], [2.0], origins=[105])
    with pytest.raises(ValueError, match="origins"):
        metrics.compare_backtests(a, b)


def test_compare_backtests_raises_without_recorded_origins():
    """A `BacktestResult` built before PYQ-266 (or by hand) has no `origins` --
    refuse to treat the comparison as paired when alignment can't be verified,
    rather than silently trusting the caller."""
    a = _windows([1.0], [2.0], origins=[])
    b = _windows([1.0], [2.0], origins=[])
    with pytest.raises(ValueError, match="origins"):
        metrics.compare_backtests(a, b)


def test_compare_backtests_raises_when_only_one_side_has_recorded_origins():
    """PYQ-245's mutation survey: the existing 'without recorded origins' test
    leaves both sides empty, which can't distinguish `or` from `and` in
    `if not a.origins or not b.origins`. Exactly one side empty is the case
    that actually tells them apart -- with `and`, execution falls through to
    the misaligned-origins check below and still raises, but with the wrong,
    less specific message, silently losing the guard's own stated purpose."""
    a = _windows([], [], origins=[])
    b = _windows([1.0], [2.0], origins=[100])
    with pytest.raises(ValueError, match="requires both sides to carry recorded window origins"):
        metrics.compare_backtests(a, b)


def test_compare_backtests_excludes_zero_is_false_when_the_interval_straddles_it():
    rng = np.random.default_rng(0)
    n = 50
    origins = list(range(n))
    noise_a = rng.normal(0, 0.5, n)
    noise_b = rng.normal(0, 0.5, n)
    a = _windows((1.0 - noise_a).tolist(), [1.0] * n, origins=origins)
    b = _windows((1.0 - noise_b).tolist(), [1.0] * n, origins=origins)

    result = metrics.compare_backtests(a, b, block_size=1)

    assert result.excludes_zero is False


# --- PYQ-141: pooled headline skill vs. the per-window skill column ----------


def test_aggregate_metrics_pooled_skill_is_not_the_mean_of_per_window_skills():
    """The bug PYQ-141 documents: `aggregate_metrics().skill_vs_baseline` is a
    ratio of pooled means, `_per_window_table`'s column is a mean of per-window
    ratios once read down, and the two can disagree without limit. Construct
    windows with unequal baseline MAEs (equal point-weight, so the divergence
    isn't just weighting) and assert the two estimators are not equal --
    documenting the divergence is deliberate so it cannot be silently
    refactored into agreement."""
    windows = [
        metrics.EvaluationMetrics(
            model_mae=0.5,
            baseline_mae=1.0,
            directional_accuracy=0.5,
            calibration_coverage=0.8,
            n_samples=1,
            n_points=5,
        ),
        metrics.EvaluationMetrics(
            model_mae=90.0,
            baseline_mae=100.0,
            directional_accuracy=0.5,
            calibration_coverage=0.8,
            n_samples=1,
            n_points=5,
        ),
    ]
    per_window_skills = [w.skill_vs_baseline for w in windows]  # [0.5, 0.1]
    mean_of_ratios = sum(per_window_skills) / len(per_window_skills)

    pooled = metrics.aggregate_metrics(windows)
    ratio_of_means = pooled.skill_vs_baseline

    assert mean_of_ratios == pytest.approx(0.3)
    assert ratio_of_means != pytest.approx(mean_of_ratios)


# --- PYQ-270: confidence interval on skill, and the block-size fix -----------


def test_skill_confidence_interval_is_none_with_fewer_than_two_windows():
    windows = [
        metrics.EvaluationMetrics(
            model_mae=1.0, baseline_mae=2.0, directional_accuracy=0.5, calibration_coverage=0.8
        )
    ]
    assert metrics.skill_confidence_interval(windows) is None


def test_directional_accuracy_confidence_interval_is_none_with_fewer_than_two_windows():
    windows = [
        metrics.EvaluationMetrics(
            model_mae=1.0, baseline_mae=2.0, directional_accuracy=0.5, calibration_coverage=0.8
        )
    ]
    assert metrics.directional_accuracy_confidence_interval(windows) is None


def test_skill_confidence_interval_has_nonzero_width_at_the_default_window_count():
    """Regression for PYQ-270: the interval used to be bootstrapped with
    ``block_size = max(1, horizon)``, a point-level overlap correction applied
    to a series whose elements are already whole windows. At the project's own
    default (5 walk-forward windows, 5-day horizon) that made
    ``block_size == len(values)``, collapsing the "95% CI" to the one possible
    resample -- a zero-width interval reported as if it were informative. Five
    windows with distinct skill is exactly that default shape; the fixed
    (block_size=1) interval must have real width."""
    windows = [
        metrics.EvaluationMetrics(
            model_mae=m,
            baseline_mae=2.0,
            directional_accuracy=0.5,
            calibration_coverage=0.8,
            n_samples=1,
            n_points=5,
        )
        for m in [0.0, 0.4, 0.8, 1.2, 1.6]  # skill = [1.0, 0.8, 0.6, 0.4, 0.2]
    ]
    lo, hi = metrics.skill_confidence_interval(windows)
    assert hi > lo


def test_directional_accuracy_confidence_interval_has_nonzero_width_at_the_default_window_count():
    windows = [
        metrics.EvaluationMetrics(
            model_mae=1.0,
            baseline_mae=2.0,
            directional_accuracy=d,
            calibration_coverage=0.8,
            n_samples=1,
            n_points=5,
        )
        for d in [0.2, 0.4, 0.6, 0.8, 1.0]
    ]
    lo, hi = metrics.directional_accuracy_confidence_interval(windows)
    assert hi > lo


# --- PYQ-267: per-horizon-step breakdown --------------------------------------


def test_evaluate_predictions_recovers_a_skill_profile_that_varies_by_horizon_step():
    """Every top-level metric averages over h=1..horizon, discarding a profile
    like 'skill improves with horizon' entirely. A synthetic case with skill
    deliberately +1.0/0.0/-1.0 across three steps must not collapse to a mean
    -- the per-step values must be individually recoverable."""
    # step 1: model predicts perfectly (mae 0) while baseline mae is 1 -> skill +1.0.
    # step 2: model predicts "no change", identical to baseline -> skill 0.0.
    # step 3: model overshoots the move baseline (naively) gets half right -> skill -1.0.
    median = np.array([[101.0, 100.0, 130.0], [99.0, 100.0, 70.0]])
    lower, upper = median - 5.0, median + 5.0
    predictions = np.stack([lower, median, upper], axis=-1)
    actuals = np.array([[101.0, 105.0, 110.0], [99.0, 95.0, 90.0]])
    last_observed = np.array([100.0, 100.0])

    result = metrics.evaluate_predictions(predictions, actuals, last_observed, [0.1, 0.5, 0.9])

    assert [p.step for p in result.per_horizon] == [1, 2, 3]
    skills = [p.skill_vs_baseline for p in result.per_horizon]
    assert skills[0] == pytest.approx(1.0)
    assert skills[1] == pytest.approx(0.0)
    assert skills[2] == pytest.approx(-1.0)
    # The mean-over-horizon headline number hides the profile: it is neither
    # the best nor the worst step, and not obviously any of the three.
    assert result.skill_vs_baseline == pytest.approx(-0.5625)


def test_evaluate_predictions_per_horizon_mae_and_coverage_match_manual_per_step_calculation():
    predictions = np.array(
        [
            [[95.0, 101.0, 106.0], [96.0, 103.0, 109.0]],
            [[90.0, 98.0, 104.0], [88.0, 96.0, 102.0]],
        ]
    )
    actuals = np.array([[102.0, 104.0], [97.0, 95.0]])
    last_observed = np.array([100.0, 100.0])

    result = metrics.evaluate_predictions(predictions, actuals, last_observed, [0.1, 0.5, 0.9])

    assert len(result.per_horizon) == 2
    for h, step in enumerate(result.per_horizon):
        assert step.model_mae == pytest.approx(
            np.mean(np.abs(actuals[:, h] - predictions[:, h, 1]))
        )
        assert step.baseline_mae == pytest.approx(
            metrics.persistence_baseline_mae(actuals[:, h : h + 1], last_observed)
        )
        assert 0.0 <= step.calibration_coverage <= 1.0


def test_aggregate_metrics_pools_per_horizon_position_wise_weighted_by_n_samples():
    a = metrics.EvaluationMetrics(
        model_mae=1.0,
        baseline_mae=2.0,
        directional_accuracy=0.5,
        calibration_coverage=0.7,
        n_samples=4,
        n_points=8,
        per_horizon=[
            metrics.PerHorizonMetrics(
                1,
                model_mae=0.0,
                baseline_mae=1.0,
                directional_accuracy=1.0,
                calibration_coverage=1.0,
            ),
            metrics.PerHorizonMetrics(
                2,
                model_mae=10.0,
                baseline_mae=10.0,
                directional_accuracy=0.0,
                calibration_coverage=0.0,
            ),
        ],
    )
    b = metrics.EvaluationMetrics(
        model_mae=3.0,
        baseline_mae=4.0,
        directional_accuracy=0.9,
        calibration_coverage=0.9,
        n_samples=1,
        n_points=2,
        per_horizon=[
            metrics.PerHorizonMetrics(
                1,
                model_mae=4.0,
                baseline_mae=1.0,
                directional_accuracy=0.0,
                calibration_coverage=0.0,
            ),
            metrics.PerHorizonMetrics(
                2,
                model_mae=20.0,
                baseline_mae=10.0,
                directional_accuracy=1.0,
                calibration_coverage=1.0,
            ),
        ],
    )

    agg = metrics.aggregate_metrics([a, b])

    assert [p.step for p in agg.per_horizon] == [1, 2]
    # step 1, weighted by n_samples (4 vs 1): (4*0.0 + 1*4.0) / 5 = 0.8
    assert agg.per_horizon[0].model_mae == pytest.approx(0.8)
    assert agg.per_horizon[0].directional_accuracy == pytest.approx((4 * 1.0 + 1 * 0.0) / 5)
    # step 2: (4*10.0 + 1*20.0) / 5 = 12.0
    assert agg.per_horizon[1].model_mae == pytest.approx(12.0)
    assert agg.per_horizon[1].calibration_coverage == pytest.approx((4 * 0.0 + 1 * 1.0) / 5)


def test_aggregate_metrics_per_horizon_is_empty_when_no_window_has_it():
    """Windows built without PYQ-267's breakdown (e.g. hand-constructed
    EvaluationMetrics in older tests/scripts) must not crash aggregation."""
    a = metrics.EvaluationMetrics(
        model_mae=1.0, baseline_mae=2.0, directional_accuracy=0.5, calibration_coverage=0.7
    )
    agg = metrics.aggregate_metrics([a])
    assert agg.per_horizon == []


def test_evaluate_predictions_requires_0_5_quantile():
    predictions = np.zeros((1, 2, 2))
    actuals = np.zeros((1, 2))
    last_observed = np.zeros(1)
    with pytest.raises(ValueError, match="0.5"):
        metrics.evaluate_predictions(predictions, actuals, last_observed, [0.25, 0.75])


def test_warn_on_quantile_crossing_flags_crossed_band(caplog):
    """A higher quantile below a lower one must be surfaced, not silently scored (PYQ-216)."""
    # For one point the "p90" (last) is below the "p10" (first): a crossed band.
    predictions = np.array([[[106.0, 100.0, 95.0]]])
    with caplog.at_level("WARNING"):
        n = metrics.warn_on_quantile_crossing(predictions, [0.1, 0.5, 0.9])
    assert n > 0
    assert any("crossing" in m.lower() for m in caplog.messages)


def test_warn_on_quantile_crossing_silent_when_monotonic(caplog):
    predictions = np.array([[[95.0, 100.0, 106.0]]])  # properly ordered
    with caplog.at_level("WARNING"):
        n = metrics.warn_on_quantile_crossing(predictions, [0.1, 0.5, 0.9])
    assert n == 0
    assert not any("crossing" in m.lower() for m in caplog.messages)


def test_evaluate_predictions_warns_on_crossing(caplog):
    predictions = np.array([[[106.0, 100.0, 95.0]]])
    actuals = np.array([[100.0]])
    last_observed = np.array([100.0])
    with caplog.at_level("WARNING"):
        metrics.evaluate_predictions(predictions, actuals, last_observed, [0.1, 0.5, 0.9])
    assert any("crossing" in m.lower() for m in caplog.messages)


# --- PYQ-117: metrics must carry their own sample size -----------------------


def test_evaluate_predictions_records_sample_size():
    """A percentage without a denominator is not a result. 3 windows x 2 days = 6 points."""
    predictions = np.zeros((3, 2, 3))
    predictions[:, :, 0] = 90.0
    predictions[:, :, 1] = 100.0
    predictions[:, :, 2] = 110.0
    actuals = np.full((3, 2), 100.0)
    last_observed = np.full(3, 99.0)

    ev = metrics.evaluate_predictions(predictions, actuals, last_observed, [0.1, 0.5, 0.9])

    assert ev.n_samples == 3
    assert ev.n_points == 6


def test_aggregate_metrics_sums_sample_counts_rather_than_averaging_them():
    """Five windows of 5 points is 25 points of evidence, not 5."""
    windows = [
        metrics.EvaluationMetrics(1.0, 2.0, 0.5, 0.9, n_samples=1, n_points=5) for _ in range(5)
    ]
    agg = metrics.aggregate_metrics(windows)
    assert agg.n_samples == 5
    assert agg.n_points == 25
    # Equal-sized windows: pooling and averaging coincide.
    assert agg.directional_accuracy == 0.5


# --- PYQ-136: the aggregate rate must be pooled over the summed denominator ---


def test_aggregate_metrics_weights_rates_by_each_window_point_count():
    """A rate reported over 30 pooled points must be the rate *of* those 30 points.

    n_points sums (PYQ-117), so an unweighted mean of the rates reports a
    numerator and a denominator computed different ways.
    """
    big = metrics.EvaluationMetrics(
        model_mae=1.0,
        baseline_mae=2.0,
        directional_accuracy=1.0,
        calibration_coverage=1.0,
        n_samples=5,
        n_points=25,
    )
    small = metrics.EvaluationMetrics(
        model_mae=11.0,
        baseline_mae=12.0,
        directional_accuracy=0.0,
        calibration_coverage=0.0,
        n_samples=1,
        n_points=5,
    )

    agg = metrics.aggregate_metrics([big, small])

    # 25 hits out of 30 points, not the midpoint of 1.0 and 0.0.
    assert agg.directional_accuracy == pytest.approx(25 / 30)
    assert agg.calibration_coverage == pytest.approx(25 / 30)
    assert agg.model_mae == pytest.approx((25 * 1.0 + 5 * 11.0) / 30)
    assert agg.baseline_mae == pytest.approx((25 * 2.0 + 5 * 12.0) / 30)
    assert agg.n_points == 30


def test_aggregate_metrics_falls_back_to_unweighted_mean_without_point_counts():
    """Metrics constructed without n_points still aggregate rather than divide by zero."""
    a = metrics.EvaluationMetrics(1.0, 2.0, 0.4, 0.6)
    b = metrics.EvaluationMetrics(3.0, 4.0, 0.8, 1.0)

    agg = metrics.aggregate_metrics([a, b])

    assert agg.directional_accuracy == pytest.approx(0.6)
    assert agg.model_mae == pytest.approx(2.0)


def test_aggregate_metrics_raises_a_clear_error_on_an_empty_list():
    """PYQ-156: np.average([], weights=[]) used to raise a bare ZeroDivisionError,
    reachable via `pyquant backtest SYMBOL --windows 0` and not in cli/app.py's
    EXPECTED_FAILURES -- so it surfaced as a raw traceback, not a clean message."""
    with pytest.raises(ValueError):
        metrics.aggregate_metrics([])


# --- PYQ-252: CRPS, Winkler and PIT -------------------------------------------


def test_crps_and_winkler_match_hand_calculation():
    """Hand-computed on a 1x2 array so the constants are derived, not copied.

    quantiles p10/p50/p90; predictions [[8,10,12],[18,20,22]]; actuals [11,19].
      pinball(0.1) = mean(0.1*3, 0.9*1)      = 0.2
      pinball(0.5) = mean(0.5*1, 0.5*1)      = 0.5
      pinball(0.9) = mean(0.9*1, 0.1*9)/...  = 0.2
      CRPS = mean(0.2, 0.5, 0.2)             = 0.3
    Both actuals are inside [p10, p90], so Winkler is pure width:
      mean(12-8, 22-18) = 4.0
    """
    quantiles = [0.1, 0.5, 0.9]
    predictions = np.array([[[8.0, 10.0, 12.0], [18.0, 20.0, 22.0]]])
    actuals = np.array([[11.0, 19.0]])

    assert metrics.crps_from_quantiles(actuals, predictions, quantiles) == pytest.approx(0.3)
    assert metrics.winkler_score(
        actuals, predictions[:, :, 0], predictions[:, :, -1], alpha=0.2
    ) == pytest.approx(4.0)


def test_winkler_penalises_a_miss_far_more_than_the_width_it_saves():
    """The whole point of the interval score: a band cannot buy a good score by
    being narrow if it then misses, nor by being enormous (PYQ-252)."""
    tight_and_wrong = metrics.winkler_score(
        np.array([[10.0]]), np.array([[0.0]]), np.array([[1.0]]), alpha=0.2
    )
    wide_and_right = metrics.winkler_score(
        np.array([[10.0]]), np.array([[0.0]]), np.array([[20.0]]), alpha=0.2
    )
    # width 1 + 2/0.2 * 9 = 91  vs  width 20 + no penalty = 20
    assert tight_and_wrong == pytest.approx(91.0)
    assert wide_and_right == pytest.approx(20.0)
    assert tight_and_wrong > wide_and_right


def test_winkler_scores_an_overwide_band_worse_than_a_calibrated_one():
    """PYQ-117's 99.3%-on-a-nominal-80% band looks perfect by coverage and is
    nearly uninformative. Winkler must say so; coverage alone cannot."""
    actuals = np.array([[10.0], [11.0], [9.0]])
    calibrated_lo, calibrated_hi = np.full((3, 1), 8.0), np.full((3, 1), 12.0)
    overwide_lo, overwide_hi = np.full((3, 1), -100.0), np.full((3, 1), 100.0)

    assert metrics.calibration_coverage(actuals, calibrated_lo, calibrated_hi) == 1.0
    assert metrics.calibration_coverage(actuals, overwide_lo, overwide_hi) == 1.0  # identical
    assert metrics.winkler_score(
        actuals, calibrated_lo, calibrated_hi, 0.2
    ) < metrics.winkler_score(actuals, overwide_lo, overwide_hi, 0.2)


def test_pit_is_uniform_for_a_calibrated_forecaster_and_clustered_for_an_overwide_one():
    """PIT is uniform when calibrated and hump-shaped when underconfident --
    the shape expected here given a band covering 99.3% of a nominal 80%."""
    rng = np.random.default_rng(0)
    quantiles = [0.1, 0.5, 0.9]
    actuals = rng.normal(0, 1, (400, 1))

    calibrated = np.tile(
        np.array([np.percentile(rng.normal(0, 1, 100_000), [10, 50, 90])]), (400, 1, 1)
    )
    overwide = np.tile(np.array([[-50.0, 0.0, 50.0]]), (400, 1, 1))

    pit_calibrated = metrics.pit_values(actuals, calibrated, quantiles)
    pit_overwide = metrics.pit_values(actuals, overwide, quantiles)

    # Calibrated: PIT spreads across the unit interval.
    assert pit_calibrated.std() > 0.25
    # Over-wide: every outcome lands near the middle of a vastly-too-large band.
    assert pit_overwide.std() < 0.05
    assert abs(pit_overwide.mean() - 0.5) < 0.05


def test_pit_values_clamp_out_of_band_actuals_to_the_outer_quantile_exactly():
    """PYQ-153: an actual outside the predicted band saturates at exactly
    0.1/0.9 rather than extending into the tails, a real limitation now
    documented on `pit_values()` -- verified here so that claim can't
    silently drift from the implementation."""
    quantiles = [0.1, 0.5, 0.9]
    predictions = np.array([[10.0, 20.0, 30.0]])

    far_below = metrics.pit_values(np.array([-1000.0]), predictions, quantiles)
    far_above = metrics.pit_values(np.array([1000.0]), predictions, quantiles)

    assert far_below[0] == 0.1
    assert far_above[0] == 0.9


def test_crps_and_winkler_survive_weighted_aggregation_across_windows():
    """Both must flow through aggregate_metrics weighted by n_points (PYQ-136),
    and PIT values concatenate rather than average -- a pooled histogram is the
    point of collecting them."""
    quantiles = [0.1, 0.5, 0.9]
    preds = np.array([[[8.0, 10.0, 12.0], [18.0, 20.0, 22.0]]])
    a = metrics.evaluate_predictions(preds, np.array([[11.0, 19.0]]), np.array([10.0]), quantiles)
    b = metrics.evaluate_predictions(preds, np.array([[9.0, 21.0]]), np.array([10.0]), quantiles)

    pooled = metrics.aggregate_metrics([a, b])

    assert pooled.crps == pytest.approx((a.crps + b.crps) / 2)
    assert pooled.winkler_score == pytest.approx((a.winkler_score + b.winkler_score) / 2)
    assert len(pooled.pit) == len(a.pit) + len(b.pit) == 4
