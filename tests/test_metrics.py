"""Tests for evaluation metrics: persistence baseline, directional accuracy, calibration."""

import numpy as np
import pytest

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
    assert abs(result.baseline_mae - metrics.persistence_baseline_mae(actuals, last_observed)) < 1e-9
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
