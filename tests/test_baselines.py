"""Tests for analysis/baselines.py: comparators beyond persistence (PYQ-275)."""

import numpy as np
import pytest

from pyquant.analysis import baselines
from pyquant.analysis.metrics import persistence_baseline_mae


def test_persistence_baseline_predicts_last_value_flat():
    history = np.array([[10.0, 20.0, 30.0], [5.0, 5.0, 5.0]])
    forecast = baselines.PersistenceBaseline().predict(history, horizon=3)
    np.testing.assert_allclose(forecast, [[30.0, 30.0, 30.0], [5.0, 5.0, 5.0]])


def test_persistence_baseline_mae_matches_the_existing_metrics_function():
    """The new baseline must reproduce the project's original comparator exactly,
    not a subtly different one -- otherwise the "strongest baseline" comparison
    below would be comparing against two different persistence baselines."""
    rng = np.random.default_rng(0)
    history = rng.normal(100, 5, size=(20, 10))
    actuals = rng.normal(100, 5, size=(20, 4))

    forecast = baselines.PersistenceBaseline().predict(history, horizon=4)
    mae = float(np.mean(np.abs(actuals - forecast)))

    assert mae == pytest.approx(persistence_baseline_mae(actuals, history[:, -1]))


def test_random_walk_with_drift_extrapolates_the_observed_linear_trend():
    # Exactly +10/step: drift should be recovered exactly and extrapolated.
    history = np.array([[10.0, 20.0, 30.0, 40.0, 50.0]])
    forecast = baselines.RandomWalkWithDriftBaseline().predict(history, horizon=3)
    np.testing.assert_allclose(forecast, [[60.0, 70.0, 80.0]])


def test_random_walk_with_drift_does_not_divide_by_zero_on_single_step_history():
    history = np.array([[42.0]])
    forecast = baselines.RandomWalkWithDriftBaseline().predict(history, horizon=2)
    np.testing.assert_allclose(forecast, [[42.0, 42.0]])


def test_seasonal_naive_repeats_the_value_from_one_season_ago():
    # n=10, season_length=5: the last 5 values are "this week"; forecast steps
    # 1..5 must repeat them in order, and step 6 must wrap back to step 1's value.
    history = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 50.0]])
    forecast = baselines.SeasonalNaiveBaseline(season_length=5).predict(history, horizon=6)
    np.testing.assert_allclose(forecast, [[10.0, 20.0, 30.0, 40.0, 50.0, 10.0]])


def test_seasonal_naive_degrades_gracefully_when_history_is_shorter_than_the_season():
    history = np.array([[1.0, 2.0, 3.0]])  # only 3 points, season_length=5
    forecast = baselines.SeasonalNaiveBaseline(season_length=5).predict(history, horizon=3)
    # Clamped to season_length=3 (the whole history): repeats [1, 2, 3].
    np.testing.assert_allclose(forecast, [[1.0, 2.0, 3.0]])


def test_climatological_baseline_predicts_the_historical_mean_flat_across_horizon():
    history = np.array([[10.0, 20.0, 30.0]])  # mean 20
    forecast = baselines.ClimatologicalBaseline().predict(history, horizon=4)
    np.testing.assert_allclose(forecast, [[20.0, 20.0, 20.0, 20.0]])


def test_ar1_baseline_recovers_a_noiseless_ar1_process():
    """y[i+1] = 0.5*y[i] + 3, generated with no noise -- OLS must recover
    phi=0.5, c=3 to float precision and continue the exact same recursion."""
    phi_true, c_true = 0.5, 3.0
    y = [10.0]
    for _ in range(19):
        y.append(phi_true * y[-1] + c_true)
    history = np.array([y])

    forecast = baselines.AR1Baseline().predict(history, horizon=3)

    expected = []
    value = y[-1]
    for _ in range(3):
        value = c_true + phi_true * value
        expected.append(value)
    np.testing.assert_allclose(forecast[0], expected, rtol=1e-6)


def test_ar1_baseline_does_not_crash_on_degenerate_short_or_constant_history():
    flat = baselines.AR1Baseline().predict(np.array([[5.0, 5.0, 5.0, 5.0]]), horizon=2)
    assert np.all(np.isfinite(flat))
    single = baselines.AR1Baseline().predict(np.array([[7.0]]), horizon=2)
    np.testing.assert_allclose(single, [[7.0, 7.0]])


def test_baseline_maes_covers_every_default_baseline():
    rng = np.random.default_rng(1)
    history = rng.normal(100, 5, size=(15, 20))
    actuals = rng.normal(100, 5, size=(15, 5))

    result = baselines.baseline_maes(actuals, history)

    expected_names = {b.name for b in baselines.DEFAULT_BASELINES}
    assert set(result) == expected_names
    assert all(mae >= 0.0 for mae in result.values())
    assert "persistence" in result


def test_baseline_maes_uses_only_the_baselines_it_is_given():
    rng = np.random.default_rng(2)
    history = rng.normal(0, 1, size=(5, 10))
    actuals = rng.normal(0, 1, size=(5, 3))

    result = baselines.baseline_maes(actuals, history, [baselines.PersistenceBaseline()])

    assert set(result) == {"persistence"}
