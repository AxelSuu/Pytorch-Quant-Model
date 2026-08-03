"""Tests for the Forecast dataclass + generate_forecast orchestration."""

import numpy as np
import pandas as pd
import pytest

from pyquant.analysis import forecast as fc_mod
from pyquant.analysis.forecast import (
    Forecast,
    log_return_quantiles_to_price_band,
    log_returns_to_prices,
)


def _make_forecast():
    dates = pd.bdate_range("2024-01-01", periods=30)
    history = pd.Series(np.linspace(100, 110, 30), index=dates)
    preds = np.array(
        [
            [98.0, 105.0, 112.0],
            [97.0, 106.0, 114.0],
            [96.0, 107.0, 116.0],
        ]
    )
    return Forecast(
        symbol="TEST",
        last_date=dates[-1],
        current_price=110.0,
        quantiles=[0.1, 0.5, 0.9],
        predictions=preds,
        history=history,
    )


def test_forecast_horizon_and_median():
    fc = _make_forecast()
    assert fc.horizon == 3
    np.testing.assert_array_equal(fc.median, [105.0, 106.0, 107.0])


def test_quantile_series_selects_correct_column():
    fc = _make_forecast()
    np.testing.assert_array_equal(fc.quantile_series(0.9), [112.0, 114.0, 116.0])


def test_expected_return_pct():
    fc = _make_forecast()
    # final median 107 vs current 110 -> ~ -2.7%
    assert abs(fc.expected_return_pct() - (-2.7272)) < 1e-3


def test_log_returns_to_prices_compounds_a_single_path_exactly():
    """log_returns_to_prices is for one deterministic path (e.g. the realized
    actual), where cumsum is exactly correct -- no quantile/dispersion question
    applies to a single path."""
    returns = np.log(np.array([101.0, 104.0]) / np.array([100.0, 101.0]))
    prices = log_returns_to_prices(returns, 100.0)
    np.testing.assert_allclose(prices, [101.0, 104.0])


def test_log_return_quantile_band_does_not_reproduce_old_naive_cumsum_band():
    """PYQ-142: naively cumsum-ing each quantile column independently (the old
    behavior) overstates the h-step band width by ~sqrt(h). The fixed
    reconstruction must not reproduce that band."""
    quantiles = [0.1, 0.5, 0.9]
    horizon, sigma = 5, 0.02
    # Same per-step quantiles at every step (iid steps).
    per_step = np.array([-1.2816 * sigma, 0.0, 1.2816 * sigma])
    log_return_quantiles = np.tile(per_step, (horizon, 1))

    band = log_return_quantiles_to_price_band(log_return_quantiles, 100.0, quantiles)
    naive = log_returns_to_prices(log_return_quantiles, 100.0)  # old buggy call shape

    fixed_p90_pct = (band[-1, -1] - 100.0) / 100.0
    naive_p90_pct = (naive[-1, -1] - 100.0) / 100.0
    # The naive band is the ~sqrt(h) too-wide one the ticket measured.
    assert naive_p90_pct > fixed_p90_pct * 1.5


def test_log_return_quantile_band_matches_analytic_iid_normal_quantiles():
    """Under an iid-per-step-normal assumption, the true h-step p90 edge is
    z_0.9 * sigma * sqrt(h). If the model's own per-step quantiles are exactly
    the true per-step normal quantiles, the fixed reconstruction should recover
    that analytic h-step edge (this is the sqrt(sum of squared per-step
    deviations) identity, not a Monte Carlo approximation)."""
    quantiles = [0.1, 0.5, 0.9]
    horizon, sigma = 5, 0.02
    z = np.array([-1.2815515655446004, 0.0, 1.2815515655446004])  # norm.ppf([0.1, 0.5, 0.9])
    per_step = z * sigma
    log_return_quantiles = np.tile(per_step, (horizon, 1))

    band = log_return_quantiles_to_price_band(log_return_quantiles, 100.0, quantiles)

    for h in range(1, horizon + 1):
        expected_log_return = z[-1] * sigma * np.sqrt(h)
        actual_log_return = np.log(band[h - 1, -1] / 100.0)
        assert abs(actual_log_return - expected_log_return) < 1e-9


def test_log_return_quantile_band_achieves_close_to_nominal_coverage():
    """Monte Carlo coverage check (PYQ-142 acceptance criterion): simulate iid
    per-step normal log-returns, build the p10-p90 band from the true per-step
    quantiles, and confirm empirical h-step coverage is close to the nominal
    80%, unlike the naive cumsum band (which the ticket measured at up to
    ~2.2x too wide, i.e. badly over-covering)."""
    rng = np.random.default_rng(0)
    quantiles = [0.1, 0.5, 0.9]
    horizon, sigma, n_sims = 5, 0.02, 20000
    z = np.array([-1.2815515655446004, 0.0, 1.2815515655446004])  # norm.ppf([0.1, 0.5, 0.9])
    per_step = z * sigma
    log_return_quantiles = np.tile(per_step, (horizon, 1))
    band = log_return_quantiles_to_price_band(log_return_quantiles, 100.0, quantiles)

    steps = rng.normal(0.0, sigma, size=(n_sims, horizon))
    cum_actual_log_return = np.cumsum(steps, axis=1)
    final_actual = 100.0 * np.exp(cum_actual_log_return[:, -1])

    covered = (final_actual >= band[-1, 0]) & (final_actual <= band[-1, -1])
    empirical_coverage = covered.mean()
    assert abs(empirical_coverage - 0.8) < 0.03


def test_median_raises_clear_error_when_0_5_not_configured():
    """0.5 // 2 would silently pick p75 for [0.05,0.25,0.75,0.95] -- must not."""
    dates = pd.bdate_range("2024-01-01", periods=3)
    fc = Forecast(
        symbol="TEST",
        last_date=dates[-1],
        current_price=100.0,
        quantiles=[0.05, 0.25, 0.75, 0.95],
        predictions=np.array([[90, 95, 105, 110]] * 3),
        history=pd.Series([100.0] * 3, index=dates),
    )
    with pytest.raises(ValueError, match="0.5"):
        _ = fc.median


def test_generate_forecast_orchestration(monkeypatch, sample_ohlcv_df):
    from pyquant.data.prices import add_technical_indicators

    panel = add_technical_indicators(sample_ohlcv_df)
    monkeypatch.setattr(fc_mod, "build_panel", lambda *a, **k: panel)
    monkeypatch.setattr(fc_mod, "panel_to_long", lambda p, s: p)

    class FakeBundle:
        meta = {"quantiles": [0.1, 0.5, 0.9]}

    monkeypatch.setattr(fc_mod.tft, "load", lambda *a, **k: FakeBundle())
    monkeypatch.setattr(fc_mod.tft, "predict_quantiles", lambda b, df: np.ones((5, 3)) * 100.0)

    fc = fc_mod.generate_forecast("test", object())
    assert fc.symbol == "TEST"
    assert fc.horizon == 5
    assert fc.current_price == float(panel["Close"].iloc[-1])


def test_generate_forecast_forwards_pin_to_build_panel(monkeypatch, sample_ohlcv_df):
    from pyquant.data.prices import add_technical_indicators

    panel = add_technical_indicators(sample_ohlcv_df)
    received = {}

    def fake_build_panel(symbol, settings, end=None, pin=None):
        received["pin"] = pin
        return panel

    monkeypatch.setattr(fc_mod, "build_panel", fake_build_panel)
    monkeypatch.setattr(fc_mod, "panel_to_long", lambda p, s: p)

    class FakeBundle:
        meta = {"quantiles": [0.1, 0.5, 0.9]}

    monkeypatch.setattr(fc_mod.tft, "load", lambda *a, **k: FakeBundle())
    monkeypatch.setattr(fc_mod.tft, "predict_quantiles", lambda b, df: np.ones((5, 3)) * 100.0)

    fc_mod.generate_forecast("test", object(), pin="exp-1")
    assert received["pin"] == "exp-1"


def test_generate_forecast_forwards_end_to_build_panel(monkeypatch, sample_ohlcv_df):
    """PYQ-284: `end` is what lets a caller simulate forecasting as of a past date --
    it must reach build_panel verbatim, with no shifting applied here."""
    from pyquant.data.prices import add_technical_indicators

    panel = add_technical_indicators(sample_ohlcv_df)
    received = {}

    def fake_build_panel(symbol, settings, end=None, pin=None):
        received["end"] = end
        return panel

    monkeypatch.setattr(fc_mod, "build_panel", fake_build_panel)
    monkeypatch.setattr(fc_mod, "panel_to_long", lambda p, s: p)

    class FakeBundle:
        meta = {"quantiles": [0.1, 0.5, 0.9]}

    monkeypatch.setattr(fc_mod.tft, "load", lambda *a, **k: FakeBundle())
    monkeypatch.setattr(fc_mod.tft, "predict_quantiles", lambda b, df: np.ones((5, 3)) * 100.0)

    fc_mod.generate_forecast("test", object(), end="2026-07-29")
    assert received["end"] == "2026-07-29"


# --- PYQ-115: the forecast must be labelled with the dates it is actually for --


def test_forecast_dates_are_the_business_days_after_the_last_observed_bar():
    fc = _make_forecast()  # last_date = 2024-02-09 (Fri), horizon 3
    dates = fc.forecast_dates
    assert len(dates) == fc.horizon
    assert (dates > fc.last_date).all()
    assert all(d.dayofweek < 5 for d in dates)


def test_forecast_dates_match_the_rows_appended_for_prediction():
    """The display dates and the rows the model actually decoded must agree."""
    from pyquant.data.dataset import future_business_dates

    fc = _make_forecast()
    expected = future_business_dates(fc.last_date, fc.horizon)
    pd.testing.assert_index_equal(pd.DatetimeIndex(fc.forecast_dates), expected)


# --- PYQ-119: forecasting must use the config the bundle was trained with ----


def test_generate_forecast_rebuilds_the_panel_with_the_bundles_recorded_config(
    monkeypatch, sample_ohlcv_df
):
    """Train with --no-sectors then forecast without flags: the panel must still
    be built with sectors off, or the feature schemas differ by construction."""
    from pyquant.config import Settings
    from pyquant.data.prices import add_technical_indicators

    panel = add_technical_indicators(sample_ohlcv_df)
    seen = {}

    def fake_build_panel(symbol, settings, end=None, pin=None):
        seen["use_sectors"] = settings.data.use_sectors
        return panel

    monkeypatch.setattr(fc_mod, "build_panel", fake_build_panel)
    monkeypatch.setattr(fc_mod, "panel_to_long", lambda p, s: p)

    class FakeBundle:
        meta = {"quantiles": [0.1, 0.5, 0.9], "config": {"data": {"use_sectors": False}}}

    monkeypatch.setattr(fc_mod.tft, "load", lambda *a, **k: FakeBundle())
    monkeypatch.setattr(fc_mod.tft, "predict_quantiles", lambda b, df: np.ones((5, 3)) * 100.0)

    settings = Settings()
    assert settings.data.use_sectors is True  # live default disagrees with the bundle

    fc_mod.generate_forecast("test", settings)

    assert seen["use_sectors"] is False
