"""Tests for the Forecast dataclass + generate_forecast orchestration."""

import numpy as np
import pandas as pd
import pytest

from pyquant.analysis import forecast as fc_mod
from pyquant.analysis.forecast import Forecast


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
    monkeypatch.setattr(
        fc_mod.tft, "predict_quantiles", lambda b, df: np.ones((5, 3)) * 100.0
    )

    fc = fc_mod.generate_forecast("test", object())
    assert fc.symbol == "TEST"
    assert fc.horizon == 5
    assert fc.current_price == float(panel["Close"].iloc[-1])


def test_generate_forecast_forwards_pin_to_build_panel(monkeypatch, sample_ohlcv_df):
    from pyquant.data.prices import add_technical_indicators

    panel = add_technical_indicators(sample_ohlcv_df)
    received = {}

    def fake_build_panel(symbol, settings, pin=None):
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

    def fake_build_panel(symbol, settings, pin=None):
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
