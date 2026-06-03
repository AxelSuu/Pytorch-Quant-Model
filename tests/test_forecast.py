"""Tests for the Forecast dataclass + generate_forecast orchestration."""

import numpy as np
import pandas as pd

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
