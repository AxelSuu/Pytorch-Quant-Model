"""Tests for interpretability: single-fetch panel reuse (PYQ-105)."""

import numpy as np
import pandas as pd

from pyquant.analysis import interpret as interp_mod
from pyquant.analysis.interpret import Interpretation, attention_to_series


def test_explain_forecast_reuses_the_panel_it_built(monkeypatch, sample_ohlcv_df):
    from pyquant.data.prices import add_technical_indicators

    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    build_panel_calls = []

    def fake_build_panel(*a, **k):
        build_panel_calls.append(1)
        return panel

    monkeypatch.setattr(interp_mod, "build_panel", fake_build_panel)
    monkeypatch.setattr(interp_mod, "panel_to_long", lambda p, s: p)

    class FakeBundle:
        meta: dict = {}  # a real ModelBundle always carries meta (PYQ-119)

    monkeypatch.setattr(
        interp_mod.tft,
        "interpret",
        lambda bundle, df: {
            "encoder_importance": {"RSI_14": 1.0},
            "attention": np.array([0.2, 0.3, 0.5]),
        },
    )

    result = interp_mod.explain_forecast("test", object(), bundle=FakeBundle())

    assert len(build_panel_calls) == 1  # build_panel() fetched exactly once
    assert result.panel_index is not None
    pd.testing.assert_index_equal(result.panel_index, panel.index)


def test_attention_to_series_uses_interpretations_own_panel_index():
    dates = pd.bdate_range("2024-01-01", periods=10)
    interp = Interpretation(
        symbol="TEST",
        feature_importance={"RSI_14": 1.0},
        attention=np.array([0.2, 0.3, 0.5]),
        panel_index=dates,
    )
    att = attention_to_series(interp)
    pd.testing.assert_index_equal(att.index, dates[-3:])


def test_explain_forecast_rebuilds_the_panel_with_the_bundles_recorded_config(
    monkeypatch, sample_ohlcv_df
):
    """PYQ-119: explain shares the forecast's schema problem, so it needs the fix too."""
    from pyquant.config import Settings
    from pyquant.data.prices import add_technical_indicators

    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    seen = {}

    def fake_build_panel(symbol, settings, *a, **k):
        seen["use_sectors"] = settings.data.use_sectors
        return panel

    monkeypatch.setattr(interp_mod, "build_panel", fake_build_panel)
    monkeypatch.setattr(interp_mod, "panel_to_long", lambda p, s: p)
    monkeypatch.setattr(
        interp_mod.tft,
        "interpret",
        lambda bundle, df: {"encoder_importance": {"RSI_14": 1.0}, "attention": np.array([1.0])},
    )

    class FakeBundle:
        meta = {"config": {"data": {"use_sectors": False}}}

    settings = Settings()
    assert settings.data.use_sectors is True

    interp_mod.explain_forecast("test", settings, bundle=FakeBundle())

    assert seen["use_sectors"] is False
