"""Tests for macro features (graceful degradation)."""

import pandas as pd

from pyquant.data import macro


def _fake_vix_ticker(sample_index):
    class FakeTicker:
        def __init__(self, symbol):
            pass

        def history(self, period=None, start=None, end=None):
            return pd.DataFrame({"Close": range(len(sample_index))}, index=sample_index)

    return FakeTicker


def test_fetch_macro_without_key_returns_vix_only(monkeypatch, sample_ohlcv_df):
    monkeypatch.setattr(macro.yf, "Ticker", _fake_vix_ticker(sample_ohlcv_df.index))
    out = macro.fetch_macro(api_key=None)
    assert "VIX" in out.columns
    assert "FedFunds" not in out.columns


def test_fetch_macro_handles_vix_failure(monkeypatch):
    class Boom:
        def __init__(self, symbol):
            raise RuntimeError("network down")

    monkeypatch.setattr(macro.yf, "Ticker", Boom)
    out = macro.fetch_macro(api_key=None)
    assert out.empty


def test_fetch_macro_with_key_adds_fred(monkeypatch, sample_ohlcv_df):
    monkeypatch.setattr(macro.yf, "Ticker", _fake_vix_ticker(sample_ohlcv_df.index))

    class FakeFred:
        def __init__(self, api_key=None):
            pass

        def get_series(self, series_id, observation_start=None, observation_end=None):
            return pd.Series(
                range(len(sample_ohlcv_df.index)), index=sample_ohlcv_df.index
            )

    import fredapi

    monkeypatch.setattr(fredapi, "Fred", FakeFred)
    out = macro.fetch_macro(api_key="dummy")
    assert "VIX" in out.columns
    assert "FedFunds" in out.columns
    assert "YieldSpread" in out.columns
