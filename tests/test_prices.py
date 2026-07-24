"""Tests for price data + technical indicators."""

import numpy as np
import pandas as pd
import pytest

from pyquant.data import prices


def test_rsi_in_range(sample_ohlcv_df):
    rsi = prices.compute_rsi(sample_ohlcv_df["Close"])
    assert rsi.dropna().between(0, 100).all()


def test_macd_histogram_is_macd_minus_signal(sample_ohlcv_df):
    macd, signal, hist = prices.compute_macd(sample_ohlcv_df["Close"])
    np.testing.assert_allclose(hist.values, (macd - signal).values, rtol=1e-9)


def test_bollinger_percent_b_shape(sample_ohlcv_df):
    width, pctb = prices.compute_bollinger_bands(sample_ohlcv_df["Close"])
    assert len(width) == len(sample_ohlcv_df)
    assert len(pctb) == len(sample_ohlcv_df)


def test_add_technical_indicators_adds_all_columns(sample_ohlcv_df):
    out = prices.add_technical_indicators(sample_ohlcv_df)
    for col in prices.INDICATOR_COLUMNS:
        assert col in out.columns


def test_add_technical_indicators_leaves_warmup_rows_genuinely_nan(sample_ohlcv_df):
    """SMA_50 needs 49 real days of history -- those rows must stay NaN so
    build_panel() can drop them, instead of being fabricated via bfill."""
    out = prices.add_technical_indicators(sample_ohlcv_df)
    assert out["SMA_50"].iloc[:49].isna().all()
    assert out["SMA_50"].iloc[49:].notna().all()


def test_add_technical_indicators_does_not_mutate_input(sample_ohlcv_df):
    before = set(sample_ohlcv_df.columns)
    prices.add_technical_indicators(sample_ohlcv_df)
    assert set(sample_ohlcv_df.columns) == before


def test_fetch_prices_uses_yfinance(monkeypatch, sample_ohlcv_df):
    class FakeTicker:
        def __init__(self, symbol):
            pass

        def history(self, period=None, start=None, end=None):
            # yfinance returns extra columns + tz-aware index
            df = sample_ohlcv_df.copy()
            df["Dividends"] = 0.0
            df.index = df.index.tz_localize("America/New_York")
            return df

    monkeypatch.setattr(prices.yf, "Ticker", FakeTicker)
    out = prices.fetch_prices("AAPL", use_indicators=True)
    assert out.index.tz is None  # normalised to tz-naive
    assert "RSI_14" in out.columns
    assert {"Open", "High", "Low", "Close", "Volume"}.issubset(out.columns)


def test_fetch_prices_honors_start_without_end(monkeypatch, sample_ohlcv_df):
    """Passing only `start` must use the date range, not silently fall back to
    `period` and discard it (PYQ-112)."""
    received = {}

    class FakeTicker:
        def __init__(self, symbol):
            pass

        def history(self, period=None, start=None, end=None):
            received.update(period=period, start=start, end=end)
            return sample_ohlcv_df.copy()

    monkeypatch.setattr(prices.yf, "Ticker", FakeTicker)
    prices.fetch_prices("AAPL", start="2020-01-01", use_indicators=False)
    assert received["start"] == "2020-01-01"
    assert received["period"] is None  # period path not taken


def test_fetch_prices_raises_on_empty(monkeypatch):
    class EmptyTicker:
        def __init__(self, symbol):
            pass

        def history(self, **kwargs):
            return pd.DataFrame()

    monkeypatch.setattr(prices.yf, "Ticker", EmptyTicker)
    with pytest.raises(ValueError):
        prices.fetch_prices("BADSYM")
