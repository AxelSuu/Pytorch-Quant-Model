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


def test_fetch_prices_recovers_from_transient_failure(monkeypatch, sample_ohlcv_df):
    """A single transient yfinance failure must be retried, not hard-fail the
    whole panel build (PYQ-215)."""
    from pyquant.data import retry

    monkeypatch.setattr(retry, "_sleep", lambda _s: None)
    calls = {"n": 0}

    class FlakyTicker:
        def __init__(self, symbol):
            pass

        def history(self, period=None, start=None, end=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient network error")
            return sample_ohlcv_df.copy()

    monkeypatch.setattr(prices.yf, "Ticker", FlakyTicker)
    out = prices.fetch_prices("AAPL", use_indicators=False)
    assert calls["n"] == 2  # failed once, then succeeded
    assert not out.empty


def test_fetch_prices_raises_on_empty(monkeypatch):
    class EmptyTicker:
        def __init__(self, symbol):
            pass

        def history(self, **kwargs):
            return pd.DataFrame()

    monkeypatch.setattr(prices.yf, "Ticker", EmptyTicker)
    with pytest.raises(ValueError):
        prices.fetch_prices("BADSYM")


# --- PYQ-121: RSI must be Wilder's RSI, not a simple moving average ----------


def _wilder_rsi_reference(values, period=14):
    """Textbook Wilder RSI, written the slow obvious way as an independent check.

    SMA seed over the first `period` changes, then the recursive smoothed average
    ((period - 1) * prev + new) / period.
    """
    gains, losses = [], []
    for prev, cur in zip(values[:-1], values[1:], strict=True):
        change = cur - prev
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    def to_rsi(avg_gain, avg_loss):
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    out = [float("nan")] * len(values)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    out[period] = to_rsi(avg_gain, avg_loss)
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i + 1] = to_rsi(avg_gain, avg_loss)
    return out


def test_compute_rsi_matches_an_independent_wilder_implementation(sample_ohlcv_df):
    close = sample_ohlcv_df["Close"]
    expected = _wilder_rsi_reference(list(close.to_numpy()), period=14)
    actual = prices.compute_rsi(close, 14)
    np.testing.assert_allclose(
        actual.to_numpy()[14:], np.array(expected)[14:], rtol=1e-9, atol=1e-9
    )


def test_compute_rsi_warmup_rows_are_nan_not_fabricated(sample_ohlcv_df):
    """min_periods=1 used to emit a value from row 2 off a one-row window; those
    survived dropna() and were only removed because SMA_50 happened to cut them."""
    rsi = prices.compute_rsi(sample_ohlcv_df["Close"], 14)
    assert rsi.iloc[:14].isna().all()
    assert rsi.iloc[14:].notna().all()


def test_compute_rsi_is_100_when_price_only_rises():
    rising = pd.Series(np.arange(1.0, 40.0))
    rsi = prices.compute_rsi(rising, 14)
    np.testing.assert_allclose(rsi.dropna().to_numpy(), 100.0)


def test_compute_rsi_is_0_when_price_only_falls():
    falling = pd.Series(np.arange(40.0, 1.0, -1.0))
    rsi = prices.compute_rsi(falling, 14)
    np.testing.assert_allclose(rsi.dropna().to_numpy(), 0.0)
