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
    """Every indicator's warm-up must stay NaN so build_panel() can drop it,
    instead of being fabricated via bfill or emitted off a one-row window.

    SMA_50 needs 49 real days. The EMA pair used to emit a value from row 1 --
    literally ``close[0]``, an average of nothing -- and MACD inherited it
    (PYQ-132), which is the same defect PYQ-121 fixed for RSI_14.
    """
    out = prices.add_technical_indicators(sample_ohlcv_df)

    # (column, number of leading rows that are genuinely undefined)
    warmups = {
        "SMA_10": 9,
        "SMA_20": 19,
        "SMA_50": 49,
        "EMA_12": 11,
        "EMA_26": 25,
        "RSI_14": 14,
        # MACD needs the slow EMA; the signal line then needs 9 MACD values.
        "MACD": 25,
        "MACD_Signal": 33,
        "MACD_Hist": 33,
    }
    for column, warmup in warmups.items():
        assert out[column].iloc[:warmup].isna().all(), f"{column} fabricates warm-up values"
        assert out[column].iloc[warmup:].notna().all(), f"{column} is NaN past its warm-up"


def test_panel_warmup_is_decided_by_the_longest_window_not_by_sma_50(sample_ohlcv_df):
    """Each indicator must cut its own warm-up, not rely on SMA_50 cutting it.

    PYQ-121 called that "an accidental dependency between two unrelated
    indicators": SMA_50 happens to drop the first 49 rows, so a shorter
    indicator emitting garbage before its window filled was masked. Dropping
    SMA_50 must therefore still exclude the MACD warm-up (PYQ-132).
    """
    out = prices.add_technical_indicators(sample_ohlcv_df)
    positions = {
        c: out.index.get_loc(out[c].first_valid_index()) for c in prices.INDICATOR_COLUMNS
    }

    first_kept = out.index.get_loc(out.dropna().index[0])
    assert first_kept == max(positions.values())

    without_sma_50 = out.drop(columns=["SMA_50"])
    first_kept_reduced = out.index.get_loc(without_sma_50.dropna().index[0])
    expected = max(v for c, v in positions.items() if c != "SMA_50")
    assert first_kept_reduced == expected
    # The MACD signal line, not SMA_50, is what binds once SMA_50 is gone.
    assert expected == positions["MACD_Signal"]


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
