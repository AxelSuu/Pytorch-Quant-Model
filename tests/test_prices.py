"""Tests for price data + technical indicators."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yfinance

from pyquant.data import prices

FIXTURES = Path(__file__).parent / "fixtures"


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

    The exponential warm-ups are ``warmup_spans * span`` rather than one span:
    masking a single span stops a value being emitted off a one-row window but
    leaves the recursion's seed weighted 2.3% at the first surviving row, worth
    5.66% of MACD's own magnitude (PYQ-137). These lengths move with
    ``DEFAULT_EMA_WARMUP_SPANS``.
    """
    out = prices.add_technical_indicators(sample_ohlcv_df)
    spans = prices.DEFAULT_EMA_WARMUP_SPANS

    # (column, number of leading rows that are genuinely undefined)
    warmups = {
        "SMA_10": 9,
        "SMA_20": 19,
        "SMA_50": 49,
        "EMA_12": spans * 12 - 1,
        "EMA_26": spans * 26 - 1,
        "RSI_14": 14,
        # MACD needs the slow EMA; the signal line then needs its own warm-up
        # of MACD values on top.
        "MACD": spans * 26 - 1,
        "MACD_Signal": (spans * 26 - 1) + (spans * 9 - 1),
        "MACD_Hist": (spans * 26 - 1) + (spans * 9 - 1),
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

        def history(self, period=None, start=None, end=None, auto_adjust=None, **kwargs):
            # yfinance returns extra columns + tz-aware index
            df = sample_ohlcv_df.copy()
            df["Dividends"] = 0.0
            df.index = df.index.tz_localize("America/New_York")
            return df

    monkeypatch.setattr(yfinance, "Ticker", FakeTicker)
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

        def history(self, period=None, start=None, end=None, auto_adjust=None, **kwargs):
            received.update(period=period, start=start, end=end)
            return sample_ohlcv_df.copy()

    monkeypatch.setattr(yfinance, "Ticker", FakeTicker)
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

        def history(self, period=None, start=None, end=None, auto_adjust=None, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient network error")
            return sample_ohlcv_df.copy()

    monkeypatch.setattr(yfinance, "Ticker", FlakyTicker)
    out = prices.fetch_prices("AAPL", use_indicators=False)
    assert calls["n"] == 2  # failed once, then succeeded
    assert not out.empty


def test_fetch_prices_raises_on_empty(monkeypatch):
    class EmptyTicker:
        def __init__(self, symbol):
            pass

        def history(self, **kwargs):
            return pd.DataFrame()

    monkeypatch.setattr(yfinance, "Ticker", EmptyTicker)
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


# --- PYQ-135: no inf may reach the panel -------------------------------------


def test_volume_change_is_nan_not_inf_on_a_zero_volume_session(sample_ohlcv_df):
    """A halted/gapped session reported as Volume=0 makes the *next* row's
    pct_change divide by zero. inf is not NaN, so it survives build_panel()'s
    dropna() and poisons GroupNormalizer's fitted scale (PYQ-135)."""
    df = sample_ohlcv_df.copy()
    df.iloc[60, df.columns.get_loc("Volume")] = 0.0

    out = prices.add_technical_indicators(df)

    assert not np.isinf(out["Volume_Change"]).any()
    assert np.isnan(out["Volume_Change"].iloc[61])  # the divide-by-zero row
    assert not np.isinf(out.select_dtypes("number").to_numpy()).any()
    assert len(out.dropna()) > 0  # the row is dropped, the panel survives


def test_no_indicator_column_emits_inf_for_a_flat_or_zero_series():
    """Belt-and-braces: the whole indicator block is inf-free even on degenerate
    input, so a new indicator cannot reintroduce the PYQ-135 class silently."""
    n = 200
    idx = pd.bdate_range("2023-01-02", periods=n, name="Date")
    df = pd.DataFrame(
        {"Open": 10.0, "High": 10.0, "Low": 10.0, "Close": 10.0, "Volume": 0.0}, index=idx
    )

    out = prices.add_technical_indicators(df)

    assert not np.isinf(out.select_dtypes("number").to_numpy()).any()


# --- PYQ-137: EMA warm-up must be long enough that the seed has decayed -------


def _full_history_ema(n_prior: int, window: pd.Series, span: int) -> pd.Series:
    """EMA_``span`` computed with ``n_prior`` extra rows of history in front.

    This is the reference PYQ-137 actually cares about: what the indicator would
    read on a charting package that has more history than our panel starts with.
    """
    rng = np.random.default_rng(99)
    prior = pd.Series(float(window.iloc[0]) + np.cumsum(rng.normal(0, 1, n_prior)))
    full = pd.concat([prior, window], ignore_index=True)
    return full.ewm(span=span, adjust=False).mean().iloc[n_prior:].reset_index(drop=True)


def test_first_surviving_ema_row_matches_a_full_history_reference(sample_ohlcv_df):
    """From the first surviving panel row onward, EMA_26 must agree with an EMA
    that had ample prior history to within 0.05% of price.

    PYQ-132 masked the first ``span`` outputs but did not change the recursion,
    which ``adjust=False`` seeds at ``close[0]``; the first rows that *were*
    emitted still carried the seed (PYQ-137). A four-span warm-up decays it.
    """
    window = sample_ohlcv_df["Close"].reset_index(drop=True)
    reference = _full_history_ema(3000, window, span=26)

    out = prices.add_technical_indicators(sample_ohlcv_df)
    ema = out["EMA_26"].reset_index(drop=True)
    first = int(ema.notna().idxmax())

    relative_error = (ema.iloc[first:] - reference.iloc[first:]).abs() / window.iloc[first:]
    assert relative_error.max() < 0.0005, (
        f"EMA_26 at the first surviving row (index {first}) is "
        f"{relative_error.iloc[0] * 100:.4f}% off a full-history reference"
    )


def test_ema_warmup_spans_is_configurable_and_trades_rows_for_accuracy(sample_ohlcv_df):
    """The warm-up length is a tunable, not a constant: a shorter one keeps more
    rows and a longer one is strictly more accurate at the front."""
    short = prices.add_technical_indicators(sample_ohlcv_df, warmup_spans=1)
    long = prices.add_technical_indicators(sample_ohlcv_df, warmup_spans=4)

    assert len(short.dropna()) > len(long.dropna())
    assert short["EMA_26"].first_valid_index() < long["EMA_26"].first_valid_index()


def test_fetch_prices_parses_a_real_recorded_yfinance_payload(monkeypatch):
    """PYQ-243: every other test here mocks at our own function boundary, which
    verifies our logic against our own assumptions about the payload -- it cannot
    catch the failure that actually happens in production, the vendor changing its
    response shape (PYQ-228: yfinance silently jumped 0.2.x -> 1.4.1 and flipped
    auto_adjust's default mid-series). This drives the real fetch_prices()/
    normalize_ohlcv() path from one real recorded Ticker.history() response instead
    of a hand-built frame.
    """
    real_response = pd.read_pickle(FIXTURES / "yfinance_prices_aapl.pkl")
    # The real payload carries Dividends/Stock Splits alongside OHLCV -- a detail a
    # hand-built fixture would only include if someone thought to add it.
    assert "Dividends" in real_response.columns

    class RecordedTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            return real_response

    monkeypatch.setattr(yfinance, "Ticker", RecordedTicker)

    out = prices.fetch_prices("AAPL", period="3mo")

    assert list(out.columns[:5]) == ["Open", "High", "Low", "Close", "Volume"]
    assert "Dividends" not in out.columns
    assert "Stock Splits" not in out.columns
    assert out.index.tz is None
    assert out.index.is_monotonic_increasing
    for col in prices.INDICATOR_COLUMNS:
        assert col in out.columns
