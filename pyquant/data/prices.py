"""Price data and technical indicators (Yahoo Finance).

Returns a date-indexed DataFrame, the building block for the unified panel in
:mod:`pyquant.data.dataset`. Technical-indicator logic is ported from the
original PyStock ``src/data.py``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf

from pyquant.data.retry import with_retry

logger = logging.getLogger(__name__)

# Columns produced by add_technical_indicators (excluding base OHLCV).
INDICATOR_COLUMNS = [
    "SMA_10",
    "SMA_20",
    "SMA_50",
    "EMA_12",
    "EMA_26",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "BB_Width",
    "BB_PercentB",
    "Price_Change",
    "Volume_Change",
    "Realized_Vol_20",
]


def _wilder_average(changes: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothed average: SMA seed, then ``((n-1)*prev + new) / n``.

    Not the same as an exponential mean with ``alpha=1/period`` and no seed, and
    not the same as a rolling mean -- the seed is what makes this match the RSI
    every charting package plots.
    """
    out = pd.Series(np.nan, index=changes.index, dtype=float)
    valid = changes.dropna()
    if len(valid) < period:
        return out
    average = float(valid.iloc[:period].mean())
    out.loc[valid.index[period - 1]] = average
    for label, value in valid.iloc[period:].items():
        average = (average * (period - 1) + float(value)) / period
        out.loc[label] = average
    return out


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index, using Wilder's smoothing.

    This previously smoothed the average gain/loss with a plain
    ``rolling(period).mean()``, which is a different indicator that happens to
    share the name: every reference implementation and charting package uses
    Wilder's smoothing, so the old values were not comparable to any external RSI
    and the usual 30/70 thresholds only loosely applied (PYQ-121).

    The first ``period`` rows are genuinely undefined and returned as NaN --
    ``build_panel()`` drops them. The old ``min_periods=1`` emitted a value from
    the second row off a one-row window, which was not NaN and therefore survived
    that cleanup; it was removed only because ``SMA_50`` happened to cut the first
    49 rows anyway.
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder_average(gain, period)
    avg_loss = _wilder_average(loss, period)

    rsi = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    # An all-gains window has avg_loss == 0; state RSI = 100 explicitly rather
    # than leaning on float division by zero.
    return rsi.mask((avg_loss == 0) & avg_gain.notna(), 100.0)


def compute_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series]:
    """Bollinger band width and %B."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    band_width = (upper - lower) / (sma + 1e-10)
    percent_b = (series - lower) / (upper - lower + 1e-10)
    return band_width, percent_b


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append technical indicators to an OHLCV DataFrame (in place-safe copy)."""
    df = df.copy()
    close = df["Close"]
    volume = df["Volume"]

    df["SMA_10"] = close.rolling(window=10).mean()
    df["SMA_20"] = close.rolling(window=20).mean()
    df["SMA_50"] = close.rolling(window=50).mean()
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    df["RSI_14"] = compute_rsi(close, 14)

    macd, signal, hist = compute_macd(close)
    df["MACD"] = macd
    df["MACD_Signal"] = signal
    df["MACD_Hist"] = hist

    bb_width, bb_pctb = compute_bollinger_bands(close)
    df["BB_Width"] = bb_width
    df["BB_PercentB"] = bb_pctb

    df["Price_Change"] = close.pct_change()
    df["Volume_Change"] = volume.pct_change()
    # Realized volatility: a genuine, historical volatility time series
    # (free-data stand-in for options-implied vol, which yfinance only
    # exposes as a current snapshot — see pyquant.data.options).
    df["Realized_Vol_20"] = close.pct_change().rolling(window=20).std() * np.sqrt(252)

    # Leading rows are genuinely NaN until each indicator's window is full
    # (e.g. SMA_50 needs 49 days of history). Leave them as NaN rather than
    # bfilling a fabricated constant -- build_panel() drops them.
    return df


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Make the index a tz-naive, normalized DatetimeIndex named 'Date'."""
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx.normalize()
    df.index.name = "Date"
    return df


def fetch_prices(
    symbol: str,
    period: str = "5y",
    start: str | None = None,
    end: str | None = None,
    use_indicators: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV history for ``symbol`` with optional technical indicators.

    Returns a DataFrame indexed by tz-naive date with at least
    Open/High/Low/Close/Volume columns.
    """
    ticker = yf.Ticker(symbol)

    # Honor an explicit range if *either* bound is given (yfinance accepts start
    # or end alone); only fall back to period when neither is set, so passing
    # just start (e.g. "everything since IPO") isn't silently ignored.
    def _load() -> pd.DataFrame:
        if start or end:
            return ticker.history(start=start, end=end)
        return ticker.history(period=period)

    # A transient yfinance hiccup here otherwise hard-fails the whole panel
    # build; retry a couple of times before giving up (PYQ-215).
    df = with_retry(_load, description=f"fetch_prices({symbol})")

    if df is None or df.empty:
        raise ValueError(f"No price data found for {symbol!r}")

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = _normalize_index(df)

    if use_indicators:
        df = add_technical_indicators(df)

    return df
