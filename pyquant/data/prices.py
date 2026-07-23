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


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


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
    if start and end:
        df = ticker.history(start=start, end=end)
    else:
        df = ticker.history(period=period)

    if df is None or df.empty:
        raise ValueError(f"No price data found for {symbol!r}")

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = _normalize_index(df)

    if use_indicators:
        df = add_technical_indicators(df)

    return df
