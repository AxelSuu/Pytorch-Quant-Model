"""Price data and technical indicators (Yahoo Finance).

Returns a date-indexed DataFrame, the building block for the unified panel in
:mod:`pyquant.data.dataset`. Technical-indicator logic is ported from the
original PyStock ``src/data.py``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Split/dividend adjustment convention for every price series the model sees.
#
# yfinance flipped this default to True during the 0.2.x series, and the declared
# constraint (>=0.2.40) already resolved to 1.4.1 -- so *whether* Close is
# adjusted, and therefore every price level, every derived indicator and every
# trained model, was decided by whichever version happened to install (PYQ-228).
# Pass it explicitly everywhere instead.
#
# True (adjusted) is the right convention here: an unadjusted series has
# discontinuities at splits and dividends that are not real price moves, and the
# indicators would read them as ones. sectors.py already assumed True; this makes
# the whole codebase agree rather than differ by file.
AUTO_ADJUST = True

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
    # An all-gains window has avg_loss == 0 and avg_gain > 0; state RSI = 100
    # explicitly rather than leaning on float division by zero. A flat/halted
    # window has avg_gain == avg_loss == 0 too (0/0 = NaN), which is neutral,
    # not overbought -- PYQ-152 (the `avg_gain > 0` guard is what tells the two
    # apart; without it a flat series was misread as maximally overbought).
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    return rsi.mask((avg_loss == 0) & (avg_gain == 0) & avg_gain.notna(), 50.0)


# How many spans of warm-up an EMA must accumulate before a value is emitted.
# The recursion is seeded at the first observation, and that seed carries weight
# (1 - alpha)**n after n rows: 14.6% at one span, 2.3% at two, 0.25% at three,
# 0.03% at four. See DEFAULT_EMA_WARMUP_SPANS's use in compute_macd (PYQ-137).
DEFAULT_EMA_WARMUP_SPANS = 4


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    warmup_spans: int = DEFAULT_EMA_WARMUP_SPANS,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram.

    Each ``ewm`` carries ``min_periods = warmup_spans * span``, so no value is
    emitted until the ``adjust=False`` seed has decayed out of it.

    PYQ-132 set ``min_periods = span``, which stopped row 1 being emitted as an
    "average" of one observation. It did not change the recursion, so the first
    rows that *were* emitted still carried the seed (PYQ-137). Measured against
    an EMA given 3000 rows of prior history -- what a charting package with more
    history than our panel actually plots -- the residual error on the first
    surviving row was 5.66% of MACD's own typical magnitude. At four spans it is
    0.08%, a 71x reduction, for 91 rows (7.2%) off a 5-year panel.

    Two alternatives were measured and rejected. ``adjust=True`` was expected to
    remove the bias exactly, being the normalised weighted average with no seed;
    it does not, because it is exact only over the *truncated* window and is
    equally blind to the missing history -- against the full-history reference it
    is 1.3-1.6x **worse** than ``adjust=False`` at rows 49-104. Seeding the
    recursion with an SMA of the first ``span`` observations (what TradingView
    does) was 1.6x worse again. Truncation, not the seed choice, is the real
    error source, and a longer warm-up is the only one of the three that
    attacks it -- while keeping the standard definition every charting package
    plots, which is the argument PYQ-121 used when it adopted Wilder's RSI.

    The signal line inherits the slow EMA's warm-up and adds its own, so
    ``MACD_Signal``/``MACD_Hist`` are first defined ``warmup_spans * signal``
    rows after ``MACD`` is.
    """
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=warmup_spans * fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=warmup_spans * slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(
        span=signal, adjust=False, min_periods=warmup_spans * signal
    ).mean()
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


def add_technical_indicators(
    df: pd.DataFrame, warmup_spans: int = DEFAULT_EMA_WARMUP_SPANS
) -> pd.DataFrame:
    """Append technical indicators to an OHLCV DataFrame (in place-safe copy).

    ``warmup_spans`` trades leading rows for front-of-panel accuracy in the
    exponential indicators; see compute_macd for the measurements behind the
    default (PYQ-137).
    """
    df = df.copy()
    close = df["Close"]
    volume = df["Volume"]

    df["SMA_10"] = close.rolling(window=10).mean()
    df["SMA_20"] = close.rolling(window=20).mean()
    df["SMA_50"] = close.rolling(window=50).mean()
    # min_periods keeps the EMA warm-up genuinely NaN rather than seeding it
    # with close[0] and calling that an average (PYQ-132), and spans it far
    # enough that the seed has actually decayed before a value is used
    # (PYQ-137).
    df["EMA_12"] = close.ewm(span=12, adjust=False, min_periods=warmup_spans * 12).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False, min_periods=warmup_spans * 26).mean()

    df["RSI_14"] = compute_rsi(close, 14)

    macd, signal, hist = compute_macd(close, warmup_spans=warmup_spans)
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

    # A zero-volume session (a halt, a thin ADR, a feed gap yfinance reports as
    # 0 rather than NaN) makes the next row's pct_change divide by zero. inf is
    # not NaN, so it survives build_panel()'s dropna() and poisons
    # GroupNormalizer's fitted scale for that group, propagating NaN through the
    # loss. Map non-finite results onto NaN so the existing row-drop handles
    # them -- applied across the whole block, not just Volume_Change, so a
    # future indicator cannot reintroduce the same class silently (PYQ-135).
    df = df.replace([np.inf, -np.inf], np.nan)

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


def _period_start(period: str) -> str:
    """First calendar date covered by a yfinance-style period, as YYYY-MM-DD.

    Providers that take an explicit date range rather than a period string
    (Tiingo, Alpha Vantage) need this to honour the same ``DataConfig.period``
    the rest of the pipeline is configured with.
    """
    text = str(period).strip().lower()
    for suffix, unit in (("mo", "months"), ("y", "years"), ("d", "days"), ("wk", "weeks")):
        if text.endswith(suffix):
            try:
                offset = pd.DateOffset(**{unit: int(text[: -len(suffix)])})
            except ValueError:
                break
            return (pd.Timestamp.today().normalize() - offset).strftime("%Y-%m-%d")
    logger.warning("Unrecognised period %r; defaulting to 5 years", period)
    return (pd.Timestamp.today().normalize() - pd.DateOffset(years=5)).strftime("%Y-%m-%d")


def fetch_prices(
    symbol: str,
    period: str = "5y",
    start: str | None = None,
    end: str | None = None,
    use_indicators: bool = True,
    provider: str | object = "yfinance",
) -> pd.DataFrame:
    """Fetch OHLCV history for ``symbol`` with optional technical indicators.

    Returns a DataFrame indexed by tz-naive date with at least
    Open/High/Low/Close/Volume columns.

    ``provider`` names a `pyquant.data.providers.PriceProvider` (or is one
    already). Switching vendors is therefore a config change rather than a
    rewrite, which is the property PYQ-258 is actually about -- yfinance being
    an unofficial scraper with no SLA behind four of the project's data sources.
    """
    from pyquant.data.providers import assert_ohlcv_contract, get_provider

    price_provider = get_provider(provider) if isinstance(provider, str) else provider
    df = price_provider.fetch_ohlcv(symbol, period=period, start=start, end=end)
    # Every provider is held to one schema, checked here rather than trusted, so
    # a new vendor's subtly different frame fails loudly at the boundary instead
    # of misaligning a join downstream.
    assert_ohlcv_contract(df)

    if use_indicators:
        df = add_technical_indicators(df)

    return df
