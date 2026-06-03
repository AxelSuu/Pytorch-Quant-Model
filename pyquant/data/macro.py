"""Macro-economic context features.

VIX comes from Yahoo Finance (no key needed). FRED series (rates, CPI, yield
curve) require a free ``FRED_API_KEY``. Missing credentials degrade gracefully:
the unavailable columns are simply omitted.
"""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# FRED series id -> output column name.
FRED_SERIES = {
    "DFF": "FedFunds",  # Effective federal funds rate (daily)
    "T10Y2Y": "YieldSpread",  # 10Y-2Y treasury spread (daily)
    "CPIAUCSL": "CPI",  # CPI, all urban consumers (monthly)
}

MACRO_COLUMNS = ["VIX", *FRED_SERIES.values()]


def _fetch_vix(start: str | None, end: str | None, period: str) -> pd.Series | None:
    """Daily VIX close from Yahoo Finance."""
    try:
        tkr = yf.Ticker("^VIX")
        df = tkr.history(start=start, end=end) if (start and end) else tkr.history(period=period)
        if df is None or df.empty:
            return None
        s = df["Close"].copy()
        idx = pd.to_datetime(s.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        s.index = idx.normalize()
        s.name = "VIX"
        return s
    except Exception as exc:  # network / symbol issues should not crash the pipeline
        logger.warning("Could not fetch VIX: %s", exc)
        return None


def _fetch_fred(api_key: str, start: str | None, end: str | None) -> pd.DataFrame | None:
    """Daily-resampled FRED series; None if the library or key is unavailable."""
    try:
        from fredapi import Fred
    except ImportError:
        logger.warning("fredapi not installed; skipping FRED macro features")
        return None

    try:
        fred = Fred(api_key=api_key)
        cols = {}
        for series_id, name in FRED_SERIES.items():
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            if s is not None and len(s):
                s.index = pd.to_datetime(s.index).normalize()
                cols[name] = s
        if not cols:
            return None
        return pd.DataFrame(cols)
    except Exception as exc:
        logger.warning("Could not fetch FRED series: %s", exc)
        return None


def fetch_macro(
    api_key: str | None = None,
    start: str | None = None,
    end: str | None = None,
    period: str = "5y",
) -> pd.DataFrame:
    """Return a date-indexed DataFrame of macro features.

    Always attempts VIX. Adds FRED columns when ``api_key`` is provided. Returns
    an empty DataFrame if nothing could be fetched (caller treats as "no macro").
    """
    frames: list[pd.DataFrame] = []

    vix = _fetch_vix(start, end, period)
    if vix is not None:
        frames.append(vix.to_frame())

    if api_key:
        fred = _fetch_fred(api_key, start, end)
        if fred is not None:
            frames.append(fred)
    else:
        logger.info("No FRED_API_KEY set; macro features limited to VIX")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1).sort_index()
    out.index.name = "Date"
    # Forward-fill lower-frequency series (e.g. monthly CPI) to daily.
    return out.ffill()
