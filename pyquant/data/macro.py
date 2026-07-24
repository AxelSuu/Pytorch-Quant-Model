"""Macro-economic context features.

VIX comes from Yahoo Finance (no key needed). FRED series (rates, CPI, yield
curve) require a free ``FRED_API_KEY``. Missing credentials degrade gracefully:
the unavailable columns are simply omitted.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class _FredSeriesSpec(NamedTuple):
    """A FRED series' output column and its real-world publication lag.

    fredapi's get_series() indexes values by economic *reference period*
    (e.g. CPIAUCSL dated 2026-06-01 is June's CPI), not the date it was
    actually published. publication_lag_days is how long after the reference
    date the value is realistically known, so it can be shifted forward
    before joining onto a daily/trading calendar -- otherwise a training row
    sees data that, in reality, wasn't available yet (look-ahead leakage).
    """

    column: str
    publication_lag_days: int


# FRED series id -> (output column name, publication lag in days).
FRED_SERIES: dict[str, _FredSeriesSpec] = {
    "DFF": _FredSeriesSpec("FedFunds", 1),  # daily rate, published next business day
    "T10Y2Y": _FredSeriesSpec("YieldSpread", 1),  # same-day market data
    "CPIAUCSL": _FredSeriesSpec("CPI", 21),  # BLS releases ~3 weeks after month-end
}

MACRO_COLUMNS = ["VIX", *(spec.column for spec in FRED_SERIES.values())]


def _fetch_vix(start: str | None, end: str | None, period: str) -> pd.Series | None:
    """Daily VIX close from Yahoo Finance."""
    try:
        tkr = yf.Ticker("^VIX")
        # Honor an explicit range if *either* bound is given (yfinance accepts
        # start or end alone); only fall back to period when neither is set.
        df = tkr.history(start=start, end=end) if (start or end) else tkr.history(period=period)
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
    except Exception as exc:
        logger.warning("Could not initialise FRED client: %s", exc)
        return None

    # Fetch each series independently: a single failing/rate-limited series
    # (e.g. CPIAUCSL) must not discard the ones that already succeeded
    # (PYQ-110, same bug shape as PYQ-104), matching _fetch_vix's degrade-and-
    # continue pattern.
    cols = {}
    for series_id, spec in FRED_SERIES.items():
        try:
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
        except Exception as exc:
            logger.warning("Could not fetch FRED series %s: %s", series_id, exc)
            continue
        if s is not None and len(s):
            s.index = pd.to_datetime(s.index).normalize() + pd.Timedelta(
                days=spec.publication_lag_days
            )
            cols[spec.column] = s
    if not cols:
        return None
    return pd.DataFrame(cols)


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
