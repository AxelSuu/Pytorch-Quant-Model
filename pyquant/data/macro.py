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

from pyquant.data.prices import AUTO_ADJUST, _period_start

logger = logging.getLogger(__name__)


class _FredSeriesSpec(NamedTuple):
    """A FRED series' output column.

    Values are indexed by their ALFRED ``realtime_start`` release date rather
    than the economic reference period. This is what made the value available
    to a historical model row, including later revisions (PYQ-257).
    """

    column: str


# FRED series id -> output column name. ALFRED release dates replace the
# approximate per-series publication lags previously maintained here.
FRED_SERIES: dict[str, _FredSeriesSpec] = {
    "DFF": _FredSeriesSpec("FedFunds"),
    "T10Y2Y": _FredSeriesSpec("YieldSpread"),
    "CPIAUCSL": _FredSeriesSpec("CPI"),
}

MACRO_COLUMNS = ["VIX", *(spec.column for spec in FRED_SERIES.values())]


def _fetch_vix(start: str | None, end: str | None, period: str) -> pd.Series | None:
    """Daily VIX close from Yahoo Finance."""
    try:
        tkr = yf.Ticker("^VIX")
        # Honor an explicit range if *either* bound is given (yfinance accepts
        # start or end alone); only fall back to period when neither is set.
        # Explicit, for the reason prices.AUTO_ADJUST records (PYQ-228). VIX is
        # an index and is never split/dividend adjusted, so the value is
        # unchanged either way -- but "unchanged either way" is a fact worth
        # pinning rather than a default worth inheriting.
        # `end` alone is not passed through with `start=None`: yfinance defaults
        # a missing start to ~1 month before `end`, not the full period (PYQ-171).
        df = (
            tkr.history(
                start=start or _period_start(period, anchor=end), end=end, auto_adjust=AUTO_ADJUST
            )
            if (start or end)
            else tkr.history(period=period, auto_adjust=AUTO_ADJUST)
        )
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


def _vintage_series(releases: pd.DataFrame) -> pd.Series:
    """Map every release date to the newest observation then known.

    ``fredapi.get_series_all_releases()`` exposes the observation's economic
    reference date as ``date`` and the date that vintage was published as
    ``realtime_start``. Applying each release in order and emitting the newest
    reference observation after it produces a point-in-time-safe feature:
    before a first release the value is absent, after it the first published
    value is visible, and later revisions become visible only on their release
    date.
    """
    required = {"date", "realtime_start", "value"}
    if not required.issubset(releases.columns):
        raise ValueError(
            f"FRED vintage response is missing columns {sorted(required - set(releases))}"
        )
    releases = releases.loc[:, ["date", "realtime_start", "value"]].copy()
    releases["date"] = pd.to_datetime(releases["date"]).dt.normalize()
    releases["realtime_start"] = pd.to_datetime(releases["realtime_start"]).dt.normalize()
    # FRED encodes a missing observation as "."; fredapi turns that into NaT, not
    # NaN, so `float(value)` raised TypeError and one market holiday took the
    # whole series down -- then graceful degradation hid the loss (PYQ-139).
    # Coerce and drop instead: a missing observation is simply not a release.
    releases["value"] = pd.to_numeric(releases["value"], errors="coerce")
    releases = releases.dropna(subset=["value", "date", "realtime_start"])
    releases = releases.sort_values(["realtime_start", "date"])
    if releases.empty:
        return pd.Series(dtype=float)

    latest_by_reference: dict[pd.Timestamp, float] = {}
    points: dict[pd.Timestamp, float] = {}
    for released_at, batch in releases.groupby("realtime_start", sort=True):
        for row in batch.itertuples(index=False):
            latest_by_reference[row.date] = float(row.value)
        newest_reference = max(latest_by_reference)
        points[released_at] = latest_by_reference[newest_reference]
    return pd.Series(points, dtype=float).sort_index()


# FRED caps one `get_series_all_releases` call at 2000 vintage dates. A daily
# series publishes a vintage every business day (~252/year), so a one-year chunk
# leaves ample headroom while keeping the number of requests small.
_VINTAGE_CHUNK = pd.DateOffset(years=1)


def _period_to_offset(period: str) -> pd.DateOffset:
    """Parse a yfinance-style period ("5y", "6mo", "250d") into an offset.

    Needed because the realtime window has to be bounded (see
    ``_vintage_windows``) and the default call path supplies only ``period``.
    """
    text = str(period).strip().lower()
    for suffix, unit in (("mo", "months"), ("y", "years"), ("d", "days"), ("wk", "weeks")):
        if text.endswith(suffix):
            try:
                return pd.DateOffset(**{unit: int(text[: -len(suffix)])})
            except ValueError:
                break
    logger.warning("Unrecognised period %r; defaulting the macro window to 5 years", period)
    return pd.DateOffset(years=5)


def _vintage_windows(start: str | None, end: str | None, period: str) -> list[tuple[str, str]]:
    """Bounded, chunked ``(realtime_start, realtime_end)`` pairs to request.

    Two live failures came from leaving this unset (PYQ-139). With no explicit
    range fredapi defaults to FRED's full real-time span, 1776-07-04 to
    9999-12-31, and FRED rejects that twice over: *"There are 3085 vintage dates
    ... exceeds the maximum number of vintage dates allowed (2000)"* for a daily
    series, and *"realtime_end can not be after today's date"*.

    So the window is derived from the history actually being requested, clamped
    to today, and tiled into chunks no single one of which can exceed the
    vintage ceiling.
    """
    # FRED's "today" is its own, US-based one, and it rejects any realtime_end
    # past it. A caller in a timezone ahead of the US is simply on a later
    # calendar day, so `pd.Timestamp.today()` asked for tomorrow and lost the
    # most recent chunk to a Bad Request (PYQ-139). Model the cause -- the
    # publisher's clock -- rather than subtracting a fudge day.
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    last = min(pd.Timestamp(end).normalize(), today) if end else today
    first = pd.Timestamp(start).normalize() if start else last - _period_to_offset(period)
    if first > last:
        first = last

    windows: list[tuple[str, str]] = []
    cursor = first
    while cursor <= last:
        chunk_end = min(cursor + _VINTAGE_CHUNK, last)
        windows.append((cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        if chunk_end >= last:
            break
        cursor = chunk_end + pd.Timedelta(days=1)
    return windows


def _fetch_fred(
    api_key: str, start: str | None, end: str | None, period: str = "5y"
) -> pd.DataFrame | None:
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

    windows = _vintage_windows(start, end, period)

    # Fetch each series independently: a single failing/rate-limited series
    # (e.g. CPIAUCSL) must not discard the ones that already succeeded
    # (PYQ-110, same bug shape as PYQ-104), matching _fetch_vix's degrade-and-
    # continue pattern. Chunk failures degrade the same way, per chunk, so one
    # bad window costs a gap rather than the whole series.
    cols = {}
    for series_id, spec in FRED_SERIES.items():
        batches: list[pd.DataFrame] = []
        for realtime_start, realtime_end in windows:
            try:
                batches.append(
                    fred.get_series_all_releases(
                        series_id, realtime_start=realtime_start, realtime_end=realtime_end
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Could not fetch FRED series %s for %s..%s: %s",
                    series_id,
                    realtime_start,
                    realtime_end,
                    exc,
                )
        if not batches:
            continue
        try:
            s = _vintage_series(pd.concat(batches, ignore_index=True))
        except Exception as exc:
            logger.warning("Could not parse FRED series %s: %s", series_id, exc)
            continue
        if s is not None and len(s):
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
        fred = _fetch_fred(api_key, start, end, period)
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
