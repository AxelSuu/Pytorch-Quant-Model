"""News sentiment features (Finnhub headlines + local FinBERT scoring).

Builds a genuine daily sentiment time series: fetch company news from Finnhub,
score each headline with FinBERT (``ProsusAI/finbert``) running locally, then
aggregate per day. Both the ``FINNHUB_API_KEY`` and the optional ``transformers``
dependency are required; if either is missing the feature degrades gracefully
to an empty frame.

**Coverage (PYQ-140).** Finnhub's free tier ignores the ``from`` parameter entirely and
always returns the same recent slice -- measured at ~6 distinct days of headlines,
regardless of whether the requested window is one week or five years back. At the
default ``DataConfig.period="5y"`` this makes ``Sentiment``/``HeadlineCount`` a
structural zero for ~99.7% of training rows (see investigations.md#pyq-301), and the
handful of non-zero rows sit at the *end* of the panel -- exactly where the prediction
encoder reads. This module does not deliver a year of history on the free tier this
project is built against; do not assume it does.

**Publication timing (PYQ-129).** Headlines are bucketed by the session that can
first act on them, not by the UTC calendar date they were published in. This is
the same convention pyquant.data.macro applies to FRED via
``publication_lag_days``: a row at time *t* must only see information that
existed at *t*. See :func:`session_date` for the rule and its one assumption.
"""

from __future__ import annotations

import datetime as dt
import logging
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from pyquant.data.retry import with_retry

logger = logging.getLogger(__name__)

SENTIMENT_COLUMNS = ["Sentiment", "HeadlineCount"]
_FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"
_MAX_HISTORY_DAYS = 365  # requested window when `start` is omitted -- the free tier
# ignores this and returns ~6 days regardless of what's asked for (PYQ-140); kept as
# the honest ask in case a paid tier ever honors it.

# The exchange whose close decides which session a headline belongs to. US
# equities close at 16:00 America/New_York; ZoneInfo handles EST/EDT so the
# boundary stays correct across DST rather than drifting by an hour for half
# the year. Module-level rather than on DataConfig because the panel is
# US-equity-shaped throughout (see also PYQ-130's calendar note).
_EXCHANGE_TZ = ZoneInfo("America/New_York")
_SESSION_CLOSE = dt.time(16, 0)

# Cache only a *successfully* constructed pipeline (PYQ-114). lru_cache would
# also memoise a None returned after a transient failure (e.g. a HuggingFace
# Hub download blip on first use), permanently disabling sentiment for the rest
# of the process -- fatal for a long-running server. A manual cache set only on
# success lets a later call retry.
_FINBERT_PIPELINE = None


def _finbert():
    """Lazily build (and cache on success) the FinBERT classification pipeline."""
    global _FINBERT_PIPELINE
    if _FINBERT_PIPELINE is not None:
        return _FINBERT_PIPELINE
    try:
        from transformers import pipeline
    except ImportError:
        logger.warning(
            "transformers not installed; install the 'sentiment' extra to enable "
            "news sentiment (uv sync --extra sentiment)"
        )
        return None
    try:
        clf = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            truncation=True,
            top_k=None,
        )
    except Exception as exc:
        logger.warning("Could not load FinBERT model: %s", exc)
        return None
    _FINBERT_PIPELINE = clf
    return clf


def fetch_news(api_key: str, symbol: str, start: str, end: str) -> list[dict]:
    """Fetch company news headlines from Finnhub between ``start`` and ``end``."""
    params = {"symbol": symbol, "from": start, "to": end, "token": api_key}

    def _get() -> list:
        """One Finnhub request; non-list payloads become an empty result."""
        resp = requests.get(_FINNHUB_NEWS_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    # Retry a transient request failure before surfacing it to the caller, which
    # otherwise degrades to "no news" -- indistinguishable from no key (PYQ-215).
    return with_retry(_get, description=f"fetch_news({symbol})")


def _signed_score(result: list[dict]) -> float:
    """Map a FinBERT top_k result to a signed score in [-1, 1]."""
    scores = {r["label"].lower(): r["score"] for r in result}
    return scores.get("positive", 0.0) - scores.get("negative", 0.0)


def score_headlines(headlines: list[str]) -> list[float]:
    """Score a batch of headlines into signed sentiment values."""
    clf = _finbert()
    if clf is None or not headlines:
        return []
    results = clf(headlines)
    return [_signed_score(r) for r in results]


def session_date(epoch_seconds: float) -> pd.Timestamp:
    """The first session date that may legitimately use a headline published at ``epoch_seconds``.

    A headline published *before* the exchange close can inform that day's
    close, so it belongs to that date. One published at or after the close
    cannot -- it is post-close information -- so it belongs to the next
    session (PYQ-129).

    Bucketing by UTC calendar date instead attaches everything from 20:00 UTC
    onward (16:00 ET, DST aside) to a row whose target is that day's close.
    That is roughly the last 3-4 hours of each UTC day, and it is the most
    market-moving slice of it: post-close earnings releases land there almost
    by definition, which makes the leaked headline close to a direct readout of
    the next move.

    Returns a *calendar* date. Weekends and holidays are rolled onto a real
    trading session later, by :func:`align_to_sessions`, which is the only
    place the trading calendar is known.
    """
    local = dt.datetime.fromtimestamp(epoch_seconds, tz=dt.timezone.utc).astimezone(_EXCHANGE_TZ)
    date = pd.Timestamp(local.date())
    return date + pd.Timedelta(days=1) if local.time() >= _SESSION_CLOSE else date


def align_to_sessions(daily: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Roll a daily sentiment series onto real trading sessions.

    :func:`session_date` works in calendar dates, so Saturday news -- and any
    Friday-post-close news it shifts onto Saturday -- has no row to land on.
    Reindexing straight onto the price calendar would silently *drop* those
    headlines, which trades PYQ-129's leak for data loss rather than fixing it.

    Each dated bucket is therefore assigned to the first session at or after
    it, and buckets landing on the same session are pooled: counts add, and
    sentiment is averaged weighted by headline count so a 10-headline day does
    not carry the same weight as a 1-headline day. News after the last session
    has nowhere to go and is dropped -- never rolled backwards, which would be
    the leak again.

    Sessions with no news are returned as NaN, not 0: "no news" and "neutral
    news" are different, and it is ``build_panel`` that decides to treat them
    alike.
    """
    sessions = pd.DatetimeIndex(sessions)
    if daily.empty or len(sessions) == 0:
        return pd.DataFrame(columns=SENTIMENT_COLUMNS, index=sessions, dtype=float)

    positions = sessions.searchsorted(pd.DatetimeIndex(daily.index), side="left")
    within = positions < len(sessions)
    n_dropped = int((~within).sum())
    if n_dropped:
        logger.info(
            "Dropping %d sentiment day(s) dated after the last trading session", n_dropped
        )

    landed = daily.loc[within].copy()
    landed["_session"] = sessions[positions[within]]
    weight = landed["HeadlineCount"].to_numpy()
    landed["_weighted"] = landed["Sentiment"].to_numpy() * weight

    grouped = landed.groupby("_session").agg(
        _weighted=("_weighted", "sum"), HeadlineCount=("HeadlineCount", "sum")
    )
    counts = grouped["HeadlineCount"].to_numpy(dtype=float)
    # A session whose headlines all carry count 0 has no weight to average by;
    # leave it NaN rather than dividing by zero.
    grouped["Sentiment"] = np.divide(
        grouped["_weighted"].to_numpy(dtype=float),
        counts,
        out=np.full(len(counts), np.nan),
        where=counts != 0,
    )
    out = grouped[SENTIMENT_COLUMNS].reindex(sessions)
    out.index.name = "Date"
    return out


def fetch_sentiment(
    api_key: str | None,
    symbol: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Return a date-indexed DataFrame with daily mean Sentiment + HeadlineCount.

    Empty DataFrame when the key or model is unavailable, or no news is found.
    """
    if not api_key:
        logger.info("No FINNHUB_API_KEY set; skipping news sentiment")
        return pd.DataFrame()
    if _finbert() is None:
        return pd.DataFrame()

    end = end or dt.date.today().isoformat()
    if start is None:
        start = (dt.date.today() - dt.timedelta(days=_MAX_HISTORY_DAYS)).isoformat()

    try:
        articles = fetch_news(api_key, symbol, start, end)
    except Exception as exc:
        logger.warning("Could not fetch news for %s: %s", symbol, exc)
        return pd.DataFrame()

    if not articles:
        return pd.DataFrame()

    usable = [a for a in articles if a.get("datetime")]
    n_dropped = len(articles) - len(usable)
    if n_dropped:
        logger.warning(
            "Dropping %d/%d articles for %s with no usable datetime",
            n_dropped,
            len(articles),
            symbol,
        )
    if not usable:
        return pd.DataFrame()

    headlines = [a.get("headline", "") for a in usable]
    # The session that may act on the headline, not the UTC date it was
    # published in (PYQ-129). Also drops the deprecated utcfromtimestamp,
    # whose DeprecationWarning PYQ-108's filter was hiding.
    dates = [session_date(a["datetime"]) for a in usable]
    scores = score_headlines(headlines)
    if not scores or len(scores) != len(dates):
        return pd.DataFrame()

    df = pd.DataFrame({"Date": dates, "score": scores})
    daily = df.groupby("Date").agg(Sentiment=("score", "mean"), HeadlineCount=("score", "size"))
    daily.index.name = "Date"
    return daily.sort_index()
