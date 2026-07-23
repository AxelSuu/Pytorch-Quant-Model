"""News sentiment features (Finnhub headlines + local FinBERT scoring).

Builds a genuine daily sentiment time series: fetch company news from Finnhub,
score each headline with FinBERT (``ProsusAI/finbert``) running locally, then
aggregate per day. Both the ``FINNHUB_API_KEY`` and the optional ``transformers``
dependency are required; if either is missing the feature degrades gracefully
to an empty frame.

Note: Finnhub's free tier serves roughly the last year of news, so older dates
in the panel get neutral (0) sentiment after the join.
"""

from __future__ import annotations

import datetime as dt
import logging
from functools import lru_cache

import pandas as pd
import requests

logger = logging.getLogger(__name__)

SENTIMENT_COLUMNS = ["Sentiment", "HeadlineCount"]
_FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"
_MAX_HISTORY_DAYS = 365  # free-tier news horizon


@lru_cache(maxsize=1)
def _finbert():
    """Lazily build (and cache) the FinBERT classification pipeline."""
    try:
        from transformers import pipeline
    except ImportError:
        logger.warning(
            "transformers not installed; install the 'sentiment' extra to enable "
            "news sentiment (uv sync --extra sentiment)"
        )
        return None
    try:
        return pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            truncation=True,
            top_k=None,
        )
    except Exception as exc:
        logger.warning("Could not load FinBERT model: %s", exc)
        return None


def fetch_news(api_key: str, symbol: str, start: str, end: str) -> list[dict]:
    """Fetch company news headlines from Finnhub between ``start`` and ``end``."""
    params = {"symbol": symbol, "from": start, "to": end, "token": api_key}
    resp = requests.get(_FINNHUB_NEWS_URL, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else []


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
    dates = [pd.Timestamp(dt.datetime.utcfromtimestamp(a["datetime"]).date()) for a in usable]
    scores = score_headlines(headlines)
    if not scores or len(scores) != len(dates):
        return pd.DataFrame()

    df = pd.DataFrame({"Date": dates, "score": scores})
    daily = df.groupby("Date").agg(Sentiment=("score", "mean"), HeadlineCount=("score", "size"))
    daily.index.name = "Date"
    return daily.sort_index()
