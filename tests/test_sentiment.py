"""Tests for news sentiment (graceful degradation + scoring logic)."""

import datetime as dt
import json
import sys
import types
import warnings
from pathlib import Path

import pandas as pd
import pytest

from pyquant.data import sentiment

FIXTURES = Path(__file__).parent / "fixtures"


def _utc(year, month, day, hour):
    """An explicit UTC timestamp, so these tests do not depend on the host tz."""
    return int(dt.datetime(year, month, day, hour, tzinfo=dt.timezone.utc).timestamp())


def test_finbert_retries_after_transient_load_failure(monkeypatch):
    """A transient pipeline-construction failure must not permanently poison the
    cache (PYQ-114): a later call in the same process should retry and succeed."""
    calls = {"n": 0}

    def flaky_pipeline(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("HuggingFace Hub download hiccup")
        return object()  # a working pipeline the second time

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.pipeline = flaky_pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(sentiment, "_FINBERT_PIPELINE", None)

    assert sentiment._finbert() is None  # first call: transient failure
    assert sentiment._finbert() is not None  # retry: not the cached None
    assert calls["n"] == 2


def test_score_headlines_maps_finbert_pipeline_output_offline(monkeypatch):
    """Cover the real _finbert() -> score_headlines -> _signed_score path with a
    recorded FinBERT-shaped pipeline output, so the label mapping/aggregation has
    coverage without a model download (PYQ-308). Only the pipeline construction
    itself (the HF model download) stays uncovered -- by decision, it degrades
    gracefully rather than warranting a slow CI job."""
    # Mirrors transformers pipeline("text-classification", top_k=None): one list
    # of {label, score} dicts per input headline.
    recorded = [
        [
            {"label": "positive", "score": 0.90},
            {"label": "negative", "score": 0.05},
            {"label": "neutral", "score": 0.05},
        ],
        [
            {"label": "negative", "score": 0.70},
            {"label": "positive", "score": 0.20},
            {"label": "neutral", "score": 0.10},
        ],
    ]

    class FakePipeline:
        def __call__(self, headlines):
            assert len(headlines) == 2
            return recorded

    # Prime the module-level cache so the real _finbert() returns our fake
    # pipeline (exercises the cache-hit path too).
    monkeypatch.setattr(sentiment, "_FINBERT_PIPELINE", FakePipeline())
    scores = sentiment.score_headlines(["great news", "bad news"])
    assert scores == [0.90 - 0.05, 0.20 - 0.70]


def test_signed_score_positive_minus_negative():
    result = [
        {"label": "positive", "score": 0.8},
        {"label": "negative", "score": 0.1},
        {"label": "neutral", "score": 0.1},
    ]
    assert abs(sentiment._signed_score(result) - 0.7) < 1e-9


def test_fetch_sentiment_without_key_is_empty():
    assert sentiment.fetch_sentiment(api_key=None, symbol="AAPL").empty


def test_fetch_news_recovers_from_transient_failure(monkeypatch):
    """A single transient Finnhub request failure must be retried (PYQ-215)."""
    from pyquant.data import retry

    monkeypatch.setattr(retry, "_sleep", lambda _s: None)
    calls = {"n": 0}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"headline": "x", "datetime": 1}]

    def flaky_get(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return FakeResp()

    monkeypatch.setattr(sentiment.requests, "get", flaky_get)
    out = sentiment.fetch_news("key", "AAPL", "2024-01-01", "2024-02-01")
    assert calls["n"] == 2
    assert out == [{"headline": "x", "datetime": 1}]


def test_fetch_news_sends_the_api_key_as_a_header_not_a_query_param(monkeypatch):
    """PYQ-150: a query-string token reaches HTTPError.__str__() (the full request
    URL) and therefore with_retry's WARNING log on every retryable failure."""
    seen = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return []

    def fake_get(url, params=None, headers=None, **kwargs):
        seen["params"] = params
        seen["headers"] = headers
        return FakeResp()

    monkeypatch.setattr(sentiment.requests, "get", fake_get)
    sentiment.fetch_news("super-secret-key", "AAPL", "2024-01-01", "2024-02-01")

    assert seen["headers"] == {"X-Finnhub-Token": "super-secret-key"}
    assert "token" not in seen["params"]
    assert "super-secret-key" not in str(seen["params"])


def test_fetch_sentiment_aggregates_daily(monkeypatch):
    # Pretend FinBERT + Finnhub are available and return deterministic data.
    monkeypatch.setattr(sentiment, "_finbert", lambda: object())

    ts = int(dt.datetime(2024, 1, 2, 12, 0).timestamp())
    articles = [
        {"headline": "great earnings", "datetime": ts},
        {"headline": "lawsuit filed", "datetime": ts},
    ]
    monkeypatch.setattr(sentiment, "fetch_news", lambda *a, **k: articles)
    monkeypatch.setattr(sentiment, "score_headlines", lambda h: [0.9, -0.5])

    out = sentiment.fetch_sentiment(api_key="dummy", symbol="AAPL")
    assert "Sentiment" in out.columns
    assert "HeadlineCount" in out.columns
    # Two articles on the same day -> one row, count 2, mean of the two scores.
    assert len(out) == 1
    assert out["HeadlineCount"].iloc[0] == 2
    assert abs(out["Sentiment"].iloc[0] - 0.2) < 1e-9


def test_fetch_sentiment_empty_news(monkeypatch):
    monkeypatch.setattr(sentiment, "_finbert", lambda: object())
    monkeypatch.setattr(sentiment, "fetch_news", lambda *a, **k: [])
    assert sentiment.fetch_sentiment(api_key="dummy", symbol="AAPL").empty


def test_fetch_sentiment_ignores_one_malformed_article(monkeypatch, caplog):
    """One article missing a usable datetime must not zero out the whole batch."""
    monkeypatch.setattr(sentiment, "_finbert", lambda: object())

    ts = int(dt.datetime(2024, 1, 2, 12, 0).timestamp())
    articles = [
        {"headline": "great earnings", "datetime": ts},
        {"headline": "malformed article", "datetime": None},
    ]
    monkeypatch.setattr(sentiment, "fetch_news", lambda *a, **k: articles)
    monkeypatch.setattr(sentiment, "score_headlines", lambda h: [0.9] * len(h))

    with caplog.at_level("WARNING"):
        out = sentiment.fetch_sentiment(api_key="dummy", symbol="AAPL")

    assert not out.empty
    assert out["HeadlineCount"].iloc[0] == 1
    assert abs(out["Sentiment"].iloc[0] - 0.9) < 1e-9
    assert any("malformed" in msg.lower() or "datetime" in msg.lower() for msg in caplog.messages)


# --- PYQ-129: post-close news belongs to the *next* session -------------------


def test_post_close_headline_is_assigned_to_the_next_session(monkeypatch):
    """A headline published after the close cannot inform that day's close.

    A US session closes at 16:00 ET (20:00/21:00 UTC), so bucketing by UTC
    calendar date attaches the last 3-4 hours of every UTC day -- the slice
    that holds post-close earnings releases -- to a row whose target is that
    day's close. Same class of leak as PYQ-101 (PYQ-129).
    """
    monkeypatch.setattr(sentiment, "_finbert", lambda: object())
    articles = [
        {"headline": "pre-close", "datetime": _utc(2024, 1, 2, 14)},  # 09:00 ET
        {"headline": "post-close", "datetime": _utc(2024, 1, 2, 22)},  # 17:00 ET
    ]
    monkeypatch.setattr(sentiment, "fetch_news", lambda *a, **k: articles)
    monkeypatch.setattr(sentiment, "score_headlines", lambda h: [0.8, -0.4])

    out = sentiment.fetch_sentiment(api_key="dummy", symbol="AAPL")

    assert list(out.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert out.loc[pd.Timestamp("2024-01-02"), "Sentiment"] == pytest.approx(0.8)
    assert out.loc[pd.Timestamp("2024-01-03"), "Sentiment"] == pytest.approx(-0.4)


def test_fetch_sentiment_raises_no_deprecation_warning(monkeypatch):
    """utcfromtimestamp is deprecated in 3.12 and PYQ-108's filter hides it."""
    monkeypatch.setattr(sentiment, "_finbert", lambda: object())
    monkeypatch.setattr(
        sentiment,
        "fetch_news",
        lambda *a, **k: [{"headline": "x", "datetime": _utc(2024, 1, 2, 14)}],
    )
    monkeypatch.setattr(sentiment, "score_headlines", lambda h: [0.5])

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        sentiment.fetch_sentiment(api_key="dummy", symbol="AAPL")


# --- PYQ-129: aligning the daily series onto real trading sessions ------------


def test_align_to_sessions_rolls_non_trading_dates_onto_the_next_session():
    """News dated on a weekend must land on Monday, not be silently dropped.

    build_panel() reindexes the daily series straight onto the price index, so
    any date that is not itself a trading day vanishes. Shifting post-close news
    forward would otherwise convert a leak into data loss.
    """
    daily = pd.DataFrame(
        {"Sentiment": [1.0, -1.0, 0.5], "HeadlineCount": [1.0, 3.0, 2.0]},
        index=pd.DatetimeIndex(
            [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-06"), pd.Timestamp("2024-01-08")],
            name="Date",
        ),
    )
    sessions = pd.DatetimeIndex([pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08")])

    aligned = sentiment.align_to_sessions(daily, sessions)

    assert list(aligned.index) == list(sessions)
    assert aligned.loc[pd.Timestamp("2024-01-05"), "HeadlineCount"] == 1.0
    # Saturday's 3 headlines roll onto Monday and pool with Monday's own 2.
    assert aligned.loc[pd.Timestamp("2024-01-08"), "HeadlineCount"] == 5.0
    # Pooled sentiment is headline-weighted: (3*-1.0 + 2*0.5) / 5.
    assert aligned.loc[pd.Timestamp("2024-01-08"), "Sentiment"] == pytest.approx(-0.4)


def test_align_to_sessions_drops_news_after_the_last_session():
    """News with no session left to land on is dropped, never rolled backwards."""
    daily = pd.DataFrame(
        {"Sentiment": [1.0], "HeadlineCount": [4.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-02-01")], name="Date"),
    )
    sessions = pd.DatetimeIndex([pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-08")])

    aligned = sentiment.align_to_sessions(daily, sessions)

    assert aligned["HeadlineCount"].sum() == 0.0


def test_fetch_news_parses_a_real_recorded_finnhub_payload(monkeypatch):
    """PYQ-243: drives the real fetch_news() JSON parsing from one real recorded
    Finnhub response (headline/summary/url/image text replaced with placeholders
    before being checked in -- the contract is the response *shape*, not the
    copyrighted article text) instead of the hand-built ``[{"headline": "x", ...}]``
    every other test in this file uses.
    """
    real_articles = json.loads((FIXTURES / "finnhub_news_aapl.json").read_text())
    assert len(real_articles) > 50  # a real multi-day response, not a token sample

    class RecordedResp:
        def raise_for_status(self):
            pass

        def json(self):
            return real_articles

    monkeypatch.setattr(sentiment.requests, "get", lambda *a, **k: RecordedResp())

    out = sentiment.fetch_news("key", "AAPL", "2024-01-01", "2024-02-01")

    assert out == real_articles
    # Every article fetch_sentiment() actually reads must carry these two fields.
    assert all("datetime" in a and "headline" in a for a in out)


def test_fetch_sentiment_scores_a_real_recorded_finnhub_payload_end_to_end(monkeypatch):
    """The fuller chain: real recorded articles -> session_date bucketing ->
    daily aggregation, with only the network call and the FinBERT model stubbed.
    """
    real_articles = json.loads((FIXTURES / "finnhub_news_aapl.json").read_text())

    class RecordedResp:
        def raise_for_status(self):
            pass

        def json(self):
            return real_articles

    monkeypatch.setattr(sentiment.requests, "get", lambda *a, **k: RecordedResp())
    monkeypatch.setattr(sentiment, "_finbert", lambda: object())
    monkeypatch.setattr(sentiment, "score_headlines", lambda hs: [0.0] * len(hs))

    out = sentiment.fetch_sentiment("key", "AAPL", start="2024-01-01", end="2024-02-01")

    assert not out.empty
    assert list(out.columns) == sentiment.SENTIMENT_COLUMNS
    # Every real article landed on exactly one day; none silently vanished.
    assert out["HeadlineCount"].sum() == len(real_articles)


# --- PYQ-140: the vendor honours a request nominally but truncates the reply -


def test_fetch_sentiment_recorded_payload_reproduces_pyq_140s_truncation(monkeypatch):
    """Finnhub's free tier was measured (live, across several `from` values --
    see bugs.md#pyq-140) to always return roughly the same ~6-day recent slice,
    regardless of how far back the request asks. This test does not re-probe
    that live behaviour; `finnhub_news_aapl.json` (recorded for an ordinary
    7-day request, PYQ-243) happens to already carry that same real shape --
    247 articles across exactly 6 distinct days -- so it doubles as a genuine,
    non-synthetic instance of it. Feeding it through `fetch_sentiment` with a
    multi-year `start` reproduces what a real 5y-period panel build actually
    sees: sparse, real coverage concentrated at the end of the window, not a
    crash and not (if some future change broke the graceful-degradation path)
    a fabricated full-window series.
    """
    real_articles = json.loads((FIXTURES / "finnhub_news_aapl.json").read_text())
    distinct_days = {sentiment.session_date(a["datetime"]).date() for a in real_articles}
    assert len(distinct_days) == 6, "fixture no longer has the shape this test relies on"

    class RecordedResp:
        def raise_for_status(self):
            pass

        def json(self):
            return real_articles

    monkeypatch.setattr(sentiment.requests, "get", lambda *a, **k: RecordedResp())
    monkeypatch.setattr(sentiment, "_finbert", lambda: object())
    monkeypatch.setattr(sentiment, "score_headlines", lambda hs: [0.0] * len(hs))

    start, end = "2021-07-27", "2026-07-27"  # a ~5y request, PYQ-140's own scale
    out = sentiment.fetch_sentiment("key", "AAPL", start=start, end=end)

    # fetch_sentiment() itself only ever returns rows that have news -- confirm
    # that stays true (no fabricated coverage) before checking the panel-level
    # effect through align_to_sessions below.
    assert len(out) == 6
    assert out["HeadlineCount"].sum() == len(real_articles)

    sessions = pd.bdate_range(start, end)
    aligned = sentiment.align_to_sessions(out, sessions)
    coverage = aligned["HeadlineCount"].notna().mean()
    # PYQ-140 measured ~0.3% coverage at DataConfig's real 5y default; this
    # smaller synthetic session count (~1300 business days vs. a real panel's
    # already-indicator-trimmed one) should land in the same neighbourhood --
    # loosely bounded so the test isn't pinned to the exact historical figure,
    # but tight enough to catch "no longer structurally sparse" as a failure.
    assert coverage < 0.01, f"coverage {coverage:.1%} is no longer ~structural-zero"
    # Sessions with no news are NaN, not silently 0 -- PYQ-140's mechanism
    # depends on that distinction (see align_to_sessions's own docstring).
    assert aligned["HeadlineCount"].isna().sum() > 0
    assert not (aligned["HeadlineCount"].fillna(0) < 0).any()
