"""Tests for news sentiment (graceful degradation + scoring logic)."""

import datetime as dt
import sys
import types

from pyquant.data import sentiment


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
