"""Tests for the shared retry helper (PYQ-215, PYQ-151)."""

import pytest
import requests

from pyquant.data import retry


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Don't actually wait between retries in tests."""
    monkeypatch.setattr(retry, "_sleep", lambda _s: None)


def _http_error(status_code: int) -> requests.exceptions.HTTPError:
    response = requests.models.Response()
    response.status_code = status_code
    return requests.exceptions.HTTPError(response=response)


def test_with_retry_recovers_after_one_transient_failure():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return "ok"

    assert retry.with_retry(flaky, description="test") == "ok"
    assert calls["n"] == 2


def test_with_retry_reraises_after_exhausting_attempts():
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        retry.with_retry(always_fails, attempts=3, description="test")
    assert calls["n"] == 3


def test_with_retry_returns_immediately_on_success():
    assert retry.with_retry(lambda: 42) == 42


def test_with_retry_redacts_query_string_secrets_from_the_warning_log(caplog):
    """PYQ-150: a retryable failure's exception message can embed the full request
    URL (e.g. requests.HTTPError.__str__()); a query-string token/api_key must not
    reach the WARNING log, per CLAUDE.md's secrets non-negotiable."""
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError(
                "404 Client Error: Not Found for url: "
                "https://example.com/news?symbol=AAPL&token=super-secret-value"
            )
        return "ok"

    with caplog.at_level("WARNING"):
        retry.with_retry(flaky, description="test")

    assert "super-secret-value" not in caplog.text
    assert "token=***REDACTED***" in caplog.text


def test_redact_handles_api_key_and_is_case_insensitive():
    assert "my-secret" not in retry._redact("...&API_KEY=my-secret&other=1")
    assert "my-secret" not in retry._redact("...&Token=my-secret")


@pytest.mark.parametrize("status", [401, 404])
def test_with_retry_does_not_retry_non_retryable_4xx(status):
    """PYQ-151: a bad key (401) or unknown symbol (404) should fail on the first
    attempt, not burn the whole retry budget before surfacing."""
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        raise _http_error(status)

    with pytest.raises(requests.exceptions.HTTPError):
        retry.with_retry(flaky, attempts=3, description="test")
    assert calls["n"] == 1


@pytest.mark.parametrize(
    "make_exc",
    [
        lambda: _http_error(500),
        lambda: _http_error(429),
        lambda: requests.exceptions.Timeout("timed out"),
        lambda: requests.exceptions.ConnectionError("reset"),
    ],
)
def test_with_retry_still_retries_5xx_429_and_connection_failures(make_exc):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise make_exc()
        return "ok"

    assert retry.with_retry(flaky, attempts=3, description="test") == "ok"
    assert calls["n"] == 2


def test_with_retry_adds_jitter_to_the_backoff_delay(monkeypatch):
    """PYQ-151: fixed exponential backoff makes concurrent callers (e.g. the
    API's threadpool) retry in lockstep; jitter should spread them out."""
    delays = []
    monkeypatch.setattr(retry, "_sleep", delays.append)
    monkeypatch.setattr(retry.random, "uniform", lambda lo, hi: hi)

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    retry.with_retry(flaky, attempts=3, base_delay=1.0, jitter=0.5, description="test")

    assert delays == pytest.approx([1.5, 3.0])
