"""Tests for the shared retry helper (PYQ-215)."""

import pytest

from pyquant.data import retry


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Don't actually wait between retries in tests."""
    monkeypatch.setattr(retry, "_sleep", lambda _s: None)


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
