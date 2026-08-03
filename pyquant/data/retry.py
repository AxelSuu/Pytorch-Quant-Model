"""Tiny retry helper for flaky external calls (no extra dependency).

A single transient network hiccup should not crash a run (fetch_prices) or
silently zero out a feature that would have succeeded on retry (the
degrade-gracefully sources). This is a few lines of exponential backoff rather
than pulling in tenacity.
"""

from __future__ import annotations

import logging
import random
import re
import time
from collections.abc import Callable
from typing import TypeVar

import requests

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _default_is_retryable(exc: BaseException) -> bool:
    """True unless ``exc`` is an HTTPError with a definitive non-retryable 4xx status.

    A 401 (bad key) or 404 (unknown symbol) is deterministic -- retrying it
    still fails, so it burns the whole retry budget (every configured sleep)
    before surfacing the same error a first attempt would have (PYQ-151).
    HTTP 429 and 5xx, connection/timeout errors, and any exception type this
    doesn't recognise are still retried -- they may be transient, and this
    keeps the previous "retry everything" default for failure modes outside
    the specific one this narrows.
    """
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        status = exc.response.status_code
        return status == 429 or status >= 500
    return True


# Module-level indirection so tests can patch out the wait without touching the
# global time module.
_sleep = time.sleep

# Query-string secrets (token=..., api_key=...) that might otherwise reach a
# WARNING log via a raised HTTPError's __str__(), which embeds the full request
# URL (PYQ-150). Defense in depth: the known offender (Finnhub) was moved to
# header auth, but this catches any other query-string secret the same way.
_SECRET_QUERY_PARAM = re.compile(r"(?i)\b(token|api[_-]?key)=[^&\s]+")


def _redact(text: str) -> str:
    """Redact ``token=``/``api_key=``-shaped query fragments from a log message."""
    return _SECRET_QUERY_PARAM.sub(r"\1=***REDACTED***", text)


def with_retry(
    func: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    is_retryable: Callable[[BaseException], bool] = _default_is_retryable,
    jitter: float = 0.25,
    description: str = "call",
) -> T:
    """Call ``func`` up to ``attempts`` times with exponential backoff.

    ``exceptions`` bounds what's caught at all; ``is_retryable`` then decides
    whether a caught exception is worth spending the remaining budget on --
    anything it rejects re-raises immediately (PYQ-151), regardless of
    attempts remaining. ``jitter`` adds up to that fraction of the computed
    delay (uniform random), so concurrent callers don't retry in lockstep.

    Returns the first successful result. Re-raises the last exception if every
    attempt fails, so callers keep their existing error handling for a genuine
    (non-transient) failure.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except exceptions as exc:
            if not is_retryable(exc):
                raise
            last_exc = exc
            if attempt == attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            if jitter:
                delay += random.uniform(0, delay * jitter)
            logger.warning(
                "%s failed (attempt %d/%d): %s; retrying in %.1fs",
                description,
                attempt,
                attempts,
                _redact(str(exc)),
                delay,
            )
            _sleep(delay)
    assert last_exc is not None  # loop always runs at least once
    raise last_exc
