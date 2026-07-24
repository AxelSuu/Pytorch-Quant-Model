"""Tiny retry helper for flaky external calls (no extra dependency).

A single transient network hiccup should not crash a run (fetch_prices) or
silently zero out a feature that would have succeeded on retry (the
degrade-gracefully sources). This is a few lines of exponential backoff rather
than pulling in tenacity.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Module-level indirection so tests can patch out the wait without touching the
# global time module.
_sleep = time.sleep


def with_retry(
    func: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    description: str = "call",
) -> T:
    """Call ``func`` up to ``attempts`` times with exponential backoff.

    Returns the first successful result. Re-raises the last exception if every
    attempt fails, so callers keep their existing error handling for a genuine
    (non-transient) failure.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except exceptions as exc:
            last_exc = exc
            if attempt == attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "%s failed (attempt %d/%d): %s; retrying in %.1fs",
                description,
                attempt,
                attempts,
                exc,
                delay,
            )
            _sleep(delay)
    assert last_exc is not None  # loop always runs at least once
    raise last_exc
