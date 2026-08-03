"""FastAPI dependencies: settings, the bundle LRU cache, per-bundle locks, API-key auth.

Per docs/api-design.md #4/#5. Module-level singletons rather than app.state, since this
is the v1 in-process scaffold the design note describes -- state is lost on restart and
invisible to a second instance, which is the documented trigger to graduate to a shared
backend, not a bug in this version.
"""

from __future__ import annotations

import hmac
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import Header, HTTPException

from pyquant.config import Settings, load_settings
from pyquant.models import tft

logger = logging.getLogger(__name__)

# Cap on how many per-name locks either registry below retains. Both are keyed
# by names that already passed SYMBOL_PATTERN (`^[A-Za-z0-9][A-Za-z0-9.\-]{0,15}$`,
# api/schemas.py) -- a large but finite space, not an arbitrary-string injection
# point -- so unbounded growth here is slow, bounded memory drift under a
# long-running process, not an unbounded DoS from one request (bugs.md#pyq-164).
# 1024 distinct names touched by one process comfortably covers any real
# operator's symbol universe while keeping the registry meaningfully bounded.
_MAX_TRACKED_LOCKS = 1024


class _BoundedLockRegistry:
    """One `threading.Lock` per name, created lazily, LRU-evicted past a cap.

    Only an *unheld* lock is ever evicted (bugs.md#pyq-164): replacing a lock
    object out from under a thread that is currently holding it or blocked in
    `.acquire()` on it would let a second, unrelated `Lock` get created for the
    same name afterward, silently breaking the "one lock per name" mutual-
    exclusion guarantee this registry exists to provide. If every tracked lock
    is currently held, the registry is allowed to grow past the cap rather
    than break that guarantee -- bounded by how many requests can be
    concurrently in flight, not by how many distinct names have ever been
    seen -- and a warning is logged so sustained overflow is visible.
    """

    def __init__(self, max_size: int = _MAX_TRACKED_LOCKS) -> None:
        self.max_size = max_size
        self._locks: OrderedDict[str, threading.Lock] = OrderedDict()
        self._registry_lock = threading.Lock()

    def get(self, name: str) -> threading.Lock:
        with self._registry_lock:
            lock = self._locks.get(name)
            if lock is not None:
                self._locks.move_to_end(name)
                return lock
            lock = threading.Lock()
            self._locks[name] = lock
            self._evict_unheld_past_cap()
            return lock

    def _evict_unheld_past_cap(self) -> None:
        """Drop least-recently-used unheld locks until at/under `max_size`.

        Caller holds `_registry_lock`. Stops early once none are evictable.
        """
        if len(self._locks) <= self.max_size:
            return
        for existing_name in list(self._locks):
            if len(self._locks) <= self.max_size:
                return
            if self._locks[existing_name].locked():
                continue
            del self._locks[existing_name]
        if len(self._locks) > self.max_size:
            logger.warning(
                "Lock registry holding %d entries, over its %d cap -- every "
                "tracked lock is currently held, so none could be evicted.",
                len(self._locks),
                self.max_size,
            )

    def __len__(self) -> int:
        return len(self._locks)


def get_settings() -> Settings:
    """FastAPI dependency: settings for the current request."""
    return load_settings()


class BundleCache:
    """LRU cache of loaded ModelBundles, bounded by max_size.

    docs/api-design.md #4: reloading per request re-pays checkpoint
    deserialisation and re-triggers the weights_only=False unpickle every time.
    """

    def __init__(self, max_size: int = 8) -> None:
        """Empty cache, evicting beyond ``max_size`` loaded bundles."""
        self.max_size = max_size
        self._cache: OrderedDict[str, tft.ModelBundle] = OrderedDict()
        self._lock = threading.Lock()
        # One lock per bundle name, held only around a cache-miss load (not
        # around a hit) -- without this, N simultaneous first-requests for the
        # same not-yet-cached symbol each pay the full checkpoint
        # deserialisation cost redundantly, since the load itself deliberately
        # happens outside `_lock` (see the comment below). Per-name, not
        # global, so concurrent loads of *different* bundles still proceed in
        # parallel -- only a stampede on the identical name is serialized.
        # LRU-bounded (bugs.md#pyq-164): this dict used to grow by one entry
        # per distinct bundle name ever requested and never shrink.
        self._load_locks = _BoundedLockRegistry()

    def _load_lock_for(self, name: str) -> threading.Lock:
        return self._load_locks.get(name)

    def get(self, name: str, settings: Settings) -> tft.ModelBundle:
        """Return the cached bundle, loading (and caching) it on a miss."""
        name = name.upper()
        with self._lock:
            cached = self._cache.get(name)
            if cached is not None:
                self._cache.move_to_end(name)
                return cached
        with self._load_lock_for(name):
            # Re-check: another thread may have loaded and cached this exact
            # name while we were waiting for the load lock above.
            with self._lock:
                cached = self._cache.get(name)
                if cached is not None:
                    self._cache.move_to_end(name)
                    return cached
            # tft.load() does real file I/O + deserialisation; keep it outside
            # `_lock` so loading one bundle does not block requests for
            # already-cached ones. The per-name load lock above is what stops
            # this from running redundantly for concurrent misses of the same
            # name -- it does not affect concurrency across different names.
            bundle = tft.load(name, settings)
            with self._lock:
                self._cache[name] = bundle
                self._cache.move_to_end(name)
                while len(self._cache) > self.max_size:
                    self._cache.popitem(last=False)
            return bundle

    def invalidate(self, name: str) -> None:
        """Evict a bundle (e.g. after retraining it) so the next get() reloads it."""
        with self._lock:
            self._cache.pop(name.upper(), None)


_BUNDLE_CACHE = BundleCache()


def get_bundle_cache() -> BundleCache:
    """FastAPI dependency: the process-wide bundle LRU cache."""
    return _BUNDLE_CACHE


class _PredictionLocks:
    """One lock per bundle name, created on first use.

    docs/api-design.md #4: do not assume TemporalFusionTransformer.predict() is
    safe to call concurrently on the same model instance -- pytorch-forecasting
    spins up an internal Lightning Trainer per call and mutates model state.
    Serializing per-bundle is not as costly as it sounds: torch intra-op
    threading already uses multiple cores per call, and parallelism *across*
    different bundles is unaffected.
    """

    def __init__(self) -> None:
        """No locks yet; each is created lazily on first get()."""
        # LRU-bounded (bugs.md#pyq-164): this dict used to grow by one entry
        # per distinct bundle name ever requested and never shrink.
        self._locks = _BoundedLockRegistry()

    def get(self, name: str) -> threading.Lock:
        """Return this bundle's lock, creating it on first use."""
        return self._locks.get(name.upper())


_PREDICTION_LOCKS = _PredictionLocks()


def get_prediction_lock(name: str) -> threading.Lock:
    """FastAPI-callable accessor for a bundle's prediction lock."""
    return _PREDICTION_LOCKS.get(name)


# How long a request will queue for a busy bundle's prediction lock before
# giving up. `with lock:` blocks indefinitely, which turns "this bundle is
# serving a slow request" into every subsequent caller also holding a request
# thread hostage on the shared anyio threadpool -- the same starvation shape
# as an unbounded background job, just one lock over. A bounded wait plus a
# clean 429 costs the caller nothing they wouldn't already pay by retrying.
PREDICTION_LOCK_TIMEOUT_SECONDS = 30.0


@contextmanager
def acquire_prediction_lock(name: str, timeout: float | None = None) -> Iterator[None]:
    """Hold `name`'s prediction lock for the block, or raise 429 within `timeout`.

    Same per-bundle lock `get_prediction_lock` already returns -- this only
    changes how long a caller is willing to wait for it before returning
    control to the client instead of blocking forever. `timeout` defaults to
    `PREDICTION_LOCK_TIMEOUT_SECONDS` read at call time (not bound into the
    signature as a default value) so tests can `monkeypatch` the module
    constant directly rather than needing to thread an override through every
    call site.
    """
    if timeout is None:
        timeout = PREDICTION_LOCK_TIMEOUT_SECONDS
    lock = get_prediction_lock(name)
    if not lock.acquire(timeout=timeout):
        raise HTTPException(
            status_code=429,
            detail=f"Bundle {name!r} is busy serving another request; try again shortly",
        )
    try:
        yield
    finally:
        lock.release()


_UNCONFIGURED_KEYS_MESSAGE = (
    "PYQUANT_API_KEYS is not set. Set it (comma-separated keys), or set "
    "PYQUANT_API_ALLOW_UNAUTHENTICATED=1 for local development only."
)


def _configured_api_keys() -> list[str]:
    return [k for k in os.environ.get("PYQUANT_API_KEYS", "").split(",") if k]


def api_auth_is_configured() -> bool:
    """True if a request would actually be checked against a real key.

    Shared by `require_api_key` (per-request) and `app.py`'s startup lifespan
    check (once, at boot) so the two can't drift on what "configured" means.
    """
    return (
        bool(_configured_api_keys()) or os.environ.get("PYQUANT_API_ALLOW_UNAUTHENTICATED") == "1"
    )


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Reject unauthenticated requests (docs/api-design.md #5).

    Keys come from the ``PYQUANT_API_KEYS`` environment variable (comma-separated),
    not ``Settings`` -- so a key is never at risk of being written into a
    ``meta.json``/log/response the way a Settings field could be, matching this
    project's secrets non-negotiable. Constant-time comparison, so a response-timing
    side channel cannot leak a valid key one byte at a time.

    Fails loudly (500) when unconfigured rather than silently allowing every
    request through: a public endpoint with no key check would spend the
    operator's FRED/Finnhub/Yahoo quota on every caller (docs/api-design.md #5).
    Set ``PYQUANT_API_ALLOW_UNAUTHENTICATED=1`` for local development only.
    `app.py`'s lifespan hook makes the same check once at process startup, so a
    misconfigured deployment fails to boot rather than accepting traffic and
    500ing on the first real request -- this per-request check stays as
    defense in depth (e.g. against the env var changing after startup).
    """
    configured = _configured_api_keys()
    if not configured:
        if os.environ.get("PYQUANT_API_ALLOW_UNAUTHENTICATED") == "1":
            return
        raise HTTPException(status_code=500, detail=_UNCONFIGURED_KEYS_MESSAGE)
    # Compare as bytes, not str: Starlette decodes headers as latin-1, so a byte
    # >127 in X-API-Key produces a non-ASCII str, and hmac.compare_digest raises
    # TypeError on non-ASCII str input (verified) -- turning a bad key into an
    # unhandled 500 instead of a clean 401 (PYQ-145). Bytes comparison has no
    # such restriction, and utf-8-encoding a latin-1-decoded str is always
    # lossless (every latin-1 codepoint is valid utf-8).
    if x_api_key is None or not any(
        hmac.compare_digest(x_api_key.encode("utf-8"), k.encode("utf-8")) for k in configured
    ):
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
