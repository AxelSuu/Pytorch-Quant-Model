"""FastAPI dependencies: settings, the bundle LRU cache, per-bundle locks, API-key auth.

Per docs/api-design.md #4/#5. Module-level singletons rather than app.state, since this
is the v1 in-process scaffold the design note describes -- state is lost on restart and
invisible to a second instance, which is the documented trigger to graduate to a shared
backend, not a bug in this version.
"""

from __future__ import annotations

import hmac
import os
import threading
from collections import OrderedDict

from fastapi import Header, HTTPException

from pyquant.config import Settings, load_settings
from pyquant.models import tft


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

    def get(self, name: str, settings: Settings) -> tft.ModelBundle:
        """Return the cached bundle, loading (and caching) it on a miss."""
        name = name.upper()
        with self._lock:
            cached = self._cache.get(name)
            if cached is not None:
                self._cache.move_to_end(name)
                return cached
        # tft.load() does real file I/O + deserialisation; keep it outside the lock
        # so loading one bundle does not block requests for already-cached ones.
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
        self._locks: dict[str, threading.Lock] = {}
        self._registry_lock = threading.Lock()

    def get(self, name: str) -> threading.Lock:
        """Return this bundle's lock, creating it on first use."""
        name = name.upper()
        with self._registry_lock:
            lock = self._locks.get(name)
            if lock is None:
                lock = threading.Lock()
                self._locks[name] = lock
            return lock


_PREDICTION_LOCKS = _PredictionLocks()


def get_prediction_lock(name: str) -> threading.Lock:
    """FastAPI-callable accessor for a bundle's prediction lock."""
    return _PREDICTION_LOCKS.get(name)


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
    """
    configured = [k for k in os.environ.get("PYQUANT_API_KEYS", "").split(",") if k]
    if not configured:
        if os.environ.get("PYQUANT_API_ALLOW_UNAUTHENTICATED") == "1":
            return
        raise HTTPException(
            status_code=500,
            detail=(
                "PYQUANT_API_KEYS is not set. Set it (comma-separated keys), or set "
                "PYQUANT_API_ALLOW_UNAUTHENTICATED=1 for local development only."
            ),
        )
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
