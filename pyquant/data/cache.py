"""Local cache for assembled data panels.

Two complementary mechanisms:
  - a TTL cache keyed by a fingerprint of (symbol, date range, enabled
    sources) -- speeds up repeated train/forecast/explain runs and is easy on
    upstream rate limits.
  - named "pins" that never expire -- an explicit, reproducible dataset
    snapshot so a specific experiment can be re-run later against the exact
    same data instead of whatever happens to be live that day.

Entries are pickled DataFrames. This is safe here specifically because the
cache is purely self-generated and never loaded from an untrusted source --
unlike a shared interchange format, there is no path for an external file to
reach ``pd.read_pickle`` (the same trust boundary already relied on for
dataset_params.pt in pyquant.models.tft).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def fingerprint_key(fingerprint: dict) -> str:
    """Stable short key for a cache fingerprint dict (order-independent)."""
    payload = json.dumps(fingerprint, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _entry_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.pkl"


def _meta_path(entry_path: Path) -> Path:
    return entry_path.with_suffix(".meta.json")


def read_cache(
    cache_dir: Path, key: str, ttl_seconds: float | None, now: float | None = None
) -> pd.DataFrame | None:
    """Return the cached panel for ``key`` if present and not past its TTL."""
    path = _entry_path(cache_dir, key)
    if not path.exists():
        return None
    if ttl_seconds is not None:
        meta_path = _meta_path(path)
        cached_at = json.loads(meta_path.read_text())["cached_at"] if meta_path.exists() else 0.0
        clock = now if now is not None else time.time()
        if clock - cached_at > ttl_seconds:
            return None
    return pd.read_pickle(path)


def write_cache(cache_dir: Path, key: str, panel: pd.DataFrame, now: float | None = None) -> None:
    """Persist ``panel`` under ``key``, timestamped for TTL expiry."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _entry_path(cache_dir, key)
    panel.to_pickle(path)
    _meta_path(path).write_text(json.dumps({"cached_at": now if now is not None else time.time()}))


def read_pin(cache_dir: Path, name: str) -> pd.DataFrame | None:
    """Return a pinned dataset snapshot, ignoring any TTL -- exact reproducibility."""
    path = _entry_path(cache_dir / "pins", name)
    return pd.read_pickle(path) if path.exists() else None


def write_pin(cache_dir: Path, name: str, panel: pd.DataFrame) -> None:
    """Save a named, TTL-exempt dataset snapshot for later exact reuse."""
    pin_dir = cache_dir / "pins"
    pin_dir.mkdir(parents=True, exist_ok=True)
    panel.to_pickle(_entry_path(pin_dir, name))
