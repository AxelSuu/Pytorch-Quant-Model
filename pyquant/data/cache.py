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
import os
import time
import uuid
from pathlib import Path

import pandas as pd

from pyquant import provenance

logger = logging.getLogger(__name__)


def _tmp_sibling(path: Path) -> Path:
    return path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so a reader never observes a partial file.

    Writes to a uniquely-named sibling temp file, then ``os.replace()``s it
    into place -- a rename is atomic on the same filesystem, unlike writing
    directly to ``path`` (PYQ-155: a crash mid-write, or two threads racing
    the same fingerprint, could otherwise leave a truncated file that a
    stale-but-present sibling would still present as valid).
    """
    tmp_path = _tmp_sibling(path)
    tmp_path.write_text(text)
    os.replace(tmp_path, path)


def _atomic_write_pickle(path: Path, panel: pd.DataFrame) -> None:
    """Same atomicity guarantee as ``_atomic_write_text``, for a pickled DataFrame."""
    tmp_path = _tmp_sibling(path)
    panel.to_pickle(tmp_path)
    os.replace(tmp_path, path)


def fingerprint_key(fingerprint: dict) -> str:
    """Stable short key for a cache fingerprint dict (order-independent)."""
    payload = json.dumps(fingerprint, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _entry_path(cache_dir: Path, key: str) -> Path:
    """Path of the pickled panel for ``key``."""
    return cache_dir / f"{key}.pkl"


def _meta_path(entry_path: Path) -> Path:
    """Path of the sidecar metadata file beside a cache entry."""
    return entry_path.with_suffix(".meta.json")


def read_pin_metadata(cache_dir: Path, name: str) -> dict | None:
    """Return a pin's provenance metadata, if it was recorded (and readable)."""
    path = _meta_path(_entry_path(cache_dir / "pins", name))
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("Pin %s metadata is corrupt; treating as unrecorded.", name)
        return None


def read_cache(
    cache_dir: Path, key: str, ttl_seconds: float | None, now: float | None = None
) -> pd.DataFrame | None:
    """Return the cached panel for ``key`` if present, valid, and not past its TTL.

    A corrupt/truncated entry (PYQ-155: possible before writes were made atomic,
    and still possible from e.g. a disk-full write) is treated as a cache miss --
    the caller refetches -- rather than propagating a raw unpickling error out of
    ``build_panel``.
    """
    path = _entry_path(cache_dir, key)
    if not path.exists():
        return None
    if ttl_seconds is not None:
        meta_path = _meta_path(path)
        try:
            cached_at = json.loads(meta_path.read_text())["cached_at"] if meta_path.exists() else 0.0
        except (json.JSONDecodeError, KeyError):
            logger.warning("Cache metadata for %s is corrupt; treating as a miss.", key)
            return None
        clock = now if now is not None else time.time()
        if clock - cached_at > ttl_seconds:
            return None
    try:
        return pd.read_pickle(path)
    except Exception:
        logger.warning("Cache entry for %s is corrupt; treating as a miss.", key)
        return None


def write_cache(cache_dir: Path, key: str, panel: pd.DataFrame, now: float | None = None) -> None:
    """Persist ``panel`` under ``key``, timestamped for TTL expiry."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _entry_path(cache_dir, key)
    _atomic_write_pickle(path, panel)
    _atomic_write_text(
        _meta_path(path), json.dumps({"cached_at": now if now is not None else time.time()})
    )


def read_pin(cache_dir: Path, name: str) -> pd.DataFrame | None:
    """Return a pinned dataset snapshot, ignoring any TTL -- exact reproducibility.

    A pin is meant to be exactly reproducible, so a corrupt entry cannot
    silently fall back to a refetch the way a TTL cache miss can (PYQ-155) --
    that would defeat the point of pinning. It raises, with a message naming
    the pin, rather than a bare unpickling error from deep in pandas.
    """
    path = _entry_path(cache_dir / "pins", name)
    if not path.exists():
        return None

    try:
        panel = pd.read_pickle(path)
    except Exception as exc:
        raise ValueError(f"Pin '{name}' at {path} is corrupt and cannot be read: {exc}") from exc
    meta = read_pin_metadata(cache_dir, name)
    if meta is None:
        logger.warning("Pin %s has no recorded metadata; it may predate the current code.", name)
        return panel

    recorded_version = meta.get("pyquant_version")
    current_version = provenance.package_version()
    if recorded_version != current_version:
        logger.warning(
            "Pin %s was created with PyQuant version %s, but this run uses %s; "
            "its feature values may not be reproducible.",
            name,
            recorded_version,
            current_version,
        )
    recorded_columns = meta.get("columns")
    columns = list(panel.columns)
    if recorded_columns != columns:
        logger.warning(
            "Pin %s metadata records columns %s, but its panel contains %s; "
            "the pin may have been created by incompatible code.",
            name,
            recorded_columns,
            columns,
        )
    return panel


def write_pin(cache_dir: Path, name: str, panel: pd.DataFrame) -> None:
    """Save a named, TTL-exempt dataset snapshot for later exact reuse."""
    pin_dir = cache_dir / "pins"
    pin_dir.mkdir(parents=True, exist_ok=True)
    path = _entry_path(pin_dir, name)
    _atomic_write_pickle(path, panel)
    _atomic_write_text(
        _meta_path(path),
        json.dumps(
            {
                "pyquant_version": provenance.package_version(),
                "git_sha": provenance.git_sha(),
                "created_at": time.time(),
                "columns": list(panel.columns),
                "n_rows": len(panel),
            },
            indent=2,
        ),
    )


# --- Management helpers (PYQ-221) -------------------------------------------
# The TTL only gates *read* validity; nothing ever deleted the underlying files,
# so `.cache/pyquant/` grew unboundedly. These back the `pyquant cache`
# subcommand. Top-level `*.pkl` are TTL entries; `pins/` is deliberately kept
# separate so pruning never touches a reproducibility pin.


def _entry_files(cache_dir: Path) -> list[Path]:
    """TTL cache entry pickles (excludes the pins/ subdirectory)."""
    if not cache_dir.exists():
        return []
    return sorted(cache_dir.glob("*.pkl"))


def list_pins(cache_dir: Path) -> list[str]:
    """Names of all saved pins."""
    pin_dir = cache_dir / "pins"
    if not pin_dir.exists():
        return []
    return sorted(p.stem for p in pin_dir.glob("*.pkl"))


def cache_stats(cache_dir: Path) -> dict:
    """Entry count, total on-disk size (entries + their meta), and pin names."""
    entries = _entry_files(cache_dir)
    total_bytes = 0
    for path in entries:
        total_bytes += path.stat().st_size
        meta = _meta_path(path)
        if meta.exists():
            total_bytes += meta.stat().st_size
    return {
        "entry_count": len(entries),
        "total_bytes": total_bytes,
        "pins": list_pins(cache_dir),
    }


def prune_expired(
    cache_dir: Path, ttl_seconds: float, now: float | None = None
) -> list[str]:
    """Delete TTL entries older than ``ttl_seconds``; return the removed keys.

    Valid (unexpired) entries and all pins are left untouched.
    """
    clock = now if now is not None else time.time()
    removed: list[str] = []
    for path in _entry_files(cache_dir):
        meta = _meta_path(path)
        cached_at = json.loads(meta.read_text())["cached_at"] if meta.exists() else 0.0
        if clock - cached_at > ttl_seconds:
            path.unlink(missing_ok=True)
            meta.unlink(missing_ok=True)
            removed.append(path.stem)
    return removed


def remove_pin(cache_dir: Path, name: str) -> bool:
    """Delete a named pin; return True if it existed."""
    path = _entry_path(cache_dir / "pins", name)
    if path.exists():
        path.unlink()
        _meta_path(path).unlink(missing_ok=True)
        return True
    return False
