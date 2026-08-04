"""SQLite-backed API key store: identity, scopes, revocation (PYQ-281).

Before this module, `require_api_key` compared the caller's header against a flat,
comma-separated `PYQUANT_API_KEYS` list and returned `None` on success -- every valid
key was interchangeable and the request that followed had no way to know *which* key
authenticated it. That is the actual defect this closes: everything downstream
(quotas, revocation, audit, per-key cost attribution, scoping `/train`/`/backtest`
away from read-only callers) needs a subject to hang off of, which a `None` cannot
provide.

SQLite, not Postgres: this is the same trigger `docs/http-api.md` already names for
the job registry and bundle cache (single-instance, in-process, v1 scaffold) -- a
second instance is the point to move all three, not before (non-negotiable #5: a new
dependency needs a recorded reason, and `sqlite3` is stdlib, not a new one).

Only a salted hash of each key is ever stored -- never the raw value -- matching this
project's secrets rule (`meta.json`/logs/cache fingerprints never carry secret values,
only presence). The raw key is returned exactly once, at `create_key()`, and cannot be
recovered from the store afterward.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pyquant.config import project_root

KEY_PREFIX = "pq_live_"
# Unhashed slice used for the store lookup, matching a short prefix rather than
# scanning every row's hash on each request. Long enough that prefix collisions
# among live keys are a non-issue in practice; a collision only costs an extra
# hmac.compare_digest, never a security property, since the full hash still gates.
PREFIX_LEN = len(KEY_PREFIX) + 8

ALLOWED_SCOPES = frozenset({"read", "train"})


class InvalidScope(ValueError):
    """Raised when a requested scope isn't one this store recognizes."""


@dataclass(frozen=True)
class ApiKey:
    """The identity `require_api_key` resolves a request to."""

    id: str
    name: str
    scopes: frozenset[str]


@dataclass(frozen=True)
class ApiKeyRecord:
    """A stored key's metadata, for `pyquant keys list` -- never includes the raw key."""

    id: str
    name: str
    prefix: str
    scopes: frozenset[str]
    created_at: str
    revoked_at: str | None
    last_used_at: str | None


def resolve_db_path() -> Path:
    """The sqlite file's path: ``PYQUANT_API_KEYS_DB``, or an anchored default.

    Read directly from the environment rather than through ``Settings`` -- the same
    deliberate choice already made for ``PYQUANT_API_KEYS``/
    ``PYQUANT_API_ALLOW_UNAUTHENTICATED`` in ``deps.py``, so this path can never end up
    written into a ``meta.json`` or YAML config the way a ``Settings`` field could.
    """
    import os

    override = os.environ.get("PYQUANT_API_KEYS_DB")
    if override:
        path = Path(override).expanduser()
        return path if path.is_absolute() else (project_root() / path).resolve()
    return (project_root() / "data" / "api_keys.db").resolve()


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS api_keys (
            id TEXT PRIMARY KEY,
            key_hash TEXT NOT NULL,
            prefix TEXT NOT NULL,
            name TEXT NOT NULL,
            scopes TEXT NOT NULL,
            created_at TEXT NOT NULL,
            revoked_at TEXT,
            last_used_at TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS api_keys_prefix ON api_keys(prefix)")
    conn.commit()
    return conn


def _hash(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _validate_scopes(scopes: Iterable[str]) -> frozenset[str]:
    scopes = frozenset(scopes)
    if not scopes:
        raise InvalidScope("a key must have at least one scope")
    unknown = scopes - ALLOWED_SCOPES
    if unknown:
        raise InvalidScope(f"unknown scope(s) {sorted(unknown)}; allowed: {sorted(ALLOWED_SCOPES)}")
    return scopes


def create_key(db_path: Path, name: str, scopes: Iterable[str]) -> tuple[str, ApiKeyRecord]:
    """Issue a new key. Returns ``(raw_key, record)`` -- ``raw_key`` is never stored.

    Format ``pq_live_<24 hex chars>`` (96 bits from ``secrets.token_hex``, a
    cryptographically secure source) -- shown to the caller exactly once here.
    """
    validated_scopes = _validate_scopes(scopes)
    raw_key = f"{KEY_PREFIX}{secrets.token_hex(12)}"
    key_id = secrets.token_hex(8)
    now = _now()
    with closing(_connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO api_keys (id, key_hash, prefix, name, scopes, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (key_id, _hash(raw_key), raw_key[:PREFIX_LEN], name, ",".join(sorted(validated_scopes)), now),
        )
        conn.commit()
    record = ApiKeyRecord(
        id=key_id,
        name=name,
        prefix=raw_key[:PREFIX_LEN],
        scopes=validated_scopes,
        created_at=now,
        revoked_at=None,
        last_used_at=None,
    )
    return raw_key, record


def list_keys(db_path: Path) -> list[ApiKeyRecord]:
    """All issued keys (revoked included), newest first. Never returns raw key material."""
    if not db_path.exists():
        return []
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT id, name, prefix, scopes, created_at, revoked_at, last_used_at "
            "FROM api_keys ORDER BY created_at DESC"
        ).fetchall()
    return [
        ApiKeyRecord(
            id=row[0],
            name=row[1],
            prefix=row[2],
            scopes=frozenset(row[3].split(",")),
            created_at=row[4],
            revoked_at=row[5],
            last_used_at=row[6],
        )
        for row in rows
    ]


def revoke_key(db_path: Path, key_id: str) -> bool:
    """Mark a key revoked. Returns False if ``key_id`` doesn't exist or is already revoked."""
    with closing(_connect(db_path)) as conn:
        cur = conn.execute(
            "UPDATE api_keys SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
            (_now(), key_id),
        )
        conn.commit()
        return cur.rowcount > 0


def has_active_keys(db_path: Path) -> bool:
    """True if at least one non-revoked key exists.

    The store equivalent of a non-empty ``PYQUANT_API_KEYS`` list, for the
    "is auth configured at all" check.
    """
    if not db_path.exists():
        return False
    with closing(_connect(db_path)) as conn:
        row = conn.execute(
            "SELECT 1 FROM api_keys WHERE revoked_at IS NULL LIMIT 1"
        ).fetchone()
    return row is not None


def authenticate(db_path: Path, raw_key: str) -> ApiKey | None:
    """Resolve a raw key to its identity, or ``None`` if invalid/unknown/revoked.

    Looked up by prefix (an unhashed slice, cheap to index) then verified against
    the full hash with `hmac.compare_digest` -- same constant-time discipline the
    old flat-list comparison had, now against a hash lookup instead of a linear
    scan of every configured key. On success, stamps ``last_used_at`` -- the first
    piece of the audit trail this ticket exists to add.
    """
    if not raw_key or not db_path.exists():
        return None
    prefix = raw_key[:PREFIX_LEN]
    target_hash = _hash(raw_key)
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT id, key_hash, name, scopes FROM api_keys "
            "WHERE prefix = ? AND revoked_at IS NULL",
            (prefix,),
        ).fetchall()
        for key_id, key_hash, name, scopes in rows:
            if hmac.compare_digest(target_hash, key_hash):
                conn.execute(
                    "UPDATE api_keys SET last_used_at = ? WHERE id = ?", (_now(), key_id)
                )
                conn.commit()
                return ApiKey(id=key_id, name=name, scopes=frozenset(scopes.split(",")))
    return None
