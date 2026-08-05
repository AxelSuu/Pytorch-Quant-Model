"""Nightly-precomputed forecast store (features.md#pyq-282).

``GET /forecast/{symbol}`` used to run the full live pipeline
(``generate_forecast``) on every request. investigations.md#pyq-319 measured
that at ~65s, ~98% of it vendor fetch/panel build -- for daily-bar data that
only actually changes once a trading session closes. This module is the
read/write side of the fix: ``pyquant precompute`` (cli/app.py), meant to run
nightly after market close, computes every trained symbol's forecast once and
writes it here; the route becomes a read.

A single SQLite file, not a new dependency (``sqlite3`` is stdlib,
non-negotiable #5) -- keyed by symbol, and unlike ``pyquant.data.cache``'s TTL
panel cache, never time-expired: a precomputed forecast is replaced wholesale
by the next night's write, not pruned by age. Staleness (has the job actually
run recently enough) is a read-time judgement -- see ``is_stale`` -- not a
write-time TTL, because "recently enough" means "since the last trading
session closed," which a fixed TTL cannot express across weekends/holidays.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pyquant.config import Settings
from pyquant.data.trading_calendar import latest_session_before

_SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    symbol TEXT PRIMARY KEY,
    as_of TEXT NOT NULL,
    computed_at TEXT NOT NULL,
    payload TEXT NOT NULL
)
"""


@dataclass
class StoredForecast:
    """One symbol's precomputed forecast, as written by ``write_forecast``."""

    symbol: str
    as_of: str
    computed_at: str
    payload: dict[str, Any]


def _connect(settings: Settings) -> sqlite3.Connection:
    db_path = Path(settings.forecast_store_db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(_SCHEMA)
    return conn


def write_forecast(
    settings: Settings, symbol: str, as_of: str, computed_at: str, payload: dict[str, Any]
) -> None:
    """Upsert ``symbol``'s precomputed forecast, replacing any prior entry whole."""
    symbol = symbol.upper()
    with _connect(settings) as conn:
        conn.execute(
            "INSERT INTO forecasts (symbol, as_of, computed_at, payload) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(symbol) DO UPDATE SET "
            "as_of = excluded.as_of, computed_at = excluded.computed_at, "
            "payload = excluded.payload",
            (symbol, as_of, computed_at, json.dumps(payload)),
        )


def read_forecast(settings: Settings, symbol: str) -> StoredForecast | None:
    """``symbol``'s precomputed forecast, or ``None`` if never written."""
    symbol = symbol.upper()
    with _connect(settings) as conn:
        row = conn.execute(
            "SELECT symbol, as_of, computed_at, payload FROM forecasts WHERE symbol = ?",
            (symbol,),
        ).fetchone()
    if row is None:
        return None
    return StoredForecast(
        symbol=row[0], as_of=row[1], computed_at=row[2], payload=json.loads(row[3])
    )


def is_stale(as_of: str, now: pd.Timestamp | None = None) -> bool:
    """True when ``as_of`` predates the session the nightly job should cover.

    Compares against ``latest_session_before(now)`` -- the last session before
    *today* -- rather than requiring ``as_of`` to already equal today's
    session, since the job has all of the intervening night to run.
    """
    now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else pd.Timestamp(now)
    expected = latest_session_before(now)
    return pd.Timestamp(as_of) < expected
