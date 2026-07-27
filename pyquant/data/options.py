"""Options-implied market context: a current snapshot, plus an accumulating history.

IMPORTANT: Yahoo Finance only exposes the *current* option chain, not history. So
``fetch_options_snapshot`` is a point-in-time market-sentiment reading used as CLI
context for `forecast`/`scan` — on its own it is NOT fed to the TFT as a
time-varying input, because a constant/lookahead value would carry no historical
signal. The model's volatility signal comes from the historical
``Realized_Vol_20`` feature in :mod:`pyquant.data.prices`.

``append_snapshot``/``load_snapshot_history`` are PYQ-254's route out of that
limitation: since there is no way to *fetch* a historical options-implied series,
the only way to ever have one is to start *recording* today's snapshot every day,
from today. Useless on day one; a genuinely proprietary dataset after enough days
accumulate. Same publication-timing discipline as PYQ-101/PYQ-129 applies: a
snapshot is recorded under the date it was actually observed (US/Eastern), so
``build_panel`` can only ever join it onto that day's row or later.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from pyquant.data.prices import AUTO_ADJUST

logger = logging.getLogger(__name__)

_EXCHANGE_TZ = ZoneInfo("America/New_York")

# Columns load_snapshot_history() produces once it has enough history to be useful.
SNAPSHOT_COLUMNS = ["OptionsPutCallRatio", "OptionsATMIV", "OptionsIVSkew"]

# Below this many distinct recorded days, the accumulated history is too sparse to
# be a meaningful per-day feature -- nearly every training row would be
# structurally missing it, the same trap PYQ-140 found for sentiment at the
# vendor's own free-tier coverage limit. Skipped with a logged notice rather than
# joined as a nearly-empty column.
MIN_SNAPSHOT_DAYS = 20


@dataclass
class OptionsSnapshot:
    """Current options-implied sentiment for a symbol."""

    put_call_ratio: float | None
    atm_iv: float | None
    iv_skew: float | None
    expiry: str | None

    @property
    def sentiment_label(self) -> str:
        """Human-readable read on the put/call ratio."""
        if self.put_call_ratio is None:
            return "n/a"
        if self.put_call_ratio > 1.2:
            return "bearish (heavy puts)"
        if self.put_call_ratio < 0.7:
            return "bullish (heavy calls)"
        return "neutral"


def _spot_price(ticker: yf.Ticker) -> float | None:
    """Best-effort current price, trying ``fast_info`` before falling back.

    Returns ``None`` rather than raising when every route fails: options data is
    display-only and must never break a forecast.
    """
    try:
        fast = ticker.fast_info
        price = fast.get("last_price") if hasattr(fast, "get") else fast["lastPrice"]
        if price:
            return float(price)
    except Exception:
        pass
    try:
        hist = ticker.history(period="1d", auto_adjust=AUTO_ADJUST)  # PYQ-228
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


def fetch_options_snapshot(symbol: str) -> OptionsSnapshot:
    """Compute a current options-sentiment snapshot. Fields are None on failure."""
    empty = OptionsSnapshot(None, None, None, None)
    try:
        ticker = yf.Ticker(symbol)
        expiries = ticker.options
        if not expiries:
            logger.info("No options listed for %s", symbol)
            return empty
        expiry = expiries[0]
        chain = ticker.option_chain(expiry)
        calls, puts = chain.calls, chain.puts
        spot = _spot_price(ticker)
        if spot is None or calls.empty or puts.empty:
            return OptionsSnapshot(None, None, None, expiry)

        call_vol = calls["volume"].fillna(0).sum()
        put_vol = puts["volume"].fillna(0).sum()
        put_call = float(put_vol / call_vol) if call_vol > 0 else None

        atm_call = calls.iloc[(calls["strike"] - spot).abs().argmin()]
        atm_put = puts.iloc[(puts["strike"] - spot).abs().argmin()]
        atm_iv = float(np.nanmean([atm_call["impliedVolatility"], atm_put["impliedVolatility"]]))

        # IV skew: OTM put IV (~10% below spot) minus OTM call IV (~10% above).
        otm_put = puts.iloc[(puts["strike"] - spot * 0.9).abs().argmin()]
        otm_call = calls.iloc[(calls["strike"] - spot * 1.1).abs().argmin()]
        iv_skew = float(otm_put["impliedVolatility"] - otm_call["impliedVolatility"])

        return OptionsSnapshot(put_call, atm_iv, iv_skew, expiry)
    except Exception as exc:
        logger.warning("Could not fetch options snapshot for %s: %s", symbol, exc)
        return empty


def append_snapshot(symbol: str, settings) -> Path:
    """Fetch today's snapshot and append it to ``symbol``'s accumulated history.

    Written to ``settings.options_history_dir / f"{symbol}.jsonl"`` (PYQ-254).
    ``settings`` is typed loosely (not ``pyquant.config.Settings``) to avoid a
    circular import; any object with an ``options_history_dir`` path attribute
    works. The recorded ``date`` is the US/Eastern calendar date the snapshot was
    actually observed on, so a later join can never backfill it onto an earlier
    row (the same discipline PYQ-101/PYQ-129 apply to macro/sentiment).
    """
    symbol = symbol.upper()
    snap = fetch_options_snapshot(symbol)
    path = Path(settings.options_history_dir) / f"{symbol}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    now_et = dt.datetime.now(dt.timezone.utc).astimezone(_EXCHANGE_TZ)
    row = {
        "date": now_et.date().isoformat(),
        "observed_at": now_et.isoformat(timespec="seconds"),
        "put_call_ratio": snap.put_call_ratio,
        "atm_iv": snap.atm_iv,
        "iv_skew": snap.iv_skew,
        "expiry": snap.expiry,
    }
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return path


def load_snapshot_history(symbol: str, settings) -> pd.DataFrame:
    """Return ``symbol``'s accumulated snapshot history as a date-indexed frame.

    Empty (with ``SNAPSHOT_COLUMNS``, so callers need no special case) until at
    least ``MIN_SNAPSHOT_DAYS`` distinct days have been recorded -- before that,
    the feature would be structurally missing from nearly every training row.
    """
    symbol = symbol.upper()
    path = Path(settings.options_history_dir) / f"{symbol}.jsonl"
    empty = pd.DataFrame(columns=SNAPSHOT_COLUMNS, dtype=float)
    if not path.exists():
        return empty

    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return empty
    rows = [json.loads(line) for line in lines]

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["date"])
    # The same day recorded more than once (re-running `snapshot`): keep the
    # latest observation rather than averaging or taking the first.
    df = df.sort_values("observed_at").drop_duplicates("Date", keep="last")
    df = df.set_index("Date").sort_index()
    if df.index.nunique() < MIN_SNAPSHOT_DAYS:
        return empty

    out = pd.DataFrame(
        {
            "OptionsPutCallRatio": df["put_call_ratio"],
            "OptionsATMIV": df["atm_iv"],
            "OptionsIVSkew": df["iv_skew"],
        }
    )
    return out.astype(float)
