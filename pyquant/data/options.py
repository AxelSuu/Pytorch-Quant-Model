"""Options-implied market context (current snapshot).

IMPORTANT: Yahoo Finance only exposes the *current* option chain, not history.
So options features here are a point-in-time market-sentiment snapshot
(put/call ratio, ATM implied vol, IV skew) used as CLI context for `forecast`
and `scan` — they are NOT fed to the TFT as time-varying inputs, because a
constant/lookahead value would carry no historical signal. The model's
volatility signal comes from the historical ``Realized_Vol_20`` feature in
:mod:`pyquant.data.prices`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import yfinance as yf

from pyquant.data.prices import AUTO_ADJUST

logger = logging.getLogger(__name__)


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
