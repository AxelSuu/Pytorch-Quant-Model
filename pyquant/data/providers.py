"""Pluggable OHLCV price providers (PYQ-258, the concrete half of PYQ-214).

yfinance is currently the sole source of OHLCV **and** VIX **and** sector ETFs
**and** options -- four of the project's data sources behind one unofficial,
unversioned, ToS-ambiguous scraper of Yahoo's internal endpoints. Every
enrichment degrades gracefully except the one everything else depends on: if
prices fail, nothing works. PYQ-228 records the concrete harm already done
(a silent 0.2 -> 1.4 major jump, and an `auto_adjust` default that flipped
mid-series and changes every price level).

Two properties matter more than which vendor is picked:

1. **An interface**, so switching is a config change rather than a rewrite.
2. **An explicit adjustment convention**, since PYQ-228 showed it was previously
   decided by whichever version resolved.

Every provider must return the same thing: a tz-naive, date-indexed frame with
exactly ``Open/High/Low/Close/Volume`` as floats, ascending by date, and
split/dividend **adjusted** (``prices.AUTO_ADJUST``). ``assert_ohlcv_contract``
is the executable statement of that, and both providers are tested against it so
they cannot drift apart.

Deliberately no new hard dependency: the Tiingo provider talks to a documented
JSON REST endpoint through ``requests``, which is already required.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol, runtime_checkable

import pandas as pd

from pyquant.data.retry import with_retry

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class PriceProviderError(RuntimeError):
    """A provider could not return usable OHLCV data."""


@runtime_checkable
class PriceProvider(Protocol):
    """Anything that can supply adjusted daily OHLCV for one symbol."""

    name: str

    def fetch_ohlcv(
        self,
        symbol: str,
        *,
        period: str = "5y",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Return adjusted daily OHLCV. Must satisfy ``assert_ohlcv_contract``."""
        ...


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a vendor frame into the shared contract.

    Shared rather than per-provider so a new vendor cannot accidentally ship a
    tz-aware index or an unsorted frame -- the differences that are invisible
    until they silently misalign a join.
    """
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise PriceProviderError(f"provider response is missing column(s): {missing}")
    out = df.loc[:, OHLCV_COLUMNS].astype(float)
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out.index.name = "Date"
    return out.sort_index()


def assert_ohlcv_contract(df: pd.DataFrame) -> None:
    """Raise unless ``df`` satisfies the cross-provider contract.

    Written as an assertion helper rather than prose so the two providers are
    checked against the *same* statement, which is what makes them substitutable.
    """
    if list(df.columns) != OHLCV_COLUMNS:
        raise PriceProviderError(f"columns must be exactly {OHLCV_COLUMNS}, got {list(df.columns)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise PriceProviderError(f"index must be a DatetimeIndex, got {type(df.index).__name__}")
    if df.index.tz is not None:
        raise PriceProviderError("index must be tz-naive")
    if df.index.name != "Date":
        raise PriceProviderError(f"index must be named 'Date', got {df.index.name!r}")
    if not df.index.is_monotonic_increasing:
        raise PriceProviderError("index must be ascending")
    if not all(str(df[c].dtype) == "float64" for c in OHLCV_COLUMNS):
        raise PriceProviderError(f"all OHLCV columns must be float64, got {dict(df.dtypes)}")


class YFinanceProvider:
    """Yahoo Finance via the unofficial ``yfinance`` client (the default)."""

    name = "yfinance"

    def fetch_ohlcv(  # noqa: D102 - contract documented on PriceProvider.fetch_ohlcv
        self,
        symbol: str,
        *,
        period: str = "5y",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        import yfinance as yf

        from pyquant.data.prices import AUTO_ADJUST

        ticker = yf.Ticker(symbol)

        # Honor an explicit range if *either* bound is given (yfinance accepts
        # start or end alone); only fall back to period when neither is set, so
        # passing just start (e.g. "everything since IPO") isn't silently ignored.
        def _load() -> pd.DataFrame:
            if start or end:
                return ticker.history(start=start, end=end, auto_adjust=AUTO_ADJUST)
            return ticker.history(period=period, auto_adjust=AUTO_ADJUST)

        # A transient hiccup here otherwise hard-fails the whole panel build;
        # retry a couple of times before giving up (PYQ-215).
        df = with_retry(_load, description=f"fetch_prices({symbol})")
        if df is None or df.empty:
            raise ValueError(f"No price data found for {symbol!r}")
        return normalize_ohlcv(df)


class TiingoProvider:
    """Tiingo's licensed daily EOD API -- a real vendor with real terms.

    Chosen as the alternative over Alpha Vantage (25 requests/day on the free
    tier, too tight even for development) and over Polygon/EODHD (paid only).
    Tiingo's free tier covers enough symbols and history to be a genuine
    fallback, and unlike yfinance it is an API the operator has agreed terms for
    -- which is the point PYQ-320 makes about serving anything publicly.

    ``adjClose``/``adjOpen``/... are requested so the adjustment convention
    matches ``prices.AUTO_ADJUST`` rather than differing per provider (PYQ-228).
    """

    name = "tiingo"
    BASE_URL = "https://api.tiingo.com/tiingo/daily"

    def __init__(self, api_key: str | None = None, session=None) -> None:
        """``session`` is injectable so tests drive the real parsing offline."""
        self.api_key = api_key or os.environ.get("TIINGO_API_KEY")
        self._session = session

    def fetch_ohlcv(  # noqa: D102 - contract documented on PriceProvider.fetch_ohlcv
        self,
        symbol: str,
        *,
        period: str = "5y",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        if not self.api_key:
            raise PriceProviderError(
                "TiingoProvider needs an API key: set TIINGO_API_KEY or pass api_key=. "
                "Free keys: https://www.tiingo.com/"
            )
        import requests

        from pyquant.data.prices import _period_start

        session = self._session or requests
        start_date = start or _period_start(period)
        params = {"startDate": start_date, "format": "json", "resampleFreq": "daily"}
        if end:
            params["endDate"] = end

        def _load():
            response = session.get(
                f"{self.BASE_URL}/{symbol.lower()}/prices",
                params=params,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Token {self.api_key}",
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()

        payload = with_retry(_load, description=f"tiingo({symbol})")
        if not payload:
            raise ValueError(f"No price data found for {symbol!r}")

        frame = pd.DataFrame(payload)
        # Adjusted fields, to match AUTO_ADJUST=True. Tiingo serves both, so the
        # convention is chosen here rather than inherited from a default.
        frame = frame.rename(
            columns={
                "adjOpen": "Open",
                "adjHigh": "High",
                "adjLow": "Low",
                "adjClose": "Close",
                "adjVolume": "Volume",
            }
        )
        frame.index = pd.to_datetime(frame["date"], format="mixed", utc=True).dt.tz_localize(None)
        return normalize_ohlcv(frame)


_PROVIDERS: dict[str, type] = {"yfinance": YFinanceProvider, "tiingo": TiingoProvider}


def get_provider(name: str = "yfinance", **kwargs) -> PriceProvider:
    """Construct the configured price provider by name."""
    try:
        factory = _PROVIDERS[name.strip().lower()]
    except KeyError:
        raise PriceProviderError(
            f"Unknown price provider {name!r}. Available: {sorted(_PROVIDERS)}."
        ) from None
    return factory(**kwargs)
