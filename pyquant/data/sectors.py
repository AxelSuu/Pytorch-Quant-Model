"""Cross-asset / sector features.

Daily returns of sector ETFs (and a broad-market proxy) provide context beyond
the target ticker — sector rotation and market beta. These are genuine
time-varying features fed to the TFT.
"""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from pyquant.data.prices import AUTO_ADJUST

logger = logging.getLogger(__name__)


def fetch_sector_returns(
    etfs: list[str],
    start: str | None = None,
    end: str | None = None,
    period: str = "5y",
) -> pd.DataFrame:
    """Return a date-indexed DataFrame of daily ETF returns.

    Columns are named ``SEC_<ETF>``. Empty DataFrame on failure.
    """
    if not etfs:
        return pd.DataFrame()
    try:
        data = yf.download(
            etfs,
            start=start,
            end=end,
            # Honor an explicit range if *either* bound is given; only fall back
            # to period when neither is set (PYQ-112).
            period=None if (start or end) else period,
            progress=False,
            auto_adjust=AUTO_ADJUST,  # PYQ-228: one convention, one place
        )
    except Exception as exc:
        logger.warning("Could not fetch sector ETFs: %s", exc)
        return pd.DataFrame()

    if data is None or data.empty:
        return pd.DataFrame()

    # yf.download returns a column MultiIndex (field, ticker) for multiple tickers.
    close = data["Close"] if "Close" in data.columns.get_level_values(0) else data
    if isinstance(close, pd.Series):
        close = close.to_frame()

    returns = close.pct_change()
    returns.columns = [f"SEC_{c}" for c in returns.columns]

    idx = pd.to_datetime(returns.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    returns.index = idx.normalize()
    returns.index.name = "Date"
    return returns.dropna(how="all")
