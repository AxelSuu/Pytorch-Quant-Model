"""Tests for sector ETF features."""

import pandas as pd

from pyquant.data import sectors


def test_fetch_sector_returns_empty_for_no_etfs():
    assert sectors.fetch_sector_returns([]).empty


def test_fetch_sector_returns_builds_named_columns(monkeypatch, sample_ohlcv_df):
    idx = sample_ohlcv_df.index
    close = pd.DataFrame(
        {("Close", "XLK"): range(1, len(idx) + 1), ("Close", "SPY"): range(2, len(idx) + 2)},
        index=idx,
    )
    close.columns = pd.MultiIndex.from_tuples(close.columns)

    monkeypatch.setattr(sectors.yf, "download", lambda *a, **k: close)
    out = sectors.fetch_sector_returns(["XLK", "SPY"])
    assert "SEC_XLK" in out.columns
    assert "SEC_SPY" in out.columns
    assert out.index.tz is None


def test_fetch_sector_returns_handles_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("download failed")

    monkeypatch.setattr(sectors.yf, "download", boom)
    assert sectors.fetch_sector_returns(["XLK"]).empty
