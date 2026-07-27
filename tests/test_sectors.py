"""Tests for sector ETF features."""

from pathlib import Path

import pandas as pd

from pyquant.data import sectors

FIXTURES = Path(__file__).parent / "fixtures"


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


def test_fetch_sector_returns_parses_a_real_recorded_yf_download_payload(monkeypatch):
    """PYQ-243: drives the real fetch_sector_returns() parsing (the (field, ticker)
    MultiIndex column selection, the SEC_ rename, the tz/dropna cleanup) from one
    real recorded yf.download() response instead of a hand-built MultiIndex frame.
    """
    real_response = pd.read_pickle(FIXTURES / "yfinance_sectors.pkl")
    assert isinstance(real_response.columns, pd.MultiIndex)
    assert "Close" in real_response.columns.get_level_values(0)

    monkeypatch.setattr(sectors.yf, "download", lambda *a, **k: real_response)

    out = sectors.fetch_sector_returns(["XLK", "SPY"])

    assert list(out.columns) == ["SEC_SPY", "SEC_XLK"] or list(out.columns) == ["SEC_XLK", "SEC_SPY"]
    assert out.index.tz is None
    assert out.index.name == "Date"
    assert not out.empty
