"""Tests for macro features (graceful degradation)."""

import pandas as pd

from pyquant.data import macro


def _fake_vix_ticker(sample_index):
    class FakeTicker:
        def __init__(self, symbol):
            pass

        def history(self, period=None, start=None, end=None):
            return pd.DataFrame({"Close": range(len(sample_index))}, index=sample_index)

    return FakeTicker


def test_fetch_macro_without_key_returns_vix_only(monkeypatch, sample_ohlcv_df):
    monkeypatch.setattr(macro.yf, "Ticker", _fake_vix_ticker(sample_ohlcv_df.index))
    out = macro.fetch_macro(api_key=None)
    assert "VIX" in out.columns
    assert "FedFunds" not in out.columns


def test_fetch_macro_handles_vix_failure(monkeypatch):
    class Boom:
        def __init__(self, symbol):
            raise RuntimeError("network down")

    monkeypatch.setattr(macro.yf, "Ticker", Boom)
    out = macro.fetch_macro(api_key=None)
    assert out.empty


def test_fetch_macro_with_key_adds_fred(monkeypatch, sample_ohlcv_df):
    monkeypatch.setattr(macro.yf, "Ticker", _fake_vix_ticker(sample_ohlcv_df.index))

    class FakeFred:
        def __init__(self, api_key=None):
            pass

        def get_series(self, series_id, observation_start=None, observation_end=None):
            return pd.Series(
                range(len(sample_ohlcv_df.index)), index=sample_ohlcv_df.index
            )

    import fredapi

    monkeypatch.setattr(fredapi, "Fred", FakeFred)
    out = macro.fetch_macro(api_key="dummy")
    assert "VIX" in out.columns
    assert "FedFunds" in out.columns
    assert "YieldSpread" in out.columns


def test_fetch_macro_keeps_series_that_succeed_when_one_fails(monkeypatch, sample_ohlcv_df):
    """One failing FRED series must not discard the ones already fetched (PYQ-110)."""
    monkeypatch.setattr(macro.yf, "Ticker", _fake_vix_ticker(sample_ohlcv_df.index))

    class FlakyFred:
        def __init__(self, api_key=None):
            pass

        def get_series(self, series_id, observation_start=None, observation_end=None):
            if series_id == "CPIAUCSL":
                raise RuntimeError("transient FRED rate limit")
            return pd.Series(range(len(sample_ohlcv_df.index)), index=sample_ohlcv_df.index)

    import fredapi

    monkeypatch.setattr(fredapi, "Fred", FlakyFred)
    out = macro.fetch_macro(api_key="dummy")
    # DFF/T10Y2Y succeeded before CPIAUCSL failed -- they must survive.
    assert "FedFunds" in out.columns
    assert "YieldSpread" in out.columns
    assert "CPI" not in out.columns


def test_fetch_macro_lags_monthly_cpi_by_publication_delay(monkeypatch, sample_ohlcv_df):
    """CPIAUCSL is indexed by BLS's reference period, not its publish date.

    A row must not reveal the value before it was actually released.
    """
    monkeypatch.setattr(macro.yf, "Ticker", _fake_vix_ticker(sample_ohlcv_df.index))
    reference_date = pd.Timestamp("2022-06-01")  # June's CPI, dated at month start

    class FakeFred:
        def __init__(self, api_key=None):
            pass

        def get_series(self, series_id, observation_start=None, observation_end=None):
            return pd.Series([111.0], index=[reference_date])

    import fredapi

    monkeypatch.setattr(fredapi, "Fred", FakeFred)
    out = macro.fetch_macro(api_key="dummy", start="2022-01-01", end="2022-12-31")

    lag_days = macro.FRED_SERIES["CPIAUCSL"].publication_lag_days
    assert lag_days > 14  # sanity: this is meant to model a multi-week real lag

    published_date = reference_date + pd.Timedelta(days=lag_days)
    # Not yet known on the reference date itself -- the pre-fix bug exposed it here.
    assert pd.isna(out.loc[reference_date, "CPI"])
    # Known once its real publication date has passed.
    assert out.loc[published_date, "CPI"] == 111.0
