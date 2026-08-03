"""Tests for macro features (graceful degradation)."""

import json
from pathlib import Path

import pandas as pd

from pyquant.data import macro

FIXTURES = Path(__file__).parent / "fixtures"


def _fake_vix_ticker(sample_index):
    class FakeTicker:
        def __init__(self, symbol):
            pass

        def history(self, period=None, start=None, end=None, auto_adjust=None, **kwargs):
            return pd.DataFrame({"Close": range(len(sample_index))}, index=sample_index)

    return FakeTicker


def _releases(index, values=None):
    """One point-in-time release per synthetic observation."""
    values = values if values is not None else range(len(index))
    return pd.DataFrame({"date": index, "realtime_start": index, "value": values})


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

        def get_series_all_releases(self, series_id, realtime_start=None, realtime_end=None):
            return _releases(sample_ohlcv_df.index)

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

        def get_series_all_releases(self, series_id, realtime_start=None, realtime_end=None):
            if series_id == "CPIAUCSL":
                raise RuntimeError("transient FRED rate limit")
            return _releases(sample_ohlcv_df.index)

    import fredapi

    monkeypatch.setattr(fredapi, "Fred", FlakyFred)
    out = macro.fetch_macro(api_key="dummy")
    # DFF/T10Y2Y succeeded before CPIAUCSL failed -- they must survive.
    assert "FedFunds" in out.columns
    assert "YieldSpread" in out.columns
    assert "CPI" not in out.columns


def test_fetch_macro_uses_the_first_published_cpi_vintage(monkeypatch, sample_ohlcv_df):
    """A historical row sees CPI only after its actual first release, not a revision."""
    monkeypatch.setattr(macro.yf, "Ticker", _fake_vix_ticker(sample_ohlcv_df.index))
    reference_date = pd.Timestamp("2022-06-01")  # June's CPI, dated at month start

    class FakeFred:
        def __init__(self, api_key=None):
            pass

        def get_series_all_releases(self, series_id, realtime_start=None, realtime_end=None):
            return pd.DataFrame(
                {
                    "date": [reference_date, reference_date],
                    "realtime_start": ["2022-06-14", "2022-07-14"],
                    "value": [111.0, 999.0],
                }
            )

    import fredapi

    monkeypatch.setattr(fredapi, "Fred", FakeFred)
    out = macro.fetch_macro(api_key="dummy", start="2022-01-01", end="2022-12-31")

    published_date = pd.Timestamp("2022-06-14")
    # Not known on its reference date -- the pre-fix bug exposed it here.
    assert pd.isna(out.loc[reference_date, "CPI"])
    # Known at the first-published value on the actual release date.
    assert out.loc[published_date, "CPI"] == 111.0
    # A later revision only appears from its own release date onwards.
    assert out.loc[pd.Timestamp("2022-07-14"), "CPI"] == 999.0


# --- PYQ-139: the vintage fetch has to survive the real API -------------------
#
# PYQ-273 asked for these three as named regression tests against hand-built
# fixtures, on the (corrected, see that ticket's resolution note) premise that
# none existed yet. They already did, as of this same pass -- and as inline
# hand-built frames rather than a file under tests/fixtures/, which is the
# right shape for this specific boundary: `fredapi.Fred` hands back a
# `pandas.DataFrame` already parsed from FRED's XML, not raw bytes, so a
# `RecordingFred`/`FakeFred` class returning a DataFrame *is* the fixture --
# see scripts/record_fixtures.py's module docstring for the fuller reasoning.


def test_missing_observations_do_not_abort_a_whole_series(monkeypatch, sample_ohlcv_df):
    """FRED encodes a missing observation as ".", which fredapi hands back as
    NaT -- not NaN. ``float(NaT)`` raises, so one market holiday in T10Y2Y took
    the entire series down, and graceful degradation then hid it (PYQ-139).

    Reproduced against the live API: T10Y2Y returned 551 such rows in a 5-year
    window and CPIAUCSL one.
    """
    monkeypatch.setattr(macro.yf, "Ticker", _fake_vix_ticker(sample_ohlcv_df.index))
    index = sample_ohlcv_df.index[:10]
    values = [1.0, pd.NaT, 3.0, 4.0, pd.NaT, 6.0, 7.0, 8.0, 9.0, 10.0]

    class FakeFred:
        def __init__(self, api_key=None):
            pass

        def get_series_all_releases(self, series_id, realtime_start=None, realtime_end=None):
            return pd.DataFrame({"date": index, "realtime_start": index, "value": values})

    import fredapi

    monkeypatch.setattr(fredapi, "Fred", FakeFred)
    out = macro.fetch_macro(api_key="dummy")

    assert "YieldSpread" in out.columns
    assert out["YieldSpread"].notna().any()
    # The gaps carry the previous known value forward, never NaT, never a crash.
    assert out["YieldSpread"].dropna().map(lambda v: isinstance(v, float)).all()


def test_vintage_requests_are_bounded_and_never_ask_for_a_future_realtime_end(
    monkeypatch, sample_ohlcv_df
):
    """Two live failures in one assertion (PYQ-139).

    With ``period="5y"`` and no explicit start/end, the realtime window was left
    unset, so fredapi defaulted to 1776-07-04..9999-12-31. FRED rejects that two
    ways: "3085 vintage dates ... exceeds the maximum 2000" for a daily series,
    and "realtime_end can not be after today's date".
    """
    monkeypatch.setattr(macro.yf, "Ticker", _fake_vix_ticker(sample_ohlcv_df.index))
    seen: list[tuple[str, str]] = []
    # FRED's clock, not ours: a caller east of the US is on a later calendar day
    # and asking for it is a Bad Request (PYQ-139).
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)

    class RecordingFred:
        def __init__(self, api_key=None):
            pass

        def get_series_all_releases(self, series_id, realtime_start=None, realtime_end=None):
            seen.append((realtime_start, realtime_end))
            return _releases(sample_ohlcv_df.index[:5])

    import fredapi

    monkeypatch.setattr(fredapi, "Fred", RecordingFred)
    macro.fetch_macro(api_key="dummy", period="5y")

    assert seen, "no FRED request was made"
    for realtime_start, realtime_end in seen:
        assert realtime_start is not None and realtime_end is not None
        assert pd.Timestamp(realtime_end) <= today
        assert pd.Timestamp(realtime_start) <= pd.Timestamp(realtime_end)
        # Chunked so no single request can exceed FRED's 2000-vintage ceiling.
        span_days = (pd.Timestamp(realtime_end) - pd.Timestamp(realtime_start)).days
        assert span_days <= 400, (
            f"chunk spans {span_days} days; a daily series would exceed 2000 vintages"
        )


def test_a_ten_year_request_is_split_into_chunks_that_cover_the_whole_window(
    monkeypatch, sample_ohlcv_df
):
    """Chunking must tile the requested window without leaving holes, or the
    feature silently starts late."""
    monkeypatch.setattr(macro.yf, "Ticker", _fake_vix_ticker(sample_ohlcv_df.index))
    seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    class RecordingFred:
        def __init__(self, api_key=None):
            pass

        def get_series_all_releases(self, series_id, realtime_start=None, realtime_end=None):
            seen.append((pd.Timestamp(realtime_start), pd.Timestamp(realtime_end)))
            return _releases(sample_ohlcv_df.index[:3])

    import fredapi

    monkeypatch.setattr(fredapi, "Fred", RecordingFred)
    macro.fetch_macro(api_key="dummy", start="2016-01-01", end="2026-01-01")

    dff_chunks = sorted(seen[: len(seen) // len(macro.FRED_SERIES)])
    assert len(dff_chunks) > 1
    assert dff_chunks[0][0] == pd.Timestamp("2016-01-01")
    for (_, prev_end), (next_start, _) in zip(dff_chunks, dff_chunks[1:], strict=False):
        assert next_start <= prev_end + pd.Timedelta(days=1), "gap between vintage chunks"


def test_fetch_macro_parses_real_recorded_vix_and_fred_payloads(monkeypatch):
    """PYQ-243: every other test in this file hand-builds its VIX/FRED frames, which
    verifies our parsing against our own assumptions rather than the vendors'
    actual shapes. fredapi in particular talks to FRED over urllib, not `requests`
    (checked directly), so `_fetch_fred`/`_vintage_series` is the real boundary
    worth pinning here -- mocking raw sockets would only test fredapi's own XML
    parsing, not ours.
    """
    real_vix = pd.read_pickle(FIXTURES / "yfinance_vix.pkl")
    assert "Dividends" in real_vix.columns  # a real Ticker.history() response shape

    class RecordedVixTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            return real_vix

    real_releases = pd.DataFrame(json.loads((FIXTURES / "fred_dff.json").read_text()))
    assert set(real_releases.columns) == {"realtime_start", "date", "value"}

    class RecordedFred:
        def __init__(self, api_key=None):
            pass

        def get_series_all_releases(self, series_id, realtime_start=None, realtime_end=None):
            return real_releases

    import fredapi

    monkeypatch.setattr(macro.yf, "Ticker", RecordedVixTicker)
    monkeypatch.setattr(fredapi, "Fred", RecordedFred)

    out = macro.fetch_macro(api_key="dummy")

    assert "VIX" in out.columns
    assert out["VIX"].notna().any()
    assert "FedFunds" in out.columns  # DFF -> FedFunds, from the real releases
    assert out.index.tz is None
