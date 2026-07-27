"""Pluggable price providers must be genuinely substitutable (PYQ-258)."""

import numpy as np
import pandas as pd
import pytest

from pyquant.data import prices
from pyquant.data.providers import (
    OHLCV_COLUMNS,
    PriceProviderError,
    TiingoProvider,
    YFinanceProvider,
    assert_ohlcv_contract,
    get_provider,
    normalize_ohlcv,
)


def _yahoo_shaped_frame():
    """What yfinance hands back: tz-aware index, extra columns, mixed dtypes."""
    idx = pd.date_range("2024-01-02", periods=5, freq="B", tz="America/New_York")
    return pd.DataFrame(
        {
            "Open": [10.0, 11, 12, 13, 14],
            "High": [11.0, 12, 13, 14, 15],
            "Low": [9.0, 10, 11, 12, 13],
            "Close": [10.5, 11.5, 12.5, 13.5, 14.5],
            "Volume": [100, 200, 300, 400, 500],  # int, not float
            "Dividends": [0.0] * 5,  # extra column the contract must drop
            "Stock Splits": [0.0] * 5,
        },
        index=idx,
    )


def _tiingo_shaped_payload():
    """What Tiingo's JSON REST endpoint hands back: ISO strings, adj* names."""
    return [
        {
            "date": f"2024-01-0{d}T00:00:00.000Z",
            "adjOpen": 10.0 + d,
            "adjHigh": 11.0 + d,
            "adjLow": 9.0 + d,
            "adjClose": 10.5 + d,
            "adjVolume": 100 * d,
        }
        for d in (5, 4, 3, 2)  # deliberately out of order
    ]


class _FakeTiingoSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        payload = self.payload

        class _Response:
            def raise_for_status(self):
                return None

            def json(self):
                return payload

        return _Response()


def test_both_providers_return_the_identical_column_schema_and_dtypes(monkeypatch):
    """The property that makes switching vendors a config change rather than a
    rewrite: both must satisfy one contract, checked against one statement."""

    class FakeTicker:
        def __init__(self, symbol):
            pass

        def history(self, **kwargs):
            return _yahoo_shaped_frame()

    import yfinance as yf

    monkeypatch.setattr(yf, "Ticker", FakeTicker)

    yahoo = YFinanceProvider().fetch_ohlcv("AAPL")
    tiingo = TiingoProvider(
        api_key="dummy", session=_FakeTiingoSession(_tiingo_shaped_payload())
    ).fetch_ohlcv("AAPL")

    assert_ohlcv_contract(yahoo)
    assert_ohlcv_contract(tiingo)
    assert list(yahoo.columns) == list(tiingo.columns) == OHLCV_COLUMNS
    assert yahoo.dtypes.to_dict() == tiingo.dtypes.to_dict()
    assert yahoo.index.tz is None and tiingo.index.tz is None
    assert yahoo.index.name == tiingo.index.name == "Date"


def test_normalization_drops_vendor_extras_and_sorts_ascending():
    """yfinance ships Dividends/Stock Splits and Tiingo ships newest-first; both
    would misalign a join if they reached the panel."""
    normalized = normalize_ohlcv(_yahoo_shaped_frame())

    assert "Dividends" not in normalized.columns
    assert "Stock Splits" not in normalized.columns
    assert normalized.index.is_monotonic_increasing
    assert str(normalized["Volume"].dtype) == "float64"  # int -> float


def test_tiingo_orders_its_response_and_requests_adjusted_fields():
    """The adjustment convention is chosen here, not inherited from a default --
    the PYQ-228 failure mode. Tiingo serves both raw and adjusted."""
    session = _FakeTiingoSession(_tiingo_shaped_payload())

    frame = TiingoProvider(api_key="k", session=session).fetch_ohlcv("AAPL", period="1y")

    assert frame.index.is_monotonic_increasing
    assert frame.index[0] == pd.Timestamp("2024-01-02")
    # adjClose for 2024-01-02 is 10.5 + 2
    assert frame["Close"].iloc[0] == pytest.approx(12.5)
    assert session.calls[0]["headers"]["Authorization"] == "Token k"
    assert session.calls[0]["params"]["startDate"]  # period honoured as a date range


def test_the_contract_rejects_a_frame_that_would_misalign_a_join():
    """assert_ohlcv_contract has to actually reject things, or it documents
    nothing. Each of these is a real way a new vendor differs."""
    good = normalize_ohlcv(_yahoo_shaped_frame())

    tz_aware = good.copy()
    tz_aware.index = tz_aware.index.tz_localize("UTC")
    with pytest.raises(PriceProviderError, match="tz-naive"):
        assert_ohlcv_contract(tz_aware)

    descending = good.iloc[::-1]
    with pytest.raises(PriceProviderError, match="ascending"):
        assert_ohlcv_contract(descending)

    renamed = good.rename(columns={"Close": "close"})
    with pytest.raises(PriceProviderError, match="columns must be exactly"):
        assert_ohlcv_contract(renamed)

    as_int = good.copy()
    as_int["Volume"] = as_int["Volume"].astype(int)
    with pytest.raises(PriceProviderError, match="float64"):
        assert_ohlcv_contract(as_int)


def test_a_missing_required_column_names_itself():
    with pytest.raises(PriceProviderError, match=r"missing column\(s\).*Volume"):
        normalize_ohlcv(_yahoo_shaped_frame().drop(columns=["Volume"]))


def test_tiingo_without_a_key_says_so_instead_of_failing_obscurely():
    with pytest.raises(PriceProviderError, match="TIINGO_API_KEY"):
        TiingoProvider(api_key=None).fetch_ohlcv("AAPL")


def test_get_provider_resolves_by_name_and_rejects_unknown_ones():
    assert isinstance(get_provider("yfinance"), YFinanceProvider)
    assert isinstance(get_provider("Tiingo", api_key="k"), TiingoProvider)
    with pytest.raises(PriceProviderError, match="Unknown price provider"):
        get_provider("bloomberg")


def test_fetch_prices_accepts_an_arbitrary_provider_object(monkeypatch):
    """The interface, not the vendor, is the deliverable: anything satisfying
    the protocol can back the whole pipeline."""

    class StubProvider:
        name = "stub"

        def fetch_ohlcv(self, symbol, *, period="5y", start=None, end=None):
            n = 300
            idx = pd.bdate_range("2023-01-02", periods=n, name="Date")
            close = 100 + np.arange(n, dtype=float)
            return pd.DataFrame(
                {
                    "Open": close,
                    "High": close + 1,
                    "Low": close - 1,
                    "Close": close,
                    "Volume": np.full(n, 1e6),
                },
                index=idx,
            )

    out = prices.fetch_prices("ANY", provider=StubProvider(), use_indicators=True)

    assert "RSI_14" in out.columns  # the indicator pipeline ran on it
    assert out.index.name == "Date"


def test_fetch_prices_rejects_a_provider_that_breaks_the_contract():
    """A new vendor's subtly different frame must fail loudly at the boundary,
    not misalign a join three modules downstream."""

    class BadProvider:
        name = "bad"

        def fetch_ohlcv(self, symbol, *, period="5y", start=None, end=None):
            idx = pd.date_range("2024-01-01", periods=3, tz="UTC", name="Date")
            return pd.DataFrame(
                {c: [1.0, 2.0, 3.0] for c in OHLCV_COLUMNS}, index=idx
            )

    with pytest.raises(PriceProviderError):
        prices.fetch_prices("ANY", provider=BadProvider(), use_indicators=False)
