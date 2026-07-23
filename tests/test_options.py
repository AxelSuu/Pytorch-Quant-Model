"""Tests for options-implied market context: nearest-strike indexing, degradation, labels."""

from dataclasses import dataclass

import pandas as pd
import pytest

from pyquant.data import options


@dataclass
class _FakeChain:
    calls: pd.DataFrame
    puts: pd.DataFrame


def _chain_df(strikes, volume, iv):
    return pd.DataFrame({"strike": strikes, "volume": volume, "impliedVolatility": iv})


class _FakeFastInfo(dict):
    """Mimics yfinance's fast_info: dict-like with a .get()."""


def _fake_ticker(
    expiries, chain, spot=100.0, fast_info_raises=False, history_df=None, history_raises=False
):
    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol
            self.options = expiries

        @property
        def fast_info(self):
            if fast_info_raises:
                raise RuntimeError("fast_info unavailable")
            return _FakeFastInfo(last_price=spot)

        def option_chain(self, expiry):
            return chain

        def history(self, period=None):
            if history_raises:
                raise RuntimeError("history unavailable")
            return history_df if history_df is not None else pd.DataFrame()

    return FakeTicker


def test_fetch_options_snapshot_normal_case(monkeypatch):
    calls = _chain_df([90, 95, 100, 105, 110], [10, 20, 30, 15, 5], [0.30, 0.28, 0.25, 0.27, 0.29])
    puts = _chain_df([90, 95, 100, 105, 110], [40, 25, 20, 10, 5], [0.35, 0.31, 0.26, 0.24, 0.22])
    chain = _FakeChain(calls=calls, puts=puts)
    monkeypatch.setattr(options.yf, "Ticker", _fake_ticker(["2024-01-19"], chain, spot=100.0))

    snap = options.fetch_options_snapshot("AAPL")

    assert snap.expiry == "2024-01-19"
    assert abs(snap.put_call_ratio - 1.25) < 1e-9  # 100 put vol / 80 call vol
    assert abs(snap.atm_iv - 0.255) < 1e-9  # mean(call@100=0.25, put@100=0.26)
    assert abs(snap.iv_skew - 0.06) < 1e-9  # put@90=0.35 - call@110=0.29
    assert snap.sentiment_label == "bearish (heavy puts)"


def test_fetch_options_snapshot_no_options_listed(monkeypatch):
    monkeypatch.setattr(options.yf, "Ticker", _fake_ticker([], _FakeChain(pd.DataFrame(), pd.DataFrame())))
    snap = options.fetch_options_snapshot("AAPL")
    assert snap == options.OptionsSnapshot(None, None, None, None)


def test_fetch_options_snapshot_empty_chain(monkeypatch):
    empty_chain = _FakeChain(calls=pd.DataFrame(), puts=pd.DataFrame())
    monkeypatch.setattr(options.yf, "Ticker", _fake_ticker(["2024-01-19"], empty_chain))
    snap = options.fetch_options_snapshot("AAPL")
    assert snap.expiry == "2024-01-19"
    assert snap.put_call_ratio is None
    assert snap.atm_iv is None
    assert snap.iv_skew is None


def test_fetch_options_snapshot_missing_spot_price(monkeypatch):
    calls = _chain_df([100], [10], [0.25])
    puts = _chain_df([100], [10], [0.26])
    chain = _FakeChain(calls=calls, puts=puts)
    monkeypatch.setattr(
        options.yf,
        "Ticker",
        _fake_ticker(["2024-01-19"], chain, fast_info_raises=True, history_df=pd.DataFrame()),
    )
    snap = options.fetch_options_snapshot("AAPL")
    assert snap.expiry == "2024-01-19"
    assert snap.put_call_ratio is None
    assert snap.atm_iv is None
    assert snap.iv_skew is None


def test_fetch_options_snapshot_falls_back_to_history_for_spot(monkeypatch):
    calls = _chain_df([100], [10], [0.25])
    puts = _chain_df([100], [5], [0.26])
    chain = _FakeChain(calls=calls, puts=puts)
    history_df = pd.DataFrame({"Close": [98.0, 99.0, 101.0]})
    monkeypatch.setattr(
        options.yf,
        "Ticker",
        _fake_ticker(["2024-01-19"], chain, fast_info_raises=True, history_df=history_df),
    )
    snap = options.fetch_options_snapshot("AAPL")
    assert snap.put_call_ratio == 0.5  # 5 put vol / 10 call vol


def test_fetch_options_snapshot_handles_history_fallback_exception(monkeypatch):
    calls = _chain_df([100], [10], [0.25])
    puts = _chain_df([100], [5], [0.26])
    chain = _FakeChain(calls=calls, puts=puts)
    monkeypatch.setattr(
        options.yf,
        "Ticker",
        _fake_ticker(["2024-01-19"], chain, fast_info_raises=True, history_raises=True),
    )
    snap = options.fetch_options_snapshot("AAPL")
    assert snap.put_call_ratio is None  # spot price unavailable via either path


def test_fetch_options_snapshot_handles_ticker_exception(monkeypatch):
    class Boom:
        def __init__(self, symbol):
            raise RuntimeError("network down")

    monkeypatch.setattr(options.yf, "Ticker", Boom)
    snap = options.fetch_options_snapshot("AAPL")
    assert snap == options.OptionsSnapshot(None, None, None, None)


@pytest.mark.parametrize(
    "ratio,expected",
    [
        (None, "n/a"),
        (1.2, "neutral"),  # boundary: not > 1.2
        (1.2000001, "bearish (heavy puts)"),
        (0.7, "neutral"),  # boundary: not < 0.7
        (0.6999999, "bullish (heavy calls)"),
        (1.0, "neutral"),
    ],
)
def test_sentiment_label_thresholds(ratio, expected):
    snap = options.OptionsSnapshot(put_call_ratio=ratio, atm_iv=None, iv_skew=None, expiry=None)
    assert snap.sentiment_label == expected
