"""Tests for options-implied market context: nearest-strike indexing, degradation, labels."""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from pyquant.data import options

FIXTURES = Path(__file__).parent / "fixtures"


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

        def history(self, period=None, auto_adjust=None, **kwargs):
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


def test_fetch_options_snapshot_parses_a_real_recorded_chain(monkeypatch):
    """PYQ-243: every other test here hand-builds a 3-column calls/puts frame
    (strike/volume/impliedVolatility). The real chain carries 14 columns
    (contractSymbol, lastTradeDate, bid/ask, inTheMoney, ...) -- this drives the
    real fetch_options_snapshot() parsing from one real recorded chain instead, so
    an unexpected dtype or a renamed/missing column would actually be caught.
    """
    recorded = pd.read_pickle(FIXTURES / "yfinance_options_aapl.pkl")
    assert "impliedVolatility" in recorded["calls"].columns
    assert len(recorded["calls"].columns) > 3  # the real chain, not the 3-column fixture

    chain = _FakeChain(calls=recorded["calls"], puts=recorded["puts"])
    monkeypatch.setattr(
        options.yf,
        "Ticker",
        _fake_ticker([recorded["expiry"]], chain, spot=recorded["fast_last_price"]),
    )

    snap = options.fetch_options_snapshot("AAPL")

    assert snap.expiry == recorded["expiry"]
    assert snap.put_call_ratio is not None and snap.put_call_ratio > 0
    assert snap.atm_iv is not None and snap.atm_iv > 0
    assert snap.iv_skew is not None


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


# --- PYQ-254: accumulated snapshot history --------------------------------------


class _Settings:
    """Minimal stand-in: append_snapshot/load_snapshot_history need only this attr."""

    def __init__(self, tmp_path):
        self.options_history_dir = tmp_path


def test_append_snapshot_writes_one_recorded_row(monkeypatch, tmp_path):
    fake_snap = options.OptionsSnapshot(put_call_ratio=1.1, atm_iv=0.3, iv_skew=0.02, expiry="2024-06-21")
    monkeypatch.setattr(options, "fetch_options_snapshot", lambda symbol: fake_snap)

    path = options.append_snapshot("aapl", _Settings(tmp_path))

    assert path.name == "AAPL.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["put_call_ratio"] == 1.1
    assert rows[0]["atm_iv"] == 0.3
    assert "date" in rows[0] and "observed_at" in rows[0]


def test_load_snapshot_history_is_empty_below_the_minimum_days(monkeypatch, tmp_path):
    """Not enough accumulated history to be a meaningful per-day feature yet --
    empty (with the right columns), not a nearly-all-missing column (PYQ-254)."""
    settings = _Settings(tmp_path)
    path = tmp_path / "AAPL.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"date": f"2024-01-{d:02d}", "observed_at": f"2024-01-{d:02d}T15:00:00", "put_call_ratio": 1.0, "atm_iv": 0.2, "iv_skew": 0.01, "expiry": "2024-06-21"}
        for d in range(1, options.MIN_SNAPSHOT_DAYS)  # one short of the threshold
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    out = options.load_snapshot_history("AAPL", settings)

    assert out.empty
    assert list(out.columns) == options.SNAPSHOT_COLUMNS


def test_load_snapshot_history_returns_data_once_enough_days_accumulate(tmp_path):
    settings = _Settings(tmp_path)
    path = tmp_path / "AAPL.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"date": f"2024-01-{d:02d}", "observed_at": f"2024-01-{d:02d}T15:00:00", "put_call_ratio": 1.0 + d * 0.01, "atm_iv": 0.2, "iv_skew": 0.01, "expiry": "2024-06-21"}
        for d in range(1, options.MIN_SNAPSHOT_DAYS + 1)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    out = options.load_snapshot_history("AAPL", settings)

    assert not out.empty
    assert list(out.columns) == options.SNAPSHOT_COLUMNS
    assert len(out) == options.MIN_SNAPSHOT_DAYS
    assert out.index.is_monotonic_increasing


def test_load_snapshot_history_keeps_the_latest_of_a_repeated_day(tmp_path):
    """Re-running `snapshot` on the same day must not duplicate that day's row."""
    settings = _Settings(tmp_path)
    path = tmp_path / "AAPL.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"date": f"2024-01-{d:02d}", "observed_at": f"2024-01-{d:02d}T10:00:00", "put_call_ratio": 1.0, "atm_iv": 0.2, "iv_skew": 0.01, "expiry": "2024-06-21"}
        for d in range(1, options.MIN_SNAPSHOT_DAYS + 1)
    ]
    # A second, later snapshot on the same first day, with a different value.
    rows.append(
        {"date": "2024-01-01", "observed_at": "2024-01-01T15:30:00", "put_call_ratio": 9.9, "atm_iv": 0.2, "iv_skew": 0.01, "expiry": "2024-06-21"}
    )
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    out = options.load_snapshot_history("AAPL", settings)

    assert out.index.nunique() == options.MIN_SNAPSHOT_DAYS  # not +1
    assert out.loc[pd.Timestamp("2024-01-01"), "OptionsPutCallRatio"] == 9.9  # the later value won
