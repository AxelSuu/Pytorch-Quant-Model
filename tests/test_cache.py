"""Tests for the local panel cache: TTL expiry + named pins."""

import pandas as pd

from pyquant.data import cache


def _df():
    return pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=pd.bdate_range("2024-01-01", periods=3))


def test_fingerprint_key_is_order_independent():
    a = cache.fingerprint_key({"symbol": "AAPL", "period": "5y"})
    b = cache.fingerprint_key({"period": "5y", "symbol": "AAPL"})
    assert a == b


def test_fingerprint_key_differs_for_different_inputs():
    a = cache.fingerprint_key({"symbol": "AAPL"})
    b = cache.fingerprint_key({"symbol": "MSFT"})
    assert a != b


def test_read_cache_returns_none_when_missing(tmp_path):
    assert cache.read_cache(tmp_path, "nope", ttl_seconds=3600) is None


def test_write_then_read_cache_round_trips(tmp_path):
    df = _df()
    cache.write_cache(tmp_path, "key1", df, now=1000.0)
    out = cache.read_cache(tmp_path, "key1", ttl_seconds=3600, now=1000.0)
    pd.testing.assert_frame_equal(out, df)


def test_read_cache_returns_none_once_ttl_expires(tmp_path):
    df = _df()
    cache.write_cache(tmp_path, "key1", df, now=1000.0)
    # 3601s later, past a 3600s TTL.
    assert cache.read_cache(tmp_path, "key1", ttl_seconds=3600, now=1000.0 + 3601) is None


def test_read_cache_still_valid_just_within_ttl(tmp_path):
    df = _df()
    cache.write_cache(tmp_path, "key1", df, now=1000.0)
    out = cache.read_cache(tmp_path, "key1", ttl_seconds=3600, now=1000.0 + 3599)
    pd.testing.assert_frame_equal(out, df)


def test_write_then_read_pin_round_trips(tmp_path):
    df = _df()
    cache.write_pin(tmp_path, "experiment-1", df)
    out = cache.read_pin(tmp_path, "experiment-1")
    pd.testing.assert_frame_equal(out, df)


def test_read_pin_returns_none_when_missing(tmp_path):
    assert cache.read_pin(tmp_path, "nope") is None
