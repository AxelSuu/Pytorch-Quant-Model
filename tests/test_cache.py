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


def test_prune_expired_removes_only_stale_entries_and_keeps_pins(tmp_path):
    """`cache prune` must remove expired entries without touching valid ones or
    any pins (PYQ-221)."""
    df = _df()
    cache.write_cache(tmp_path, "stale", df, now=0.0)
    cache.write_cache(tmp_path, "fresh", df, now=10000.0)
    cache.write_pin(tmp_path, "keeper", df)

    removed = cache.prune_expired(tmp_path, ttl_seconds=3600, now=10000.0)

    assert removed == ["stale"]
    assert not (tmp_path / "stale.pkl").exists()
    assert not (tmp_path / "stale.meta.json").exists()
    assert cache.read_cache(tmp_path, "fresh", ttl_seconds=3600, now=10000.0) is not None
    assert cache.read_pin(tmp_path, "keeper") is not None  # pin untouched
    assert cache.list_pins(tmp_path) == ["keeper"]


def test_cache_stats_counts_entries_and_pins(tmp_path):
    df = _df()
    cache.write_cache(tmp_path, "e1", df, now=1.0)
    cache.write_pin(tmp_path, "p1", df)
    stats = cache.cache_stats(tmp_path)
    assert stats["entry_count"] == 1  # pin not counted as a TTL entry
    assert stats["total_bytes"] > 0
    assert stats["pins"] == ["p1"]


def test_remove_pin(tmp_path):
    df = _df()
    cache.write_pin(tmp_path, "p1", df)
    assert cache.remove_pin(tmp_path, "p1") is True
    assert cache.remove_pin(tmp_path, "p1") is False
