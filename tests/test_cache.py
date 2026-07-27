"""Tests for the local panel cache: TTL expiry + named pins."""

import json
from pathlib import Path

import pandas as pd

from pyquant import provenance
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


# --- PYQ-133: a pin must not outlive the code that defined its columns -------


def test_pin_metadata_records_the_version_and_the_column_list(tmp_path):
    """A pin is TTL-exempt and permanent; nothing recorded what computed it."""
    df = pd.DataFrame(
        {"Close": [1.0, 2.0], "RSI_14": [40.0, 60.0]},
        index=pd.bdate_range("2024-01-01", periods=2),
    )
    cache.write_pin(tmp_path, "experiment-1", df)

    meta = cache.read_pin_metadata(tmp_path, "experiment-1")

    assert meta["columns"] == ["Close", "RSI_14"]
    assert meta["pyquant_version"] == provenance.package_version()
    assert meta["n_rows"] == 2
    assert "created_at" in meta


def test_read_pin_warns_when_the_recorded_version_differs(tmp_path, caplog):
    """PYQ-121 redefined RSI_14. A pin created before it replays the *old*
    definition forever, into a bundle whose provenance records the new sha --
    a reproducibility claim that is false and undetectably so."""
    cache.write_pin(tmp_path, "experiment-1", _df())
    meta_path = tmp_path / "pins" / "experiment-1.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["pyquant_version"] = "0.0.1-ancient"
    meta_path.write_text(json.dumps(meta))

    with caplog.at_level("WARNING"):
        out = cache.read_pin(tmp_path, "experiment-1")

    assert out is not None  # still usable -- warned about, not refused
    assert any("0.0.1-ancient" in m for m in caplog.messages)
    assert any("experiment-1" in m for m in caplog.messages)


def test_read_pin_warns_when_the_recorded_columns_differ(tmp_path, caplog):
    """The cheap high-value half: a renamed or added column is caught exactly,
    even though a silently *redefined* one cannot be."""
    cache.write_pin(tmp_path, "experiment-1", _df())
    meta_path = tmp_path / "pins" / "experiment-1.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["columns"] = ["Close", "SomeColumnThatNoLongerExists"]
    meta_path.write_text(json.dumps(meta))

    with caplog.at_level("WARNING"):
        cache.read_pin(tmp_path, "experiment-1")

    assert any("SomeColumnThatNoLongerExists" in m for m in caplog.messages)


def test_read_pin_is_silent_when_version_and_columns_match(tmp_path, caplog):
    cache.write_pin(tmp_path, "experiment-1", _df())
    with caplog.at_level("WARNING"):
        cache.read_pin(tmp_path, "experiment-1")
    assert not caplog.messages


def test_read_pin_warns_for_a_pin_written_before_metadata_existed(tmp_path, caplog):
    """Pins on disk from before this ticket have no sibling metadata at all."""
    cache.write_pin(tmp_path, "experiment-1", _df())
    (tmp_path / "pins" / "experiment-1.meta.json").unlink()

    with caplog.at_level("WARNING"):
        out = cache.read_pin(tmp_path, "experiment-1")

    assert out is not None
    assert any("no recorded metadata" in m.lower() for m in caplog.messages)


def test_remove_pin_also_removes_its_metadata(tmp_path):
    cache.write_pin(tmp_path, "experiment-1", _df())
    assert cache.remove_pin(tmp_path, "experiment-1") is True
    assert not (tmp_path / "pins" / "experiment-1.meta.json").exists()


def test_remove_pin(tmp_path):
    df = _df()
    cache.write_pin(tmp_path, "p1", df)
    assert cache.remove_pin(tmp_path, "p1") is True
    assert cache.remove_pin(tmp_path, "p1") is False


def test_git_sha_returns_none_when_the_package_lives_in_an_unrelated_repo(tmp_path, monkeypatch):
    """A pip-installed package can sit inside *some other* git repo (a vendored
    dependency tree, a conda env under version control, a monorepo with a
    committed venv). Recording that repo's sha as PyQuant's provenance is
    silently wrong, which is worse than absent (PYQ-134)."""
    import subprocess

    unrelated = tmp_path / "unrelated"
    (unrelated / "site-packages" / "pyquant").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "."], cwd=unrelated, check=True)
    subprocess.run(
        ["git", "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-q",
         "--allow-empty", "-m", "unrelated"],
        cwd=unrelated, check=True,
    )
    foreign_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=unrelated,
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    monkeypatch.setattr(
        provenance, "__file__", str(unrelated / "site-packages" / "pyquant" / "provenance.py")
    )

    sha = provenance.git_sha()
    assert sha != foreign_sha
    assert sha is None


def test_git_sha_still_reports_the_sha_from_a_real_source_checkout():
    """The fix must not break the case that actually works today."""
    import subprocess

    expected = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=Path(provenance.__file__).resolve().parent,
        capture_output=True, text=True,
    )
    if expected.returncode != 0:  # not a checkout (installed wheel) -- nothing to assert
        return
    assert provenance.git_sha() == expected.stdout.strip()
