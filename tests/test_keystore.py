"""Tests for the PYQ-281 SQLite API-key store (network-free, no API extra needed)."""

import pytest

from pyquant.api import keystore


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "api_keys.db"


def test_create_key_returns_a_raw_value_matching_the_documented_format(db_path):
    raw_key, record = keystore.create_key(db_path, "ci-bot", ["read"])
    assert raw_key.startswith(keystore.KEY_PREFIX)
    assert record.prefix == raw_key[: keystore.PREFIX_LEN]
    assert record.name == "ci-bot"
    assert record.scopes == frozenset({"read"})
    assert record.revoked_at is None
    assert record.last_used_at is None


def test_two_created_keys_have_different_raw_values_and_ids(db_path):
    raw_a, record_a = keystore.create_key(db_path, "a", ["read"])
    raw_b, record_b = keystore.create_key(db_path, "b", ["read"])
    assert raw_a != raw_b
    assert record_a.id != record_b.id


def test_create_key_rejects_an_unknown_scope(db_path):
    with pytest.raises(keystore.InvalidScope):
        keystore.create_key(db_path, "bad", ["admin"])


def test_create_key_rejects_no_scopes(db_path):
    with pytest.raises(keystore.InvalidScope):
        keystore.create_key(db_path, "no-scopes", [])


def test_authenticate_resolves_a_freshly_created_key_to_its_identity(db_path):
    raw_key, record = keystore.create_key(db_path, "ci-bot", ["read", "train"])
    identity = keystore.authenticate(db_path, raw_key)
    assert identity is not None
    assert identity.id == record.id
    assert identity.name == "ci-bot"
    assert identity.scopes == frozenset({"read", "train"})


def test_authenticate_rejects_an_unknown_key(db_path):
    keystore.create_key(db_path, "ci-bot", ["read"])
    assert keystore.authenticate(db_path, "pq_live_" + "0" * 24) is None


def test_authenticate_rejects_a_key_from_a_store_that_does_not_exist_yet(tmp_path):
    assert keystore.authenticate(tmp_path / "never-created.db", "pq_live_anything") is None


def test_authenticate_stamps_last_used_at_on_success(db_path):
    raw_key, record = keystore.create_key(db_path, "ci-bot", ["read"])
    assert keystore.list_keys(db_path)[0].last_used_at is None
    keystore.authenticate(db_path, raw_key)
    assert keystore.list_keys(db_path)[0].last_used_at is not None


def test_revoke_key_makes_authenticate_reject_it(db_path):
    raw_key, record = keystore.create_key(db_path, "ci-bot", ["read"])
    assert keystore.revoke_key(db_path, record.id) is True
    assert keystore.authenticate(db_path, raw_key) is None


def test_revoke_key_is_not_idempotently_true(db_path):
    """A second revoke of an already-revoked key reports False (nothing changed),
    not True -- so a caller can distinguish "just revoked it" from "already was"."""
    _raw_key, record = keystore.create_key(db_path, "ci-bot", ["read"])
    assert keystore.revoke_key(db_path, record.id) is True
    assert keystore.revoke_key(db_path, record.id) is False


def test_revoke_key_on_an_unknown_id_returns_false(db_path):
    assert keystore.revoke_key(db_path, "does-not-exist") is False


def test_has_active_keys_is_false_for_a_store_that_does_not_exist_yet(tmp_path):
    assert keystore.has_active_keys(tmp_path / "never-created.db") is False


def test_has_active_keys_is_false_once_the_only_key_is_revoked(db_path):
    _raw_key, record = keystore.create_key(db_path, "ci-bot", ["read"])
    assert keystore.has_active_keys(db_path) is True
    keystore.revoke_key(db_path, record.id)
    assert keystore.has_active_keys(db_path) is False


def test_list_keys_never_exposes_the_raw_key_or_its_hash(db_path):
    keystore.create_key(db_path, "ci-bot", ["read"])
    records = keystore.list_keys(db_path)
    assert len(records) == 1
    dumped = str(records[0])
    assert "key_hash" not in dumped


def test_list_keys_orders_newest_first(db_path):
    keystore.create_key(db_path, "first", ["read"])
    keystore.create_key(db_path, "second", ["read"])
    names = [r.name for r in keystore.list_keys(db_path)]
    assert names[0] == "second"


def test_resolve_db_path_honors_the_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("PYQUANT_API_KEYS_DB", str(tmp_path / "custom.db"))
    assert keystore.resolve_db_path() == (tmp_path / "custom.db").resolve()


def test_resolve_db_path_defaults_under_project_root(monkeypatch):
    monkeypatch.delenv("PYQUANT_API_KEYS_DB", raising=False)
    from pyquant.config import project_root

    assert keystore.resolve_db_path() == (project_root() / "data" / "api_keys.db").resolve()
