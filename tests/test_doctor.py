"""Tests for analysis/doctor.py -- environment/bundle health checks (PYQ-263, PYQ-272).

`doctor` exists *because* PYQ-139 was invisible: a whole vendor's features
vanished from every panel and only a log line said so. A diagnostic whose own
failure mode is silence needs direct tests more than most code -- each
unhealthy condition it claims to detect is constructed here and the report is
asserted to actually name it, rather than relying on the CLI-level smoke tests
in test_cli.py to exercise every branch incidentally.
"""

from __future__ import annotations

import json

from pyquant.analysis import doctor
from pyquant.config import Settings


def _settings(tmp_path, **overrides):
    s = Settings()
    s.checkpoint_dir = tmp_path / "checkpoints"
    s.data.cache_dir = tmp_path / "cache"
    s.fred_api_key = None
    s.finnhub_api_key = None
    for key, value in overrides.items():
        target, _, attr = key.rpartition("__")
        setattr(getattr(s, target) if target else s, attr, value)
    return s


def _bundle(checkpoint_dir, name, features, *, data_cfg=None, meta_overrides=None):
    d = checkpoint_dir / name
    d.mkdir(parents=True)
    meta = {
        "symbol": name,
        "symbols": [name],
        "trained_at": "2026-07-27T10:00:00",
        "features": features,
        "target": "close",
        "config": {"data": data_cfg or {}},
        **(meta_overrides or {}),
    }
    (d / "meta.json").write_text(json.dumps(meta))
    return d


def test_bundle_with_no_meta_json_is_unhealthy():
    from pathlib import Path

    health = doctor._bundle_health(Path("/nonexistent/AAPL"), Settings())
    assert health.schema_ok is False
    assert health.problem == "no meta.json"


def test_bundle_with_unreadable_meta_json_is_unhealthy(tmp_path):
    d = tmp_path / "checkpoints" / "AAPL"
    d.mkdir(parents=True)
    (d / "meta.json").write_text("{not valid json")

    health = doctor._bundle_health(d, Settings())

    assert health.schema_ok is False
    assert "unreadable meta.json" in health.problem


def test_bundle_needing_macro_features_is_unhealthy_when_macro_is_disabled(tmp_path):
    settings = _settings(tmp_path, data__use_macro=False)
    d = _bundle(tmp_path / "checkpoints", "AAPL", ["Close", "VIX"])

    health = doctor._bundle_health(d, settings)

    assert health.schema_ok is False
    assert "use_macro" in health.problem


def test_bundle_needing_sector_features_is_unhealthy_when_sectors_are_disabled(tmp_path):
    settings = _settings(tmp_path, data__use_sectors=False)
    d = _bundle(tmp_path / "checkpoints", "AAPL", ["Close", "SEC_XLK"])

    health = doctor._bundle_health(d, settings)

    assert health.schema_ok is False
    assert "use_sectors" in health.problem


def test_bundle_needing_sentiment_features_is_unhealthy_when_sentiment_is_disabled(tmp_path):
    settings = _settings(tmp_path, data__use_sentiment=False)
    d = _bundle(tmp_path / "checkpoints", "AAPL", ["Close", "Sentiment"])

    health = doctor._bundle_health(d, settings)

    assert health.schema_ok is False
    assert "use_sentiment" in health.problem


def test_bundle_needing_fred_features_is_unhealthy_without_a_fred_key(tmp_path):
    settings = _settings(tmp_path, fred_api_key=None)
    d = _bundle(tmp_path / "checkpoints", "AAPL", ["Close", "FedFunds", "CPI"])

    health = doctor._bundle_health(d, settings)

    assert health.schema_ok is False
    assert "FRED_API_KEY" in health.problem


def test_bundle_needing_sentiment_is_unhealthy_without_a_finnhub_key(tmp_path):
    settings = _settings(tmp_path, finnhub_api_key=None)
    d = _bundle(tmp_path / "checkpoints", "AAPL", ["Close", "Sentiment"])

    health = doctor._bundle_health(d, settings)

    assert health.schema_ok is False
    assert "FINNHUB_API_KEY" in health.problem


def test_bundle_needing_sentiment_is_unhealthy_without_the_sentiment_extra(tmp_path, monkeypatch):
    settings = _settings(tmp_path, finnhub_api_key="configured")
    monkeypatch.setattr(doctor, "find_spec", lambda name: None)
    d = _bundle(tmp_path / "checkpoints", "AAPL", ["Close", "Sentiment"])

    health = doctor._bundle_health(d, settings)

    assert health.schema_ok is False
    assert "sentiment" in health.problem


def test_bundle_is_healthy_when_every_needed_feature_source_is_available(tmp_path):
    settings = _settings(tmp_path)
    d = _bundle(tmp_path / "checkpoints", "MSFT", ["Close", "RSI_14", "SMA_50"])

    health = doctor._bundle_health(d, settings)

    assert health.schema_ok is True
    assert health.problem is None
    assert health.n_features == 3
    assert health.symbols == ["MSFT"]


def test_run_doctor_is_unhealthy_if_any_bundle_is_unhealthy(tmp_path):
    settings = _settings(tmp_path, data__use_sentiment=False)
    _bundle(tmp_path / "checkpoints", "MSFT", ["Close", "RSI_14"])
    _bundle(tmp_path / "checkpoints", "AAPL", ["Close", "Sentiment"])

    report = doctor.run_doctor(settings)

    assert report.healthy is False
    assert {b.name for b in report.bundles} == {"MSFT", "AAPL"}


def test_run_doctor_is_healthy_with_no_bundles_at_all(tmp_path):
    report = doctor.run_doctor(_settings(tmp_path))
    assert report.healthy is True
    assert report.bundles == []


def test_run_doctor_never_puts_a_secret_value_in_the_report(tmp_path):
    """Secrets non-negotiable: key *presence* is reported, key *values* never
    enter meta.json, runs.jsonl, logs -- or, as tested here, this report."""
    secret = "sk-super-secret-value-12345"
    settings = _settings(tmp_path, fred_api_key=secret, finnhub_api_key=secret)

    report = doctor.run_doctor(settings)
    serialized = json.dumps(report.to_dict())

    assert secret not in serialized
    assert report.keys == {"FRED_API_KEY": True, "FINNHUB_API_KEY": True}
