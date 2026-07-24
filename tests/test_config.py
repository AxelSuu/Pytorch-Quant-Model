"""Tests for typed configuration validation."""

import pytest

from pyquant.config import TFTConfig, TrainingConfig, load_settings


def test_tft_quantiles_reject_unsorted():
    """evaluate_predictions treats the first/last quantile as the calibration
    band bounds; an unsorted list would silently invert them (PYQ-219)."""
    with pytest.raises(ValueError, match="ascending"):
        TFTConfig(quantiles=[0.9, 0.1, 0.5])


def test_tft_quantiles_accept_sorted():
    cfg = TFTConfig(quantiles=[0.1, 0.5, 0.9])
    assert cfg.quantiles == [0.1, 0.5, 0.9]


def test_training_config_reproducibility_and_perf_defaults():
    """Defaults keep today's behavior: fixed seed, single-process loading, fp32."""
    cfg = TrainingConfig()
    assert cfg.seed == 42
    assert cfg.num_workers == 0
    assert cfg.precision == "32-true"


def test_yaml_config_overrides_defaults(tmp_path):
    """A YAML config overrides built-in defaults for the fields it sets (PYQ-209)."""
    cfg = tmp_path / "exp.yaml"
    cfg.write_text("tft:\n  hidden_size: 128\ntraining:\n  max_epochs: 7\n")
    s = load_settings(cfg)
    assert s.tft.hidden_size == 128
    assert s.training.max_epochs == 7
    assert s.tft.attention_head_size == 4  # unspecified field keeps its default


def test_yaml_config_via_env_var(tmp_path, monkeypatch):
    cfg = tmp_path / "exp.yaml"
    cfg.write_text("training:\n  max_epochs: 9\n")
    monkeypatch.setenv("PYQUANT_CONFIG", str(cfg))
    assert load_settings().training.max_epochs == 9


def test_no_config_uses_defaults(monkeypatch):
    monkeypatch.delenv("PYQUANT_CONFIG", raising=False)
    s = load_settings()
    assert s.tft.hidden_size == 32
    assert s.training.max_epochs == 30


def test_cli_flag_overrides_yaml_config(tmp_path):
    """Explicit CLI flags must still win over the config file (PYQ-209)."""
    from pyquant.cli.app import _build_settings

    cfg = tmp_path / "exp.yaml"
    cfg.write_text("data:\n  period: '10y'\n")

    from_config = _build_settings(None, False, False, False, config=cfg)
    assert from_config.data.period == "10y"  # config applied

    cli_wins = _build_settings("3y", False, False, False, config=cfg)
    assert cli_wins.data.period == "3y"  # explicit --period beats the config
