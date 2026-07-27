"""Tests for typed configuration validation."""

from pathlib import Path

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


def test_training_config_holdout_is_longer_than_one_horizon():
    """PYQ-117: a holdout of exactly one horizon yields a single validation window."""
    cfg = TrainingConfig()
    assert cfg.validation_days > cfg.max_prediction_length
    # The dead `train_split` knob is gone -- it configured nothing (PYQ-125).
    assert not hasattr(cfg, "train_split")


def test_early_stopping_patience_is_configurable_with_todays_default():
    """PYQ-224: previously a hardcoded literal in both Trainer constructions."""
    assert TrainingConfig().early_stopping_patience == 5
    assert TrainingConfig(early_stopping_patience=9).early_stopping_patience == 9


def test_load_settings_rejects_a_config_path_that_does_not_exist(tmp_path):
    """PYQ-128: a typo'd --config silently trained on defaults, invalidating any
    A/B comparison between two experiment configs."""
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError, match="nope.yaml"):
        load_settings(missing)


def test_load_settings_without_a_config_stays_silent(monkeypatch):
    """Absent-by-default is not the same as explicitly-requested-and-missing."""
    monkeypatch.delenv("PYQUANT_CONFIG", raising=False)
    assert load_settings().tft.hidden_size == 32


# --- PYQ-220: paths anchor to the project, not the ambient cwd ----------------


def test_relative_paths_resolve_the_same_from_any_working_directory(tmp_path, monkeypatch):
    """`train` from the repo root then `forecast` from elsewhere must find the
    same bundle. Resolved against the ambient cwd they did not (PYQ-220)."""
    from pyquant.config import Settings

    monkeypatch.chdir(tmp_path)
    from_tmp = Settings()
    monkeypatch.chdir(Path(__file__).resolve().parent.parent)
    from_root = Settings()

    assert from_tmp.checkpoint_dir == from_root.checkpoint_dir
    assert from_tmp.data.cache_dir == from_root.data.cache_dir
    assert from_tmp.checkpoint_dir.is_absolute()
    assert from_tmp.data.cache_dir.is_absolute()


def test_pyquant_home_moves_the_anchor(tmp_path, monkeypatch):
    """A deployment that wants bundles outside the source tree gets one env var."""
    from pyquant.config import Settings

    monkeypatch.setenv("PYQUANT_HOME", str(tmp_path))
    s = Settings()
    assert s.checkpoint_dir == tmp_path.resolve() / "checkpoints"
    assert s.data.cache_dir == tmp_path.resolve() / ".cache/pyquant"


def test_an_absolute_configured_path_is_left_alone(tmp_path, monkeypatch):
    """Anchoring must not override an explicit absolute path."""
    from pyquant.config import Settings

    explicit = tmp_path / "elsewhere"
    monkeypatch.setenv("CHECKPOINT_DIR", str(explicit))
    assert Settings().checkpoint_dir == explicit
