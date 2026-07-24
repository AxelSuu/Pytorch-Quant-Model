"""Tests for typed configuration validation."""

import pytest

from pyquant.config import TFTConfig, TrainingConfig


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
