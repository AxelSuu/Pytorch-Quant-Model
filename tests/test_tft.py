"""Network-free smoke test for the TFT wrapper.

Trains for 1 epoch on synthetic data (build_panel mocked) to verify the
train -> save -> load -> predict bundle round-trips. Kept small so it runs in CI.
"""

import json
import warnings

import pytest

from pyquant.data.prices import add_technical_indicators
from pyquant.models import tft

warnings.filterwarnings("ignore")


@pytest.fixture
def fast_settings(tmp_path, settings):
    settings.checkpoint_dir = tmp_path / "checkpoints"
    settings.training.max_encoder_length = 20
    settings.training.max_prediction_length = 3
    settings.training.batch_size = 32
    settings.tft.hidden_size = 8
    settings.tft.hidden_continuous_size = 4
    return settings


def test_train_load_predict_roundtrip(monkeypatch, sample_ohlcv_df, fast_settings):
    # build_panel() itself drops indicator warm-up rows (see dataset.py); this
    # test bypasses build_panel(), so it must replicate that cleanup itself.
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)

    result = tft.train("TEST", fast_settings, max_epochs=1, progress=False)
    assert result.bundle_dir.exists()
    assert (result.bundle_dir / "model.ckpt").exists()
    assert (result.bundle_dir / "dataset_params.pt").exists()
    assert (result.bundle_dir / "meta.json").exists()
    assert result.n_features > 0

    assert 0.0 <= result.evaluation.directional_accuracy <= 1.0
    assert 0.0 <= result.evaluation.calibration_coverage <= 1.0
    assert result.evaluation.model_mae >= 0.0
    assert result.evaluation.baseline_mae >= 0.0

    bundle = tft.load("TEST", fast_settings)
    assert bundle.meta["symbol"] == "TEST"
    assert len(bundle.meta["features"]) == result.n_features
    assert "evaluation" in bundle.meta
    assert "directional_accuracy" in bundle.meta["evaluation"]


def test_train_appends_to_run_log_instead_of_overwriting_history(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    """Retraining the same symbol must not erase the previous run's record."""
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)

    tft.train("TEST", fast_settings, max_epochs=1, progress=False)
    tft.train("TEST", fast_settings, max_epochs=1, progress=False)

    log_path = tft._bundle_dir(fast_settings, "TEST") / "runs.jsonl"
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        run = json.loads(line)
        assert run["symbol"] == "TEST"
        assert "trained_at" in run
        assert "val_loss" in run


def test_load_missing_model_raises(fast_settings):
    with pytest.raises(FileNotFoundError):
        tft.load("NOPE", fast_settings)


def test_train_rejects_insufficient_history(monkeypatch, sample_ohlcv_df, fast_settings):
    short = add_technical_indicators(sample_ohlcv_df).iloc[:15]
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: short)
    with pytest.raises(ValueError):
        tft.train("TEST", fast_settings, max_epochs=1, progress=False)


def test_walk_forward_backtest_aggregates_across_windows(monkeypatch, sample_ohlcv_df, fast_settings):
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)

    result = tft.walk_forward_backtest("TEST", fast_settings, n_windows=2, max_epochs=1, progress=False)

    assert result.symbol == "TEST"
    assert result.n_windows == 2
    assert len(result.per_window) == 2
    assert 0.0 <= result.aggregated.directional_accuracy <= 1.0
    assert 0.0 <= result.aggregated.calibration_coverage <= 1.0
    # No trained model artifacts should be persisted for a backtest.
    assert not (fast_settings.checkpoint_dir / "TEST").exists()


def test_train_pools_multiple_symbols_into_one_dataset(monkeypatch, sample_ohlcv_df, fast_settings):
    panel_a = add_technical_indicators(sample_ohlcv_df).dropna()
    panel_b = add_technical_indicators(sample_ohlcv_df * 1.01).dropna()

    def fake_build_panel(symbol, settings, *a, **k):
        return panel_a if symbol == "AAA" else panel_b

    monkeypatch.setattr(tft, "build_panel", fake_build_panel)

    result = tft.train(["AAA", "BBB"], fast_settings, max_epochs=1, progress=False)

    assert result.symbols == ["AAA", "BBB"]
    assert result.bundle_dir.name == "AAA_BBB"
    assert result.bundle_dir.exists()

    bundle = tft.load("AAA_BBB", fast_settings)
    assert set(bundle.meta["symbols"]) == {"AAA", "BBB"}


def test_train_single_symbol_still_uses_symbol_as_bundle_name(monkeypatch, sample_ohlcv_df, fast_settings):
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)

    result = tft.train("TEST", fast_settings, max_epochs=1, progress=False)
    assert result.symbols == ["TEST"]
    assert result.bundle_dir.name == "TEST"


def test_train_forwards_pin_to_build_panel(monkeypatch, sample_ohlcv_df, fast_settings):
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    received = {}

    def fake_build_panel(symbol, settings, start=None, end=None, pin=None):
        received["pin"] = pin
        return panel

    monkeypatch.setattr(tft, "build_panel", fake_build_panel)
    tft.train("TEST", fast_settings, max_epochs=1, progress=False, pin="exp-1")
    assert received["pin"] == "exp-1"


def test_walk_forward_backtest_rejects_insufficient_history(monkeypatch, sample_ohlcv_df, fast_settings):
    short = add_technical_indicators(sample_ohlcv_df).iloc[:15]
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: short)
    with pytest.raises(ValueError):
        tft.walk_forward_backtest("TEST", fast_settings, n_windows=3, max_epochs=1, progress=False)
