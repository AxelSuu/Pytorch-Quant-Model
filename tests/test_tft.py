"""Network-free smoke test for the TFT wrapper.

Trains for 1 epoch on synthetic data (build_panel mocked) to verify the
train -> save -> load -> predict bundle round-trips. Kept small so it runs in CI.
"""

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
    panel = add_technical_indicators(sample_ohlcv_df)
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)

    result = tft.train("TEST", fast_settings, max_epochs=1, progress=False)
    assert result.bundle_dir.exists()
    assert (result.bundle_dir / "model.ckpt").exists()
    assert (result.bundle_dir / "dataset_params.pt").exists()
    assert (result.bundle_dir / "meta.json").exists()
    assert result.n_features > 0

    bundle = tft.load("TEST", fast_settings)
    assert bundle.meta["symbol"] == "TEST"
    assert len(bundle.meta["features"]) == result.n_features


def test_load_missing_model_raises(fast_settings):
    with pytest.raises(FileNotFoundError):
        tft.load("NOPE", fast_settings)


def test_train_rejects_insufficient_history(monkeypatch, sample_ohlcv_df, fast_settings):
    short = add_technical_indicators(sample_ohlcv_df).iloc[:15]
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: short)
    with pytest.raises(ValueError):
        tft.train("TEST", fast_settings, max_epochs=1, progress=False)
