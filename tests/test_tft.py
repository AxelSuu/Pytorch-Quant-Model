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


def test_train_evaluates_best_checkpoint_not_live_model(monkeypatch, sample_ohlcv_df, fast_settings):
    """Reported metrics must come from the reloaded best checkpoint (the deployed
    model), not the live post-fit model EarlyStopping leaves past the best epoch
    (PYQ-109). The two are distinct objects; assert evaluation used the reload."""
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)

    built = {}
    real_build_model = tft.build_model

    def spy_build_model(ds, settings):
        model = real_build_model(ds, settings)
        built["live_model"] = model
        return model

    monkeypatch.setattr(tft, "build_model", spy_build_model)

    evaluated = {}
    real_eval = tft._evaluate_validation

    def spy_eval(model, val_loader, quantiles):
        evaluated["model"] = model
        return real_eval(model, val_loader, quantiles)

    monkeypatch.setattr(tft, "_evaluate_validation", spy_eval)

    tft.train("TEST", fast_settings, max_epochs=3, progress=False)

    assert "live_model" in built and "model" in evaluated
    # The evaluated model is a freshly reloaded checkpoint, a different object
    # from the live model that was fit. On the pre-fix code these were identical.
    assert evaluated["model"] is not built["live_model"]


def test_pooling_preserves_valid_sector_column_for_other_symbols(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    """Pooling a sector-ETF symbol must not strip its own SEC_ column from the
    other pooled symbols that legitimately have it (PYQ-111)."""
    base = add_technical_indicators(sample_ohlcv_df).dropna()
    panel_aaa = base.copy()
    panel_aaa["SEC_SPY"] = 0.01  # AAA legitimately carries SPY as a sector feature
    panel_spy = base.copy()  # SPY's own SEC_SPY was dropped for self-leakage

    def fake_build_panel(symbol, settings, *a, **k):
        return panel_aaa if symbol == "AAA" else panel_spy

    monkeypatch.setattr(tft, "build_panel", fake_build_panel)
    pooled = tft._build_pooled_long_df(["AAA", "SPY"], fast_settings, None, None)

    assert "SEC_SPY" in pooled.columns  # survived the pool-wide intersection
    aaa_rows = pooled[pooled["symbol"] == "AAA"]
    spy_rows = pooled[pooled["symbol"] == "SPY"]
    assert (aaa_rows["SEC_SPY"] == 0.01).all()  # real values kept for AAA
    assert (spy_rows["SEC_SPY"] == 0.0).all()  # neutralised (not leaked) for SPY


def test_train_seeds_everything_and_records_seed(monkeypatch, sample_ohlcv_df, fast_settings):
    """train() must seed before the fit and persist the seed for reproducibility (PYQ-210)."""
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)

    seeds = []
    monkeypatch.setattr(tft, "seed_everything", lambda s, **k: seeds.append(s))

    fast_settings.training.seed = 123
    tft.train("TEST", fast_settings, max_epochs=1, progress=False)

    assert 123 in seeds
    meta = json.loads((tft._bundle_dir(fast_settings, "TEST") / "meta.json").read_text())
    assert meta["seed"] == 123


def test_train_threads_num_workers_into_dataloaders(monkeypatch, sample_ohlcv_df, fast_settings):
    """TrainingConfig.num_workers must reach the DataLoader construction (PYQ-218)."""
    from pytorch_forecasting import TimeSeriesDataSet

    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)
    fast_settings.training.num_workers = 3

    recorded = []
    real_to_dataloader = TimeSeriesDataSet.to_dataloader

    def spy_to_dataloader(self, *args, **kwargs):
        recorded.append(kwargs.get("num_workers"))
        kwargs["num_workers"] = 0  # don't actually spawn workers inside the test
        return real_to_dataloader(self, *args, **kwargs)

    monkeypatch.setattr(TimeSeriesDataSet, "to_dataloader", spy_to_dataloader)
    tft.train("TEST", fast_settings, max_epochs=1, progress=False)

    train_val_workers = [w for w in recorded if w is not None]
    assert train_val_workers  # train + val loaders were built
    assert all(w == 3 for w in train_val_workers)


def test_train_threads_precision_into_trainer(monkeypatch, sample_ohlcv_df, fast_settings):
    """TrainingConfig.precision must reach the Trainer (PYQ-223), run safely regardless."""
    from lightning.pytorch import Trainer as RealTrainer

    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)
    fast_settings.training.precision = "bf16-mixed"

    recorded = {}

    def spy_trainer(*args, **kwargs):
        recorded["precision"] = kwargs.get("precision")
        kwargs["precision"] = "32-true"  # run at fp32 on CPU regardless of request
        return RealTrainer(*args, **kwargs)

    monkeypatch.setattr(tft, "Trainer", spy_trainer)
    tft.train("TEST", fast_settings, max_epochs=1, progress=False)

    assert recorded["precision"] == "bf16-mixed"
