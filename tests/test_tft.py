"""Network-free smoke test for the TFT wrapper.

Trains for 1 epoch on synthetic data (build_panel mocked) to verify the
train -> save -> load -> predict bundle round-trips. Kept small so it runs in CI.
"""

import json
import logging
import warnings

import numpy as np
import pandas as pd
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
    # compute_signals defaults off (PYQ-255): no extra forward pass unless asked.
    assert result.signals == []
    assert result.signal_returns_pct == []
    # Window identity, in per_window order -- what compare_backtests (PYQ-266)
    # verifies before treating two backtests' per-window differences as paired.
    assert len(result.origins) == 2
    assert result.origins == sorted(result.origins)
    assert len(set(result.origins)) == 2  # distinct origins, not the same window twice


def test_walk_forward_backtest_computes_a_signal_per_window_when_requested(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    """PYQ-255: compute_signals=True must populate one BUY/SELL/HOLD signal and one
    realized return per window, using the same classify_signal() scan() uses."""
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)

    result = tft.walk_forward_backtest(
        "TEST", fast_settings, n_windows=2, max_epochs=1, progress=False, compute_signals=True
    )

    assert len(result.signals) == 2
    assert all(s in ("BUY", "SELL", "HOLD") for s in result.signals)
    assert len(result.signal_returns_pct) == 2
    assert all(isinstance(r, float) for r in result.signal_returns_pct)
    # PYQ-149: no conformal offset is ever applied here, so this must say so
    # explicitly rather than silently claiming parity with scan().
    assert result.signals_calibrated is False


# --- PYQ-149: --signals measures an uncalibrated band, and must say so -------


def test_signals_with_calibration_configured_warns_and_reports_uncalibrated(
    monkeypatch, sample_ohlcv_df, fast_settings, caplog
):
    """walk_forward_backtest() never fits a conformal offset (unlike train()),
    so compute_signals=True with calibration_days > 0 measures a different,
    uncalibrated case than scan() would show for an equivalent bundle. That
    divergence must be surfaced, not silent -- a warning at call time, and a
    machine-readable BacktestResult.signals_calibrated=False regardless of
    calibration_days, so a JSON/programmatic consumer sees it too."""
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)
    fast_settings.training.calibration_days = 5

    with caplog.at_level(logging.WARNING, logger="pyquant.models.tft"):
        result = tft.walk_forward_backtest(
            "TEST", fast_settings, n_windows=2, max_epochs=1, progress=False, compute_signals=True
        )

    assert result.signals_calibrated is False
    assert "uncalibrated" in caplog.text
    assert "PYQ-149" in caplog.text


def test_signals_without_compute_signals_does_not_warn_about_calibration(
    monkeypatch, sample_ohlcv_df, fast_settings, caplog
):
    """The warning is specifically about --signals' own uncalibrated band --
    a plain backtest (no signals requested) has nothing to warn about."""
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)
    fast_settings.training.calibration_days = 5

    with caplog.at_level(logging.WARNING, logger="pyquant.models.tft"):
        tft.walk_forward_backtest("TEST", fast_settings, n_windows=2, max_epochs=1, progress=False)

    assert "PYQ-149" not in caplog.text


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

    def spy_eval(model, val_loader, quantiles, *args, **kwargs):
        evaluated["model"] = model
        return real_eval(model, val_loader, quantiles, *args, **kwargs)

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


def test_two_identically_seeded_runs_produce_identical_metrics(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    """PYQ-246: PYQ-210's own acceptance criterion was "two consecutive train() calls
    with the same seed and a pinned dataset produce identical val_loss" -- but the test
    that shipped for it only checked that the seed is *passed* and *recorded*
    (test_train_seeds_everything_and_records_seed, above), not that the run is
    actually reproducible. seed_everything() does not by itself guarantee determinism
    on every backend (cuDNN autotuning, num_workers > 0 ordering), so the property this
    project claims could have been false on some configurations without anything
    noticing.

    On this CPU-only suite it holds exactly, with default num_workers=0 *and* with
    num_workers=2 (checked manually; not asserted here since the second is redundant
    with this test's own default DataLoader config and would only double the runtime).
    GPU determinism is not covered -- there is no GPU in this environment to check it
    against, and cuDNN autotuning is a real, materially different source of
    nondeterminism this test cannot see. Document rather than claim it: `runs.jsonl`
    comparisons across GPU-trained bundles should not assume bit-identical
    reproducibility the way this test verifies for CPU.
    """
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)

    result_a = tft.train("TEST", fast_settings, max_epochs=3, progress=False)
    result_b = tft.train("TEST", fast_settings, max_epochs=3, progress=False)

    assert result_a.val_loss == result_b.val_loss
    assert result_a.evaluation == result_b.evaluation


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


def test_prediction_decoder_covers_steps_after_last_observed_bar(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    """PYQ-115: the forecast must decode FUTURE steps, not re-predict observed days.

    `predict=True` anchors the decoder to the last `max_prediction_length`
    timesteps present in the frame. Handed a frame that ends at the last
    observed bar, it re-predicts days that already happened -- so the decoder
    window must start strictly after the last observed time_idx.
    """
    from pyquant.data.dataset import panel_to_long

    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)
    tft.train("TEST", fast_settings, max_epochs=1, progress=False)
    bundle = tft.load("TEST", fast_settings)

    df = panel_to_long(panel, "TEST")
    observed_max = int(df["time_idx"].max())

    ds = tft._prediction_dataset(bundle, df)
    x, _ = next(iter(ds.to_dataloader(train=False, batch_size=1, num_workers=0)))
    assert int(x["decoder_time_idx"].min()) > observed_max


def test_prediction_encoder_ends_on_the_last_observed_bar(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    """PYQ-115: attention_to_series() labels attention with the *last* observed
    panel dates, so the encoder must genuinely end on the last observed bar."""
    from pyquant.data.dataset import panel_to_long

    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)
    tft.train("TEST", fast_settings, max_epochs=1, progress=False)
    bundle = tft.load("TEST", fast_settings)

    df = panel_to_long(panel, "TEST")
    observed_max = int(df["time_idx"].max())

    ds = tft._prediction_dataset(bundle, df)
    x, _ = next(iter(ds.to_dataloader(train=False, batch_size=1, num_workers=0)))
    # The encoder ends exactly where the observed data ends: the first decoded
    # step is the first future day.
    assert int(x["decoder_time_idx"].min()) == observed_max + 1


def test_pooling_date_aligns_symbols_with_unequal_history(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    """PYQ-116: a late-listing symbol's validation window must not sit inside training.

    train() derives `cutoff` from the *global* max time_idx. With per-symbol
    positional indices, a shorter symbol's entire series -- including the window
    predict=True later returns as validation -- falls below that cutoff.
    """
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    panels = {"LONG": panel, "SHORT": panel.tail(80)}
    monkeypatch.setattr(tft, "build_panel", lambda symbol, *a, **k: panels[symbol])

    df = tft._build_pooled_long_df(["LONG", "SHORT"], fast_settings, None, None)

    horizon = fast_settings.training.max_prediction_length
    cutoff = int(df["time_idx"].max()) - horizon
    last_per_symbol = df.groupby("symbol")["time_idx"].max()
    assert (last_per_symbol > cutoff).all(), (
        f"cutoff={cutoff} but per-symbol max time_idx is {last_per_symbol.to_dict()}"
    )


def test_train_warns_when_a_symbols_history_ends_before_the_cutoff(
    monkeypatch, sample_ohlcv_df, fast_settings, caplog
):
    """PYQ-116: date alignment fixes late *starts*; a stale symbol that stops early
    still has its validation window inside training, so say so out loud."""
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    panels = {"LIVE": panel, "STALE": panel.head(200)}
    monkeypatch.setattr(tft, "build_panel", lambda symbol, *a, **k: panels[symbol])

    with caplog.at_level(logging.WARNING, logger="pyquant.models.tft"):
        tft.train(["LIVE", "STALE"], fast_settings, max_epochs=1, progress=False)

    assert "STALE" in caplog.text
    assert "LIVE" not in caplog.text  # the up-to-date symbol is fine
    assert "cutoff" in caplog.text.lower()


def test_train_evaluates_many_validation_windows_not_a_single_one(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    """PYQ-117: `predict=True` plus a one-horizon holdout gave exactly 1 sample --
    5 points driving every reported metric AND early stopping."""
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)

    result = tft.train("TEST", fast_settings, max_epochs=1, progress=False)

    horizon = fast_settings.training.max_prediction_length
    assert result.evaluation.n_samples > 1
    assert result.evaluation.n_points == result.evaluation.n_samples * horizon


def test_train_evaluation_scores_every_default_baseline_beyond_persistence(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    """PYQ-275: `_evaluate_validation` now threads the encoder history it
    already has through to `evaluate_predictions`, so a real bundle's
    reported metrics carry more than the single persistence comparator."""
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)

    result = tft.train("TEST", fast_settings, max_epochs=1, progress=False)

    from pyquant.analysis.baselines import DEFAULT_BASELINES

    expected_names = {b.name for b in DEFAULT_BASELINES}
    assert set(result.evaluation.baseline_maes) == expected_names
    assert result.evaluation.baseline_maes["persistence"] == pytest.approx(
        result.evaluation.baseline_mae
    )
    assert result.evaluation.strongest_baseline is not None


def test_validation_predictions_actuals_and_persistence_baseline_share_price_units(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    """PYQ-240/PYQ-313: upstream prediction outputs must stay in raw target units.

    This independently maps the decoder indices back to raw Close values. It
    fails if pytorch-forecasting starts returning any of the arrays normalised.
    """
    from pytorch_forecasting import TimeSeriesDataSet

    from pyquant.analysis.metrics import persistence_baseline_mae
    from pyquant.data.dataset import align_time_index, make_dataset, panel_to_long

    fast_settings.training.target = "close"
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)
    tft.train("TEST", fast_settings, max_epochs=1, progress=False)
    bundle = tft.load("TEST", fast_settings)

    df = align_time_index(panel_to_long(panel, "TEST"))
    cutoff = int(df["time_idx"].max()) - fast_settings.training.validation_days
    training = make_dataset(df, fast_settings, training_cutoff=cutoff)
    validation = TimeSeriesDataSet.from_dataset(
        training, df, min_prediction_idx=cutoff + 1, stop_randomization=True
    )
    loader = validation.to_dataloader(train=False, batch_size=fast_settings.training.batch_size)
    result = bundle.model.predict(loader, mode="quantiles", return_x=True, return_y=True)

    predictions = result.output.cpu().numpy()
    actuals = result.y[0].cpu().numpy()
    last_observed = result.x["encoder_target"][:, -1].cpu().numpy()
    decoder_idx = result.x["decoder_time_idx"].cpu().numpy()
    raw_close = df.set_index("time_idx")["Close"]
    expected_actuals = np.array([[raw_close[i] for i in row] for row in decoder_idx])
    expected_last = np.array([raw_close[row[0] - 1] for row in decoder_idx])

    np.testing.assert_allclose(actuals, expected_actuals)
    np.testing.assert_allclose(last_observed, expected_last)
    assert np.all(predictions > 10.0)
    assert persistence_baseline_mae(actuals, last_observed) == pytest.approx(
        persistence_baseline_mae(expected_actuals, expected_last)
    )


def test_walk_forward_window_validation_targets_its_own_origin(sample_ohlcv_df, settings):
    """PYQ-127: every rolling origin evaluated the *same* final window, so a
    5-window backtest was 5 models scored on one identical 5 days."""
    from pyquant.data.dataset import align_time_index, make_dataset, panel_to_long

    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    df = align_time_index(panel_to_long(panel, "TEST"))
    horizon = settings.training.max_prediction_length

    windows = []
    for cutoff in (200, 220, 240):
        training = make_dataset(df, settings, training_cutoff=cutoff)
        ds = tft._window_validation_dataset(training, df, cutoff, horizon)
        x, _ = next(iter(ds.to_dataloader(train=False, batch_size=1, num_workers=0)))
        decoded = x["decoder_time_idx"][0].tolist()
        assert decoded[0] == cutoff + 1, f"origin {cutoff} evaluated {decoded}"
        windows.append(tuple(decoded))

    assert len(set(windows)) == 3, f"origins collapsed onto the same window: {windows}"


# --- PYQ-118 / PYQ-119: schema drift and recorded config ---------------------


def _rich_and_lean_panels(sample_ohlcv_df):
    """A panel with an enrichment column, and the same panel without it."""
    import numpy as np

    lean = add_technical_indicators(sample_ohlcv_df).dropna()
    rich = lean.copy()
    rng = np.random.default_rng(0)
    rich["SEC_SPY"] = rng.normal(scale=0.01, size=len(rich))
    return rich, lean


def test_predict_raises_a_clear_error_when_a_trained_feature_is_missing(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    """PYQ-118: this was a bare `KeyError: 'SEC_SPY'` from inside pytorch-forecasting."""
    from pyquant.data.dataset import panel_to_long

    rich, lean = _rich_and_lean_panels(sample_ohlcv_df)
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: rich)
    tft.train("TEST", fast_settings, max_epochs=1, progress=False)
    bundle = tft.load("TEST", fast_settings)

    with pytest.raises(tft.FeatureSchemaMismatch) as exc:
        tft.predict_quantiles(bundle, panel_to_long(lean, "TEST"))

    message = str(exc.value)
    assert "SEC_SPY" in message
    assert "sector" in message.lower()  # names the source that went missing


def test_predict_ignores_columns_not_seen_during_training(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    """Extra columns at predict time were already harmless -- keep them harmless."""
    from pyquant.data.dataset import panel_to_long

    rich, lean = _rich_and_lean_panels(sample_ohlcv_df)
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: lean)
    tft.train("TEST", fast_settings, max_epochs=1, progress=False)
    bundle = tft.load("TEST", fast_settings)

    predictions = tft.predict_quantiles(bundle, panel_to_long(rich, "TEST"))
    assert predictions.shape == (fast_settings.training.max_prediction_length, 3)


def test_prediction_rejects_a_multi_symbol_frame_instead_of_returning_group_zero(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    """PYQ-131: batched frames must not silently return another symbol's path."""
    from pyquant.data.dataset import panel_to_long

    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)
    tft.train("TEST", fast_settings, max_epochs=1, progress=False)
    bundle = tft.load("TEST", fast_settings)
    multi_symbol = pd.concat(
        [panel_to_long(panel, "AAA"), panel_to_long(panel, "BBB")], ignore_index=True
    )

    with pytest.raises(ValueError, match="exactly one symbol"):
        tft.predict_quantiles(bundle, multi_symbol)


def test_interpret_raises_on_encoder_name_weight_length_mismatch(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    """PYQ-147: interpret() used to zip names to weights with strict=False, which
    would silently truncate and misattribute weights to the wrong feature names
    on any length mismatch instead of failing loudly."""
    from pyquant.data.dataset import panel_to_long

    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)
    tft.train("TEST", fast_settings, max_epochs=1, progress=False)
    bundle = tft.load("TEST", fast_settings)

    # Leave encoder_variables (names) alone -- the forward pass needs it to line
    # up with the model's real inputs. Instead shorten what interpret_output
    # (weights) reports, the same shape of drift a pytorch-forecasting version
    # bump could introduce.
    real_interpret_output = bundle.model.interpret_output

    def _one_weight_short(output, reduction="sum"):
        result = dict(real_interpret_output(output, reduction=reduction))
        result["encoder_variables"] = result["encoder_variables"][:-1]
        return result

    monkeypatch.setattr(bundle.model, "interpret_output", _one_weight_short)

    with pytest.raises(ValueError):
        tft.interpret(bundle, panel_to_long(panel, "TEST"))


def test_train_records_the_resolved_data_config(monkeypatch, sample_ohlcv_df, fast_settings):
    """PYQ-119: the feature schema is a function of the data toggles, so the bundle
    must record them -- nothing else can recover them later."""
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)
    fast_settings.data.use_sectors = False
    fast_settings.data.period = "3y"

    tft.train("TEST", fast_settings, max_epochs=1, progress=False)
    bundle = tft.load("TEST", fast_settings)

    recorded = bundle.meta["config"]["data"]
    assert recorded["use_sectors"] is False
    assert recorded["period"] == "3y"


def test_settings_for_bundle_restores_the_recorded_data_toggles(fast_settings):
    """A bundle trained with an enrichment off must not be forecast with it on."""

    class Bundle:
        meta = {"config": {"data": {"use_sectors": False, "period": "3y"}}}

    fast_settings.data.use_sectors = True
    fast_settings.data.period = "5y"

    restored = tft.settings_for_bundle(Bundle(), fast_settings)

    assert restored.data.use_sectors is False
    assert restored.data.period == "3y"
    # The caller's own settings object is left alone.
    assert fast_settings.data.use_sectors is True


# --- PYQ-224 / PYQ-225: configurable patience, recorded provenance -----------


def _spy_trainer_patience(monkeypatch, recorded):
    from lightning.pytorch.callbacks import EarlyStopping

    real_trainer = tft.Trainer

    def spy(**kwargs):
        for cb in kwargs.get("callbacks", []):
            if isinstance(cb, EarlyStopping):
                recorded.append(cb.patience)
        return real_trainer(**kwargs)

    monkeypatch.setattr(tft, "Trainer", spy)


def test_train_threads_early_stopping_patience_into_the_callback(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    """PYQ-224: patience was a literal 5, while every neighbouring knob was configurable."""
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)
    fast_settings.training.early_stopping_patience = 2

    recorded: list[int] = []
    _spy_trainer_patience(monkeypatch, recorded)
    tft.train("TEST", fast_settings, max_epochs=1, progress=False)

    assert recorded == [2]


def test_backtest_threads_early_stopping_patience_into_the_callback(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)
    fast_settings.training.early_stopping_patience = 3

    recorded: list[int] = []
    _spy_trainer_patience(monkeypatch, recorded)
    tft.walk_forward_backtest("TEST", fast_settings, n_windows=2, max_epochs=1, progress=False)

    assert recorded == [3, 3]


def test_train_records_provenance_including_the_pin(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    """PYQ-225: seed + pinned data only reproduce a run if the code version is known too."""
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)

    tft.train("TEST", fast_settings, max_epochs=1, progress=False, pin="exp-7")
    bundle = tft.load("TEST", fast_settings)

    provenance = bundle.meta["provenance"]
    assert provenance["pyquant_version"]
    assert provenance["pin"] == "exp-7"
    assert "git_sha" in provenance  # best-effort: present, may be None


# --- PYQ-250: purge + embargo around every split ------------------------------


def _decoder_range(ds) -> tuple[int, int]:
    """(min, max) decoder ``time_idx`` across every sample in a TimeSeriesDataSet.

    Each row of ``ds.index`` is one sample starting at ``time`` and running for
    ``sequence_length`` steps; the decoder is the last ``max_prediction_length``
    of those.
    """
    end = ds.index["time"] + ds.index["sequence_length"] - 1
    start = end - ds.max_prediction_length + 1
    return int(start.min()), int(end.max())


def _long_df(sample_ohlcv_df):
    from pyquant.data.dataset import align_time_index, panel_to_long

    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    return align_time_index(panel_to_long(panel, "TEST"))


def test_no_training_decoder_overlaps_the_validation_window_at_any_origin(
    sample_ohlcv_df, fast_settings
):
    """The remaining known member of the leak family (PYQ-250).

    A training sample whose decoder reaches the days immediately before the
    split shares target days with the period about to be scored -- the
    validation sample reads exactly those days through its own encoder. Purge
    drops them; embargo drops a further buffer for serial correlation. Asserted
    at *every* walk-forward origin, not just one, because PYQ-127's defect was
    that the origins were not actually distinct.
    """
    from pyquant.data.dataset import make_dataset

    df = _long_df(sample_ohlcv_df)
    settings = fast_settings
    horizon = settings.training.max_prediction_length
    settings.training.purge_horizon = horizon
    settings.training.embargo_days = 2

    max_idx = int(df["time_idx"].max())
    for origin in range(max_idx - horizon - 40, max_idx - horizon + 1, 10):
        train_cutoff = tft.purged_training_cutoff(origin, settings)
        training = make_dataset(df, settings, training_cutoff=train_cutoff)
        validation = tft._window_validation_dataset(training, df, origin, horizon)

        _, train_decoder_end = _decoder_range(training)
        val_decoder_start, _ = _decoder_range(validation)

        assert train_decoder_end < val_decoder_start, (
            f"origin {origin}: a training decoder reaches {train_decoder_end}, "
            f"into the validation window starting at {val_decoder_start}"
        )
        # And not merely non-overlapping -- separated by the full buffer.
        assert val_decoder_start - train_decoder_end > horizon + 2


def test_purge_and_embargo_shrink_the_training_slice_by_exactly_their_sum(fast_settings):
    """Both knobs are configurable and additive, and zeroing them restores the
    pre-PYQ-250 geometry exactly -- which is what makes the before/after
    skill comparison in the ticket a controlled one."""
    fast_settings.training.max_prediction_length = 5

    fast_settings.training.purge_horizon = 0
    fast_settings.training.embargo_days = 0
    assert tft.purged_training_cutoff(100, fast_settings) == 100

    fast_settings.training.purge_horizon = None  # -> max_prediction_length
    fast_settings.training.embargo_days = 2
    assert tft.purged_training_cutoff(100, fast_settings) == 100 - 5 - 2

    fast_settings.training.purge_horizon = 3
    assert tft.purged_training_cutoff(100, fast_settings) == 100 - 3 - 2


def test_train_still_validates_on_the_full_holdout_after_purging(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    """Purging must shrink *training* only. If it moved the validation window
    too, the sample size PYQ-117 fought for would quietly shrink with it."""
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)
    fast_settings.training.validation_days = 30

    fast_settings.training.purge_horizon = 0
    fast_settings.training.embargo_days = 0
    unpurged = tft.train("TEST", fast_settings, max_epochs=1, progress=False)

    fast_settings.training.purge_horizon = 5
    fast_settings.training.embargo_days = 2
    purged = tft.train("TEST", fast_settings, max_epochs=1, progress=False)

    assert purged.evaluation.n_points == unpurged.evaluation.n_points
    assert purged.evaluation.n_samples == unpurged.evaluation.n_samples


# --- PYQ-143: checkpoint selection disjoint from the reported test window -----


def _spy_trainer_val_dataset(monkeypatch, captured, key):
    """Patch tft.Trainer so trainer.fit(...)'s val_dataloaders is recorded."""
    real_trainer_cls = tft.Trainer

    def spy_trainer(**kwargs):
        trainer = real_trainer_cls(**kwargs)
        real_fit = trainer.fit

        def fit(model, train_dataloaders=None, val_dataloaders=None):
            captured[key] = val_dataloaders.dataset
            return real_fit(model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)

        trainer.fit = fit
        return trainer

    monkeypatch.setattr(tft, "Trainer", spy_trainer)


def test_train_fits_against_a_selection_window_disjoint_from_the_reported_test_window(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    """PYQ-143: EarlyStopping/ModelCheckpoint must select the checkpoint against
    a window that is neither the training data nor the window EvaluationMetrics
    is later reported from. Before this fix both used the same `val_loader`,
    so the reported metrics were a best-of-many-epochs statistic (the same
    selection-event bias `TuneResult`'s own docstring names for Optuna trials,
    applied to epochs instead)."""
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)

    captured: dict = {}
    _spy_trainer_val_dataset(monkeypatch, captured, "fit_val_dataset")

    real_evaluate = tft._evaluate_validation

    def spy_evaluate(model, val_loader, *a, **k):
        captured["reported_val_dataset"] = val_loader.dataset
        return real_evaluate(model, val_loader, *a, **k)

    monkeypatch.setattr(tft, "_evaluate_validation", spy_evaluate)

    tft.train("TEST", fast_settings, max_epochs=1, progress=False)

    assert "fit_val_dataset" in captured
    assert "reported_val_dataset" in captured
    fit_range = _decoder_range(captured["fit_val_dataset"])
    reported_range = _decoder_range(captured["reported_val_dataset"])
    assert fit_range != reported_range
    # Selection strictly precedes the reported test window -- not just a
    # different window, but the ordered, purged geometry PYQ-143 asks for.
    assert fit_range[1] < reported_range[0]


def test_walk_forward_backtest_fits_against_a_selection_window_per_origin(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    """The walk-forward path is worse pre-fix: predict=True gives exactly one
    sample per origin, so that single window was simultaneously what
    early-stopping/checkpoint-selection optimized against and what the
    per-window (and aggregate) reported metric came from. Each origin must now
    select against its own disjoint selection window."""
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)

    captured: dict = {}
    _spy_trainer_val_dataset(monkeypatch, captured, "fit_val_dataset")

    real_evaluate = tft._evaluate_best_checkpoint

    def spy_evaluate(best_path, model, val_loader, *a, **k):
        captured["reported_val_dataset"] = val_loader.dataset
        return real_evaluate(best_path, model, val_loader, *a, **k)

    monkeypatch.setattr(tft, "_evaluate_best_checkpoint", spy_evaluate)

    tft.walk_forward_backtest("TEST", fast_settings, n_windows=1, max_epochs=1, progress=False)

    fit_range = _decoder_range(captured["fit_val_dataset"])
    reported_range = _decoder_range(captured["reported_val_dataset"])
    assert fit_range != reported_range
    assert fit_range[1] < reported_range[0]


def test_selection_days_is_configurable_and_recorded_on_the_bundle(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    """selection_days is a TrainingConfig field like every other tunable split
    knob, and (via the existing whole-config recording) ends up in meta.json."""
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)
    fast_settings.training.selection_days = 15

    tft.train("TEST", fast_settings, max_epochs=1, progress=False)
    bundle = tft.load("TEST", fast_settings)

    assert bundle.meta["config"]["training"]["selection_days"] == 15


# --- PYQ-265: repeat a backtest across seeds -----------------------------------


def test_walk_forward_backtest_multi_seed_runs_once_per_seed_and_retains_each_result(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)

    result = tft.walk_forward_backtest_multi_seed(
        "TEST", fast_settings, seeds=[1, 2], n_windows=2, max_epochs=1, progress=False
    )

    assert result.symbol == "TEST"
    assert result.seeds == [1, 2]
    assert len(result.per_seed) == 2
    assert all(isinstance(r, tft.BacktestResult) for r in result.per_seed)
    assert all(r.n_windows == 2 for r in result.per_seed)
    # The caller's own settings object must be left alone (each seed gets a
    # deep-copied Settings, not a mutation of the shared one).
    assert fast_settings.training.seed == 42


def test_walk_forward_backtest_multi_seed_defaults_to_configured_training_seeds(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    """Omitting `seeds=` falls back to `settings.training.seeds`, which itself
    defaults to a single-element list -- so a caller who never opts in gets
    exactly today's one-seed behaviour, just wrapped."""
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)
    assert fast_settings.training.seeds == [42]

    result = tft.walk_forward_backtest_multi_seed(
        "TEST", fast_settings, n_windows=2, max_epochs=1, progress=False
    )

    assert result.seeds == [42]
    assert len(result.per_seed) == 1


def test_walk_forward_backtest_multi_seed_same_seeds_reproduce_identical_results(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)

    a = tft.walk_forward_backtest_multi_seed(
        "TEST", fast_settings, seeds=[7, 8], n_windows=2, max_epochs=1, progress=False
    )
    b = tft.walk_forward_backtest_multi_seed(
        "TEST", fast_settings, seeds=[7, 8], n_windows=2, max_epochs=1, progress=False
    )

    assert [r.aggregated for r in a.per_seed] == [r.aggregated for r in b.per_seed]
    assert a.skill_mean == b.skill_mean


def test_walk_forward_backtest_multi_seed_different_seeds_give_different_results(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)

    a = tft.walk_forward_backtest_multi_seed(
        "TEST", fast_settings, seeds=[1], n_windows=2, max_epochs=1, progress=False
    )
    b = tft.walk_forward_backtest_multi_seed(
        "TEST", fast_settings, seeds=[2], n_windows=2, max_epochs=1, progress=False
    )

    assert a.per_seed[0].aggregated != b.per_seed[0].aggregated


def test_seed_sweep_result_summary_stats_match_manual_calculation(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)

    result = tft.walk_forward_backtest_multi_seed(
        "TEST", fast_settings, seeds=[1, 2, 3], n_windows=2, max_epochs=1, progress=False
    )

    skills = [r.aggregated.skill_vs_baseline for r in result.per_seed]
    assert result.skill_mean == pytest.approx(sum(skills) / len(skills))
    assert result.skill_min == pytest.approx(min(skills))
    assert result.skill_max == pytest.approx(max(skills))
    assert result.skill_sd >= 0.0


def test_seed_sweep_result_skill_sd_is_zero_for_a_single_seed(monkeypatch, sample_ohlcv_df, fast_settings):
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)

    result = tft.walk_forward_backtest_multi_seed(
        "TEST", fast_settings, seeds=[1], n_windows=2, max_epochs=1, progress=False
    )

    assert result.skill_sd == 0.0


# --- PYQ-248: the conformal offset travels with the bundle --------------------


def test_calibration_slice_produces_an_offset_that_forecast_reuses(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    """The offset must be fitted on a slice disjoint from training and from the
    scored window, persisted, and applied at predict time -- otherwise the
    coverage a bundle reports is not the coverage of the band it prints."""
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)
    fast_settings.training.validation_days = 30
    fast_settings.training.calibration_days = 20

    tft.train("TEST", fast_settings, max_epochs=1, progress=False)
    bundle = tft.load("TEST", fast_settings)

    recorded = bundle.meta["conformal"]
    assert recorded is not None
    assert recorded["n_calibration"] > 0
    assert recorded["nominal_coverage"] == pytest.approx(0.8)

    offset = tft.bundle_conformal_offset(bundle)
    assert offset.offset == pytest.approx(recorded["offset"])

    # And the prediction path actually applies it.
    from pyquant.data.dataset import panel_to_long

    df = panel_to_long(panel, "TEST")
    calibrated = tft.predict_quantiles(bundle, df)
    monkeypatch.setattr(tft, "bundle_conformal_offset", lambda _b: None)
    raw = tft.predict_quantiles(bundle, df)

    assert not np.allclose(calibrated, raw)
    np.testing.assert_allclose(calibrated[:, 1], raw[:, 1])  # median untouched


def test_a_bundle_without_a_calibration_slice_records_no_offset(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    """calibration_days defaults to 0, so nothing changes for existing bundles
    and no coverage figure moves without someone asking for it."""
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)
    assert fast_settings.training.calibration_days == 0

    tft.train("TEST", fast_settings, max_epochs=1, progress=False)
    bundle = tft.load("TEST", fast_settings)

    assert bundle.meta["conformal"] is None
    assert tft.bundle_conformal_offset(bundle) is None


def test_permutation_importance_ranks_the_injected_signal_above_pure_noise_features(
    monkeypatch, tmp_path
):
    """PYQ-314: sanity-checks permutation_importance() itself, independent of any
    question about what interpret()'s TFT weights mean -- on a panel where exactly
    one feature (Signal) actually drives the target and the rest are pure noise, a
    working implementation must rank Signal at or near the top.

    Deliberately skips add_technical_indicators(): a first version of this test
    included them and Signal scored *zero* while SMA_10 scored highest, because
    every indicator is a smoothing of Close, and Close's own path necessarily
    encodes the cumulative signal history the log-return target is derived from
    (panel_to_long() computes LogReturn from Close directly). Shuffling Signal
    alone left correlated echoes of it intact in every indicator column, which
    is a genuine, useful finding about permutation importance's blind spot with
    collinear features (see the PYQ-314 resolution note and PYQ-316) -- but it
    would make a poor mechanics check for this function specifically, so this
    test isolates the mechanism instead of exercising that interaction.
    """
    rng = np.random.default_rng(11)
    n = 300
    dates = pd.bdate_range("2022-01-03", periods=n)
    signal = rng.choice([-1.0, 1.0], size=n)
    log_returns = 0.08 * np.roll(signal, 1) + rng.normal(0, 0.002, n)
    log_returns[0] = 0.0
    close = 100 * np.exp(np.cumsum(log_returns))
    panel = pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.001,
            "Low": close * 0.999,
            "Close": close,
            "Volume": np.abs(rng.normal(0, 1, n)) * 1_000_000,
            "Signal": signal,
            "NoiseA": rng.normal(0, 1, n),
            "NoiseB": rng.normal(0, 1, n),
        },
        index=dates,
    )
    panel.index.name = "Date"
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)

    from pyquant.config import Settings

    settings = Settings()
    settings.data.use_macro = False
    settings.data.use_sectors = False
    settings.data.use_sentiment = False
    settings.data.cache_enabled = False
    settings.training.target = "log_return"
    settings.training.max_encoder_length = 15
    settings.training.max_prediction_length = 1
    settings.training.validation_days = 40
    settings.training.batch_size = 32
    settings.training.max_epochs = 15
    settings.tft.hidden_size = 8
    settings.tft.hidden_continuous_size = 4
    settings.checkpoint_dir = tmp_path / "checkpoints"

    tft.train("TEST", settings, progress=False)
    bundle = tft.load("TEST", settings)

    from pyquant.data.dataset import panel_to_long

    long_df = panel_to_long(panel, "TEST")
    importance = tft.permutation_importance(bundle, long_df, settings)

    assert "Signal" in importance
    top_feature = max(importance, key=importance.get)
    assert top_feature == "Signal", f"expected Signal on top, got {importance}"
    assert importance["Signal"] > importance.get("NoiseA", 0.0)
    assert importance["Signal"] > importance.get("NoiseB", 0.0)


def test_tune_writes_a_config_and_scores_the_winner_on_a_held_out_split(
    monkeypatch, sample_ohlcv_df, fast_settings
):
    """PYQ-253: needs the 'tuning' extra (optuna/statsmodels/tensorboard), which CI's
    default job does not install -- skips cleanly there, same disposition PYQ-308
    already established for a real-FinBERT CI job. Verified locally with the extra
    installed (see the ticket's resolution note for a real run's output).
    """
    pytest.importorskip("optuna")
    pytest.importorskip("statsmodels")
    pytest.importorskip("tensorboard")

    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)

    result = tft.tune("TEST", fast_settings, n_trials=1, max_epochs=1, held_out_days=20, progress=False)

    assert result.symbol == "TEST"
    assert result.n_trials == 1
    assert result.best_params  # at least one tuned hyperparameter recorded
    assert result.held_out_evaluation.n_samples > 0
    assert 0.0 <= result.held_out_evaluation.calibration_coverage <= 1.0
    assert result.config_path.exists()
    assert result.config_path.suffix == ".yaml"
    assert (result.bundle_dir / "optuna_study.db").exists()
    # The bundle behind the held-out score is the real, loadable thing train() makes.
    bundle = tft.load("TEST_TUNED", fast_settings)
    assert bundle.meta["symbol"] == "TEST_TUNED"

    result.config_path.unlink()  # scripts/configs/ is real repo state, not a tmp dir


def test_tune_without_the_extra_installed_fails_clearly(monkeypatch, fast_settings):
    """A missing 'tuning' extra must not surface as a bare ImportError deep inside
    pytorch-forecasting -- name the fix."""
    monkeypatch.setitem(__import__("sys").modules, "optuna", None)
    with pytest.raises(ImportError, match="tuning"):
        tft.tune("TEST", fast_settings, n_trials=1)
