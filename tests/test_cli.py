"""CLI smoke tests using Typer's CliRunner (network-free, mocked forecasts)."""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from pyquant.analysis.forecast import Forecast
from pyquant.analysis.metrics import EvaluationMetrics
from pyquant.cli import app as app_mod
from pyquant.models.tft import TrainResult

runner = CliRunner()


def _fake_forecast(symbol="AAPL", predictions=None):
    dates = pd.bdate_range("2024-01-01", periods=20)
    if predictions is None:
        predictions = np.array([[95.0, 105.0, 115.0]] * 5)
    return Forecast(
        symbol=symbol,
        last_date=dates[-1],
        current_price=100.0,
        quantiles=[0.1, 0.5, 0.9],
        predictions=predictions,
        history=pd.Series(np.linspace(90, 100, 20), index=dates),
    )


def test_forecast_command(monkeypatch):
    monkeypatch.setattr(app_mod, "generate_forecast", lambda *a, **k: _fake_forecast())

    class NoOptions:
        put_call_ratio = None

    monkeypatch.setattr(app_mod, "fetch_options_snapshot", lambda s: NoOptions())
    result = runner.invoke(app_mod.app, ["forecast", "AAPL", "--no-chart"])
    assert result.exit_code == 0
    assert "5-day forecast" in result.stdout
    assert "$105.00" in result.stdout  # median day value


def test_scan_handles_untrained(monkeypatch):
    def raise_missing(symbol, settings):
        raise FileNotFoundError("no model")

    monkeypatch.setattr(app_mod, "generate_forecast", raise_missing)
    result = runner.invoke(app_mod.app, ["scan", "AAPL,MSFT"])
    assert result.exit_code == 0
    assert "not trained" in result.stdout


def test_scan_survives_one_symbol_raising_non_filenotfound(monkeypatch):
    """A non-FileNotFoundError from one symbol must not crash the whole scan (PYQ-113)."""

    def flaky(symbol, settings):
        if symbol == "BAD":
            raise RuntimeError("transient data-source error")
        return _fake_forecast(symbol=symbol, predictions=np.array([[102.0, 105.0, 108.0]] * 5))

    monkeypatch.setattr(app_mod, "generate_forecast", flaky)
    result = runner.invoke(app_mod.app, ["scan", "GOOD,BAD"])
    assert result.exit_code == 0
    assert "GOOD" in result.stdout  # the healthy symbol still rendered
    assert "error" in result.stdout.lower()  # the flaky one shown as an error row


def test_train_command_reports_evaluation_metrics(monkeypatch):
    fake_result = TrainResult(
        symbols=["AAPL"],
        bundle_dir=Path("checkpoints/AAPL"),
        val_loss=0.123,
        n_features=10,
        epochs_run=5,
        evaluation=EvaluationMetrics(
            model_mae=1.5,
            baseline_mae=2.0,
            directional_accuracy=0.6,
            calibration_coverage=0.8,
        ),
    )
    monkeypatch.setattr(app_mod.tft, "train", lambda *a, **k: fake_result)
    result = runner.invoke(app_mod.app, ["train", "AAPL"])
    assert result.exit_code == 0
    assert "Directional accuracy" in result.stdout
    assert "60.0%" in result.stdout
    assert "Calibration" in result.stdout
    assert "80.0%" in result.stdout
    assert "Skill vs. baseline" in result.stdout


def test_train_command_pools_multiple_symbols(monkeypatch):
    captured = {}

    def fake_train(symbols, settings, **kwargs):
        captured["symbols"] = symbols
        return TrainResult(
            symbols=symbols,
            bundle_dir=Path("checkpoints/AAPL_MSFT"),
            val_loss=0.1,
            n_features=5,
            epochs_run=1,
            evaluation=EvaluationMetrics(
                model_mae=1.0, baseline_mae=1.0, directional_accuracy=0.5, calibration_coverage=0.8
            ),
        )

    monkeypatch.setattr(app_mod.tft, "train", fake_train)
    result = runner.invoke(app_mod.app, ["train", "AAPL,MSFT"])
    assert result.exit_code == 0
    assert captured["symbols"] == ["AAPL", "MSFT"]
    assert "AAPL, MSFT" in result.stdout


def test_backtest_command_reports_aggregated_metrics(monkeypatch):
    from pyquant.models.tft import BacktestResult

    ev = EvaluationMetrics(
        model_mae=1.2, baseline_mae=1.6, directional_accuracy=0.55, calibration_coverage=0.75
    )
    fake_result = BacktestResult(symbol="AAPL", n_windows=3, per_window=[ev, ev, ev], aggregated=ev)
    monkeypatch.setattr(app_mod.tft, "walk_forward_backtest", lambda *a, **k: fake_result)
    result = runner.invoke(app_mod.app, ["backtest", "AAPL"])
    assert result.exit_code == 0
    assert "3 windows" in result.stdout
    assert "55.0%" in result.stdout
    assert "75.0%" in result.stdout


def test_default_logging_level_is_warning(monkeypatch):
    monkeypatch.setattr(app_mod, "generate_forecast", lambda *a, **k: _fake_forecast())
    runner.invoke(app_mod.app, ["forecast", "AAPL", "--no-chart"])
    assert logging.getLogger().level == logging.WARNING
    assert logging.getLogger("lightning.pytorch").level == logging.ERROR


def test_verbose_flag_enables_info_logging(monkeypatch):
    monkeypatch.setattr(app_mod, "generate_forecast", lambda *a, **k: _fake_forecast())
    result = runner.invoke(app_mod.app, ["--verbose", "forecast", "AAPL", "--no-chart"])
    assert result.exit_code == 0
    assert logging.getLogger().level == logging.INFO


def test_debug_flag_enables_debug_logging_and_lightning_chatter(monkeypatch):
    monkeypatch.setattr(app_mod, "generate_forecast", lambda *a, **k: _fake_forecast())
    result = runner.invoke(app_mod.app, ["--debug", "forecast", "AAPL", "--no-chart"])
    assert result.exit_code == 0
    assert logging.getLogger().level == logging.DEBUG
    assert logging.getLogger("lightning.pytorch").level <= logging.INFO


def _has_ignore_filter(category: type) -> bool:
    # warnings.filters entries are plain tuples: (action, message, category, module, lineno).
    return any(f[0] == "ignore" and f[2] is category for f in warnings.filters)


def test_default_run_suppresses_user_and_deprecation_warnings(monkeypatch):
    monkeypatch.setattr(app_mod, "generate_forecast", lambda *a, **k: _fake_forecast())
    with warnings.catch_warnings():
        warnings.resetwarnings()
        runner.invoke(app_mod.app, ["forecast", "AAPL", "--no-chart"])
        assert _has_ignore_filter(UserWarning)
        assert _has_ignore_filter(DeprecationWarning)
        assert _has_ignore_filter(FutureWarning)  # e.g. torch's LeafSpec deprecation


def test_debug_flag_restores_default_warning_filters(monkeypatch):
    monkeypatch.setattr(app_mod, "generate_forecast", lambda *a, **k: _fake_forecast())
    with warnings.catch_warnings():
        warnings.resetwarnings()
        # Simulate a prior non-debug invocation having already suppressed warnings
        # in this process; --debug must actively restore them, not just "happen"
        # to leave them alone.
        warnings.filterwarnings("ignore", category=UserWarning)
        runner.invoke(app_mod.app, ["--debug", "forecast", "AAPL", "--no-chart"])
        assert not _has_ignore_filter(UserWarning)


def test_scan_signal_buy_when_whole_band_is_positive(monkeypatch):
    # p10=102, p50=105, p90=108 vs current 100: every quantile is a gain.
    fc = _fake_forecast(predictions=np.array([[102.0, 105.0, 108.0]] * 5))
    monkeypatch.setattr(app_mod, "generate_forecast", lambda *a, **k: fc)
    result = runner.invoke(app_mod.app, ["scan", "AAPL"])
    assert result.exit_code == 0
    assert "BUY" in result.stdout


def test_scan_signal_sell_when_whole_band_is_negative(monkeypatch):
    # p10=92, p50=95, p90=98 vs current 100: every quantile is a loss.
    fc = _fake_forecast(predictions=np.array([[92.0, 95.0, 98.0]] * 5))
    monkeypatch.setattr(app_mod, "generate_forecast", lambda *a, **k: fc)
    result = runner.invoke(app_mod.app, ["scan", "AAPL"])
    assert result.exit_code == 0
    assert "SELL" in result.stdout


def test_scan_signal_hold_when_band_straddles_zero(monkeypatch):
    # median +5% looks confident, but p10=95 (a loss) means the band straddles
    # zero -- must not render identically to a genuinely confident +5%.
    fc = _fake_forecast(predictions=np.array([[95.0, 105.0, 115.0]] * 5))
    monkeypatch.setattr(app_mod, "generate_forecast", lambda *a, **k: fc)
    result = runner.invoke(app_mod.app, ["scan", "AAPL"])
    assert result.exit_code == 0
    assert "HOLD" in result.stdout
    assert "BUY" not in result.stdout
    assert "SELL" not in result.stdout


def test_forecast_json_output_is_clean_parseable_json(monkeypatch):
    """`--format json` must emit valid JSON with no ANSI escape codes (PYQ-212)."""
    monkeypatch.setattr(app_mod, "generate_forecast", lambda *a, **k: _fake_forecast())

    class NoOptions:
        put_call_ratio = None

    monkeypatch.setattr(app_mod, "fetch_options_snapshot", lambda s: NoOptions())
    result = runner.invoke(app_mod.app, ["--format", "json", "forecast", "AAPL"])
    assert result.exit_code == 0
    assert "\x1b[" not in result.stdout  # no ANSI escapes
    data = json.loads(result.stdout)  # parses cleanly
    assert data["symbol"] == "AAPL"
    assert data["median"] == [105.0] * 5  # p50 column of the fake forecast


def test_scan_json_output_lists_per_symbol_records(monkeypatch):
    fc = _fake_forecast(predictions=np.array([[102.0, 105.0, 108.0]] * 5))
    monkeypatch.setattr(app_mod, "generate_forecast", lambda *a, **k: fc)
    result = runner.invoke(app_mod.app, ["--format", "json", "scan", "AAPL"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert isinstance(data, list) and data[0]["symbol"] == "AAPL"
    assert data[0]["signal"] == "BUY"


def test_cache_list_and_prune_commands(monkeypatch, tmp_path):
    """`pyquant cache` list/prune wire the helpers to the CLI (PYQ-221)."""
    from pyquant.config import Settings

    def fake_load_settings():
        s = Settings()
        s.data.cache_dir = tmp_path / "cache"
        return s

    monkeypatch.setattr(app_mod, "load_settings", fake_load_settings)

    listed = runner.invoke(app_mod.app, ["--format", "json", "cache", "list"])
    assert listed.exit_code == 0
    assert json.loads(listed.stdout)["entry_count"] == 0

    pruned = runner.invoke(app_mod.app, ["cache", "prune"])
    assert pruned.exit_code == 0
    assert "Pruned 0" in pruned.stdout


def test_train_json_output_serializes_result(monkeypatch):
    fake_result = TrainResult(
        symbols=["AAPL"],
        bundle_dir=Path("checkpoints/AAPL"),
        val_loss=0.123,
        n_features=10,
        epochs_run=5,
        evaluation=EvaluationMetrics(
            model_mae=1.5, baseline_mae=2.0, directional_accuracy=0.6, calibration_coverage=0.8
        ),
    )
    monkeypatch.setattr(app_mod.tft, "train", lambda *a, **k: fake_result)
    result = runner.invoke(app_mod.app, ["--format", "json", "train", "AAPL"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["symbols"] == ["AAPL"]
    assert data["evaluation"]["directional_accuracy"] == 0.6
