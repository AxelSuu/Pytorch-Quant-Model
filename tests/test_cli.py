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
from pyquant.models.tft import BacktestResult, TrainResult

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


# --- PYQ-117: a percentage must never be shown without its denominator -------


def _train_result_with(**metric_kwargs):
    defaults = dict(
        model_mae=1.5,
        baseline_mae=2.0,
        directional_accuracy=0.6,
        calibration_coverage=0.8,
        n_samples=56,
        n_points=280,
    )
    defaults.update(metric_kwargs)
    return TrainResult(
        symbols=["AAPL"],
        bundle_dir=Path("checkpoints/AAPL"),
        val_loss=0.123,
        n_features=10,
        epochs_run=5,
        evaluation=EvaluationMetrics(**defaults),
    )


def test_train_table_reports_the_metric_sample_size(monkeypatch):
    monkeypatch.setattr(app_mod.tft, "train", lambda *a, **k: _train_result_with())
    result = runner.invoke(app_mod.app, ["train", "AAPL"])
    assert result.exit_code == 0
    assert "56" in result.stdout and "280" in result.stdout


def test_train_json_output_includes_the_metric_sample_size(monkeypatch):
    monkeypatch.setattr(app_mod.tft, "train", lambda *a, **k: _train_result_with())
    result = runner.invoke(app_mod.app, ["--format", "json", "train", "AAPL"])
    assert result.exit_code == 0
    ev = json.loads(result.stdout)["evaluation"]
    assert ev["n_samples"] == 56
    assert ev["n_points"] == 280


# --- PYQ-122: the calibration band label must follow the configured quantiles -


def test_train_table_labels_the_calibration_band_from_configured_quantiles(monkeypatch, tmp_path):
    """With quantiles [0.05, 0.5, 0.95] the band is p5-p95, not the hardcoded p10-p90."""
    cfg = tmp_path / "wide.yaml"
    cfg.write_text("tft:\n  quantiles: [0.05, 0.5, 0.95]\n")
    monkeypatch.setattr(app_mod.tft, "train", lambda *a, **k: _train_result_with())

    result = runner.invoke(app_mod.app, ["train", "AAPL", "--config", str(cfg)])

    assert result.exit_code == 0
    assert "p5-p95" in result.stdout
    assert "p10-p90" not in result.stdout


def test_backtest_table_labels_the_calibration_band_from_configured_quantiles(
    monkeypatch, tmp_path
):
    cfg = tmp_path / "wide.yaml"
    cfg.write_text("tft:\n  quantiles: [0.05, 0.5, 0.95]\n")
    fake = BacktestResult(
        symbol="AAPL",
        n_windows=2,
        per_window=[],
        aggregated=EvaluationMetrics(1.0, 2.0, 0.5, 0.7, n_samples=2, n_points=10),
    )
    monkeypatch.setattr(app_mod.tft, "walk_forward_backtest", lambda *a, **k: fake)

    result = runner.invoke(app_mod.app, ["backtest", "AAPL", "--config", str(cfg)])

    assert result.exit_code == 0
    assert "p5-p95" in result.stdout
    assert "p10-p90" not in result.stdout


# --- PYQ-120: failures must be clean messages, not tracebacks ----------------


def test_forecast_on_untrained_symbol_exits_cleanly_without_a_traceback(monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "load_settings", lambda *a, **k: _settings_in(tmp_path))
    result = runner.invoke(app_mod.app, ["forecast", "NEVERTRAINED"])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit), f"leaked {result.exception!r}"
    assert "No trained model" in result.output
    assert "Traceback" not in result.output


def test_explain_on_untrained_symbol_exits_cleanly_without_a_traceback(monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "load_settings", lambda *a, **k: _settings_in(tmp_path))
    result = runner.invoke(app_mod.app, ["explain", "NEVERTRAINED"])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit), f"leaked {result.exception!r}"
    assert "No trained model" in result.output


def _settings_in(tmp_path):
    from pyquant.config import Settings

    s = Settings()
    s.checkpoint_dir = tmp_path / "checkpoints"
    return s


# --- PYQ-124: a crossed band must never reach display or signalling ----------


def test_forecast_table_renders_a_crossed_band_monotonically(monkeypatch):
    crossed = Forecast(
        symbol="AAPL",
        last_date=pd.Timestamp("2024-03-01"),
        current_price=100.0,
        quantiles=[0.1, 0.5, 0.9],
        predictions=np.array([[110.0, 100.0, 90.0]]),  # p90 below p10
        history=pd.Series([100.0], index=pd.bdate_range("2024-03-01", periods=1)),
    )
    table = app_mod._forecast_table(crossed)
    rendered = [
        c._cells[0] for c in table.columns if c.header.startswith("p")
    ]
    assert rendered == ["$90.00", "$100.00", "$110.00"]


# --- PYQ-125: use_options must actually gate the options fetch ---------------


def test_forecast_skips_the_options_fetch_when_use_options_is_false(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(
        app_mod, "fetch_options_snapshot", lambda s: called.append(s) or _empty_snapshot()
    )

    settings = _settings_in(tmp_path)
    settings.data.use_options = False
    monkeypatch.setattr(app_mod, "load_settings", lambda *a, **k: settings)
    monkeypatch.setattr(app_mod.tft, "load", lambda *a, **k: object())
    monkeypatch.setattr(app_mod, "generate_forecast", lambda *a, **k: _simple_forecast())

    result = runner.invoke(app_mod.app, ["forecast", "AAPL", "--no-chart"])
    assert result.exit_code == 0
    assert called == []


def _empty_snapshot():
    from pyquant.data.options import OptionsSnapshot

    return OptionsSnapshot(None, None, None, None)


def _simple_forecast():
    return Forecast(
        symbol="AAPL",
        last_date=pd.Timestamp("2024-03-01"),
        current_price=100.0,
        quantiles=[0.1, 0.5, 0.9],
        predictions=np.array([[95.0, 100.0, 105.0]]),
        history=pd.Series([100.0], index=pd.bdate_range("2024-03-01", periods=1)),
    )


# --- PYQ-126: no unreachable branch in _fmt_bytes ----------------------------


def test_fmt_bytes_formats_every_unit():
    assert app_mod._fmt_bytes(512) == "512 B"
    assert app_mod._fmt_bytes(2 * 1024) == "2.0 KB"
    assert app_mod._fmt_bytes(3 * 1024**2) == "3.0 MB"
    assert app_mod._fmt_bytes(4 * 1024**3) == "4.0 GB"
    # Beyond GB there is no larger unit: it stays GB rather than falling through.
    assert app_mod._fmt_bytes(5000 * 1024**3) == "5000.0 GB"


# --- PYQ-226: the spread across backtest windows, not just the mean ----------


def test_backtest_table_shows_per_window_spread(monkeypatch):
    from pyquant.analysis.metrics import aggregate_metrics

    per_window = [
        EvaluationMetrics(1.0, 2.0, 1.0, 1.0, n_samples=1, n_points=5),
        EvaluationMetrics(3.0, 2.0, 0.2, 0.4, n_samples=1, n_points=5),
    ]
    fake = BacktestResult(
        symbol="AAPL", n_windows=2, per_window=per_window, aggregated=aggregate_metrics(per_window)
    )
    monkeypatch.setattr(app_mod.tft, "walk_forward_backtest", lambda *a, **k: fake)

    result = runner.invoke(app_mod.app, ["backtest", "AAPL"])

    assert result.exit_code == 0
    # The mean directional accuracy is 60%; the window range 20%-100% is the point.
    assert "20.0%" in result.stdout
    assert "100.0%" in result.stdout


# --- PYQ-231: failure paths, which had zero coverage -------------------------


def test_train_with_a_missing_config_file_fails_instead_of_using_defaults(monkeypatch, tmp_path):
    """PYQ-128 via the CLI: silently training on defaults is the dangerous outcome."""
    called = []
    monkeypatch.setattr(app_mod.tft, "train", lambda *a, **k: called.append(1))
    result = runner.invoke(app_mod.app, ["train", "AAPL", "--config", str(tmp_path / "nope.yaml")])
    assert result.exit_code != 0
    assert called == [], "trained anyway despite an unusable config"


def test_invalid_output_format_is_rejected():
    result = runner.invoke(app_mod.app, ["--format", "bogus", "forecast", "AAPL"])
    assert result.exit_code != 0
    assert "rich" in result.output and "json" in result.output


def test_train_on_insufficient_history_reports_cleanly(monkeypatch, tmp_path):
    def raise_short(*a, **k):
        raise ValueError("Not enough history for TINY: need more than 80 rows, got 15.")

    monkeypatch.setattr(app_mod.tft, "train", raise_short)
    result = runner.invoke(app_mod.app, ["train", "TINY"])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit), f"leaked {result.exception!r}"
    assert "Not enough history" in result.output
    assert "Traceback" not in result.output
