"""CLI smoke tests using Typer's CliRunner (network-free, mocked forecasts)."""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from pyquant.analysis.forecast import Forecast
from pyquant.analysis.metrics import EvaluationMetrics, PerHorizonMetrics
from pyquant.cli import app as app_mod
from pyquant.data import dataset as ds_mod
from pyquant.data import options as options_mod
from pyquant.data.prices import add_technical_indicators
from pyquant.models.tft import BacktestResult, TrainResult, TuneResult

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


def test_forecast_command_forwards_as_of_to_generate_forecast(monkeypatch):
    """PYQ-284: --as-of must reach generate_forecast's `end` kwarg verbatim."""
    captured = {}

    def fake_generate_forecast(symbol, settings, bundle=None, pin=None, end=None):
        captured["end"] = end
        return _fake_forecast()

    monkeypatch.setattr(app_mod, "generate_forecast", fake_generate_forecast)

    class NoOptions:
        put_call_ratio = None

    monkeypatch.setattr(app_mod, "fetch_options_snapshot", lambda s: NoOptions())
    result = runner.invoke(app_mod.app, ["forecast", "AAPL", "--no-chart", "--as-of", "2026-07-29"])
    assert result.exit_code == 0
    assert captured["end"] == "2026-07-29"


def test_forecast_command_rejects_malformed_as_of(monkeypatch):
    monkeypatch.setattr(app_mod, "generate_forecast", lambda *a, **k: _fake_forecast())
    result = runner.invoke(app_mod.app, ["forecast", "AAPL", "--as-of", "not-a-date"])
    assert result.exit_code == 1
    assert "--as-of" in result.stdout


def test_forecast_command_rejects_as_of_combined_with_pin(monkeypatch):
    monkeypatch.setattr(app_mod, "generate_forecast", lambda *a, **k: _fake_forecast())
    result = runner.invoke(
        app_mod.app, ["forecast", "AAPL", "--as-of", "2026-07-29", "--pin", "exp-1"]
    )
    assert result.exit_code == 1
    assert "--pin" in result.stdout


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


def test_train_command_forwards_as_of_to_tft_train(monkeypatch):
    """PYQ-284: --as-of must reach tft.train's `end` kwarg verbatim, so a checkpoint
    trained "as of" a past date sees no data after it."""
    captured = {}

    def fake_train(symbols, settings, **kwargs):
        captured["end"] = kwargs.get("end")
        return TrainResult(
            symbols=symbols,
            bundle_dir=Path("checkpoints/AAPL"),
            val_loss=0.1,
            n_features=5,
            epochs_run=1,
            evaluation=EvaluationMetrics(
                model_mae=1.0, baseline_mae=1.0, directional_accuracy=0.5, calibration_coverage=0.8
            ),
        )

    monkeypatch.setattr(app_mod.tft, "train", fake_train)
    result = runner.invoke(app_mod.app, ["train", "AAPL", "--as-of", "2026-07-29"])
    assert result.exit_code == 0
    assert captured["end"] == "2026-07-29"


def test_train_command_rejects_malformed_as_of(monkeypatch):
    monkeypatch.setattr(
        app_mod.tft,
        "train",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("tft.train should not be called")),
    )
    result = runner.invoke(app_mod.app, ["train", "AAPL", "--as-of", "07/29/2026"])
    assert result.exit_code == 1
    assert "--as-of" in result.stdout


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


def test_backtest_signals_flag_reports_pnl_and_reaches_json(monkeypatch):
    """PYQ-255: --signals wires walk_forward_backtest's per-window signals through
    evaluate_signals() and into both the Rich table and --format json."""
    from pyquant.models.tft import BacktestResult

    ev = EvaluationMetrics(
        model_mae=1.2, baseline_mae=1.6, directional_accuracy=0.55, calibration_coverage=0.75
    )
    fake_result = BacktestResult(
        symbol="AAPL",
        n_windows=3,
        per_window=[ev, ev, ev],
        aggregated=ev,
        signals=["BUY", "SELL", "HOLD"],
        signal_returns_pct=[3.0, -2.0, 0.5],
    )

    captured = {}

    def fake_backtest(symbol, settings, **kwargs):
        captured["compute_signals"] = kwargs.get("compute_signals")
        return fake_result

    monkeypatch.setattr(app_mod.tft, "walk_forward_backtest", fake_backtest)

    result = runner.invoke(app_mod.app, ["backtest", "AAPL", "--signals"])
    assert result.exit_code == 0
    assert captured["compute_signals"] is True
    assert "Signal evaluation" in result.stdout
    assert "1 BUY / 1 SELL / 1 HOLD" in result.stdout

    json_result = runner.invoke(app_mod.app, ["--format", "json", "backtest", "AAPL", "--signals"])
    assert json_result.exit_code == 0
    payload = json.loads(json_result.stdout)
    assert payload["signal_evaluation"]["n_buy"] == 1
    assert "strategy_pnl_pct" in payload["signal_evaluation"]


def test_backtest_signals_flag_labels_the_uncalibrated_band(monkeypatch):
    """PYQ-149: --signals must say explicitly that its band is uncalibrated,
    both in the printed note and in --format json, rather than silently
    diverging from what scan() would show once calibration is configured."""
    from pyquant.models.tft import BacktestResult

    ev = EvaluationMetrics(
        model_mae=1.2, baseline_mae=1.6, directional_accuracy=0.55, calibration_coverage=0.75
    )
    fake_result = BacktestResult(
        symbol="AAPL",
        n_windows=1,
        per_window=[ev],
        aggregated=ev,
        signals=["BUY"],
        signal_returns_pct=[3.0],
    )
    monkeypatch.setattr(app_mod.tft, "walk_forward_backtest", lambda *a, **k: fake_result)

    result = runner.invoke(app_mod.app, ["backtest", "AAPL", "--signals"])
    assert result.exit_code == 0
    assert "uncalibrated" in result.stdout
    assert "PYQ-149" in result.stdout

    json_result = runner.invoke(app_mod.app, ["--format", "json", "backtest", "AAPL", "--signals"])
    payload = json.loads(json_result.stdout)
    assert payload["signals_calibrated"] is False


def test_backtest_without_signals_flag_skips_the_extra_pass(monkeypatch):
    from pyquant.models.tft import BacktestResult

    ev = EvaluationMetrics(
        model_mae=1.2, baseline_mae=1.6, directional_accuracy=0.55, calibration_coverage=0.75
    )
    fake_result = BacktestResult(symbol="AAPL", n_windows=1, per_window=[ev], aggregated=ev)
    captured = {}

    def fake_backtest(symbol, settings, **kwargs):
        captured["compute_signals"] = kwargs.get("compute_signals")
        return fake_result

    monkeypatch.setattr(app_mod.tft, "walk_forward_backtest", fake_backtest)
    result = runner.invoke(app_mod.app, ["backtest", "AAPL"])

    assert result.exit_code == 0
    assert captured["compute_signals"] is False
    assert "Signal evaluation" not in result.stdout


# --- PYQ-265: repeat a backtest across seeds ------------------------------------


def _seed_sweep_fixture():
    from pyquant.models.tft import BacktestResult, SeedSweepResult

    def _ev(mae):
        return EvaluationMetrics(
            model_mae=mae, baseline_mae=2.0, directional_accuracy=0.5, calibration_coverage=0.8
        )

    per_seed = [
        BacktestResult(
            symbol="AAPL", n_windows=2, per_window=[_ev(1.0)], aggregated=_ev(1.0)
        ),  # skill 0.5
        BacktestResult(
            symbol="AAPL", n_windows=2, per_window=[_ev(1.5)], aggregated=_ev(1.5)
        ),  # skill 0.25
    ]
    return SeedSweepResult(symbol="AAPL", seeds=[0, 1], per_seed=per_seed)


def test_backtest_without_seeds_flag_uses_the_single_run_path_unchanged(monkeypatch):
    """`--seeds` defaults to 1, which must reach `walk_forward_backtest`, not
    the multi-seed path -- existing behaviour is unchanged (PYQ-265)."""
    from pyquant.models.tft import BacktestResult

    ev = EvaluationMetrics(
        model_mae=1.0, baseline_mae=2.0, directional_accuracy=0.5, calibration_coverage=0.8
    )
    fake_result = BacktestResult(symbol="AAPL", n_windows=1, per_window=[ev], aggregated=ev)
    monkeypatch.setattr(app_mod.tft, "walk_forward_backtest", lambda *a, **k: fake_result)

    def fail_if_called(*a, **k):
        raise AssertionError("multi-seed path must not run when --seeds is not passed")

    monkeypatch.setattr(app_mod.tft, "walk_forward_backtest_multi_seed", fail_if_called)

    result = runner.invoke(app_mod.app, ["backtest", "AAPL"])
    assert result.exit_code == 0


def test_backtest_seeds_flag_expands_to_a_deterministic_sequence_and_reports_summary_stats(
    monkeypatch,
):
    fake_sweep = _seed_sweep_fixture()
    captured = {}

    def fake_multi_seed(symbol, settings, **kwargs):
        captured["seeds"] = kwargs.get("seeds")
        return fake_sweep

    monkeypatch.setattr(app_mod.tft, "walk_forward_backtest_multi_seed", fake_multi_seed)

    result = runner.invoke(app_mod.app, ["backtest", "AAPL", "--seeds", "2"])

    assert result.exit_code == 0
    assert captured["seeds"] == [0, 1]  # "the first N of a deterministic sequence"
    assert "Per-seed results" in result.stdout
    assert "+37.5%" in result.stdout  # mean of skills 0.5 and 0.25


def test_backtest_seeds_flag_json_output_includes_summary_and_per_seed(monkeypatch):
    fake_sweep = _seed_sweep_fixture()
    monkeypatch.setattr(app_mod.tft, "walk_forward_backtest_multi_seed", lambda *a, **k: fake_sweep)

    result = runner.invoke(app_mod.app, ["--format", "json", "backtest", "AAPL", "--seeds", "2"])

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["seeds"] == [0, 1]
    assert len(data["per_seed"]) == 2
    assert data["skill_mean"] == pytest.approx(0.375)
    assert data["skill_min"] == pytest.approx(0.25)
    assert data["skill_max"] == pytest.approx(0.5)


def test_tune_command_reports_the_held_out_score_not_the_in_search_value(monkeypatch, tmp_path):
    """PYQ-253: the in-search value and the held-out evaluation are different numbers
    for a reason -- both must reach the user, clearly distinguished."""
    ev = EvaluationMetrics(
        model_mae=1.0, baseline_mae=1.2, directional_accuracy=0.6, calibration_coverage=0.8
    )
    fake_result = TuneResult(
        symbol="AAPL",
        n_trials=5,
        best_params={"hidden_size": 32, "learning_rate": 0.01},
        best_value=0.1234,
        held_out_evaluation=ev,
        bundle_dir=tmp_path / "AAPL_TUNED",
        config_path=tmp_path / "aapl_tuned.yaml",
    )
    captured = {}

    def fake_tune(symbol, settings, **kwargs):
        captured.update(kwargs)
        return fake_result

    monkeypatch.setattr(app_mod.tft, "tune", fake_tune)

    result = runner.invoke(app_mod.app, ["tune", "AAPL", "--trials", "5"])

    assert result.exit_code == 0
    assert captured["n_trials"] == 5
    assert "0.1234" in result.stdout  # in-search value shown...
    assert "held-out" in result.stdout.lower()  # ...clearly labelled apart from it
    assert "60.0%" in result.stdout  # the held-out directional accuracy


def test_tune_command_json_output(monkeypatch, tmp_path):
    ev = EvaluationMetrics(
        model_mae=1.0, baseline_mae=1.2, directional_accuracy=0.6, calibration_coverage=0.8
    )
    fake_result = TuneResult(
        symbol="AAPL",
        n_trials=3,
        best_params={"hidden_size": 32},
        best_value=0.5,
        held_out_evaluation=ev,
        bundle_dir=tmp_path / "AAPL_TUNED",
        config_path=tmp_path / "aapl_tuned.yaml",
    )
    monkeypatch.setattr(app_mod.tft, "tune", lambda symbol, settings, **kwargs: fake_result)

    result = runner.invoke(app_mod.app, ["--format", "json", "tune", "AAPL"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["best_params"] == {"hidden_size": 32}
    assert payload["held_out_evaluation"]["directional_accuracy"] == 0.6


def test_tune_command_reports_a_missing_extra_clearly(monkeypatch):
    def raise_missing_extra(symbol, settings, **kwargs):
        raise ImportError("pyquant tune needs the 'tuning' extra: uv sync --extra tuning")

    monkeypatch.setattr(app_mod.tft, "tune", raise_missing_extra)
    result = runner.invoke(app_mod.app, ["tune", "AAPL"])
    assert result.exit_code == 1
    assert "tuning" in result.output


# --- PYQ-268: multi-symbol sweep harness ---------------------------------------


def _fake_sweep_result():
    from pyquant.experiments.sweep import SweepCell, SweepResult
    from pyquant.models.tft import BacktestResult

    def _ev(mae):
        return EvaluationMetrics(
            model_mae=mae, baseline_mae=2.0, directional_accuracy=0.5, calibration_coverage=0.8
        )

    def _result(symbol, mae):
        ev = _ev(mae)
        return BacktestResult(
            symbol=symbol, n_windows=1, per_window=[ev], aggregated=ev, origins=[100]
        )

    cells = [
        SweepCell("AAA", "close", result=_result("AAA", 1.0)),  # skill 0.5
        SweepCell("AAA", "log_return", result=_result("AAA", 0.0)),  # skill 1.0
        SweepCell("BBB", "close", result=_result("BBB", 1.5)),  # skill 0.25
        SweepCell("BBB", "log_return", error="not enough history"),
    ]
    return SweepResult(symbols=["AAA", "BBB"], arm_names=["close", "log_return"], cells=cells)


def test_sweep_command_reports_the_cell_matrix_and_pooled_skill(monkeypatch):
    captured = {}

    def fake_run_sweep(symbols, arms, settings, **kwargs):
        captured["symbols"] = symbols
        captured["arm_specs"] = [(a.name, a.overrides) for a in arms]
        return _fake_sweep_result()

    monkeypatch.setattr(app_mod, "run_sweep", fake_run_sweep)

    result = runner.invoke(
        app_mod.app,
        ["sweep", "--symbols", "aaa,bbb", "--arm", "target=close", "--arm", "target=log_return"],
    )

    assert result.exit_code == 0
    assert captured["symbols"] == ["AAA", "BBB"]
    assert captured["arm_specs"] == [
        ("target=close", {"target": "close"}),
        ("target=log_return", {"target": "log_return"}),
    ]
    assert "failed" in result.stdout  # BBB/log_return's recorded gap
    assert "scored higher" in result.stdout  # helped-summary line


def test_sweep_command_json_output_includes_the_full_cell_matrix(monkeypatch):
    monkeypatch.setattr(app_mod, "run_sweep", lambda *a, **k: _fake_sweep_result())

    result = runner.invoke(
        app_mod.app,
        [
            "--format",
            "json",
            "sweep",
            "--symbols",
            "AAA,BBB",
            "--arm",
            "target=close",
            "--arm",
            "target=log_return",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["symbols"] == ["AAA", "BBB"]
    assert data["arms"] == ["close", "log_return"]
    assert len(data["cells"]) == 4
    failed_cell = next(
        c for c in data["cells"] if c["symbol"] == "BBB" and c["arm"] == "log_return"
    )
    assert failed_cell["result"] is None
    assert failed_cell["error"] == "not enough history"
    assert data["pooled_skill"]["close"] == pytest.approx((0.5 + 0.25) / 2)
    assert data["pooled_skill"]["log_return"] == pytest.approx(1.0)  # BBB failed, only AAA counts


def test_sweep_command_rejects_a_malformed_arm_spec(monkeypatch):
    monkeypatch.setattr(app_mod, "run_sweep", lambda *a, **k: _fake_sweep_result())

    result = runner.invoke(
        app_mod.app, ["sweep", "--symbols", "AAA", "--arm", "not-key-equals-value"]
    )

    assert result.exit_code == 1
    assert "key=value" in result.output


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


def test_snapshot_command_records_to_the_configured_history_dir(monkeypatch, tmp_path):
    """PYQ-254: `pyquant snapshot SYMBOL` -- the only way to ever accumulate a
    historical options-implied series, since yfinance exposes only a current chain.
    """
    from pyquant.config import Settings
    from pyquant.data.options import OptionsSnapshot

    def fake_load_settings():
        s = Settings()
        s.options_history_dir = tmp_path / "options_history"
        return s

    monkeypatch.setattr(app_mod, "load_settings", fake_load_settings)
    # append_snapshot() calls fetch_options_snapshot() from within options.py's own
    # namespace, not app_mod's imported reference -- that one is only used by the
    # `forecast` command's display snapshot.
    monkeypatch.setattr(
        options_mod,
        "fetch_options_snapshot",
        lambda symbol: OptionsSnapshot(
            put_call_ratio=1.1, atm_iv=0.3, iv_skew=0.02, expiry="2024-06-21"
        ),
    )

    result = runner.invoke(app_mod.app, ["snapshot", "aapl"])

    assert result.exit_code == 0
    written = tmp_path / "options_history" / "AAPL.jsonl"
    assert written.exists()
    assert json.loads(written.read_text().splitlines()[0])["put_call_ratio"] == 1.1


def test_snapshot_command_json_output(monkeypatch, tmp_path):
    from pyquant.config import Settings
    from pyquant.data.options import OptionsSnapshot

    def fake_load_settings():
        s = Settings()
        s.options_history_dir = tmp_path / "options_history"
        return s

    monkeypatch.setattr(app_mod, "load_settings", fake_load_settings)
    monkeypatch.setattr(
        options_mod,
        "fetch_options_snapshot",
        lambda symbol: OptionsSnapshot(None, None, None, None),
    )

    result = runner.invoke(app_mod.app, ["--format", "json", "snapshot", "AAPL"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["symbol"] == "AAPL"
    assert "path" in payload


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


# --- PYQ-267: per-horizon-step breakdown --------------------------------------


def test_train_json_output_includes_per_horizon_breakdown(monkeypatch):
    monkeypatch.setattr(
        app_mod.tft,
        "train",
        lambda *a, **k: _train_result_with(
            per_horizon=[
                PerHorizonMetrics(
                    1,
                    model_mae=1.0,
                    baseline_mae=2.0,
                    directional_accuracy=0.6,
                    calibration_coverage=0.8,
                ),
                PerHorizonMetrics(
                    2,
                    model_mae=1.5,
                    baseline_mae=1.5,
                    directional_accuracy=0.5,
                    calibration_coverage=0.7,
                ),
            ]
        ),
    )
    result = runner.invoke(app_mod.app, ["--format", "json", "train", "AAPL"])
    assert result.exit_code == 0
    per_horizon = json.loads(result.stdout)["evaluation"]["per_horizon"]
    assert [step["step"] for step in per_horizon] == [1, 2]
    assert per_horizon[0]["skill_vs_baseline"] == pytest.approx(0.5)
    assert per_horizon[1]["skill_vs_baseline"] == pytest.approx(0.0)


def test_train_table_shows_a_per_horizon_breakdown_when_horizon_exceeds_one(monkeypatch):
    monkeypatch.setattr(
        app_mod.tft,
        "train",
        lambda *a, **k: _train_result_with(
            per_horizon=[
                PerHorizonMetrics(
                    1,
                    model_mae=1.0,
                    baseline_mae=2.0,
                    directional_accuracy=0.6,
                    calibration_coverage=0.8,
                ),
                PerHorizonMetrics(
                    2,
                    model_mae=1.5,
                    baseline_mae=1.5,
                    directional_accuracy=0.5,
                    calibration_coverage=0.7,
                ),
            ]
        ),
    )
    result = runner.invoke(app_mod.app, ["train", "AAPL"])
    assert result.exit_code == 0
    assert "Per-horizon" in result.stdout
    assert "+50.0%" in result.stdout  # step 1's skill


# --- PYQ-275: baselines beyond persistence -------------------------------------


def test_train_table_names_the_strongest_baseline_and_its_skill(monkeypatch):
    monkeypatch.setattr(
        app_mod.tft,
        "train",
        lambda *a, **k: _train_result_with(
            model_mae=5.0,
            baseline_mae=10.0,  # persistence
            baseline_maes={"persistence": 10.0, "seasonal_naive": 4.0, "ar1": 8.0},
        ),
    )
    result = runner.invoke(app_mod.app, ["train", "AAPL"])
    assert result.exit_code == 0
    assert "seasonal_naive" in result.stdout
    assert "ar1" in result.stdout
    # skill vs. the strongest (seasonal_naive, mae 4.0): (4-5)/4 = -25.0%
    assert "-25.0%" in result.stdout


def test_train_table_omits_the_strongest_baseline_row_without_extra_baselines(monkeypatch):
    """A bundle scored without `history` (PYQ-275 is additive) only ever has the
    "persistence" entry -- no strongest-baseline row to show."""
    monkeypatch.setattr(
        app_mod.tft,
        "train",
        lambda *a, **k: _train_result_with(baseline_maes={"persistence": 2.0}),
    )
    result = runner.invoke(app_mod.app, ["train", "AAPL"])
    assert result.exit_code == 0
    assert "Skill vs. strongest baseline" not in result.stdout


def test_train_json_output_includes_baseline_maes_and_strongest_baseline(monkeypatch):
    monkeypatch.setattr(
        app_mod.tft,
        "train",
        lambda *a, **k: _train_result_with(
            model_mae=5.0,
            baseline_mae=10.0,
            baseline_maes={"persistence": 10.0, "seasonal_naive": 4.0},
        ),
    )
    result = runner.invoke(app_mod.app, ["--format", "json", "train", "AAPL"])
    assert result.exit_code == 0
    ev = json.loads(result.stdout)["evaluation"]
    assert ev["baseline_maes"] == {"persistence": 10.0, "seasonal_naive": 4.0}
    assert ev["strongest_baseline"] == {"name": "seasonal_naive", "mae": 4.0}
    assert ev["skill_vs_strongest_baseline"] == pytest.approx((4.0 - 5.0) / 4.0)


def test_train_table_still_shows_crps_and_winkler_rows_at_exactly_zero(monkeypatch):
    """PYQ-156: `if ev.crps:`/`if ev.winkler_score:` were bare truthy checks, so a
    legitimate 0.0 (e.g. a degenerate/perfect band) silently dropped the row."""
    monkeypatch.setattr(
        app_mod.tft, "train", lambda *a, **k: _train_result_with(crps=0.0, winkler_score=0.0)
    )
    result = runner.invoke(app_mod.app, ["train", "AAPL"])
    assert result.exit_code == 0
    assert "CRPS" in result.stdout
    assert "Winkler" in result.stdout


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


def _fake_interpretation(bundle_skill):
    from pyquant.analysis.interpret import Interpretation

    dates = pd.bdate_range("2024-01-01", periods=5)
    return Interpretation(
        symbol="AAPL",
        feature_importance={"RSI_14": 0.6, "SMA_10": 0.4},
        attention=np.array([0.1, 0.2, 0.3, 0.2, 0.2]),
        panel_index=dates,
        bundle_skill=bundle_skill,
    )


def test_explain_warns_when_the_bundle_does_not_beat_the_baseline(monkeypatch):
    """PYQ-314: an interpretation of a model that does not beat persistence describes
    what it attends to, not what moves the price -- explain must say so."""
    monkeypatch.setattr(app_mod, "explain_forecast", lambda *a, **k: _fake_interpretation(-0.05))
    monkeypatch.setattr(app_mod.tft, "load", lambda *a, **k: object())

    result = runner.invoke(app_mod.app, ["explain", "AAPL", "--no-chart"])

    assert result.exit_code == 0
    assert "skill vs. persistence" in result.stdout
    assert "-5.0%" in result.stdout


def test_explain_stays_quiet_when_the_bundle_beats_the_baseline(monkeypatch):
    monkeypatch.setattr(app_mod, "explain_forecast", lambda *a, **k: _fake_interpretation(0.12))
    monkeypatch.setattr(app_mod.tft, "load", lambda *a, **k: object())

    result = runner.invoke(app_mod.app, ["explain", "AAPL", "--no-chart"])

    assert result.exit_code == 0
    assert "skill vs. persistence" not in result.stdout


def test_explain_json_carries_bundle_skill(monkeypatch):
    monkeypatch.setattr(app_mod, "explain_forecast", lambda *a, **k: _fake_interpretation(0.12))
    monkeypatch.setattr(app_mod.tft, "load", lambda *a, **k: object())

    result = runner.invoke(app_mod.app, ["--format", "json", "explain", "AAPL", "--no-chart"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["bundle_skill"] == 0.12


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
    rendered = [c._cells[0] for c in table.columns if c.header.startswith("p")]
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


# --- PYQ-277: `--provider` actually reaches DataConfig.price_provider --------


def test_build_settings_defaults_to_yfinance_when_provider_is_not_passed(monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "load_settings", lambda *a, **k: _settings_in(tmp_path))
    settings = app_mod._build_settings(None, False, False, False, provider=None)
    assert settings.data.price_provider == "yfinance"


def test_build_settings_applies_a_valid_provider_override(monkeypatch, tmp_path):
    monkeypatch.setattr(app_mod, "load_settings", lambda *a, **k: _settings_in(tmp_path))
    settings = app_mod._build_settings(None, False, False, False, provider="tiingo")
    assert settings.data.price_provider == "tiingo"


def test_build_settings_rejects_an_unknown_provider_with_a_clean_message(monkeypatch, tmp_path):
    """PriceProviderError is a RuntimeError, not one of EXPECTED_FAILURES, so an
    unknown name must be rejected here rather than surface as an uncaught
    traceback from inside build_panel (PYQ-120's convention)."""
    monkeypatch.setattr(app_mod, "load_settings", lambda *a, **k: _settings_in(tmp_path))
    with pytest.raises(ValueError, match="tiingo"):
        app_mod._build_settings(None, False, False, False, provider="polygon")


def test_train_command_threads_provider_through_to_build_panel(monkeypatch, tmp_path):
    """End-to-end through the CLI, not just `_build_settings` in isolation:
    `pyquant train --provider tiingo` must reach the price fetch, not just
    the resolved Settings object."""
    seen = {}
    settings = _settings_in(tmp_path)
    monkeypatch.setattr(app_mod, "load_settings", lambda *a, **k: settings)

    def fake_train(symbols, settings, **kwargs):
        seen["price_provider"] = settings.data.price_provider
        return TrainResult(
            symbols=symbols,
            bundle_dir=Path("checkpoints/AAPL"),
            val_loss=0.1,
            n_features=5,
            epochs_run=1,
            evaluation=EvaluationMetrics(
                model_mae=1.0, baseline_mae=1.0, directional_accuracy=0.5, calibration_coverage=0.8
            ),
        )

    monkeypatch.setattr(app_mod.tft, "train", fake_train)

    result = runner.invoke(app_mod.app, ["train", "AAPL", "--provider", "tiingo"])

    assert result.exit_code == 0
    assert seen["price_provider"] == "tiingo"


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


def test_backtest_with_zero_windows_reports_cleanly(monkeypatch):
    """PYQ-156: `--windows 0` reaches `aggregate_metrics([])`, which used to raise a
    bare ZeroDivisionError -- not in EXPECTED_FAILURES, so it leaked a traceback.
    aggregate_metrics() now raises ValueError itself on an empty list; simulate
    what walk_forward_backtest(n_windows=0) drives it to raise."""

    def raise_empty_aggregate(*a, **k):
        from pyquant.analysis.metrics import aggregate_metrics

        aggregate_metrics([])

    monkeypatch.setattr(app_mod.tft, "walk_forward_backtest", raise_empty_aggregate)
    result = runner.invoke(app_mod.app, ["backtest", "AAPL", "--windows", "0"])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit), f"leaked {result.exception!r}"
    assert "Traceback" not in result.output


# --- PYQ-263: `pyquant doctor` ------------------------------------------------


def _bundle(checkpoint_dir, name, features, *, data_cfg=None, target="Close"):
    """Write a minimal meta.json bundle the way train() would."""
    d = checkpoint_dir / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(
        json.dumps(
            {
                "symbol": name,
                "symbols": [name],
                "trained_at": "2026-07-27T10:00:00",
                "features": features,
                "target": target,
                "config": {"data": data_cfg or {}},
            }
        )
    )
    return d


def _doctor_settings(monkeypatch, tmp_path, **overrides):
    from pyquant.config import Settings

    def fake_load_settings(*a, **k):
        s = Settings()
        s.checkpoint_dir = tmp_path / "checkpoints"
        s.data.cache_dir = tmp_path / "cache"
        # Settings() reads the developer's real .env, so a machine with keys
        # configured would answer differently from CI. Start from "nothing set"
        # and let each test opt in.
        s.fred_api_key = None
        s.finnhub_api_key = None
        for key, value in overrides.items():
            target, _, attr = key.rpartition("__")
            setattr(getattr(s, target) if target else s, attr, value)
        return s

    monkeypatch.setattr(app_mod, "load_settings", fake_load_settings)
    return fake_load_settings()


def test_doctor_reports_key_presence_without_ever_printing_a_value(monkeypatch, tmp_path):
    """Secrets never enter logs or output -- presence only."""
    _doctor_settings(monkeypatch, tmp_path, fred_api_key="super-secret-value")
    (tmp_path / "checkpoints").mkdir()

    result = runner.invoke(app_mod.app, ["--format", "json", "doctor"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["keys"]["FRED_API_KEY"] is True
    assert payload["keys"]["FINNHUB_API_KEY"] is False
    assert "super-secret-value" not in result.stdout


def test_doctor_exits_non_zero_when_a_bundles_schema_can_no_longer_be_satisfied(
    monkeypatch, tmp_path
):
    """The genuinely useful check (PYQ-263): a bundle trained with sentiment
    cannot be served with sentiment switched off -- the PYQ-118 mismatch, found
    by asking rather than by a forecast blowing up later."""
    _doctor_settings(monkeypatch, tmp_path, data__use_sentiment=False)
    _bundle(tmp_path / "checkpoints", "AAPL", ["RSI_14", "Sentiment", "HeadlineCount"])

    result = runner.invoke(app_mod.app, ["doctor"])

    assert result.exit_code == 1
    assert "AAPL" in result.stdout


def test_doctor_is_healthy_when_every_bundle_can_still_be_built(monkeypatch, tmp_path):
    _doctor_settings(monkeypatch, tmp_path)
    _bundle(tmp_path / "checkpoints", "MSFT", ["RSI_14", "SMA_50", "MACD"])

    result = runner.invoke(app_mod.app, ["--format", "json", "doctor"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["healthy"] is True
    assert payload["bundles"][0]["name"] == "MSFT"
    assert payload["bundles"][0]["n_features"] == 3


def test_doctor_flags_a_bundle_needing_a_key_that_is_not_set(monkeypatch, tmp_path):
    """FRED-derived features are unbuildable without the key, whatever the toggle says."""
    _doctor_settings(monkeypatch, tmp_path, fred_api_key=None)
    _bundle(tmp_path / "checkpoints", "NVDA", ["RSI_14", "FedFunds", "CPI"])

    result = runner.invoke(app_mod.app, ["--format", "json", "doctor"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["healthy"] is False
    assert "FRED_API_KEY" in payload["bundles"][0]["problem"]


def test_doctor_succeeds_with_no_bundles_at_all(monkeypatch, tmp_path):
    """A fresh install is healthy, not broken."""
    _doctor_settings(monkeypatch, tmp_path)

    result = runner.invoke(app_mod.app, ["doctor"])

    assert result.exit_code == 0
    assert "No trained bundles" in result.stdout


def test_full_cli_journey_across_every_command_and_both_output_formats(
    monkeypatch, tmp_path, sample_ohlcv_df
):
    """PYQ-241: train -> forecast -> explain -> scan -> backtest -> cache list, using
    each command's *real* output as the next command's input, run once per --format.

    Every other CLI test in this file starts from a mocked mid-state (a fake
    Forecast, a fake TrainResult). That's the right tool for testing one command in
    isolation, but it cannot catch a write-side/read-side contract break -- a bundle
    `train` writes that `forecast` can't read, a field `explain` expects that `train`
    stopped writing (PYQ-119 was exactly this class of bug, found by reasoning rather
    than a test). Here only the network boundary (fetch_prices) and the
    options-snapshot call are stubbed; everything else -- the bundle on disk, the
    dataset params, meta.json, the cache dir -- is the real thing, isolated to a temp
    PYQUANT_HOME.
    """
    monkeypatch.setenv("PYQUANT_HOME", str(tmp_path))
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)

    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    monkeypatch.setattr(ds_mod, "fetch_prices", lambda *a, **k: panel)

    class NoOptions:
        put_call_ratio = None

    monkeypatch.setattr(app_mod, "fetch_options_snapshot", lambda s: NoOptions())

    # A real YAML experiment config (PYQ-209), not a mocked Settings object -- CLI
    # commands other than train/backtest have no settings-injection point at all,
    # so keeping the model/encoder tiny has to go through the same front door a
    # user would use.
    config = tmp_path / "tiny.yaml"
    config.write_text(
        "tft:\n  hidden_size: 8\n  hidden_continuous_size: 4\n"
        "training:\n  max_encoder_length: 15\n  max_prediction_length: 3\n"
        "  batch_size: 32\n  max_epochs: 1\n  validation_days: 20\n"
        "data:\n  use_macro: false\n  use_sentiment: false\n  use_sectors: false\n"
    )

    for fmt in ("rich", "json"):
        args = ["--format", fmt]

        result = runner.invoke(app_mod.app, [*args, "train", "TEST", "--config", str(config)])
        assert result.exit_code == 0, result.output
        if fmt == "json":
            payload = json.loads(result.stdout)
            for key in (
                "symbols",
                "bundle_dir",
                "val_loss",
                "n_features",
                "epochs_run",
                "evaluation",
            ):
                assert key in payload, f"train JSON missing {key!r}: {payload}"
            assert payload["symbols"] == ["TEST"]
        else:
            assert "Training complete" in result.stdout

        result = runner.invoke(app_mod.app, [*args, "forecast", "TEST", "--no-chart"])
        assert result.exit_code == 0, result.output
        if fmt == "json":
            payload = json.loads(result.stdout)
            for key in (
                "symbol",
                "last_date",
                "current_price",
                "horizon",
                "forecast_dates",
                "quantiles",
                "predictions",
                "n_quantile_crossings",
            ):
                assert key in payload, f"forecast JSON missing {key!r}: {payload}"
            assert len(payload["forecast_dates"]) == payload["horizon"]
        else:
            assert "forecast" in result.stdout.lower()

        result = runner.invoke(app_mod.app, [*args, "explain", "TEST", "--no-chart"])
        assert result.exit_code == 0, result.output
        if fmt == "json":
            payload = json.loads(result.stdout)
            for key in ("symbol", "feature_importance", "attention"):
                assert key in payload, f"explain JSON missing {key!r}: {payload}"
            assert payload["feature_importance"], "no features reported"
        else:
            assert "importance" in result.stdout.lower()

        result = runner.invoke(app_mod.app, [*args, "scan", "TEST"])
        assert result.exit_code == 0, result.output
        if fmt == "json":
            rows = json.loads(result.stdout)
            assert rows and rows[0]["symbol"] == "TEST"
            assert rows[0]["status"] == "ok", rows
            for key in (
                "current_price",
                "median_target",
                "expected_return_pct",
                "band_width_pct",
                "signal",
            ):
                assert key in rows[0], f"scan JSON missing {key!r}: {rows[0]}"
        else:
            assert "TEST" in result.stdout

        result = runner.invoke(
            app_mod.app, [*args, "backtest", "TEST", "--windows", "2", "--config", str(config)]
        )
        assert result.exit_code == 0, result.output
        if fmt == "json":
            payload = json.loads(result.stdout)
            for key in ("symbol", "n_windows", "aggregated", "per_window"):
                assert key in payload, f"backtest JSON missing {key!r}: {payload}"
            assert payload["n_windows"] == 2
            assert len(payload["per_window"]) == 2
        else:
            assert "backtest" in result.stdout.lower()

        result = runner.invoke(app_mod.app, [*args, "cache", "list"])
        assert result.exit_code == 0, result.output
        if fmt == "json":
            payload = json.loads(result.stdout)
            for key in ("entry_count", "total_bytes", "pins"):
                assert key in payload, f"cache list JSON missing {key!r}: {payload}"
        else:
            assert "Cache dir" in result.stdout


def test_keys_create_shows_the_raw_key_exactly_once(tmp_path, monkeypatch):
    """PYQ-281's acceptance criterion: `pyquant keys create` issues a key whose
    raw value is shown exactly once. Confirmed here by checking it's present in
    `create`'s own output and, separately, that `keys list` never prints it."""
    monkeypatch.setenv("PYQUANT_API_KEYS_DB", str(tmp_path / "api_keys.db"))

    created = runner.invoke(app_mod.app, ["keys", "create", "--name", "ci-bot"])
    assert created.exit_code == 0, created.output
    assert "pq_live_" in created.stdout

    raw_key = next(line.strip() for line in created.stdout.splitlines() if "pq_live_" in line)

    listed = runner.invoke(app_mod.app, ["keys", "list"])
    assert listed.exit_code == 0, listed.output
    assert raw_key not in listed.stdout
    assert "ci-bot" in listed.stdout


def test_keys_create_json_output_includes_the_raw_key(tmp_path, monkeypatch):
    monkeypatch.setenv("PYQUANT_API_KEYS_DB", str(tmp_path / "api_keys.db"))
    result = runner.invoke(
        app_mod.app,
        ["--format", "json", "keys", "create", "--name", "ci-bot", "--scopes", "read,train"],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["key"].startswith("pq_live_")
    assert sorted(data["scopes"]) == ["read", "train"]


def test_keys_create_rejects_an_unknown_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("PYQUANT_API_KEYS_DB", str(tmp_path / "api_keys.db"))
    result = runner.invoke(app_mod.app, ["keys", "create", "--name", "ci-bot", "--scopes", "admin"])
    assert result.exit_code != 0
    assert "scope" in result.output.lower()


def test_keys_revoke_rejects_the_key_on_a_second_authenticate(tmp_path, monkeypatch):
    monkeypatch.setenv("PYQUANT_API_KEYS_DB", str(tmp_path / "api_keys.db"))
    from pyquant.api import keystore

    db_path = keystore.resolve_db_path()
    raw_key, record = keystore.create_key(db_path, "ci-bot", ["read"])
    assert keystore.authenticate(db_path, raw_key) is not None

    result = runner.invoke(app_mod.app, ["keys", "revoke", record.id])
    assert result.exit_code == 0, result.output
    assert "Revoked" in result.stdout
    assert keystore.authenticate(db_path, raw_key) is None


def test_keys_revoke_reports_a_missing_id_without_erroring(tmp_path, monkeypatch):
    monkeypatch.setenv("PYQUANT_API_KEYS_DB", str(tmp_path / "api_keys.db"))
    result = runner.invoke(app_mod.app, ["keys", "revoke", "does-not-exist"])
    assert result.exit_code == 0, result.output
    assert "No active key" in result.stdout


def test_keys_list_with_no_keys_yet_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PYQUANT_API_KEYS_DB", str(tmp_path / "api_keys.db"))
    result = runner.invoke(app_mod.app, ["keys", "list"])
    assert result.exit_code == 0, result.output
    assert "No API keys" in result.stdout
