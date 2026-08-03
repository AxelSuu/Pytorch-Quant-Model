"""Tests for analysis/serialize.py -- the machine-readable contract behind
``--format json``, ``meta.json`` and every pyquant/api/ response body (PYQ-272).

A silent key rename here breaks downstream consumers with nothing failing, so
every serializer gets an explicit key-set assertion (fails loudly on a rename
or an accidental removal) plus a round-trip through ``json.dumps`` (fails
loudly on a numpy scalar/array leaking through, which ``json`` cannot encode).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pyquant.analysis.forecast import Forecast
from pyquant.analysis.interpret import Interpretation
from pyquant.analysis.metrics import EvaluationMetrics
from pyquant.analysis.serialize import (
    backtest_to_dict,
    evaluation_to_dict,
    forecast_to_dict,
    interpretation_to_dict,
    scan_row_to_dict,
    seed_sweep_to_dict,
    signal_evaluation_to_dict,
    sweep_result_to_dict,
    train_result_to_dict,
)
from pyquant.analysis.signals import SignalEvaluation
from pyquant.experiments.sweep import SweepCell, SweepResult
from pyquant.models.tft import BacktestResult, SeedSweepResult, TrainResult


def _ev(model_mae=1.0, baseline_mae=2.0, n_samples=5, n_points=25):
    return EvaluationMetrics(
        model_mae=model_mae,
        baseline_mae=baseline_mae,
        directional_accuracy=0.55,
        calibration_coverage=0.8,
        n_samples=n_samples,
        n_points=n_points,
        quantile_exceedance={0.1: 0.05, 0.9: 0.05},
        pinball_losses={0.1: 0.3, 0.5: 0.5, 0.9: 0.3},
    )


def _backtest_result(symbol="AAPL", origins=(100, 105)):
    ev = _ev()
    return BacktestResult(
        symbol=symbol,
        n_windows=len(origins),
        per_window=[ev, ev],
        aggregated=ev,
        origins=list(origins),
    )


def _forecast():
    dates = pd.bdate_range("2024-01-01", periods=20)
    return Forecast(
        symbol="TEST",
        last_date=dates[-1],
        current_price=110.0,
        quantiles=[0.1, 0.5, 0.9],
        predictions=np.array([[98.0, 105.0, 112.0], [97.0, 106.0, 114.0]]),
        history=pd.Series(np.linspace(90, 110, 20), index=dates),
    )


def _interpretation():
    dates = pd.bdate_range("2024-01-01", periods=5)
    return Interpretation(
        symbol="TEST",
        feature_importance={"Close": 0.6, "RSI_14": 0.4},
        attention=np.array([0.1, 0.2, 0.3, 0.2, 0.2]),
        panel_index=dates,
        bundle_skill=0.05,
    )


def _signal_evaluation():
    return SignalEvaluation(
        n_buy=3,
        n_sell=2,
        n_hold=1,
        hit_rate_buy=0.66,
        hit_rate_sell=0.5,
        avg_return_buy_pct=1.2,
        avg_return_sell_pct=-0.8,
        turnover=0.4,
        strategy_pnl_pct=3.1,
        buy_and_hold_pnl_pct=2.0,
        cost_bps=5.0,
        n_periods=6,
    )


def _assert_json_round_trips(d: dict) -> None:
    """Every serializer's output must be directly `json.dumps`-able -- a
    lingering numpy scalar/array (rather than a plain float/list) would raise
    here instead of failing silently downstream."""
    json.dumps(d)


# --- evaluation_to_dict -------------------------------------------------------


def test_evaluation_to_dict_keys_are_stable():
    d = evaluation_to_dict(_ev())
    assert set(d) == {
        "model_mae",
        "baseline_mae",
        "skill_vs_baseline",
        "directional_accuracy",
        "calibration_coverage",
        "quantile_exceedance",
        "pinball_losses",
        "crps",
        "winkler_score",
        "pit",
        "n_samples",
        "n_points",
        "effective_n_samples",
        "per_horizon",
        "baseline_maes",
        "strongest_baseline",
        "skill_vs_strongest_baseline",
    }
    _assert_json_round_trips(d)


def test_evaluation_to_dict_keeps_every_rate_next_to_its_sample_size():
    """PYQ-117: a rate reported without its denominator is unreadable."""
    d = evaluation_to_dict(_ev(n_samples=5, n_points=25))
    assert d["n_samples"] == 5
    assert d["n_points"] == 25


# --- forecast_to_dict ----------------------------------------------------------


def test_forecast_to_dict_keys_are_stable():
    d = forecast_to_dict(_forecast())
    assert set(d) == {
        "symbol",
        "last_date",
        "current_price",
        "horizon",
        "forecast_dates",
        "quantiles",
        "predictions",
        "n_quantile_crossings",
        "median",
        "expected_return_pct",
    }
    _assert_json_round_trips(d)


def test_forecast_to_dict_omits_median_when_0_5_is_not_configured():
    """PYQ-106: median/expected_return_pct require 0.5 among the quantiles."""
    fc = _forecast()
    fc.quantiles = [0.1, 0.9]
    d = forecast_to_dict(fc)
    assert "median" not in d
    assert "expected_return_pct" not in d


def test_forecast_to_dict_dates_equal_the_forecasts_own_dates():
    """Invariant 8: the dates in the JSON must be the same set the table/PNG use."""
    fc = _forecast()
    d = forecast_to_dict(fc)
    assert d["forecast_dates"] == [dt.date().isoformat() for dt in fc.forecast_dates]


# --- scan_row_to_dict ------------------------------------------------------------


def test_scan_row_to_dict_keys_are_stable():
    d = scan_row_to_dict("TEST", _forecast())
    assert set(d) == {
        "symbol",
        "status",
        "current_price",
        "median_target",
        "expected_return_pct",
        "band_width_pct",
        "signal",
    }
    _assert_json_round_trips(d)


# --- train_result_to_dict -------------------------------------------------------


def test_train_result_to_dict_keys_are_stable_and_skill_ci_is_explicitly_null():
    tr = TrainResult(
        symbols=["AAPL"],
        bundle_dir="/tmp/checkpoints/AAPL",
        val_loss=0.5,
        n_features=12,
        epochs_run=10,
        evaluation=_ev(),
    )
    d = train_result_to_dict(tr)
    assert set(d) == {
        "symbols",
        "bundle_dir",
        "val_loss",
        "n_features",
        "epochs_run",
        "evaluation",
        "skill_ci",
    }
    # PYQ-270: a single held-out split has no per-window series to bootstrap
    # from -- explicitly null, not a fabricated interval, and not omitted
    # either (a consumer distinguishing "no CI" from "field doesn't exist").
    assert d["skill_ci"] is None
    _assert_json_round_trips(d)


# --- backtest_to_dict ------------------------------------------------------------


def test_backtest_to_dict_keys_are_stable():
    d = backtest_to_dict(_backtest_result())
    assert set(d) == {
        "symbol",
        "n_windows",
        "aggregated",
        "per_window",
        "origins",
        "skill_ci",
        "directional_accuracy_ci",
        "signals_calibrated",
    }
    _assert_json_round_trips(d)


def test_backtest_to_dict_signals_calibrated_defaults_false():
    """PYQ-149: signals never carry a conformal offset today -- a JSON
    consumer must see that explicitly rather than assume `scan()` parity."""
    d = backtest_to_dict(_backtest_result())
    assert d["signals_calibrated"] is False


def test_backtest_to_dict_origins_match_per_window_order():
    """PYQ-266: origins are what lets compare_backtests verify two backtests
    were scored on the same windows -- order must be preserved through serialization."""
    br = _backtest_result(origins=(50, 60, 70))
    br.per_window = [_ev(), _ev(), _ev()]
    br.n_windows = 3
    d = backtest_to_dict(br)
    assert d["origins"] == [50, 60, 70]
    assert len(d["per_window"]) == 3


# --- seed_sweep_to_dict ----------------------------------------------------------


def test_seed_sweep_to_dict_keys_are_stable():
    sweep = SeedSweepResult(
        symbol="AAPL", seeds=[0, 1], per_seed=[_backtest_result(), _backtest_result()]
    )
    d = seed_sweep_to_dict(sweep)
    assert set(d) == {
        "symbol",
        "seeds",
        "per_seed",
        "skill_mean",
        "skill_sd",
        "skill_min",
        "skill_max",
    }
    assert len(d["per_seed"]) == 2
    _assert_json_round_trips(d)


# --- sweep_result_to_dict ---------------------------------------------------------


def test_sweep_result_to_dict_keys_are_stable():
    cells = [
        SweepCell("AAPL", "close", result=_backtest_result("AAPL")),
        SweepCell("MSFT", "close", error="not enough history"),
    ]
    result = SweepResult(symbols=["AAPL", "MSFT"], arm_names=["close"], cells=cells)
    d = sweep_result_to_dict(result)
    assert set(d) == {"symbols", "arms", "cells", "pooled_skill"}
    _assert_json_round_trips(d)


def test_sweep_result_to_dict_distinguishes_a_zero_skill_result_from_a_failed_cell():
    """A failed cell must serialize as null+error, not silently coerce to a
    zero result -- the same distinction SweepResult's own aggregates make."""
    cells = [
        SweepCell("AAPL", "close", result=_backtest_result("AAPL")),
        SweepCell("MSFT", "close", error="boom"),
    ]
    result = SweepResult(symbols=["AAPL", "MSFT"], arm_names=["close"], cells=cells)
    d = sweep_result_to_dict(result)

    ok_cell = next(c for c in d["cells"] if c["symbol"] == "AAPL")
    failed_cell = next(c for c in d["cells"] if c["symbol"] == "MSFT")
    assert ok_cell["result"] is not None
    assert ok_cell["error"] is None
    assert failed_cell["result"] is None
    assert failed_cell["error"] == "boom"


# --- signal_evaluation_to_dict -----------------------------------------------------


def test_signal_evaluation_to_dict_keys_are_stable():
    d = signal_evaluation_to_dict(_signal_evaluation())
    assert set(d) == {
        "n_buy",
        "n_sell",
        "n_hold",
        "hit_rate_buy",
        "hit_rate_sell",
        "avg_return_buy_pct",
        "avg_return_sell_pct",
        "turnover",
        "strategy_pnl_pct",
        "buy_and_hold_pnl_pct",
        "cost_bps",
        "n_periods",
    }
    _assert_json_round_trips(d)


# --- interpretation_to_dict --------------------------------------------------------


def test_interpretation_to_dict_keys_are_stable():
    d = interpretation_to_dict(_interpretation())
    assert set(d) == {"symbol", "feature_importance", "attention", "bundle_skill"}
    _assert_json_round_trips(d)


def test_interpretation_to_dict_features_are_importance_sorted_descending():
    d = interpretation_to_dict(_interpretation())
    weights = [f["weight"] for f in d["feature_importance"]]
    assert weights == sorted(weights, reverse=True)


def test_interpretation_to_dict_top_truncates_without_changing_the_order():
    interp = _interpretation()
    full = interpretation_to_dict(interp)
    top1 = interpretation_to_dict(interp, top=1)
    assert len(top1["feature_importance"]) == 1
    assert top1["feature_importance"][0] == full["feature_importance"][0]
