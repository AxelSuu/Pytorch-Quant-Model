"""Tests for pyquant/experiments/sweep.py: the multi-symbol sweep harness (PYQ-268)."""

from types import SimpleNamespace

import pytest

from pyquant.experiments import sweep as sweep_mod
from pyquant.experiments.sweep import (
    Arm,
    SweepCell,
    SweepResult,
    _resolve_override_target,
    apply_overrides,
    run_sweep,
)
from pyquant.models.tft import BacktestResult


def _ev(model_mae, baseline_mae=2.0, n_samples=1, n_points=5):
    from pyquant.analysis.metrics import EvaluationMetrics

    return EvaluationMetrics(
        model_mae=model_mae,
        baseline_mae=baseline_mae,
        directional_accuracy=0.5,
        calibration_coverage=0.8,
        n_samples=n_samples,
        n_points=n_points,
    )


def _backtest_result(symbol, skill_mae, origins=(100,)):
    ev = _ev(skill_mae)
    return BacktestResult(
        symbol=symbol,
        n_windows=len(origins),
        per_window=[ev] * len(origins),
        aggregated=ev,
        origins=list(origins),
    )


# --- apply_overrides / key resolution ------------------------------------------


def test_apply_overrides_sets_the_field_and_leaves_the_original_untouched(settings):
    result = apply_overrides(settings, {"target": "log_return"})
    assert result.training.target == "log_return"
    assert settings.training.target == "close"  # original object untouched


def test_apply_overrides_resolves_a_dotted_path():
    from pyquant.config import Settings

    result = apply_overrides(Settings(), {"training.target": "log_return"})
    assert result.training.target == "log_return"


def test_apply_overrides_coerces_string_values_against_the_current_field_type():
    from pyquant.config import Settings

    result = apply_overrides(Settings(), {"use_sentiment": "false", "validation_days": "30"})
    assert result.data.use_sentiment is False
    assert result.training.validation_days == 30


def test_apply_overrides_raises_a_clear_error_on_an_unknown_key(settings):
    with pytest.raises(ValueError, match="unknown override key"):
        apply_overrides(settings, {"not_a_real_field": "x"})


def test_resolve_override_target_raises_on_ambiguous_key_without_a_dotted_path():
    """No real field collides across training/data/tft today (verified), but the
    resolver must still refuse to silently guess if one ever does."""
    fake_settings = SimpleNamespace(
        training=SimpleNamespace(target="close"),
        data=SimpleNamespace(target="also_here"),
        tft=SimpleNamespace(hidden_size=8),
    )
    with pytest.raises(ValueError, match="ambiguous"):
        _resolve_override_target(fake_settings, "target")


# --- run_sweep -------------------------------------------------------------


def test_run_sweep_covers_the_full_symbol_by_arm_cell_matrix(monkeypatch, settings):
    calls = []

    def fake_backtest(symbol, s, **kwargs):
        calls.append((symbol, s.training.target))
        return _backtest_result(symbol, skill_mae=1.0)

    monkeypatch.setattr(sweep_mod, "walk_forward_backtest", fake_backtest)

    arms = [Arm("close", {"target": "close"}), Arm("log_return", {"target": "log_return"})]
    result = run_sweep(["AAA", "BBB"], arms, settings, n_windows=2)

    assert result.symbols == ["AAA", "BBB"]
    assert result.arm_names == ["close", "log_return"]
    assert len(result.cells) == 4
    assert set(calls) == {("AAA", "close"), ("AAA", "log_return"), ("BBB", "close"), ("BBB", "log_return")}
    assert all(c.ok for c in result.cells)


def test_run_sweep_degrades_a_failing_cell_to_a_recorded_gap(monkeypatch, settings):
    def fake_backtest(symbol, s, **kwargs):
        if symbol == "BAD":
            raise ValueError("not enough history")
        return _backtest_result(symbol, skill_mae=1.0)

    monkeypatch.setattr(sweep_mod, "walk_forward_backtest", fake_backtest)

    arms = [Arm("close", {"target": "close"})]
    result = run_sweep(["AAA", "BAD"], arms, settings, n_windows=2)

    assert len(result.cells) == 2
    good = result.cell("AAA", "close")
    bad = result.cell("BAD", "close")
    assert good.ok and good.error is None
    assert not bad.ok and "not enough history" in bad.error


# --- SweepResult -------------------------------------------------------------


def test_sweep_result_skill_by_symbol_excludes_failed_cells():
    cells = [
        SweepCell("AAA", "close", result=_backtest_result("AAA", 0.0)),  # skill 1.0
        SweepCell("BBB", "close", error="boom"),
    ]
    result = SweepResult(symbols=["AAA", "BBB"], arm_names=["close"], cells=cells)
    assert result.skill_by_symbol("close") == {"AAA": 1.0}


def test_sweep_result_pooled_skill_averages_over_successful_symbols_only():
    cells = [
        SweepCell("AAA", "close", result=_backtest_result("AAA", 0.0)),  # skill 1.0
        SweepCell("BBB", "close", result=_backtest_result("BBB", 1.0)),  # skill 0.5
        SweepCell("CCC", "close", error="boom"),
    ]
    result = SweepResult(symbols=["AAA", "BBB", "CCC"], arm_names=["close"], cells=cells)
    assert result.pooled_skill("close") == pytest.approx(0.75)


def test_sweep_result_pooled_skill_is_none_when_every_cell_failed():
    cells = [SweepCell("AAA", "close", error="boom")]
    result = SweepResult(symbols=["AAA"], arm_names=["close"], cells=cells)
    assert result.pooled_skill("close") is None


def test_sweep_result_helped_summary_counts_symbols_where_the_other_arm_scored_higher():
    cells = [
        SweepCell("AAA", "base", result=_backtest_result("AAA", 1.0)),  # skill 0.5
        SweepCell("AAA", "other", result=_backtest_result("AAA", 0.0)),  # skill 1.0 (helped)
        SweepCell("BBB", "base", result=_backtest_result("BBB", 0.0)),  # skill 1.0
        SweepCell("BBB", "other", result=_backtest_result("BBB", 1.0)),  # skill 0.5 (hurt)
        SweepCell("CCC", "base", error="boom"),  # excluded: no "base" result
        SweepCell("CCC", "other", result=_backtest_result("CCC", 0.0)),
    ]
    result = SweepResult(symbols=["AAA", "BBB", "CCC"], arm_names=["base", "other"], cells=cells)
    helped, total = result.helped_summary("base", "other")
    assert (helped, total) == (1, 2)


def test_sweep_result_paired_comparison_uses_compare_backtests_on_aligned_windows():
    cells = [
        SweepCell("AAA", "base", result=_backtest_result("AAA", 0.0, origins=[100, 105])),  # skill 1.0
        SweepCell("AAA", "other", result=_backtest_result("AAA", 1.0, origins=[100, 105])),  # skill 0.5
    ]
    result = SweepResult(symbols=["AAA"], arm_names=["base", "other"], cells=cells)
    comparison = result.paired_comparison("AAA", "base", "other")
    assert comparison is not None
    assert comparison.mean_diff == pytest.approx(0.5)  # base - other = 1.0 - 0.5


def test_sweep_result_paired_comparison_is_none_when_a_cell_failed():
    cells = [
        SweepCell("AAA", "base", result=_backtest_result("AAA", 0.0)),
        SweepCell("AAA", "other", error="boom"),
    ]
    result = SweepResult(symbols=["AAA"], arm_names=["base", "other"], cells=cells)
    assert result.paired_comparison("AAA", "base", "other") is None


def test_sweep_result_paired_comparison_is_none_when_windows_do_not_align():
    cells = [
        SweepCell("AAA", "base", result=_backtest_result("AAA", 0.0, origins=[100, 105])),
        SweepCell("AAA", "other", result=_backtest_result("AAA", 1.0, origins=[200, 205])),
    ]
    result = SweepResult(symbols=["AAA"], arm_names=["base", "other"], cells=cells)
    assert result.paired_comparison("AAA", "base", "other") is None
