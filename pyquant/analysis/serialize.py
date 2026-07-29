"""JSON-able serializers for the analysis/model dataclasses.

Centralizes the numpy -> plain-Python conversion so machine-readable output
(the CLI's ``--format json``, PYQ-212) and a future REST layer (PYQ-213) share
exactly one place that knows how each domain object maps to JSON.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyquant.analysis.forecast import Forecast
from pyquant.analysis.interpret import Interpretation
from pyquant.analysis.metrics import EvaluationMetrics
from pyquant.analysis.signals import SignalEvaluation, classify_signal
from pyquant.models.tft import BacktestResult, TrainResult


def evaluation_to_dict(ev: EvaluationMetrics) -> dict[str, Any]:
    """Serialize evaluation metrics, keeping every rate next to its sample size.

    ``n_samples``/``n_points``/``effective_n_samples`` are included deliberately:
    PYQ-117 showed a rate reported without its denominator is unreadable, since
    "100% directional accuracy" turned out to mean five points.
    """
    return {
        "model_mae": ev.model_mae,
        "baseline_mae": ev.baseline_mae,
        "skill_vs_baseline": ev.skill_vs_baseline,
        "directional_accuracy": ev.directional_accuracy,
        "calibration_coverage": ev.calibration_coverage,
        "quantile_exceedance": ev.quantile_exceedance,
        "pinball_losses": ev.pinball_losses,
        # Proper scoring rule + width-aware interval score (PYQ-252). `pit` is
        # the raw per-point transform behind the calibration histogram, so a
        # consumer can render it without re-running the model.
        "crps": ev.crps,
        "winkler_score": ev.winkler_score,
        "pit": list(ev.pit),
        # The denominator behind every rate above (PYQ-117).
        "n_samples": ev.n_samples,
        "n_points": ev.n_points,
        "effective_n_samples": ev.effective_n_samples,
        # The profile every field above is a mean over h=1..horizon of (PYQ-267).
        "per_horizon": [
            {
                "step": step.step,
                "model_mae": step.model_mae,
                "baseline_mae": step.baseline_mae,
                "skill_vs_baseline": step.skill_vs_baseline,
                "directional_accuracy": step.directional_accuracy,
                "calibration_coverage": step.calibration_coverage,
            }
            for step in ev.per_horizon
        ],
        # MAE against every comparator beyond persistence (PYQ-275), plus which
        # one is hardest for the model to beat and skill against it.
        "baseline_maes": dict(ev.baseline_maes),
        "strongest_baseline": (
            {"name": ev.strongest_baseline[0], "mae": ev.strongest_baseline[1]}
            if ev.strongest_baseline is not None
            else None
        ),
        "skill_vs_strongest_baseline": ev.skill_vs_strongest_baseline,
    }


def forecast_to_dict(fc: Forecast) -> dict[str, Any]:
    """Serialize a quantile forecast, including the dates it is a forecast *for*.

    ``median`` and ``expected_return_pct`` are omitted rather than raising when
    0.5 is not among the configured quantiles (PYQ-106).
    """
    out: dict[str, Any] = {
        "symbol": fc.symbol,
        "last_date": fc.last_date.date().isoformat(),
        "current_price": fc.current_price,
        "horizon": fc.horizon,
        # Which dates the predictions are for -- consumers should not have to
        # re-derive a business-day calendar to find out (PYQ-115).
        "forecast_dates": [d.date().isoformat() for d in fc.forecast_dates],
        "quantiles": list(fc.quantiles),
        "predictions": np.asarray(fc.predictions).tolist(),  # (horizon, n_quantiles)
        # Non-zero means the raw band was non-monotonic and has been reordered
        # (PYQ-124) -- a consumer should not have to guess whether that happened.
        "n_quantile_crossings": fc.n_quantile_crossings,
    }
    # median / expected return need 0.5 configured (see PYQ-106); omit otherwise
    # rather than raise on the serialization path.
    if 0.5 in fc.quantiles:
        out["median"] = np.asarray(fc.median).tolist()
        out["expected_return_pct"] = fc.expected_return_pct()
    return out


def scan_row_to_dict(symbol: str, fc: Forecast) -> dict[str, Any]:
    """One `scan` comparison row for a successfully-forecast symbol.

    The single implementation the CLI's `scan` command and the PYQ-261 API's
    `/scan` route both call, so the two front-ends' `signal`/`band_width_pct`
    cannot drift apart the way PYQ-119 found for config handling.
    """
    pct = fc.expected_return_pct()
    lo = fc.quantile_series(fc.quantiles[0])[-1]
    hi = fc.quantile_series(fc.quantiles[-1])[-1]
    lo_pct = (lo - fc.current_price) / fc.current_price * 100
    hi_pct = (hi - fc.current_price) / fc.current_price * 100
    band = (hi - lo) / fc.current_price * 100
    return {
        "symbol": symbol,
        "status": "ok",
        "current_price": fc.current_price,
        "median_target": float(fc.median[-1]),
        "expected_return_pct": pct,
        "band_width_pct": band,
        "signal": classify_signal(pct, lower_pct=lo_pct, upper_pct=hi_pct),
    }


def train_result_to_dict(tr: TrainResult) -> dict[str, Any]:
    """Serialize a training run: where the bundle landed and how it scored."""
    return {
        "symbols": list(tr.symbols),
        "bundle_dir": str(tr.bundle_dir),
        "val_loss": tr.val_loss,
        "n_features": tr.n_features,
        "epochs_run": tr.epochs_run,
        "evaluation": evaluation_to_dict(tr.evaluation),
    }


def backtest_to_dict(br: BacktestResult) -> dict[str, Any]:
    """Serialize a walk-forward backtest, keeping per-window results alongside the aggregate.

    The per-window list is not redundant with ``aggregated``: it is what lets a
    consumer see dispersion across origins rather than only the mean.
    """
    return {
        "symbol": br.symbol,
        "n_windows": br.n_windows,
        "aggregated": evaluation_to_dict(br.aggregated),
        "per_window": [evaluation_to_dict(w) for w in br.per_window],
        # Window identity, in `per_window` order -- what lets `compare_backtests`
        # (PYQ-266) verify two backtests were scored on the same windows.
        "origins": list(br.origins),
    }


def signal_evaluation_to_dict(ev: SignalEvaluation) -> dict[str, Any]:
    """Serialize a PYQ-255 signal P&L evaluation."""
    return {
        "n_buy": ev.n_buy,
        "n_sell": ev.n_sell,
        "n_hold": ev.n_hold,
        "hit_rate_buy": ev.hit_rate_buy,
        "hit_rate_sell": ev.hit_rate_sell,
        "avg_return_buy_pct": ev.avg_return_buy_pct,
        "avg_return_sell_pct": ev.avg_return_sell_pct,
        "turnover": ev.turnover,
        "strategy_pnl_pct": ev.strategy_pnl_pct,
        "buy_and_hold_pnl_pct": ev.buy_and_hold_pnl_pct,
        "cost_bps": ev.cost_bps,
        "n_periods": ev.n_periods,
    }


def interpretation_to_dict(interp: Interpretation, top: int | None = None) -> dict[str, Any]:
    """Serialize feature importances and attention weights, importance-sorted.

    Args:
        interp: The interpretation to serialize.
        top: Keep only the ``top`` most important features. ``None`` keeps all,
            still sorted, so a consumer can truncate rather than re-rank.
    """
    features = interp.top_features(top) if top is not None else sorted(
        interp.feature_importance.items(), key=lambda kv: kv[1], reverse=True
    )
    return {
        "symbol": interp.symbol,
        "feature_importance": [{"feature": name, "weight": w} for name, w in features],
        "attention": np.asarray(interp.attention).tolist(),
        # None when the bundle predates evaluation being recorded. A consumer
        # should treat weights from a bundle with non-positive skill as a
        # description of the model's attention, not of what moves the price
        # (investigations.md#pyq-314).
        "bundle_skill": interp.bundle_skill,
    }
