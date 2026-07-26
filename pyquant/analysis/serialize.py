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
from pyquant.models.tft import BacktestResult, TrainResult


def evaluation_to_dict(ev: EvaluationMetrics) -> dict[str, Any]:
    return {
        "model_mae": ev.model_mae,
        "baseline_mae": ev.baseline_mae,
        "skill_vs_baseline": ev.skill_vs_baseline,
        "directional_accuracy": ev.directional_accuracy,
        "calibration_coverage": ev.calibration_coverage,
        # The denominator behind every rate above (PYQ-117).
        "n_samples": ev.n_samples,
        "n_points": ev.n_points,
    }


def forecast_to_dict(fc: Forecast) -> dict[str, Any]:
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


def train_result_to_dict(tr: TrainResult) -> dict[str, Any]:
    return {
        "symbols": list(tr.symbols),
        "bundle_dir": str(tr.bundle_dir),
        "val_loss": tr.val_loss,
        "n_features": tr.n_features,
        "epochs_run": tr.epochs_run,
        "evaluation": evaluation_to_dict(tr.evaluation),
    }


def backtest_to_dict(br: BacktestResult) -> dict[str, Any]:
    return {
        "symbol": br.symbol,
        "n_windows": br.n_windows,
        "aggregated": evaluation_to_dict(br.aggregated),
        "per_window": [evaluation_to_dict(w) for w in br.per_window],
    }


def interpretation_to_dict(interp: Interpretation, top: int | None = None) -> dict[str, Any]:
    features = interp.top_features(top) if top is not None else sorted(
        interp.feature_importance.items(), key=lambda kv: kv[1], reverse=True
    )
    return {
        "symbol": interp.symbol,
        "feature_importance": [{"feature": name, "weight": w} for name, w in features],
        "attention": np.asarray(interp.attention).tolist(),
    }
