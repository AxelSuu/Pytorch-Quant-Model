"""GET /symbols, GET /metrics/{symbol} -- discovering what's trained (PYQ-283).

Every other read endpoint (``/forecast/{symbol}``, ``/explain/{symbol}``)
requires already knowing a trained symbol, and the only way to find out was to
try one and read the 404. These two are read-only and cheap: everything they
return already sits in a bundle's ``meta.json``, written at train time.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Path as PathParam

from pyquant.analysis.metrics import effective_sample_size, skill_vs_baseline_from_maes
from pyquant.api.deps import get_settings, require_api_key
from pyquant.api.schemas import SYMBOL_PATTERN, BundleSummary, EvaluationResponse
from pyquant.config import Settings
from pyquant.models import tft

router = APIRouter(dependencies=[Depends(require_api_key)])


def _effective_n_samples(evaluation: dict) -> int:
    """Same derivation as ``EvaluationMetrics.effective_n_samples``, from the persisted dict."""
    n_samples = evaluation.get("n_samples") or 0
    n_points = evaluation.get("n_points") or 0
    if not n_samples or not n_points:
        return 0
    horizon = max(1, round(n_points / n_samples))
    return effective_sample_size(n_samples, horizon)


def _evaluation_response(meta: dict) -> EvaluationResponse:
    ev = meta.get("evaluation") or {}
    return EvaluationResponse(
        model_mae=ev.get("model_mae", 0.0),
        baseline_mae=ev.get("baseline_mae", 0.0),
        skill_vs_baseline=skill_vs_baseline_from_maes(ev.get("baseline_mae"), ev.get("model_mae"))
        or 0.0,
        directional_accuracy=ev.get("directional_accuracy", 0.0),
        calibration_coverage=ev.get("calibration_coverage", 0.0),
        quantile_exceedance=ev.get("quantile_exceedance", {}),
        pinball_losses=ev.get("pinball_losses", {}),
        crps=ev.get("crps", 0.0),
        winkler_score=ev.get("winkler_score", 0.0),
        pit=ev.get("pit", []),
        n_samples=ev.get("n_samples", 0),
        n_points=ev.get("n_points", 0),
        effective_n_samples=_effective_n_samples(ev),
    )


@router.get("/symbols", response_model=list[BundleSummary])
def list_symbols(settings: Settings = Depends(get_settings)) -> list[BundleSummary]:
    """Every trained bundle under ``checkpoint_dir``, most recently trained first."""
    return [
        BundleSummary(
            symbol=meta.get("symbol", ""),
            trained_at=meta.get("trained_at"),
            bundle_skill=skill_vs_baseline_from_maes(
                (meta.get("evaluation") or {}).get("baseline_mae"),
                (meta.get("evaluation") or {}).get("model_mae"),
            ),
        )
        for meta in tft.list_bundles(settings)
    ]


@router.get("/metrics/{symbol}", response_model=EvaluationResponse)
def get_metrics(
    symbol: str = PathParam(..., pattern=SYMBOL_PATTERN),
    settings: Settings = Depends(get_settings),
) -> EvaluationResponse:
    """A bundle's recorded evaluation (skill, calibration, directional accuracy).

    Reads straight off ``meta.json`` -- no forecast is generated, and nothing is
    recomputed beyond the two derived fields (``skill_vs_baseline``,
    ``effective_n_samples``) that ``meta.json`` never stored because both are
    an ``@property``, not a dataclass field.
    """
    symbol = symbol.upper()
    try:
        meta = tft.load_meta(symbol, settings)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for {symbol}. Run `pyquant train` first.",
        ) from exc
    return _evaluation_response(meta)
