"""GET /explain/{symbol} (docs/api-design.md)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Path as PathParam

from pyquant.analysis import serialize
from pyquant.analysis.interpret import explain_forecast
from pyquant.api.deps import (
    BundleCache,
    get_bundle_cache,
    get_prediction_lock,
    get_settings,
    require_api_key,
)
from pyquant.api.schemas import SYMBOL_PATTERN, InterpretationResponse
from pyquant.config import Settings
from pyquant.models import tft

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/explain/{symbol}", response_model=InterpretationResponse)
def get_explanation(
    symbol: str = PathParam(..., pattern=SYMBOL_PATTERN),
    settings: Settings = Depends(get_settings),
    bundle_cache: BundleCache = Depends(get_bundle_cache),
) -> InterpretationResponse:
    """Feature importance + temporal attention for symbol's most recent forecast."""
    symbol = symbol.upper()
    try:
        bundle = bundle_cache.get(symbol, settings)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    # Same per-bundle lock as /forecast (docs/api-design.md #4).
    with get_prediction_lock(symbol):
        try:
            interp = explain_forecast(symbol, settings, bundle=bundle)
        except tft.FeatureSchemaMismatch as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    return InterpretationResponse(**serialize.interpretation_to_dict(interp))
