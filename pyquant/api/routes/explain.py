"""GET /explain/{symbol} (docs/api-design.md)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Path as PathParam

from pyquant.analysis import serialize
from pyquant.analysis.interpret import explain_forecast
from pyquant.api.deps import (
    BundleCache,
    acquire_prediction_lock,
    get_bundle_cache,
    get_settings,
    require_api_key,
)
from pyquant.api.schemas import SYMBOL_PATTERN, InterpretationResponse
from pyquant.config import Settings
from pyquant.models import tft

router = APIRouter(dependencies=[Depends(require_api_key)])
logger = logging.getLogger(__name__)


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
        # See forecast.py's identical comment: the raw message names an
        # absolute checkpoint path, logged here rather than sent to the caller.
        logger.info("Explanation requested for untrained bundle: %s", exc)
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for {symbol}. Run `pyquant train` first.",
        ) from exc
    # Same per-bundle lock as /forecast (docs/api-design.md #4), same bounded wait.
    with acquire_prediction_lock(symbol):
        try:
            interp = explain_forecast(symbol, settings, bundle=bundle)
        except tft.FeatureSchemaMismatch as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    return InterpretationResponse(**serialize.interpretation_to_dict(interp))
