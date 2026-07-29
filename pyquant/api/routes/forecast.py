"""GET /forecast/{symbol}, POST /scan (docs/api-design.md)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Path as PathParam

from pyquant.analysis import serialize
from pyquant.analysis.forecast import generate_forecast
from pyquant.api.deps import (
    BundleCache,
    acquire_prediction_lock,
    get_bundle_cache,
    get_settings,
    require_api_key,
)
from pyquant.api.schemas import SYMBOL_PATTERN, ForecastResponse, ScanRequest, ScanRow
from pyquant.config import Settings
from pyquant.models import tft

router = APIRouter(dependencies=[Depends(require_api_key)])
logger = logging.getLogger(__name__)


def _get_forecast(symbol: str, settings: Settings, bundle_cache: BundleCache):
    symbol = symbol.upper()
    try:
        bundle = bundle_cache.get(symbol, settings)
    except FileNotFoundError as exc:
        # The underlying message includes an absolute checkpoint path
        # (tft.py's `_load`); logged in full server-side, but not handed to a
        # remote caller, who only needs "this symbol isn't trained."
        logger.info("Forecast requested for untrained bundle: %s", exc)
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for {symbol}. Run `pyquant train` first.",
        ) from exc
    # Per-bundle lock (docs/api-design.md #4): serialize predictions against this
    # bundle's model instance; concurrent requests for *different* bundles proceed
    # unblocked since each has its own lock. Bounded wait (429 past
    # PREDICTION_LOCK_TIMEOUT_SECONDS), not an indefinite block.
    with acquire_prediction_lock(symbol):
        try:
            return generate_forecast(symbol, settings, bundle=bundle)
        except tft.FeatureSchemaMismatch as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/forecast/{symbol}", response_model=ForecastResponse)
def get_forecast(
    symbol: str = PathParam(..., pattern=SYMBOL_PATTERN),
    settings: Settings = Depends(get_settings),
    bundle_cache: BundleCache = Depends(get_bundle_cache),
) -> ForecastResponse:
    """p10/p50/p90 quantile forecast for symbol, from its trained bundle."""
    fc = _get_forecast(symbol, settings, bundle_cache)
    return ForecastResponse(**serialize.forecast_to_dict(fc))


@router.post("/scan", response_model=list[ScanRow])
def scan(
    request: ScanRequest,
    settings: Settings = Depends(get_settings),
    bundle_cache: BundleCache = Depends(get_bundle_cache),
) -> list[ScanRow]:
    """Forecast every requested symbol; one flaky symbol must not fail the rest."""
    rows: list[ScanRow] = []
    for symbol in request.symbols:
        ticker = symbol.strip().upper()
        if not ticker:
            continue
        try:
            fc = _get_forecast(ticker, settings, bundle_cache)
        except HTTPException as exc:
            status = "not_trained" if exc.status_code == 404 else "error"
            rows.append(ScanRow(symbol=ticker, status=status, error=str(exc.detail)))
            continue
        except Exception as exc:
            # One flaky symbol must not sink the whole comparison (PYQ-113, same
            # discipline as the CLI's scan).
            rows.append(ScanRow(symbol=ticker, status="error", error=str(exc)))
            continue
        rows.append(ScanRow(**serialize.scan_row_to_dict(ticker, fc)))
    return rows
