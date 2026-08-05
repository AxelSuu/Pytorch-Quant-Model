"""GET /forecast/{symbol}, POST /scan (docs/api-design.md)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Response
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
from pyquant.data import forecast_store
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
    response: Response,
    symbol: str = PathParam(..., pattern=SYMBOL_PATTERN),
    settings: Settings = Depends(get_settings),
) -> ForecastResponse:
    """p10/p50/p90 quantile forecast for symbol.

    Read from the nightly precompute store (features.md#pyq-282) rather than
    run live -- a millisecond response on a warm store instead of the ~65s
    live pipeline (investigations.md#pyq-319).

    ``pyquant precompute`` (cli/app.py) writes the store; run it on a nightly
    schedule (a cron-triggered CLI invocation is the cheapest starting point,
    per the ticket's own scoping) after market close.
    """
    symbol = symbol.upper()
    stored = forecast_store.read_forecast(settings, symbol)
    if stored is None:
        try:
            tft.load_meta(symbol, settings)
        except FileNotFoundError as exc:
            # See the identical comment on the old live path: the underlying
            # message names an absolute checkpoint path, logged server-side only.
            logger.info("Forecast requested for untrained bundle: %s", exc)
            raise HTTPException(
                status_code=404,
                detail=f"No trained model for {symbol}. Run `pyquant train` first.",
            ) from exc
        raise HTTPException(
            status_code=503,
            detail=f"{symbol} is trained but has no precomputed forecast yet. "
            "Run `pyquant precompute` (or wait for the nightly job).",
        )
    if forecast_store.is_stale(stored.as_of):
        raise HTTPException(
            status_code=503,
            detail=f"Precomputed forecast for {symbol} is stale (as_of {stored.as_of}); "
            "the nightly `pyquant precompute` job may not have run. Not serving "
            "arbitrarily old data -- run it manually to refresh.",
        )
    response.headers["ETag"] = f'"{symbol}-{stored.computed_at}"'
    return ForecastResponse(**stored.payload, as_of=stored.as_of, computed_at=stored.computed_at)


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
