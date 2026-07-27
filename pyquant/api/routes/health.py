"""GET /healthz -- liveness, no auth (docs/api-design.md)."""

from __future__ import annotations

from fastapi import APIRouter

from pyquant.api.schemas import HealthResponse

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """Liveness check: always 200 if the process is up."""
    return HealthResponse()
