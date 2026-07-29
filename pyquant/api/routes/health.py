"""GET /healthz -- liveness, no auth (docs/api-design.md)."""

from __future__ import annotations

from fastapi import APIRouter

from pyquant.api.schemas import HealthResponse

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """Liveness check: always 200 if the process is up.

    Deliberately `async def`, not `def` -- a sync route handler dispatches
    through FastAPI's `run_in_threadpool`, which shares anyio's single default
    worker-thread limiter with every other sync endpoint *and* with
    `BackgroundTasks` (verified against the installed Starlette/anyio source:
    both paths call `anyio.to_thread.run_sync` with no dedicated limiter). A
    sync liveness probe would queue behind a long-running `POST /train`
    background job under load -- exactly when an orchestrator's liveness
    check needs to answer fastest. This handler does no I/O, so it costs
    nothing to keep off that shared pool entirely.
    """
    return HealthResponse()
