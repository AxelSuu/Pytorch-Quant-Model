"""POST /backtest -> job id; GET /backtest/{job_id} (features.md#pyq-271).

Same shape as /train (routes/train.py): `walk_forward_backtest()` trains
`n_windows` models sequentially and is emphatically not a request-cycle
operation (investigations.md#pyq-319 measured a *single* cold forecast at
~65s; a 5-window backtest trains five of those), so it runs as a background
job against the shared JobRegistry rather than blocking the request thread.

Unlike /train, a backtest never writes a persistent bundle -- each window's
model is discarded after evaluation -- so there is nothing here for
bugs.md#pyq-161's per-bundle-name lock to guard; two concurrent backtests for
the same symbol just duplicate work, they cannot corrupt a shared checkpoint
directory.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, HTTPException

from pyquant.analysis import serialize
from pyquant.api.deps import get_settings, require_api_key
from pyquant.api.jobs import JobRegistry, get_job_executor, get_job_registry
from pyquant.api.schemas import BacktestJobResponse, BacktestJobStatusResponse, BacktestRequest
from pyquant.config import Settings
from pyquant.models import tft

router = APIRouter(dependencies=[Depends(require_api_key)])
logger = logging.getLogger(__name__)


def _run_backtest_job(
    job_id: str,
    request: BacktestRequest,
    settings: Settings,
    registry: JobRegistry,
) -> None:
    registry.mark_running(job_id)
    try:
        result = tft.walk_forward_backtest(
            request.symbol,
            settings,
            n_windows=request.windows,
            max_epochs=request.epochs,
            progress=False,
        )
    except Exception as exc:
        logger.warning("Backtest job %s failed: %s", job_id, exc)
        registry.mark_failed(job_id, str(exc))
        return
    registry.mark_succeeded(job_id, result)


@router.post("/backtest", response_model=BacktestJobResponse, status_code=202)
async def start_backtest(
    request: BacktestRequest,
    settings: Settings = Depends(get_settings),
    registry: JobRegistry = Depends(get_job_registry),
    executor: ThreadPoolExecutor = Depends(get_job_executor),
) -> BacktestJobResponse:
    """Queue a walk-forward backtest; poll GET /backtest/{job_id} for its status/result."""
    if request.period:
        settings.data.period = request.period
    job_id = registry.create(kind="backtest")
    # A dedicated executor, not BackgroundTasks.add_task -- see the matching
    # comment in routes/train.py (bugs.md#pyq-163).
    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        executor, functools.partial(_run_backtest_job, job_id, request, settings, registry)
    )
    return BacktestJobResponse(job_id=job_id, status="queued")


@router.get("/backtest/{job_id}", response_model=BacktestJobStatusResponse)
def get_backtest_job(
    job_id: str, registry: JobRegistry = Depends(get_job_registry)
) -> BacktestJobStatusResponse:
    """Current status of a backtest job, and its result once it succeeds."""
    record = registry.get(job_id)
    if record is None or record.kind != "backtest":
        raise HTTPException(status_code=404, detail=f"No job {job_id!r}")
    result = serialize.backtest_to_dict(record.result) if record.result is not None else None
    return BacktestJobStatusResponse(
        job_id=record.job_id,
        status=record.status,
        result=result,
        error=record.error,
    )
