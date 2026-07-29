"""POST /train -> job id; GET /train/{job_id} (docs/api-design.md #2).

`tft.train()` blocks for a full Lightning fit, so it runs via FastAPI's
`BackgroundTasks` against the in-process job registry (jobs.py) rather than on the
request thread.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from pyquant.analysis import serialize
from pyquant.api.deps import BundleCache, get_bundle_cache, get_settings, require_api_key
from pyquant.api.jobs import JobRegistry, get_job_registry
from pyquant.api.schemas import TrainJobResponse, TrainJobStatusResponse, TrainRequest
from pyquant.config import Settings
from pyquant.models import tft

router = APIRouter(dependencies=[Depends(require_api_key)])
logger = logging.getLogger(__name__)


def _run_train_job(
    job_id: str,
    request: TrainRequest,
    settings: Settings,
    registry: JobRegistry,
    bundle_cache: BundleCache,
) -> None:
    registry.mark_running(job_id)
    try:
        result = tft.train(
            request.symbols,
            settings,
            bundle_name=request.bundle_name,
            max_epochs=request.epochs,
            progress=False,
        )
    except Exception as exc:
        logger.warning("Training job %s failed: %s", job_id, exc)
        registry.mark_failed(job_id, str(exc))
        return
    # The bundle on disk just changed; an LRU-cached copy from before this run
    # would silently serve stale weights to the next /forecast or /explain call.
    bundle_cache.invalidate(result.bundle_dir.name)
    registry.mark_succeeded(job_id, result)


@router.post("/train", response_model=TrainJobResponse, status_code=202)
def start_train(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
    registry: JobRegistry = Depends(get_job_registry),
    bundle_cache: BundleCache = Depends(get_bundle_cache),
) -> TrainJobResponse:
    """Queue a training run; poll GET /train/{job_id} for its status/result."""
    if not request.symbols:
        raise HTTPException(status_code=422, detail="symbols must not be empty")
    if request.period:
        settings.data.period = request.period
    # Mirrors tft.train()'s own default (bundle_name or "_".join(symbols)).upper())
    # so the in-flight check below can run before scheduling (bugs.md#pyq-161).
    # tft.train() still computes this itself; duplicated here only for the lock key.
    bundle_name = (request.bundle_name or "_".join(request.symbols)).upper()
    job_id = registry.try_start_train(bundle_name)
    if job_id is None:
        raise HTTPException(
            status_code=409,
            detail=f"A training job for bundle {bundle_name!r} is already queued or running",
        )
    background_tasks.add_task(_run_train_job, job_id, request, settings, registry, bundle_cache)
    return TrainJobResponse(job_id=job_id, status="queued")


@router.get("/train/{job_id}", response_model=TrainJobStatusResponse)
def get_train_job(
    job_id: str, registry: JobRegistry = Depends(get_job_registry)
) -> TrainJobStatusResponse:
    """Current status of a training job, and its result once it succeeds."""
    record = registry.get(job_id)
    if record is None or record.kind != "train":
        raise HTTPException(status_code=404, detail=f"No job {job_id!r}")
    result = (
        serialize.train_result_to_dict(record.result) if record.result is not None else None
    )
    return TrainJobStatusResponse(
        job_id=record.job_id,
        status=record.status,
        result=result,
        error=record.error,
    )
