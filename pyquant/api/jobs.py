"""In-process training-job registry (v1), per docs/api-design.md #2.

`tft.train()` blocks for a full Lightning fit; a request thread must not block on it.
This registry backs `POST /train` (returns a job id immediately) and `GET /train/{job_id}`
(polls status), run via FastAPI's `BackgroundTasks`.

Where this stops scaling -- the trigger to graduate to a real queue (arq/Celery + Redis),
per the design note: job state lives in process memory, so it is lost on restart/redeploy
and invisible to a second instance; there is no concurrency control, no retries, no
backpressure, no cancellation. None of that is needed for a single-instance v1.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Literal

from pyquant.models.tft import TrainResult

JobStatus = Literal["queued", "running", "succeeded", "failed"]


@dataclass
class JobRecord:
    """One training job's status, and its result or error once it finishes."""

    job_id: str
    status: JobStatus = "queued"
    result: TrainResult | None = None
    error: str | None = None


class JobRegistry:
    """Thread-safe in-process store of JobRecords, keyed by job id."""

    def __init__(self) -> None:
        """No jobs yet."""
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}

    def create(self) -> str:
        """Register a new queued job and return its id."""
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = JobRecord(job_id=job_id)
        return job_id

    def get(self, job_id: str) -> JobRecord | None:
        """Return the record for job_id, or None if it doesn't exist."""
        with self._lock:
            return self._jobs.get(job_id)

    def mark_running(self, job_id: str) -> None:
        """Flip a queued job to running, once its background task starts."""
        with self._lock:
            self._jobs[job_id].status = "running"

    def mark_succeeded(self, job_id: str, result: TrainResult) -> None:
        """Record a job's successful TrainResult."""
        with self._lock:
            record = self._jobs[job_id]
            record.status = "succeeded"
            record.result = result

    def mark_failed(self, job_id: str, error: str) -> None:
        """Record a job's failure message."""
        with self._lock:
            record = self._jobs[job_id]
            record.status = "failed"
            record.error = error


_REGISTRY = JobRegistry()


def get_job_registry() -> JobRegistry:
    """FastAPI dependency: the process-wide job registry."""
    return _REGISTRY
