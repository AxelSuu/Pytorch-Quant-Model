"""In-process training/backtest-job registry (v1), per docs/api-design.md #2.

`tft.train()`/`tft.walk_forward_backtest()` both block for one or more full Lightning
fits; a request thread must not block on either. This registry backs `POST /train` +
`GET /train/{job_id}` and `POST /backtest` + `GET /backtest/{job_id}` (features.md#pyq-271
reuses this registry rather than building a second job mechanism), run via FastAPI's
`BackgroundTasks`.

Where this stops scaling -- the trigger to graduate to a real queue (arq/Celery + Redis),
per the design note: job state lives in process memory, so it is lost on restart/redeploy
and invisible to a second instance; there is no concurrency control, no retries, no
backpressure, no cancellation. None of that is needed for a single-instance v1.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

from pyquant.models.tft import BacktestResult, TrainResult

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "succeeded", "failed"]
JobKind = Literal["train", "backtest"]
JobResult = TrainResult | BacktestResult

# Bound in-process job history (bugs.md#pyq-159): with no cap, a long-running server
# accumulates one JobRecord per POST /train or /backtest forever. Oldest-first eviction
# is enough for v1 -- the same tradeoff deps.BundleCache's LRU already makes for loaded
# bundles, just without the "reload on next use" recovery (an evicted job's status is
# just gone, matching a queue-backed job store's own retention limits).
_DEFAULT_MAX_JOBS = 500


@dataclass
class JobRecord:
    """One job's status, and its result or error once it finishes."""

    job_id: str
    kind: JobKind
    # Set only for "train" jobs (PYQ-161's in-flight guard); None for "backtest",
    # which never writes a persistent bundle so has nothing to race on disk.
    bundle_name: str | None = None
    # Set only for "backtest" jobs (PYQ-327's dedup guard): identifies the request
    # shape (symbol/windows/epochs/period) this job is currently the in-flight
    # representative for, so an identical concurrent request can be folded into it
    # instead of spinning up a duplicate multi-window Lightning run.
    backtest_key: str | None = None
    status: JobStatus = "queued"
    result: JobResult | None = None
    error: str | None = None


class JobRegistry:
    """Thread-safe in-process store of JobRecords, keyed by job id.

    Shared by /train and /backtest: both are "runs in the background, poll for
    status" jobs, and duplicating the bookkeeping for a second resource would
    be exactly the second job mechanism features.md#pyq-271 says not to build.
    `kind` keeps their id spaces from colliding in meaning even though they
    share one dict -- polling a train job id via GET /backtest/{job_id} 404s
    rather than trying to serialize a TrainResult as a BacktestResponse.
    """

    def __init__(self, max_jobs: int = _DEFAULT_MAX_JOBS) -> None:
        """No jobs yet; evict beyond `max_jobs`, oldest first."""
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._order: list[str] = []
        self._max_jobs = max_jobs
        # bundle_name -> job_id, for "train" jobs currently queued/running
        # (bugs.md#pyq-161). Lets a second POST /train for the same bundle be
        # rejected before it races the first onto the same on-disk checkpoint
        # directory, mirroring deps.py's per-bundle prediction lock.
        self._active_bundle_names: dict[str, str] = {}
        # backtest request key -> job_id, for "backtest" jobs currently
        # queued/running (PYQ-327). A backtest can't corrupt shared state the
        # way a second concurrent train for one bundle can, so this folds an
        # identical concurrent request into the existing job rather than
        # rejecting it outright.
        self._active_backtest_keys: dict[str, str] = {}

    def create(self, kind: JobKind = "train") -> str:
        """Register a new queued job (no bundle-name guard) and return its id."""
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = JobRecord(job_id=job_id, kind=kind)
            self._order.append(job_id)
            self._evict_locked()
        return job_id

    def try_start_train(self, bundle_name: str) -> str | None:
        """Atomically register a queued "train" job for bundle_name.

        Returns the new job id, or ``None`` if a job for the same bundle_name
        is already queued/running -- the caller should reject the request
        (409) rather than schedule a second fit onto the same checkpoint
        directory (bugs.md#pyq-161).
        """
        bundle_name = bundle_name.upper()
        with self._lock:
            if bundle_name in self._active_bundle_names:
                return None
            job_id = uuid.uuid4().hex
            self._jobs[job_id] = JobRecord(job_id=job_id, kind="train", bundle_name=bundle_name)
            self._order.append(job_id)
            self._active_bundle_names[bundle_name] = job_id
            self._evict_locked()
            return job_id

    def try_start_backtest(self, key: str) -> tuple[str, bool]:
        """Atomically register a queued "backtest" job for `key`, or fold it into an existing in-flight job for the same request.

        Unlike `try_start_train`, this never rejects a request outright --
        two concurrent backtests for the same symbol can't corrupt shared disk
        state (nothing here for bugs.md#pyq-161's guard to protect). But
        nothing bounded how much duplicate work an identical retry/double-click
        could pile onto the shared 4-worker `_JOB_EXECUTOR` either (PYQ-327).
        `key` should identify the full request shape (symbol + windows +
        epochs + period); an identical key gets the *same* job_id back rather
        than a new job, and the caller should not schedule new work in that
        case. Returns `(job_id, created)`.
        """
        with self._lock:
            existing = self._active_backtest_keys.get(key)
            if existing is not None:
                return existing, False
            job_id = uuid.uuid4().hex
            self._jobs[job_id] = JobRecord(job_id=job_id, kind="backtest", backtest_key=key)
            self._order.append(job_id)
            self._active_backtest_keys[key] = job_id
            self._evict_locked()
            return job_id, True

    def _evict_locked(self) -> None:
        while len(self._order) > self._max_jobs:
            oldest = self._order.pop(0)
            record = self._jobs.pop(oldest, None)
            self._release_locks_locked(record)

    def get(self, job_id: str) -> JobRecord | None:
        """Return the record for job_id, or None if it doesn't exist."""
        with self._lock:
            return self._jobs.get(job_id)

    def mark_running(self, job_id: str) -> None:
        """Flip a queued job to running, once its background task starts.

        A no-op (logged) for an unknown job_id rather than a bare KeyError
        (bugs.md#pyq-159) -- this runs inside a BackgroundTask, where an
        unhandled exception has no request to surface a 500 to.
        """
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                logger.warning("mark_running: unknown job_id %r", job_id)
                return
            record.status = "running"

    def mark_succeeded(self, job_id: str, result: JobResult) -> None:
        """Record a job's successful result and release its in-flight guard."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                logger.warning("mark_succeeded: unknown job_id %r", job_id)
                return
            record.status = "succeeded"
            record.result = result
            self._release_locks_locked(record)

    def mark_failed(self, job_id: str, error: str) -> None:
        """Record a job's failure message and release its in-flight guard."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                logger.warning("mark_failed: unknown job_id %r", job_id)
                return
            record.status = "failed"
            record.error = error
            self._release_locks_locked(record)

    def _release_locks_locked(self, record: JobRecord | None) -> None:
        if record is None:
            return
        if record.bundle_name is not None:
            if self._active_bundle_names.get(record.bundle_name) == record.job_id:
                del self._active_bundle_names[record.bundle_name]
        if record.backtest_key is not None:
            if self._active_backtest_keys.get(record.backtest_key) == record.job_id:
                del self._active_backtest_keys[record.backtest_key]


_REGISTRY = JobRegistry()


def get_job_registry() -> JobRegistry:
    """FastAPI dependency: the process-wide job registry."""
    return _REGISTRY


# bugs.md#pyq-163: a sync FastAPI route and a sync BackgroundTasks function both
# dispatch through starlette.concurrency.run_in_threadpool, which draws from
# anyio's single process-wide default thread limiter (40 slots) -- the exact
# same pool /forecast, /explain and /scan need. A training job holding a slot
# for minutes then queues a concurrent read endpoint behind it. This executor
# is dedicated to /train and /backtest's background work, so those jobs stop
# drawing from the pool request handling needs; bounded (not unbounded) so an
# unlimited number of concurrent POST /train calls cannot each spin up their
# own OS thread without limit. Smallest fix per the ticket: the real fix
# (a process/queue boundary -- arq/Celery + Redis, or a ProcessPoolExecutor so
# torch's own CPU usage is isolated from the request-serving process) is a new
# dependency, out of scope here (docs/api-design.md #2, non-negotiable #5).
_JOB_EXECUTOR_MAX_WORKERS = 4
_JOB_EXECUTOR = ThreadPoolExecutor(
    max_workers=_JOB_EXECUTOR_MAX_WORKERS, thread_name_prefix="pyquant-job"
)


def get_job_executor() -> ThreadPoolExecutor:
    """FastAPI dependency: the dedicated executor /train and /backtest jobs run on."""
    return _JOB_EXECUTOR
