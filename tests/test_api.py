"""Tests for the PYQ-261 FastAPI service layer (network-free, mocked domain calls).

Needs the 'api' extra (fastapi/uvicorn), which CI's default job does not install --
skips cleanly there, the same disposition already used for PYQ-253's Optuna tests
(and PYQ-308's precedent: verify a real-dependency integration locally, don't gate
default CI on an optional extra).
"""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

try:
    import fastapi.testclient as fastapi_testclient
except Exception as exc:  # noqa: BLE001 - see comment: any failure here means "skip"
    # Deliberately broader than pytest.importorskip's ImportError-only net.
    # starlette.testclient raises RuntimeError (not ImportError) when fastapi is
    # importable but no HTTP client (httpx/httpx2) is installed alongside it --
    # a state CI has actually produced (a shared-venv step installing fastapi
    # without also installing a test client). This file's whole premise is "the
    # api extra isn't fully usable here, skip cleanly"; a narrower catch would
    # turn that exact case back into a collection error instead of a skip.
    pytest.skip(f"fastapi.testclient not usable: {exc}", allow_module_level=True)

from pyquant.analysis.forecast import Forecast  # noqa: E402
from pyquant.analysis.interpret import Interpretation  # noqa: E402
from pyquant.analysis.metrics import EvaluationMetrics  # noqa: E402
from pyquant.api import (
    deps,  # noqa: E402
    keystore,  # noqa: E402
)
from pyquant.api import jobs as jobs_mod  # noqa: E402
from pyquant.api.app import app  # noqa: E402
from pyquant.api.jobs import JobRegistry  # noqa: E402
from pyquant.cli import app as cli_app_mod  # noqa: E402
from pyquant.models.tft import BacktestResult, TrainResult  # noqa: E402

TestClient = fastapi_testclient.TestClient

client = TestClient(app)
cli_runner = CliRunner()


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _bypass_auth():
    """Most tests aren't about auth; the auth-specific tests override this back.

    Returns a full-scope identity (not ``None``, PYQ-281) so routes gated by
    ``require_scope("train")`` -- which itself depends on ``require_api_key`` --
    also resolve cleanly for tests that aren't exercising scope enforcement.
    """
    app.dependency_overrides[deps.require_api_key] = lambda: keystore.ApiKey(
        id="test", name="test-bypass", scopes=keystore.ALLOWED_SCOPES
    )
    yield


def _fake_forecast(symbol="AAPL", predictions=None):
    dates = pd.bdate_range("2024-01-01", periods=20)
    if predictions is None:
        predictions = np.array([[95.0, 105.0, 115.0]] * 5)
    return Forecast(
        symbol=symbol,
        last_date=dates[-1],
        current_price=100.0,
        quantiles=[0.1, 0.5, 0.9],
        predictions=predictions,
        history=pd.Series(np.linspace(90, 100, 20), index=dates),
    )


class _FakeBundleCache:
    """A stand-in for deps.BundleCache that never touches disk."""

    def __init__(self, bundle=object(), raises: Exception | None = None):
        self._bundle = bundle
        self._raises = raises

    def get(self, name, settings):
        if self._raises:
            raise self._raises
        return self._bundle

    def invalidate(self, name):
        pass


def _override_settings_and_cache(bundle_cache=None):
    app.dependency_overrides[deps.get_settings] = lambda: object()
    app.dependency_overrides[deps.get_bundle_cache] = lambda: bundle_cache or _FakeBundleCache()


def _fake_train_result(symbols=("AAPL",)):
    ev = EvaluationMetrics(
        model_mae=1.0, baseline_mae=1.2, directional_accuracy=0.6, calibration_coverage=0.8
    )
    return TrainResult(
        symbols=list(symbols),
        bundle_dir=Path("/tmp/checkpoints/" + "_".join(symbols)),
        val_loss=0.1,
        n_features=10,
        epochs_run=5,
        evaluation=ev,
    )


def _wait_for_job(path: str, timeout: float = 2.0) -> dict:
    """Poll a job status endpoint until it leaves queued/running.

    /train and /backtest now schedule their work on a real dedicated
    ThreadPoolExecutor rather than FastAPI's BackgroundTasks (bugs.md#pyq-163)
    -- unlike BackgroundTasks, which TestClient happened to run synchronously
    before returning the response, an executor job is genuinely concurrent
    with the request that queued it, so a status check right after POST is no
    longer guaranteed to see a finished job. Poll instead of assuming.
    """
    deadline = time.monotonic() + timeout
    body = {}
    while time.monotonic() < deadline:
        body = client.get(path).json()
        if body.get("status") not in ("queued", "running"):
            return body
        time.sleep(0.01)
    raise AssertionError(f"job at {path!r} did not finish within {timeout}s: {body}")


def _fake_backtest_result(symbol="AAPL"):
    ev = EvaluationMetrics(
        model_mae=1.0, baseline_mae=1.2, directional_accuracy=0.6, calibration_coverage=0.8
    )
    return BacktestResult(
        symbol=symbol, n_windows=2, per_window=[ev, ev], aggregated=ev, origins=[10, 15]
    )


def test_healthz_needs_no_auth():
    app.dependency_overrides.pop(deps.require_api_key, None)  # deliberately not bypassed
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_healthz_handler_is_async():
    """Regression guard: a sync `def` here would dispatch through FastAPI's
    run_in_threadpool, sharing anyio's single default worker-thread limiter
    with every other sync endpoint *and* with BackgroundTasks -- a liveness
    probe queuing behind a long-running POST /train job is the failure mode
    this guards against."""
    import inspect

    from pyquant.api.routes.health import healthz

    assert inspect.iscoroutinefunction(healthz)


def test_forecast_returns_the_serialized_forecast(monkeypatch):
    _override_settings_and_cache()
    monkeypatch.setattr(
        "pyquant.api.routes.forecast.generate_forecast", lambda *a, **k: _fake_forecast()
    )

    r = client.get("/forecast/AAPL")

    assert r.status_code == 200
    data = r.json()
    assert data["symbol"] == "AAPL"
    assert data["horizon"] == 5
    assert data["median"][-1] == 105.0


def test_forecast_404_for_an_untrained_symbol():
    _override_settings_and_cache(_FakeBundleCache(raises=FileNotFoundError("no bundle")))
    r = client.get("/forecast/NEVERTRAINED")
    assert r.status_code == 404


def test_forecast_404_does_not_leak_the_absolute_checkpoint_path():
    """The underlying FileNotFoundError names an absolute filesystem path
    (tft.py's `_load`) -- a remote caller only needs "not trained", not the
    server's directory layout."""
    leaky = FileNotFoundError(
        "No trained model for NEVERTRAINED at /home/svc/checkpoints/NEVERTRAINED/model.ckpt. "
        "Run `pyquant train` first."
    )
    _override_settings_and_cache(_FakeBundleCache(raises=leaky))
    r = client.get("/forecast/NEVERTRAINED")
    assert r.status_code == 404
    assert "/home/svc" not in r.json()["detail"]
    assert "NEVERTRAINED" in r.json()["detail"]


def test_explain_404_does_not_leak_the_absolute_checkpoint_path():
    leaky = FileNotFoundError(
        "No trained model for NEVERTRAINED at /home/svc/checkpoints/NEVERTRAINED/model.ckpt. "
        "Run `pyquant train` first."
    )
    _override_settings_and_cache(_FakeBundleCache(raises=leaky))
    r = client.get("/explain/NEVERTRAINED")
    assert r.status_code == 404
    assert "/home/svc" not in r.json()["detail"]


def test_forecast_409_on_a_feature_schema_mismatch(monkeypatch):
    from pyquant.models.tft import FeatureSchemaMismatch

    _override_settings_and_cache()

    def raise_mismatch(*a, **k):
        raise FeatureSchemaMismatch("missing feature X")

    monkeypatch.setattr("pyquant.api.routes.forecast.generate_forecast", raise_mismatch)
    r = client.get("/forecast/AAPL")
    assert r.status_code == 409


def test_explain_returns_bundle_skill(monkeypatch):
    _override_settings_and_cache()
    interp = Interpretation(
        symbol="AAPL",
        feature_importance={"RSI_14": 0.6, "SMA_10": 0.4},
        attention=np.array([0.2, 0.3, 0.5]),
        panel_index=pd.bdate_range("2024-01-01", periods=5),
        bundle_skill=-0.1,
    )
    monkeypatch.setattr("pyquant.api.routes.explain.explain_forecast", lambda *a, **k: interp)

    r = client.get("/explain/AAPL")

    assert r.status_code == 200
    data = r.json()
    assert data["bundle_skill"] == -0.1
    assert {"feature": "RSI_14", "weight": 0.6} in data["feature_importance"]


def test_scan_reports_one_flaky_symbol_without_failing_the_rest(monkeypatch):
    """Same discipline as the CLI's scan (PYQ-113): one bad symbol must not sink
    the whole multi-symbol comparison."""
    _override_settings_and_cache()

    def fake_get_forecast(symbol, settings, bundle_cache):
        if symbol == "BAD":
            raise RuntimeError("transient data-source error")
        return _fake_forecast(symbol=symbol)

    monkeypatch.setattr("pyquant.api.routes.forecast._get_forecast", fake_get_forecast)

    r = client.post("/scan", json={"symbols": ["GOOD", "BAD"]})

    assert r.status_code == 200
    rows = {row["symbol"]: row for row in r.json()}
    assert rows["GOOD"]["status"] == "ok"
    assert rows["BAD"]["status"] == "error"


def test_forecast_serializes_concurrent_requests_against_the_same_bundle(monkeypatch):
    """docs/api-design.md #4: pytorch-forecasting's predict() is not safe to call
    concurrently on one model instance -- two /forecast requests for the same
    symbol must not overlap in time."""
    _override_settings_and_cache()
    call_spans: list[tuple[float, float]] = []
    spans_lock = threading.Lock()

    def slow_generate_forecast(*a, **k):
        start = time.monotonic()
        time.sleep(0.05)
        end = time.monotonic()
        with spans_lock:
            call_spans.append((start, end))
        return _fake_forecast()

    monkeypatch.setattr("pyquant.api.routes.forecast.generate_forecast", slow_generate_forecast)

    threads = [threading.Thread(target=lambda: client.get("/forecast/AAPL")) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(call_spans) == 2
    (_, first_end), (second_start, _) = sorted(call_spans)
    assert second_start >= first_end, f"concurrent calls overlapped: {call_spans}"


# --- prediction lock now times out instead of blocking forever --------------------


def test_acquire_prediction_lock_returns_429_when_the_bundle_is_busy():
    """`with lock:` blocks indefinitely; acquire_prediction_lock must not --
    a caller stuck behind a slow request should get a clean 429, not hang."""
    from fastapi import HTTPException

    lock = deps.get_prediction_lock("BUSYTEST")
    lock.acquire()
    try:
        with pytest.raises(HTTPException) as exc_info:
            with deps.acquire_prediction_lock("BUSYTEST", timeout=0.05):
                pass  # pragma: no cover -- must never be reached
        assert exc_info.value.status_code == 429
    finally:
        lock.release()


def test_acquire_prediction_lock_releases_on_success():
    with deps.acquire_prediction_lock("FREETEST", timeout=1.0):
        pass
    # A second immediate acquire must succeed -- proves the first `with` released it.
    with deps.acquire_prediction_lock("FREETEST", timeout=0.5):
        pass


def test_forecast_returns_429_when_the_bundle_is_busy(monkeypatch):
    _override_settings_and_cache()
    monkeypatch.setattr(deps, "PREDICTION_LOCK_TIMEOUT_SECONDS", 0.05)
    lock = deps.get_prediction_lock("AAPL")
    lock.acquire()
    try:
        r = client.get("/forecast/AAPL")
        assert r.status_code == 429
    finally:
        lock.release()


# --- BundleCache: concurrent misses for the same name must not stampede -----------


def test_bundle_cache_dedupes_concurrent_loads_of_the_same_bundle_name(monkeypatch):
    load_calls: list[str] = []
    calls_lock = threading.Lock()

    def slow_load(name, settings):
        with calls_lock:
            load_calls.append(name)
        time.sleep(0.05)
        return object()

    monkeypatch.setattr("pyquant.api.deps.tft.load", slow_load)

    cache = deps.BundleCache()
    results: list[object] = []
    results_lock = threading.Lock()

    def get():
        bundle = cache.get("AAPL", object())
        with results_lock:
            results.append(bundle)

    threads = [threading.Thread(target=get) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert load_calls == ["AAPL"]  # exactly one real load, not five
    assert len(results) == 5
    assert len({id(r) for r in results}) == 1  # every caller got the same instance


def test_bundle_cache_still_loads_different_names_in_parallel(monkeypatch):
    call_spans: list[tuple[float, float]] = []
    spans_lock = threading.Lock()

    def slow_load(name, settings):
        start = time.monotonic()
        time.sleep(0.05)
        with spans_lock:
            call_spans.append((start, time.monotonic()))
        return object()

    monkeypatch.setattr("pyquant.api.deps.tft.load", slow_load)
    cache = deps.BundleCache()

    threads = [
        threading.Thread(target=lambda s=s: cache.get(s, object())) for s in ("AAPL", "MSFT")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(call_spans) == 2
    (first_start, first_end), (second_start, _) = sorted(call_spans)
    # Overlapping spans (second starts before the first ends) is what "parallel" means here.
    assert second_start < first_end, f"different-name loads serialized: {call_spans}"


# --- PYQ-164: per-name lock registries stay bounded --------------------------------


def test_bounded_lock_registry_evicts_the_least_recently_used_unheld_lock():
    registry = deps._BoundedLockRegistry(max_size=3)
    for name in ("A", "B", "C"):
        registry.get(name)
    registry.get("D")  # over cap; "A" is least-recently-used and unheld
    assert len(registry) == 3


def test_bounded_lock_registry_never_evicts_a_currently_held_lock():
    """A lock a thread is holding (or waiting to acquire) must never be
    replaced out from under it -- a second Lock object created for the same
    name afterward would silently break mutual exclusion for that name."""
    registry = deps._BoundedLockRegistry(max_size=2)
    held = registry.get("HELD")
    held.acquire()
    try:
        for i in range(10):  # far past the cap, all unheld and evictable
            registry.get(f"OTHER{i}")
        assert registry.get("HELD") is held  # same lock object, not re-created
    finally:
        held.release()


def test_prediction_locks_registry_stays_bounded_across_many_symbols(monkeypatch):
    fresh = deps._PredictionLocks()
    fresh._locks.max_size = 4
    monkeypatch.setattr(deps, "_PREDICTION_LOCKS", fresh)

    for i in range(50):
        deps.get_prediction_lock(f"SYM{i}")

    assert len(fresh._locks) <= 4


def test_bundle_cache_load_locks_stay_bounded_across_many_names(monkeypatch):
    monkeypatch.setattr("pyquant.api.deps.tft.load", lambda name, settings: object())
    cache = deps.BundleCache(max_size=100)  # bundle cache itself not under test here
    cache._load_locks.max_size = 4

    for i in range(50):
        cache.get(f"SYM{i}", object())

    assert len(cache._load_locks) <= 4


# --- PYQ-163: /train, /backtest run on a dedicated executor, not the shared pool --


def test_start_train_and_start_backtest_are_async_and_no_longer_take_background_tasks():
    """Regression guard, same shape as test_healthz_handler_is_async: an
    `async def` route that submits to jobs.get_job_executor() directly never
    touches starlette.concurrency.run_in_threadpool -- unlike a sync route or
    a FastAPI BackgroundTasks callback, both of which dispatch through it and
    share anyio's single 40-slot default thread limiter (bugs.md#pyq-163)."""
    import inspect

    from pyquant.api.routes.backtest import start_backtest
    from pyquant.api.routes.train import start_train

    assert inspect.iscoroutinefunction(start_train)
    assert inspect.iscoroutinefunction(start_backtest)
    assert "background_tasks" not in inspect.signature(start_train).parameters
    assert "background_tasks" not in inspect.signature(start_backtest).parameters


def test_train_job_is_dispatched_via_the_dedicated_job_executor(monkeypatch):
    """The acceptance criterion itself: /train's background work must run on
    jobs.get_job_executor()'s pool, not FastAPI's BackgroundTasks. Override
    the executor dependency with a spy wrapping a real ThreadPoolExecutor and
    confirm the job is actually submitted through it."""
    registry = JobRegistry()
    app.dependency_overrides[deps.get_settings] = lambda: object()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: registry
    app.dependency_overrides[deps.get_bundle_cache] = lambda: _FakeBundleCache()

    submissions = []
    real_executor = ThreadPoolExecutor(max_workers=2)

    class _SpyExecutor:
        def submit(self, fn, *a, **k):
            submissions.append(fn)
            return real_executor.submit(fn, *a, **k)

    app.dependency_overrides[jobs_mod.get_job_executor] = lambda: _SpyExecutor()
    monkeypatch.setattr("pyquant.api.routes.train.tft.train", lambda *a, **k: _fake_train_result())

    try:
        r = client.post("/train", json={"symbols": ["AAPL"]})
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        body = _wait_for_job(f"/train/{job_id}")
    finally:
        real_executor.shutdown(wait=True)

    assert body["status"] == "succeeded"
    assert len(submissions) == 1, "training job was not dispatched via the dedicated executor"


def test_backtest_job_is_dispatched_via_the_dedicated_job_executor(monkeypatch):
    registry = JobRegistry()
    app.dependency_overrides[deps.get_settings] = lambda: object()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: registry

    submissions = []
    real_executor = ThreadPoolExecutor(max_workers=2)

    class _SpyExecutor:
        def submit(self, fn, *a, **k):
            submissions.append(fn)
            return real_executor.submit(fn, *a, **k)

    app.dependency_overrides[jobs_mod.get_job_executor] = lambda: _SpyExecutor()
    monkeypatch.setattr(
        "pyquant.api.routes.backtest.tft.walk_forward_backtest",
        lambda *a, **k: _fake_backtest_result(),
    )

    try:
        r = client.post("/backtest", json={"symbol": "AAPL", "windows": 2})
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        body = _wait_for_job(f"/backtest/{job_id}")
    finally:
        real_executor.shutdown(wait=True)

    assert body["status"] == "succeeded"
    assert len(submissions) == 1, "backtest job was not dispatched via the dedicated executor"


def test_a_slow_training_job_does_not_delay_a_concurrent_sync_read_endpoint(monkeypatch):
    """The ticket's own acceptance test, against a route that actually shares
    the pool: GET /forecast/{symbol} is a sync `def` route, dispatched via
    FastAPI's run_in_threadpool (unlike /healthz, which is async and was
    already isolated by bugs.md#pyq-165 -- it would pass this test even on
    the pre-fix code, so it is not a meaningful regression guard for this
    ticket). Block the training job on an Event and confirm a concurrent
    forecast call still returns promptly rather than queuing behind it."""
    _override_settings_and_cache()
    registry = JobRegistry()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: registry

    release = threading.Event()

    def blocking_train(*a, **k):
        assert release.wait(timeout=5), "test setup: job was never released"
        return _fake_train_result()

    monkeypatch.setattr("pyquant.api.routes.train.tft.train", blocking_train)
    monkeypatch.setattr(
        "pyquant.api.routes.forecast.generate_forecast", lambda *a, **k: _fake_forecast()
    )

    try:
        r = client.post("/train", json={"symbols": ["AAPL"]})
        assert r.status_code == 202
        job_id = r.json()["job_id"]

        start = time.monotonic()
        forecast_resp = client.get("/forecast/AAPL")
        elapsed = time.monotonic() - start

        assert forecast_resp.status_code == 200
        assert elapsed < 2.0, (
            f"a concurrent sync route took {elapsed:.2f}s -- queued behind the job?"
        )
    finally:
        release.set()

    body = _wait_for_job(f"/train/{job_id}")
    assert body["status"] == "succeeded"


def test_train_returns_202_and_a_pollable_job_id(monkeypatch):
    registry = JobRegistry()
    app.dependency_overrides[deps.get_settings] = lambda: object()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: registry
    app.dependency_overrides[deps.get_bundle_cache] = lambda: _FakeBundleCache()

    monkeypatch.setattr("pyquant.api.routes.train.tft.train", lambda *a, **k: _fake_train_result())

    r = client.post("/train", json={"symbols": ["AAPL"]})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert r.json()["status"] == "queued"

    # Runs on the dedicated job executor now (bugs.md#pyq-163), genuinely
    # concurrent with this request -- poll rather than assume it's done.
    body = _wait_for_job(f"/train/{job_id}")
    assert body["status"] == "succeeded"
    assert body["result"]["symbols"] == ["AAPL"]


def test_train_job_404_for_an_unknown_id():
    registry = JobRegistry()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: registry
    r = client.get("/train/does-not-exist")
    assert r.status_code == 404


# --- PYQ-162: the documented `failed` job status path had no test coverage ---------


def test_train_job_reports_failed_status_and_error(monkeypatch):
    registry = JobRegistry()
    app.dependency_overrides[deps.get_settings] = lambda: object()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: registry
    app.dependency_overrides[deps.get_bundle_cache] = lambda: _FakeBundleCache()

    def raise_error(*a, **k):
        raise RuntimeError("not enough history for AAPL")

    monkeypatch.setattr("pyquant.api.routes.train.tft.train", raise_error)

    r = client.post("/train", json={"symbols": ["AAPL"]})
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    body = _wait_for_job(f"/train/{job_id}")
    assert body["status"] == "failed"
    assert body["result"] is None
    assert "not enough history for AAPL" in body["error"]


# --- PYQ-161: a second POST /train for a bundle already in flight must not race ----


def test_train_rejects_a_second_request_for_a_bundle_already_in_flight(monkeypatch):
    """Two concurrent fits for the same bundle both mkdir/torch.save into the
    same checkpoint directory -- the second must be rejected, not scheduled."""
    registry = JobRegistry()
    app.dependency_overrides[deps.get_settings] = lambda: object()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: registry
    app.dependency_overrides[deps.get_bundle_cache] = lambda: _FakeBundleCache()

    def slow_train(*a, **k):
        time.sleep(0.1)
        return _fake_train_result()

    monkeypatch.setattr("pyquant.api.routes.train.tft.train", slow_train)

    responses: list[int] = []
    resp_lock = threading.Lock()

    def post():
        r = client.post("/train", json={"symbols": ["AAPL"]})
        with resp_lock:
            responses.append(r.status_code)

    threads = [threading.Thread(target=post) for _ in range(2)]
    for t in threads:
        t.start()
        time.sleep(0.02)  # let the first request register its bundle-name guard first
    for t in threads:
        t.join()

    assert sorted(responses) == [202, 409]


def test_train_conflict_names_the_bundle_in_the_error(monkeypatch):
    registry = JobRegistry()
    app.dependency_overrides[deps.get_settings] = lambda: object()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: registry
    app.dependency_overrides[deps.get_bundle_cache] = lambda: _FakeBundleCache()

    monkeypatch.setattr(
        "pyquant.api.routes.train.tft.train",
        lambda *a, **k: time.sleep(0.1) or _fake_train_result(),
    )

    responses = []

    def post():
        responses.append(client.post("/train", json={"symbols": ["AAPL"], "bundle_name": "AAPL"}))

    threads = [threading.Thread(target=post) for _ in range(2)]
    for t in threads:
        t.start()
        time.sleep(0.02)
    for t in threads:
        t.join()

    conflicts = [r for r in responses if r.status_code == 409]
    assert len(conflicts) == 1
    assert "AAPL" in conflicts[0].json()["detail"]


# --- PYQ-159: JobRegistry is bounded and its mark_* methods tolerate an unknown id --


def test_job_registry_bounds_its_size_under_sustained_job_creation():
    # PYQ-325: eviction now skips queued/running records, so jobs must actually
    # finish (mark_succeeded) to be eviction-eligible -- a bare create() leaves
    # them "queued" forever, which is the exact case the fix protects.
    registry = JobRegistry(max_jobs=3)
    job_ids = []
    for _ in range(5):
        job_id = registry.create(kind="backtest")
        registry.mark_succeeded(job_id, _fake_backtest_result())
        job_ids.append(job_id)
    remaining = [jid for jid in job_ids if registry.get(jid) is not None]
    assert len(remaining) == 3
    assert remaining == job_ids[-3:]  # oldest-first eviction: newest 3 survive


def test_job_registry_mark_methods_are_a_no_op_for_an_unknown_job_id():
    registry = JobRegistry()
    # None of these must raise, unlike the old unguarded self._jobs[job_id].
    registry.mark_running("nope")
    registry.mark_succeeded("nope", _fake_backtest_result())
    registry.mark_failed("nope", "boom")
    assert registry.get("nope") is None


def test_job_registry_eviction_never_drops_a_still_running_job():
    # PYQ-325: a running train job's bundle-name lock must survive eviction --
    # dropping it early lets a second POST /train for the same bundle_name start
    # writing to the same checkpoint directory while the first fit is still live.
    registry = JobRegistry(max_jobs=3)
    running_job_id = registry.try_start_train("AAPL")
    registry.mark_running(running_job_id)

    # Finished (never-started) jobs to force eviction past the cap.
    for _ in range(5):
        registry.create(kind="backtest")

    assert registry.get(running_job_id) is not None
    assert registry._active_bundle_names.get("AAPL") == running_job_id

    # The registry is allowed to grow past max_jobs while a tracked job is
    # still live -- mirrors deps.py's _BoundedLockRegistry never evicting a
    # held lock.
    assert len(registry._order) > registry._max_jobs

    registry.mark_succeeded(running_job_id, _fake_train_result())
    record = registry.get(running_job_id)
    assert record.status == "succeeded"
    assert "AAPL" not in registry._active_bundle_names


def test_train_request_period_overrides_settings_data_period(monkeypatch):
    from pyquant.config import Settings

    captured = {}
    registry = JobRegistry()
    app.dependency_overrides[deps.get_settings] = lambda: Settings()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: registry
    app.dependency_overrides[deps.get_bundle_cache] = lambda: _FakeBundleCache()

    def fake_train(symbols, settings, **kwargs):
        captured["period"] = settings.data.period
        return _fake_train_result()

    monkeypatch.setattr("pyquant.api.routes.train.tft.train", fake_train)
    r = client.post("/train", json={"symbols": ["AAPL"], "period": "10y"})
    assert r.status_code == 202
    assert captured["period"] == "10y"


def test_train_rejects_an_empty_symbol_list():
    app.dependency_overrides[deps.get_settings] = lambda: object()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: JobRegistry()
    app.dependency_overrides[deps.get_bundle_cache] = lambda: _FakeBundleCache()
    r = client.post("/train", json={"symbols": []})
    assert r.status_code == 422


# --- PYQ-145: request-body symbol/bundle_name fields cannot become path traversal


def test_train_rejects_a_path_traversal_bundle_name():
    """A JSON body field isn't subject to Starlette's `/`-rejecting path-parameter
    matching the way GET /forecast/{symbol} is -- unvalidated, it reaches
    _bundle_dir -> mkdir/torch.save directly."""
    app.dependency_overrides[deps.get_settings] = lambda: object()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: JobRegistry()
    app.dependency_overrides[deps.get_bundle_cache] = lambda: _FakeBundleCache()
    r = client.post("/train", json={"symbols": ["AAPL"], "bundle_name": "../../etc"})
    assert r.status_code == 422


def test_train_rejects_a_path_traversal_symbol():
    app.dependency_overrides[deps.get_settings] = lambda: object()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: JobRegistry()
    app.dependency_overrides[deps.get_bundle_cache] = lambda: _FakeBundleCache()
    r = client.post("/train", json={"symbols": ["../x"]})
    assert r.status_code == 422


def test_scan_rejects_a_path_traversal_symbol():
    _override_settings_and_cache()
    r = client.post("/scan", json={"symbols": ["../x"]})
    assert r.status_code == 422


# --- unbounded request fields: one authenticated request must not be able to -------
# --- demand unbounded vendor-quota spend or training compute -----------------------


def test_scan_rejects_more_symbols_than_the_cap():
    from pyquant.api.schemas import MAX_SYMBOLS_PER_REQUEST

    _override_settings_and_cache()
    too_many = [f"SYM{i}" for i in range(MAX_SYMBOLS_PER_REQUEST + 1)]
    r = client.post("/scan", json={"symbols": too_many})
    assert r.status_code == 422


def test_train_rejects_more_symbols_than_the_cap():
    from pyquant.api.schemas import MAX_SYMBOLS_PER_REQUEST

    app.dependency_overrides[deps.get_settings] = lambda: object()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: JobRegistry()
    app.dependency_overrides[deps.get_bundle_cache] = lambda: _FakeBundleCache()
    too_many = [f"SYM{i}" for i in range(MAX_SYMBOLS_PER_REQUEST + 1)]
    r = client.post("/train", json={"symbols": too_many})
    assert r.status_code == 422


def test_train_rejects_an_out_of_range_epochs():
    from pyquant.api.schemas import MAX_EPOCHS

    app.dependency_overrides[deps.get_settings] = lambda: object()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: JobRegistry()
    app.dependency_overrides[deps.get_bundle_cache] = lambda: _FakeBundleCache()
    assert client.post("/train", json={"symbols": ["AAPL"], "epochs": 0}).status_code == 422
    assert (
        client.post("/train", json={"symbols": ["AAPL"], "epochs": MAX_EPOCHS + 1}).status_code
        == 422
    )


def test_train_rejects_an_invalid_period():
    app.dependency_overrides[deps.get_settings] = lambda: object()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: JobRegistry()
    app.dependency_overrides[deps.get_bundle_cache] = lambda: _FakeBundleCache()
    r = client.post("/train", json={"symbols": ["AAPL"], "period": "'; DROP TABLE x"})
    assert r.status_code == 422


def test_backtest_rejects_an_out_of_range_windows():
    from pyquant.api.schemas import MAX_BACKTEST_WINDOWS

    app.dependency_overrides[deps.get_settings] = lambda: object()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: JobRegistry()
    assert client.post("/backtest", json={"symbol": "AAPL", "windows": 0}).status_code == 422
    assert (
        client.post(
            "/backtest", json={"symbol": "AAPL", "windows": MAX_BACKTEST_WINDOWS + 1}
        ).status_code
        == 422
    )


def test_backtest_rejects_an_invalid_period():
    app.dependency_overrides[deps.get_settings] = lambda: object()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: JobRegistry()
    r = client.post("/backtest", json={"symbol": "AAPL", "period": "forever"})
    assert r.status_code == 422


def test_bundle_dir_never_resolves_outside_checkpoint_dir(tmp_path):
    from pyquant.config import Settings
    from pyquant.models import tft as tft_mod

    settings = Settings(checkpoint_dir=tmp_path / "checkpoints")
    for bad_name in ("..", "../../etc", "foo/../../bar"):
        with pytest.raises(ValueError, match="checkpoint_dir"):
            tft_mod._bundle_dir(settings, bad_name)


# --- PYQ-271: POST /backtest -> job id; GET /backtest/{job_id} ---------------------


def test_backtest_returns_202_and_a_pollable_job_id(monkeypatch):
    registry = JobRegistry()
    app.dependency_overrides[deps.get_settings] = lambda: object()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: registry

    monkeypatch.setattr(
        "pyquant.api.routes.backtest.tft.walk_forward_backtest",
        lambda *a, **k: _fake_backtest_result(),
    )

    r = client.post("/backtest", json={"symbol": "AAPL", "windows": 2})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert r.json()["status"] == "queued"

    body = _wait_for_job(f"/backtest/{job_id}")
    assert body["status"] == "succeeded"
    assert body["result"]["symbol"] == "AAPL"
    assert body["result"]["n_windows"] == 2
    assert body["result"]["origins"] == [10, 15]
    assert len(body["result"]["per_window"]) == 2


def test_backtest_job_404_for_an_unknown_id():
    registry = JobRegistry()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: registry
    r = client.get("/backtest/does-not-exist")
    assert r.status_code == 404


def test_backtest_job_reports_failed_status_and_error(monkeypatch):
    registry = JobRegistry()
    app.dependency_overrides[deps.get_settings] = lambda: object()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: registry

    def raise_error(*a, **k):
        raise RuntimeError("not enough history for AAPL")

    monkeypatch.setattr("pyquant.api.routes.backtest.tft.walk_forward_backtest", raise_error)

    r = client.post("/backtest", json={"symbol": "AAPL"})
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    body = _wait_for_job(f"/backtest/{job_id}")
    assert body["status"] == "failed"
    assert body["result"] is None
    assert "not enough history for AAPL" in body["error"]


def test_backtest_rejects_a_path_traversal_symbol():
    r = client.post("/backtest", json={"symbol": "../x"})
    assert r.status_code == 422


def test_backtest_request_period_overrides_settings_data_period(monkeypatch):
    from pyquant.config import Settings

    captured = {}
    registry = JobRegistry()
    app.dependency_overrides[deps.get_settings] = lambda: Settings()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: registry

    def fake_backtest(symbol, settings, **kwargs):
        captured["period"] = settings.data.period
        return _fake_backtest_result(symbol)

    monkeypatch.setattr("pyquant.api.routes.backtest.tft.walk_forward_backtest", fake_backtest)
    r = client.post("/backtest", json={"symbol": "AAPL", "period": "10y"})
    assert r.status_code == 202
    assert captured["period"] == "10y"


# --- PYQ-327: an identical in-flight /backtest request must not duplicate work ----


def test_backtest_deduplicates_an_identical_in_flight_request(monkeypatch):
    """Two concurrent identical POST /backtest calls must not both spin up a
    full multi-window Lightning run on the shared executor -- the second
    should get back the first's job id rather than scheduling a duplicate."""
    registry = JobRegistry()
    app.dependency_overrides[deps.get_settings] = lambda: object()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: registry

    call_count = 0
    call_count_lock = threading.Lock()

    def slow_backtest(*a, **k):
        nonlocal call_count
        with call_count_lock:
            call_count += 1
        time.sleep(0.1)
        return _fake_backtest_result()

    monkeypatch.setattr("pyquant.api.routes.backtest.tft.walk_forward_backtest", slow_backtest)

    responses: list = []
    resp_lock = threading.Lock()

    def post():
        r = client.post("/backtest", json={"symbol": "AAPL", "windows": 2})
        with resp_lock:
            responses.append(r)

    threads = [threading.Thread(target=post) for _ in range(2)]
    for t in threads:
        t.start()
        time.sleep(0.02)  # let the first request register its dedup key first
    for t in threads:
        t.join()

    assert all(r.status_code == 202 for r in responses)
    job_ids = {r.json()["job_id"] for r in responses}
    assert len(job_ids) == 1

    _wait_for_job(f"/backtest/{job_ids.pop()}")
    assert call_count == 1


def test_backtest_does_not_deduplicate_requests_with_different_parameters(monkeypatch):
    """A different `windows` is a different request -- both must actually run,
    not get folded into one job the way an identical retry should be."""
    registry = JobRegistry()
    app.dependency_overrides[deps.get_settings] = lambda: object()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: registry

    monkeypatch.setattr(
        "pyquant.api.routes.backtest.tft.walk_forward_backtest",
        lambda *a, **k: _fake_backtest_result(),
    )

    r1 = client.post("/backtest", json={"symbol": "AAPL", "windows": 2})
    r2 = client.post("/backtest", json={"symbol": "AAPL", "windows": 3})
    assert r1.status_code == 202
    assert r2.status_code == 202
    assert r1.json()["job_id"] != r2.json()["job_id"]


def test_backtest_dedup_key_releases_once_the_job_finishes(monkeypatch):
    """A later request with the same parameters, after the first job has
    finished, must start a fresh job rather than being folded into a
    long-gone one -- the dedup key is only held while genuinely in flight."""
    registry = JobRegistry()
    app.dependency_overrides[deps.get_settings] = lambda: object()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: registry

    monkeypatch.setattr(
        "pyquant.api.routes.backtest.tft.walk_forward_backtest",
        lambda *a, **k: _fake_backtest_result(),
    )

    first = client.post("/backtest", json={"symbol": "AAPL", "windows": 2})
    _wait_for_job(f"/backtest/{first.json()['job_id']}")

    second = client.post("/backtest", json={"symbol": "AAPL", "windows": 2})
    assert second.status_code == 202
    assert second.json()["job_id"] != first.json()["job_id"]


def test_train_job_id_not_found_via_backtest_endpoint(monkeypatch):
    """Job ids share one registry (PYQ-271 reuses JobRegistry rather than
    building a second job mechanism) but not one namespace across kinds --
    polling the wrong endpoint 404s rather than trying to serialize a
    TrainResult as a BacktestResponse."""
    registry = JobRegistry()
    app.dependency_overrides[deps.get_settings] = lambda: object()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: registry
    app.dependency_overrides[deps.get_bundle_cache] = lambda: _FakeBundleCache()

    monkeypatch.setattr("pyquant.api.routes.train.tft.train", lambda *a, **k: _fake_train_result())
    job_id = client.post("/train", json={"symbols": ["AAPL"]}).json()["job_id"]

    r = client.get(f"/backtest/{job_id}")
    assert r.status_code == 404


def _configured_keystore(tmp_path, monkeypatch, scopes=("read",), name="test-key"):
    """Point PYQUANT_API_KEYS_DB at a fresh temp store and issue one real key.

    Returns the raw key string -- the only place it's ever available outside
    `keystore.create_key`'s own return value, per PYQ-281's "shown once" rule.
    """
    monkeypatch.setenv("PYQUANT_API_KEYS_DB", str(tmp_path / "api_keys.db"))
    raw_key, _record = keystore.create_key(keystore.resolve_db_path(), name, scopes)
    return raw_key


def test_require_api_key_rejects_a_non_ascii_key_with_401_not_500(tmp_path, monkeypatch):
    """PYQ-145: Starlette decodes headers as latin-1, so a byte >127 in
    X-API-Key produces a non-ASCII str -- hmac.compare_digest used to raise
    TypeError on that, which FastAPI's default handler turned into a 500."""
    app.dependency_overrides.pop(deps.require_api_key, None)
    _configured_keystore(tmp_path, monkeypatch)
    _override_settings_and_cache()

    # httpx's header-value normalization ASCII-encodes str values by default, so
    # a raw non-ASCII byte must be passed as bytes to reach the server at all --
    # matching what Starlette's own latin-1 header decoding accepts on the wire.
    r = client.get("/forecast/AAPL", headers={"X-API-Key": "café".encode("latin-1")})

    assert r.status_code == 401


# --- auth (deliberately does NOT use the _bypass_auth fixture's override) --------


def test_require_api_key_fails_loudly_when_unconfigured(tmp_path, monkeypatch):
    app.dependency_overrides.pop(deps.require_api_key, None)
    monkeypatch.setenv("PYQUANT_API_KEYS_DB", str(tmp_path / "empty.db"))
    monkeypatch.delenv("PYQUANT_API_ALLOW_UNAUTHENTICATED", raising=False)
    r = client.get("/forecast/AAPL")
    assert r.status_code == 500  # not a silent 200 -- an unconfigured gate must not open


def test_app_refuses_to_start_when_api_keys_unconfigured(tmp_path, monkeypatch):
    """The lifespan hook makes the same check require_api_key does, once at
    boot -- a misconfigured deployment should crash-loop immediately rather
    than start, pass a liveness check, and 500 on the first real request.
    Only entering TestClient as a context manager runs ASGI lifespan events
    (verified against the installed Starlette source); the module-level
    `client` used elsewhere never does, so this is the one test that does."""
    monkeypatch.setenv("PYQUANT_API_KEYS_DB", str(tmp_path / "empty.db"))
    monkeypatch.delenv("PYQUANT_API_ALLOW_UNAUTHENTICATED", raising=False)
    with pytest.raises(RuntimeError, match="No active API keys"):
        with TestClient(app):
            pass  # pragma: no cover -- startup must fail before this runs


def test_app_starts_cleanly_when_api_keys_configured(tmp_path, monkeypatch):
    _configured_keystore(tmp_path, monkeypatch)
    with TestClient(app) as c:
        assert c.get("/healthz").status_code == 200


def test_app_starts_cleanly_with_the_explicit_dev_opt_out(tmp_path, monkeypatch):
    monkeypatch.setenv("PYQUANT_API_KEYS_DB", str(tmp_path / "empty.db"))
    monkeypatch.setenv("PYQUANT_API_ALLOW_UNAUTHENTICATED", "1")
    with TestClient(app) as c:
        assert c.get("/healthz").status_code == 200


def test_require_api_key_allows_the_explicit_dev_opt_out(tmp_path, monkeypatch):
    app.dependency_overrides.pop(deps.require_api_key, None)
    monkeypatch.setenv("PYQUANT_API_KEYS_DB", str(tmp_path / "empty.db"))
    monkeypatch.setenv("PYQUANT_API_ALLOW_UNAUTHENTICATED", "1")
    _override_settings_and_cache()
    monkeypatch.setattr(
        "pyquant.api.routes.forecast.generate_forecast", lambda *a, **k: _fake_forecast()
    )
    r = client.get("/forecast/AAPL")
    assert r.status_code == 200


def test_require_api_key_rejects_a_missing_or_wrong_key(tmp_path, monkeypatch):
    app.dependency_overrides.pop(deps.require_api_key, None)
    _configured_keystore(tmp_path, monkeypatch, name="correct-key")
    r_missing = client.get("/forecast/AAPL")
    assert r_missing.status_code == 401
    r_wrong = client.get("/forecast/AAPL", headers={"x-api-key": "wrong"})
    assert r_wrong.status_code == 401


def test_require_api_key_accepts_a_correct_key(tmp_path, monkeypatch):
    app.dependency_overrides.pop(deps.require_api_key, None)
    raw_key = _configured_keystore(tmp_path, monkeypatch, name="correct-key")
    _override_settings_and_cache()
    monkeypatch.setattr(
        "pyquant.api.routes.forecast.generate_forecast", lambda *a, **k: _fake_forecast()
    )
    r = client.get("/forecast/AAPL", headers={"x-api-key": raw_key})
    assert r.status_code == 200


def test_require_api_key_rejects_a_revoked_key(tmp_path, monkeypatch):
    """PYQ-281's identity model makes this possible at all -- the old flat
    PYQUANT_API_KEYS list had no way to invalidate one key without rotating
    every key sharing that env var. A second, still-active key is issued
    alongside the revoked one so the store is genuinely "configured" (has an
    active key) -- otherwise this collapses into the unconfigured-service 500
    case rather than testing revocation's own 401."""
    app.dependency_overrides.pop(deps.require_api_key, None)
    monkeypatch.setenv("PYQUANT_API_KEYS_DB", str(tmp_path / "api_keys.db"))
    db_path = keystore.resolve_db_path()
    raw_key, record = keystore.create_key(db_path, "revoke-me", ["read"])
    keystore.create_key(db_path, "still-active", ["read"])
    assert keystore.revoke_key(db_path, record.id) is True

    r = client.get("/forecast/AAPL", headers={"x-api-key": raw_key})
    assert r.status_code == 401


def test_require_scope_rejects_a_read_only_key_on_train(tmp_path, monkeypatch):
    """PYQ-281's scopes requirement: a read-only key must not be able to
    trigger a fit and spend the operator's training compute budget."""
    app.dependency_overrides.pop(deps.require_api_key, None)
    raw_key = _configured_keystore(tmp_path, monkeypatch, scopes=("read",))

    r = client.post("/train", json={"symbols": ["AAPL"]}, headers={"x-api-key": raw_key})

    assert r.status_code == 403


def test_require_scope_accepts_a_train_scoped_key_on_train(tmp_path, monkeypatch):
    app.dependency_overrides.pop(deps.require_api_key, None)
    raw_key = _configured_keystore(tmp_path, monkeypatch, scopes=("read", "train"))
    app.dependency_overrides[deps.get_settings] = lambda: object()
    app.dependency_overrides[deps.get_bundle_cache] = lambda: _FakeBundleCache()
    monkeypatch.setattr("pyquant.api.routes.train.tft.train", lambda *a, **k: _fake_train_result())

    r = client.post("/train", json={"symbols": ["AAPL"]}, headers={"x-api-key": raw_key})

    assert r.status_code == 202


# --- PYQ-160: /docs, /redoc, /openapi.json must honor the same auth contract -------


def test_openapi_json_requires_auth(tmp_path, monkeypatch):
    app.dependency_overrides.pop(deps.require_api_key, None)
    raw_key = _configured_keystore(tmp_path, monkeypatch, name="correct-key")

    r_missing = client.get("/openapi.json")
    assert r_missing.status_code == 401

    r_ok = client.get("/openapi.json", headers={"x-api-key": raw_key})
    assert r_ok.status_code == 200
    assert "paths" in r_ok.json()


def test_docs_and_redoc_require_auth(tmp_path, monkeypatch):
    app.dependency_overrides.pop(deps.require_api_key, None)
    raw_key = _configured_keystore(tmp_path, monkeypatch, name="correct-key")

    assert client.get("/docs").status_code == 401
    assert client.get("/redoc").status_code == 401
    assert client.get("/docs", headers={"x-api-key": raw_key}).status_code == 200
    assert client.get("/redoc", headers={"x-api-key": raw_key}).status_code == 200


# --- PYQ-261's explicit acceptance criterion: API and CLI --format json agree -----


def test_forecast_response_matches_the_cli_format_json_field_for_field(monkeypatch):
    """Both front-ends must call forecast_to_dict(); this proves it, rather than
    trusting that the pydantic model's fields happen to have been typed the same."""
    fc = _fake_forecast()

    monkeypatch.setattr(cli_app_mod, "generate_forecast", lambda *a, **k: fc)

    class NoOptions:
        put_call_ratio = None

    monkeypatch.setattr(cli_app_mod, "fetch_options_snapshot", lambda s: NoOptions())
    cli_result = cli_runner.invoke(
        cli_app_mod.app, ["--format", "json", "forecast", "AAPL", "--no-chart"]
    )
    assert cli_result.exit_code == 0
    cli_payload = json.loads(cli_result.stdout)

    _override_settings_and_cache()
    monkeypatch.setattr("pyquant.api.routes.forecast.generate_forecast", lambda *a, **k: fc)
    api_payload = client.get("/forecast/AAPL").json()

    assert api_payload == cli_payload


# --- PYQ-283: GET /symbols, GET /metrics/{symbol} ---------------------------------


def _write_bundle_meta(checkpoint_dir: Path, symbol: str, **overrides) -> None:
    bundle_dir = checkpoint_dir / symbol
    bundle_dir.mkdir(parents=True)
    meta = {
        "symbol": symbol,
        "trained_at": "2026-01-01T00:00:00",
        "evaluation": {
            "model_mae": 1.5,
            "baseline_mae": 2.0,
            "directional_accuracy": 0.6,
            "calibration_coverage": 0.8,
            "n_samples": 25,
            "n_points": 125,
            "quantile_exceedance": {},
            "pinball_losses": {},
            "crps": 0.5,
            "winkler_score": 3.0,
            "pit": [0.4, 0.6],
        },
    }
    meta.update(overrides)
    (bundle_dir / "meta.json").write_text(json.dumps(meta))


def test_list_symbols_returns_every_trained_bundle_most_recent_first(tmp_path):
    from pyquant.config import Settings

    settings = Settings(checkpoint_dir=tmp_path / "checkpoints")
    _write_bundle_meta(settings.checkpoint_dir, "AAA", trained_at="2026-01-01T00:00:00")
    _write_bundle_meta(settings.checkpoint_dir, "BBB", trained_at="2026-02-01T00:00:00")
    app.dependency_overrides[deps.get_settings] = lambda: settings

    r = client.get("/symbols")
    assert r.status_code == 200
    body = r.json()
    assert [row["symbol"] for row in body] == ["BBB", "AAA"]
    assert body[0]["bundle_skill"] == pytest.approx(0.25)


def test_list_symbols_is_empty_when_nothing_is_trained(tmp_path):
    from pyquant.config import Settings

    app.dependency_overrides[deps.get_settings] = lambda: Settings(
        checkpoint_dir=tmp_path / "checkpoints"
    )
    r = client.get("/symbols")
    assert r.status_code == 200
    assert r.json() == []


def test_get_metrics_returns_the_bundles_recorded_evaluation(tmp_path):
    from pyquant.config import Settings

    settings = Settings(checkpoint_dir=tmp_path / "checkpoints")
    _write_bundle_meta(settings.checkpoint_dir, "AAPL")
    app.dependency_overrides[deps.get_settings] = lambda: settings

    r = client.get("/metrics/AAPL")
    assert r.status_code == 200
    body = r.json()
    assert body["model_mae"] == pytest.approx(1.5)
    assert body["skill_vs_baseline"] == pytest.approx(0.25)
    assert body["n_samples"] == 25
    assert body["effective_n_samples"] == 5  # 25 samples / horizon 5 (125/25)


def test_get_metrics_404_for_an_untrained_symbol(tmp_path):
    from pyquant.config import Settings

    app.dependency_overrides[deps.get_settings] = lambda: Settings(
        checkpoint_dir=tmp_path / "checkpoints"
    )
    r = client.get("/metrics/NOPE")
    assert r.status_code == 404


def test_get_metrics_rejects_a_path_traversal_symbol(tmp_path):
    from pyquant.config import Settings

    app.dependency_overrides[deps.get_settings] = lambda: Settings(
        checkpoint_dir=tmp_path / "checkpoints"
    )
    r = client.get("/metrics/..%2F..%2Fetc")
    assert r.status_code in (404, 422)
