"""Tests for the PYQ-261 FastAPI service layer (network-free, mocked domain calls).

Needs the 'api' extra (fastapi/uvicorn), which CI's default job does not install --
skips cleanly there, the same disposition already used for PYQ-253's Optuna tests
(and PYQ-308's precedent: verify a real-dependency integration locally, don't gate
default CI on an optional extra).
"""

import json
import threading
import time
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
from pyquant.api import deps  # noqa: E402
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
    """Most tests aren't about auth; the auth-specific tests override this back."""
    app.dependency_overrides[deps.require_api_key] = lambda: None
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


def test_train_returns_202_and_a_pollable_job_id(monkeypatch):
    registry = JobRegistry()
    app.dependency_overrides[deps.get_settings] = lambda: object()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: registry
    app.dependency_overrides[deps.get_bundle_cache] = lambda: _FakeBundleCache()

    monkeypatch.setattr(
        "pyquant.api.routes.train.tft.train", lambda *a, **k: _fake_train_result()
    )

    r = client.post("/train", json={"symbols": ["AAPL"]})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert r.json()["status"] == "queued"

    # TestClient runs BackgroundTasks synchronously before the response returns
    # in-process, so the job should already be resolved by the time we poll it.
    status = client.get(f"/train/{job_id}")
    assert status.status_code == 200
    body = status.json()
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

    status = client.get(f"/train/{job_id}")
    assert status.status_code == 200
    body = status.json()
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
        "pyquant.api.routes.train.tft.train", lambda *a, **k: time.sleep(0.1) or _fake_train_result()
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
    registry = JobRegistry(max_jobs=3)
    job_ids = [registry.create(kind="backtest") for _ in range(5)]
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

    status = client.get(f"/backtest/{job_id}")
    assert status.status_code == 200
    body = status.json()
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

    status = client.get(f"/backtest/{job_id}")
    assert status.status_code == 200
    body = status.json()
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


def test_train_job_id_not_found_via_backtest_endpoint(monkeypatch):
    """Job ids share one registry (PYQ-271 reuses JobRegistry rather than
    building a second job mechanism) but not one namespace across kinds --
    polling the wrong endpoint 404s rather than trying to serialize a
    TrainResult as a BacktestResponse."""
    registry = JobRegistry()
    app.dependency_overrides[deps.get_settings] = lambda: object()
    app.dependency_overrides[jobs_mod.get_job_registry] = lambda: registry
    app.dependency_overrides[deps.get_bundle_cache] = lambda: _FakeBundleCache()

    monkeypatch.setattr(
        "pyquant.api.routes.train.tft.train", lambda *a, **k: _fake_train_result()
    )
    job_id = client.post("/train", json={"symbols": ["AAPL"]}).json()["job_id"]

    r = client.get(f"/backtest/{job_id}")
    assert r.status_code == 404


def test_require_api_key_rejects_a_non_ascii_key_with_401_not_500(monkeypatch):
    """PYQ-145: Starlette decodes headers as latin-1, so a byte >127 in
    X-API-Key produces a non-ASCII str -- hmac.compare_digest used to raise
    TypeError on that, which FastAPI's default handler turned into a 500."""
    app.dependency_overrides.pop(deps.require_api_key, None)
    monkeypatch.setenv("PYQUANT_API_KEYS", "a-real-key")
    _override_settings_and_cache()

    # httpx's header-value normalization ASCII-encodes str values by default, so
    # a raw non-ASCII byte must be passed as bytes to reach the server at all --
    # matching what Starlette's own latin-1 header decoding accepts on the wire.
    r = client.get("/forecast/AAPL", headers={"X-API-Key": "café".encode("latin-1")})

    assert r.status_code == 401


# --- auth (deliberately does NOT use the _bypass_auth fixture's override) --------


def test_require_api_key_fails_loudly_when_unconfigured(monkeypatch):
    app.dependency_overrides.pop(deps.require_api_key, None)
    monkeypatch.delenv("PYQUANT_API_KEYS", raising=False)
    monkeypatch.delenv("PYQUANT_API_ALLOW_UNAUTHENTICATED", raising=False)
    r = client.get("/forecast/AAPL")
    assert r.status_code == 500  # not a silent 200 -- an unconfigured gate must not open


def test_require_api_key_allows_the_explicit_dev_opt_out(monkeypatch):
    app.dependency_overrides.pop(deps.require_api_key, None)
    monkeypatch.delenv("PYQUANT_API_KEYS", raising=False)
    monkeypatch.setenv("PYQUANT_API_ALLOW_UNAUTHENTICATED", "1")
    _override_settings_and_cache()
    monkeypatch.setattr(
        "pyquant.api.routes.forecast.generate_forecast", lambda *a, **k: _fake_forecast()
    )
    r = client.get("/forecast/AAPL")
    assert r.status_code == 200


def test_require_api_key_rejects_a_missing_or_wrong_key(monkeypatch):
    app.dependency_overrides.pop(deps.require_api_key, None)
    monkeypatch.setenv("PYQUANT_API_KEYS", "correct-key")
    r_missing = client.get("/forecast/AAPL")
    assert r_missing.status_code == 401
    r_wrong = client.get("/forecast/AAPL", headers={"x-api-key": "wrong"})
    assert r_wrong.status_code == 401


def test_require_api_key_accepts_a_correct_key(monkeypatch):
    app.dependency_overrides.pop(deps.require_api_key, None)
    monkeypatch.setenv("PYQUANT_API_KEYS", "correct-key,another-key")
    _override_settings_and_cache()
    monkeypatch.setattr(
        "pyquant.api.routes.forecast.generate_forecast", lambda *a, **k: _fake_forecast()
    )
    r = client.get("/forecast/AAPL", headers={"x-api-key": "correct-key"})
    assert r.status_code == 200


# --- PYQ-160: /docs, /redoc, /openapi.json must honor the same auth contract -------


def test_openapi_json_requires_auth(monkeypatch):
    app.dependency_overrides.pop(deps.require_api_key, None)
    monkeypatch.setenv("PYQUANT_API_KEYS", "correct-key")

    r_missing = client.get("/openapi.json")
    assert r_missing.status_code == 401

    r_ok = client.get("/openapi.json", headers={"x-api-key": "correct-key"})
    assert r_ok.status_code == 200
    assert "paths" in r_ok.json()


def test_docs_and_redoc_require_auth(monkeypatch):
    app.dependency_overrides.pop(deps.require_api_key, None)
    monkeypatch.setenv("PYQUANT_API_KEYS", "correct-key")

    assert client.get("/docs").status_code == 401
    assert client.get("/redoc").status_code == 401
    assert client.get("/docs", headers={"x-api-key": "correct-key"}).status_code == 200
    assert client.get("/redoc", headers={"x-api-key": "correct-key"}).status_code == 200


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
