# HTTP API

A FastAPI service over the same `analysis/` and `models/` calls the CLI makes (PYQ-261).
It was additive rather than a rewrite because of the two structural rules in
{ref}`Architecture <what-each-layer-owns>`: `cli/app.py` was already a mapping from command
to a plain function returning a plain dataclass, so a second front-end wraps the same
functions instead of reimplementing them.

[The design note](api-design.md) records the shape and the reasoning behind each choice;
this page is how to run what was built.

## Running it

```bash
uv sync --extra api
export PYQUANT_API_KEYS=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
uv run uvicorn pyquant.api.app:app
```

Interactive documentation — generated from the same pydantic response models the endpoints
return — is at `/docs`, and the OpenAPI schema at `/openapi.json`.

The service reads bundles from `checkpoints/` on the local filesystem, so a symbol must
have been trained (by `pyquant train` or by `POST /train`) before `/forecast` or `/explain`
can answer for it.

## Authentication

Every endpoint except `/healthz` requires an `X-API-Key` header matching one of the
comma-separated keys in the `PYQUANT_API_KEYS` environment variable. Comparison is
constant-time, so response timing cannot leak a valid key one byte at a time.

Keys come from the environment rather than from {py:class}`~pyquant.config.Settings`
deliberately: a `Settings` field can end up in a `meta.json`, a log line or a cache
fingerprint, and this project's secrets rule is that key values never do.

**An unconfigured service fails loudly with `500`, it does not fall open.** A public
forecasting endpoint with no key check would spend the operator's FRED, Finnhub and Yahoo
quota on every caller. For local development only:

```bash
PYQUANT_API_ALLOW_UNAUTHENTICATED=1 uv run uvicorn pyquant.api.app:app
```

## Endpoints

| Method | Path | Auth | Returns |
|---|---|---|---|
| `GET` | `/healthz` | — | `{"status": "ok"}`. 200 whenever the process is up. |
| `GET` | `/forecast/{symbol}` | ✔ | p10/p50/p90 quantile forecast. |
| `POST` | `/scan` | ✔ | One comparison row per requested symbol. |
| `GET` | `/explain/{symbol}` | ✔ | Feature importances and temporal attention. |
| `POST` | `/train` | ✔ | `202` with a `job_id` to poll. |
| `GET` | `/train/{job_id}` | ✔ | That job's status, and its result once it succeeds. |
| `POST` | `/backtest` | ✔ | `202` with a `job_id` to poll. |
| `GET` | `/backtest/{job_id}` | ✔ | That job's status, and its result once it succeeds. |

Symbols are upper-cased on the way in, so `/forecast/aapl` and `/forecast/AAPL` are the
same bundle.

`/docs`, `/redoc` and `/openapi.json` also require `X-API-Key` (bugs.md#pyq-160) — unlike
FastAPI's defaults, which mount independently of any router's `dependencies=` and would
otherwise disclose the full endpoint/schema surface with no key. They are hand-mounted in
`app.py` with the same `require_api_key` dependency the other routers use, so "every
endpoint except `/healthz`" is exact, not aspirational.

### Status codes

| Code | When |
|---|---|
| `401` | Missing or invalid `X-API-Key`. Body: `{"detail": "Missing or invalid API key"}`. |
| `404` | No trained bundle for that symbol, or no job with that id. Body: `{"detail": "No trained model for ... Run \`pyquant train\` first."}` or `{"detail": "No job '<id>'"}`. |
| `409` | {py:class}`~pyquant.models.tft.FeatureSchemaMismatch` — the bundle was trained on features that can no longer be assembled. A conflict, not a server error: the model is fine, the world moved. Also returned by `POST /train` when a job for the same bundle name is already queued or running (bugs.md#pyq-161). |
| `422` | An empty `symbols` list on `POST /train` (`{"detail": "symbols must not be empty"}`), or a body that fails pydantic validation — e.g. an invalid symbol shape returns FastAPI's structured form: `{"detail": [{"type": "value_error", "loc": ["body", "symbols"], "msg": "Value error, Invalid symbol/bundle name '../x': must match ...", ...}]}`. Note the two `422` shapes differ (plain string vs. a list of error objects) depending on which check rejected the request. |
| `500` | `PYQUANT_API_KEYS` is unset and unauthenticated access was not explicitly enabled. |

### Forecasting

```bash
curl -H "X-API-Key: $KEY" http://localhost:8000/forecast/AAPL
```

The response body is {py:class}`~pyquant.api.schemas.ForecastResponse`, constructed
directly from `serialize.forecast_to_dict` — the same function behind
`pyquant --format json forecast`. `forecast_dates` and `n_quantile_crossings` are present
for the same reasons they are in the [CLI's JSON output](cli.md#json-output).

```bash
curl -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"symbols": ["AAPL", "MSFT", "NVDA"]}' \
     http://localhost:8000/scan
```

`POST /scan` returns `200` even when individual symbols fail. Each row carries a `status`
of `ok`, `not_trained` or `error`, so one flaky symbol does not sink the comparison — the
same discipline as the CLI's `scan`, and the same `scan_row_to_dict` implementation, so the
`signal` and `band_width_pct` a caller sees cannot drift between the two front-ends. An
empty `symbols` list is accepted and returns `[]` (unlike `POST /train`, which rejects an
empty list with `422` — `/scan` treats "compare nothing" as a valid no-op).

A real two-symbol response, one trained and one not, looks like this:

```json
[
  {
    "symbol": "AAPL", "status": "ok", "error": null,
    "current_price": 336.91, "median_target": 267.31,
    "expected_return_pct": -20.66, "band_width_pct": 7.97, "signal": "SELL"
  },
  {
    "symbol": "ZZZZNOPE", "status": "not_trained",
    "error": "No trained model for ZZZZNOPE at /.../checkpoints/ZZZZNOPE/model.ckpt. Run `pyquant train` first.",
    "current_price": null, "median_target": null,
    "expected_return_pct": null, "band_width_pct": null, "signal": null
  }
]
```

### Explaining a forecast

```bash
curl -H "X-API-Key: $KEY" http://localhost:8000/explain/AAPL
```

The response is {py:class}`~pyquant.api.schemas.InterpretationResponse` — the same
`feature_importance`/`attention`/`bundle_skill` fields as `pyquant explain --format json`.
`bundle_skill` is the bundle's own recorded skill from training (can be negative — the
project does not dress up a bad number); use it as a "should I trust this explanation"
signal without re-deriving it.

### Training

Training takes minutes, so `POST /train` returns immediately:

```bash
curl -X POST -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"symbols": ["AAPL"], "epochs": 5}' \
     http://localhost:8000/train
# 202 {"job_id": "…", "status": "queued"}

curl -H "X-API-Key: $KEY" http://localhost:8000/train/<job_id>
```

Body fields: `symbols` (required, non-empty; more than one pools them), `bundle_name`,
`epochs`, `period` (overrides `settings.data.period`, e.g. `"10y"`). Job status moves
through `queued` → `running` → `succeeded` or `failed`; `result` is populated on success
and `error` on failure.

When a job succeeds, the bundle it retrained is **evicted from the bundle cache**, so the
next `/forecast` or `/explain` reloads it from disk. Without that step a cached copy from
before the run would keep serving the old weights.

A shell polling loop:

```bash
job_id=$(curl -s -X POST -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"symbols": ["AAPL"], "epochs": 5}' \
     http://localhost:8000/train | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')

until [ "$(curl -s -H "X-API-Key: $KEY" http://localhost:8000/train/$job_id \
     | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')" != "running" ]; do
  sleep 5
done
curl -s -H "X-API-Key: $KEY" http://localhost:8000/train/$job_id
```

A second `POST /train` for a bundle that already has a job queued or running is rejected
with `409` rather than racing the first job onto the same on-disk checkpoint directory
(bugs.md#pyq-161) — poll the existing `job_id` instead of retrying a slow-looking job:

```bash
curl -X POST -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"symbols": ["AAPL"]}' http://localhost:8000/train
# 409 {"detail": "A training job for bundle 'AAPL' is already queued or running"}
```

The bundle name in the conflict message is `bundle_name` if given, else the request's
`symbols` joined with `_` and upper-cased — the same default `tft.train()` itself applies,
so pooled requests (`{"symbols": ["AAPL", "MSFT"]}`) are keyed as `AAPL_MSFT`, not per symbol.

### Backtesting

`pyquant backtest` is the command that produces every quality number this project
reports — skill vs. persistence, calibration coverage, directional accuracy — so it is the
one capability an API consumer most plausibly wants beyond a forecast. Like training, a
walk-forward backtest trains `windows` models sequentially (investigations.md#pyq-319
measured a *single* cold forecast at ~65s; a 5-window backtest trains five), so it is a
background job on the same `JobRegistry` `/train` uses, not a request-cycle operation:

```bash
curl -X POST -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"symbol": "AAPL", "windows": 5}' \
     http://localhost:8000/backtest
# 202 {"job_id": "…", "status": "queued"}

curl -H "X-API-Key: $KEY" http://localhost:8000/backtest/<job_id>
```

Body fields: `symbol` (required, singular — a backtest is never pooled), `windows`
(default `5`), `epochs`, `period`. The result, once `status` is `"succeeded"`, is
{py:class}`~pyquant.api.schemas.BacktestResponse` — exactly `analysis.serialize
.backtest_to_dict()`'s shape, the same function `pyquant --format json backtest` prints, so
no reshaping was needed to serve both front-ends. `aggregated` is the pooled
{py:class}`~pyquant.api.schemas.EvaluationResponse` across all windows; `per_window` is the
same shape per window, in the same order as `origins` (each window's cutoff time index) —
present so a caller can see dispersion across origins rather than only the mean
(bugs.md#pyq-141 is the reminder that the two are different estimators and can diverge).

`--signals`, `--seeds` and `--cost-bps` are CLI-only for this endpoint — the same judgement
features.md#pyq-271 already makes for `tune`/`doctor`/`cache`/`snapshot`: not obviously
wanted remotely, and each would need its own reasoning about job-result shape rather than
being folded in silently. A backtest never persists a bundle (each window's model is
discarded after evaluation), so unlike `/train` there is no per-bundle-name conflict lock —
two concurrent backtests for the same symbol just duplicate work rather than racing
on-disk state.

`job_id`s from `/train` and `/backtest` are not interchangeable even though they share one
registry: polling a train job's id via `GET /backtest/{job_id}` (or vice versa) returns
`404`, the same as an unknown id.

### Minimal Python client

```python
import requests

BASE, KEY = "http://localhost:8000", "..."
headers = {"X-API-Key": KEY}

fc = requests.get(f"{BASE}/forecast/AAPL", headers=headers).json()
print(fc["median"], fc["expected_return_pct"])

job = requests.post(
    f"{BASE}/train", headers=headers, json={"symbols": ["AAPL"], "epochs": 5}
).json()
status = requests.get(f"{BASE}/train/{job['job_id']}", headers=headers).json()

bt_job = requests.post(
    f"{BASE}/backtest", headers=headers, json={"symbol": "AAPL", "windows": 5}
).json()
bt_status = requests.get(f"{BASE}/backtest/{bt_job['job_id']}", headers=headers).json()
```

## Concurrency model

Two mechanisms, both from measured behaviour rather than assumption:

**An LRU cache of loaded bundles**, holding 8. Loading a bundle is real file I/O plus
checkpoint deserialisation; reloading per request would re-pay it every time. The load
happens *outside* the cache lock, so loading one bundle does not block requests for
bundles already resident.

**One lock per bundle.** `TemporalFusionTransformer.predict()` is not safe to call
concurrently on the same model instance — pytorch-forecasting spins up an internal
Lightning `Trainer` per call and mutates model state. Requests against *different* bundles
proceed in parallel; requests against the same one serialise. That costs less than it
sounds: torch already uses multiple cores within a single call.

Both are informed by `investigations.md#pyq-319`, which measured where forecast latency
actually goes: a cold call is **~98% vendor fetch and panel build (~65 s)**, and the
forward pass itself is under a second either way. Optimising the model path would have been
optimising 2% of the wall clock.

## What v1 deliberately does not do

The design note calls this an in-process scaffold, and the limits are documented triggers
to graduate rather than defects to file:

- **The job registry is in-process**, and bounded to the 500 most recent jobs
  (oldest-first eviction, same shape as the bundle LRU cache above). Jobs are lost on
  restart and invisible to a second worker regardless of the bound. Running more than one
  uvicorn worker is the point at which this needs a real queue.
- **The bundle cache is per-process**, so a `POST /train` in one worker does not invalidate
  another worker's cached copy. Same trigger.
- **Bundles live on the local filesystem.** No object storage, so instances do not share
  trained models.
- **No rate limiting.** API keys authenticate; they do not meter. An authenticated caller
  can still exhaust the operator's vendor quota.

Each of these is cheap to live with for a single instance and wrong to pretend away for
more than one.

## Known gaps (tracked, not fixed here)

- **`tune`, `doctor`, `cache`, `snapshot` and `calibration` have no API equivalent.** The
  first is the same shape as `/backtest` and the obvious next candidate if wanted
  remotely; the rest are local-operator commands with no obvious remote meaning
  (features.md#pyq-271's own scoping judgement).
- **`/backtest` does not expose `--signals`, `--seeds` or `--cost-bps`.** CLI-only for this
  pass — each would need its own reasoning about job-result shape rather than being folded
  in silently (see the Backtesting section above).
- **No rate limiting and no per-key request quota** — see "What v1 deliberately does not
  do" above.
