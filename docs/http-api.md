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

Symbols are upper-cased on the way in, so `/forecast/aapl` and `/forecast/AAPL` are the
same bundle.

### Status codes

| Code | When |
|---|---|
| `401` | Missing or invalid `X-API-Key`. |
| `404` | No trained bundle for that symbol, or no job with that id. |
| `409` | {py:class}`~pyquant.models.tft.FeatureSchemaMismatch` — the bundle was trained on features that can no longer be assembled. A conflict, not a server error: the model is fine, the world moved. |
| `422` | An empty `symbols` list on `POST /train`, or a body that fails validation. |
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
`signal` and `band_width_pct` a caller sees cannot drift between the two front-ends.

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
`epochs`, `period`. Job status moves through `queued` → `running` → `succeeded` or
`failed`; `result` is populated on success and `error` on failure.

When a job succeeds, the bundle it retrained is **evicted from the bundle cache**, so the
next `/forecast` or `/explain` reloads it from disk. Without that step a cached copy from
before the run would keep serving the old weights.

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

- **The job registry is in-process.** Jobs are lost on restart and invisible to a second
  worker. Running more than one uvicorn worker is the point at which this needs a real
  queue.
- **The bundle cache is per-process**, so a `POST /train` in one worker does not invalidate
  another worker's cached copy. Same trigger.
- **Bundles live on the local filesystem.** No object storage, so instances do not share
  trained models.
- **No rate limiting.** API keys authenticate; they do not meter. An authenticated caller
  can still exhaust the operator's vendor quota.

Each of these is cheap to live with for a single instance and wrong to pretend away for
more than one.
