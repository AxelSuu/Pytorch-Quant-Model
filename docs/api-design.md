# PyQuant FastAPI service layer — design note (PYQ-213)

Status: **implemented** (PYQ-261, 2026-07-27) — `pyquant/api/`, following the shape
this note decided. The note is kept as-is below since the implementation follows it
directly; see `backlog/features.md#pyq-261` for what shipped versus what was
deliberately deferred (a real job queue, object storage, rate-limiting).

**To run the service rather than read its design, go to [HTTP API](http-api.md).** This
page is the reasoning; that one is the interface.

## Why this is additive, not a rewrite

The analysis/model/data layers are already fully decoupled from Typer/Rich. The
CLI in `pyquant/cli/app.py` is a thin caller over plain functions and
dataclasses:

| CLI command | Underlying call | Returns |
|---|---|---|
| `forecast` | `analysis.forecast.generate_forecast(symbol, settings, ...)` | `Forecast` |
| `explain` | `analysis.interpret.explain_forecast(symbol, settings, ...)` | `Interpretation` |
| `train` | `models.tft.train(symbols, settings, ...)` | `TrainResult` |
| `backtest` | `models.tft.walk_forward_backtest(symbol, settings, ...)` | `BacktestResult` |
| `scan` | N× `generate_forecast` | list of `Forecast` |
| (load) | `models.tft.load(symbol, settings)` | `ModelBundle` |

A FastAPI app wraps exactly these functions. Nothing in the core needs to move;
the API is a second front-end beside the CLI.

## Proposed package layout

```
pyquant/api/
    __init__.py
    app.py          # FastAPI() instance, router mounting, exception handlers
    schemas.py      # pydantic response models (see below)
    deps.py         # Settings, bundle cache, API-key auth dependencies
    jobs.py         # in-process training-job registry (v1) + status model
    routes/
        forecast.py # GET  /forecast/{symbol}, POST /scan
        explain.py  # GET  /explain/{symbol}
        train.py    # POST /train  -> job id;  GET /train/{job_id}
        health.py   # GET  /healthz
```

Add an optional dependency group so the base install stays lean:

```toml
[project.optional-dependencies]
api = ["fastapi>=0.110", "uvicorn[standard]>=0.29"]
```

## 1. Response models (dovetails with PYQ-212)

`Forecast`, `Interpretation`, `TrainResult`, `EvaluationMetrics`,
`BacktestResult` are already plain dataclasses. Two viable paths:

- **Pydantic mirrors in `api/schemas.py`** — an explicit `ForecastResponse`
  etc. with a `from_domain(obj)` classmethod. Keeps numpy→list conversion and
  field-shaping (e.g. `predictions` is an `np.ndarray` that must become nested
  lists; `last_date` a `pd.Timestamp` → ISO string) at the boundary, and keeps
  the API contract decoupled from internal dataclass changes.
- **Shared serializers with the CLI's `--format json` (PYQ-212)** — implement
  the domain→JSON-able-dict functions once (e.g. `forecast_to_dict(fc)`), have
  the CLI print them and the API return them. Recommended: build PYQ-212 first,
  then the API reuses those serializers and wraps them in thin pydantic models
  for OpenAPI schema generation.

Numpy is the sharp edge: every `np.ndarray`/`np.float64` must be converted
(`.tolist()`, `float(...)`) or JSON encoding fails. Centralize this in the
serializers so it is done in exactly one place.

Sketch:

```python
class ForecastResponse(BaseModel):
    symbol: str
    last_date: str
    current_price: float
    quantiles: list[float]
    predictions: list[list[float]]   # (horizon, n_quantiles)
    expected_return_pct: float

    @classmethod
    def from_domain(cls, fc: Forecast) -> "ForecastResponse":
        return cls(
            symbol=fc.symbol,
            last_date=fc.last_date.date().isoformat(),
            current_price=fc.current_price,
            quantiles=list(fc.quantiles),
            predictions=fc.predictions.tolist(),
            expected_return_pct=fc.expected_return_pct(),
        )
```

## 2. Training is long-running → background jobs

`tft.train()` blocks for a full Lightning fit (seconds on synthetic data,
minutes-to-longer on real 5y panels). A request thread must not block on it.

**v1 (single node):** an in-process job registry, run on a dedicated
`concurrent.futures.ThreadPoolExecutor` (`jobs.get_job_executor`, 4 workers) —
**not** FastAPI `BackgroundTasks`. `BackgroundTasks` dispatches a sync
function through `starlette.concurrency.run_in_threadpool`, which draws from
anyio's single process-wide default thread limiter (40 slots) — the same pool
every other sync route handler needs. A training job holding a slot for
minutes would queue a concurrent `GET /forecast` behind it (bugs.md#pyq-163).
`/train` and `/backtest` are `async def` routes that submit the job to the
dedicated executor via `loop.run_in_executor(executor, ...)` instead, so
scheduling the job never itself touches the shared pool.

```
POST /train {"symbols": ["AAPL"], ...}  -> 202 {"job_id": "...", "status": "queued"}
GET  /train/{job_id}                    -> {"status": "running|succeeded|failed",
                                            "result": TrainResultResponse | null,
                                            "error": str | null}
```

`jobs.py` holds a `dict[str, JobRecord]` guarded by a lock; the executor task
updates the record on completion. Good enough for one instance and a low
training concurrency.

**In-flight request guards differ between the two routes,** because the two
jobs fail differently under concurrency. `/train` rejects (409) a second
request for a bundle already queued/running (`try_start_train`,
bugs.md#pyq-161) — two concurrent fits racing the same on-disk checkpoint
directory would corrupt it. `/backtest` never writes a persistent bundle, so
concurrent backtests for the same symbol can't corrupt anything; instead
`try_start_backtest` folds an identical in-flight request (same
symbol/windows/epochs/period) into the existing job rather than rejecting or
duplicating it — bounding how much redundant work a retry/double-click can
pile onto the shared 4-worker executor (PYQ-327). Two backtests with
*different* parameters for the same symbol still both run.

**Where v1 stops scaling — the trigger to graduate to a real queue
(arq/RQ/Celery + Redis):**
- Job state is in process memory → **lost on restart/redeploy**, and invisible
  to any second instance.
- The dedicated executor bounds *this process's* concurrency (4 workers) but
  not CPU/GPU oversubscription across a multi-instance deployment, and every
  worker still runs in-process — torch's own CPU usage is not isolated from
  the request-serving process the way a separate worker/process pool would
  isolate it.
- No retries, no backpressure beyond the executor's own queue, no scheduling,
  no cancellation.

Move to arq or Celery + Redis the moment you need durable jobs, more than one
instance, or bounded training concurrency.

## 3. Bundle storage

`tft.load()` / `train()` read and write `checkpoints/<BUNDLE>/` on local disk
(`model.ckpt`, `dataset_params.pt`, `meta.json`, `runs.jsonl`).

- **One instance:** local disk (or a mounted volume) is fine.
- **Multi-instance / serverless:** the box that served `POST /train` is not
  necessarily the one that serves `GET /forecast` later → the second box can't
  see the bundle. Needs shared/object storage (S3-compatible): write the bundle
  to a bucket on train, pull-and-cache on load.

**Decide this before building anything stateful.** Two concrete prerequisites
regardless of backend:
- **PYQ-220** (`checkpoint_dir`/`cache_dir` resolve against CWD) must be fixed
  first — a server's working directory is not the repo root, so relative
  `checkpoints/` silently lands in the wrong place (or a different place per
  restart). Pin to an absolute, configured location.
- A storage abstraction (`load_bundle(name) -> local_path`, `save_bundle(...)`)
  so local-disk vs S3 is one implementation swap, not scattered `Path` joins.

## 4. Inference concurrency

**Measured (PYQ-319, `scripts/profile_forecast.py`, AAPL, default `period=5y`):** the
question this section used to answer by guesswork now has real numbers.

| Phase | Cold (panel cache empty) | Warm (panel cache hit) |
|---|---|---|
| Bundle load (checkpoint deserialize) | 261 ms | 203 ms |
| Fetch + panel build (4 vendors) | **64,300 ms** | 5 ms |
| `predict()` (the actual forward pass) | 812 ms | 632 ms |
| `interpret()`'s extra raw-mode predict (`explain` only) | +835 ms | +740 ms |
| **Total, `forecast`** | **~65.4 s** | **~0.84 s** |

Request counts behind the cold call: 8 `yfinance.Ticker.history`/related calls (prices +
VIX + options), 15 `fredapi.get_series_all_releases` calls (5 yearly-chunked windows × 3
FRED series, at the default 5-year period), 1 `yfinance.download` (sector ETFs), 1
`requests.get` (Finnhub). The warm call makes zero outbound requests.

This answers the concurrency design's central open question directly: **the forward pass
is not the bottleneck.** Cold, fetch/panel-build is 98% of total latency; the model
inference this section spends most of its words worrying about (locking, LRU eviction,
thread-safety) is under a second regardless of cache state, and `explain`'s second
raw-mode predict adds well under a second more. The real scaling constraint is vendor
latency and rate limits on a cache miss, and cache hit rate is therefore the single number
that determines whether a deployed `/forecast` endpoint feels instant (sub-second) or feels
like a timeout risk (60+ seconds) — which reframes the panel TTL cache (`pyquant.data.
cache`) from a nice-to-have into the thing that actually decides the service's perceived
latency. It also means a naive per-request panel rebuild is the wrong default for any
server deployment: the LRU **bundle** cache below saves ~250-450ms; a **panel** cache (or a
background pre-warm) saves ~64 seconds. Quota is the other real constraint this implies:
FRED's 15 requests/call at the default period means an operator serving many distinct
symbols cold will hit FRED's rate limit long before CPU or GPU becomes the bottleneck.

Caveats on these numbers: one symbol, one run, on whatever the live vendors' latency
happened to be at measurement time (2026-07-27) — not a controlled benchmark, and vendor
latency varies. The qualitative conclusion (fetch dominates, predict is cheap) is robust to
that noise; the exact millisecond figures are not.

- **Thread-safety of `predict()`:** do **not** assume
  `TemporalFusionTransformer.predict()` is safe to call concurrently on the
  same model instance — pytorch-forecasting spins up an internal Lightning
  `Trainer` per call and mutates model state. Safest v1: a **per-bundle lock**
  (serialize predictions against a given loaded model); parallelism across
  *different* bundles is fine. Revisit with a proper load test before removing
  the lock. Torch intra-op threading already uses multiple cores per call, so
  serializing per-model is not as costly as it sounds.
- **Keep bundles loaded vs. reload per request:** reloading per request pays
  the checkpoint deserialization cost every time and re-triggers the
  `weights_only=False` unpickle (PYQ-306). Prefer an in-process **LRU cache**
  of `ModelBundle` keyed by bundle name (bounded size to cap memory), evicting
  least-recently-used. Invalidate/replace an entry when that bundle is
  retrained.
- **FinBERT cache (PYQ-114) matters more here:** the now-fixed sentiment
  pipeline cache only memoizes on success, so a transient first-request failure
  no longer permanently disables sentiment for the life of the server process.
  This design depends on that fix.
- **Schema drift (PYQ-302):** a long-running server is exactly where "source
  enabled at train time, rate-limited at predict time" happens. Resolve PYQ-302
  before relying on the forecast endpoint against live data — otherwise a
  transient upstream outage yields malformed predictions rather than a clean
  error.

## 5. Auth / rate-limiting

Nothing in the stack has any today. A public endpoint would spend the
operator's FRED/Finnhub/Yahoo quota on every caller.

- **Minimum:** an API-key gate as a FastAPI dependency
  (`Depends(require_api_key)`), keys from config/secret store, constant-time
  compare. Reject unauthenticated with 401.
- **Rate-limiting:** per-key limits (e.g. `slowapi`, or the reverse
  proxy/API-gateway layer) — training especially must be tightly capped since
  it is expensive and quota-hungry.
- Never echo upstream API keys in responses or logs.

## 6. Deployment

Pairs with **PYQ-217** (Dockerfile): a CPU-only image is enough for the
inference/serving path (`forecast`/`explain`/`scan`); training wants the CUDA
variant. Build via `uv sync --frozen --extra api` (plus `--extra sentiment` if
sentiment is wanted server-side). `uvicorn pyquant.api.app:app` as the entry
point.

## Endpoint summary (v1)

| Method | Path | Wraps | Notes |
|---|---|---|---|
| GET | `/healthz` | — | liveness |
| GET | `/forecast/{symbol}` | `generate_forecast` | per-bundle lock + LRU cache |
| POST | `/scan` | N× `generate_forecast` | body: list of symbols |
| GET | `/explain/{symbol}` | `explain_forecast` | as forecast |
| POST | `/train` | `tft.train` (background) | 202 + job id |
| GET | `/train/{job_id}` | job registry | status/result |

## Recommended build order

1. **PYQ-212** (JSON serializers) — the API reuses them; do not duplicate
   numpy→JSON logic.
2. **PYQ-220** (absolute bundle/cache paths) — prerequisite for any server.
3. Read-only endpoints (`/forecast`, `/explain`, `/scan`) + API-key auth +
   bundle LRU cache with per-bundle prediction lock.
4. `/train` background-job endpoints (in-process registry).
5. **PYQ-217** Dockerfile; deploy single-instance.
6. Only if/when multi-instance or durability is needed: object-storage bundle
   backend + Redis-backed job queue (arq/Celery). Note PYQ-302 must be resolved
   before trusting live-data forecasts under sustained load.

## Open follow-ups this note spawns (not filed yet)

- A storage-abstraction ticket (local ↔ S3) — blocked on PYQ-220.
- Load-test `predict()` concurrency to decide whether the per-bundle lock can
  be relaxed.
- Decide server-side sentiment: ship `--extra sentiment` in the image (heavier)
  or run inference-only bundles without it.
