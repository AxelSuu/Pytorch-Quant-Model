# Architecture

PyQuant is five layers with one hard rule between them: **the machine-learning stack does
not leak upwards**. Everything above `models/` works on pandas frames, numpy arrays and
plain dataclasses, and is served by two independent front-ends — `cli/` and `api/` — over
the same calls, without either being a rewrite of the other.

```
                 vendors                          pyquant/
   ┌──────────────────────────────┐
   │ yfinance   OHLCV + indicators│──►  data/prices.py       ─┐
   │ FRED       macro (ALFRED)    │──►  data/macro.py         │
   │ Finnhub    headlines         │──►  data/sentiment.py     ├─► data/dataset.py
   │ yfinance   sector ETFs       │──►  data/sectors.py       │      build_panel()
   │ yfinance   options chain     │──►  data/options.py  ◐   ─┘      panel_to_long()
   └──────────────────────────────┘     (display + accum.)          make_dataset()
                                                                          │
                                                       ┌──────────────────┘
                                                       ▼
                                        models/tft.py  ── the only module that
                                        train() / walk_forward_backtest()  imports
                                        predict_quantiles() / interpret() / tune()  Lightning
                                                       │
                        ┌──────────────────────────────┼───────────────────────┐
                        ▼                              ▼                       ▼
              analysis/forecast.py          analysis/interpret.py     analysis/metrics.py
              analysis/calibrate.py         analysis/serialize.py     analysis/signals.py
                        │                              │                       │
                        ├──────────────► cli/app.py (Typer + Rich)             │
                        └──────────────► api/app.py (FastAPI, optional) ◄──────┘
```

(what-each-layer-owns)=
## What each layer owns

`config.py`
: A pydantic-settings tree — {py:class}`~pyquant.config.Settings` wrapping
  {py:class}`~pyquant.config.TFTConfig`, {py:class}`~pyquant.config.TrainingConfig` and
  {py:class}`~pyquant.config.DataConfig`. Precedence, highest first: CLI flags >
  environment > `.env` > YAML experiment file > built-in defaults. The YAML source sits
  *below* the environment deliberately, so a checked-in experiment stays overridable.
  Anything a user might tune belongs here rather than as a literal in `models/tft.py`;
  hardcoded literals in that file have generated four separate tickets. Full field list:
  [configuration reference](api/configuration.md).

`data/`
: One module per vendor, each returning a date-indexed frame and each responsible for its
  own publication-time semantics. {py:mod}`pyquant.data.dataset` is where they meet:
  `build_panel()` joins them into one wide frame, `panel_to_long()` adds `time_idx` /
  `symbol` / calendar columns, and `make_dataset()` produces the
  {py:class}`~pytorch_forecasting.data.timeseries.TimeSeriesDataSet`. `cache.py` adds a
  TTL panel cache plus TTL-exempt named *pins* for reproducible experiments, `retry.py` is
  a dependency-free exponential backoff used by the flaky fetches, and `providers.py` is
  the `PriceProvider` protocol (yfinance, Tiingo) behind `fetch_prices`.

`models/tft.py`
: Every pytorch-forecasting and Lightning call in the project. Training, walk-forward
  backtesting, Optuna hyperparameter search (`tune()`), checkpoint/bundle persistence,
  prediction and interpretation extraction. A trained bundle is a directory under
  `checkpoints/<name>/` holding `model.ckpt`, `dataset_params.pt` and `meta.json` (symbol,
  feature names, metrics, resolved config, seed, code version, pin).

`analysis/`
: Library-agnostic post-processing on numpy arrays and dataclasses — quantile forecasts
  ({py:class}`~pyquant.analysis.forecast.Forecast`), interpretation, evaluation metrics,
  conformal recalibration, BUY/SELL/HOLD signal P&L accounting
  ({py:mod}`pyquant.analysis.signals`), and the JSON serializers both front-ends share.

`cli/app.py`
: Typer commands and Rich rendering. A thin caller: each command resolves settings, calls
  one `analysis/` or `models/` function, and renders the dataclass it gets back.

`api/`
: FastAPI, optional (`uv sync --extra api`) — the second front-end the design note below
  planned, now built. `app.py` mounts one router per resource (`routes/{health,forecast,
  explain,train}.py`); response models in `schemas.py` are thin pydantic wrappers
  constructed directly from `analysis/serialize.py`'s functions, so a response and the
  CLI's `--format json` output cannot drift — they are produced by the same code, not two
  independently-typed schemas that happen to agree today. `deps.py` holds the per-bundle
  prediction lock and LRU bundle cache; `jobs.py` is the in-process training-job registry
  behind `POST /train`. Full shape and the reasoning behind each choice:
  [FastAPI design note](api-design.md).

## Two structural rules

**1. pytorch-forecasting and Lightning are confined to `models/tft.py` and
`data/dataset.py`.** Nothing in `analysis/`, `cli/` or `api/` may import them.

**2. `analysis/` and `models/` never import Typer, Rich, or FastAPI.**

Together these are what made `api/` additive rather than a rewrite when it was built:
`cli/app.py` was already a mapping from command to a plain function returning a plain
dataclass, so the second front-end wraps exactly the same calls.
`analysis/calibrate.py` is a good illustration of rule 1 being load-bearing rather than
aesthetic — split-conformal calibration operates on raw quantile arrays, so it lives in
`analysis/` and stays testable without the ML stack installed.

The same split governs the test suite: `test_tft`, `test_dataset`, `test_cli`,
`test_forecast` and `test_interpret` need torch; `test_api` needs the `api` extra (skipped
cleanly via `pytest.importorskip` when absent, matching `test_tune`'s disposition for the
`tuning` extra); everything else runs without either.

(graceful-degradation)=
## Graceful degradation is a training-time contract

Every enrichment flag on {py:class}`~pyquant.config.DataConfig` is a *request*, not an
assertion. A source that is disabled, missing its key, rate-limited or simply returning
nothing is skipped, logged, and left out of the feature schema — which is derived from the
columns that actually materialised rather than from a fixed list. That is why
`pyquant train AAPL` works with no API keys at all.

**The contract deliberately stops at predict time.** A model cannot run without the
columns it was trained on, so a bundle whose panel is missing a trained feature raises
{py:class}`~pyquant.models.tft.FeatureSchemaMismatch`, naming every absent column *and*
the source it came from — a rotated-out `FRED_API_KEY`, a failing sector fetch, a toggle
flipped between runs. Extra columns at predict time remain a deliberate no-op.

This asymmetry is the subject of PYQ-118, and it is a decision rather than an oversight:
silent degradation during training costs you some features; silent degradation at predict
time would produce a confident forecast from a model running on the wrong inputs.

Two related mechanisms keep a bundle self-describing:

- `meta.json` records the *resolved* `DataConfig` a bundle was trained with, and
  `settings_for_bundle()` replays it, so `forecast` cannot quietly rebuild a different
  panel than `train` used (PYQ-119).
- Reproducibility rests on three legs — the recorded seed, a pinned dataset snapshot, and
  the code version (package version + git sha). Feature definitions do change; `RSI_14`
  was redefined once, and the cache fingerprint includes the code version so a pin can
  never be replayed across an incompatible implementation.

## Options data: display today, accumulating toward a real feature

`data/options.py`'s `fetch_options_snapshot()` returns a put/call ratio, ATM implied
volatility and skew, and the `forecast` command prints them under the table. Yahoo Finance
exposes only the *current* chain, not history, so there is no way to fetch a historical
options-implied series — joining a live snapshot onto every historical training row would
attach today's option prices to the past, a leak by construction.

`pyquant snapshot SYMBOL` (PYQ-254) is the honest way out of that limitation: it appends
one row a day to an append-only per-symbol history
(`Settings.options_history_dir/<SYMBOL>.jsonl`), and `build_panel()` joins the accumulated
history back in — once at least `options.MIN_SNAPSHOT_DAYS` (20) distinct days exist,
following the same publication-timing discipline as the macro/sentiment joins (a snapshot
recorded on day *T* can only ever land on day *T*'s row or later). Below that threshold, or
on a fresh checkout with no accumulated history at all, the columns are simply absent —
useless on day one, and the only path to ever having a genuine historical options feature.

## Where to go next

- [Leakage invariants](invariants.md) — the properties this shape exists to protect.
- [Methodology](methodology.md) — how the resulting model is scored.
- [CLI reference](cli.md) and [HTTP API](http-api.md) — the two front-ends this shape
  exists to make interchangeable.
- [API reference](api/index.md) — module-by-module autodoc.
- [Development](development.md) — testing conventions, and the scripts behind the measured
  numbers.
