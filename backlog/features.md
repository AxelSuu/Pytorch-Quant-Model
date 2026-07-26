# Features (PYQ-2xx)

Things to build — see [`README.md`](README.md) for the format.
Next free ID: **PYQ-232**.

| ID | Priority | Status | Title |
|----|----------|--------|-------|
| [PYQ-201](#pyq-201) | High | Resolved | Naive-baseline + directional-accuracy + calibration reporting |
| [PYQ-202](#pyq-202) | High | Resolved | Proper rolling / walk-forward backtest command |
| [PYQ-203](#pyq-203) | Medium | Resolved | Experiment/version tracking for trained bundles |
| [PYQ-204](#pyq-204) | Medium | Resolved | Pooled / cross-sectional multi-ticker training |
| [PYQ-205](#pyq-205) | Medium | Resolved | Local data caching / dataset pinning |
| [PYQ-206](#pyq-206) | Medium | Resolved | `scan`'s BUY/SELL respects the forecast's uncertainty band |
| [PYQ-207](#pyq-207) | Low | Resolved | `--verbose`/`--debug` CLI flag |
| [PYQ-208](#pyq-208) | Medium | Resolved | `test_options.py` — was 0% covered |
| [PYQ-209](#pyq-209) | High | Resolved | Config-file (YAML) support for experiments |
| [PYQ-210](#pyq-210) | Medium | Resolved | `seed_everything` + recorded seed for reproducibility |
| [PYQ-211](#pyq-211) | Medium | Open | Learning-rate tuning instead of one fixed lr for every run |
| [PYQ-212](#pyq-212) | Medium | Resolved | Machine-readable output mode (`--format json` / `--quiet`) |
| [PYQ-213](#pyq-213) | High | Resolved | Design (and scaffold) a FastAPI service layer alongside the CLI |
| [PYQ-214](#pyq-214) | Medium | Open | Broaden and harden external data providers |
| [PYQ-215](#pyq-215) | Medium | Resolved | Retry/backoff for flaky external calls |
| [PYQ-216](#pyq-216) | Low | Resolved | Detect/guard against quantile crossing |
| [PYQ-217](#pyq-217) | Medium | Open | Dockerfile for reproducible training/serving environments |
| [PYQ-218](#pyq-218) | Medium | Resolved | Make DataLoader `num_workers` configurable (hardcoded to 0 everywhere) |
| [PYQ-219](#pyq-219) | Low | Resolved | Validate `TFTConfig.quantiles` is sorted ascending |
| [PYQ-220](#pyq-220) | Medium | Open | `checkpoint_dir`/`cache_dir` resolve relative to CWD |
| [PYQ-221](#pyq-221) | Low | Resolved | `pyquant cache` subcommand — the local panel cache has no eviction/pruning |
| [PYQ-222](#pyq-222) | Low | Resolved | `train`'s `console.status()` spinner competes with Lightning's own progress bar |
| [PYQ-223](#pyq-223) | Low | Resolved | Mixed-precision training option (Trainer has no `precision=` set, fp32 only) |
| [PYQ-224](#pyq-224) | Medium | Resolved | Make `EarlyStopping` patience configurable (hardcoded to 5 in both trainers) |
| [PYQ-225](#pyq-225) | Medium | Resolved | Record full provenance in `meta.json` (version, git sha, pin, resolved config) |
| [PYQ-226](#pyq-226) | Medium | Resolved | Report metric dispersion across backtest windows, not just the mean |
| [PYQ-227](#pyq-227) | Medium | Open | Per-quantile calibration + pinball loss alongside band coverage |
| [PYQ-228](#pyq-228) | Low | Open | Bound dependency majors; pass `auto_adjust` to `yfinance` explicitly |
| [PYQ-229](#pyq-229) | Low | Open | CI: Python matrix, frozen install, `uv lock --check`, `ruff format --check` |
| [PYQ-230](#pyq-230) | Low | Open | CI: measure and report test coverage |
| [PYQ-231](#pyq-231) | Medium | Resolved | CLI failure-path test coverage — every existing test asserts `exit_code == 0` |

---

## [PYQ-201]
Naive-baseline + directional-accuracy + calibration reporting
Status: Resolved — commit b616184, 2026-07-23
Priority: High

Problem: model quality was reported only as an absolute QuantileLoss number
on one held-out window, with nothing to compare it against.

Resolution: `pyquant/analysis/metrics.py` now reports (a) accuracy vs. a
persistence baseline (predict last observed close), (b) directional
hit-rate, and (c) empirical calibration coverage, surfaced in `train`'s
summary table and `backtest` (PYQ-202).

---

## [PYQ-202]
Proper rolling / walk-forward backtest command
Status: Resolved — commit b616184, 2026-07-23
Priority: High

Problem: `train()`'s validation was exactly one 5-day held-out window per
symbol — a small, high-variance sample to be driving `EarlyStopping` and
`ModelCheckpoint`'s model-selection decisions (see investigations.md#pyq-303).

Resolution: `pyquant backtest` (`tft.walk_forward_backtest`) trains/evaluates
across many rolling origins and aggregates PYQ-201's metrics across them.

---

## [PYQ-203]
Experiment/version tracking for trained bundles
Status: Resolved — commit b616184, 2026-07-23
Priority: Medium

Problem: `checkpoints/<SYMBOL>/` was a single mutable slot — retraining the
same symbol with a different config silently overwrote the previous bundle
and `meta.json` with no history.

Resolution: an append-only `runs.jsonl` per bundle now accumulates every
training run's metadata alongside the mutable `meta.json` (which still
reflects only the latest/deployable bundle).

---

## [PYQ-204]
Pooled / cross-sectional multi-ticker training
Status: Resolved — commit b616184, 2026-07-23
Priority: Medium

Problem: `TimeSeriesDataSet` was already built with `group_ids=["symbol"]`
(pytorch-forecasting's mechanism for exactly this), but `train()` and the
CLI only ever accepted one symbol at a time.

Resolution: `pyquant train AAPL,MSFT,NVDA` pools symbols into one
`TimeSeriesDataSet`/model; `forecast`/`explain --bundle <name>` query an
individual symbol's forecast from a pooled bundle.

---

## [PYQ-205]
Local data caching / dataset pinning
Status: Resolved — commit b616184, 2026-07-23
Priority: Medium

Problem: every train/forecast/explain call re-fetched live from
Yahoo/FRED/Finnhub — slow, harder on informal rate limits than necessary,
and not reproducible.

Resolution: `pyquant/data/cache.py` — a local TTL cache
(`.cache/pyquant/`, 1h default) keyed by a fingerprint of
(symbol, date range, enabled sources), plus `--pin NAME` for a named,
TTL-exempt dataset snapshot that replays byte-identical data later.

---

## [PYQ-206]
`scan`'s BUY/SELL signal ignored the forecast's own uncertainty band
Status: Resolved — commit b616184, 2026-07-23
Priority: Medium
Files: `pyquant/cli/app.py` (`scan`)

Problem: the signal heuristic (expected return > +2% -> BUY, < -2% -> SELL)
used only the median forecast — a confident +2.1% and a wildly uncertain
+2.1% (huge p10-p90 spread) rendered identically as "BUY."

Resolution: `scan` now also requires the whole p10-p90 band to sit on one
side of 0% before signaling BUY/SELL, falling back to HOLD otherwise.

---

## [PYQ-207]
`--verbose`/`--debug` CLI flag
Status: Resolved — commit b616184, 2026-07-23
Priority: Low
Files: `pyquant/cli/app.py`, `pyproject.toml`

Problem: logging was hard-set to WARNING with no way to turn diagnostics
back on (e.g. NaN training loss, unexpected feature drift) without editing
source.

Resolution: `--verbose`/`--debug` global flags re-enable INFO/DEBUG logging
and un-silence Lightning's own output. (A narrower gap found later — they
didn't cover `warnings.warn` output — is bugs.md#pyq-108, now also
resolved.)

---

## [PYQ-208]
Add `test_options.py` — was 0% covered
Status: Resolved — commit b616184, 2026-07-23
Priority: Medium
Files: `pyquant/data/options.py`

Problem: put/call ratio, ATM IV, and IV skew all depended on nearest-strike
lookups via `(strike - spot).abs().argmin()` over live option-chain data —
the fiddliest indexing logic in the whole data layer — with zero test
coverage.

Resolution: `test_options.py` added, covering the normal case, an empty
chain, a missing spot price, and the `sentiment_label` threshold boundaries.

---

## [PYQ-209]
Config-file (YAML) support for experiments
Status: Resolved — 2026-07-24
Priority: High
Files: `pyquant/config.py`, `pyquant/cli/app.py`

Problem: `Settings` currently loads only from environment variables /
`.env`. Every hyperparameter (hidden_size, quantiles, learning_rate,
encoder/prediction length, data toggles) is either a hardcoded default or a
one-off CLI flag — there's no way to check a named, reusable experiment
config into version control (e.g. "AAPL baseline", "wide-quantile
aggressive"). The flag surface is also inconsistent today: train/backtest
expose `--period`/`--no-macro`/etc., forecast/explain/scan don't.

Ask: add `--config path.yaml` (or a `PYQUANT_CONFIG` env var), loaded before
env-var overrides via pydantic-settings' YAML config source (or a manual
`yaml.safe_load` merged into `Settings(**overrides)`), so a full experiment
is one file: `uv run pyquant train AAPL --config configs/aapl_baseline.yaml`.
Ship 1–2 example configs under `configs/`.

Acceptance criteria: a YAML file overriding `tft.hidden_size` and
`training.max_epochs` produces a bundle whose `meta.json` reflects those
values; explicit CLI flags still win over the config file.

Resolution: `load_settings(config_path)` layers a YAML file (via
pydantic-settings' `YamlConfigSettingsSource`, wired through
`settings_customise_sources`) *below* env vars but above defaults; a
`PYQUANT_CONFIG` env var is also honored. `train`/`backtest` gained a
`--config` option, threaded through `_build_settings` so explicit CLI flags
still win. Added `pyyaml` to dependencies and two example configs under
`configs/` (`aapl_baseline.yaml`, `wide_quantile_aggressive.yaml`). Covered by
`test_yaml_config_overrides_defaults`, `test_yaml_config_via_env_var`,
`test_no_config_uses_defaults`, and `test_cli_flag_overrides_yaml_config`. (The
inconsistent flag surface the ticket also noted — forecast/explain/scan lacking
`--config` — is left as-is; those don't take hyperparameters.)

---

## [PYQ-210]
`seed_everything` + recorded seed for reproducibility
Status: Resolved — 2026-07-24
Priority: Medium
Files: `pyquant/models/tft.py` (`train`, `walk_forward_backtest`), `pyquant/config.py`

Problem: `Trainer.fit()` is never preceded by
`lightning.pytorch.seed_everything`, and no seed is recorded anywhere. Two
`pyquant train AAPL` runs with byte-identical config — even against a
`--pin`ned dataset — can produce different weights and different val_loss,
undermining `runs.jsonl` (PYQ-203) as a way to compare experiments, and
adding noise on top of the small-validation-window variance
investigations.md#pyq-303 already flags.

Ask: call `seed_everything(seed, workers=True)` at the top of
`train`/`walk_forward_backtest` (default configurable via a new
`TrainingConfig.seed: int = 42`), and record the seed in `meta.json`.

Acceptance criteria: two consecutive `train()` calls with the same seed and
a pinned dataset produce identical val_loss.

Resolution: added `TrainingConfig.seed: int = 42`; `train()` and
`walk_forward_backtest()` call `seed_everything(seed, workers=True)` before any
data loading / weight init, and `train()` records `seed` in `meta.json`.
Covered by `test_train_seeds_everything_and_records_seed` (asserts the seed is
passed to `seed_everything` and persisted to `meta.json`). The seed-variance
comparison in investigations.md#pyq-303 is now unblocked.

---

## [PYQ-211]
Learning-rate tuning instead of one fixed lr for every run
Status: Open
Priority: Medium (downgraded from High — see update)
Files: `pyquant/models/tft.py` (`build_model`, `train`), `pyquant/config.py` (`TrainingConfig.learning_rate`)

Problem: `learning_rate` is a single hardcoded default (0.01) applied
identically regardless of symbol, feature set, or hidden_size, with no
scheduler.

Ask: use Lightning's `Tuner(trainer).lr_find(...)` (the pattern
pytorch-forecasting's own TFT tutorials use) to pick a per-run learning rate
before the real fit, or at minimum swap the flat rate for a scheduler
(`ReduceLROnPlateau`/`CosineAnnealingLR`).

Acceptance criteria: an lr-tuned run on the same symbol/data shows
measurably better calibration_coverage and skill_vs_baseline than the
current fixed-lr default, evaluated via `backtest` rather than a single
5-day window, *after* bugs.md#pyq-109 is fixed.

Update (2026-07-24): this ticket's original motivating evidence (bad
training numbers) turned out to be explained by bugs.md#pyq-109 (evaluating
the wrong checkpoint), not primarily by the LR. A same-epoch-budget
comparison run *before* PYQ-109 was understood (lr=0.01 vs lr=0.001, both
capped at 15 epochs, both suffering from PYQ-109) showed lr=0.001 scoring
worse, not better — the opposite of "lr too high." That comparison is
unreliable until it's re-run against correctly-evaluated (best-checkpoint)
models. LR tuning is still a reasonable improvement on its own merits, just
no longer the leading theory for the bad numbers, hence the priority
downgrade.

---

## [PYQ-212]
Machine-readable output mode (`--format json` / `--quiet`)
Status: Resolved — 2026-07-24
Priority: Medium
Files: `pyquant/cli/app.py`, `pyquant/analysis/serialize.py`

Problem: every command's only output is Rich-formatted tables/panels/charts
for a human terminal. No way to consume a forecast, scan comparison, or
training summary programmatically without scraping rendered text — a
blocker both for scripting and for reusing the same serialization in a
future API (PYQ-213).

Ask: a global `--format json` option that serializes the same data
(`Forecast`, `TrainResult`, `EvaluationMetrics`, etc. are already plain
dataclasses) to stdout instead of Rich tables, plus a `--quiet` flag
suppressing the progress bar/banner.

Acceptance criteria: `pyquant forecast AAPL --format json | jq .median`
works; JSON output has no ANSI escape codes.

Resolution: added global `--format rich|json` and `--quiet`/`-q` options on the
Typer callback. JSON is emitted via stdlib `json.dumps` + `print` (never Rich),
so no ANSI escapes leak in; `--format json` implies quiet. New
`pyquant/analysis/serialize.py` holds the numpy→JSON serializers for
`Forecast`/`EvaluationMetrics`/`TrainResult`/`BacktestResult`/`Interpretation`,
deliberately reusable by the PYQ-213 API. All five commands (forecast, scan,
train, backtest, explain) support both modes. Covered by
`test_forecast_json_output_is_clean_parseable_json` (asserts no `\x1b[` and
parses to `.median`), plus scan/train JSON tests.

---

## [PYQ-213]
Design (and scaffold) a FastAPI service layer alongside the CLI
Status: Resolved — 2026-07-24
Priority: High
Files: new `pyquant/api/`, `pyquant/analysis/*`, `pyquant/models/tft.py`

Problem: the goal is to eventually run PyQuant as a backend for a web
service, not only a CLI. The good news: `analysis/forecast.py`,
`analysis/interpret.py`, `analysis/metrics.py`, `models/tft.py`, and
`data/*` are already fully decoupled from Typer/Rich — `generate_forecast`,
`explain_forecast`, `tft.train`, `tft.load` are plain
functions/dataclasses the CLI happens to call. A REST layer is additive,
not a rewrite.

Ask: research and (once scoped) scaffold a `pyquant/api/` FastAPI app
wrapping the same functions `cli/app.py` already calls, covering at
minimum:
- Response models: pydantic-wrap Forecast/Interpretation/TrainResult/
  EvaluationMetrics for JSON responses (dovetails with PYQ-212's
  serializers).
- Training is long-running (the `train` CLI command already blocks for the
  full fit) — needs a background-job model. FastAPI `BackgroundTasks` + a
  job-status endpoint is enough for a single node; note where it stops
  scaling (no queue, no retries, job state lost on restart) as the trigger
  to move to arq/RQ/Celery + Redis.
- Bundle storage: `checkpoints/<SYMBOL>/` on local disk is fine for one
  instance; multi-instance/serverless deployment needs shared/object
  storage (S3-compatible) — decide this before building anything stateful
  on top.
- Inference concurrency: confirm `TemporalFusionTransformer.predict()` is
  safe under concurrent requests (or serialize per-model), and decide
  whether bundles stay loaded in-process (LRU cache) vs. reloaded per
  request.
- Auth/rate-limiting: nothing in the stack has any today — a public-facing
  API needs at least an API-key gate, given it would be spending the
  operator's FRED/Finnhub quota.
- Deployment: pairs with PYQ-217 (Dockerfile).

Acceptance criteria (as filed): a written design note covering the above —
this is a research/design ticket, not an implementation one yet.

Resolution: written design note delivered at [`docs/api-design.md`](../docs/api-design.md),
covering all six areas — pydantic response models (dovetailing with PYQ-212's
serializers), the background-job model and where in-process `BackgroundTasks`
stops scaling (→ arq/Celery + Redis), local-disk vs. object-storage bundles
(gated on PYQ-220), inference concurrency (per-bundle lock + LRU bundle cache,
plus the PYQ-114/PYQ-306/PYQ-302 dependencies), API-key auth + rate-limiting,
and Docker deployment (PYQ-217). Includes a recommended build order. Scaffolding
remains future work as filed (spawns the follow-ups listed in the note); the
ticket's stated deliverable — the design note — is complete.

---

## [PYQ-214]
Broaden and harden external data providers
Status: Open
Priority: Medium
Files: `pyquant/data/macro.py`, `pyquant/data/prices.py`, `pyquant/data/sentiment.py`

Problem/Ask: three concrete gaps:
1. `FRED_SERIES` only wires up DFF/T10Y2Y/CPIAUCSL; investigations.md#pyq-305's
   `publication_lag_days` convention is implemented and ready to extend to
   unemployment (UNRATE), PCE, or GDP with the same pattern.
2. yfinance is an unofficial, frequently-breaking scraper of Yahoo's
   internal endpoints (no SLA, no key) and is the sole price source for
   OHLCV, VIX, sector ETFs, and options. Worth evaluating a licensed
   fallback (Alpha Vantage, Polygon.io, Tiingo) for the core price feed
   specifically, since it's load-bearing for every other feature.
3. Finnhub's free tier caps news history at 365 days
   (investigations.md#pyq-301) — worth pricing/evaluating an alternative or
   paid tier with deeper history if PYQ-301 concludes sentiment is being
   meaningfully underused.

Acceptance criteria: n/a — research ticket that may spawn concrete
per-source follow-ups.

---

## [PYQ-215]
Retry/backoff for flaky external calls
Status: Resolved — 2026-07-24
Priority: Medium
Files: `pyquant/data/retry.py` (new), `pyquant/data/prices.py` (`fetch_prices`), `pyquant/data/sentiment.py` (`fetch_news`)

Problem: no source retries a transient failure. `fetch_prices` raises
immediately and hard-fails the whole panel build on any yfinance hiccup;
macro/sectors/sentiment/options each catch broadly and silently degrade to
"unavailable," which — per bugs.md#pyq-104's own finding — is
indistinguishable from "no key configured" or "genuinely no data."

Ask: a small shared retry helper (e.g. `tenacity`, or a few lines of manual
backoff) around each network call, so a single flaky request doesn't crash
the run or silently zero out a feature that would have succeeded on retry.

Acceptance criteria: a mocked transient failure (raises once, then
succeeds) is recovered from without the caller seeing an error, for at
least `fetch_prices` and `fetch_news`.

Resolution: added a dependency-free `pyquant/data/retry.py` with
`with_retry(func, attempts=3, base_delay=0.5, ...)` — exponential backoff, logs
each retry, re-raises the last exception if all attempts fail (so callers keep
their existing handling for genuine failures). Wired into `fetch_prices`
(yfinance history) and `fetch_news` (Finnhub request). Covered by
`tests/test_retry.py` plus `test_fetch_prices_recovers_from_transient_failure`
and `test_fetch_news_recovers_from_transient_failure`. Other sources
(macro/sectors/options) already degrade gracefully and were left for a
follow-up if their flakiness proves material.

---

## [PYQ-216]
Detect/guard against quantile crossing
Status: Resolved — 2026-07-24
Priority: Low
Files: `pyquant/analysis/forecast.py`, `pyquant/analysis/metrics.py`

Problem: pytorch-forecasting's `QuantileLoss` does not enforce
`p10 <= p50 <= p90` pointwise, and nothing in `Forecast` or
`evaluate_predictions` checks for it. If it happens, the CLI would silently
render a table with e.g. p10 > p50 for some day, or compute
`calibration_coverage` against a crossed (lower > upper) band with no
signal anything's wrong.

Ask: add a cheap monotonicity check (log a warning, or clip) wherever
predictions are consumed (`Forecast` construction and/or
`evaluate_predictions`), so crossing is visible rather than silently
displayed/scored as-is.

Acceptance criteria: a synthetic crossed-quantile prediction array triggers
a logged warning in a test.

Resolution: added `metrics.warn_on_quantile_crossing(predictions, quantiles)`
— it counts points where a higher quantile falls below a lower one (via
`np.diff` along the quantile axis) and logs a warning. `evaluate_predictions`
calls it on every batch. Covered by `test_warn_on_quantile_crossing_flags_crossed_band`,
`test_warn_on_quantile_crossing_silent_when_monotonic`, and
`test_evaluate_predictions_warns_on_crossing`. (Relies on ascending quantiles,
now enforced by PYQ-219.)

---

## [PYQ-217]
Dockerfile for reproducible training/serving environments
Status: Open
Priority: Medium
Files: new `Dockerfile`

Problem: no containerization exists. This matters for two reasons: (a)
torch/lightning/pytorch-forecasting resolve CUDA wheels in a
host-dependent way, so training has real "works on my machine" risk; (b)
PYQ-213's API service will need a deployable image regardless.

Ask: a CPU-only base image good enough for inference/serving (small, fast
to build), plus notes on a CUDA variant for training, both built via
`uv sync --frozen`.

Acceptance criteria: `docker build` + `docker run ... pyquant forecast
AAPL` (against a mounted checkpoint) works.

---

## [PYQ-218]
Make DataLoader `num_workers` configurable (hardcoded to 0 everywhere)
Status: Resolved — 2026-07-24
Priority: Medium
Files: `pyquant/models/tft.py` (every `to_dataloader(...)` call)

Problem: every `to_dataloader(...)` call in `train()`,
`walk_forward_backtest()`, `predict_quantiles()`, and `interpret()` hardcodes
`num_workers=0` — single-process data loading, no parallelism. Lightning
itself flags this at runtime ("The 'predict_dataloader' does not have many
workers which may be a bottleneck. Consider increasing the value of the
`num_workers` argument to `num_workers=19`..."). That warning is a
`PossibleUserWarning` (a `UserWarning` subclass), which means bugs.md#pyq-108's
fix — filtering `UserWarning` by default — now silences this exact hint by
default, permanently hiding it from users unless they pass `--debug`.

Ask: add a `TrainingConfig.num_workers: int = 0` setting (or a sensible
non-zero default, e.g. based on `os.cpu_count()`), threaded through every
`to_dataloader()` call, so the hint PYQ-108 just silenced actually has a
knob to act on.

Acceptance criteria: setting `TRAINING__NUM_WORKERS=4` measurably changes
DataLoader worker count (observable via the dataloader's `num_workers`
attribute in a test, without requiring an actual multi-core speed
comparison).

Resolution: added `TrainingConfig.num_workers: int = 0` (0 keeps today's
behavior), threaded into the train/val `to_dataloader(...)` calls in both
`train()` and `walk_forward_backtest()`. The predict/interpret paths keep
`num_workers=0` deliberately (batch_size=1 single-sample inference gains
nothing from workers). Covered by
`test_train_threads_num_workers_into_dataloaders`.

---

## [PYQ-219]
Validate `TFTConfig.quantiles` is sorted ascending
Status: Resolved — 2026-07-24
Priority: Low
Files: `pyquant/config.py` (`TFTConfig.quantiles`), `pyquant/analysis/metrics.py` (`evaluate_predictions`)

Problem: `evaluate_predictions()` takes `lower = predictions[:, :, 0]` and
`upper = predictions[:, :, -1]` — i.e. it assumes `quantiles` is sorted
ascending, so the first configured quantile is the lowest and the last is
the highest. Nothing validates this. `Forecast`'s own rendering
(`quantile_series`, the CLI table) is actually order-independent since it
looks up quantiles by value via `.index(q)` — but the calibration/skill
metrics are not. A user setting `TFT__QUANTILES='[0.9,0.1,0.5]'` (order
matters even though the *set* of values would be identical to
`[0.1,0.5,0.9]`) would silently get `calibration_coverage` computed against
p90 as the "lower" bound and p10 as the "upper" bound — inverted and
nonsensical, with no error. This gets more likely once PYQ-209 (YAML
config) makes hand-editing a quantile list easy.

Ask: a pydantic validator on `TFTConfig.quantiles` enforcing the list is
sorted ascending (and, per bugs.md#pyq-106's existing check elsewhere,
ideally also that 0.5 is present, though that's already enforced downstream
at use time).

Acceptance criteria: constructing `TFTConfig(quantiles=[0.9, 0.1, 0.5])`
raises a clear validation error instead of silently accepting it.

Resolution: added a pydantic `field_validator` on `TFTConfig.quantiles` that
rejects any list not equal to its own sorted order, with a message explaining
the first/last entries are the calibration-band bounds. Covered by
`test_tft_quantiles_reject_unsorted` / `test_tft_quantiles_accept_sorted` in the
new `tests/test_config.py`. (0.5-presence is left enforced downstream at use
time per bugs.md#pyq-106.)

---

## [PYQ-220]
`checkpoint_dir`/`cache_dir` resolve relative to CWD
Status: Open
Priority: Medium
Files: `pyquant/config.py` (`Settings.checkpoint_dir`, `DataConfig.cache_dir`)

Problem: both default to relative paths (`Path("checkpoints")`,
`Path(".cache/pyquant")`), resolved against whatever the current working
directory happens to be at runtime. For CLI use this is a mild paper cut
(`pyquant train AAPL` from the repo root, then `pyquant forecast AAPL` from
a different directory, fails to find the bundle with a — reasonably clear —
`FileNotFoundError`). It's a bigger problem for PYQ-213's proposed API
service: a server process's working directory is not guaranteed to be the
repo root, and getting this wrong means checkpoints/cache silently land
somewhere unexpected (or, worse, a different working directory per restart
means the service can't find bundles it itself created).

Ask: default to an absolute, XDG-style location (e.g. under
`platformdirs.user_data_dir("pyquant")`) or at minimum resolve the
configured relative path against a fixed anchor (e.g. the project root)
rather than the ambient CWD, while keeping today's paths as an explicit
override for anyone who wants repo-local bundles.

Acceptance criteria: running the equivalent of `train` then `forecast` from
two different working directories (same `Settings`) finds the same bundle.

---

## [PYQ-221]
`pyquant cache` subcommand — the local panel cache has no eviction/pruning
Status: Resolved — 2026-07-24
Priority: Low
Files: `pyquant/data/cache.py`, `pyquant/cli/app.py`

Problem: `write_cache()` writes a new `.pkl`/`.meta.json` pair per unique
fingerprint forever; `write_pin()` similarly never expires. TTL only gates
*read* validity (`read_cache` returns `None` past TTL) — nothing ever
deletes the underlying files. Over months of varied symbol/date-range/
settings combinations, `.cache/pyquant/` grows unboundedly, and pins
accumulate with no way to list or remove them short of `rm` by hand.

Ask: a `pyquant cache` subcommand — at minimum `list` (size, entry count,
pin names) and `prune` (delete TTL-expired entries); `pyquant cache rm-pin
NAME` for pin cleanup would round it out.

Acceptance criteria: `pyquant cache prune` removes expired cache files
without touching valid ones or any pins.

Resolution: added `cache.cache_stats()`, `cache.prune_expired()`,
`cache.list_pins()`, and `cache.remove_pin()` (pruning globs only top-level TTL
entries, never the `pins/` subdir), plus a `pyquant cache` Typer sub-app with
`list`, `prune`, and `rm-pin` commands (all `--format json`-aware). Covered by
`test_prune_expired_removes_only_stale_entries_and_keeps_pins`,
`test_cache_stats_counts_entries_and_pins`, `test_remove_pin`, and a CLI smoke
test.

---

## [PYQ-222]
`train`'s `console.status()` spinner competes with Lightning's own progress bar
Status: Resolved — 2026-07-24
Priority: Low
Files: `pyquant/cli/app.py` (`train`)

Problem: `train` wraps `tft.train(..., progress=True)` in
`with console.status("Fetching data and training..."):` — but
`progress=True` means Lightning renders its own live progress bar to the
same terminal for the duration of the fit, on a separate `Console`/`Live`
instance. In practice this doesn't crash (different `Live` instances don't
raise Rich's "only one live display" error), but the Rich spinner's message
is immediately superseded by Lightning's own output rather than being
visible at all — dead UI code, not actively harmful, just not doing
anything. `backtest` avoids this entirely by passing `progress=False` to
keep its own `console.status()` spinner as the only live region, which is
the more deliberate pattern.

Ask: either drop the redundant `console.status()` wrapper around `train`
(let Lightning's progress bar be the only live indicator, matching what
users actually see today), or make `train` consistent with `backtest` by
defaulting `progress=False` and relying on the Rich spinner instead
(loses per-epoch visibility, gains a cleaner default view — worth deciding
which is preferred rather than leaving the current accidental mix).

Acceptance criteria: n/a — this is a small design decision, not a
correctness fix; acceptance is "pick one pattern and apply it consistently
across train/backtest."

Resolution (pattern chosen): dropped the redundant `console.status()` wrapper
in the `train` command, letting Lightning's own live progress bar
(`progress=True`) be the single live indicator — matching what users actually
saw anyway. `backtest` keeps its `console.status()` spinner with
`progress=False`, so each command now has exactly one deliberate live region.

---

## [PYQ-223]
Mixed-precision training option (Trainer has no `precision=` set, fp32 only)
Status: Resolved — 2026-07-24
Priority: Low
Files: `pyquant/models/tft.py` (`train`, `walk_forward_backtest`)

Problem: every `Trainer(...)` construction leaves `precision` at its
Lightning default (32-bit). On hardware that supports it (most modern GPUs,
and Apple Silicon via MPS to a lesser extent), `precision="bf16-mixed"` (or
`"16-mixed"`) is a standard, low-risk way to speed up training and reduce
memory use with minimal accuracy impact — a reasonable "best PyTorch
practice" gap for a project explicitly aiming to use current best practices.

Ask: expose `TrainingConfig.precision: str = "32-true"` (Lightning's own
precision string values) and thread it into both `Trainer` constructions,
defaulting to today's behavior so nothing changes unless a user opts in.

Acceptance criteria: setting `TRAINING__PRECISION=bf16-mixed` on capable
hardware measurably reduces peak memory or wall-clock time for a training
run, with model quality (skill_vs_baseline) not meaningfully worse.

Resolution: added `TrainingConfig.precision: str = "32-true"` (Lightning's own
precision strings), threaded into both `Trainer(...)` constructions, defaulting
to today's fp32 so nothing changes unless opted in. Covered by
`test_train_threads_precision_into_trainer` (asserts the configured value
reaches the `Trainer`; the hardware speed/memory comparison in the acceptance
criteria needs capable GPU hardware and is left to the operator).

---

## [PYQ-224]
Make `EarlyStopping` patience configurable (hardcoded to 5 in both trainers)
Status: Resolved — 2026-07-26
Priority: Medium
Files: `pyquant/models/tft.py` (`train`, `walk_forward_backtest`), `pyquant/config.py` (`TrainingConfig`)

Problem: `EarlyStopping(monitor="val_loss", patience=5, mode="min")` is
constructed with a literal in both places. Meanwhile `max_epochs`,
`learning_rate`, `gradient_clip_val`, `precision`, `num_workers` and `seed`
are all configurable via `TrainingConfig` and YAML. Patience is arguably the
most consequential of the set given how noisy the selection metric is
(bugs.md#pyq-117): with a small validation sample, 5 is either far too
impatient or meaningless depending on the run.

Ask: add `TrainingConfig.early_stopping_patience: int = 5` (preserving today's
behaviour as the default) and thread it into both `EarlyStopping`
constructions, the same way PYQ-218/PYQ-223 threaded `num_workers`/`precision`.

Acceptance criteria: a test asserting the configured value reaches the
callback in both `train` and `walk_forward_backtest`.

Resolution: added `TrainingConfig.early_stopping_patience: int = 5` (preserving
today's behaviour) and threaded it into both `EarlyStopping` constructions, matching
how PYQ-218/PYQ-223 threaded `num_workers`/`precision`.

Covered by `test_train_threads_early_stopping_patience_into_the_callback`,
`test_backtest_threads_early_stopping_patience_into_the_callback` (asserting it
reaches all of the backtest's per-window trainers) and a config default test.

---

## [PYQ-225]
Record full provenance in `meta.json` (version, git sha, pin, resolved config)
Status: Resolved — 2026-07-26
Priority: Medium
Files: `pyquant/models/tft.py` (`train`), `pyquant/config.py`

Problem: the project has invested deliberately in reproducibility — PYQ-210's
`seed_everything` plus a recorded seed, PYQ-205's TTL cache and TTL-exempt
dataset pins, PYQ-209's checked-in YAML experiment configs. But `meta.json`
records only `seed`, `quantiles`, `max_encoder_length`,
`max_prediction_length`, `features` and `trained_at`. It does not record:

- the `pyquant` version or a git sha — so you cannot tell which code produced
  a bundle, and the PYQ-121 RSI change (for instance) silently changes what a
  feature name means across bundles;
- the `pin` name, if the run used one — so the reproducibility mechanism does
  not record that it was used;
- the resolved `data`/`training` config — see bugs.md#pyq-119, which needs
  this anyway.

Together those are the missing third leg: seed + pinned data + *code version*
is what actually reproduces a run.

Ask: write a `provenance` block into `meta.json` (and therefore into the
append-only `runs.jsonl`) carrying the package version, git sha when
available, pin name, and the resolved `data`/`training` sub-configs.

Acceptance criteria: a test asserting a freshly trained bundle's `meta.json`
carries the package version and, when a `pin` was passed, the pin name.

Resolution: `train()` writes a `provenance` block into `meta.json`/`runs.jsonl`
carrying `pyquant_version` (via `importlib.metadata`, `"unknown"` when running from
an uninstalled source tree), `git_sha` (best-effort `git rev-parse --short HEAD`,
`None` outside a repo) and the `pin` name when one was used. The resolved
`data`/`training`/`tft` config is recorded alongside it by bugs.md#pyq-119.

Together with PYQ-210's seed and PYQ-205's dataset pins, that is the set which
actually reproduces a run. It is not hypothetical: bugs.md#pyq-121 redefined what
`RSI_14` means, so two bundles with identical feature *names* can have been trained
on different data.

Covered by `test_train_records_provenance_including_the_pin`.

---

## [PYQ-226]
Report metric dispersion across backtest windows, not just the mean
Status: Resolved — 2026-07-26
Priority: Medium
Files: `pyquant/cli/app.py` (`backtest`), `pyquant/analysis/metrics.py` (`aggregate_metrics`)

Problem: `walk_forward_backtest` computes `per_window: list[EvaluationMetrics]`
and `aggregate_metrics()` averages them. The Rich table prints only the
averages. With `--windows 5` the spread across windows *is* the finding — a
mean directional accuracy of 60% built from windows at 100/20/100/40/40 says
something very different from five windows at 60%. `--format json` already
exposes `per_window`, so the information exists and is simply not shown.

Directly serves the stability question in investigations.md#pyq-303, and pairs
with bugs.md#pyq-127 (until that lands, the "windows" are all the same window,
so dispersion would read as zero for the wrong reason).

Ask: print min/max (or standard deviation) per metric alongside the mean, and
consider a compact per-window row listing so the walk is visible.

Acceptance criteria: a test asserting the backtest table shows per-window
spread, not only the aggregate.

Resolution: `backtest` now prints a `Per-window results` table beneath the
aggregate whenever there is more than one window, listing each window's model MAE,
baseline MAE, skill, directional accuracy and band coverage. The aggregate row also
carries `n_samples`/`n_points` from bugs.md#pyq-117.

This only became meaningful once bugs.md#pyq-127 landed: before that every origin
evaluated the same final window, so the "spread" would have reflected model-init
noise rather than performance across time.

Covered by `test_backtest_table_shows_per_window_spread`, which asserts the 20%/100%
window values appear and not merely their 60% mean.

---

## [PYQ-227]
Per-quantile calibration + pinball loss alongside band coverage
Status: Open
Priority: Medium
Files: `pyquant/analysis/metrics.py`

Problem: `calibration_coverage` measures only the outermost band (p10-p90 by
default) as a single number. For a quantile model that is the least
informative calibration statistic available: it cannot distinguish a
well-calibrated band from one that is too wide on one side and too narrow on
the other, and it says nothing about the interior quantiles that
`configs/wide_quantile_aggressive.yaml` deliberately adds.

Two additions are standard and cheap here:

- **per-quantile exceedance**: for each configured quantile q, the empirical
  fraction of actuals below the prediction. A calibrated p10 should sit near
  0.10. This is the diagnostic that tells you *which side* of the band is
  wrong.
- **pinball (quantile) loss**: the proper scoring rule the model is actually
  trained on, reported per quantile and averaged — directly comparable across
  configs in a way `val_loss` is not (it is reported on normalized targets).

Also worth noting: `model_mae`/`baseline_mae` are in dollars and therefore not
comparable across symbols in `scan`. `skill_vs_baseline` is effectively
1 - MASE and *is* scale-free, which is the right primary number — it is simply
shown third.

Ask: add per-quantile exceedance rates and pinball loss to
`EvaluationMetrics`, surface them in `--format json` always and in the Rich
table for `backtest` (where there is room).

Acceptance criteria: unit tests for both statistics against hand-computed
values on a small known array.

---

## [PYQ-228]
Bound dependency majors; pass `auto_adjust` to `yfinance` explicitly
Status: Open
Priority: Low
Files: `pyproject.toml`, `pyquant/data/prices.py` (`fetch_prices`), `pyquant/data/macro.py` (`_fetch_vix`)

Problem: every dependency is specified with a lower bound and no upper bound.
For most of the stack that is tolerable; for `yfinance` it is not. The
declared constraint is `yfinance>=0.2.40` and `uv.lock` has already resolved
to **1.4.1** — a major-version jump on the least stable dependency in the
project, whose `Ticker.history` signature is now just `(self, *args, **kwargs)`.

Concretely: `fetch_prices()` and `_fetch_vix()` never pass `auto_adjust`, whose
default flipped to `True` during the 0.2.x series. So whether `Close` is
split/dividend-adjusted — which changes every price level, every technical
indicator derived from it, and therefore every trained model — is decided by
whatever version happens to resolve. `sectors.py` already passes
`auto_adjust=True` explicitly, so the codebase is internally inconsistent on
the point too.

`uv.lock` protects developers and CI. It does not protect anyone who installs
the package, and it did not prevent the silent 0.2 -> 1.4 jump.

Ask: pass `auto_adjust` explicitly everywhere prices are fetched (and document
which convention the model assumes), and cap majors on at least `yfinance`,
`torch`, `lightning` and `pytorch-forecasting`.

Acceptance criteria: a test asserting `fetch_prices` passes an explicit
`auto_adjust` to `yfinance`; constraints updated in `pyproject.toml`.

---

## [PYQ-229]
CI: Python matrix, frozen install, `uv lock --check`, `ruff format --check`
Status: Open
Priority: Low
Files: `.github/workflows/ci.yml`, `pyproject.toml`

Problem: four gaps in an otherwise solid pipeline (lint + backlog check +
tests all gated):

- `pyproject.toml` declares `requires-python = ">=3.10,<3.13"` but CI installs
  only 3.12. 3.10 and 3.11 are completely unverified, despite being supported
  on paper. `from __future__ import annotations` is used consistently, so the
  risk is mostly in third-party resolution rather than syntax — but that is
  exactly what a matrix would tell us.
- `uv sync --extra dev` is not `--frozen`, and nothing runs `uv lock --check`,
  so a `pyproject.toml` edit that desyncs the lockfile passes CI silently.
- No `ruff format --check`. Lint rules are enforced; formatting is not, so
  style drift is invisible until someone runs the formatter and produces a
  large unrelated diff.

Ask: add a `python-version: ["3.10", "3.11", "3.12"]` matrix, switch to a
frozen install plus `uv lock --check`, and add `ruff format --check .`.

Acceptance criteria: CI green across all matrix entries; a deliberate lockfile
desync fails the build.

---

## [PYQ-230]
CI: measure and report test coverage
Status: Open
Priority: Low
Files: `.github/workflows/ci.yml`, `pyproject.toml`

Problem: `pytest-cov` is not a dev dependency and coverage is never measured.
investigations.md#pyq-304 is marked Resolved on the strength of a one-off
local coverage run, but nothing makes that repeatable or visible, so the number
it produced is already stale and unverifiable. With 122 tests the suite is
substantial enough that the interesting question is no longer "is there
coverage" but "which paths are uncovered" — and the answer to that turned out
to matter: bugs.md#pyq-120 notes that no CLI failure path is tested at all.

Ask: add `pytest-cov` to the dev extra, a `[tool.coverage]` config excluding
the obvious untestable branches, and a CI step printing the report. A
threshold gate is optional and probably premature; visibility is the point.

Acceptance criteria: CI prints a per-module coverage table; the number is
reproducible locally with one documented command.

---

## [PYQ-231]
CLI failure-path test coverage — every existing test asserts `exit_code == 0`
Status: Resolved — 2026-07-26
Priority: Medium
Files: `tests/test_cli.py`

Problem: all 18 CLI tests assert `result.exit_code == 0`. Not one exercises a
failure: no untrained bundle, no invalid `--format`, no unreadable config
path, no data-source failure, no insufficient-history error. The CLI is the
only user-facing surface in the project and half of its behaviour — what it
does when something is wrong — is untested. bugs.md#pyq-120 (a raw traceback
on the most likely first-time user error) is a direct consequence.

Ask: add failure-path tests covering at minimum: forecast/explain on an
untrained symbol, an invalid `--format` value, a `--config` pointing at a
missing file, and `train` on a symbol with insufficient history. Assert exit
codes *and* that the message is a clean one-liner rather than a traceback.

Acceptance criteria: each listed failure has a test asserting a non-zero exit
code and a readable message.

Resolution: added failure-path coverage for every case the ticket listed --
`forecast`/`explain` on an untrained symbol, an invalid `--format`, a `--config`
pointing at a missing file, and `train` on insufficient history. Each asserts a
non-zero exit code, that the raised object is a `SystemExit` rather than the
original exception leaking through, and that the output is a readable one-liner
with no traceback.

Writing them found two real defects rather than merely documenting existing
behaviour: bugs.md#pyq-120 (the traceback) and bugs.md#pyq-128 (a missing
`--config` silently training on defaults), which is the argument for the ticket.
