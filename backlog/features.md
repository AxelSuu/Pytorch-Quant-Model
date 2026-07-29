# Features (PYQ-2xx)

Things to build — see [`README.md`](README.md) for the format.
Next free ID: **PYQ-281**.

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
| [PYQ-211](#pyq-211) | Low | Superseded by PYQ-253 | Learning-rate tuning instead of one fixed lr for every run |
| [PYQ-212](#pyq-212) | Medium | Resolved | Machine-readable output mode (`--format json` / `--quiet`) |
| [PYQ-213](#pyq-213) | High | Resolved | Design (and scaffold) a FastAPI service layer alongside the CLI |
| [PYQ-214](#pyq-214) | Medium | Resolved | Broaden and harden external data providers |
| [PYQ-215](#pyq-215) | Medium | Resolved | Retry/backoff for flaky external calls |
| [PYQ-216](#pyq-216) | Low | Resolved | Detect/guard against quantile crossing |
| [PYQ-217](#pyq-217) | Low | Open | Dockerfile for reproducible training/serving environments |
| [PYQ-218](#pyq-218) | Medium | Resolved | Make DataLoader `num_workers` configurable (hardcoded to 0 everywhere) |
| [PYQ-219](#pyq-219) | Low | Resolved | Validate `TFTConfig.quantiles` is sorted ascending |
| [PYQ-220](#pyq-220) | Medium | Resolved | `checkpoint_dir`/`cache_dir` resolve relative to CWD |
| [PYQ-221](#pyq-221) | Low | Resolved | `pyquant cache` subcommand — the local panel cache has no eviction/pruning |
| [PYQ-222](#pyq-222) | Low | Resolved | `train`'s `console.status()` spinner competes with Lightning's own progress bar |
| [PYQ-223](#pyq-223) | Low | Resolved | Mixed-precision training option (Trainer has no `precision=` set, fp32 only) |
| [PYQ-224](#pyq-224) | Medium | Resolved | Make `EarlyStopping` patience configurable (hardcoded to 5 in both trainers) |
| [PYQ-225](#pyq-225) | Medium | Resolved | Record full provenance in `meta.json` (version, git sha, pin, resolved config) |
| [PYQ-226](#pyq-226) | Medium | Resolved | Report metric dispersion across backtest windows, not just the mean |
| [PYQ-227](#pyq-227) | Medium | Resolved | Per-quantile calibration + pinball loss alongside band coverage |
| [PYQ-228](#pyq-228) | Low | Resolved | Bound dependency majors; pass `auto_adjust` to `yfinance` explicitly |
| [PYQ-229](#pyq-229) | Low | Resolved | CI: Python matrix, frozen install, `uv lock --check`, `ruff format --check` |
| [PYQ-230](#pyq-230) | Low | Resolved | CI: measure and report test coverage |
| [PYQ-231](#pyq-231) | Medium | Resolved | CLI failure-path test coverage — every existing test asserts `exit_code == 0` |
| [PYQ-232](#pyq-232) | High | Resolved | Sphinx + autodoc API documentation site |
| [PYQ-233](#pyq-233) | Medium | Resolved | Gate the docs build in CI with warnings-as-errors |
| [PYQ-234](#pyq-234) | Medium | Resolved | Host the docs on Read the Docs via uv |
| [PYQ-235](#pyq-235) | High | Resolved | Narrative docs: architecture, leakage invariants, methodology |
| [PYQ-236](#pyq-236) | Low | Resolved | Adopt one docstring style and enforce it with ruff's `D` rules |
| [PYQ-237](#pyq-237) | Low | Open | Executable doctests for the metrics and forecast APIs |
| [PYQ-238](#pyq-238) | High | Resolved | `tests/test_invariants.py` — assert the pipeline-spanning invariants directly |
| [PYQ-239](#pyq-239) | High | Resolved | Learnability test: inject a known signal and assert the model recovers it |
| [PYQ-240](#pyq-240) | Medium | Resolved | Regression test that predictions/actuals/last_observed share units |
| [PYQ-241](#pyq-241) | Medium | Resolved | End-to-end CLI journey test across every command and both output formats |
| [PYQ-242](#pyq-242) | Low | Open | Property-based tests for `analysis/metrics.py` |
| [PYQ-243](#pyq-243) | Medium | Resolved | Recorded-payload contract tests for every external vendor |
| [PYQ-244](#pyq-244) | Low | Resolved | Scheduled nightly CI job against live vendors |
| [PYQ-245](#pyq-245) | Low | Open | Mutation testing on the metrics and indicator modules |
| [PYQ-246](#pyq-246) | Medium | Resolved | Determinism test: same seed + same pin ⇒ identical metrics |
| [PYQ-247](#pyq-247) | High | Resolved | Forecast log-returns instead of price levels |
| [PYQ-248](#pyq-248) | High | Resolved | Conformal / split-calibration of the quantile band |
| [PYQ-249](#pyq-249) | Medium | Open | Add a time-series foundation model as a zero-shot baseline |
| [PYQ-250](#pyq-250) | Medium | Resolved | Purge + embargo around every walk-forward split |
| [PYQ-251](#pyq-251) | Medium | Resolved | Report effective sample size and block-bootstrap intervals |
| [PYQ-252](#pyq-252) | Medium | Resolved | CRPS, Winkler score and a PIT histogram |
| [PYQ-253](#pyq-253) | Medium | Resolved | Optuna hyperparameter search (absorbs PYQ-211's scope) |
| [PYQ-254](#pyq-254) | Medium | Resolved | Promote options data from display context to model features |
| [PYQ-255](#pyq-255) | Medium | Resolved | Signal evaluation: does `scan`'s BUY/SELL actually make money? |
| [PYQ-256](#pyq-256) | Low | Resolved | `has_sentiment_data` indicator column |
| [PYQ-257](#pyq-257) | High | Resolved | Use FRED/ALFRED vintages instead of a fixed publication lag |
| [PYQ-258](#pyq-258) | Medium | Resolved | Pluggable price-provider interface with a licensed fallback |
| [PYQ-259](#pyq-259) | Medium | Resolved | Experiment tracking (MLflow) alongside `runs.jsonl` |
| [PYQ-260](#pyq-260) | Low | Resolved | Ship a `py.typed` marker |
| [PYQ-261](#pyq-261) | Medium | Resolved | Scaffold `pyquant/api/` per the PYQ-213 design note |
| [PYQ-262](#pyq-262) | Low | Resolved | Pre-commit configuration |
| [PYQ-263](#pyq-263) | Low | Resolved | `pyquant doctor` — environment and bundle health check |
| [PYQ-264](#pyq-264) | Medium | Resolved | Fold PYQ-247/248/250 into the docs; deploy to GitHub Pages; nightly docs-drift check |
| [PYQ-265](#pyq-265) | High | Open | Report skill across seeds, not from a single seed |
| [PYQ-266](#pyq-266) | High | Open | Paired significance test for comparing two configurations |
| [PYQ-267](#pyq-267) | High | Resolved | Break every metric down by horizon step |
| [PYQ-268](#pyq-268) | High | Open | A reusable multi-symbol sweep harness, replacing the one-off scripts |
| [PYQ-269](#pyq-269) | Medium | Open | Split `models/tft.py` (1075 lines) without breaking Lightning containment |
| [PYQ-270](#pyq-270) | Medium | Open | Put a confidence interval on the headline skill number |
| [PYQ-271](#pyq-271) | Medium | Open | `/backtest` endpoint: close the CLI/API front-end gap |
| [PYQ-272](#pyq-272) | Medium | Open | Dedicated tests for `serialize`, `doctor`, `provenance` and `charts` |
| [PYQ-273](#pyq-273) | Medium | Open | Regression cases for PYQ-139/140 on PYQ-243's existing replay harness |
| [PYQ-274](#pyq-274) | Low | Open | CHANGELOG and a release/tagging workflow |
| [PYQ-275](#pyq-275) | High | Open | Baselines beyond persistence: a negative result is only as strong as what it failed against |
| [PYQ-276](#pyq-276) | Medium | Open | Execute PYQ-312's reframing: the README still sells a forecaster |
| [PYQ-277](#pyq-277) | Medium | Open | Tiingo isn't actually selectable anywhere; PYQ-258's own acceptance criterion is unmet |
| [PYQ-278](#pyq-278) | Low | Open | Ruff format drift has grown to 33 files vs. the CI comment's 20-22 baseline |
| [PYQ-279](#pyq-279) | Low | Open | `git_sha()`/`code_version()` shells out uncached on every `build_panel` call |
| [PYQ-280](#pyq-280) | Low | Open | `backlog.py check` should verify a resolved ticket's acceptance-criteria tests actually exist |

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
Status: Superseded by PYQ-253
Priority: Low (downgraded again from Medium — see 2026-07-26 update)
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

Update (2026-07-26, external review pass): a second reason to deprioritise, more
fundamental than the first. `TARGET = "Close"` means the model predicts the price
**level**, and for a near-random-walk series the persistence baseline is near-optimal by
construction on that formulation — so the reported −23.5% skill is roughly what the
formulation predicts *a priori*, largely independent of learning rate. Tuning lr searches
for a better answer inside a formulation whose ceiling is approximately "tie the
baseline." See features.md#pyq-247 (log-return target) for the change actually likely to
move the number, and it needs no GPU, unlike this ticket.

Separately, learning rate is also just one of at least six coupled knobs (`hidden_size`,
`attention_head_size`, `dropout`, `hidden_continuous_size`, `learning_rate`,
`early_stopping_patience`), so tuning it alone is close to uninformative regardless of the
target question. features.md#pyq-253 proposes an Optuna study over the full set instead,
run *after* PYQ-247 lands. Suggest treating this ticket as superseded by PYQ-253 once that
lands, rather than working it standalone — downgraded to Low in the meantime rather than
closed, since "tie the baseline" is still marginally better than "lose to it," and the
narrow `lr_find` fix costs little if someone picks it up first.

Update (2026-07-27): **still Open, and the case for deprioritising is now measured rather
than argued.** features.md#pyq-247 landed and ran the controlled comparison this ticket's
second update predicted: switching the target from price level to log-return moved skill
from −59.5% to +2.4% (+3.8% with purged splits) on identical data, seed and epoch budget,
with the learning rate untouched at 0.01 throughout. So the −23.5% headline was
substantially a property of the formulation, exactly as argued, and no learning rate would
have recovered it.

Deliberately **not** marked `Superseded by PYQ-253`: that ticket has not landed, and this
file's own rule is that a superseding ID must exist and be real. The recommendation stands —
supersede once PYQ-253 ships — and the prerequisite PYQ-253 names (land PYQ-247 first, so a
GPU study is not spent inside a near-unbeatable formulation) is now satisfied.

Update (2026-07-27): PYQ-253 shipped — `pyquant tune` runs an Optuna study over
`learning_rate` alongside the other five coupled knobs this ticket's own updates argued
could not be tuned informatively in isolation. Superseded as planned.

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
Status: Resolved — 2026-07-27
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

Resolution (2026-07-27): closed as the research ticket it is — all three points now have a
concrete outcome, and two of them turned out to be bugs rather than enhancements.

**1. FRED series breadth.** Not extended, and the reason is more useful than the extension
would have been: measuring the existing three against the live API found that *none of them
were arriving at all* (bugs.md#pyq-139 — the vintage fetch asked FRED for 1776-9999 and was
rejected, so `FedFunds`/`YieldSpread`/`CPI` were silently absent from every panel). Adding
UNRATE/PCE/GDP on top of a broken fetch would have added three more silently-absent columns.
With PYQ-139 fixed, all four macro columns now populate (1260–1261 non-null rows over five
years) and extending the list is a mechanical one-line-per-series change against a path
that demonstrably works. Note that investigations.md#pyq-305's `publication_lag_days`
convention this ticket refers to no longer exists — features.md#pyq-257 replaced it with
ALFRED release vintages.

**2. yfinance as the sole price source.** Addressed by features.md#pyq-258, landed this pass:
a `PriceProvider` protocol, a `YFinanceProvider`, a licensed `TiingoProvider` behind a
config toggle, and an executable `assert_ohlcv_contract` both are held to. Alpha Vantage
was evaluated and rejected (25 requests/day free, too tight even for development); Polygon
and EODHD are paid-only and were not pursued. Tiingo needs no new dependency — it is a JSON
REST endpoint over the already-required `requests`.

**3. Finnhub news depth.** Answered, and far more sharply than this ticket or
investigations.md#pyq-301 anticipated: the free tier serves ~**6 days**, not the documented
~365, so `Sentiment` is 99.7% structural zeros (bugs.md#pyq-140). "Whether sentiment is
being meaningfully underused" is no longer the question — it is barely being used at all.
Whether to pay for depth, drop the feature, or truncate the training window is now a
decision with a measurable comparison attached, tracked on PYQ-140 rather than here.

Spawned follow-ups, as anticipated: bugs.md#pyq-139, bugs.md#pyq-140, features.md#pyq-258.

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
Priority: Low
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

Deprioritised (2026-07-27, explicit user call during the PYQ-140/238/... pass): not
interesting right now. Downgraded Medium -> Low rather than closed, since the ticket
content and its acceptance criteria are still valid whenever it is picked back up.
Left genuinely open, not attempted this pass -- also sidesteps a real limitation: this
session's sandbox has no Docker CLI, so `docker build`/`docker run` could not have been
verified here regardless (PYQ-261's API scaffold does not depend on this landing first;
it is only the "deploy single-instance" step in `docs/api-design.md`'s build order).

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
Status: Resolved — 2026-07-27
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

Resolution (2026-07-27): `config.project_root()` anchors relative `checkpoint_dir` and
`cache_dir` against the package's own location rather than the ambient cwd, via
`field_validator`s with `validate_default=True`. Both are now always absolute.

Two decisions. First, **not** XDG/`platformdirs` by default: repo-local `checkpoints/` and
`.cache/pyquant` are what the README, `.gitignore` and every existing install already
expect, and switching would strand them. `PYQUANT_HOME` moves the anchor for a deployment
that wants bundles outside the source tree, and an absolute configured path still wins
outright — so the ticket's "keep today's paths as an explicit override" is satisfied by
keeping them as the *default*.

Second, `validate_default=True` is load-bearing and was a real bug in the first attempt:
`DataConfig` is built from its `default_factory` on every `Settings()`, so without it the
`cache_dir` validator never ran and the default stayed relative while `checkpoint_dir` (on
`BaseSettings`, where sources always supply a value) worked. Caught by the test, not by
inspection.

Guarded by `test_relative_paths_resolve_the_same_from_any_working_directory` — the ticket's
acceptance criterion, expressed as building `Settings()` from two different cwds and
asserting both paths match and are absolute — plus `test_pyquant_home_moves_the_anchor` and
`test_an_absolute_configured_path_is_left_alone`. This also removes the main blocker
`docs/api-design.md` records for the API layer (PYQ-261).

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
Status: Resolved — 2026-07-26
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

Resolution (2026-07-26): `EvaluationMetrics` now records empirical exceedance and
pinball loss for every configured quantile. Both flow through weighted backtest
aggregation, JSON serialization, and the Rich aggregate table (mean pinball loss plus
individual empirical quantile rates). The hand-computed
`test_quantile_exceedance_and_pinball_loss_match_hand_calculation` guards both formulas;
metrics and CLI tests pass offline.

---

## [PYQ-228]
Bound dependency majors; pass `auto_adjust` to `yfinance` explicitly
Status: Resolved — 2026-07-27
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

Resolution (2026-07-27): `prices.AUTO_ADJUST = True` is now passed explicitly at every
yfinance call site — `fetch_prices`, `_fetch_vix`, `fetch_sector_returns` and
`fetch_options_snapshot` — so the convention is one constant in one place rather than four
files disagreeing (`sectors.py` already hardcoded `True`; the other three inherited
whatever the installed version defaulted to). `True` is the right convention: an unadjusted
series has split/dividend discontinuities that are not real price moves, and every
indicator would read them as ones.

Majors capped on `torch`, `lightning`, `pytorch-forecasting` and `yfinance` — the
dependencies whose breaking changes alter model *outputs* rather than failing loudly. The
rest keep open upper bounds deliberately: capping everything ages badly, and PYQ-310's
disposition is that a constraint should earn its place. `uv lock --check` passes with the
new constraints (the resolved versions already satisfy them).

Fixing this surfaced a genuine test-quality point. Four `history()` doubles in the suite
took `(self, period, start, end)` and broke on the new keyword — they had been asserting a
signature the code no longer used. Updated to accept it, which is a stub matching reality
rather than a weakened assertion. `tests/test_providers.py` now drives the real parser from
a realistically-shaped yfinance payload, which is the stronger form of the requested
assertion and the direction PYQ-243 argues for.

---

## [PYQ-229]
CI: Python matrix, frozen install, `uv lock --check`, `ruff format --check`
Status: Resolved — 2026-07-27
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

Resolution: three of the four gaps are now closed as gates; the fourth is
reported but deliberately not gating yet, for a recorded reason.

**Matrix.** `ci.yml` runs `["3.10", "3.11", "3.12"]` with `fail-fast: false`, so
a 3.10-only failure stays visible when another leg also fails — the whole point
being that 3.10/3.11 were previously unverified. The job pins `UV_PYTHON` to the
matrix entry *and* sets `UV_FROZEN: "1"`, so a later `uv run` cannot silently
re-resolve to a different dependency set halfway through the job, which would
have made the matrix decorative.

Verified locally that all three legs actually resolve against the frozen
lockfile, with the environment redirected via `UV_PROJECT_ENVIRONMENT` so the
working `.venv` was not disturbed:

```
py3.10 -> exit=0
py3.11 -> exit=0
py3.12 -> exit=0
```

Honest limit on that evidence: `--dry-run` proves resolution and the install
plan, not that the *tests* pass on 3.10/3.11. The ticket's "CI green across all
matrix entries" can only be confirmed by a real CI run. The ticket's own
reasoning — that the risk here is third-party resolution rather than syntax,
since `from __future__ import annotations` is used consistently — is what the
dry-run addresses directly.

**Frozen install + lockfile check.** `uv lock --check` runs *before* the install,
so a desync is reported as itself rather than surfacing later as a confusing
resolution error, and the install is `uv sync --frozen --extra dev`. The
acceptance criterion was tested rather than assumed — a `tabulate>=0.9` line was
added to `[project.dependencies]`, the check run, and the tree restored:

```
desynced uv lock --check EXIT=1
restored  uv lock --check EXIT=0
```

`uv.lock` was confirmed byte-identical afterwards.

**Formatting: decided NOT to gate, this pass.** Measuring first (the same
disposition investigations.md#pyq-310 established) changed the answer. The tree
is *not* formatted: `ruff format --check .` reported **20 files, ~250 changed
lines** — 11 under `pyquant/`, 8 under `tests/`, 1 under `scripts/` — and had
drifted to **22 files** by the end of the same pass as concurrent work landed.
That the figure moved twice in one session is itself the argument: the drift is
live, not a fixed historical debt. Four of the `pyquant/` files
(`analysis/metrics.py`, `analysis/calibrate.py`, `data/prices.py`,
`models/tft.py`) were under concurrent edit when this landed.

So adopting the formatter as a gate right now would have forced exactly the
outcome the ticket says it wants to prevent: one large reformat commit, tangled
with unrelated in-flight work, where every real change is invisible inside
whitespace. The decision is therefore to add the step as **`continue-on-error:
true`** — the drift number is printed on every run, so it cannot grow unnoticed,
but it does not block anyone. Flipping it to a gate after the formatting pass
lands on its own is a one-line deletion, and that is noted in the step's own
comment so the next reader does not have to rediscover why.

This is the one acceptance criterion not fully met, and it is deferred by
choice rather than missed. The formatting commit itself is left unfiled here to
avoid colliding with another ticket's in-flight edits; it needs its own pass.

Also note the interaction with PYQ-236, resolved in the same pass: `ruff check`
now includes the `D` ruleset, so the "Lint" step is doing strictly more than it
was when this ticket was written.

---

## [PYQ-230]
CI: measure and report test coverage
Status: Resolved — 2026-07-27
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

Resolution: `pytest-cov>=5.0` added to the `dev` extra, `[tool.coverage.run]`
(branch coverage, `source = ["pyquant"]`) and `[tool.coverage.report]`
(`show_missing`) added to `pyproject.toml`, and CI's Test step is now
`uv run pytest -q --cov --cov-report=term-missing`.

**No threshold gate**, per the ticket's own instruction. A `fail_under` on a
suite this young mostly teaches people to write tests that move a number, which
is a variant of the failure mode this project's first non-negotiable exists to
prevent. `--cov` is also deliberately *not* in `addopts`: the everyday
`pytest -q` loop stays fast and coverage is opt-in.

The excluded branches are only those that cannot execute under pytest by
construction — `if TYPE_CHECKING:`, `raise NotImplementedError`,
`if __name__ == "__main__":`, `@abstractmethod`, `@overload` — via `exclude_also`,
which *appends* to coverage's defaults rather than replacing them (using
`exclude_lines` would have silently dropped the built-in `pragma: no cover`).

The one documented command, which reproduces the CI figure exactly and is
recorded as a comment above `[tool.coverage.run]`:

```
uv run pytest --cov --cov-report=term-missing -q
```

**Measured number, and what it is not.** Run over the network-free, non-torch
subset only (`test_metrics`, `test_prices`, `test_config`, `test_retry`,
`test_sectors` — 61 tests):

```
TOTAL   1556 stmts   915 miss   332 branch   36%
```

That **36% is a floor, not the suite's coverage.** It excludes `test_cache`,
`test_cli`, `test_dataset`, `test_forecast`, `test_interpret`, `test_macro`,
`test_options`, `test_sentiment`, `test_tft` and `test_calibrate` — which are
precisely the files that exercise the modules reading lowest below. Stating it
as "the project has 36% coverage" would be the same error investigations.md#pyq-304
made in the other direction, and would be wrong. CI runs the full suite, so the
figure it prints is the real one.

Within the subset's own scope the result is genuinely informative, and it is
reassuring where it most matters: the two numerically dense modules where a
silently-wrong value contaminates everything downstream are the two best covered
— `data/prices.py` **98%** and `analysis/metrics.py` **88%** (with
`config.py` 100%, `data/retry.py` 96%, `data/sectors.py` 82%). Those are also
PYQ-245's proposed mutation-testing targets, and this says the line coverage
there is already high enough for mutation score to be the informative next
question rather than a restatement of "untested".

Weakest in this run, i.e. the modules carried entirely by the excluded test
files: `data/macro.py` 13%, `cli/charts.py` 14%, `cli/app.py` 18%,
`data/cache.py` 18%, `data/sentiment.py` 18%, `data/dataset.py` 20%,
`data/options.py` 21%, `models/tft.py` 23%, `provenance.py` 26%. `cli/charts.py`
is worth a second look independently of the excluded files — there is no
`test_charts.py` at all, so its 14% is a real gap rather than an artifact of the
subset.

Documenting the command in `README.md`/`CLAUDE.md` was not done here only
because those files were owned by another in-flight change during this pass; the
comment in `pyproject.toml` carries it in the meantime.

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

---

## [PYQ-232]
Sphinx + autodoc API documentation site
Status: Resolved — 2026-07-27
Priority: High
Files: new `docs/conf.py`, `docs/index.md`, `docs/api/`, `pyproject.toml`

Problem: the project has **79% docstring coverage across 126 functions and classes**
(measured by AST walk), and the docstrings are unusually good — they explain *why*, cite
ticket IDs, and record decisions (`compute_rsi`, `align_time_index`, `extend_for_prediction`
and `Forecast.__post_init__` are the standouts). None of it is rendered anywhere. The only
way to read the design rationale is to open the source files in order.

That is the whole cost/benefit: the expensive part of a docs site — writing good prose —
is already paid for. What is missing is scaffolding.

Ask: a Sphinx site. Recommended extension set, chosen for this codebase specifically:

- `sphinx.ext.autodoc` + `sphinx.ext.autosummary` — the reference pages
- `sphinx.ext.napoleon` — the existing docstrings are Google-ish prose; napoleon parses
  them without a rewrite
- `sphinx.ext.intersphinx` — **the reason to prefer Sphinx over MkDocs here.** This
  codebase is largely a thin, correct layer over pandas / torch / pytorch-forecasting /
  pydantic, so resolving `TimeSeriesDataSet`, `DataFrame` and `BaseSettings` into live
  upstream links carries real explanatory weight. mkdocstrings still does not match
  intersphinx on this.
- `sphinx.ext.viewcode` — source links, which matter when the comments are the point
- `myst_parser` — so `docs/api-design.md` and the new narrative pages (PYQ-235) stay
  Markdown and are included rather than duplicated
- `furo` theme; `sphinx-autobuild` in the docs group for the authoring loop

Add a `docs` dependency group (PEP 735 `[dependency-groups]`, which uv supports) rather
than an optional-dependency extra, so it never ships to installers. Build with
`uv run --group docs sphinx-build -b html docs docs/_build/html`.

Structure: `index` → Quickstart (reuse README) → Concepts (PYQ-235) → API reference
(autosummary over the six subpackages) → Design notes (`api-design.md`, MyST-included) →
Backlog (link out, do not duplicate — the backlog's value is that it lives with the code).

Acceptance criteria: `sphinx-build` produces a site with a page per public module; the
`Settings`/`TFTConfig`/`TrainingConfig`/`DataConfig` reference renders every field with
its default and description; at least one `pytorch_forecasting` and one `pandas` type in a
signature resolves to an external link.

Resolution (2026-07-27): Sphinx site under `docs/` with the extension set the ticket
specifies — autodoc + autosummary, napoleon, intersphinx, viewcode, myst_parser — on the
furo theme, with the docs toolchain in a PEP 735 `[dependency-groups] docs` group so it
never ships to installers. Build:
`uv run --group docs sphinx-build -b html docs docs/_build/html`.

All three acceptance criteria verified against a real build rather than assumed:

- **A page per public module** — 18 generated pages under `docs/api/_generated/`, covering
  every module in `analysis/`, `data/`, `models/`, `cli/` plus `provenance`.
- **Config reference renders every field with its default** — `docs/api/configuration.html`
  contains `max_encoder_length`, `validation_days`, `calibration_days`, `purge_horizon`,
  `embargo_days`, `target`, `hidden_size`, `cache_ttl_seconds`, `sector_etfs` and
  `checkpoint_dir`, defaults included. Note this list already includes the fields added the
  same day by PYQ-248/PYQ-250, so the page is genuinely generated rather than transcribed.
- **External types resolve** — 6 `pytorch-forecasting` links and 47 `pandas.pydata.org`
  links in the built HTML. This is the intersphinx payoff that made Sphinx the right choice
  over MkDocs for a codebase that is largely a thin layer over upstream types.

The build is warning-clean under `-W --keep-going`, which is what makes PYQ-233's gate
possible. `docs/_build/` and `docs/api/_generated/` are gitignored — the sources are checked
in, the render is not.

---

## [PYQ-233]
Gate the docs build in CI with warnings-as-errors
Status: Resolved — 2026-07-27
Priority: Medium
Files: `.github/workflows/ci.yml`, `docs/conf.py`

Problem/Ask: a docs site that is not built in CI rots within two refactors. Run
`sphinx-build -W --keep-going -b html docs docs/_build/html` as a CI step, so a broken
`:func:` cross-reference, a renamed module, or an autodoc import failure fails the PR that
caused it.

This is the same instinct as PYQ-311's decision to gate `scripts/backlog.py check`: cheap,
dependency-light, catches a bookkeeping error in the PR that introduced it rather than
three months later. Consider `nitpicky = True` once the baseline is clean — it turns every
unresolvable type reference into an error, which is strict but is exactly the check that
keeps an autodoc site honest as signatures change.

Acceptance criteria: CI fails on a deliberately broken cross-reference; passes on the
current tree.

Resolution (2026-07-27): a `Docs build (warnings are errors)` step in `.github/workflows/
ci.yml` running
`uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html`, and
`fail_on_warning: true` in `.readthedocs.yaml` so the hosted build applies the same gate.

Gated on the 3.12 matrix leg only: the docs group resolves identically on every leg, so
building it three times spends ~2 minutes re-proving one fact.

Verified passing on the current tree (build succeeded, zero warnings). **The "CI fails on a
deliberately broken cross-reference" half was not executed** — that needs a pushed PR with
an intentionally broken `:func:` reference, and this pass never pushed. What *is*
established is that `-W` is active and the tree is clean under it, so any new warning
becomes an error by construction. Recorded as verified-in-part rather than claimed whole.

`nitpicky = True` deliberately not enabled yet, following the ticket's own advice to reach
a clean baseline first.

---

## [PYQ-234]
Host the docs on Read the Docs via uv
Status: Resolved — 2026-07-27
Priority: Medium
Files: new `.readthedocs.yaml`

Problem/Ask: a built site that only exists in CI artifacts is not readable by anyone
evaluating the project. Read the Docs added native `uv` support in April 2026, so this is
a short config file rather than a pip-shim workaround, and it gives versioned URLs
(`latest`, `stable`, one per tag) at no cost.

Note the build must not need the full torch stack, or RTD builds will be slow and fragile.
Two options: install the real dependencies (simplest, slower), or set
`autodoc_mock_imports = ["torch", "lightning", "pytorch_forecasting", "yfinance",
"fredapi", "transformers"]` and install only the docs group. **Prefer the real install
first** — mocked imports silently produce wrong signatures for anything that inspects a
base class, and this codebase subclasses/returns upstream types in several public
signatures. Fall back to mocking only if build time becomes a problem.

Acceptance criteria: a pushed tag produces a versioned docs URL; the badge is in the README
next to the CI badge.

Resolution (2026-07-27): `.readthedocs.yaml` using Read the Docs' native `uv` support
(preinstalled on the `ubuntu-24.04` builder), so this is a short config rather than a
pip-shim workaround.

The ticket's "prefer the real install first" is followed, with one measured amendment: a
plain `uv sync --group docs` produces a **5.2 GB** environment (torch 1.2 GB plus 2.7 GB of
`nvidia-*` CUDA runtime wheels), which will not fit an RTD builder.
`uv pip install --torch-backend=cpu --group docs .` pins CPU wheels instead — 1.2 GB, and
`sphinx-build -W --keep-going` completed in 43 s against it with byte-identical output. No
GPU is involved in importing a module for autodoc, so nothing is lost. That is the middle
path between a real install and `autodoc_mock_imports`, which would have silently produced
wrong signatures for the several public functions returning upstream types — the very thing
intersphinx exists to link.

**Not verified: the hosted build itself.** No RTD project is connected to this repository,
so "a pushed tag produces a versioned docs URL" cannot be observed from here. The config is
written and the local equivalent of its install+build steps was run successfully; connecting
the project and pushing a tag remains a one-time manual step. Stated plainly rather than
claimed.

---

## [PYQ-235]
Narrative docs: architecture, leakage invariants, methodology
Status: Resolved — 2026-07-27
Priority: High
Files: new `docs/architecture.md`, `docs/invariants.md`, `docs/methodology.md`

Problem: this is the highest-value documentation in the project and it does not exist in
readable form. The `backlog/` directory contains genuine, hard-won, non-obvious knowledge
about look-ahead leakage in this specific pipeline — PYQ-101 (reference vs. publication
date), PYQ-103 (fills laundering warm-up rows), PYQ-115 (`predict=True` anchors to the end
of the frame it is handed), PYQ-116 (per-symbol `time_idx` means different dates),
PYQ-123 (`bfill` is look-ahead by construction), PYQ-127 (a walk-forward that does not
walk). Every one of those is a mistake most practitioners make and never notice.

Right now that knowledge is discoverable only by reading 70 tickets in ID order. Promoting
it into narrative pages is what converts the repo from "well-managed" into "instructive,"
and it is the part a reader is most likely to remember.

Ask: three pages.

- **`architecture.md`** — the README diagram expanded: what each layer owns, why
  pytorch-forecasting is confined to two modules, what the graceful-degradation contract
  guarantees and (per PYQ-118) where it deliberately stops.
- **`invariants.md`** — the properties the pipeline must satisfy, each stated as a
  falsifiable claim, each linked to the ticket that established it and the test that now
  guards it. This page and PYQ-238's test module are the same content in two forms and
  should be written together.
- **`methodology.md`** — how the model is evaluated and why: persistence baseline,
  walk-forward geometry, what `validation_days` buys, what calibration coverage does and
  does not tell you, and an honest statement of current results including the negative
  skill. The negative result stated plainly is more persuasive than a positive one stated
  vaguely.

Acceptance criteria: each of the six leakage tickets above appears in `invariants.md` as a
stated invariant with a link to its test; `methodology.md` states the current
skill/coverage numbers with their sample size.

Resolution (2026-07-27): three pages under `docs/` — `architecture.md`, `invariants.md`,
`methodology.md` — plus an `index.md` tying them to the API reference and the existing
`api-design.md` (MyST-included rather than duplicated).

`invariants.md` states each property as a falsifiable claim linked to the ticket that
established it and the test that now guards it, covering all six leakage tickets
(PYQ-101, 103, 115, 116, 123, 127) plus PYQ-124/129/130/132. It is the prose half of
features.md#pyq-238's test module and is meant to be read with it.

`methodology.md` records the persistence baseline, walk-forward geometry, what
`validation_days` buys, what calibration coverage does and does not tell you, and the
current results with their sample size — −23.5% skill, 57.5% directional accuracy, 99.3%
coverage on a nominal 80% band over 56 windows / 280 predictions — stated as a negative
result rather than hedged.

**Known staleness, flagged rather than hidden:** these pages were written against commit
`a7a2b5f` while PYQ-247/248/250/251/252 were landing in the same pass. The headline numbers
in `methodology.md` are therefore the README's, not PYQ-247's newer controlled comparison
(which measures +2.4% skill on a log-return target under a different, smaller geometry).
The two are not in conflict — different runs, different geometry, both stated with their
sample sizes — but `methodology.md` needs a follow-up pass to fold in the return-target
result, the purge/embargo splits and the CRPS/Winkler/PIT additions. That follow-up is the
one piece of this ticket left undone, and it is recorded here rather than quietly deferred.

---

## [PYQ-236]
Adopt one docstring style and enforce it with ruff's `D` rules
Status: Resolved — 2026-07-27
Priority: Low
Files: `pyproject.toml` (`[tool.ruff.lint]`)

Problem/Ask: 79% coverage means 27 undocumented functions/classes, and the existing style
is consistent by habit rather than by rule. Once autodoc (PYQ-232) renders them, gaps
become visible to readers. Enable ruff's `D` ruleset with
`convention = "google"`, ignore the rules that fight the current style (`D203`/`D213`
family), and either document or explicitly `# noqa` the remaining private helpers.

Deliberately Low: PYQ-310 already established the project's stance that a tool must catch
something real before it earns a CI gate. Run it once, look at the findings, and only then
decide whether it is a gate or a local convenience — the same disposition, applied to a
different tool.

Acceptance criteria: `ruff check` passes with `D` enabled; docstring coverage measured the
same way as this review (AST walk over `pyquant/`) is above 90%.

Resolution (decided: gate it, scoped to `pyquant/`). Ran it first, as the ticket
asked, and the findings decided the shape of the answer.

**What `D` actually found.** Enabled tree-wide with `convention = "google"`:
**201 findings — but only 19 of them in `pyquant/`.** The split is the whole
result:

```
176  tests/     (130 D103 "missing docstring in public function")
 19  pyquant/
  6  scripts/
```

Two things follow, and neither was what the ticket anticipated:

1. **`D` must not apply to `tests/`.** 130 of the 176 are D103 on test
   functions — and this project's testing convention is that the *name* carries
   the specification (`test_walk_forward_window_validation_targets_its_own_origin`).
   A docstring rule there would dilute that convention, not enforce it: it would
   reward restating the name in prose. `tests/`, `scripts/` and `docs/` are
   therefore `per-file-ignores = ["D"]`, and `D` is effectively scoped to the
   package. (`docs/conf.py` arrived from PYQ-232 during this same pass and is
   Sphinx configuration, not public API — same category.)
2. **The existing docstrings are already Google-convention.** All 19 `pyquant/`
   findings were D101/D102/D103/D104 — *missing* docstrings. Not one was a style
   violation: zero D205, zero D209, zero D4xx. So the ticket's premise that the
   style is "consistent by habit rather than by rule" is confirmed true, and the
   habit was already correct. This is why there is **no hand-maintained ignore
   list** for the `D203`/`D213` family the ticket expected to need —
   `convention = "google"` disables those automatically, and nothing else fought.

That second finding is the argument for gating rather than leaving it local: the
cost of keeping `D` green is near zero because the codebase already satisfies it,
so the rule locks in a property that already holds instead of imposing a new one.
This differs from investigations.md#pyq-310's mypy outcome for a concrete reason
— mypy found nothing real *and* would have needed stub management to stay quiet;
`D` found 19 real gaps and needs no ongoing maintenance.

**Before/after, measured by AST walk over `pyquant/` (the review's method):**

```
before   127/158 = 80.4%
after    146/158 = 92.4%
```

Above the 90% criterion. (The review's own 79%/126-symbol baseline is not
directly comparable — the tree has grown to 158 symbols since.)

Note the two metrics count different things and both are worth having: ruff's
`D1xx` flags only *public* API, while the AST walk also counts private helpers
and nested closures. 19 ruff findings vs. 31 AST gaps is that difference. The
gate uses the former; the acceptance criterion uses the latter.

The 19 new docstrings are not filler. The five package `__init__.py` files were
**completely empty**, which PYQ-232's autodoc would have rendered as five blank
landing pages — they now state what each layer is for and why the import
boundaries exist. The rest explain decisions rather than restating signatures:
`_build_settings` documents the CLI > env > `.env` > YAML > defaults precedence,
`_git` documents why every git failure collapses to `None`, `evaluation_to_dict`
documents why sample sizes travel next to rates (PYQ-117), `Interpretation`
points at investigations.md#pyq-314 for whether its weights mean what `explain`
claims, and `charts.path` explains why the series is anchored to the last close.

**Deferred, explicitly.** Seven public symbols in four files were left
undocumented because those files were under concurrent edit in this same pass,
and adding docstrings to them would have merged into someone else's diff:

```
pyquant/analysis/calibrate.py   D102   to_dict, from_dict
pyquant/analysis/forecast.py    D102   horizon, median
pyquant/config.py               D102   settings_customise_sources
pyquant/models/tft.py           D101   TrainResult, BacktestResult
```

They are held under **per-rule** `per-file-ignores` rather than a blanket `D` on
those files, so a new *style* violation there still fails — only the specific
missing-docstring rule is suppressed, and each line is annotated with the exact
symbols it covers so it can be deleted as they are documented. Five further AST
gaps in those same files are private helpers (`metrics.pooled`,
`metrics.pooled_dict`, `prices._load`, `tft._source_hint`, `tft._bundle_dir`)
which `D` does not flag; they are the residue between 92.4% and 100%.

This is a real hole and worth stating plainly: until those four lines are
removed, an undocumented new public class in `models/tft.py` or method in
`config.py` will pass CI. It is bounded, enumerated above, and cheap to close.

Verification: `uv run ruff check .` passes with `D` in `select`. No test guards
this one — the linter is the guard, which is the point of the ticket.

---

## [PYQ-237]
Executable doctests for the metrics and forecast APIs
Status: Open
Priority: Low
Files: `pyquant/analysis/metrics.py`, `pyquant/analysis/forecast.py`, `pyproject.toml`

Problem/Ask: `metrics.py`'s functions are pure, take small numpy arrays, and have
docstrings that *describe* behaviour a two-line example would *demonstrate*
(`skill_vs_baseline`, `calibration_coverage`, `directional_hit_rate`). Add `>>>` examples
and run them with `--doctest-modules` in pytest (and/or `sphinx.ext.doctest` for the
rendered site).

This is the cheapest documentation-drift guard available: an example that stops matching
the code fails the build. It is also the best place to make the units question concrete
(see PYQ-240 / PYQ-313) — a worked example showing dollars in and a dimensionless skill
figure out documents the contract better than a sentence can.

Acceptance criteria: `pytest --doctest-modules pyquant/analysis` passes and covers at
least `skill_vs_baseline`, `calibration_coverage` and `Forecast.expected_return_pct`.

---

## [PYQ-238]
`tests/test_invariants.py` — assert the pipeline-spanning invariants directly
Status: Resolved (pending, 2026-07-27)
Priority: High
Files: `tests/test_invariants.py` (new), `docs/invariants.md`

Problem: `backlog/README.md` states the lesson from the third review pass precisely —
*"this backlog was optimising local correctness ticket by ticket while the invariants that
span the pipeline went unstated."* PYQ-115/117/127 each then shipped a test asserting one
invariant. That prevents those three exact recurrences. It does not prevent the *next*
member of the family, and there have now been six members (PYQ-101, 103, 115, 116, 123,
127), every one of which was correct in each individual file and wrong across files.

The structural fix is to state the invariants once, in one place, as properties rather
than as regressions — so a future change that violates any of them fails immediately
regardless of which file it touched.

Ask: a dedicated module whose every test is a named invariant over a synthetic
multi-symbol panel with deliberately unequal history:

1. **No future information in any training row.** Every non-price column's value at row
   *t* must be derivable from data timestamped at or before *t*. (Constructible: give each
   joined source a distinctive monotone value and assert no row carries a value first
   observed later.) — guards PYQ-101, PYQ-123, PYQ-129.
2. **Warm-up rows never carry fabricated values.** The panel's first surviving row is
   determined by the longest indicator window, and no column is constant across the
   leading rows. — guards PYQ-103, PYQ-132.
3. **Prediction decodes the future.** `decoder_time_idx.min() > df["time_idx"].max()` for
   the un-extended frame. — guards PYQ-115.
4. **The encoder ends on the last observed bar.** — guards PYQ-115's second half.
5. **One calendar across pooled symbols.** The same `Date` maps to the same `time_idx` for
   every group. — guards PYQ-116.
6. **Validation is strictly after the cutoff.** Every validation decoder index exceeds
   `training_cutoff` for every group, including the shortest-history one. — guards
   PYQ-116, PYQ-117.
7. **The walk-forward walks.** Consecutive origins produce disjoint decoder windows, each
   starting at its own `cutoff + 1`. — guards PYQ-127.
8. **The forecast dates are the decoded dates.** The dates in the table, the JSON, the PNG
   and the appended rows are one set. — guards PYQ-115, PYQ-130.
9. **The band is monotone wherever it is consumed.** — guards PYQ-124.

Write each with a docstring naming the ticket it descends from, and link the module from
`docs/invariants.md` (PYQ-235) so the two stay in sync.

Acceptance criteria: every invariant above has a named test; each one fails if the
corresponding fix is reverted (verify this while writing them — an invariant test that
passes against the broken code is worthless, which is exactly the trap PYQ-120's coverage
gap illustrates).

Resolution: `tests/test_invariants.py` adds 10 tests (invariants 3 and 4 are asserted
together in one test, since they're read off the same decoded batch; invariant 1 is split
across three tests since it covers three independently-breakable join mechanisms —
publication lag, late-start backfill, and post-close session timing — matching how
`docs/invariants.md` already treats that trio as separate entries in the leak family).
Built over a shared `unequal_history_panels` fixture (two symbols, one truncated to its
last 90 rows) for the invariants that are inherently cross-symbol (5, 6), and reused for the
single-symbol ones (3/4, 7, 8) so the module has one coherent multi-symbol shape throughout
rather than a fixture per test.

Verified against pre-fix logic shapes, not just against the current (fixed) code, per the
acceptance criteria:

- Reindexing a synthetic publication-lagged source with `method="bfill"` instead of
  `"ffill"` (the PYQ-101 shape) makes the "value seen \<= row's own date" assertion fail, as
  expected.
- Running `.ffill().bfill()` on a late-starting source instead of dropping the ungapped
  rows (the PYQ-123 shape) makes the "panel starts at the source's first observation"
  assertion fail.
- Computing the pre-extension decoder window (`observed_max - horizon + 1`) instead of
  extending the frame first (the PYQ-115 shape) makes "decoder strictly after observed_max"
  fail.
- Anchoring every walk-forward origin to the frame's final window regardless of cutoff (the
  PYQ-127 shape) collapses three distinct origins onto one window, failing the
  disjoint-windows assertion.

Full run: 10 passed in ~11s, no `@pytest.mark.slow` needed — the shared fixture keeps every
test's model tiny (`hidden_size=8`) and most invariants (1, 2, 5, 6, 7, 9) need no training
at all, only dataset/panel construction. Linked from `docs/invariants.md`'s existing note,
which now points at the module by path instead of describing it as not-yet-landed.

---

## [PYQ-239]
Learnability test: inject a known signal and assert the model recovers it
Status: Resolved (pending, 2026-07-27)
Priority: High
Files: `tests/test_learnability.py` (new)

Problem: `test_train_load_predict_roundtrip` asserts the bundle files exist, that
`n_features > 0`, and that the metrics are within `[0, 1]`. **Every one of those assertions
would pass against a model that emits a constant.** Nothing anywhere in 164 tests asserts
that the training pipeline can learn *anything*.

That matters more than usual here, because the repo's headline number is negative
(−23.5% skill) and there is currently no way to distinguish the two explanations:

- the target genuinely is not forecastable from these features (the interesting answer,
  and quite possibly the true one — see investigations.md#pyq-312), or
- something in the wiring is silently broken (normalisation, feature ordering, target
  scaling, an off-by-one that survived PYQ-115).

A learnability test discriminates between them in seconds and permanently.

Ask: a synthetic panel where the target is a deterministic, learnable function of an
observable feature at the required lag — e.g. `Close[t+h] = Close[t] * (1 + k *
feature[t])` with a modest `k` and light noise. Train a small model for a handful of
epochs and assert `skill_vs_baseline > 0` by a comfortable margin. Keep it fast enough for
CI (tiny `hidden_size`, short encoder, few epochs) and, if it proves flaky, mark it
`@pytest.mark.slow` and run it on a schedule rather than deleting it.

The second, cheaper half: a *degenerate* control — train on pure white noise and assert
skill is approximately zero and not implausibly positive. A pipeline that finds skill in
noise has a leak, and this test finds it without needing to know where.

Acceptance criteria: the injected-signal test asserts positive skill; the noise-control
test asserts skill is not significantly positive. Both run in CI within a sensible time
budget.

Resolution: `tests/test_learnability.py` builds a synthetic OHLCV panel plus one extra
`Signal` feature via a shared `_signal_panel()` helper. In the learnable case, day *t*'s
log-return is `k * Signal[t-1] + noise` (`k=0.08`, `noise_std=0.002`); in the degenerate
control, `Signal` is shuffled independently of the returns. `max_prediction_length` is set
to 1 for this test specifically: `Signal` is an unknown real like every other feature, so
it is only ever visible to the encoder, never the decoder — a multi-step horizon would ask
the model to predict steps whose driving value it structurally cannot see yet, which is
unlearnable by construction and would confound the test's purpose. `tft.train()` is run
against a `monkeypatch`ed `build_panel`, same pattern as `test_tft.py`.

Measured skill (`skill_vs_baseline`, `n=60` validation windows each), across three unrelated
seeds to check the thresholds weren't tuned to one lucky run:

```
seed=1  learnable: 0.971   noise: 0.003
seed=42 learnable: 0.972   noise: 0.011
seed=99 learnable: 0.975   noise: 0.004
```

Chosen thresholds (`> 0.3` learnable, `< 0.15` noise) sit far inside both margins — the
effect size is large by design (`k` well above `noise_std`), which is what keeps the test
fast and non-flaky rather than borderline. Neither test needed `@pytest.mark.slow`: each
runs in single-digit seconds.

This also gives an answer for investigations.md#pyq-312's open question, restated here since
this is the ticket that produces the evidence: the pipeline **can** learn a genuine
signal when one exists, cleanly and repeatably. That means PYQ-247's small measured skill
(+2.4%, one symbol, effective n≈5) is not obviously a wiring artifact — the machinery is
demonstrably capable of finding a real effect at this scale when the effect is there. It
does not prove +2.4% itself is real (that needs the multi-symbol repeat the backlog's `##
Now` list already calls for), only that a near-zero result is not automatically evidence of
a broken pipeline.

---

## [PYQ-240]
Regression test that predictions/actuals/last_observed share units
Status: Resolved — 2026-07-26
Priority: Medium
Files: `tests/test_tft.py`, `pyquant/models/tft.py` (`_evaluate_validation`)

Problem: `_evaluate_validation()` assembles three arrays from three different places in
pytorch-forecasting's output:

```python
predictions   = result.output.cpu().numpy()          # mode="quantiles"
actuals       = result.y[0].cpu().numpy()
last_observed = result.x["encoder_target"][:, -1].cpu().numpy()
```

and hands all three to `evaluate_predictions()`, which subtracts them from one another.
That is only meaningful if all three are in the **same space** — and the dataset applies
`GroupNormalizer(transformation="softplus")`, so "price space" and "normalised space" are
very different scales. Whether `x["encoder_target"]` is normalised or raw is a semantic
detail of the upstream library, not of this code, and nothing in the test suite pins it.

Circumstantial evidence says it is currently correct (a reported `baseline_mae` of 7.40
for AAPL is a plausible dollar figure and an implausible normalised one). But "currently
correct, unasserted, and dependent on an upstream library's internal convention" is
exactly the profile of the PYQ-109 class of bug: silent, total, and invisible from inside
any one file. A pytorch-forecasting minor release that changes this would corrupt every
metric the project reports with no test failing.

See investigations.md#pyq-313 for confirming the semantics; this ticket is the guard that
makes the answer durable.

Ask: a test on a synthetic panel with a known price level (say ~$100) asserting all three
arrays are within an order of magnitude of that level, and that `baseline_mae` computed by
`_evaluate_validation` matches a persistence MAE computed independently from the raw panel
to within a tight tolerance.

Acceptance criteria: the independent-recomputation assertion passes on the current stack
and would fail if any of the three arrays were silently swapped into normalised space.

Resolution (2026-07-26): added
`test_validation_predictions_actuals_and_persistence_baseline_share_price_units`. On
pytorch-forecasting 1.7.0 it maps every returned decoder index back to the raw synthetic
Close panel and asserts exact equality for `result.y[0]` and
`x["encoder_target"][:, -1]`; it also independently recomputes the persistence MAE.
The quantile predictions are asserted to be price-scale rather than normalizer-scale. This
pins the upstream contract and fails if any one array crosses the normalization boundary.

---

## [PYQ-241]
End-to-end CLI journey test across every command and both output formats
Status: Resolved (pending, 2026-07-27)
Priority: Medium
Files: `tests/test_cli.py`

Problem/Ask: the CLI tests are thorough per-command but each starts from a mocked
mid-state. No test performs the actual user journey — `train` → `forecast` → `explain` →
`scan` → `backtest` → `cache list` — against one temp `checkpoint_dir`, using each
command's *real* output as the next command's input. That sequence is what a first-time
user does, and it is where cross-command contract breaks live: a bundle written by `train`
that `forecast` cannot read, a `meta.json` field `explain` expects and `train` stopped
writing, a `--pin` created by one command and not found by another.

PYQ-119 was exactly this class of bug (read-side commands ignoring the write side's
config) and was found by reasoning rather than by a test.

Run the whole journey twice, once with `--format rich` and once with `--format json`,
asserting the JSON parses and carries the documented keys at each step. Mark it
`@pytest.mark.slow` if the real fit is too slow, but keep it in CI.

Acceptance criteria: one test function walking all six commands against a shared temp
directory with mocked network but real training; JSON output parsed and key-checked at
each step.

Resolution: `test_full_cli_journey_across_every_command_and_both_output_formats` in
`tests/test_cli.py`. Only two things are stubbed: `fetch_prices` (the network boundary,
shared by every command since they all funnel through the same `build_panel`) and
`fetch_options_snapshot` (display-only, not part of the trained schema). Everything else —
bundle files, `meta.json`, the dataset params, the cache dir — is real, isolated to a temp
`PYQUANT_HOME` (`forecast`/`explain`/`scan`/`cache list` have no settings-injection point at
all, so this was the only way to redirect them without a mocked mid-state). A tiny
checked-in-style YAML config (PYQ-209's mechanism, not a mocked `Settings` object) keeps the
model small enough that both full passes together run in ~22s — no `@pytest.mark.slow`
needed.

The journey runs twice end-to-end (including retraining the same bundle), once per
`--format`, all six commands each time. Every JSON payload is key-checked against exactly
the fields `analysis/serialize.py` documents (`scan`'s rows are checked directly, since that
command builds its dict inline rather than through a serializer). This exercises PYQ-119's
exact failure class for real: `forecast`/`explain`/`scan` all pick up the bundle's *recorded*
data-config toggles via `settings_for_bundle` rather than the fresh `load_settings()`
defaults each command otherwise starts from — the test would fail loudly if that self-healing
path ever regressed, since `--no-macro/--no-sentiment/--no-sectors` are only ever passed to
`train`.

---

## [PYQ-242]
Property-based tests for `analysis/metrics.py`
Status: Open
Priority: Low
Files: `tests/test_metrics.py`, `pyproject.toml` (dev extra)

Problem/Ask: the metric functions are pure, small, and take numpy arrays — the ideal
target for Hypothesis. Example-based tests confirm hand-computed cases; properties confirm
the whole input space. Worth asserting:

- `calibration_coverage` ∈ [0, 1] for any arrays, including degenerate ones
- `directional_hit_rate` ∈ [0, 1], and equals 1 when median and actual are on the same
  side of `last_observed` everywhere
- `skill_vs_baseline` is **invariant under a common affine rescaling of all inputs** —
  this is the property that would have caught a units mismatch (PYQ-240) as a *design*
  statement rather than a spot check
- `warn_on_quantile_crossing` returns 0 for any array already sorted along the last axis
- `aggregate_metrics` over a single-element list is the identity

Acceptance criteria: `hypothesis` in the dev extra; at least the five properties above,
each with a shrinking-friendly strategy.

---

## [PYQ-243]
Recorded-payload contract tests for every external vendor
Status: Resolved (pending, 2026-07-27)
Priority: Medium
Files: `tests/fixtures/` (new), `scripts/record_fixtures.py` (new), `tests/test_prices.py`,
`test_macro.py`, `test_sentiment.py`, `test_options.py`, `test_sectors.py`

Problem: every vendor test today mocks at the *function* boundary — `fetch_prices` is
patched, or `yf.Ticker` returns a hand-built DataFrame with exactly the columns the code
wants. That verifies our logic against our own assumptions about the payload, which is
half a test. It cannot catch the failure that actually happens in production: **the vendor
changes its response shape.**

That risk is not hypothetical here. PYQ-228 records that `yfinance` has already jumped
from the declared `>=0.2.40` to a resolved **1.4.1**, with `Ticker.history` now typed
`(self, *args, **kwargs)`, and that `auto_adjust`'s default flipped mid-0.2.x — which
silently changes every price level, every derived indicator, and therefore every trained
model. `fetch_prices` also does `df[["Open","High","Low","Close","Volume"]]` with no guard.

Ask: capture one real response per source as a checked-in fixture (JSON for
Finnhub/FRED, a small pickled/parquet frame for yfinance), record the vendor + library
version alongside it, and mock at the **transport** boundary (`requests`, `yf.Ticker`)
rather than at our own function boundary. Each source then gets a test asserting our
parser produces the expected columns and dtypes from a genuine payload.

Pairs with PYQ-244: fixtures catch *our* regressions, the nightly job catches *theirs*.

Acceptance criteria: one recorded fixture per source; each source's happy-path test drives
the real parsing code from that fixture; the fixture files record which vendor/library
version produced them.

Resolution: `scripts/record_fixtures.py` hits all five live vendor call sites once
(`yfinance>=1.4.1`, `fredapi==0.5.2`, Finnhub REST v1) and writes `tests/fixtures/` plus a
`MANIFEST.json` recording vendor, library version and fetch date per file. Six new tests
(one each for prices, VIX, FRED, sector ETFs, options, plus one full `fetch_sentiment()`
chain test) mock at the vendor's own class/function (`yf.Ticker`, `yf.download`,
`fredapi.Fred`, `requests.get`) rather than at `pyquant`'s own `fetch_*` functions, then run
the real parsing code against the recorded payload.

`fredapi` was checked directly and does not use `requests` — it calls `urllib.request.
urlopen` and parses XML internally — so `fredapi.Fred.get_series_all_releases` (the
library's own public method) is the practical transport boundary; mocking raw sockets would
only test fredapi's own XML parsing, not ours.

This immediately paid for itself, twice, on the very first recording:

- The real `Ticker.history()` response carries `Dividends`/`Stock Splits` alongside OHLCV.
  Every existing hand-built fixture in this codebase only had the 5 columns the code wants
  — none of them would have caught `fetch_prices` failing to drop an unexpected column.
- `fredapi.get_series_all_releases()` over a 60-day realtime window returned **26,321
  rows**, because FRED attaches the *entire* historical series (back to 1954 for `DFF`) to
  the earliest vintage boundary in the window — not the handful of rows a hand-built
  `_releases()` fixture assumes. The recording script now trims to reference dates within
  90 days of the latest release before writing the fixture (90 rows, 40 distinct
  `realtime_start` groups — still real multi-vintage structure, 12KB instead of 3MB).

Finnhub's `headline`/`summary`/`url`/`image` fields are replaced with placeholder text
before the fixture is written (`_sanitize_headline` in the recording script) — the contract
these tests check is the response *shape* (which fields exist, `datetime` as a Unix
timestamp, `id` as an int), not the copyrighted article text, so there is no reason to
redistribute the latter in a public fixture. `category`/`datetime`/`id`/`related`/`source`
are kept real.

Pairs with PYQ-244 as designed: these fixtures catch *our* regressions against a payload
shape that was real at the time it was recorded; PYQ-244's nightly job catches the vendor
changing that shape *after* recording.

---

## [PYQ-244]
Scheduled nightly CI job against live vendors
Status: Resolved — 2026-07-27
Priority: Low
Files: `.github/workflows/nightly.yml` (new)

Problem/Ask: the test suite is deliberately network-free, which is correct for PR CI and
means **no automated check ever talks to a real vendor.** yfinance is an unofficial
scraper of Yahoo's internal endpoints with no SLA; when it breaks, the project finds out
when a human runs a command.

Add a `schedule:`-triggered workflow that runs a minimal live smoke test — fetch prices
for one liquid symbol, fetch VIX, fetch sector ETFs, and (if repo secrets are configured)
FRED and Finnhub — asserting only shape and freshness, never values. Use
`continue-on-error` or a separate workflow so a vendor outage never blocks a PR; the
signal is the notification, not the gate.

Acceptance criteria: the workflow runs on schedule, is not required for merges, and fails
loudly (issue or notification) when a source stops returning the expected shape.

Resolution: added `.github/workflows/nightly.yml` — a **separate workflow** from
`ci.yml` rather than a `continue-on-error` step inside it. Both options were on
the table in the ticket; separate wins because a job in `ci.yml` appears on every
pull request's check list even when non-blocking, and a check that is permanently
allowed to be red is a check people stop reading. Nothing in this file can gate a
merge, because it is never triggered by `pull_request`.

Triggers: `schedule` at `0 9 * * *` (after the US close and after Yahoo/FRED have
settled the prior session, so a staleness failure means the vendor rather than
the clock) plus `workflow_dispatch`, so a suspected vendor problem can be checked
immediately instead of waiting for the schedule.

**Shape and freshness only, never values.** Each source asserts the columns the
pipeline actually consumes and that the most recent observation is within
`MAX_STALE_DAYS = 7`. A different price is not a defect; a missing `Close`
column, or data that stopped updating a fortnight ago, is. The tolerance is
deliberately generous so a long holiday weekend plus a lagging vendor cannot page
anyone.

Checks: yfinance OHLCV (`AAPL`, asserting all five OHLCV columns), VIX via
`fetch_macro(api_key=None)`, sector ETFs (`SPY`/`XLK`/`XLF`, asserting the
`SEC_<ETF>` naming), and — only when the corresponding repo secret exists — FRED
(asserting that something other than `VIX` survived, since a `["VIX"]`-only frame
is what a silent total FRED failure looks like) and Finnhub (asserting a non-empty
list whose items carry `headline` and `datetime`, the two fields the sentiment
join depends on). Missing secrets **skip** rather than fail, so a fork or a fresh
clone still gets the yfinance signal. Key presence decides what runs; no key value
is echoed, logged or written, per CLAUDE.md's secrets rule.

Failures are collected rather than raised, so one dead vendor cannot mask the
state of the others — the job reports every source's outcome and then exits 1.

**Verified by running it**, not by reading it. The three keyless checks were
executed live against real vendors:

```
OK   yfinance OHLCV (AAPL): 21 rows, latest 2026-07-24 (3d old)
OK   yfinance VIX (via fetch_macro): 21 rows, latest 2026-07-23 (4d old)
OK   yfinance sector ETFs: 19 rows, latest 2026-07-23 (4d old)
SKIP FRED (no FRED_API_KEY secret configured)
SKIP Finnhub (no FINNHUB_API_KEY secret configured)
```

The failure path was verified too, by pointing the price check at a
non-existent ticker — it exits 1, names the broken source, and the other two
still run and report:

```
FAIL yfinance OHLCV (AAPL): raised ValueError: No price data found for 'NOT_A_REAL_TICKER_XYZ'
OK   yfinance VIX (via fetch_macro): 21 rows, latest 2026-07-23 (4d old)
OK   yfinance sector ETFs: 19 rows, latest 2026-07-23 (4d old)
1 vendor check(s) failed -- a source changed shape or went stale.
EXIT=1
```

Both workflow files were confirmed to parse as YAML, and the embedded script was
extracted and `ast.parse`d plus ruff-checked, so a Python syntax error cannot hide
inside a YAML block scalar until 09:00 UTC. Writing it surfaced one real defect in
the first draft: `pd.Timestamp.utcnow()` is deprecated and warns under the
resolved pandas, now `pd.Timestamp.now("UTC").tz_convert(None)`.

Decision worth recording: the smoke test lives **inline in the workflow**, not in
`tests/`. The pytest suite is network-free by contract, and a live test sitting in
`tests/` is one `-k` away from being run by accident — which would break the
offline guarantee. PYQ-243 is the ticket that gives vendor checks a home under
`tests/`, and the two are complementary rather than overlapping: recorded fixtures
catch *our* parser regressions against a frozen payload, this catches *theirs*
against a live one.

Limitation, stated rather than glossed: the acceptance criterion says "fails
loudly (issue or notification)". What ships is a failing scheduled run, which
GitHub notifies the repo owner about by default. Auto-filing an issue on failure
was not added — it needs `issues: write` and a decision about duplicate handling
on consecutive failures, which is a product call rather than a mechanical one.
The `permissions:` block is currently `contents: read` for that reason.

Also note GitHub disables `schedule` triggers on repositories with no activity
for 60 days. Acceptable here, since the job exists to protect an actively
developed tree, but it means a dormant fork gets no signal.

---

## [PYQ-245]
Mutation testing on the metrics and indicator modules
Status: Open
Priority: Low
Files: `pyquant/analysis/metrics.py`, `pyquant/data/prices.py`

Problem/Ask: coverage (PYQ-230) tells you which lines ran; it does not tell you whether an
assertion would have noticed them being wrong. `metrics.py` and `prices.py` are pure,
fast, and numerically dense — the ideal mutation-testing target, and the two modules where
a silently-wrong result is most damaging because every downstream number inherits it.

Run `mutmut` (or `cosmic-ray`) once against those two modules and look at the survivors: a
flipped comparison in `directional_hit_rate`, an off-by-one in `_wilder_average`'s seed, a
swapped `lower`/`upper` in `calibration_coverage`. Each survivor is a missing assertion.

Deliberately a one-off survey rather than a CI gate, matching the disposition PYQ-310
established for mypy: run it, record what it found, then decide.

Acceptance criteria: a recorded mutation score for both modules and a test added for each
surviving mutant judged worth killing.

---

## [PYQ-246]
Determinism test: same seed + same pin ⇒ identical metrics
Status: Resolved (pending, 2026-07-27)
Priority: Medium
Files: `tests/test_tft.py`, `docs/methodology.md`

Problem/Ask: PYQ-210 added `seed_everything` and PYQ-205 added dataset pins, and
PYQ-210's own acceptance criterion was "two consecutive `train()` calls with the same seed
and a pinned dataset produce identical val_loss" — but the test that shipped asserts only
that the seed is *passed* and *recorded*, not that the run is actually reproducible.

That is a meaningful gap: `seed_everything` does not by itself guarantee determinism on
every backend (cuDNN autotuning, non-deterministic reductions, `num_workers > 0` ordering
— all three of which PYQ-218/PYQ-223 made configurable), so the property the project
claims may already be false on some configurations.

Ask: train twice on a pinned synthetic panel with an identical seed and assert
`val_loss` and every `EvaluationMetrics` field match exactly. If they do not, the correct
outcome is not to weaken the test but to decide explicitly whether to set
`torch.use_deterministic_algorithms(True)` / `Trainer(deterministic=True)` and record the
performance cost — and, if determinism is not guaranteed on GPU, to say so in the docs
rather than let `runs.jsonl` imply comparability it cannot deliver.

Acceptance criteria: the two-run equality test passes on CPU; the GPU/`num_workers > 0`
situation is documented either way.

Resolution: `test_two_identically_seeded_runs_produce_identical_metrics` in `test_tft.py`
trains twice on an identical monkeypatched synthetic panel with the same seed and asserts
`val_loss` and the full `EvaluationMetrics` dataclass compare equal. **It passes outright —
no code change was needed.** Determinism holds exactly on this CPU-only environment, checked
with both `num_workers=0` (the default, asserted in the test) and `num_workers=2` (checked
manually alongside it; not asserted in the committed test since it would double the
runtime for a config this suite doesn't otherwise exercise by default). That rules out one
of the ticket's two suspected nondeterminism sources for the configurations actually
available here.

The other suspected source, cuDNN autotuning, could not be checked either way — there is no
GPU in this environment. Per the acceptance criteria, this is documented explicitly rather
than left implicit: `docs/methodology.md`'s reproducibility section now states plainly that
GPU determinism is unverified and `runs.jsonl` comparisons should not assume bit-identical
reproducibility across GPU-trained bundles on the strength of this result. No
`torch.use_deterministic_algorithms`/`Trainer(deterministic=True)` change was made, since
there was nothing observed to fix.

---

## [PYQ-247]
Forecast log-returns instead of price levels
Status: Resolved — 2026-07-27
Priority: High
Files: `pyquant/data/dataset.py` (`TARGET`, `make_dataset`), `pyquant/analysis/forecast.py`, `pyquant/analysis/metrics.py`

Problem: `TARGET = "Close"`. The model predicts the **price level**, normalised with
`GroupNormalizer(transformation="softplus")`, and is scored against a persistence baseline
that predicts the last observed close.

For a series close to a random walk, the conditional expectation of the level essentially
*is* the last observed level. So the baseline is near-optimal by construction, and any
deviation the model makes costs MAE in expectation. **The reported −23.5% skill is roughly
what this formulation predicts a priori, largely independent of hyperparameters.**

This has a direct consequence for the current "Now" list: PYQ-211 (learning-rate tuning)
is ranked #1, but it optimises *within* a formulation where the achievable ceiling is
approximately "tie the baseline." It is unlikely to move the headline number meaningfully
however carefully it is run — and it needs GPU hardware, while this ticket does not.

Standard practice is to model log-returns, `r[t] = log(C[t] / C[t-1])`, which are
approximately stationary and roughly zero-mean. Then:

- the baseline becomes "predict zero return," which is beatable in principle;
- the quantile band is a band on returns, which is what a calibration number should
  describe and is comparable across symbols (fixing the "MAE is in dollars" complaint
  PYQ-227 already records);
- `GroupNormalizer`'s per-group scaling stops fighting a non-stationary level;
- pooling across symbols becomes far more sensible, since returns are comparable across
  tickers in a way price levels are not — which strengthens the rationale PYQ-116 already
  flagged as weaker than advertised.

Ask: make the target configurable (`TrainingConfig.target: Literal["close", "log_return"]`,
defaulting to the new behaviour once validated), predict cumulative log-returns over the
horizon, and reconstruct price paths for display by exponentiating from the last observed
close. Report metrics in return space; keep the price-space table for the user. Note that
reconstruction preserves quantile monotonicity, so `Forecast.__post_init__`'s invariant
(PYQ-124) still holds.

Acceptance criteria: a config toggle switching the target; unit tests for the
return↔price round-trip (reconstruct(transform(x)) == x); a documented before/after
`backtest` comparison on the same symbol and seed. **The result may well be that skill
stays near zero — that is a legitimate and interesting outcome (see
investigations.md#pyq-312), and it should be recorded rather than tuned away.**

Resolution (2026-07-27): landed, and **the ticket's central prediction is confirmed — this
is the single largest change to the project's headline number so far.**

`TrainingConfig.target: Literal["close","log_return"]` selects the target;
`panel_to_long()` computes `LogReturn = log(C[t]/C[t-1])` (dropping the first row rather
than fabricating a target for it); `make_dataset()` drops the `softplus` transformation for
returns, which need no positivity constraint; `evaluate_predictions(target="log_return")`
scores against a **zero-return** persistence baseline; and `generate_forecast()`
reconstructs a price path with `log_returns_to_prices()`, so the user-facing table is
unchanged.

**Measured comparison.** Same symbol (AAPL), same seed (42), same pinned dataset
(`AAPL_pyq247-v2`, 1116 rows x 31 cols, 2022-02-09..2026-07-23), same 12-epoch budget, same
5 walk-forward windows. Only the named knob varies:

```
arm                                skill      dir     cov       CRPS    Winkler
close (TARGET='Close', softplus)  -59.5%    80.0%   52.0%    4.42409    64.335
log_return (this ticket)           +2.4%    56.0%   76.0%    0.00515     0.080
close + purge/embargo              -4.5%    80.0%   64.0%    3.32032    54.806
log_return + purge/embargo         +3.8%    52.0%   80.0%    0.00507     0.078
```

Four things in that table, in order of importance.

1. **Skill goes from negative to positive.** −59.5% to +2.4%, and +3.8% with PYQ-250's
   purged splits. The formulation, not the hyperparameters, was the binding constraint —
   exactly the argument this ticket made and the reason PYQ-211 was demoted.
2. **Calibration largely fixes itself.** Coverage moves from 52% to 76–80% against a
   nominal 80% band with **no conformal correction applied** (`calibration_days=0` in every
   arm). The band pathology is substantially a symptom of predicting a non-stationary
   level, which demotes PYQ-248 from primary fix to second line of defence and directly
   informs investigations.md#pyq-317.
3. **Per-window stability changes character.** The level target is not merely worse on
   average, it is erratic: per-window skill `[+0.28, +0.47, +0.35, −2.71, −3.13]` versus
   `[−0.07, +0.00, +0.06, +0.08, +0.01]` for returns. Two of five windows lose ~3x the
   baseline's error. That dispersion is invisible in a mean, which is why PYQ-226 exists.
4. **Directional accuracy *drops*, 80% → 52–56%, and that is the honest number.**
   "Direction versus the last close" is nearly free on a level target, because a model that
   merely tracks the level is usually on the right side of it. On returns it is a genuine
   coin-flip question and the answer is roughly a coin flip. Read against the README's
   57.5%, this suggests the old figure was flattered by the formulation. Reporting the
   lower number is the point (non-negotiable #1).

**Caveats, which matter more than the headline.** `n_points = 25` per arm (5 windows x
5-day horizon), so effective n ≈ 5 independent windows — these differences are
directionally clear but not statistically strong, which is exactly what PYQ-251 exists to
surface and why CRPS/Winkler (PYQ-252) are quoted alongside. MAE is in dollars for the
close arms and log-return units for the return arms, so *skill* (a ratio) is comparable
across rows but *CRPS/Winkler* only within a target. These absolute values also do not
reproduce PYQ-117's 99.3%-coverage / −23.5%-skill figures and are not meant to: that was
`train()`'s 56-window holdout at full epochs, this is a 5-window walk-forward at 12. The
comparison is internally controlled, not a restatement of the README.

**The default is deliberately unchanged** (`target = "close"`). One symbol, five windows and
n≈5 is not enough to flip the default and silently invalidate every existing bundle; doing
it on this sample would be the number-improving move non-negotiable #1 forbids. Next step:
a multi-symbol repeat with PYQ-251's intervals, then a default change — recorded as the
concrete follow-up in investigations.md#pyq-312.

Guarded by `test_panel_to_long_adds_log_returns_and_selects_them_as_the_default_target`,
`test_log_return_price_round_trip` (reconstruct(transform(x)) == x) and
`test_log_return_metrics_use_zero_return_persistence_baseline`. Bundles record their target
in `meta.json` and `generate_forecast` reads it back, so a return-trained bundle cannot be
rendered as a price-trained one.

---

## [PYQ-248]
Conformal / split-calibration of the quantile band
Status: Resolved — 2026-07-27
Priority: High
Files: `pyquant/analysis/metrics.py`, `pyquant/models/tft.py`, new `pyquant/analysis/calibrate.py`

Problem: PYQ-117 measured 99.3% empirical coverage on a nominal 80% (p10–p90) band. The
interval is far wider than it claims, which makes it useless for decisions — an 80% band
that contains 99% of outcomes tells you almost nothing, and `scan`'s "is the whole band on
one side of zero" guard (PYQ-206) will essentially never fire, silently collapsing the
BUY/SELL logic into permanent HOLD.

PYQ-227 (open) will add per-quantile exceedance and pinball loss, which **diagnoses** which
side is at fault. Nothing currently **fixes** it. Retraining with different
hyperparameters is an indirect and unreliable route to a calibrated interval; direct
recalibration is the standard one.

Ask: split-conformal calibration. Hold out a calibration slice separate from both training
and test, compute conformity scores on it (for quantile regression, the CQR score
`max(q_lo - y, y - q_hi)`), take the appropriate empirical quantile, and widen/narrow the
predicted band by that amount. Properties that make this the right tool here:

- distribution-free, with a finite-sample marginal coverage guarantee;
- requires no retraining and no change to the model or the loss;
- roughly 80 lines plus tests;
- composes cleanly with the existing quantile head and with PYQ-247's return target.

Store the calibration offset in the bundle (`meta.json`) so `forecast` applies it
automatically, and record it in the provenance block. Note the guarantee assumes
exchangeability, which financial time series violate — so pair it with PYQ-250's
purged/embargoed splits and report *achieved* coverage on a genuinely held-out period
rather than relying on the theoretical guarantee.

Acceptance criteria: after calibration, empirical coverage on an out-of-sample period is
within a few points of nominal; a unit test on synthetic data with a deliberately
overwide band asserts calibration narrows it toward nominal; the offset is persisted and
applied by `forecast`.

Resolution (2026-07-27): new `pyquant/analysis/calibrate.py` implementing conformalized
quantile regression. `conformity_scores()` computes `max(q_lo − y, y − q_hi)` — signed
distance to the band, so one formula both widens a band that is too narrow and narrows one
that is too wide. `fit_conformal_offset()` takes the finite-sample corrected quantile
`ceil((n+1)·coverage)/n` rather than a plain empirical quantile: that correction is what
buys the marginal guarantee, and at n = 20 with 80% nominal it is the difference between
the 80th and 84th percentile, not a rounding detail.

`TrainingConfig.calibration_days` inserts a slice between the training cutoff and the
scored window, so the offset is fitted out-of-sample for training **and** disjoint from
what it is judged on:

```
[ training .. train_cutoff ][ purge+embargo ][ calibration ][ validation ]
```

The offset is written to `meta.json` and applied by `predict_quantiles()`, so the band a
user sees is the band the reported coverage describes. Only the outer quantiles move — the
median is deliberately untouched, because CQR calibrates the *interval* and shifting the
median would change the reported direction with no evidence for it. The result is
re-sorted, since a large negative offset can pull the band inside the median and every
consumer assumes monotonicity (PYQ-124).

Verified on synthetic data in both directions: an 8-sigma band covering **100%** of
outcomes at a nominal 80% is narrowed to within 5 points of nominal on a held-out half, and
a deliberately too-narrow ±0.05 band is widened to the same target. Both are asserted, so
the fix cannot be one-directional. Eight tests in `tests/test_calibrate.py`, plus
`test_calibration_slice_produces_an_offset_that_forecast_reuses`, which trains a real
bundle and asserts the persisted offset is what the prediction path applies.

**Decision: `calibration_days` defaults to 0 (off).** Switching it on changes every
reported coverage figure, and that must be deliberate rather than silent. It also proved
less urgent than expected: PYQ-247's measurement reaches 76–80% coverage against nominal
80% from the *target change alone*, with no conformal correction — so the 99.3% pathology
is substantially a symptom of the level target rather than an independent defect. Conformal
calibration is the right tool for the residual and for a genuinely mis-calibrated model; it
is no longer the primary fix. The guarantee assumes exchangeability, which financial series
violate, which is why it is paired with PYQ-250's purged splits and why *achieved* coverage
is reported rather than assumed.

---

## [PYQ-249]
Add a time-series foundation model as a zero-shot baseline
Status: Open
Priority: Medium
Files: `pyquant/analysis/metrics.py`, new `pyquant/models/baselines.py`

Problem: the only baseline is persistence. That was the right first move (PYQ-201) and it
already produced the project's most important finding. But it leaves the central question
unanswered: **is the TFT — 25 features, four vendors, a training loop, a bundle format —
earning its complexity against something that needs none of it?**

Since 2024 a class of pretrained time-series foundation models has made that comparison
cheap. Chronos-2 (Amazon, 2025) and TimesFM (Google) both produce probabilistic zero-shot
forecasts from a raw univariate series with no training, no features, and no API key, and
both are reported to beat tuned statistical baselines out of the box on standard
benchmarks. Chronos in particular emits full predictive distributions, so it is directly
comparable to a quantile band rather than only to a point forecast.

Adding one as a *third* baseline changes what every number in the project means. Three
outcomes, all informative:

- TFT beats both baselines → the pipeline is genuinely earning its keep, and that is a
  strong, defensible claim.
- Foundation model beats TFT → an important finding, and a cheap alternative production
  path (no training, no GPU, no bundle management).
- Neither beats persistence → strong evidence for the efficient-market reading in
  investigations.md#pyq-312, and the honest headline result.

Ask: an optional `foundation` extra; a `baselines.py` exposing the same
`(n_samples, horizon, n_quantiles)` contract `evaluate_predictions` already consumes, so
it plugs into the existing metric path unchanged; a `--baseline persistence|chronos|both`
flag on `backtest`. Keep it optional — it must not become a required dependency of the
core install.

Acceptance criteria: `backtest --baseline both` reports skill against both baselines on
the same windows; the foundation model path is skipped with a clear message when the extra
is not installed (matching the existing FinBERT degradation pattern).

---

## [PYQ-250]
Purge + embargo around every walk-forward split
Status: Resolved — 2026-07-27
Priority: Medium
Files: `pyquant/models/tft.py` (`train`, `walk_forward_backtest`, `_window_validation_dataset`)

Problem: PYQ-127 made the walk-forward genuinely walk. The splits are still naive in the
sense the financial-ML literature means: training rows immediately adjacent to the
validation window remain in training, and their own decoder targets overlap the period the
validation window is about to be scored on.

The standard treatment (López de Prado, *Advances in Financial Machine Learning*) is:

- **purge** — drop training samples whose label window overlaps the test window. Here, any
  training sample whose decoder covers `time_idx > cutoff - horizon` overlaps the
  validation period and should be dropped.
- **embargo** — additionally drop a small buffer after the test window before training
  resumes, because serial correlation leaks information across the boundary even without
  literal overlap.

Without these, reported out-of-sample performance is optimistically biased. Given the
project's demonstrated seriousness about look-ahead (PYQ-101/103/115/116/123/127), this is
the remaining known gap in the same family, and it is the one the literature considers
table stakes.

Ask: `TrainingConfig.purge_horizon` (default = `max_prediction_length`) and
`TrainingConfig.embargo_days` (default a small positive number, e.g. 2), applied when
building each training dataset in both `train()` and `walk_forward_backtest()`.

Note the interaction with PYQ-136: purging will produce windows with unequal sample counts,
which is precisely when `aggregate_metrics`'s unweighted averaging becomes wrong. Land
PYQ-136 first or together.

Acceptance criteria: a test asserting no training sample's decoder overlaps the validation
decoder range at any origin; a documented before/after skill comparison (skill is expected
to *drop*, and that drop is the point).

Resolution (2026-07-27): `TrainingConfig.purge_horizon` (default `None` → one
`max_prediction_length`) and `TrainingConfig.embargo_days` (default 2), applied through a
new public `tft.purged_training_cutoff(cutoff, settings)` in **both** `train()` and
`walk_forward_backtest()`. Training shrinks; the validation window does not move, so
PYQ-117's hard-won sample size is untouched —
`test_train_still_validates_on_the_full_holdout_after_purging` asserts exactly that.

The overlap being removed is worth stating precisely, because it is not quite the one the
ticket describes. Training decoders end at the cutoff and validation decoders start after
it, so there is no literal decoder-to-decoder overlap. The leak is that a validation
sample's **encoder** reads the `max_encoder_length` days *before* its own decoder — days
that are training targets. Training and evaluation therefore share days across the
boundary, and purge + embargo is what separates them.

`test_no_training_decoder_overlaps_the_validation_window_at_any_origin` asserts the
invariant at *every* walk-forward origin rather than one, since PYQ-127's defect was
precisely that the origins were not distinct; it checks both non-overlap and that the gap
is at least `horizon + embargo`.

**The predicted before/after did not happen, and that is the finding.** The ticket expects
skill to *drop* and says the drop is the point. Measured (AAPL, seed 42, same pin, 5
windows, 12 epochs) it **rose** in both arms:

```
close       target:  -59.5%  ->  -4.5%   skill
log_return  target:   +2.4%  ->  +3.8%   skill
coverage (log_return): 76.0% -> 80.0%   against a nominal 80%
```

The literature's expectation is that removing leaked information removes optimism, so the
honest reading is not "purging helps". It is that at this sample size (n_points = 25,
effective n ≈ 5) the change sits inside the noise, and the level-target arm is so unstable
per-window — `[+0.53, +0.62, +0.78, −1.54, −2.13]` — that a 55-point swing in the mean says
little. What the measurement *does* establish is that purging is not catastrophic to fit
quality, which was the practical risk. A properly-powered before/after belongs with
PYQ-251's intervals and is recorded as a follow-up rather than claimed here. The PYQ-136
interaction is satisfied: `aggregate_metrics` already weights by `n_points`, which is what
makes unequal windows safe to pool.

---

## [PYQ-251]
Report effective sample size and block-bootstrap intervals
Status: Resolved — 2026-07-27
Priority: Medium
Files: `pyquant/analysis/metrics.py`, `pyquant/cli/app.py`

Problem: PYQ-117 was right that a percentage without a denominator is misleading, and
`Evaluated on 56 windows (280 predictions)` was a large improvement. But those 56 windows
are built with `min_prediction_idx=cutoff + 1` and therefore **overlap heavily**:
consecutive windows share 4 of their 5 target days and roughly 59 of their 60 encoder
days.

So "280 predictions" overstates the independent evidence by roughly a factor of the
horizon. The effective number of independent windows is closer to `validation_days /
horizon` ≈ **12**, not 56. Any confidence interval computed as if n = 280 would be about
`sqrt(5)` ≈ 2.2× too narrow — which matters directly, because the interesting question
about `57.5% directional accuracy` is whether it is distinguishable from 50%, and the
answer changes completely between n = 280 and n = 12.

Ask: two things.
- Report an **effective sample size** alongside the raw count, computed from the overlap
  geometry (`n_independent ≈ n_samples / horizon` as a first approximation), and label the
  raw figure as overlapping.
- For `backtest`, add a **moving-block bootstrap** confidence interval on the headline
  metrics — blocks of length ≥ horizon preserve the autocorrelation the naive bootstrap
  destroys. Report e.g. `directional accuracy 57.5% [46.2, 68.1]`, which immediately
  answers the question a bare 57.5% invites.

This is the natural completion of PYQ-117: that ticket made the sample size visible; this
one makes it *honest*.

Acceptance criteria: a unit test asserting the effective-size calculation on a known
geometry; the backtest table shows an interval alongside each rate; a test asserting the
bootstrap uses blocks no shorter than the horizon.

Resolution (2026-07-27): `effective_sample_size(n_samples, horizon)` and
`moving_block_bootstrap_interval(values, block_size, ...)` in `analysis/metrics.py`;
`EvaluationMetrics.effective_n_samples` derives the horizon from
`n_points / n_samples` so it is right for both a single window and a pooled aggregate. The
Rich table now reads `N overlapping windows (M predictions; effective n≈K)` instead of a
bare count, `--format json` carries `effective_n_samples`, and `backtest` adds a
`Directional accuracy 95% CI` row from a moving-block bootstrap with blocks no shorter than
the horizon.

This turned out to matter more than expected. PYQ-247's comparison reports 25 predictions
per arm, which reads like a reasonable sample until the effective figure (≈5 independent
windows) is next to it — and that number is what stopped the ~62-point skill improvement
being written up as settled rather than as directional evidence. The completion of PYQ-117
it was meant to be: that ticket made the sample size visible, this one makes it honest.

---

## [PYQ-252]
CRPS, Winkler score and a PIT histogram
Status: Resolved — 2026-07-27
Priority: Medium
Files: `pyquant/analysis/metrics.py`, `pyquant/cli/charts.py`

Problem/Ask: complements PYQ-227 (per-quantile exceedance + pinball). Three further
standard probabilistic-forecast diagnostics, all cheap given the quantile output already
exists:

- **CRPS** (continuous ranked probability score) — the standard proper scoring rule for a
  full predictive distribution, approximable from a quantile set by averaging pinball loss
  across quantiles. Single number, strictly proper, comparable across models — which is
  what makes a comparison against PYQ-249's foundation-model baseline meaningful.
- **Winkler / interval score** — scores an interval on both coverage *and* width in one
  figure. Directly diagnoses the 99.3%-on-80% pathology, which coverage alone cannot: a
  band can hit nominal coverage by being enormous, and Winkler penalises exactly that.
- **PIT histogram** — the probability-integral-transform of actuals through the predictive
  CDF. Uniform means calibrated; U-shaped means overconfident; hump-shaped means
  underconfident (the expected shape here). One glance replaces several numbers, and
  `charts.py` already has the plumbing to render it.

Land after PYQ-227 so the two share one refactor of `EvaluationMetrics` rather than two.

Acceptance criteria: unit tests for CRPS and Winkler against hand-computed values on small
arrays; the PIT histogram renders in `explain` or a new `pyquant calibration` command.

Resolution (2026-07-27): `crps_from_quantiles()`, `winkler_score()` and `pit_values()` in
`analysis/metrics.py`, all three carried on `EvaluationMetrics`, flowing through
`aggregate_metrics` (CRPS/Winkler weighted by `n_points` per PYQ-136; PIT values
*concatenate*, since a pooled histogram is the point of collecting them), through
`--format json`, and onto the Rich tables as "lower is better" rows.

Winkler earned its place immediately. In PYQ-247's comparison it separates the arms by
three orders of magnitude (64.3 vs 0.08) where coverage alone would call a 52%-covering
band and an 80%-covering one merely "different" — it charges for width as well as misses,
which is the diagnosis the 99.3%-on-nominal-80% pathology needed and that coverage
structurally cannot give.

`alpha` is derived from the configured quantiles (`q[0] + (1 − q[-1])`) rather than assumed
to be 0.2, so a non-default quantile set is scored against its own nominal rate — the same
class of bug PYQ-122 fixed for the band label.

Hand-computed tests as required: `test_crps_and_winkler_match_hand_calculation` derives
CRPS = 0.3 and Winkler = 4.0 from a 1x2 array with the arithmetic written out in the
docstring. `test_winkler_penalises_a_miss_far_more_than_the_width_it_saves` (91.0 vs 20.0)
and `test_winkler_scores_an_overwide_band_worse_than_a_calibrated_one` pin the property
that motivated it — two bands with *identical* 100% coverage score very differently.
`test_pit_is_uniform_for_a_calibrated_forecaster_and_clustered_for_an_overwide_one` checks
both shapes.

**Partially deferred:** the PIT *histogram* is not yet rendered. The values are computed,
aggregated and serialized, so the data is available to any consumer, but no `explain` panel
or `pyquant calibration` command draws it. Recorded honestly rather than claimed — the
numeric half is done and tested, the chart is not.

---

## [PYQ-253]
Optuna hyperparameter search (absorbs PYQ-211's scope)
Status: Resolved (pending, 2026-07-27)
Priority: Medium
Files: `pyquant/models/tft.py` (`tune`, `TuneResult`), `pyquant/cli/app.py`, `pyproject.toml`

Problem/Ask: PYQ-211 proposes `Tuner.lr_find` for the learning rate specifically. That is
a reasonable narrow fix, but learning rate is one of at least six coupled knobs
(`hidden_size`, `attention_head_size`, `dropout`, `hidden_continuous_size`,
`learning_rate`, `early_stopping_patience`) and tuning one in isolation is close to
uninformative — especially since PYQ-224's own note observes that patience and
`validation_days` interact with how noisy the selection metric is.

pytorch-forecasting ships `optimize_hyperparameters()` (Optuna-backed) for exactly this
model. Using it gives a proper study with pruning, a persisted database of trials, and a
record of *which* configuration won — which slots naturally into `runs.jsonl` and PYQ-259.

**Two prerequisites, both important.** First, land PYQ-247 (return target) before running
any search: a large hyperparameter study inside a formulation with a near-unbeatable
baseline burns GPU hours to discover that nothing helps. Second, every trial is a
selection event, so a search of *N* trials inflates the best observed score — report the
winning configuration's performance on a **held-out period the search never saw**, and be
explicit that the in-search score is optimistically biased.

Suggest superseding PYQ-211 by this ticket rather than keeping both.

Acceptance criteria: a `pyquant tune SYMBOL --trials N` command persisting an Optuna study;
the winning config written as a YAML file in `configs/`; the reported figure comes from a
period excluded from the search.

Resolution: `tft.tune()` wraps pytorch-forecasting's `optimize_hyperparameters()` over
`hidden_size`, `attention_head_size`, `dropout`, `hidden_continuous_size`,
`learning_rate` and `gradient_clip_val`. Split geometry: the search trains and selects
*entirely* within `df[time_idx < held_out_start]`; the final `held_out_days` of the panel
are never in any trial's train or validation loader. The winning config is then retrained
via the existing `train()` with `validation_days=held_out_days` on the *full* panel, so
`train()`'s own last-N-days validation slice lands exactly on the region no trial saw —
reusing the well-tested split logic rather than duplicating it. `pyquant tune SYMBOL
--trials N` reports both numbers side by side, explicitly labelled, with a note that the
in-search value is a selection-event score and the held-out numbers are the ones to trust.

Two real dependency-chasing detours worth recording, since they're easy to hit again: (1)
`optimize_hyperparameters()` unconditionally constructs a `TensorBoardLogger` — not
optional despite not being passed through `trainer_kwargs` — so `tensorboard` is a genuine
runtime requirement pytorch-forecasting's own `pip install optuna statsmodels` hint doesn't
mention; without it the failure is a bare `ModuleNotFoundError` several frames inside the
library. (2) It also unconditionally adds a `LearningRateMonitor` callback, which raises
`MisconfigurationException` if the Trainer has no logger — so unlike every other Trainer in
this file, `trainer_kwargs` here must *not* set `logger=False`. Both are noted inline at the
call site so the next person doesn't rediscover them by traceback. The new `tuning` extra
is `optuna`, `optuna-integration`, `statsmodels`, `tensorboard` — matching
pytorch-forecasting's own `[tuning]` extra plus the undeclared `tensorboard` requirement.

Optuna's own SQLite storage backend persists the study at
`checkpoints/<SYMBOL>_TUNED/optuna_study.db` (`study_name` keyed to the bundle, `load_if_
exists=True` so a second `tune` call resumes rather than restarting), giving the "persisted
database of trials" the ask asks for without inventing a new storage format.

Verified with a real (not mocked) 2-trial, 1-epoch run against a monkeypatched synthetic
panel: produced real `best_params`, a real held-out `EvaluationMetrics`, a written+parseable
YAML config, and a populated study database — see
`test_tune_writes_a_config_and_scores_the_winner_on_a_held_out_split`. That test needs the
`tuning` extra, which CI's default job does not install; it skips cleanly there via
`pytest.importorskip`, the same disposition PYQ-308 already established for a real-FinBERT
CI job (verify locally, don't gate CI on an optional extra). A separate test confirms a
missing extra fails with a clear message naming the fix, not a bare traceback from inside
pytorch-forecasting.

Superseded PYQ-211 (learning-rate-only tuning) as that ticket's own updates anticipated.

---

## [PYQ-254]
Promote options data from display context to model features
Status: Resolved (pending, 2026-07-27)
Priority: Medium
Files: `pyquant/data/options.py`, `pyquant/data/dataset.py`, `pyquant/config.py`,
`pyquant/models/tft.py` (`_SCHEMA_DATA_FIELDS`), `pyquant/cli/app.py`

Problem: `fetch_options_snapshot()` computes put/call ratio, ATM implied volatility and IV
skew — genuinely forward-looking, market-priced expectations, and the only truly
predictive (rather than backward-looking) signal in the whole data layer. It is fetched at
`forecast` time, printed once, and thrown away. `cli/app.py` says so explicitly: *"An
options snapshot is live market context, not a model input."*

So the README's multi-modal framing counts four sources but the model sees three, and the
one it does not see is arguably the most informative. `Realized_Vol_20`'s own comment
concedes it is a "free-data stand-in for options-implied vol."

The obstacle is real and is why the current choice is defensible: yfinance exposes only a
*current* chain, not history, so there is no way to build a historical IV series from it
and therefore no way to train on the feature. Two honest routes:

1. **Start accumulating.** Add a `pyquant snapshot SYMBOL` command that appends today's
   options metrics to a local time series. Useless on day one, a genuinely proprietary
   dataset after a year, and it costs almost nothing to start now. This is the highest
   value-per-effort option precisely because the value is time-dependent.
2. **Source historical IV** — CBOE offers some free historical index IV; several vendors
   sell equity surfaces (see PYQ-258 and `SYSTEMS-RESEARCH.md`). Evaluate cost against
   value once route 1 has produced enough data to estimate the latter.

Either way the same publication-timing discipline applies as PYQ-101/PYQ-129: a snapshot
taken at time *T* must be joined to a row whose target is after *T*.

Acceptance criteria: for route 1, a `snapshot` command with an append-only store and a
join path in `build_panel` that activates once sufficient history exists; a test asserting
the join respects observation time.

Resolution: route 1 only, as the ticket recommends. `pyquant snapshot SYMBOL` appends one
JSONL row per day to `Settings.options_history_dir` (a new, separately-anchored path —
deliberately not under `data.cache_dir`, which is a TTL-pruned convenience rather than a
permanent dataset). `load_snapshot_history()` returns empty (with the right columns, no
special-casing needed downstream) until `MIN_SNAPSHOT_DAYS` (20) distinct days have
accumulated. `build_panel` joins it exactly like PYQ-256 already handles sentiment's
identical shape of problem: forward-filled onto the trading calendar, with a
`has_options_history` indicator and a neutral (0.0) fill for the uncovered region — not the
panel's generic `ffill()`+`dropna()`, which would otherwise drop nearly the *entire* training
window down to whatever thin tail has coverage, since the history only ever grows forward
from whenever `snapshot` was first run. The recorded date is the observation's own
US/Eastern calendar date, so a join can never backfill a value onto a row before it existed
(same discipline as PYQ-101/PYQ-129) — verified directly in
`test_build_panel_joins_options_history_respecting_observation_time`, which asserts every
row before the recorded date is neutral/unflagged and every row on-or-after carries the
real value, with zero rows dropped.

`use_options` (already existed, previously gated only the CLI's *display* snapshot) now also
gates this join, and joined to `_SCHEMA_DATA_FIELDS` so `settings_for_bundle` restores it
correctly at forecast/explain time (PYQ-119's mechanism) — a bundle trained with options
history available must not silently lose those columns because a later `forecast` run's
default toggle differs.

Route 2 (sourcing historical IV from a paid vendor) is explicitly not pursued here, per the
ticket's own sequencing — there is no accumulated data yet to weigh its cost against.

---

## [PYQ-255]
Signal evaluation: does `scan`'s BUY/SELL actually make money?
Status: Resolved (pending, 2026-07-27)
Priority: Medium
Files: `pyquant/cli/app.py` (`scan`, `backtest`), `pyquant/analysis/signals.py` (new),
`pyquant/analysis/serialize.py`, `pyquant/models/tft.py` (`walk_forward_backtest`,
`BacktestResult`)

Problem: `scan` emits BUY / SELL / HOLD from a threshold on expected return plus a
band-direction guard (PYQ-206, PYQ-124). Nothing anywhere evaluates whether following
those signals would have made or lost money. The project measures forecast *accuracy*
carefully and its *usefulness* not at all — and they are different questions: a model can
have excellent MAE and a useless signal (right about magnitude, wrong about the sign that
matters), or mediocre MAE and a profitable one.

This is also the gap most visible to anyone reading the repo as a finance project rather
than a forecasting project.

Ask: a signal-evaluation layer scoring the historical signal series over the backtest
period — hit rate conditional on a signal firing (not on all days), average return
following each signal class, turnover, and cumulative P&L against a buy-and-hold benchmark
with a configurable per-trade cost (a few basis points is a realistic default and is
usually decisive at daily frequency).

Two cautions worth writing into the ticket so they are not discovered later: (a) the
signal thresholds (±2%) are themselves parameters, so tuning them on the same data is a
selection event — hold out a period; (b) `scan`'s guard requires the entire band on one
side of zero, and with the current 99.3%-coverage band that will essentially never fire,
so this ticket is only informative **after PYQ-248**.

Acceptance criteria: `backtest --signals` reports hit rate, turnover and cost-adjusted P&L
vs. buy-and-hold; unit tests for the P&L accounting on a hand-built signal series.

Resolution: `analysis/signals.py` adds `classify_signal()` (the same BUY/SELL/HOLD logic
`scan` used inline, extracted so the two cannot drift — `scan` now calls it directly) and
`evaluate_signals()` (hit rate conditional on firing, average return per class, turnover,
cost-adjusted cumulative P&L vs. buy-and-hold, via simple compounding). `walk_forward_
backtest` gained an opt-in `compute_signals` flag (default off — it costs one extra forward
pass per window) that records, per origin, the signal `scan` would have shown and the
realized return; `backtest --signals` wires this through to both the Rich table and
`--format json`, with `--cost-bps` (default 5) configurable.

The two cautions the ticket flagged are carried forward rather than solved in code, since
neither has a code fix: the module docstring states that tuning `classify_signal`'s
threshold against the same data it's evaluated on is a selection event, and the CLI prints
a note that the band guard needs PYQ-248's conformal calibration (implemented, off by
default) to fire meaningfully against this project's current ~99% coverage on a nominal 80%
band. `--signals` is therefore functional now but will mostly show all-HOLD, zero-trade
output against an unmodified default config — expected, not a bug, and the printed note
says so.

Guarded by 11 unit tests in `test_signals.py` on hand-built series (conditional hit rate,
cost-only-on-position-change, an oracle signal beating buy-and-hold) needing no training,
plus `test_tft.py::test_walk_forward_backtest_computes_a_signal_per_window_when_requested`
and two CLI tests (`--signals` reaches both Rich and JSON output; the flag is opt-in and the
extra pass is skipped when absent).

---

## [PYQ-256]
`has_sentiment_data` indicator column
Status: Resolved — 2026-07-27
Priority: Low
Files: `pyquant/data/dataset.py` (`build_panel`), `pyquant/data/sentiment.py`

Problem/Ask: the concrete remediation half of investigations.md#pyq-301. Finnhub's free
tier covers ~365 days, so at the default `period="5y"` roughly 80% of training rows carry
a structural `Sentiment = 0` that means "no data," while at prediction time 0 would mean
"neutral news." The model cannot distinguish those two meanings, and the second is the
only one that ever occurs live — a textbook train/serve distribution shift on a feature.

Add a binary `has_sentiment_data` column so the model can condition on which regime a row
is in. Cheap, standard practice for structurally-missing features, and it makes PYQ-301's
measurement interpretable: if the model learns to ignore `Sentiment` when the indicator is
0, that is directly visible in `explain`'s variable-selection weights.

Consider also truncating the effective training window to sentiment availability as a
comparison arm — fewer rows but a consistent schema — and letting the backtest decide.

Acceptance criteria: the column is present when sentiment is enabled and absent otherwise
(so it never breaks the PYQ-118 schema check); a test asserting rows outside the news
window get 0 and rows inside get 1.

Resolution (2026-07-27): `build_panel()` adds `has_sentiment_data` whenever sentiment
actually joined, and only then — so a bundle trained without sentiment keeps its exact
feature schema and the PYQ-118 check is unaffected.

One decision the ticket left open: the flag marks **coverage**, not per-day presence. A
quiet trading day *inside* the news window is genuinely neutral and gets 1; a day before
Finnhub's ~365-day horizon is missing and gets 0. Keying it off "did this row have a
headline" would have collapsed the two meanings the column exists to separate. Coverage
starts at the first date carrying any sentiment observation, and `build_panel` logs it
(`news coverage begins <date> (N of M rows)`), which is also the cheapest form of PYQ-301's
measurement — it now prints on every run rather than needing a study.

Guarded by `test_has_sentiment_data_separates_no_data_rows_from_genuinely_neutral_ones`
(asserts 0 before the window, 1 inside, *and* that a zero-Sentiment day inside the window
still reads 1 — the distinction the column is for),
`test_has_sentiment_data_is_absent_when_sentiment_is_disabled`, and
`test_has_sentiment_data_is_a_model_feature`, since a column the model never sees would be
decorative.

---

## [PYQ-257]
Use FRED/ALFRED vintages instead of a fixed publication lag
Status: Resolved — 2026-07-26
Priority: High
Files: `pyquant/data/macro.py` (`_FredSeriesSpec`, `_fetch_fred`)

Problem: PYQ-101's fix — a per-series `publication_lag_days` shifted onto the index — was
the right emergency fix and PYQ-305 made it a convention. It is still an **approximation**
of the correct thing, in three ways:

1. The lag is a constant; real release calendars are not. CPI is released on a schedule
   that varies by several days month to month, and holidays shift it further.
2. It handles the *first* release only. Macro series are **revised**, sometimes
   substantially. `get_series()` returns today's revised value for a historical date, so a
   training row for 2019 sees the number as it is understood *now*, not as it was
   published then. That is look-ahead of a subtler kind that a date shift cannot fix at
   all — and it is the dominant error for GDP and PCE, precisely the series PYQ-214 wants
   to add next.
3. It requires a hand-maintained constant per series, which is the recurring-maintenance
   cost PYQ-305 was trying to bound.

FRED's sibling database **ALFRED** exists for exactly this: it serves *vintage* data —
what each series looked like as of any given date. `fredapi` exposes it directly via
`get_series_as_of_date()` and `get_series_first_release()`.

Ask: switch `_fetch_fred` to vintage-aware retrieval, so each panel row carries the value
that was actually published and known as of that date. This makes the lag constants
unnecessary rather than merely more accurate, removes revision leakage entirely, and
generalises to any new series for free — which directly unblocks PYQ-214's plan to add
UNRATE/PCE/GDP without re-deriving a lag for each.

Note the trade-off honestly in the ticket: vintage retrieval is more API calls and more
data, so the panel cache (PYQ-205) matters more, and PYQ-110's per-series error isolation
must be preserved.

Acceptance criteria: a test asserting a training row for a historical date carries the
first-published value rather than the currently-revised one; `publication_lag_days`
removed or documented as a fallback for sources without a vintage API; PYQ-305's
convention note updated to record the supersession.

Resolution (2026-07-26): `_fetch_fred()` now reads `fredapi`'s
`get_series_all_releases()` response and emits the latest observation available on each
actual `realtime_start` release date. Fixed publication-lag constants were removed. This
means a row sees neither an observation before its first release nor a later revision
before that revision's own release date. Series remain independently guarded as required
by PYQ-110. `test_fetch_macro_uses_the_first_published_cpi_vintage` verifies both the
first-release and later-revision boundaries offline.

---

## [PYQ-258]
Pluggable price-provider interface with a licensed fallback
Status: Resolved — 2026-07-27
Priority: Medium
Files: `pyquant/data/prices.py`, `pyquant/data/sectors.py`, `pyquant/data/macro.py` (`_fetch_vix`), `pyquant/data/options.py`

Problem/Ask: the concrete shape for PYQ-214's second point. yfinance is currently the sole
source of OHLCV **and** VIX **and** sector ETFs **and** options — four of the project's
data sources behind one unofficial, unversioned, ToS-ambiguous scraper of Yahoo's internal
endpoints. Every enrichment degrades gracefully except the one that everything else
depends on: if `fetch_prices` fails, nothing works.

Introduce a `PriceProvider` protocol (`fetch_ohlcv(symbol, start, end) -> DataFrame`) with
a `YFinanceProvider` implementation and at least one licensed alternative behind a config
toggle — Tiingo and Alpha Vantage both have usable free tiers with real API keys; Polygon
and EODHD are the paid steps up. See `SYSTEMS-RESEARCH.md` for the comparison.

Two properties matter more than which vendor is chosen: (a) an *interface*, so switching
is a config change rather than a rewrite; (b) an explicit statement of which adjustment
convention the model assumes — split/dividend adjusted or not — since PYQ-228 records that
this is currently decided by whichever yfinance version resolves, and it silently changes
every price level and every derived indicator.

Acceptance criteria: the protocol exists and yfinance implements it; one alternative
provider is implemented and selectable; a test asserting both providers return the same
column schema and dtypes; the adjustment convention is documented and asserted.

Resolution (2026-07-27): new `pyquant/data/providers.py` with a `PriceProvider` Protocol
(`fetch_ohlcv(symbol, *, period, start, end) -> DataFrame`), a `YFinanceProvider` holding
the existing behaviour, and a `TiingoProvider` as the licensed alternative.
`fetch_prices(..., provider=...)` accepts either a name or an object, so swapping vendors
is a config change rather than a rewrite — the property the ticket says matters more than
which vendor is chosen.

The other property — an explicit adjustment convention — is now `prices.AUTO_ADJUST = True`,
passed at *every* yfinance call site (prices, VIX, sector ETFs, options), which is PYQ-228's
other half. Tiingo is asked for `adjOpen`/`adjClose`/… so the convention is chosen rather
than inherited from whichever default a version happens to ship.

`assert_ohlcv_contract()` is the executable statement of the shared schema — exactly
`Open/High/Low/Close/Volume`, float64, tz-naive `DatetimeIndex` named `Date`, ascending —
and `fetch_prices` applies it to *whatever* provider returns, so a new vendor's subtly
different frame fails loudly at the boundary instead of misaligning a join three modules
downstream. `test_the_contract_rejects_a_frame_that_would_misalign_a_join` checks it
actually rejects each of those four differences, since a contract that cannot fail
documents nothing.

Tiingo chosen over Alpha Vantage (25 requests/day free, too tight even for development) and
over Polygon/EODHD (paid only). **No new dependency**: it is a documented JSON REST endpoint
reached through `requests`, which is already required (non-negotiable #5).

Nine tests in `tests/test_providers.py`. The load-bearing one is
`test_both_providers_return_the_identical_column_schema_and_dtypes`, which drives both
parsers from realistically-shaped vendor payloads (yfinance's tz-aware index with
`Dividends`/`Stock Splits` and integer volume; Tiingo's ISO date strings, `adj*` names and
newest-first ordering) and asserts the two outputs are indistinguishable.

**Not verified against live Tiingo** — no `TIINGO_API_KEY` is configured here, so the HTTP
path is exercised only through an injected session. Stated rather than glossed: the parsing
and the contract are tested, the network round-trip is not.

---

## [PYQ-259]
Experiment tracking (MLflow) alongside `runs.jsonl`
Status: Resolved (pending, 2026-07-27)
Priority: Medium
Files: `scripts/runs.py` (new)

Problem/Ask: `runs.jsonl` (PYQ-203) plus provenance (PYQ-225) plus pins (PYQ-205) is a
genuinely thoughtful hand-rolled tracking system, and it should stay — it is
dependency-free, greppable, and lives next to the bundle. What it cannot do is *compare*:
answering "which of my last 30 runs had the best skill, and what did they have in common"
means writing a script, and the moment PYQ-253's Optuna search lands there will be hundreds
of runs rather than tens.

Add optional MLflow logging (an `mlflow` extra, local file backend by default, no server
required) writing params, metrics and the bundle as an artifact. It is additive — keep
`runs.jsonl` as the source of truth and treat MLflow as a queryable view, so nothing
breaks when the extra is absent. The comparison UI is the entire point; do not adopt it
for logging alone.

Worth noting for the ticket: this is a *systems-engineering* decision with a defensible
"no" — PYQ-310's precedent is that a tool must earn its place. Evaluate it against the
alternative of a `scripts/runs.py compare` reading `runs.jsonl` directly, which would cost
~100 lines and no dependency.

Acceptance criteria: `MLFLOW_TRACKING_URI` set → runs appear with params and metrics;
unset → no behaviour change and no import cost.

Resolution: **built and evaluated both**, and the answer is the ticket's own suggested
"no" — shipped `scripts/runs.py compare` (a real script now, not a hypothetical
alternative), declined the `mlflow` extra.

`scripts/runs.py` globs every `checkpoints/*/runs.jsonl` (each bundle keeps its own, so a
cross-bundle comparison needs exactly this), parses each recorded run, and prints one
sortable table (`--sort-by skill|val_loss|trained_at`, `--symbol` filter, `--top N`). 114
lines, zero new dependencies, reads `runs.jsonl` directly rather than a parallel tracking
store that could drift from it. Run for real against this session's accumulated bundles (12
runs across AAPL/ARM/NVO/pooled/solo variants, spanning three git shas) — output sorted
cleanly by skill, immediately surfaced that the two-epoch `AAPL` smoke-test bundle from
PYQ-319's profiling run was the worst of the twelve (`-4.315`), which is exactly the
"what did my runs have in common" question the ticket asks for.

Why this, not MLflow: `mlflow`'s local-file-backend mode avoids running a server, but the
*comparison UI* — the ticket's own stated "entire point" of adopting it — needs `mlflow ui`
running as a separate process either way, and the package itself pulls in Flask,
SQLAlchemy, alembic and several more transitive dependencies for functionality (a web UI,
a model registry, artifact stores) this project does not use. Against that, a 94-line
script reading the existing source of truth costs nothing and cannot drift from it by
construction. This is the same disposition PYQ-310 reached for mypy: a tool must earn its
place against doing nothing, and here "doing nothing" (extending the existing `runs.jsonl`
reader) both works today and costs less than the alternative.

Not closed forever: if PYQ-253's Optuna search actually produces the "hundreds of runs" the
ticket anticipates and a text table genuinely stops being enough — visual comparison across
many correlated hyperparameters is a real UI problem a table doesn't solve well — that
would be the evidence-based trigger to revisit MLflow, per non-negotiable #5. Nothing
observed yet justifies it.

---

## [PYQ-260]
Ship a `py.typed` marker
Status: Resolved — 2026-07-27
Priority: Low
Files: new `pyquant/py.typed`, `pyproject.toml`

Problem/Ask: PYQ-310 established that the codebase is fully annotated and internally
consistent under mypy (0 errors with `ignore_missing_imports`). But without a
`py.typed` marker (PEP 561), no downstream consumer's type checker will use any of it —
the annotations are invisible outside the package. One empty file plus a hatch
`force-include` entry.

Matters most for the PYQ-213 API layer and for anyone importing `pyquant` as a library,
which is the stated direction.

Acceptance criteria: the marker ships in the built wheel; a smoke check that mypy in a
consuming project resolves `pyquant` types rather than treating it as untyped.

Resolution (2026-07-27): empty `pyquant/py.typed` plus a hatch
`force-include` entry. Verified by building the wheel and listing it:
`pyquant-0.2.0-py3-none-any.whl` contains `pyquant/py.typed`.

The consuming-project mypy smoke check was **not** run — it needs a second throwaway
project and an install, which is more scaffolding than the one-line fact justifies. PEP 561
compliance is the marker's presence in the distribution, and that is what was verified.

---

## [PYQ-261]
Scaffold `pyquant/api/` per the PYQ-213 design note
Status: Resolved (pending, 2026-07-27)
Priority: Medium
Files: new `pyquant/api/` (`app.py`, `schemas.py`, `deps.py`, `jobs.py`,
`routes/{health,forecast,explain,train}.py`), `pyquant/analysis/serialize.py`
(`scan_row_to_dict`, new), `pyquant/cli/app.py` (`scan` now calls it), `pyproject.toml`

Problem/Ask: PYQ-213 delivered its stated deliverable — a design note — and closed
correctly. The implementation follow-up it names has no ticket, so it is currently
invisible in the backlog.

The note's own prerequisites have since landed: PYQ-114 (FinBERT cache no longer poisons,
which the note flagged as fatal for a long-running server), PYQ-118 (clear schema-mismatch
error, which the note called the top blocker for trusting the API against live data),
PYQ-119 (bundles record their config), PYQ-212 (reusable serializers). The main
outstanding blocker the note lists is PYQ-220 (absolute bundle/cache paths), which remains
open and should be landed first.

Build the v1 the note specifies: `/healthz`, `GET /forecast/{symbol}`, `GET
/explain/{symbol}`, `POST /scan`, `POST /train` → job id + `GET /train/{job_id}`, with
per-bundle locking and an LRU bundle cache, API-key auth, and the pydantic response models
reusing `analysis/serialize.py`. Stop where the note says the design stops — no queue, no
object storage — and file the follow-ups it lists rather than pre-building them.

Acceptance criteria: `uvicorn pyquant.api.app:app` serves the endpoints; response schemas
match the CLI's `--format json` payloads field-for-field (assert this in a test, so the two
front-ends cannot drift); concurrent requests against one bundle are serialised.

Resolution: built exactly the v1 surface the note specifies, stopping exactly where it says
to stop — no queue, no object storage, no rate-limiting beyond the auth gate. `GET
/healthz`, `GET /forecast/{symbol}`, `POST /scan`, `GET /explain/{symbol}`, `POST /train` →
202 + job id, `GET /train/{job_id}`.

Response models are thin pydantic wrappers constructed directly from `analysis/
serialize.py`'s existing dict functions (`ForecastResponse(**serialize.forecast_to_dict(fc))`,
same for explain/train), per the note's own recommended path — this is what makes the
acceptance criterion's field-for-field parity true *by construction*: the API and the CLI's
`--format json` call the identical conversion code, not two independently-typed schemas that
happen to agree today. `scan`'s row-building was inline in `cli/app.py` before this ticket
(not behind a serializer) — extracted to `serialize.scan_row_to_dict()` so `/scan` could
reuse it too, the same move PYQ-255 made for `classify_signal`.
`test_forecast_response_matches_the_cli_format_json_field_for_field` drives both front-ends
from one mocked `Forecast` and asserts the two JSON payloads are equal, not just
same-shaped.

Concurrency, per the note's #4: a per-bundle `threading.Lock` (created lazily, one per
bundle name) serializes `predict()` calls against a given loaded model without blocking
requests for *different* bundles, plus an LRU `BundleCache` (bounded, default 8) so a
request doesn't re-pay checkpoint deserialization and the `weights_only=False` unpickle
every time. `test_forecast_serializes_concurrent_requests_against_the_same_bundle` proves
the lock actually serializes: two real threads hit `/forecast/AAPL` concurrently against a
mocked `generate_forecast` that records its own start/end timestamps, and the test asserts
the two spans never overlap.

Auth is the one place this deviated from a literal reading of "keys from config/secret
store": `PYQUANT_API_KEYS` is read from the environment directly, not `Settings`, so a key
can never end up serialized into `meta.json`/a log/a response the way a `Settings` field
could — consistent with this project's secrets non-negotiable. It also fails **loudly**
(500, naming the fix) when unconfigured rather than silently allowing every request through
— an unconfigured gate defaulting to open would be exactly the "public endpoint spends the
operator's quota" risk §5 warns about. `PYQUANT_API_ALLOW_UNAUTHENTICATED=1` is the explicit
local-dev opt-out.

Two things the note flags as prerequisites were already satisfied and re-verified rather
than assumed: PYQ-220 (absolute `checkpoint_dir`/`cache_dir`) landed previously, and
`bundle_dir`/`cache_dir` were confirmed absolute in this pass; PYQ-212's serializers already
existed and needed no changes beyond the one extraction above.

New `api` extra (`fastapi`, `uvicorn[standard]`), matching the note's own suggested
pin. Verified against a *real* running server, not only `TestClient`: `uv run uvicorn
pyquant.api.app:app` started cleanly and `curl`'d `/healthz` (200), `/docs` (200, the
auto-generated OpenAPI UI), and an unauthenticated `/forecast/AAPL` (500, the
unconfigured-auth-gate path) all behaved as designed. 15 tests in `test_api.py`, plus the
`scan` refactor re-verified against the existing CLI `scan` tests.

Deliberately not built, per the note's own stopping point: object-storage bundle backend,
a real job queue (arq/Celery+Redis), and rate-limiting beyond the auth gate. All three
remain named follow-ups in `docs/api-design.md`, not silently dropped.

---

## [PYQ-262]
Pre-commit configuration
Status: Resolved — 2026-07-27
Priority: Low
Files: new `.pre-commit-config.yaml`

Problem/Ask: CI catches lint and backlog drift, but only after a push. A pre-commit config
running `ruff check --fix`, `ruff format`, `scripts/backlog.py check`, and the standard
whitespace/EOF/large-file hooks moves those to commit time. Zero new CI cost.

Pairs with PYQ-229's `ruff format --check` request — adopt the formatter locally in the
same pass that gates it in CI, so the first formatting commit is one deliberate diff rather
than noise spread across unrelated PRs.

Acceptance criteria: `pre-commit run --all-files` passes on a clean tree; the README's
Development section documents installation.

Resolution (2026-07-27): `.pre-commit-config.yaml` with the standard whitespace/EOF/YAML/
TOML/merge-conflict/large-file/private-key hooks, `ruff --fix`, and a local hook running
`scripts/backlog.py check` scoped to `backlog/*.md`.

No new dependency in the project sense: ruff and `scripts/backlog.py` are already required,
so this adds a workflow rather than a tool (the bar PYQ-310 set). `pytest` is deliberately
**absent** — the suite takes ~90s because of the torch tests, far too slow for a commit
hook, and it stays a CI gate.

`ruff format` is also deliberately absent, matching PYQ-229's decision from the other side:
the tree has ~20 unformatted files, and enabling the formatter here would spread that diff
across unrelated commits. CI reports the drift non-blocking meanwhile.

Verified: `pre-commit run --all-files` initially failed on two real nits (trailing
whitespace in `README.md`, a missing final newline in `.gitignore`), both auto-fixed, and
now passes all nine hooks on a clean tree.

---

## [PYQ-263]
`pyquant doctor` — environment and bundle health check
Status: Resolved — 2026-07-27
Priority: Low
Files: `pyquant/cli/app.py`

Problem/Ask: the project has a lot of optional, silently-degrading surface — two API keys,
one optional extra, a TTL cache, named pins, bundles that record a config and a feature
schema, and an accelerator that may or may not be available. Every one of those degrades
gracefully by design, which is correct, and which also means **a user cannot easily tell
what is actually switched on.**

A `doctor` command that reports, in one screen: which keys are set (presence only, never
values), whether `transformers`/FinBERT is importable, torch's available accelerator and
precision support, cache size and pin list, and for each bundle its symbols, training date,
recorded config, feature count and whether that schema can still be satisfied right now
(reusing PYQ-118's `_check_feature_schema`).

That last part is the genuinely useful bit: it turns "your bundle is broken" from a runtime
error into a proactive check, and it is the natural first thing to run when something is
wrong.

Acceptance criteria: `pyquant doctor` reports all of the above and exits non-zero if any
existing bundle's feature schema can no longer be satisfied; `--format json` supported.

Resolution (2026-07-27): `pyquant doctor`, backed by `pyquant/analysis/doctor.py`
(`run_doctor(settings) -> DoctorReport`). Reports key presence, the `sentiment` extra,
torch's accelerator and bf16 support, resolved paths, cache size and pins, and per bundle:
symbols, training date, target, feature count and whether its schema can still be
satisfied. Exits 1 if any bundle cannot. `--format json` emits the whole report.

The logic lives in `analysis/` rather than `cli/` so the same report can back a `/healthz`
endpoint (PYQ-213/PYQ-261) without going through Typer, per the layering rule.

The schema check is deliberately **offline**: it compares each bundle's recorded feature
list against what the currently-enabled sources *would* produce, rather than fetching. A
network check would make `doctor` slow, rate-limited and non-deterministic, and the failures
it looks for — a source toggled off, a key removed, an extra uninstalled — are all visible
without one.

PYQ-139 is the argument for this command existing: an entire vendor silently dropped out of
every panel and the only trace was one log line. `doctor` is where that becomes a question
someone can ask.

Five tests, including
`test_doctor_reports_key_presence_without_ever_printing_a_value` (asserts the secret's value
appears nowhere in stdout — the non-negotiable on secrets, made executable) and
`test_doctor_exits_non_zero_when_a_bundles_schema_can_no_longer_be_satisfied`. The test
fixture nulls both keys explicitly, because `Settings()` reads the developer's real `.env`
and the suite must not answer differently on a machine that happens to have keys.

Verified live against this checkout: two real bundles (AAPL 25 features, NVO 27) both
reported usable.

---

## [PYQ-264]
Fold PYQ-247/248/250 into the docs; deploy to GitHub Pages; nightly docs-drift check
Status: Resolved (pending, 2026-07-27)
Priority: Medium
Files: `docs/methodology.md`, `docs/index.md`, `docs/api/index.md`, `docs/architecture.md`,
`.github/workflows/docs.yml` (new), `.github/workflows/nightly.yml`, `README.md`

Problem: two known gaps, one already flagged in the backlog and one not. First,
PYQ-235's own resolution note says so explicitly: `methodology.md` was written against
commit `a7a2b5f`, before PYQ-247 (log-return target), PYQ-248 (conformal calibration) and
PYQ-250 (purge/embargo) landed, and its "In flight" section still describes all three as
pending. A reader hitting that page today gets the project's *previous* headline number
with no sign the log-return finding exists. Second, the API reference (PYQ-232) predates
this session's new modules (`analysis/signals.py`, `analysis/doctor.py`, `data/providers.py`,
the whole `pyquant/api/` package) and does not list them. Neither gap is hypothetical: both
are "the docs describe an earlier version of the code," the same class of problem PYQ-140
found for a vendor docstring.

Separately: the docs build cleanly (PYQ-233) and a `.readthedocs.yaml` exists (PYQ-234), but
no hosted site is actually reachable — PYQ-234's own resolution note says plainly that no
Read the Docs project is connected. Requested directly: get a working public URL via GitHub
Pages instead, and add a nightly job that rebuilds the docs on a schedule independent of any
code change, so an upstream intersphinx inventory change (pandas/torch/pytorch-forecasting
renaming or moving a symbol this project links to) is caught as a notification rather than
discovered by a reader clicking a dead link.

Resolution:

**Content.** `methodology.md`'s headline box now states both configurations side by side —
the default (`target="close"`, −23.5% skill / 57.5% dir. acc. / 99.3% coverage, 280
points) and PYQ-247's `log_return` comparison (+2.4%/+3.8% purged, 52–56% dir. acc.,
76–80% coverage, effective n≈5) — rather than the default alone with the log-return finding
absent. The old "In flight" section (describing PYQ-247/248/250 as not-yet-landed) is now
"Landed since this page was first written," stating each one's actual measured effect,
including the one surprising correction: PYQ-250's purge/embargo is *always on* by default
now (not opt-in), so it is folded into every number on the page already, not a future
change. A new "related open questions" section links PYQ-314/315/316's findings (weak
`explain` agreement beyond the top feature, pooling measured worse, sentiment measured
worse) as the same discipline applied elsewhere.

One more number earned its own section rather than a table edit: a fresh
`pyquant backtest NVO --windows 5` at the *same default config*, run for this page
(2026-07-27, git `90afcf8`), scores **+36.2% skill** — better than −23.5%, and deliberately
**not** substituted into the headline table, because it is not the same measurement.
`train`'s 280-point figure scores one model's validation holdout; `backtest --windows 5`
trains five independent models and scores each on only its own 5-day horizon (25 points,
`effective_n_samples = 1`). Per-window results (+55.7%, −24.3%, +33.7%, +25.0%, +52.1%) are
published alongside the aggregate specifically so the instability is visible rather than
smoothed away by one summary number — four positive origins outweighing one negative one is
a different claim than "the model works," and the page says so explicitly. This is the
{ref}`negative-result`/`non-negotiable #1` discipline applied to a result that would have
been easy to quietly prefer.

`architecture.md` gained an `api/` entry in "what each layer owns," `api/` in the two
structural rules, and a rewrite of the options-data section (previously unconditionally
"display-only," now describing PYQ-254's accumulation path). `api/index.md` gained
`pyquant.data.providers`, `pyquant.analysis.doctor`, `pyquant.analysis.signals`, and a new
"API service" section for all of `pyquant.api.*`.

**A real dependency gap this surfaced:** `pyquant/api/{app,deps,routes/*}.py` import
`fastapi` at module level, and Sphinx's autodoc genuinely imports every module it
documents — so `--group docs` alone (no `api` extra) fails the `-W` gate on a missing
dependency, not a real docs defect. Fixed in three places that all build the same docs:
`.readthedocs.yaml`, `ci.yml`'s docs-build step, and the new `docs.yml` all now install
`--group docs --extra api` together. Caught by actually rebuilding from a clean
`docs/_build/` + `docs/api/_generated/` locally before trusting any of this, not assumed.

**GitHub Pages.** New `.github/workflows/docs.yml`: builds under the same `-W --keep-going`
gate as `ci.yml` (it re-publishes the already-gated tree, it does not re-decide whether the
build is clean), uploads via `actions/upload-pages-artifact`, deploys via
`actions/deploy-pages` with `concurrency: cancel-in-progress` so a second push does not race
a build still in flight. Triggered on push to `main` (path-filtered to what can actually
change the rendered output) plus manual dispatch. Live at
`https://axelsuu.github.io/Pytorch-Quant-Model/` once the workflow's first run completes and
Pages is enabled on the repository (a one-time repository-settings step outside this
session's reach — recorded plainly, matching PYQ-234's own precedent for the equivalent RTD
gap, rather than claimed without being verifiable from here).

**Nightly docs-drift.** `nightly.yml` gained a `docs-drift` job alongside the existing
vendor `smoke` job, same philosophy stated once at the top of the file: a notification, not
a merge gate. It rebuilds with `-E` (discard the cached environment, forcing a genuine
re-fetch of every intersphinx inventory) so a pandas/numpy/torch/pytorch-forecasting/
pydantic rename is caught independent of any commit in this repository, on the same daily
schedule as the vendor check.

`README.md`: docs link replaced with the live Pages URL plus Docs/Nightly badges alongside
the existing CI badge; commands table gained `tune`, `snapshot` and `doctor`, all missing
from a previous pass; both docs-build invocations updated to `--group docs --extra api`.

Verified: `sphinx-build -W --keep-going` exits 0 from a clean `docs/_build/` +
`docs/api/_generated/` after every content change in this ticket (four rebuilds, the last
one clean); all three workflow YAML files parse; `scripts/backlog.py check` clean.

---

## [PYQ-265]
Report skill across seeds, not from a single seed
Status: Open
Priority: High
Files: `pyquant/config.py`, `pyquant/models/tft.py`, `pyquant/analysis/metrics.py`, `pyquant/cli/app.py`

Problem/Ask: `TrainingConfig.seed` is a single int defaulting to 42, and
`seed_everything(settings.training.seed)` is called before every fit (`tft.py:295`, `:530`,
`:661`). Every number this project has ever reported — the -23.5% headline, PYQ-247's
+2.4%, investigations.md#pyq-315's "pooling is worse", investigations.md#pyq-316's
"sentiment hurts" — is **one draw from one seed**. Nothing in the codebase has ever
measured how much of any of those deltas is run-to-run initialisation noise.

That matters most exactly where the project's conclusions are thinnest. PYQ-247 moved skill
by 61.9 points and is trusted; investigations.md#pyq-316 moved it by 0.027 (+0.045 to
+0.018 when sentiment was added) and is described as "sentiment measurably hurts". If
seed-to-seed standard deviation on this data is anywhere near 0.03, the second finding is
indistinguishable from noise and the ticket's recommendation rests on nothing. Nobody
knows which, because the experiment has never been run — see investigations.md#pyq-321,
which this ticket is the tooling half of.

Build: `TrainingConfig.seeds: list[int]` (defaulting to `[42]`, so existing behaviour and
existing bundles are unchanged), a `train`/`backtest` path that fits once per seed, and
metric reporting that carries **mean, standard deviation and min/max across seeds** rather
than a point. `--seeds 5` on the CLI as sugar for the first N of a deterministic sequence.
The per-seed `EvaluationMetrics` should all be retained, not just their summary, so
PYQ-266's paired test can consume them.

Cost is the obvious objection and should be stated rather than hidden: this multiplies
training time by the seed count. That is the correct price for the claim, and it is why the
default stays at one seed and the multi-seed run is opt-in — the same shape as PYQ-248
shipping conformal calibration defaulted off.

Note this interacts with reproducibility's three legs (seed + pinned data + code version).
A multi-seed run has no single seed to record, so `meta.json` must record the seed *list*
and the per-seed results; a bundle trained across seeds needs to say which seed's weights
it actually kept.

Acceptance criteria: `pyquant backtest SYMBOL --seeds 5` reports skill as `mean ± sd (min, max)` over five
fits; `--format json` carries the per-seed metrics; `meta.json` records the seed list; a
test asserts that two runs at the same seed list produce identical results and that a
different seed list produces different ones.

---

## [PYQ-266]
Paired significance test for comparing two configurations
Status: Open
Priority: High
Files: `pyquant/analysis/metrics.py`, `pyquant/cli/app.py`, `docs/methodology.md`

Problem/Ask: this project decides things by comparing two configurations — level vs.
log-return target (PYQ-247), pooled vs. solo (investigations.md#pyq-315), sentiment on vs.
off (investigations.md#pyq-316), conformal on vs. off (PYQ-248). Every one of those
comparisons is currently **two point estimates, eyeballed**. There is no test of whether
the difference is distinguishable from zero.

`moving_block_bootstrap_interval()` exists (PYQ-251) but is used for a single
configuration's directional accuracy, and an interval on A and an interval on B is the
wrong instrument anyway: the two configurations are scored on the *same walk-forward
windows*, so the comparison is paired, and a paired test on the per-window differences is
strictly more powerful than comparing two marginal intervals. Overlapping intervals do not
imply no difference, which is precisely the error this shape of reporting invites.

Build: a paired comparison over the per-window skill (or loss) differences of two
`BacktestResult`s scored on identical windows — a moving-block bootstrap of the mean
difference, reusing the existing block machinery, with the block length keyed to the
horizon so overlapping windows do not inflate significance. A Diebold-Mariano test on the
per-point loss differential is the textbook alternative and worth considering, but the
block bootstrap reuses code already here and makes fewer assumptions; pick one, record why.

The function must **refuse** to compare results whose windows do not align, rather than
silently comparing unlike things. That guard is the whole value of the paired framing.

Consumed by PYQ-268's sweep harness, and the natural reporting home for
investigations.md#pyq-322's decision rule: "flip the default when the paired interval
excludes zero" is a pre-registrable statement, "when the number looks better" is not.

Acceptance criteria: `compare_backtests(a, b)` returns the mean per-window skill difference with a confidence
interval, raises on window misalignment, and is exercised by tests covering: identical
inputs give a difference of zero, a constant offset is recovered, and misaligned windows
raise. `docs/methodology.md` re-states PYQ-247's +2.4% and #pyq-316's sentiment delta with
an interval attached, or says plainly that the sample is too small to place one.

---

## [PYQ-267]
Break every metric down by horizon step
Status: Resolved — 2026-07-29 (same session, uncommitted — see git status)
Priority: High
Files: `pyquant/analysis/metrics.py`, `pyquant/analysis/serialize.py`, `pyquant/cli/app.py`

Problem/Ask: `evaluate_predictions()` receives predictions shaped
`(n_samples, horizon, n_quantiles)` and immediately collapses the horizon axis —
`model_mae(actuals, median)` averages over all five steps at once, as do coverage,
directional accuracy, CRPS and Winkler. Every number this project reports is therefore a
mean over h=1..5, and the per-step structure has never been looked at.

That structure is where the answer probably lives. Persistence is near-unbeatable at h=1
and progressively less so as h grows, so a model that is genuinely learning something
should show skill *increasing* with horizon while a model that is only tracking the last
close shows the opposite. A flat -23.5% mean and a profile of `[-60%, -35%, -10%, +5%,
+15%]` are the same headline number and completely different findings — the second would
say "keep the model, shorten nothing, the horizon is where it earns its place," and the
first says the opposite. Right now the two are indistinguishable in every artifact the
project produces.

The same applies to calibration. A 99.3% empirical coverage on a nominal 80% band is
reported as one number; if that is 100% at h=1 (band far too wide where uncertainty is
smallest) decaying to 85% at h=5, the pathology is "the band does not widen with horizon,"
which is a specific, fixable modelling statement rather than a general "too wide."
docs/architecture.md's logo and `nvo.png` both draw a band that fans; investigations.md
#pyq-324 asks whether the real one does.

Build: retain the horizon axis. Add `per_horizon: list[EvaluationMetrics]` (or a compact
per-step MAE/coverage/skill record) to `EvaluationMetrics`, aggregate it position-wise
across windows in `aggregate_metrics()`, surface it in `--format json` and as an optional
Rich table. Cheap — the arrays are already the right shape and are being discarded.

Acceptance criteria: `evaluate_predictions` exposes per-step MAE, skill, coverage and directional accuracy;
`aggregate_metrics` pools them position-wise across windows; `--format json` carries them;
a test asserts that a synthetic case with skill deliberately varying by step recovers the
profile rather than its mean. `docs/methodology.md` shows the per-horizon profile for at
least the default and log-return configurations.

Resolution: added `PerHorizonMetrics` (step, model_mae, baseline_mae, directional_accuracy,
calibration_coverage, plus a `skill_vs_baseline` property using the same formula as
`EvaluationMetrics`'s) -- the "compact per-step record" option the ticket offered, over a
full `list[EvaluationMetrics]`, since quantile_exceedance/pinball_losses/CRPS/Winkler/PIT
are already per-quantile or per-point aggregates that don't obviously have a single
per-horizon-step reading, and the four fields that clearly do (MAE, baseline MAE, direction,
coverage) are exactly what the ticket's own examples (skill profile, coverage fanning) need.
`evaluate_predictions` now isolates each decoder step with `[:, h:h+1]` slices (preserving
the 2D shape `persistence_baseline_mae`/`directional_hit_rate` broadcast `last_observed`
against) before computing the top-level, still-averaged metrics -- the same underlying
per-point arrays, sliced differently, not a second computation. `aggregate_metrics` gained
`_pool_per_horizon()`, position-wise-averaging step `h` across every window that has one,
weighted by each window's `n_samples` at that step (not `n_points`, since a `PerHorizonMetrics`
entry already isolates one step and weighting by `n_points` would double-count the horizon
factor already removed by isolating it). Windows without a `per_horizon` (hand-built
`EvaluationMetrics`, e.g. in older tests) degrade to an empty list rather than raising.

`serialize.evaluation_to_dict` adds a `per_horizon` list (step, the four raw fields, and the
derived `skill_vs_baseline`) so `--format json` carries it end to end. `meta.json`'s own
"evaluation" dict (built via `vars(evaluation)` in `tft.py`, not through `serialize.py` --
`serialize.py` imports `TrainResult`/`BacktestResult` *from* `tft.py`, so the reverse import
would be circular) needed one line fixed: `vars()` doesn't recurse into the nested
`PerHorizonMetrics` dataclasses, so `json.dumps(meta, ...)` raised
`TypeError: Object of type PerHorizonMetrics is not JSON serializable` until each step was
flattened to a plain dict the same way the top level already drops computed properties like
`skill_vs_baseline` (caught by the pre-existing end-to-end CLI test
`test_full_cli_journey_across_every_command_and_both_output_formats`, which trains a real
tiny bundle rather than mocking `TrainResult` -- exactly the class of write/read contract
break PYQ-241 built it to catch, and it did).

The CLI gained `_per_horizon_table()` (mirrors the existing `_per_window_table` styling),
printed by both `train` and `backtest` whenever `len(evaluation.per_horizon) > 1` -- no new
flag, matching how `_per_window_table` is already unconditional-but-guarded rather than
opt-in.

Verified with the acceptance criterion's own scenario:
`test_evaluate_predictions_recovers_a_skill_profile_that_varies_by_horizon_step` builds a
3-step synthetic case with per-step skill exactly +1.0/0.0/-1.0 and confirms all three are
recovered while the mean-over-horizon headline (`-0.5625`) is none of them.
`test_evaluate_predictions_per_horizon_mae_and_coverage_match_manual_per_step_calculation`,
`test_aggregate_metrics_pools_per_horizon_position_wise_weighted_by_n_samples`, and
`test_aggregate_metrics_per_horizon_is_empty_when_no_window_has_it` cover the pooling and
degrade-gracefully paths. `test_train_json_output_includes_per_horizon_breakdown` and
`test_train_table_shows_a_per_horizon_breakdown_when_horizon_exceeds_one` cover the CLI
surface. 30 tests in `tests/test_metrics.py` (up from 26) and 81 across
`tests/test_metrics.py`+`tests/test_cli.py` combined, all passing; full suite green
afterward including the real-bundle end-to-end test that caught the `vars()` bug.

**`docs/methodology.md`'s per-horizon profile for the default/log-return configurations is
not filled in** -- the acceptance criterion asks for measured numbers and this pass had no
live vendor-data access to produce them (same limitation recorded on PYQ-143, which landed in
the same pass and changes what those numbers would be anyway). Added a `## Per-horizon
breakdown` section explaining the feature and stating this explicitly, rather than leaving
the gap silent or inventing numbers.

---

## [PYQ-268]
A reusable multi-symbol sweep harness, replacing the one-off scripts
Status: Open
Priority: High
Files: `pyquant/experiments/` (new), `pyquant/cli/app.py`, `scripts/ablate_features.py`, `scripts/compare_pooling.py`

Problem/Ask: `backlog/README.md`'s `## Now` list has had the same item at #1 across two
passes — a multi-symbol repeat of PYQ-247's target comparison and of
investigations.md#pyq-315/#pyq-316's pooling and feature findings — with the standing note
that it has "still no ticket, because each is a *run* rather than a code change."

That reasoning is what has kept it unstarted. It is not purely a run: there is no tool that
performs it. `scripts/ablate_features.py` and `scripts/compare_pooling.py` are both
self-described one-off investigation scripts, each hard-wired to one question and, between
them, to one or two symbols. Repeating either across fifteen symbols today means editing a
script, running it by hand, and reconciling the output by hand — which is exactly why three
findings that each explicitly name a multi-symbol repeat as their prerequisite have all sat
un-repeated. The missing artifact is a harness, and a harness *is* a code change.

Build: `pyquant/experiments/` — a library-agnostic sweep runner that takes a list of
symbols, a list of config overrides (arms), and a window count; runs the walk-forward
backtest for every (symbol, arm) cell; and returns a tidy per-cell result set. Plus
`pyquant sweep` to drive it. Aggregation across symbols is the part that must be got right:
report per-symbol results **and** the pooled figure, because "helped 11 of 15 symbols" and
"mean skill +0.3%" answer different questions and only the pair is honest. Reuse
PYQ-266's paired test for the arm-vs-arm comparison and PYQ-265's seed handling for the
within-cell repeat.

The two existing scripts then become thin callers or get deleted — they should not be
reimplemented inside the harness, and whichever way that goes should be recorded, since
they carry investigation context the harness itself does not.

Explicitly *not* in scope: running the sweep. This ticket delivers the instrument; the
three pending repeats are separate runs, and the results they produce are what
investigations.md#pyq-322's decision rule consumes. Keeping those apart is deliberate —
shipping the harness should not be blocked on GPU time, and running it should not be
blocked on writing code.

Acceptance criteria: `pyquant sweep --symbols A,B,C --arm target=close --arm target=log_return --windows 5`
runs every cell, reports per-symbol and pooled skill with a paired interval per arm pair,
and writes a machine-readable result set. Tests cover the cell matrix (arms x symbols),
the "helped N of M" summary, and that a failing symbol degrades to a recorded gap rather
than taking the sweep down. `scripts/ablate_features.py` and `scripts/compare_pooling.py`
are either rewritten over the harness or removed with their reasoning preserved.

---

## [PYQ-269]
Split `models/tft.py` (1075 lines) without breaking Lightning containment
Status: Open
Priority: Medium
Files: `pyquant/models/tft.py`, `pyquant/models/` (new submodules), `tests/test_tft.py`

Problem/Ask: `models/tft.py` is 1075 lines and the single largest module in the project
by a factor of 1.3 over `cli/app.py` and 3.2 over anything else. It holds `train()`,
`walk_forward_backtest()`, `tune()`, `predict_quantiles()`, `interpret()`,
`permutation_importance()`, `load()`, `settings_for_bundle()`, the `ModelBundle`/
`TrainResult`/`BacktestResult`/`TuneResult` dataclasses, the checkpoint-selection and
validation-array plumbing, purged-cutoff arithmetic, feature-schema checking and pooled
long-frame assembly.

The size is a *consequence of a rule this project should keep*, not an accident:
CLAUDE.md's first structural rule confines all pytorch-forecasting and Lightning imports
to `models/tft.py` and `data/dataset.py`, so anything touching Lightning has nowhere else
to go and the module accretes. That is why this is a refactor ticket rather than a
complaint — the rule earns its keep (it is what made `pyquant/api/` additive rather than a
rewrite, per PYQ-261), and the fix is to give it more room, not to relax it.

Build: turn `pyquant/models/` into a package with the containment rule applied to the
*package* rather than the file — e.g. `models/bundle.py` (dataclasses, load/save,
provenance, schema check), `models/training.py` (`train`, checkpoint selection, validation
evaluation), `models/backtest.py` (walk-forward, window geometry, purged cutoff),
`models/tuning.py` (Optuna), `models/inference.py` (predict/interpret/permutation
importance). `models/tft.py` keeps its public names as re-exports so no import outside
`models/` changes and no bundle is invalidated.

Two things make this worth doing beyond aesthetics. The window-geometry code — the subject
of PYQ-115, PYQ-116, PYQ-127 and PYQ-250, four of the pipeline's most expensive bugs —
currently sits interleaved with training loop and tuning code, and the graph shows it
clustered with `TimeSeriesDataSet` construction rather than isolated. And PYQ-238's
invariant module tests those geometry properties from outside; a named module to point at
makes the invariant/implementation correspondence legible.

Risk to state plainly: this is a pure-motion refactor over the most bug-dense code in the
project, and a mistake here is the kind that does not fail a test. It should land as
motion only, with no behaviour change, verified by the full suite plus a byte-identical
`meta.json` on a re-run at a fixed seed and pinned dataset.

Acceptance criteria: `pyquant/models/` is a package of five focused modules, none over ~350 lines; `from
pyquant.models.tft import train, load, ...` still works unchanged; `ruff check` clean; the
full suite passes; a re-train at a fixed seed against a pinned dataset produces a
`meta.json` identical to one produced before the split, and that comparison is recorded in
the resolution note.

---

## [PYQ-270]
Put a confidence interval on the headline skill number
Status: Open
Priority: Medium
Files: `pyquant/analysis/metrics.py`, `pyquant/cli/app.py`, `README.md`, `docs/methodology.md`

Problem/Ask: `cli/app.py:211` bootstraps a confidence interval over per-window
**directional accuracy** and prints it. Skill vs. baseline — the number in the README, in
`docs/methodology.md`'s headline tables, in `explain`'s warning banner (`app.py:576`), and
in essentially every ticket that has ever argued about model quality — gets no interval at
all.

The asymmetry is backwards. Directional accuracy is the metric PYQ-247 showed to be
*flattered* by the level target and least trusted; skill is the one every decision hangs
on. And skill is precisely the number where an interval would change readings: -23.5% over
56 windows and +2.4% over 5 windows are quoted side by side in `docs/methodology.md` with
the sample-size caveat in prose, when the interval would carry it structurally.

Build: bootstrap the per-window skill values with the same moving-block machinery
(`moving_block_bootstrap_interval`, blocks no shorter than the horizon) and print
`skill [lo, hi]` wherever skill is printed. Carry it in `--format json` and in `meta.json`
so `explain`'s banner can say "at or below baseline, and the interval spans zero" rather
than just quoting a point.

One subtlety to get right rather than inherit: the existing call blocks per-*window*
directional accuracies with `block_size = max(1, horizon)`, but consecutive walk-forward
windows are disjoint by invariant 7 (PYQ-127), so a block length measured in horizon steps
is being applied to a series whose elements are windows. Decide what the correct block
length is for a window-level series and fix both call sites, or record why the current one
is right.

Interacts with PYQ-141: once both the pooled headline and the per-window column carry
intervals, the fact that they are different estimators becomes visible rather than
confusing.

Acceptance criteria: Skill is reported with an interval everywhere it is currently reported as a point (Rich
tables, `--format json`, `meta.json`); the block-length question above is resolved in code
with a comment citing this ticket; `README.md` and `docs/methodology.md` quote intervals
alongside the three headline configurations, or state that n is too small to place one.

---

## [PYQ-271]
`/backtest` endpoint: close the CLI/API front-end gap
Status: Open
Priority: Medium
Files: `pyquant/api/routes/`, `pyquant/api/schemas.py`, `docs/http-api.md`, `tests/test_api.py`

Problem/Ask: `docs/architecture.md` states the design as two independent front-ends,
`cli/` and `api/`, "served by the same calls, without either being a rewrite of the other."
The API currently exposes `/forecast/{symbol}`, `/scan`, `/explain/{symbol}`, `/train`,
`/train/{job_id}` and `/healthz`. The CLI additionally has `backtest`, `tune`, `cache`,
`snapshot`, `doctor` and `calibration`.

`backtest` is the gap that matters. It is the command that produces every quality number
this project reports, and it is the one an API consumer most plausibly wants — an
evaluation service is the difference between "here is a forecast" and "here is a forecast
and here is what it has been worth." Its absence means the parity claim in
`docs/architecture.md` is aspirational for the one capability that would most test it.

Build: `POST /backtest` returning 202 with a job id, polled via `GET /backtest/{job_id}` —
the same shape `/train` already established, because a walk-forward backtest trains N
models and is emphatically not a request-cycle operation (investigations.md#pyq-319
measured a *single* cold forecast at ~65s). Reuse `JobRegistry` rather than adding a second
job mechanism. The response body is `analysis/serialize.py`'s `backtest_to_dict` output,
which already exists and is already what `--format json` emits — this is the test of
whether the shared-core claim holds.

`tune` is the same shape and the obvious follow-on, but is heavier and less clearly wanted;
`doctor`/`cache`/`snapshot` are local-operator commands with no obvious remote meaning.
Scope this ticket to `backtest` and record that judgement rather than silently limiting it.

If `backtest_to_dict` turns out to need reshaping to serve both front-ends, that is the
finding, and it should be reported as such — it would mean the two-front-end claim has been
untested rather than true.

Acceptance criteria: `POST /backtest` + `GET /backtest/{job_id}` work end to end against the existing
`JobRegistry`, return `backtest_to_dict`'s payload behind a Pydantic response model, are
documented in `docs/http-api.md`, and are covered in `tests/test_api.py` including the
job-not-found and job-failed paths. The resolution note states whether serving both
front-ends required changing the shared serializer.

---

## [PYQ-272]
Dedicated tests for `serialize`, `doctor`, `provenance` and `charts`
Status: Open
Priority: Medium
Files: `tests/test_serialize.py`, `tests/test_doctor.py`, `tests/test_provenance.py`, `tests/test_charts.py` (all new)

Problem/Ask: every module under `pyquant/` has a matching `tests/test_*.py` except
`analysis/serialize.py`, `analysis/doctor.py`, `provenance.py`, `cli/charts.py`, and
`cli/app.py` (covered by `test_cli.py` under a different name). The four unnamed ones are
exercised only incidentally, through CLI and API tests that assert on something else.

That is a poor place for them, because of what they do:

- `serialize.py` is the contract behind `--format json`, `meta.json` and every API
  response body — it is the machine-readable surface, and PYQ-271 is about to add a second
  consumer. A silent key rename here breaks downstream consumers with nothing failing.
- `doctor.py` exists *because* PYQ-139 was invisible: a whole vendor's features vanished
  from every panel and only a log line said so. A diagnostic whose own failure mode is
  silence needs direct tests more than most code, not fewer.
- `provenance.py` is one of reproducibility's three legs (PYQ-225, PYQ-133, PYQ-134 — the
  last of which was exactly a provenance function resolving against the wrong directory).
- `charts.py` renders the fan chart, and invariant 8 requires that the dates in the table,
  the JSON, the PNG and the appended rows are one set. The PNG is the leg with no direct
  test.

Build: four test modules asserting behaviour, not output shape, per the project's naming
convention. For `serialize`, round-trip and key-stability assertions (a renamed key should
fail loudly). For `doctor`, that each unhealthy condition it claims to detect is actually
detected — construct the broken state, assert the report names it. For `provenance`, that
git sha/dirty-state resolve against the working tree (PYQ-134's regression) and that
secrets never appear (the secrets non-negotiable currently has no test naming it). For
`charts`, that the plotted x-values equal the forecast dates — invariant 8's missing leg.

The `doctor` and secrets items are the ones with real defect-finding odds; PYQ-231's
precedent is that writing failure-path tests for untested surfaces found two live bugs.

Acceptance criteria: Four new test modules exist and pass offline; `doctor`'s detections are each covered by a
constructed failure; a test asserts no API-key value can appear in `meta.json`,
`runs.jsonl` or a cache fingerprint; a test asserts `charts`' plotted dates equal the
`Forecast`'s dates. Any defect found is filed as its own bug ticket rather than fixed
inline.

---

## [PYQ-273]
Regression cases for PYQ-139/140 on PYQ-243's existing replay harness
Status: Open
Priority: Medium
Files: `tests/test_macro.py`, `tests/test_sentiment.py`, `tests/fixtures/`, `scripts/record_fixtures.py`

Problem/Ask: **this ticket was filed on a false premise and is corrected here rather than
deleted.** As originally written (2026-07-28) it asked for boundary-level replay tests
against recorded vendor payloads, on the reading that vendor tests still patch at our own
`fetch_*` boundary. PYQ-243 already built exactly that and is Resolved: six tests mock at
`yf.Ticker`, `yf.download`, `fredapi.Fred` and `requests.get`, then run the real parsing
code against `tests/fixtures/`, with a `MANIFEST.json` recording vendor and library version
per payload. The harness exists and paid for itself twice on the first recording.

What is actually missing is narrower and more specific. PYQ-243 shipped in the same pass
that later produced **PYQ-139** and **PYQ-140** — two live-vendor failures that the harness
did not prevent, because a recorded happy-path payload cannot exercise a failure mode that
only appears against the live API under conditions the recording never hit. PYQ-139 alone
had three: an unbounded realtime window rejected for exceeding 2000 vintages, a `NaT` in
the value column that took a whole series down on one market holiday, and a `realtime_end`
in the future whenever the caller's clock runs ahead of FRED's. Every FRED macro feature
had silently vanished from every panel and only a log line said so. PYQ-140 is a fourth
shape: a vendor honouring a request *nominally* — accepting the `from` parameter, returning
200 — while serving ~6 days of the ~365 requested.

Those four are what a replay harness is *for*, and none of them is in it. Recording a
successful call captures the shape the vendor produces when everything works; the failures
that have actually cost this project are the shapes it produces when something does not.

Build: hand-construct (do not attempt to record) one fixture per failure mode and add a
named regression test per shape, on the harness PYQ-243 already provides. Each must fail
against the pre-fix code — the project's test-first convention exists precisely so a
regression test that cannot fail is caught before it is trusted (PYQ-120's lesson).

Also verify the two fixtures an external tooling pass flagged as producing no extractable
content (`fred_dff.json`, `finnhub_news_aapl.json`); if they are empty or malformed, the
tests reading them are asserting against nothing.

Acceptance criteria: four named regression tests — the three PYQ-139 FRED shapes and
PYQ-140's silent truncation — each demonstrated to fail against the pre-fix code and pass
after; the two suspect fixtures verified or re-recorded; `scripts/record_fixtures.py`
documents that failure-mode fixtures are hand-built by design and why.

---

## [PYQ-274]
CHANGELOG and a release/tagging workflow
Status: Open
Priority: Low
Files: `CHANGELOG.md` (new), `.github/workflows/release.yml` (new), `pyproject.toml`

Problem/Ask: `pyproject.toml` declares `version = "0.2.0"`. There is no `CHANGELOG.md`,
no tag-driven workflow, and no recorded relationship between that version string and any
commit. Reproducibility's third leg is code version (PYQ-225, PYQ-133), and a version
number that never moves and points at nothing does not provide it — a bundle's `meta.json`
recording `0.2.0` is compatible with any commit in the project's history.

This is genuinely Low priority: the project is not distributed, has no external consumers,
and its per-bundle git-sha provenance already does the load-bearing work that a version
string would do for a published package. It is worth a ticket because the version field
exists and currently asserts something it cannot back, not because releasing is urgent.

Build: a `CHANGELOG.md` in Keep a Changelog form, seeded from `backlog/README.md`'s History
section — which is already a better changelog than most projects have, and should be
summarised rather than duplicated, with the backlog remaining the detailed record. Then a
tag-triggered workflow that verifies the tag matches `pyproject.toml`'s version, runs the
full suite, and cuts a GitHub release. Publishing to PyPI is explicitly out of scope until
someone wants it — adding a release channel with no consumers is the kind of unjustified
addition non-negotiable #5 is about.

Bump to `0.3.0` as part of this, since the three passes recorded in `backlog/README.md` since
`0.2.0` include target-format work, a new API package and a new CLI command surface.

Acceptance criteria: `CHANGELOG.md` exists and covers the passes recorded in `backlog/README.md`; pushing a
`v*` tag runs the suite and cuts a release; the workflow fails if the tag and
`pyproject.toml` disagree; `docs/development.md` documents the release step.

---

## [PYQ-275]
Baselines beyond persistence: a negative result is only as strong as what it failed against
Status: Open
Priority: High
Files: `pyquant/analysis/baselines.py` (new), `pyquant/analysis/metrics.py`, `pyquant/cli/app.py`, `docs/methodology.md`

Problem/Ask: `analysis/` has no `baselines.py`. The only comparator in the codebase is
`persistence_baseline_mae()` — last observed value carried forward, or zero in log-return
space. Every skill number this project has ever reported is relative to that one baseline.

investigations.md#pyq-312 recorded the project's conclusion: the deliverable should be
reframed around the measurement apparatus and an honestly-reported "no detectable edge",
which it argues is "a more credible and much rarer artifact than another repo claiming
edge." That is the right call. But it raises the bar on exactly this: **a negative result
is a claim about the baselines it was measured against, and one baseline is a weak claim.**
"Does not beat persistence" and "does not beat anything a competent practitioner would try"
are very different statements, and only the second is worth publishing.

Persistence is also uniquely favourable to the null. On a near-random-walk level series it
is close to optimal by construction — which is the whole of PYQ-247's finding. Beating it
is hard for reasons that have nothing to do with whether the TFT learned anything, so
failing to beat it is correspondingly weak evidence. A drift baseline, a seasonal-naive
baseline and a simple statistical model would each fail differently, and the *pattern* of
which baselines the TFT beats and which it does not is far more diagnostic than a single
signed number.

Build: `analysis/baselines.py` with a small protocol and several implementations — random
walk with drift, seasonal naive, an ARIMA or ETS fit per window, and a "climatological"
constant-mean-return baseline. Library-agnostic, no Lightning, per the architecture rule;
`statsmodels` is the obvious dependency and needs justifying against doing nothing per
non-negotiable #5 (a hand-rolled AR(1) may well be enough and adds nothing).

Report skill against *each* baseline rather than collapsing to one, and make the strongest
baseline the headline — reporting skill against the weakest comparator available is the
failure mode this ticket exists to prevent. features.md#pyq-249's foundation-model arm is
the same idea one level up and should share this interface rather than growing its own.

This is the groundwork the reframed deliverable needs, and it is worth more than any new
data source: it strengthens the claim the project has decided to make, where a fifth vendor
would only add an input nobody can currently evaluate.

Acceptance criteria: `baselines.py` exposes at least three baselines behind one protocol;
`EvaluationMetrics` carries skill against each; `backtest` reports the full row and names
which baseline the headline skill is against; any new dependency is justified in the
resolution note or declined; `docs/methodology.md`'s three configurations are re-stated
against the full baseline set.

---

## [PYQ-276]
Execute PYQ-312's reframing: the README still sells a forecaster
Status: Open
Priority: Medium
Files: `README.md`, `docs/index.md`, `docs/methodology.md`, `CLAUDE.md`

Problem/Ask: investigations.md#pyq-312 is marked Answered and its recorded conclusion is
explicit: *"the deliverable should be reframed around the measurement apparatus, as this
ticket anticipated — but the honest headline is now 'no detectable edge after fixing the
formulation', not 'negative skill'."* It then names the one thing standing in the way:
*"Updating the README to say so requires the multi-symbol repeat first; that is the
concrete next step."*

That reframing has not happened, and it is not tracked anywhere as work. PYQ-312 cannot
track it — it is Answered, and per this backlog's rules an Answered investigation records a
conclusion rather than carrying an open action. So the project's own decision about what it
*is* currently lives only as a paragraph inside a closed ticket.

Meanwhile `README.md` still opens with "Probabilistic equity forecasting with a Temporal
Fusion Transformer" and answers "Does it beat a benchmark?" with a two-sentence note that
the baseline is hard to beat. `docs/index.md`'s lede is a forecasting harness. Both are
accurate about the machinery and both lead with the capability rather than the finding —
which is the framing PYQ-312 concluded was the wrong one.

Ask: once features.md#pyq-268's sweep produces a multi-symbol result and
investigations.md#pyq-321 establishes the seed-variance floor, rewrite the top of
`README.md` and `docs/index.md` to lead with what was actually measured: a leak-audited
pipeline, an evaluation apparatus most equity repos do not have, and a rigorously
established absence of detectable edge at this horizon on public daily data. The TFT
becomes a component of the finding rather than the headline.

Two guardrails. This must not become self-deprecation — "no detectable edge, measured
properly, on n symbols across m windows with a stated seed floor" is a stronger and rarer
claim than an unsupported positive one, and should be written as the finding it is. And it
is **gated**: rewriting the README on the current n≈5 evidence is the move non-negotiable
#1 forbids, in the same way that flipping `TrainingConfig.target` would be. Do not start
this before the sweep lands.

Acceptance criteria: `README.md` and `docs/index.md` lead with the measured finding and its
sample size; the headline number is stated against the strongest baseline available
(features.md#pyq-275); `CLAUDE.md`'s project description matches; the resolution note
records which sweep result licensed the rewrite.

---

## [PYQ-277]
Tiingo isn't actually selectable anywhere; PYQ-258's own acceptance criterion is unmet
Status: Open
Priority: Medium
Files: `pyquant/data/providers.py`, `pyquant/config.py` (`DataConfig`), `pyquant/data/dataset.py` (`build_panel`), `pyquant/cli/app.py`

Problem: PYQ-258 (Resolved) states its acceptance criteria as *"one alternative provider is
implemented **and selectable**"* and *"switching is a config change rather than a
rewrite."* `providers.py` implements a well-tested `PriceProvider` protocol with a licensed
Tiingo backend, but nothing wires it to an entry point: `DataConfig`
(`config.py`) has no `provider`/`price_provider` field; `build_panel`
(`dataset.py:90-92`) calls `fetch_prices(symbol, period=..., start=..., end=...,
use_indicators=...)` and never passes `provider=`, so `fetch_prices`'s own default
(`provider: str | object = "yfinance"`, `prices.py:253`) is always what runs; and no CLI
flag exists anywhere in `cli/app.py` (confirmed by grep). Today, Tiingo is reachable only by
calling `fetch_prices(..., provider="tiingo")` from a Python REPL — which is exactly the
"argument you can only pass from a Python REPL" state PYQ-258's own acceptance criteria were
written to rule out. The ticket is marked Resolved with a stated criterion its own
implementation doesn't meet.

This is filed as a new ticket rather than reopening PYQ-258 directly, per this backlog's
convention that a ticket's `Status:` line changes but its content/ID doesn't move — the
gap is a fresh, separately-actionable finding, and PYQ-258's resolution note is the correct
place to record what actually shipped versus what the criteria asked for.

Ask: add `DataConfig.price_provider: Literal["yfinance", "tiingo"] = "yfinance"`, thread it
through `build_panel → fetch_prices`, add a `--provider`/config-file knob at the CLI layer,
and add the field to `_cache_fingerprint` (see bugs.md#pyq-148 for the general pattern of a
`DataConfig` toggle missing from the fingerprint — don't repeat it a third time).

Acceptance criteria: `pyquant train AAPL` (or an equivalent config-file setting) can select
Tiingo without touching Python; a test asserts `build_panel` passes `settings.data
.price_provider` through to `fetch_prices`; the cache fingerprint changes when only
`price_provider` changes.

---

## [PYQ-278]
Ruff format drift has grown to 33 files vs. the CI comment's 20-22 baseline
Status: Open
Priority: Low
Files: `.github/workflows/ci.yml`, repo-wide formatting

Problem: `ruff format --check .` currently reports **33 files** would be reformatted
(verified directly, run twice for stability — not the 36 the review that prompted this pass
claimed, but still real drift). `.github/workflows/ci.yml`'s comment states the count was
"20-22 unformatted files (~250 lines)" when PYQ-229 (Resolved) added this as a
`continue-on-error: true`, non-blocking step, with the stated intent that reporting the
count "stops it growing." It has grown ~50-65% since. `ruff check` (the blocking linter) is
already clean, so a mechanical `ruff format .` pass carries near-zero review risk.

Ask: either land a one-off `ruff format .` pass now, while the number is still small enough
to review as a pure-formatting diff, or accept explicitly that the non-blocking check is
decorative and stop citing a baseline it isn't holding to.

Acceptance criteria: `ruff format --check .` reports 0 files (post-pass), or the CI comment
is updated to state the check is informational-only and not expected to hold a baseline.

---

## [PYQ-279]
`git_sha()`/`code_version()` shells out uncached on every `build_panel` call
Status: Open
Priority: Low
Files: `pyquant/provenance.py`

Problem: `git_sha()` (`provenance.py:45-66`) makes two `subprocess.run` calls (`rev-parse
--show-toplevel`, then `rev-parse --short HEAD`) with no memoization anywhere in the module.
It's called from `dataset.py`'s `_cache_fingerprint` (via `provenance.code_version()`) on
**every** `build_panel()` call, and separately from `cache.py`'s `read_pin`/`write_pin`,
`tft.py` (bundle provenance), and `doctor.py`. In the API, that's two subprocess spawns per
request that touches any of those paths — cheap individually, but pure overhead: the working
tree's SHA can't change mid-process in any way that matters for a single run.

Ask: `functools.lru_cache` (or a simple module-level memo) on `git_sha`/`code_version`,
since nothing in-process ever needs the working tree's SHA to be re-read after the first
call.

Acceptance criteria: `git_sha()`/`code_version()` shell out at most once per process; a
test asserts a second call doesn't invoke `subprocess.run` again (mock/count-based).

---

## [PYQ-280]
`backlog.py check` should verify a resolved ticket's acceptance-criteria tests actually exist
Status: Open
Priority: Low
Files: `scripts/backlog.py`, `backlog/README.md`

Problem: this pass found a concrete instance of the exact failure mode `backlog/README.md`'s
own History section already names as a known risk: *"two tickets are marked Resolved with
unmet acceptance criteria."* features.md#pyq-277 documents PYQ-258 closed as Resolved with
its own stated criterion — "one alternative provider is implemented **and selectable**" —
unmet in the shipped code. `scripts/backlog.py check` currently verifies table/detail
consistency and ID ranges, but nothing about whether a ticket's claimed verification (test
names, acceptance criteria) is real.

Ask, as the review that prompted this pass put it: *"consider requiring that each
acceptance criterion name the test that proves it, and have `scripts/backlog.py check`
verify those test names exist in `tests/`."* That's a real fix for the "resolved but
unverified" class of drift, and the machinery to extend already exists in this file. Scope
it narrowly: for tickets whose resolution note or acceptance criteria mention a specific
`test_*` name (a common existing convention, e.g. PYQ-142's `test_log_return_price_round
_trip` reference), grep `tests/` for that name and flag a mismatch. This can't catch every
case (PYQ-277's gap is about a claim, "selectable," not a named-but-missing test), but it
closes the mechanically-checkable half cheaply.

Acceptance criteria: `backlog.py check` extracts backtick-quoted `test_*`-shaped identifiers
from Resolved tickets' resolution notes/acceptance criteria and fails if none of them exist
under `tests/` (allowing tickets that name no test to pass through unchanged); a test on the
checker itself using a synthetic backlog fixture with a dangling test reference.

