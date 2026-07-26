# Investigations (PYQ-3xx)

Open questions that need reasoning/experimentation before they become a bug
or feature ticket (or get answered and closed as-is) — see
[`README.md`](README.md) for the format.
Next free ID: **PYQ-312**.

| ID | Priority | Status | Title |
|----|----------|--------|-------|
| [PYQ-301](#pyq-301) | Medium | Open | How much of the training window actually has non-neutral sentiment? |
| [PYQ-302](#pyq-302) | High | Answered | Schema drift between train-time and predict-time panels |
| [PYQ-303](#pyq-303) | High | Superseded | Is a single 5-day validation window statistically reliable for model selection? |
| [PYQ-304](#pyq-304) | Medium | Resolved | Re-run full test suite + coverage with the complete ML stack installed |
| [PYQ-305](#pyq-305) | Medium | Resolved | Establish a documented publication-lag convention for future macro/fundamental sources |
| [PYQ-306](#pyq-306) | Low | Answered | Confirm whether `weights_only=False` is actually required for `dataset_params.pt` |
| [PYQ-307](#pyq-307) | High | Answered | Why is the default training config not beating a do-nothing baseline? |
| [PYQ-308](#pyq-308) | Medium | Answered | FinBERT/Finnhub sentiment scoring path has zero CI coverage |
| [PYQ-309](#pyq-309) | Low | Resolved | No LICENSE file despite `pyproject.toml` declaring MIT |
| [PYQ-310](#pyq-310) | Low | Answered | Would type-checking (mypy/pyright) catch anything real, cheaply? |
| [PYQ-311](#pyq-311) | Low | Resolved | Should `scripts/backlog.py check` run in CI? |

---

## [PYQ-301]
How much of the training window actually has non-neutral sentiment?
Status: Open
Priority: Medium
Files: `pyquant/data/sentiment.py`, `pyquant/config.py` (`DataConfig.period` default `5y`)

Question: `sentiment.py`'s own docstring discloses that Finnhub's free tier
only covers ~365 days of news, and everything older gets neutral
(Sentiment=0, HeadlineCount=0) after the join. With the default
`period="5y"`, roughly 80% of training rows could be structurally zero for
this feature. Does the TFT end up learning to ignore Sentiment, or learning
"0 is normal" in a way that doesn't transfer to live use (where sentiment is
always populated)? Worth checking learned feature importance for Sentiment
via `explain`, and considering a companion `has_sentiment_data` indicator
column or truncating the effective training window to sentiment
availability. See features.md#pyq-214 and #pyq-308 for related
data-provider and test-coverage angles.

---

## [PYQ-302]
Schema drift between train-time and predict-time panels
Status: Answered — 2026-07-26
Priority: High
Files: `pyquant/data/dataset.py` (`build_panel`), `pyquant/models/tft.py` (`_prediction_dataset`)

Question: `build_panel()`'s per-source graceful degradation means the exact
column set depends on which optional sources succeeded on that specific
call. `TimeSeriesDataSet.from_parameters()` (used at predict/explain time)
expects the same reals the model was trained with. If, say, Finnhub is
rate-limited only at prediction time — after training succeeded with
sentiment enabled — reasoning through the code this looks likely to error or
misbehave. Worth a deliberate test: train with a source enabled, then force
that source to fail at predict time, and see what actually happens.

Answer (2026-07-26): ran the deliberate test both directions. Results:

- **extra** columns at predict time (trained lean, predicted rich):
  `from_parameters` silently ignores them and returns a normal forecast.
  Benign.
- **missing** column at predict time (trained rich, predicted lean):
  `KeyError: 'SEC_SPY'`, raised from inside pytorch-forecasting with no
  indication of which source vanished or why.

So the suspicion was correct and the failure mode is the worse of the two
possibilities: a hard crash with an opaque message. Two things follow, both
filed as bugs:

- bugs.md#pyq-118 — validate `meta["features"]` against the rebuilt panel and
  raise one clear, actionable error instead of the bare `KeyError`.
- bugs.md#pyq-119 — the *reason* this is easy to hit: `forecast`/`explain`/
  `scan` never see the config the bundle was trained with, so they rebuild the
  panel from defaults. Train with `--no-sectors` and forecast without it and
  the schemas differ by construction, not by bad luck.

Closing as answered; the remediation lives in those two tickets. This also
unblocks PYQ-213 — the API cannot be trusted against live data until PYQ-118
lands, which was the concern that made this High.

---

## [PYQ-303]
Is a single 5-day validation window statistically reliable for model selection?
Status: Superseded by PYQ-117 — 2026-07-26
Priority: High (raised from Medium — see update)
Files: `pyquant/models/tft.py` (`train` — `training_cutoff`, `EarlyStopping`, `ModelCheckpoint`)

Question: ties to features.md#pyq-202 (now implemented as `backtest`).
`best_model_score`, `EarlyStopping`, and `ModelCheckpoint` all key off one
held-out window per symbol. Is that score meaningfully noisy run-to-run
depending on which particular week happens to land in validation? Worth
training the same config twice (different random seeds) and comparing
val_loss variance before trusting it for early stopping / hyperparameter
comparisons.

Update (2026-07-24): now has real supporting evidence, not just reasoning. A
live `pyquant train AAPL` run (default config, no API keys, 25 features,
early-stopped at epoch 10) reported: Model MAE 16.94 vs. persistence
baseline MAE 7.40 (skill −128.9%), directional accuracy 100.0%, calibration
coverage (p10–p90) 0.0%. 100% directional accuracy from 5 points is exactly
the kind of small-sample noise this ticket describes. (0% calibration
coverage and the large MAE deficit turned out to have a different, now
fully-explained cause — see #pyq-307/bugs.md#pyq-109 — but the sample-size
question this ticket asks is still open and separate.) features.md#pyq-210's
`seed_everything` is a prerequisite for running the seed-variance comparison
this ticket asks for, and should be re-run after bugs.md#pyq-109 lands so
the comparison isn't itself measuring the wrong checkpoint.

Superseded (2026-07-26) by bugs.md#pyq-117. The question does not need the
seed-variance experiment to be answered: the validation set was measured and
contains exactly **one** sample per group (5 points at the default horizon),
because `predict=True` yields one window and `cutoff = max_idx - horizon`
leaves a holdout exactly one horizon long. A 5-point sample is not a
reliable basis for early stopping or model selection, and no amount of
seed-variance measurement would make it one.

Reframing this as a bug rather than a question matters: the numbers derived
from those 5 points ship to the `train` table as `Directional accuracy 100.0%`
and `Calibration coverage 100.0%`, with no denominator shown. That is a defect
in what the tool reports, not an open research question — so remediation
(a real holdout span, plus reporting the sample size) moved to PYQ-117.

Two adjacent defects surfaced while measuring this and are filed separately:
bugs.md#pyq-127 (every backtest origin evaluates the *same* final window, so
"5 windows" was 5 models scored on the same 5 days) and bugs.md#pyq-116
(pooled training leaks a shorter symbol's validation window into training).
Together with PYQ-117 those three were the reason validation numbers here
never looked trustworthy.

---

## [PYQ-304]
Re-run full test suite + coverage with the complete ML stack installed
Status: Resolved — commit b616184, 2026-07-23
Priority: Medium

Problem: the original review could only install/run the torch-independent
test files (17/17 passed, with coverage); `test_dataset.py`, `test_tft.py`,
`test_cli.py`, and `test_forecast.py` were read but not executed
(installing torch+lightning+pytorch-forecasting exceeded the review
sandbox's disk quota).

Resolution: confirmed — the full suite (88 tests as of commit b616184, 90 as
of 2026-07-24) passes with the complete ML stack installed, and this
session ran it directly multiple times.

---

## [PYQ-305]
Establish a documented publication-lag convention for future macro/fundamental sources
Status: Resolved — commit b616184, 2026-07-23
Priority: Medium
Related: bugs.md#pyq-101

Question: as more sources get added, how should each new slow-moving series
declare its real-world publication lag so the PYQ-101 bug class doesn't
recur per-series?

Resolution: `macro.py`'s `_FredSeriesSpec(column, publication_lag_days)`
convention, applied uniformly across `FRED_SERIES` and consumed by
`_fetch_fred`.

---

## [PYQ-306]
Confirm whether `weights_only=False` is actually required for `dataset_params.pt`
Status: Answered — 2026-07-24
Priority: Low
Files: `pyquant/models/tft.py` (`load`)

Question: `TimeSeriesDataSet.get_parameters()` likely serializes
pytorch-forecasting objects (normalizers/encoders) that may not be on
PyTorch's safe-unpickling allowlist. Confirm whether `weights_only=True`
works; if not, document that `dataset_params.pt`/`model.ckpt` should only
ever be loaded from your own trusted training runs, since
`weights_only=False` deserialization can execute arbitrary code if the
file's provenance is ever untrusted. The same trust boundary is now also
relied on explicitly in `pyquant/data/cache.py`'s pickle-based panel cache
(see its module docstring) — worth confirming that reasoning together.

Answer (2026-07-24): confirmed empirically — `torch.load(dataset_params.pt,
weights_only=True)` raises `UnpicklingError` (the pytorch-forecasting
normalizers/encoders are not on PyTorch's safe-unpickling allowlist), so
`weights_only=False` is genuinely required, not incidental. Documented the
trust boundary in a comment at the `tft.load()` load site: bundles must only
ever be loaded from your own trusted training runs, matching the identical
boundary already documented for `pyquant/data/cache.py`'s pickle panel cache.
No code change beyond the doc comment; the requirement is inherent to the
serialization format.

---

## [PYQ-307]
Why is the default training config not beating a do-nothing baseline?
Status: Answered — 2026-07-24; superseded by bugs.md#pyq-109
Priority: High (historical -- no further action needed here directly, see bugs.md#pyq-109)
Files: `pyquant/models/tft.py` (`train`), `pyquant/config.py` (`TrainingConfig` defaults)

Original question: from a real `pyquant train AAPL` run (defaults:
hidden_size=32, lr=0.01, max_epochs=30, early-stopped at epoch 10, no API
keys so macro/sentiment were off, 25 features): Model MAE 16.94 vs.
persistence baseline MAE 7.40 (skill −128.9%), calibration coverage
(p10–p90) 0.0%, directional accuracy 100.0%.

Answer (2026-07-24): primarily bugs.md#pyq-109, not undertraining or LR. A
controlled comparison (same trained run, same validation data) evaluating
the live post-fit model vs. the actual best checkpoint that gets saved to
disk found: FINAL (what `train()` currently reports) scored skill −30.5%,
calibration 20.0%; BEST (what `forecast`/`explain` actually load and use)
scored skill **+15.9%**, calibration **100.0%**. i.e. the deployed model is
genuinely decent; the number shown to the user was measuring a different,
worse, already-discarded checkpoint. Two secondary experiments run before
this was understood (more epochs, and a lower learning rate) both made the
*reported* metric worse — consistent with PYQ-109 (more epochs under
EarlyStopping means more room to drift further from the best epoch before
patience triggers) rather than with an LR/capacity problem. See
bugs.md#pyq-109 for the fix and features.md#pyq-211 for the now-downgraded
LR-tuning ticket. #pyq-303's small-sample-size concern is still valid and
separate — re-run its seed-variance comparison after PYQ-109 lands.

---

## [PYQ-308]
FinBERT/Finnhub sentiment scoring path has zero CI coverage
Status: Answered — 2026-07-24
Priority: Medium
Files: `tests/test_sentiment.py`, `.github/workflows/ci.yml`

Question: every test in `test_sentiment.py` monkeypatches `_finbert()` and
`score_headlines()` — the real HuggingFace pipeline construction and
inference is never exercised. CI's `uv sync --extra dev` never installs the
`sentiment` extra either, so the actual
`pipeline("text-classification", model="ProsusAI/finbert", ...)` call path
(model download, tokenization, label mapping in `_signed_score`) has no
automated coverage. A break in that real integration (a `transformers` API
change, the model being renamed/removed from the Hub) would pass CI
silently and only surface live as silent "no sentiment features"
degradation. Worth deciding between a slow/optional CI job that installs
`--extra sentiment` and runs one real FinBERT call, vs. an offline fixture
that at least replays a recorded pipeline output through `_signed_score`.

Answer (2026-07-24, decision: offline fixture): added
`test_score_headlines_maps_finbert_pipeline_output_offline`, which primes the
module-level pipeline cache with a fake pipeline returning recorded
FinBERT-shaped output and asserts the `_finbert()` → `score_headlines` →
`_signed_score` path maps/aggregates it correctly — coverage for the
label-mapping logic with no model download. A slow real-FinBERT CI job
(installing `--extra sentiment` and calling the live model) was decided
against: the real `pipeline(...)` construction degrades gracefully if the model
is renamed/removed, and bugs.md#pyq-114 now prevents a transient failure from
poisoning the cache — so the marginal CI cost isn't justified. The only
remaining uncovered line is the model download itself, by design. `ci.yml`
unchanged.

---

## [PYQ-309]
No LICENSE file despite `pyproject.toml` declaring MIT
Status: Resolved — 2026-07-24
Priority: Low
Files: repo root

Question: `pyproject.toml` sets `license = { text = "MIT" }` and the README
makes no license statement at all, but there is no `LICENSE`/`LICENSE.txt`
file in the repo. Confirm MIT is actually intended and add the file — until
then the declared license isn't really in effect for anyone trying to rely
on it.

Resolution: MIT confirmed as intended (matches `pyproject.toml`); added a
standard `LICENSE` file at the repo root (MIT, "Copyright (c) 2026 Axel",
matching the `pyproject.toml` author). The declared license is now actually in
effect.

---

## [PYQ-310]
Would type-checking (mypy/pyright) catch anything real, cheaply?
Status: Answered — 2026-07-24
Priority: Low
Files: `pyproject.toml`, `.github/workflows/ci.yml`

Question: the codebase is fully type-hinted (`from __future__ import
annotations` in every module, typed dataclasses/pydantic models throughout)
but nothing checks the annotations are internally consistent — only ruff
(lint) runs in CI, no mypy/pyright, no pre-commit config, and (separately)
no coverage tooling (pytest-cov) is in the dev dependencies despite past
reviews having used coverage manually. Worth a one-off local mypy run to see
how many real issues surface before deciding whether it's worth the
CI-time cost of enforcing permanently, and whether pytest-cov + a coverage
gate belong in the same pass.

Answer (2026-07-24, decision: local-only, no CI gate): ran `mypy` at three
settings. Default (no config) → 40 errors, **all** missing-stub noise
(`import-untyped`/`import-not-found` for torch/pandas/yfinance/etc.), zero real
issues. With `ignore_missing_imports` → **0 errors**: the annotations are
internally consistent. With `--disallow-untyped-defs` → 18 findings, **all**
`no-untyped-def` (missing annotations on internal helpers), i.e.
completeness nits, not correctness bugs. Conclusion: type-checking catches
nothing real here today, so it's not worth a blocking CI step. Added a
`[tool.mypy]` section (`ignore_missing_imports = true`) so a local
`uvx mypy pyquant` runs clean and repeatable, with a comment recording this
decision. The pytest-cov/coverage-gate half of the question is left as a
separate concern (not adopted in this pass).

---

## [PYQ-311]
Should `scripts/backlog.py check` run in CI?
Status: Resolved — 2026-07-24
Priority: Low
Files: `.github/workflows/ci.yml`, `scripts/backlog.py`

Question: the backlog restructure (2026-07-24) added `scripts/backlog.py
check` to catch duplicate IDs, out-of-range IDs, and table/detail drift
across `backlog/*.md` before it accumulates unnoticed — exactly the kind of
manual-bookkeeping error the restructure was meant to fix. It currently has
to be run by hand. Is it worth a CI step (fast, dependency-free, would catch
a bad backlog edit in the same PR that introduced it) or is that overkill
for a backlog that's edited far less often than code? Leaning toward "yes,
it's nearly free" but filed as a question rather than done outright since it
touches `ci.yml`, which affects every PR, for a tool that's one session old
and hasn't proven itself yet.

Resolution (decided: yes): added a "Backlog consistency" step to
`.github/workflows/ci.yml` running `python3 scripts/backlog.py check` (no venv
or extra deps needed) between Lint and Test, so a bad backlog edit fails in the
same PR that introduced it. Confirmed the check passes on the current tree.
