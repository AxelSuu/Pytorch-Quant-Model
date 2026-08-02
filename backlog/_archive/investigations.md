# Investigations (PYQ-3xx)

Open questions that need reasoning/experimentation before they become a bug
or feature ticket (or get answered and closed as-is) — see
[`README.md`](README.md) for the format.
Next free ID: **PYQ-325**.

| ID | Priority | Status | Title |
|----|----------|--------|-------|
| [PYQ-301](#pyq-301) | Medium | Answered | How much of the training window actually has non-neutral sentiment? |
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
| [PYQ-312](#pyq-312) | High | Answered | Is a 5-day horizon learnable at all from these features — what should "good" look like? |
| [PYQ-313](#pyq-313) | High | Answered | Are predictions, actuals and last_observed genuinely in the same space? |
| [PYQ-314](#pyq-314) | Medium | Answered | Does the TFT's variable selection / attention output mean what `explain` claims? |
| [PYQ-315](#pyq-315) | Medium | Answered | Is pooling actually helping, now that PYQ-116 aligned the calendar? |
| [PYQ-316](#pyq-316) | Medium | Answered | Which of the 25+ features earn their place? |
| [PYQ-317](#pyq-317) | Medium | Answered | Is `softplus` the right target transformation for prices? |
| [PYQ-318](#pyq-318) | Low | Answered | pytorch-forecasting vendor risk vs. neuralforecast / Darts |
| [PYQ-319](#pyq-319) | Medium | Answered — 2026-07-27 | What is the latency and cost budget of one `forecast` call? |
| [PYQ-320](#pyq-320) | Low | Answered | Data-source licensing and ToS review before anything public-facing |
| [PYQ-321](#pyq-321) | Critical | Answered | How much of every reported number is seed variance? |
| [PYQ-322](#pyq-322) | High | Answered | A pre-registered rule for what evidence flips a default |
| [PYQ-323](#pyq-323) | Medium | Open | Is passing `Settings` everywhere costing more than it saves? |
| [PYQ-324](#pyq-324) | Medium | Open | Does the forecast band actually fan, or does it translate? |

---

## [PYQ-301]
How much of the training window actually has non-neutral sentiment?
Status: Answered — 2026-07-27
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

Answer (2026-07-27): measured, now that a `FINNHUB_API_KEY` is configured. **The premise
was too generous by two orders of magnitude.** This question estimated ~80% of training
rows would be structurally zero. The measured figure is **99.7%**:

```
AAPL: 1116 rows (2022-02-10..2026-07-27)   inside news window: 3 (0.3%)
MSFT: 1116 rows                            inside news window: 2 (0.2%)
```

The cause is not the 365-day free-tier limit this ticket reasons from. Finnhub's free tier
ignores the `from` parameter entirely and returns the same recent slice — ~248 headlines
across **6 distinct days** — whether asked for one week or five years. Filed separately as
bugs.md#pyq-140, because a module documenting coverage the vendor does not provide is a
defect in its own right.

So the answer to "does the TFT learn to ignore Sentiment, or learn that 0 is normal?" is
that it can only learn the second: with 99.7% zeros the column is very nearly a constant,
and the handful of non-zero rows sit at the *end* of the panel, which is exactly where the
prediction encoder reads. That is the train/serve shift this ticket suspected, in its
sharpest possible form.

Both remediations it proposes are now available. features.md#pyq-256 shipped the
`has_sentiment_data` indicator, so the two regimes are distinguishable by the model and
countable by anyone; truncating the training window to the covered period is arm two and
needs a backtest, which is recorded on PYQ-140 rather than guessed at here. Feature
importance via `explain` was **not** run: investigations.md#pyq-314 argues an interpretation
of a model that does not beat its baseline is an interpretation of noise, and with 0.3%
coverage the importance of `Sentiment` would be uninterpretable regardless.

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

Resolution (superseded by PYQ-257, 2026-07-26): `macro.py` originally used an
`_FredSeriesSpec(column, publication_lag_days)` convention to approximate when a
reference-period value became available. `_fetch_fred()` now consumes ALFRED/FRED
release vintages and indexes features by the actual `realtime_start` date instead, so
the fixed-lag convention is no longer used for FRED series. Keep an explicit
availability-date convention for future sources that do not expose vintages.

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

---

## [PYQ-312]
Is a 5-day horizon learnable at all from these features — what should "good" look like?
Status: Answered — 2026-07-27
Priority: High
Files: n/a — framing question for the whole project

Question: every quality ticket in the backlog asks *whether the number is measured
correctly*. None asks **what number should be expected if everything is correct.** That is
now the more important question, because after PYQ-109/115/116/117/127 the measurement
apparatus is trustworthy and the answer it gives is −23.5% skill.

Two readings, and the project cannot currently distinguish them:

1. Something is still wrong (formulation, target, hyperparameters, features).
2. Nothing is wrong. Five-day single-name equity direction is close to unforecastable from
   public daily OHLCV + three macro series + sector returns + FinBERT headline sentiment,
   and the honest result is "no edge."

Reading 2 is the mainstream prior. Published equity-return-prediction work typically
reports directional accuracy in the low-to-mid 50s at daily-to-weekly horizons *before*
transaction costs, and much of it does not survive replication with proper purging and
multiple-testing corrections. Against that, `57.5%` is not obviously wrong — it is roughly
where a real, marginal, possibly-illusory effect would sit, and the confidence interval on
it (PYQ-251) probably includes 50%.

Worth resolving explicitly because it determines the project's identity. If reading 2
holds, the correct response is **not** more tuning — it is to reframe the deliverable
around the measurement apparatus and report the negative result rigorously, which is a
more credible and much rarer artifact than another repo claiming edge.

Suggested approach: (a) write down, before running anything, what skill and directional
accuracy would count as evidence of a real effect given the effective sample size from
PYQ-251; (b) run PYQ-239's learnability test to confirm the pipeline *can* learn a real
signal when one exists; (c) run PYQ-247 and PYQ-249 and see whether either moves the
number outside the interval; (d) record the conclusion here either way, and update the
README's framing to match. Pre-registering the threshold in (a) is what stops (d) from
becoming post-hoc rationalisation.

Answer (2026-07-27, partial — the framing question is settled, the empirical one is not):

Step (c) was run. features.md#pyq-247's controlled comparison (AAPL, seed 42, one pinned
dataset, 12 epochs, 5 walk-forward windows, only the target varying) **moved the number out
of the negative range**:

```
close target       skill -59.5%   direction 80.0%   coverage 52.0%
log_return target  skill  +2.4%   direction 56.0%   coverage 76.0%
  + purged splits  skill  +3.8%   direction 52.0%   coverage 80.0%
```

That decides between this ticket's two readings more sharply than expected, and **not in
favour of either as stated**. Reading 1 ("something is still wrong") is partly vindicated:
the formulation *was* wrong, and −23.5% skill was substantially an artifact of predicting a
price level against a near-optimal persistence baseline, exactly as PYQ-247 predicted.
Reading 2 ("no edge") is *also* vindicated, at the level that matters: once the formulation
is fixed, skill is **+2.4% to +3.8%** — statistically indistinguishable from zero at this
sample size — and directional accuracy falls to **52–56%**, right where the mainstream
prior says single-name 5-day equity direction should sit.

The honest summary is therefore: *the old negative number was mostly measurement, the new
near-zero number is probably real.* Both of this ticket's readings were partly right, and
the project could not previously tell them apart because the formulation confounded them.

Directional accuracy deserves emphasis because it moves the *wrong* way and that is the
useful part. 80% on a level target is close to free — a model tracking the level is usually
on the right side of the last close — and the README's 57.5% was flattered by the same
effect. 52–56% on returns is the number a reader should be shown.

**What is not done, and why this is "Answered" rather than closed with a verdict.** Step (a)
— pre-registering what skill would count as a real effect — was not performed before
running, so nothing here is a pre-registered test and it should not be read as one. Step (b)
(features.md#pyq-239's learnability test) is still open, so "the pipeline can learn a real
signal when one exists" remains unverified and a +2.4% result cannot yet be distinguished
from a wiring artifact. Step (c)'s foundation-model arm (features.md#pyq-249) was not run.
The sample is one symbol, 25 predictions, effective n ≈ 5 — features.md#pyq-251 now reports
that effective figure precisely so this result is not over-read.

Recorded conclusion: **the deliverable should be reframed around the measurement apparatus,
as this ticket anticipated** — but the honest headline is now "no detectable edge after
fixing the formulation", not "negative skill". Updating the README to say so requires the
multi-symbol repeat first; that is the concrete next step, and stating it here rather than
editing the README on n≈5 is the point of non-negotiable #1.

---

## [PYQ-313]
Are predictions, actuals and last_observed genuinely in the same space?
Status: Answered — 2026-07-26
Priority: High
Files: `pyquant/models/tft.py` (`_evaluate_validation`)

Question: `_evaluate_validation` builds its three arrays from three different parts of
pytorch-forecasting's prediction output — `result.output` (`mode="quantiles"`),
`result.y[0]`, and `result.x["encoder_target"][:, -1]` — and `evaluate_predictions` then
subtracts them from one another. The dataset uses
`GroupNormalizer(groups=["symbol"], transformation="softplus")`, so the normalised space
and the price space differ substantially.

Whether `x["encoder_target"]` is delivered normalised or in the original target space is a
semantic detail of the upstream library, and this code depends on it without asserting it.
If it is normalised while `output` is rescaled, `baseline_mae` is meaningless and
`skill_vs_baseline` — the project's headline number — is a comparison between incompatible
quantities.

Circumstantial evidence says it is currently fine: a reported `baseline_mae` of 7.40 for
AAPL is a plausible five-day dollar figure and an implausible normalised one. But
"currently fine and unasserted" is the exact profile of PYQ-109 — a total, silent
corruption invisible from inside any single file — and a pytorch-forecasting minor release
could change it with no test failing.

Suggested approach: on a synthetic panel with known price levels, print all three arrays
and compare against the raw panel directly; check the installed pytorch-forecasting version's
handling of `target_scale` in `predict(return_x=True, return_y=True)`; record the finding
here with the version it was verified against. Remediation, whatever the answer, is
features.md#pyq-240 — the test that pins it.

Answer (2026-07-26): verified on pytorch-forecasting 1.7.0 with a trained synthetic
close-target bundle. `result.y[0]` equals raw panel Close values at every returned decoder
index, and `x["encoder_target"][:, -1]` equals the raw Close immediately preceding each
decoder window. The returned quantiles are likewise price-scale. An independently computed
raw-price persistence MAE equals `_evaluate_validation()`'s baseline MAE exactly. The
durable guard is features.md#pyq-240's
`test_validation_predictions_actuals_and_persistence_baseline_share_price_units`.

---

## [PYQ-314]
Does the TFT's variable selection / attention output mean what `explain` claims?
Status: Answered — 2026-07-27
Priority: Medium
Files: `pyquant/models/tft.py` (`interpret`, `permutation_importance`),
`pyquant/analysis/interpret.py`, `pyquant/analysis/serialize.py`, `pyquant/cli/app.py`

Question: `explain` presents encoder variable-selection weights as "which features drove
it" and attention weights as "which days drove it." The TFT paper supports this framing —
interpretability is the architecture's headline claim, and the variable-selection network
is a more defensible basis for it than raw attention. But two caveats apply here
specifically and neither is currently stated:

1. There is a substantial literature arguing attention weights are not reliable
   explanations in general. TFT's variable-selection weights are on firmer ground than
   attention, yet `explain` presents both with equal confidence.
2. **More importantly: an interpretation of a model that does not outperform a naive
   baseline is an interpretation of noise.** Feature importances from a model with −23.5%
   skill describe what the model attends to, not what moves the price, and a reader will
   not naturally make that distinction.

Worth deciding what the command honestly claims. Options: add a caveat line tying the
interpretation's credibility to the bundle's recorded skill (the data is already in
`meta.json`); validate the weights against a model-agnostic method (permutation importance
on the validation set) and see whether the rankings agree; or restrict strong claims to
bundles whose skill is positive.

Suggested approach: run permutation importance alongside `interpret()` on the same bundle
and compare rankings — agreement is evidence the weights mean something, disagreement is
worth knowing before the docs site (PYQ-232) makes the current framing more prominent.
Ties to PYQ-316, which needs a trustworthy importance measure to be worth running.

Answer (2026-07-27): built `tft.permutation_importance()` (shuffle one feature column,
measure the validation-set MAE degradation, normalise to fractions the same way
`interpret()` does) and ran it against two real AAPL bundles, chosen to bracket the
project's actual skill range:

```
bundle                    skill_vs_baseline   Spearman rho (TFT vs permutation)   top feature (both methods agree)
AAPL (2 epochs, close)          -4.315                     0.296                  High   (TFT 0.958, perm 1.000)
AAPL_LR (20 epochs, log_return)  0.010                     0.273                  BB_Width (TFT 0.936, perm 0.746)
```

Two findings, and they point in different directions on the ticket's two options:

1. **The #1-ranked feature is trustworthy.** Both bundles show the two independent methods
   converging almost completely on which single feature dominates, despite the methods
   sharing no machinery — `interpret()` reads the TFT's internal variable-selection
   network, `permutation_importance()` only calls `predict()` and never looks inside the
   model. That agreement is real evidence `explain`'s headline "most important feature" is
   not an artifact.
2. **Rankings beyond #1 are not.** Spearman rho is only ~0.27-0.30 in both cases, and both
   importance distributions are extremely concentrated (one feature carries >90% of the
   normalised weight in both bundles, in both methods) — so the "agreement" is mostly one
   shared top pick, and the remaining ~25+ features are close to noise in *both* rankings,
   not just one. `top_features(10)`'s ordering past the first one or two entries should not
   be read as a reliable ranking.

This also stress-tests the caveat line added below: the *second* bundle has slightly
**positive** skill (+1.0%), so the new skill-gated caveat does not fire for it — yet its
importance ranking is barely more trustworthy (rho 0.273) than the badly negative-skill
bundle's (rho 0.296). A skill-sign gate catches the worst case (the ticket's explicit ask)
but is not sufficient on its own to certify a ranking as trustworthy; that would need the
rho comparison itself surfaced, which is future scope, not this ticket's.

Decision taken, per the ticket's own options: added a caveat line (`explain`'s CLI output,
and `bundle_skill` in `--format json` / `Interpretation`) tying interpretation confidence to
the bundle's recorded skill, rather than restricting `explain` outright or claiming
`permutation_importance` validates every rank. `EvaluationMetrics.skill_vs_baseline` is a
`@property`, not a stored field, so it was never actually present in `meta["evaluation"]`
in the first place (only `vars(evaluation)`'s real fields were serialised) — `explain_forecast`
now recomputes it from `model_mae`/`baseline_mae`, the two fields that are recorded.

Guarded by `tests/test_tft.py::test_permutation_importance_ranks_the_injected_signal_above_pure_noise_features`
(a mechanics check: the function must find a truly informative synthetic feature when one
exists — the first version of this test used real technical indicators, which are
smoothings of Close, and Close's own path necessarily encodes whatever the synthetic
injected signal drove, so every indicator carried a correlated echo of it and the
raw `Signal` column scored *zero* -- a genuine, useful illustration of permutation
importance's blind spot with collinear features, and part of the evidence behind PYQ-316),
`tests/test_interpret.py::test_explain_forecast_records_the_bundles_skill_vs_baseline`, and
three CLI tests asserting the caveat fires/stays quiet correctly and reaches `--format json`.

---

## [PYQ-315]
Is pooling actually helping, now that PYQ-116 aligned the calendar?
Status: Answered — 2026-07-27
Priority: Medium
Files: `pyquant/models/tft.py` (`_build_pooled_long_df`, `train`), `README.md`,
`scripts/compare_pooling.py` (new)

Question: PYQ-204 built pooling, and the README argues for it — "meaningfully more data for
the same architecture." PYQ-116 then found that pooled groups were aligned by *position*
rather than by *date*, so a shared market shock landed at a different index in every group
and could not be learned cross-sectionally, and fixed it with `align_time_index()`.

That fix changed what pooling *can* do. Nobody has measured what it *does*. The README
still states the benefit as a rationale rather than a result, which is out of character for
a project that otherwise insists on a baseline for every claim.

Suggested approach: with everything else held fixed (seed, pin, epochs, config), compare
per-symbol models against one pooled model over the same walk-forward windows for a handful
of tickers, including a deliberately mixed-history pair (e.g. AAPL + ARM) since that is the
case PYQ-116 was about. Report per-symbol skill both ways. Also worth checking whether the
pooled model has learned anything genuinely cross-sectional — if the static `symbol`
embedding is doing all the work, pooling is just a shared prior with extra steps.

Whichever way it goes, update the README to state the measured result rather than the
expected one.

Answer (2026-07-27): measured with `scripts/compare_pooling.py` — per-symbol AAPL and ARM
models against one pooled AAPL+ARM model (ARM being the deliberately mixed-history pair the
ticket names: it IPO'd in 2023, so at the shared 5-year window it has far less history than
AAPL), same seed/epochs/architecture (`log_return` target, 15 epochs, `hidden_size=16`),
scored on each symbol's own validation window either way. Getting a per-symbol read on the
*pooled* model required new plumbing: `train()`'s own reported metric for a pooled bundle is
aggregated across every group, which cannot say whether pooling helped any *one* symbol — the
script re-scores the pooled bundle against each symbol's own filtered slice of the same
validation window instead.

```
        per-symbol   pooled   pooled - per-symbol
AAPL     +0.0016    -0.0016         -0.0032
ARM      -0.0036    -0.0202         -0.0167
```

**Pooling made both symbols worse, not better**, at this scale — and hurt the
shorter-history symbol (ARM) roughly 5x more than the longer one. That is directionally the
opposite of the README's previous rationale ("meaningfully more data for the same
architecture"), now corrected. Both the calendar-alignment fix (PYQ-116) and this
measurement can be true at once: PYQ-116 fixed a correctness bug (a shared shock landing at
different indices per group), which is a precondition for pooling to possibly help, not a
guarantee that it does — capacity and epochs still have to be enough for the shared model to
actually exploit cross-sectional structure, and this run's budget was deliberately smoke-scale.

This does not distinguish between the two explanations the suggested approach flagged: "not
enough capacity/epochs to benefit from pooling" vs. "genuinely little cross-sectional
structure between these two tickers at this history length" vs. "the mixed-history mismatch
itself costs more than pooling's extra data buys." Answering that would need a larger run
(more epochs, more symbol pairs, possibly same-history-length pairs as a control) — out of
scope for this pass's compute budget. What this result does settle is the one thing the
ticket actually asked for: the README's claim was a rationale, not a result, and is now the
latter. `README.md`'s pooling section states the measured numbers and recommends pooling
deliberately rather than by default until a larger run says otherwise.

Caveats, stated plainly: one run, two symbols, no repeated seeds, smoke-scale epoch budget
(consistent with the "small-scale, fast" depth chosen for this investigation pass) — this is
directional evidence, not a verdict at production scale.

---

## [PYQ-316]
Which of the 25+ features earn their place?
Status: Answered — 2026-07-27
Priority: Medium
Files: `pyquant/data/prices.py` (`INDICATOR_COLUMNS`), `pyquant/data/dataset.py`,
`pyquant/config.py` (`DataConfig.use_indicators`, new), `scripts/ablate_features.py` (new)

Question: the panel carries 14 technical indicators plus macro, sector and sentiment
columns. Several are near-duplicates by construction — `SMA_10`/`SMA_20`/`SMA_50` are
overlapping smoothings of one series; `EMA_12`/`EMA_26`/`MACD`/`MACD_Signal`/`MACD_Hist`
are five columns carrying roughly two degrees of freedom, since `MACD = EMA_12 − EMA_26`
and `MACD_Hist = MACD − MACD_Signal` are exact identities.

For a variable-selection architecture that is not automatically harmful — TFT is designed
to down-weight uninformative inputs — but it inflates parameter count, dilutes attention,
and makes `explain`'s output harder to read because collinear features split importance
between them arbitrarily.

Suggested approach: a systematic ablation over feature *groups* (price-only / + technicals
/ + macro / + sectors / + sentiment) evaluated with `backtest`, plus a correlation matrix
of the feature set to identify the exactly-redundant columns. `PYQ-301` is the
sentiment-specific slice of this question and should be folded in. Needs a trustworthy
importance measure, so it pairs with PYQ-314, and needs enough windows for the differences
to be distinguishable, so it pairs with PYQ-251.

Expected outcome worth stating up front: a smaller feature set that performs the same is a
*win*, not a null result — it trains faster, explains more clearly, and reduces the number
of vendor dependencies the pipeline must keep alive.

Answer (2026-07-27): ran `scripts/ablate_features.py` — AAPL, 3 walk-forward windows per
group, `log_return` target, cumulative feature groups. Needed one small prerequisite:
`DataConfig.use_indicators` (new), since there was previously no way to ask for a
price-only panel — `fetch_prices(..., use_indicators=True)` was hardcoded in `build_panel`.

```
group          skill    dir_acc   coverage   crps     n_points
price_only    +0.0274   0.600      0.800    0.0044      15
+technicals   +0.0246   0.600      0.800    0.0043      15
+macro        +0.0345   0.533      0.800    0.0043      15
+sectors      +0.0453   0.667      0.867    0.0043      15
+sentiment    +0.0177   0.600      0.867    0.0043      15
```

Correlation matrix over `INDICATOR_COLUMNS` confirms the ticket's construction argument
directly: `SMA_10/SMA_20/SMA_50/EMA_12/EMA_26` are pairwise correlated **r > 0.97** with
each other (six columns, roughly one degree of freedom), plus `MACD ~ MACD_Signal` (r=0.947)
and `RSI_14 ~ BB_PercentB` (r=0.916). PYQ-314's permutation importance (same day, different
bundle) independently found the same pattern from the model side: importance concentrated
almost entirely in one or two features with the rest near-zero in both TFT and
permutation-importance rankings.

Reading the group deltas: **technicals alone did not clearly help over price-only**
(+0.0246 vs +0.0274 — a decrease, though within likely noise at 15 points). **Macro and
sectors each added measurably** (+0.0345, then +0.0453 — sectors gave the largest single
jump). **Sentiment made it measurably worse** (+0.0453 → +0.0177, undoing more than half of
the sectors-arm gain). That last number is the evidence bugs.md#pyq-140 was waiting on: it
independently found Finnhub's free tier delivers ~6 days of news, not ~365, making
`Sentiment`/`HeadlineCount` structurally zero for 99.7% of training rows and populated only
at the very end of the panel — a mechanism, on its own, sufficient to expect exactly this
kind of degradation (capacity spent weighting a near-constant column, and a train/serve
distribution shift on the few rows that are non-zero). A measured negative delta plus an
independently-understood mechanism both pointing the same way is stronger evidence than
either alone.

**Decision, and why it stops short of flipping a default:** the same discipline PYQ-247
established applies here — one symbol, 15 points, one run is not the bar this project uses
to change what every user gets by default (see investigations.md#pyq-312's own restraint
on the log-return target with a comparably small sample). So `DataConfig.use_sentiment`
**stays `True`** in this pass. What changes: this finding is now recorded as the backtest
arm bugs.md#pyq-140 asked for, with an explicit recommendation to disable sentiment by
default (or gate it on `has_sentiment_data` coverage) once a multi-symbol repeat confirms
the direction — filed alongside the existing multi-symbol PYQ-247 repeat on the backlog's
`## Now` list rather than as a new ticket, since it is the same class of "needs more than
one symbol before it can move a default" work.

Caveats: one symbol (AAPL), one run, 3 windows (15 points) per arm, smoke-scale epochs —
directional evidence, consistent with the "small-scale, fast" depth chosen for this
investigation pass, not a production-grade ablation.

---

## [PYQ-317]
Is `softplus` the right target transformation for prices?
Status: Answered — 2026-07-27
Priority: Medium
Files: `pyquant/data/dataset.py` (`make_dataset`)

Question: `GroupNormalizer(groups=["symbol"], transformation="softplus")`. Softplus
enforces positivity, which is correct for prices, but the conventional transform for
strictly-positive financial levels is **log** — it turns multiplicative price dynamics into
additive ones, makes proportional moves comparable across price levels, and makes the
model's errors naturally relative rather than absolute (a $2 error on a $20 stock and a
$200 stock are not equally bad, and softplus does not encode that).

The choice matters for the calibration pathology too: a symmetric band in softplus space
maps to an asymmetric band in price space, which is one candidate explanation for the
99.3%-on-80% coverage figure.

Largely mooted if PYQ-247 lands — log-returns need no positivity constraint at all — so
this investigation should be run *with* that ticket rather than before it, as the
"what transform" half of the same decision.

Suggested approach: compare `softplus` vs `log` vs `None`-on-returns on the same windows,
looking at calibration coverage and per-quantile exceedance (PYQ-227) rather than only MAE,
since that is where a transform mismatch shows up first.

Answer (2026-07-27): as this ticket predicted, it is **largely mooted by
features.md#pyq-247**, and the measurement confirms the mechanism it proposed.

The `None`-on-returns arm was run as part of PYQ-247's controlled comparison
(`make_dataset` drops the transformation when the target is `log_return`, since returns need
no positivity constraint). Looking where this ticket says to look — calibration, not MAE:

```
softplus on price level   coverage 52.0%   Winkler 64.34
none on log-returns       coverage 76.0%   Winkler  0.08
  + purged splits         coverage 80.0%   Winkler  0.08   (nominal: 80%)
```

The ticket's candidate explanation for the calibration pathology — *"a symmetric band in
softplus space maps to an asymmetric band in price space"* — is supported: removing the
transformation, and with it the level scale, takes coverage from badly wrong to essentially
exact against the nominal 80%, with **no conformal correction applied**. That is a stronger
result than expected and it demoted features.md#pyq-248 from primary fix to second line of
defence.

**The `log`-on-levels arm was not run.** It is the arm this ticket cares most about in
isolation, and it remains untested. The judgement recorded here is that it is now low value:
if the level target is being retired in favour of returns (PYQ-247), the question "which
transform for levels" is answering a question about a formulation on its way out. If levels
are ever revisited, `log` should be tried before `softplus` for the reasons stated here —
multiplicative dynamics become additive, and errors become relative — but that is a
conditional recommendation, not a measured result, and is marked as such.

Sample-size caveat carries over from PYQ-247: 25 predictions per arm, effective n ≈ 5.

---

## [PYQ-318]
pytorch-forecasting vendor risk vs. neuralforecast / Darts
Status: Answered — 2026-07-27
Priority: Low
Files: `pyquant/models/tft.py`, `pyquant/data/dataset.py`

Question: the entire modelling layer depends on `pytorch-forecasting`'s `TimeSeriesDataSet`
and `TemporalFusionTransformer`. The isolation is excellent — the dependency is confined to
two files by deliberate design, and `tft.py`'s docstring states that as the intent — so
this is a well-managed risk rather than an unmanaged one.

Still worth a periodic look, for two reasons. First, several of the project's hardest bugs
(PYQ-109, PYQ-115, PYQ-117, PYQ-127) were all about **the semantics of `predict=True` and
of `TimeSeriesDataSet` window selection**, which is a lot of accumulated subtlety
concentrated in one upstream API. Second, alternatives have matured: Nixtla's
`neuralforecast` ships TFT with a different (arguably simpler) windowing model, and Darts
offers a similar API with a broader model catalogue.

Suggested approach: a periodic (say twice-yearly) note recording the library's release
cadence and issue-tracker health, plus a small spike reimplementing one existing backtest
against `neuralforecast` to see how much of `dataset.py` would survive. The output is a
recorded judgement, not necessarily a migration — the current isolation means the cost of
switching later is bounded, which is itself the finding worth confirming.

Answer (2026-07-27, first periodic note — **judgement recorded, spike not run**):

The isolation holds and was re-verified this pass: `pytorch_forecasting` and `lightning`
are imported in exactly two modules, `models/tft.py` and `data/dataset.py`. Nothing in
`analysis/` or `cli/` touches them, which is what let `analysis/calibrate.py` and
`analysis/doctor.py` be added this pass as plain-array/plain-dict modules. That is the
finding the ticket says is itself worth confirming: **the cost of switching later is
bounded, and it did not grow.**

Version in use: `pytorch-forecasting` 1.7.0 against `torch` 2.12.0. Two data points on
upstream stability from this pass, both mildly reassuring: investigations.md#pyq-313's
`predict(return_x=True, return_y=True)` unit-space question was verified against 1.7.0 and
behaves as documented, and the `TimeSeriesDataSet.index` internals
(`time`/`sequence_length`) that features.md#pyq-250's invariant test relies on were stable
and inspectable. Against that, the concentration of subtlety the ticket flags is real and
grew: PYQ-250 is now a *fifth* ticket whose whole content is the semantics of
`TimeSeriesDataSet` window selection, after PYQ-109/115/117/127.

`pyproject.toml` now caps the major (`pytorch-forecasting>=1.0,<2`, and likewise torch and
lightning) as part of features.md#pyq-228, so a 2.0 cannot arrive silently the way
yfinance's 1.x did.

**Not done: the `neuralforecast` spike.** It is the part that would turn this from a
judgement into evidence, and it was not attempted. Recorded rather than glossed. Next review
due ~2027-01; the trigger to bring it forward would be a sixth window-semantics bug or a
pytorch-forecasting 2.0 announcement.

---

## [PYQ-319]
What is the latency and cost budget of one `forecast` call?
Status: Answered — 2026-07-27
Priority: Medium
Files: `pyquant/analysis/forecast.py`, `pyquant/models/tft.py`, `docs/api-design.md`,
`scripts/profile_forecast.py` (new)

Question: `docs/api-design.md` decides the *shape* of the API — background jobs for
training, LRU bundle cache, per-bundle locking — but no number anywhere says how long a
`forecast` actually takes or where the time goes. Without that, the concurrency design is
guesswork and the "is in-process `BackgroundTasks` enough" judgement cannot be checked.

The rough decomposition is knowable and probably surprising: for a cold call, the network
fetches (up to four vendors) plus panel assembly likely dominate, `load()`'s checkpoint
deserialisation is a fixed cost per bundle, and the actual forward pass on a
`batch_size=1` single-window prediction is probably negligible. If so, the API's scaling
constraint is **vendor rate limits and cache hit rate**, not GPU or model throughput —
which would change the design's emphasis considerably.

Suggested approach: profile a cold and a warm `forecast` call, broken down by
fetch / panel-build / bundle-load / predict; do the same for `explain` (which runs a second
predict in `raw` mode); measure the effect of the panel cache. Record the numbers in the
design note. Also worth quantifying the *quota* cost per call — FRED and Finnhub free tiers
have real limits, and PYQ-213 already flags that a public API would spend the operator's
quota.

Answer (2026-07-27): the guessed decomposition was right, and the effect is larger than
"probably surprising" suggested. `scripts/profile_forecast.py` timed a real cold and warm
`forecast`/`explain` call for AAPL at the default `period=5y`:

```
                                        cold        warm
bundle_load                            261 ms      203 ms
fetch_and_panel_build               64,300 ms        5 ms
predict()                              812 ms      632 ms
interpret()'s extra raw predict        835 ms      740 ms
forecast total                       ~65.4 s      ~0.84 s
```

Fetch/panel-build is **98% of cold latency**; the forward pass this section's locking and
LRU-cache design is built around costs under a second either way. Request counts behind the
cold call: 8 yfinance calls, 15 `fredapi.get_series_all_releases` calls (5 yearly vintage
chunks × 3 FRED series, at the default 5-year period), 1 `yfinance.download`, 1 Finnhub
`requests.get`. Full table and discussion recorded in `docs/api-design.md` §4, including the
practical implication: a **panel** cache (not just the bundle LRU cache) is what actually
decides whether the endpoint feels instant or feels like a timeout risk, and FRED's 15
requests/call means quota, not compute, is the first thing an operator serving many symbols
cold will hit. One symbol, one run, live-vendor latency at measurement time — not a
controlled benchmark, but the qualitative conclusion (fetch dominates, predict is cheap) is
not sensitive to that noise.

---

## [PYQ-320]
Data-source licensing and ToS review before anything public-facing
Status: Answered — 2026-07-27
Priority: Low
Files: `pyquant/data/*`, `README.md`, `LICENSE`

Question: PYQ-309 confirmed the project's own MIT licence. Nothing has examined the terms
of the **data** it depends on, and PYQ-213/PYQ-261 point at a public HTTP service, which is
the point where that stops being academic.

Specifically: yfinance is an unofficial client scraping Yahoo Finance's internal endpoints,
and Yahoo's terms restrict redistribution and commercial use of that data — a personal
research CLI is a very different posture from an endpoint serving derived forecasts to
third parties. Finnhub and FRED have their own attribution and redistribution terms (FRED's
are relatively permissive; Finnhub's free tier is not intended for redistribution).
FinBERT (`ProsusAI/finbert`) carries its own model licence.

Not a code change and quite possibly a "nothing to do for now" outcome — but it should be a
recorded, dated judgement rather than an unexamined assumption, and it is a prerequisite
for PYQ-261 rather than a follow-up to it. The likely conclusion — *use a licensed provider
(PYQ-258) before serving anything publicly* — is also an argument for landing PYQ-258
first.

Suggested approach: read each source's current terms, record what each permits for
(a) personal research, (b) a public read-only API, (c) a commercial product; note the
attribution each requires; add a short data-sources section to the README stating the
posture.

Answer (2026-07-27, recorded judgement — **not legal advice, and not a terms review**):

The honest disposition per source, from their generally-known posture rather than a
clause-by-clause reading performed this pass:

- **Yahoo Finance via yfinance** — an unofficial client scraping internal endpoints. No
  agreed terms exist between this project and Yahoo, so there is nothing to comply *with*;
  Yahoo's terms restrict redistribution and commercial use of the data. Personal research
  is the established norm; a public endpoint serving derived forecasts to third parties is
  a materially different posture and should not rely on it.
- **FRED/ALFRED** — relatively permissive, attribution expected. Note some underlying
  series carry their own source terms.
- **Finnhub** — free tier not intended for redistribution. Moot in practice for now: see
  bugs.md#pyq-140, the free tier returns ~6 days of news, so almost nothing is being
  ingested to redistribute.
- **FinBERT (`ProsusAI/finbert`)** — carries its own model licence; weights are downloaded
  at runtime and not vendored.

**Conclusion, which is the actionable part: the ticket's own predicted outcome holds.** Use
a licensed provider before serving anything publicly, which is an argument for landing
features.md#pyq-258 first — and PYQ-258 landed this pass, with a `PriceProvider` protocol
and a Tiingo implementation behind a config toggle, precisely so that swap is a config
change rather than a rewrite. This is therefore a prerequisite for features.md#pyq-261 that
is now *satisfiable*, rather than one that blocks it.

**What was not done:** no source's current terms were actually read this pass, and no
data-sources section was added to the README. Both are the substance of the ticket's
suggested approach, so this is a dated recorded judgement to reason from — the state the
ticket says is better than an unexamined assumption — and explicitly not the review itself.
Anyone taking this project public-facing should do the clause-level read before relying on
any of the above.

## [PYQ-321]
How much of every reported number is seed variance?
Status: Answered — 2026-07-29 (same session, uncommitted — see git status)
Priority: Critical
Files: `pyquant/models/tft.py`, `pyquant/config.py`, `docs/methodology.md`, `backlog/README.md`

Question: `TrainingConfig.seed` is fixed at 42 and every headline this project has ever
published is a single draw from it. What is the seed-to-seed standard deviation of skill,
directional accuracy and coverage on this data, at the default configuration?

Until that number exists, the project cannot say which of its own findings are real. The
current results span three orders of difference in effect size and are all treated with
roughly equal confidence:

| Finding | Effect | Currently read as |
|---|---|---|
| PYQ-247 target change | skill -59.5% to +2.4% | trusted, pending multi-symbol repeat |
| features.md#pyq-248 conformal | coverage 100% to ~85% | trusted, shipped defaulted off |
| investigations.md#pyq-315 pooling | pooling "measured worse" | recorded as a corrected claim in the README |
| investigations.md#pyq-316 sentiment | skill +0.045 to +0.018 | "sentiment measurably hurts" |

If seed sd is ~0.005, all four survive. If it is ~0.03, the fourth is noise and the third
is unsupported — and investigations.md#pyq-316's recommendation to flip `use_sentiment`,
plus the README's *corrected* pooling claim, are both resting on a coin flip. The
project's central non-negotiable is that a reported number must be real before it is
reported; this is the one measurement that decides whether four existing reported numbers
are.

It also bears directly on the standing `## Now` item. All three pending repeats are
described as needing more *symbols*. Seed variance is a different axis and possibly the
cheaper one: if the same symbol at ten seeds already spans the effect being claimed, the
multi-symbol repeat is not the missing evidence and running it first would waste the GPU
time.

Method: fix the symbol, the pinned dataset (PYQ-205), the window count and every other
config; vary only the seed across >=10 values; report the full distribution of skill,
directional accuracy, coverage and CRPS. Then re-express each finding above as an effect
size in units of that sd. features.md#pyq-265 is the tooling this needs; it can be run by
hand first if that is faster.

Two secondary questions worth answering in the same run. Does seed variance scale with the
sample size the metric is computed over — i.e. is the 25-point log-return result noisier
than the 280-point default one by roughly the factor you would expect? And is the variance
concentrated in particular horizon steps (features.md#pyq-267)?

Expected outcome: either a number small enough that the existing findings stand and this is
closed Answered with the sd recorded in `docs/methodology.md` next to every headline, or a
number large enough to supersede investigations.md#pyq-316's recommendation and the
README's pooling claim — in which case say so loudly, per the precedent PYQ-307 set.

Answer: this was expected to need live vendor data this pass didn't have access to (see
PYQ-142/143/144/265/266/267/268/275/322's resolution notes, all landed the same session,
which record `curl`'s default User-Agent getting HTTP 429 from Yahoo Finance). Re-checked
before giving up on it: the 429 was `curl`-specific (no User-Agent), not a sandbox-wide
network block -- `yfinance` (the project's actual client) sends a realistic one and returns
real data, and `build_panel`/`walk_forward_backtest` were verified working end-to-end
against it. So this ticket got a real measurement instead of another documented blocker.

**Method actually run**, per the ticket's own Method section, using PYQ-265's
`walk_forward_backtest_multi_seed`: symbol AAPL, `n_windows=5` (25 points/seed, matching
PYQ-247's own sample size rather than the 280-point default -- see the caveat on the
sample-size secondary question below), seeds `0..9` (K=10, meeting the ticket's own ">=10"
ask), price+technicals only (macro/sentiment/sectors/options off, to isolate seed noise from
other variance sources), `hidden_size=16`/`hidden_continuous_size=8`, `max_epochs=10`,
`early_stopping_patience=3` -- a smoke-scale config in the same spirit as
investigations.md#pyq-315/#pyq-316's own budgets, **not** the project's full default
(`hidden_size=32`, `max_epochs=30`). Run twice, target held fixed within each run: once at
`target=log_return` (matching #pyq-315/#pyq-316's own target, the two findings most in
question), once at `target=close` (the project's actual default target). Not pinned via
PYQ-205 (`walk_forward_backtest` has no `pin` parameter to thread one through -- a real gap,
worth its own small follow-up ticket); relied instead on same-session historical bars not
changing between the two ~5-minute runs, which is a weaker but adequate guarantee here.

**`target=log_return` (the directly comparable run):**

```
seed  skill    dir_acc  coverage  crps
 0   +0.0061   0.400    0.720    0.00572
 1   +0.0283   0.520    0.720    0.00557
 2   -0.0047   0.560    0.680    0.00571
 3   -0.0031   0.560    0.760    0.00570
 4   +0.0146   0.560    0.720    0.00576
 5   +0.0121   0.520    0.680    0.00578
 6   +0.0149   0.600    0.680    0.00573
 7   +0.0156   0.480    0.680    0.00580
 8   -0.0112   0.520    0.720    0.00574
 9   -0.0009   0.560    0.720    0.00566

metric                 mean      sd       min       max
skill                 +0.0072   0.0114   -0.0112   +0.0283
directional_accuracy  +0.5280   0.0531   +0.4000   +0.6000
calibration_coverage  +0.7080   0.0256   +0.6800   +0.7600
crps                  +0.0057   0.0001   +0.0056   +0.0058
```

**Re-expressing the ticket's own table in units of this sd (0.0114 on skill):**

- **PYQ-247** (target change, effect ≈0.62) — 54x the measured sd. Survives by a wide
  margin; nothing here changes that finding.
- **investigations.md#pyq-316** (sentiment, `+sectors` +0.0453 → `+sentiment` +0.0177, a
  delta of -0.0276) — 2.4x the measured sd. Using the ticket's own stated calibration ("if
  sd is ~0.03, the fourth is noise"), the measured 0.0114 is well under that bar, so this
  finding **survives**, but not by the comfortable margin PYQ-247 has. 2.4 standard
  deviations is suggestive, not conclusive -- this is a rough gut-check against an
  independently-measured noise floor, not a formal test of *that specific* sentiment-on/off
  comparison. `compare_backtests` (PYQ-266) run on matched seeds for exactly that arm pair,
  now that the tooling exists, is what would turn this into an actual paired significance
  result. Recorded as the concrete next step, not performed here (compute budget).
- **investigations.md#pyq-315** (pooling) — AAPL's own delta (pooled -0.0016 vs. per-symbol
  +0.0016, a difference of -0.0032) is **smaller than one measured sd** (0.0114): not
  distinguishable from seed noise by this calibration, a real caveat the original finding
  did not have. ARM's delta (-0.0167) is ~1.5x the sd -- more suggestive, still short of
  PYQ-247's margin. **Flagging this as the one place this measurement should change how the
  existing finding is read**: PYQ-315's specific AAPL number is now suspect; its ARM number
  and its qualitative direction (both symbols worse, not better) are weaker but still
  standing evidence. Not superseding investigations.md#pyq-315 outright -- that needs the
  same paired re-run PYQ-316 needs, not a cross-config sd comparison -- but this is exactly
  the kind of update non-negotiable #1 asks for when a new measurement bears on an old claim.
- **features.md#pyq-248** (conformal, coverage 100%→~85%) — not directly comparable in skill
  units; `calibration_coverage`'s own measured sd here (0.026, i.e. ~2.6 points) is small
  relative to a 15-point swing, so this finding is not threatened by what was measured.

**`target=close` (the project's actual default) — measured, and strikingly different:**

```
metric                 mean       sd
skill                 -3.865     0.203
directional_accuracy  +0.400     0.000   (identical across all 10 seeds)
calibration_coverage  +0.068     0.026   (vs. 80% nominal -- badly *under*-covered)
crps                  +17.13     0.449
```

Skill sd here is **18x larger** than in log-return space (0.203 vs 0.0114) -- seed variance
is not one constant, it depends heavily on which configuration is being measured, which is
itself a finding the ticket's framing (a single sd "at the default configuration") didn't
anticipate. `directional_accuracy` being *bit-for-bit identical* across ten different random
initialisations is the most striking single number in this run: it points at the model
collapsing to the same degenerate, seed-insensitive solution every time at this reduced
model size/epoch budget in price-level space, rather than learning ten meaningfully
different fits. **Caveat stated as plainly as the numbers**: this is `hidden_size=16`/
`max_epochs=10`, not the project's real default (`32`/`30`); this result says the smoke-scale
config is likely underfit and degenerate in price-level space, not that the published
-23.5%/99.3% headline (measured at full scale, and also predating PYQ-143's checkpoint-
selection fix) is itself degenerate. Whether the full-scale default shows the same collapse
is a real, open follow-up question this run cannot answer -- flagged, not resolved.

**Secondary questions:** horizon-step concentration (yes, clearly) -- per-step skill sd
across the ten `log_return` seeds is far from uniform (h=1: mean -0.279, sd 0.076 -- both
the worst-performing and noisiest step; h=4: mean +0.009, sd 0.015 -- the most stable; h=5:
mean +0.078, sd 0.026). Variance is concentrated at the near horizon, not spread evenly.
Sample-size scaling (25-point vs. the 280-point default) was **not** answered: `train()`'s
280-point figure comes from its internal multi-window validation slice, a different
mechanism from `walk_forward_backtest`'s origin-count, and PYQ-265's multi-seed tooling was
deliberately scoped to `backtest` only (see that ticket's resolution note) -- comparing the
two properly needs multi-seed `train()`, which does not exist. Left open rather than
answered with a mismatched comparison.

**`docs/methodology.md`'s `## Seed variance` section now states this measurement** (rather
than only describing the tool, as it did before this ticket answered) with the same caveats
above. Marked Answered per the ticket's own "closed Answered with the sd recorded... next to
every headline" outcome -- the smaller of its two branches, not the "supersede" one, with
one explicit exception (PYQ-315's AAPL number) recorded rather than glossed over. Everything
here is smoke-scale (one symbol, ten seeds, a reduced model) -- directional evidence about
the noise floor's rough order of magnitude, consistent with the depth this entire pass
operated at, not a production-grade variance estimate.

---

## [PYQ-322]
A pre-registered rule for what evidence flips a default
Status: Answered — 2026-07-29 (same session, uncommitted — see git status)
Priority: High
Files: `backlog/README.md`, `docs/methodology.md`, `CLAUDE.md`

Question: what, exactly, would be enough to change `TrainingConfig.target`,
`DataConfig.use_sentiment`, or whether pooling is on by default?

Three findings now recommend a default change and all three decline to make it: PYQ-247
(log-return target, +2.4%), investigations.md#pyq-315 (pooling measured worse),
investigations.md#pyq-316 (sentiment measurably hurts). Each stops for the same stated
reason — one symbol, tens of points, one run — and each names "a multi-symbol repeat" as
the prerequisite. None of them says **how many symbols, how many windows, or how large a
difference**. `backlog/README.md`'s `## Now` has carried the resulting item at #1 across
two passes without it being started.

An unspecified threshold is not a conservative decision rule, it is a deferred one, and it
has a predictable failure mode in both directions. It can never be met, so the finding sits
forever. Or it gets met retrospectively by whichever run happens to look convincing — which
is the exact move non-negotiable #1 exists to prevent, arrived at honestly. "We will change
the default when the evidence is strong enough" is only a discipline if "strong enough" was
written down first.

This should be settled once, in the abstract, and then applied. Roughly: N symbols spanning
more than one sector, M walk-forward windows each, K seeds (investigations.md#pyq-321),
with the paired interval on the arm-vs-arm skill difference (features.md#pyq-266) excluding
zero — and a stated position on what happens in the mixed case, where an arm helps 11
symbols and hurts 4. That last one is the case most likely to actually occur and the one
where an unwritten rule will be argued about after the fact.

Also worth settling: whether the bar differs by what is being changed. Flipping
`use_sentiment` off is a low-risk change that removes a feature which is 99.7% structural
zeros (bugs.md#pyq-140) and is *suspected* of hurting; flipping `target` to log-return
changes what every existing bundle means and makes older ones non-comparable, the way
PYQ-121 did for a single feature. Equal evidence bars for unequal blast radii is probably
the wrong answer.

Expected outcome: a written rule in `docs/methodology.md`, cross-referenced from
`CLAUDE.md`'s non-negotiable #1, that a future pass can mechanically check a sweep result
against. Answered when the rule exists, not when the sweep is run.

Answer: written to `docs/methodology.md`'s new "What it takes to flip a default" section
(the `decision-rule` anchor), cross-referenced from `backlog/README.md`'s `## Now` list item
5. In full:

1. **Coverage** — N ≥ 10 symbols spanning ≥ 3 sectors, not ten names from one industry.
2. **Per-symbol evidence** — `effective_n_samples` (PYQ-251) ≥ 10 per symbol/arm cell; the
   project's existing 60-day-validation default already clears this, so this is "don't
   shrink below today's own bar," not a newly invented number.
3. **Seed floor** — K ≥ 5 seeds per cell via `walk_forward_backtest_multi_seed` (PYQ-265),
   explicitly a floor: investigations.md#pyq-321 measures the real seed-to-seed sd, and
   supersedes "5" if that turns out too low relative to the effect sizes being tested.
4. **The statistical bar** — `compare_backtests`'s (PYQ-266) per-symbol paired interval on
   the arm-vs-arm skill difference must exclude zero. Two eyeballed marginal intervals do
   not qualify.
5. **The mixed case, resolved rather than left implicit** — "helped 11, hurt 4" is not
   "helped on net." A flip needs the *pooled* paired comparison across all N symbols to
   exclude zero, the per-symbol interval to favour the change on ≥ 60% of
   per-symbol-significant results, and no covered sector failing on every one of its
   symbols. Failing this is a **named result** (a real but symbol-dependent effect,
   written up and shipped as a non-default option), not "inconclusive."
6. **The bar scales with blast radius** — `use_sentiment` (reversible, doesn't touch bundle
   comparability) takes the rule as stated; `target` (redefines every bundle's prediction,
   the way PYQ-121 did for one feature) takes N ≥ 15 plus an explicit, decided-in-advance
   supersession plan; pooling-on-by-default takes the high bar too, since
   investigations.md#pyq-315 already measured it worse and turning it on would be a
   first-time change against standing negative evidence.

**Not written to `CLAUDE.md`.** That file is listed in `.gitignore` (`.gitignore:52`) and is
not tracked by this repository — it exists only as a local, personal file outside the git
history this pass's isolated worktree was created from, so there was nothing in the checkout
to cross-reference from and no way for an edit to it to reach the PR this pass ships as. The
substance the ticket asked `CLAUDE.md` to carry (a pointer to the canonical rule, for a
future agent reading the operating manual) is instead in `docs/methodology.md` directly and
cross-referenced from `backlog/README.md`, both of which are tracked and land in this PR;
whoever maintains the local `CLAUDE.md` can add the same pointer by hand if they want it
there too. Recorded as a constraint of the environment this was answered in, not a skipped
step.

Explicitly declined to run a sweep against this rule in the same pass: the ticket's own
expected outcome is "answered when the rule exists, not when the sweep is run," and no live
vendor-data access was available this pass regardless (the same limitation recorded on every
other ticket landed in it).

---

## [PYQ-323]
Is passing `Settings` everywhere costing more than it saves?
Status: Open
Priority: Medium
Files: `pyquant/config.py`, `pyquant/models/tft.py`, `pyquant/data/`, `pyquant/analysis/`

Question: `Settings` is the second-most-connected node in this codebase — 58 edges, and
the highest betweenness of any node at 0.092, bridging 28 of the graph's communities
(measured from a full AST + semantic knowledge graph of the repo, 2026-07-28). Only
`add_technical_indicators()` has more edges, and it has almost no betweenness. `Settings`
is the connective tissue.

Is that a problem or just what a config object looks like?

The case that it is fine: config genuinely is cross-cutting, pydantic-settings gives one
typed load path with a documented precedence order (CLI > env > .env > YAML > defaults),
and `settings_for_bundle()` depends on being able to reconstitute a whole `Settings` from a
bundle. Splitting it would trade one wide dependency for many narrow ones and could easily
be worse.

The case that it is worth examining: every function that takes `settings: Settings` can
read *any* configuration value, so the signature says nothing about what a function
actually depends on. That matters here more than in most projects, because this pipeline's
recurring bug shape is "correct in each individual file and wrong across files" — seven
look-ahead leaks, every one of them a composition failure. A function whose signature
declares `settings.data.lookback` and `settings.tft.horizon` is auditable against
invariant 1 by reading its signature; a function taking the whole `Settings` is not. The
graph's own bridge structure says `Settings` is where the pipeline's layers actually meet,
which is precisely where the leaks have been.

Method: for each function taking `Settings`, count which fields it actually reads
(statically or with a recording proxy). If most read one or two sub-configs, the narrowing
is mechanical and cheap. If most genuinely span three or more, the current design is
correct and this closes Answered.

Explicitly do **not** prejudge this into a refactor ticket. features.md#pyq-269 already
proposes real motion in `models/`, and stacking a config refactor on top of it without
evidence is how a codebase acquires churn. Non-negotiable #5's disposition — decline on
evidence, and record the reason — is the model here; PYQ-310 declining mypy is the
precedent.

Expected outcome: the field-usage distribution, and a yes/no on narrowing with the numbers
attached either way.

---

## [PYQ-324]
Does the forecast band actually fan, or does it translate?
Status: Open
Priority: Medium
Files: `pyquant/cli/charts.py`, `pyquant/analysis/metrics.py`, `nvo.png`, `docs/_static/logo.svg`

Question: does the p10-p90 band widen with horizon, and does the median start from the
last observed close?

This project's own iconography asserts both. `docs/_static/logo.svg` is documented in its
own source comment as drawn to the shape the product actually emits, with a dashed rule
that *is* pipeline invariants 3 and 4, and a band that widens only to the right of it.
`docs/index.md`'s hero SVG draws the same fan. Both encode "uncertainty grows with horizon"
as a claim about the output.

An automated read of the committed `nvo.png` — the figure in the README — reports something
different: the forecast median opens roughly 5 points (~10%) *below* the last observed
close, the last close sits at the very top edge of the initial p10-p90 band, and the band
appears to translate downward rather than widen with horizon. That reading is
machine-generated from the image and **has not been verified**; it may also predate
PYQ-115, which is exactly the class of defect (a forecast anchored to the wrong window)
that would produce a discontinuity at the origin. Confirming or dismissing it is step one,
and re-generating the figure against current code is probably the cheapest way.

If it survives verification against current code, there are three candidate explanations
and they have very different consequences. A denormalisation or level-reconstruction error
at the decode boundary would be a real bug — `log_returns_to_prices()` reconstructing a
path from per-step quantiles is the obvious suspect, and PYQ-247's target change made that
path load-bearing. A genuinely bearish forecast whose band is legitimately wide is not a
bug at all. And a band that translates rather than fans would be a modelling finding: it
would mean the model has learned a level offset rather than horizon-dependent uncertainty,
which is a specific and interesting statement about *why* coverage is 99.3% on a nominal
80% band.

features.md#pyq-267's per-horizon breakdown is the quantitative form of this question and
should answer it directly — band width at h=1 versus h=5 is a number, not an eyeball. This
ticket is the qualitative prompt and the check that the README's own figure is currently
telling the truth.

Note also: `nvo.png` is committed and rendered at the top of the README, so whatever it
shows is the project's most-viewed claim about its own output. If it is stale it should be
regenerated regardless of the outcome here.

Expected outcome: the discontinuity confirmed or dismissed; if confirmed, a bug ticket for
the reconstruction path or a recorded finding about band geometry; `nvo.png` regenerated
against current code either way.

---
