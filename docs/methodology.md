# Methodology

How PyQuant decides whether its forecasts are any good, and what the current numbers do
and do not license you to conclude.

:::{admonition} Headline result: two configurations, two sample sizes
:class: warning

**Default configuration** (`TrainingConfig.target = "close"`, price *level*), over **56
walk-forward windows / 280 predictions**:

| Metric | Value | Reading |
|---|---|---|
| Skill vs. persistence baseline | **−23.5%** | The model's MAE is 23.5% *worse* than "predict no change". |
| Directional accuracy | **57.5%** | Slightly better than a coin flip, on 280 points. |
| Calibration coverage (p10–p90) | **99.3%** | A nominal 80% band that contains 99.3% of outcomes is far too wide to be useful. |

Measured on commit `a7a2b5f`, `pyquant train NVO` at defaults. **The default forecaster
does not beat a naive persistence baseline.** That is the project's central open problem,
and it is reported rather than tuned out of sight.

**`log_return` target** (PYQ-247, everything else — seed, epoch budget, window count —
held fixed), over **one symbol, 25 predictions, effective n≈5**:

| Metric | Value | Reading |
|---|---|---|
| Skill vs. persistence baseline | **+2.4%** (+3.8% purged) | The *only* configuration measured so far that beats "predict no change". |
| Directional accuracy | **52–56%** | *Falls* from 80%, not a typo — see below. |
| Calibration coverage (p10–p90) | **76–80%** | Close to nominal, with **no conformal correction applied**. |

Both are real, reproducible measurements, not a before/after of the same thing — different
target, different geometry, different sample size. The second result is the more
interesting one and the less trustworthy one, for the same reason: n≈5 is not enough to
change what every user gets by default (non-negotiable #1), so `TrainingConfig.target`
still defaults to `"close"` pending a multi-symbol repeat. See {ref}`negative-result` for
why the first number is close to what the *formulation* predicts regardless of tuning, and
{ref}`related-open-questions` for what happened when the same discipline was applied to
pooling and to the feature set.

**Both headline measurements above predate PYQ-143** (checkpoint selection was fixed to use
a window disjoint from the one these metrics are reported on; see {ref}`split-geometry`).
They were measured with `EarlyStopping`/`ModelCheckpoint` selecting the best of many epochs
against the *same* window scored above — a selection-event bias, so the true numbers are
expected to be somewhat worse than shown, per the resolution note on PYQ-143. Neither figure
has been re-measured under the corrected geometry as of this pass; this codebase had no live
vendor-data access available to re-run `pyquant train`/`pyquant backtest` for real. Re-running
both configurations under the fixed geometry is the natural next step and should replace this
table when done, rather than being read as still current.
:::

## A third number, from a third protocol — and why it is not in the table above

`pyquant backtest NVO --windows 5` at the same default configuration (target `"close"`,
2026-07-27, git `90afcf8`), measured freshly for this page, reports **+36.2% skill**, 88%
directional accuracy, 100% coverage. That is a *better* number than the headline table's
−23.5%, and it is deliberately **not** promoted to the table above, for one reason: it is
not the same measurement.

`pyquant train`'s 56-window figure comes from **one** trained model, scored on every
overlapping window in a single 60-day validation holdout. `backtest --windows 5` trains
**five independent models** at five rolling origins and scores each *only* on its own
5-day horizon — 25 points total, `effective_n_samples = 1`. The two protocols answer
related but different questions ("how good is the one model I'd actually deploy" versus
"how stable is training across time"), and per-window results here make the instability
concrete rather than abstract:

| Window | Skill | Directional acc. | Coverage |
|---|---|---|---|
| 1 | +55.7% | 80% | 100% |
| 2 | −24.3% | 100% | 100% |
| 3 | +33.7% | 100% | 100% |
| 4 | +25.0% | 60% | 100% |
| 5 | +52.1% | 100% | 100% |

Four of five origins are positive, one is not, and the aggregate is positive mainly because
four positive windows outweigh one negative one — not because the model is reliably
skillful. At `effective_n_samples = 1` this is barely more than a single data point wearing
five costumes; the honest reading is "highly variable across origins, sign not yet settled,"
not "the default configuration works." It is recorded here, not discarded, in the same
spirit as every other number on this page: a result that complicates the headline is not a
reason to leave it out.

## Why there is a baseline at all

An absolute loss with nothing to compare it against is not a result. A quantile loss of
0.67 is meaningless in isolation; a MAE of $7.40 on a $150 stock over a 5-day horizon is
meaningful only once you know what doing nothing would have cost.

So every reported metric is relative to a **persistence baseline** — "predict no change".
In price space that is the last observed close, carried flat across the horizon. In
log-return space it is a predicted return of exactly zero. Skill is the relative MAE
improvement:

```
skill = (baseline_mae - model_mae) / baseline_mae
```

Positive means the model beat doing nothing. It is currently negative.

:::{important}
Persistence is a *strong* baseline for a price level, not a weak one. For a series close to
a random walk the conditional expectation of tomorrow's level essentially **is** today's
level, so any deviation the model makes costs MAE in expectation. See
{ref}`negative-result`.
:::

## What is measured

{py:class}`~pyquant.analysis.metrics.EvaluationMetrics` carries the following, all computed
from plain numpy arrays so the same code scores a single training holdout and a
multi-origin backtest:

Model MAE / baseline MAE / skill
: Point accuracy of the median (p50) forecast against persistence.

Directional accuracy
: Fraction of predictions whose *sign* relative to the last observed value matched the
  actual move. Reported separately from MAE because a forecaster can be directionally
  useful and numerically poor, or the reverse.

Calibration coverage
: Fraction of actuals falling inside the outermost quantile band. With the default
  p10/p50/p90 the nominal figure is 80%. Coverage far above nominal means the band is too
  wide; far below means overconfident.

Per-quantile exceedance and pinball loss
: Coverage tells you the band is wrong; exceedance tells you *which edge* is wrong. Pinball
  loss is the proper scoring rule the model is actually trained on.

CRPS, Winkler score, PIT
: A proper score over the whole predictive distribution, an interval score that charges for
  *width* as well as coverage, and the probability-integral-transform values behind a
  calibration histogram. All three exist because coverage alone cannot distinguish a
  well-calibrated band from an absurdly wide one — a band from −∞ to +∞ scores 100%
  coverage.

(split-geometry)=
## Split geometry

This is where most of the historical defects lived, so it is worth stating precisely. A
single `train` run lays the timeline out left to right:

```
[ training .. train_cutoff ][ purge+embargo ][ selection ][ purge+embargo ][ calibration ][ validation ]
                                                                                           ^ scored here
```

- **`validation_days`** (default 60 trading days) sets the scored holdout — what every
  reported `EvaluationMetrics` comes from. It is *not* one horizon. A holdout of exactly one
  horizon admits exactly one window, and that is where "directional accuracy 100.0%" came
  from — it was 5/5. At the default 5-day horizon a 60-day holdout yields `60 − 5 + 1 = 56`
  windows, or 280 individual predictions. The validation set is built with
  `min_prediction_idx = cutoff + 1` rather than `predict=True`, so *every* window after the
  cutoff is scored.
- **`selection_days`** (default 30 trading days, PYQ-143) sets a *second*, earlier holdout
  that `EarlyStopping` and `ModelCheckpoint` monitor instead. Before this existed, the
  scored window above was the same window checkpoint selection watched — the best of up to
  `max_epochs` epochs chosen against the exact data later reported as "the" metrics, a
  selection-event bias identical in kind to the one `tune()`'s own held-out split exists to
  avoid for Optuna trials. Every reported metric got worse when this landed (see the
  resolution note on PYQ-143); that was expected, not a regression.
- **`purge_horizon` / `embargo_days`** shrink the *training* slice, and now also the gap
  either side of `selection`, never the scored window itself. See
  {ref}`invariant 10 <invariant-purge-embargo>`.
- **`calibration_days`** carves out a slice between `selection` and the scored window, used
  solely to fit the conformal offset — out-of-sample for training and selection, and
  disjoint from what the model is later judged on.

`walk_forward_backtest()` repeats the whole thing at rolling origins, training a fresh
model per origin and discarding it. Each origin is scored on *its own* out-of-sample window
starting at `cutoff + 1`; consecutive origins evaluate disjoint windows. That sounds
obvious and was wrong for a long time — see
{ref}`invariant 8 <invariant-walk-forward>`.

A backtest reports **per-window** metrics as well as the aggregate, because the spread
across time is the reason to run more than one window. A single mean over five origins
hides whether the model is consistently mediocre or wildly unstable.

(per-horizon)=
## Per-horizon breakdown

Every metric above is also a mean over decoder steps h=1..horizon, which hides a second kind
of structure (PYQ-267): persistence is hardest to beat at h=1 and progressively less so as h
grows, so a model that has learned something should show skill *increasing* with horizon,
while one that is only tracking the last close should show the opposite. A flat headline
number and a profile like `[-60%, -35%, -10%, +5%, +15%]` are the same mean and very
different findings. `EvaluationMetrics.per_horizon` (one `PerHorizonMetrics` per decoder
step, pooled position-wise across windows the same weighted way the aggregate is) now carries
this; `--format json` includes it, and `train`/`backtest` print it as a "Per-horizon
breakdown" table whenever the horizon exceeds one step.

The same applies to calibration: a 99.3% coverage on a nominal 80% band could be ~100% at
h=1 (the band is far too wide where uncertainty is smallest) decaying toward nominal at h=5,
or flat across every step — two different pathologies with the same headline number, and
`investigations.md#pyq-324` (see {ref}`related-open-questions`) is the open question of which
one this project's own band shows.

**This section does not yet show the actual profile** for the default or `log_return`
configurations, unlike the rest of this document's measured numbers — no live vendor-data
access was available in the pass that added `per_horizon` to re-run `pyquant train`/`backtest`
and capture it. Filling in the real profile for both configurations (ideally after PYQ-143's
geometry fix has also been re-measured, since both landed in the same pass) is the natural
next step, not a placeholder to leave standing.

(baselines-beyond-persistence)=
## Baselines beyond persistence

Every skill number on this page, including the headline −23.5%, is relative to a single
comparator: persistence ("predict no change"). Persistence is uniquely favourable to the
null on a near-random-walk level series — it is close to optimal by construction, which is
the whole of {ref}`the negative-result reading <negative-result>` above. Failing to beat it
is therefore weak evidence that the model learned nothing; a drift baseline, a seasonal-naive
baseline and a simple autoregressive model would each fail differently, and the *pattern* of
which baselines a model beats is far more diagnostic than one signed number (PYQ-275).

`analysis/baselines.py` now provides four more point-forecast comparators alongside
persistence — random-walk-with-drift, seasonal-naive (default 5-day/weekly season),
climatological (the historical mean, flat across the horizon), and a hand-rolled AR(1) fit
per sample by closed-form OLS. The AR(1) baseline is deliberately not `statsmodels`'
ARIMA/ETS: `statsmodels` is already an optional dependency (the `tuning` extra), but this
module sits on `train`/`backtest`'s core path rather than only `tune`'s, and a per-window
ARIMA fit is materially slower than a closed-form AR(1) for a comparator whose entire point
is to be a cheap floor — declined per non-negotiable #5, not overlooked.

`EvaluationMetrics.baseline_maes` now carries MAE against every one of the five,
`strongest_baseline` names whichever has the lowest MAE (the hardest for the model to beat,
and the honest one to headline skill against rather than the weakest available comparator),
and `train`/`backtest`'s Rich tables and `--format json` output both surface the full row.

**None of this page's headline numbers are re-stated against the fuller baseline set yet.**
The −23.5%/+2.4% figures above were measured before this module existed, and there was no
live vendor-data access available in the pass that added it to re-run either configuration.
Re-running both under `pyquant backtest` and reporting skill against each baseline — and
specifically against whichever is strongest for that symbol — is the natural next step.

(seed-variance)=
## Seed variance

Every number on this page is one draw from one seed: `TrainingConfig.seed` is fixed at 42,
and until now nothing measured how much of any reported delta is run-to-run initialisation
noise rather than signal. `TrainingConfig.seeds: list[int]` (default `[42]`, so nothing
changes unless a caller opts in) and `models.tft.walk_forward_backtest_multi_seed()` now
repeat a walk-forward backtest once per seed and report skill as mean ± sd (min, max) across
them; `pyquant backtest SYMBOL --seeds N` is the CLI surface, expanding to seeds `0..N-1`
(`--format json` carries every individual seed's result, not only the summary, so a pair of
seeds can be fed into `compare_backtests`).

Deliberately scoped to `backtest` only, not `train`: a multi-seed **backtest** needs nothing
extra to make sense (each seed is an independent, disposable fit, the existing
`walk_forward_backtest` semantics). A multi-seed **train**, by contrast, would fit N models
and has to decide which one's weights actually get deployed in the persisted bundle — a real
product decision this pass did not make. `TrainingConfig.seeds` is still recorded in every
bundle's `meta.json` (generic whole-config recording already does this), but `train()` itself
does not loop across it.

**No seed sweep has been run against live data yet.** This section describes the tool, not a
result — investigations.md#pyq-321 is the open question this tool exists to answer, and is
where the actual seed-to-seed standard deviation, once measured, will be reported.

(sample-size)=
## Sample size

Every rate prints with the denominator behind it — `Evaluated on 56 windows (280
predictions)` — and `--format json` emits both. This is not a formatting preference. The
project's previous headline claim was "directional accuracy 100.0%", which was 5/5 on a
single validation window, printed to one decimal place with no denominator. Reported that
way it read as a strong result when it was noise.

Two consequences worth internalising:

- **Overlapping windows are not independent observations.** 56 windows over a 60-day
  holdout at a 5-day horizon share most of their days. `EvaluationMetrics.effective_n_samples`
  reports the approximate number of *non-overlapping* windows, which is the honest
  denominator for a confidence interval.
- **Aggregation weights rates by their window's point count** and sums the counts. Summing
  the denominator while unweighted-averaging the numerator computes the two halves of a
  fraction differently, and reads as a pooled figure it is not.

(negative-result)=
## What the negative result probably means

The most likely explanation is **not** hyperparameters. `dataset.TARGET = "Close"` means
the model predicts the price *level*, and is scored against a baseline that predicts the
last close. For a near-random-walk level series that baseline is close to unbeatable by
construction, so −23.5% skill may be roughly what this formulation predicts *a priori*,
largely independent of learning rate or capacity. That reframing is why learning-rate
tuning was demoted out of the project's shortlist: it optimises inside a formulation whose
ceiling is approximately "tie the baseline".

The second finding is the band. 99.3% coverage on a nominal 80% interval is not a
near-miss — it means the interval is so wide it is close to uninformative, and it silently
disables the `scan` command's "is the whole band on one side of zero" guard, collapsing
BUY/SELL into permanent HOLD.

Both of these are more interesting than a tuned-up number would be, and neither was
visible until the sample size was fixed. Worth stating plainly: **a negative result stated
precisely is more useful than a positive one stated vaguely**, and the numbers above
replaced an earlier, wrong "+64.9% skill, 100.0% directional accuracy" that came from
scoring five points of a discarded checkpoint.

## How to reproduce

```bash
uv run pyquant train NVO                      # prints the evaluation table
uv run pyquant backtest NVO --windows 5       # per-window spread + aggregate
uv run pyquant --format json backtest NVO --windows 5 | jq '.aggregated'
```

Reproducibility rests on three legs, all recorded in the bundle's `meta.json`:

1. the **seed** (`TrainingConfig.seed`, passed to `seed_everything` before any data
   loading or weight init);
2. a **pinned dataset** — `train --pin NAME` snapshots the assembled panel TTL-exempt, so a
   later run replays byte-identical data rather than whatever is live that day;
3. the **code version** — package version plus git sha, because feature definitions do
   change; the cache fingerprint includes it so a pin cannot be replayed across an
   incompatible implementation.

A change that breaks any one of those needs a ticket.

**Verified, not just claimed (PYQ-246):** two consecutive `train()` calls with the same
seed against the same pinned data produce bit-identical `val_loss` and every
`EvaluationMetrics` field, on this project's CPU-only test environment — checked with both
`num_workers=0` (the default) and `num_workers=2`. This was previously only tested as "the
seed is passed and recorded," not as "the run is actually reproducible," which is a real
gap: `seed_everything` does not by itself guarantee determinism on every backend. **GPU
determinism is untested** — there is no GPU in the environment this was verified on, and
cuDNN autotuning is a real, materially different nondeterminism source a CPU-only check
cannot see. Do not assume bit-identical reproducibility across GPU-trained bundles in
`runs.jsonl` on the strength of this result; it has only been shown on CPU.

## Caveats on the headline numbers

- The default-config reference run was **3 epochs**; the log-return comparison held epoch
  budget, seed and window count fixed at whatever PYQ-247's run used. Neither is a verdict
  on model quality — the point of both is that the numbers are measured on enough data to
  mean something, at the scale that was actually feasible to run.
- 56 overlapping windows on one symbol (default config) and 25 predictions on one symbol
  (log-return config) are both small, serially-correlated samples. Treat the figures as
  directional — "the default doesn't obviously work, the alternative might" — not as
  precise effect sizes. `EvaluationMetrics.effective_n_samples` is the honest denominator
  for either.
- Coverage and skill are reported over the *whole* holdout, not conditioned on regime.
- MAE in price space is denominated in dollars, so it is not comparable across symbols;
  the log-return target fixes that as a side effect, which is one more argument for it
  beyond the skill number itself.
- **Neither PYQ-247's +2.4% nor `investigations.md#pyq-316`'s sentiment delta has a
  confidence interval attached here.** `analysis.metrics.compare_backtests` (PYQ-266) now
  exists to do this properly -- a paired moving-block bootstrap over the per-window skill
  *difference* between two configurations scored on the same walk-forward windows, which is
  strictly more powerful than eyeballing two marginal point estimates. It was not run against
  either finding in this pass: both were originally measured before `BacktestResult` recorded
  window origins, so their raw per-window arrays cannot be verified to align, and there was no
  live vendor-data access available to re-run either comparison from scratch. Re-running both
  under `compare_backtests` is the natural next step, and is what
  `investigations.md#pyq-322`'s pre-registered decision rule should require before either
  finding is allowed to flip a default.

## Landed since this page was first written

Three changes were "in flight" here as of commit `a7a2b5f`; all three have since shipped,
with measured before/after numbers rather than being merely described:

- **Log-return target** (PYQ-247) — `TrainingConfig.target: Literal["close", "log_return"]`.
  The headline box above is that measurement. `TrainingConfig.target` still defaults to
  `"close"`: the result needs a multi-symbol repeat before it earns a default change, per
  non-negotiable #1. See {ref}`related-open-questions`.
- **Conformal calibration of the band** (PYQ-248) — a distribution-free split-conformal
  offset, fitted on a calibration slice disjoint from both training and the scored
  validation window. Verified to pull a 100%-coverage band to within 5 points of nominal,
  and to *widen* a too-narrow one rather than only shrinking. `TrainingConfig.
  calibration_days` still defaults to `0` (off): PYQ-247 showed the 99.3%-on-80% pathology
  is largely a property of the price-level target, not something calibration alone should
  paper over, so turning it on is deferred to the same multi-symbol decision as the target
  itself.
- **Purged and embargoed splits** (PYQ-250) — see
  {ref}`invariant 10 <invariant-purge-embargo>`, now unconditionally part of every split
  (`purge_horizon` defaults to one horizon, `embargo_days` to 2). This is the one change of
  the three that is *always on*: the +2.4%→+3.8% jump in the log-return comparison above is
  purging removing an optimistic bias that survived every earlier measurement on this page,
  the default-config −23.5% included.

(related-open-questions)=
## Related open questions, same discipline applied elsewhere

Three further investigations repeated PYQ-247's "measure a controlled comparison, report
the sample size, don't move a default on it" approach against other assumptions this
project used to state as rationale rather than result:

- **Is pooling actually helping?** (`investigations.md#pyq-315`) — measured *worse*, not
  better, on an AAPL+ARM comparison. The README's pooling section now states the measured
  numbers.
- **Which features earn their place?** (`investigations.md#pyq-316`) — a feature-group
  ablation found technicals added ~nothing over price alone, and sentiment *hurt*
  measurably, consistent with `bugs.md#pyq-140`'s finding that Finnhub's free tier delivers
  ~6 days of news, not 365.
- **Does `explain` mean what it claims?** (`investigations.md#pyq-314`) — permutation
  importance agrees with the TFT's own variable-selection weights on the single top
  feature, but only weakly beyond it. `explain` now prints a caveat when a bundle's
  recorded skill is non-positive, tying interpretation confidence to model quality rather
  than presenting both with equal authority.

All three share the same caveat as the headline log-return result: one run, small samples,
directional rather than definitive. The pattern across all four is itself worth noting —
every one of them moved in the direction of "the rationale was optimistic," which is why
this project's non-negotiable #1 exists.

(decision-rule)=
## What it takes to flip a default

Three findings above each recommend a default change and each declines to make it, for the
same stated reason: one symbol, tens of points, one run. Each names "a multi-symbol repeat"
as the prerequisite and none says how many symbols, how many windows, or how large a
difference (investigations.md#pyq-322). `backlog/README.md`'s `## Now` list carried the
resulting item at #1 across two review passes without it being started — an unspecified
threshold is not a conservative decision rule, it is a deferred one: it can never be met, so
the finding sits forever, or it gets met retrospectively by whichever run happens to look
convincing, which is the exact move non-negotiable #1 exists to prevent. This section settles
it once, using the instruments PYQ-265/266/268 built in the same pass that asked the
question.

**To flip any of `TrainingConfig.target`, `DataConfig.use_sentiment`, or whether pooling
defaults on:**

1. **Coverage.** N ≥ 10 symbols, spanning at least 3 distinct sectors — not ten names from
   one industry, which would be one bet dressed as ten.
2. **Per-symbol evidence.** Enough walk-forward windows that
   `EvaluationMetrics.effective_n_samples` (PYQ-251) is ≥ 10 for *each* symbol/arm cell — the
   project's existing 60-day-validation default already clears this (~12 effective windows),
   so this is "don't shrink below today's own bar," not a new number invented for this rule.
3. **Seed floor.** K ≥ 5 seeds per (symbol, arm) cell, run via
   `models.tft.walk_forward_backtest_multi_seed` (PYQ-265). This is a floor, not necessarily
   sufficient — investigations.md#pyq-321 measures the actual seed-to-seed standard
   deviation, and if that turns out to be large relative to the effect sizes below, K must
   rise until it isn't; that finding supersedes the "5" here when it lands, per the
   supersession discipline this backlog already uses for corrected numbers.
4. **The statistical bar.** `analysis.metrics.compare_backtests`'s paired interval (PYQ-266)
   on the arm-vs-arm skill difference, computed *per symbol* on that symbol's own walk-forward
   windows, must exclude zero. Two overlapping marginal intervals do not qualify — that is
   exactly the weaker, wrong instrument PYQ-266 replaced.
5. **The mixed case — the one most likely to actually happen.** An arm that helps 11 symbols
   and hurts 4 is not "helped, on net." A default flips only if **both** (a) the pooled paired
   comparison across all N symbols excludes zero, **and** (b) the arm's per-symbol paired
   interval excludes zero in the favourable direction for at least 60% of symbols with a
   per-symbol-significant result, **and** (c) no covered sector hurts on every one of its
   symbols (a clean per-sector failure is evidence of a real interaction a pooled number
   would hide, and blocks the flip even if (a) and (b) clear). Failing this test is not
   "inconclusive, gather more data" — it is a **named result**: the effect is real but
   symbol-dependent, gets written up as such, and the config becomes a documented **non-default
   option**, not a new default.
6. **The bar scales with blast radius, not just the arm.** Equal evidence bars for unequal
   consequences is the wrong answer:
   - **Low blast radius** (`use_sentiment` off — removes a feature, changes nothing about
     what an existing bundle means, fully reversible): the rule above applies as stated.
   - **High blast radius** (`target` → `log_return` — redefines what every bundle predicts,
     making older ones non-comparable the way PYQ-121 did for a single feature): the same
     statistical bar, **plus** N ≥ 15 rather than 10, **plus** an explicit supersession plan
     naming which existing tickets and docs get marked `Superseded by PYQ-XXX` before the
     flip lands — decided in advance, not improvised after.
   - **Pooling on by default:** investigations.md#pyq-315 already measured pooling *worse*.
     Turning it on by default would be a first-time change against standing negative
     evidence, so it takes the high-blast-radius bar; leaving it off takes nothing further,
     since the status quo already matches the evidence.

This is deliberately conservative in the "no free upgrades" direction: it is written to be
hard to satisfy by accident, and easy to point to when a run *does* satisfy it. If it turns
out to be miscalibrated once real multi-symbol data exists — too strict to ever pass, or
loose enough that a marginal result clears it — say so and revise the rule explicitly, the
same way a wrong headline number gets corrected loudly rather than quietly (non-negotiable
#1), rather than working around it informally.
