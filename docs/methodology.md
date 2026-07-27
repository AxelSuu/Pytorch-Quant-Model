# Methodology

How PyQuant decides whether its forecasts are any good, and what the current numbers do
and do not license you to conclude.

:::{admonition} Headline result
:class: warning

Over **56 walk-forward windows / 280 predictions** at the default configuration:

| Metric | Value | Reading |
|---|---|---|
| Skill vs. persistence baseline | **−23.5%** | The model's MAE is 23.5% *worse* than "predict no change". |
| Directional accuracy | **57.5%** | Slightly better than a coin flip, on 280 points. |
| Calibration coverage (p10–p90) | **99.3%** | A nominal 80% band that contains 99.3% of outcomes is far too wide to be useful. |

Measured on commit `a7a2b5f`, `pyquant train NVO` at defaults. **The forecaster does not
currently beat a naive persistence baseline.** That is the project's central open problem,
and it is reported rather than tuned out of sight.
:::

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

## Split geometry

This is where most of the historical defects lived, so it is worth stating precisely. A
single `train` run lays the timeline out left to right:

```
[ training ................ train_cutoff ][ purge + embargo ][ calibration ][ validation ]
                                                                             ^ scored here
```

- **`validation_days`** (default 60 trading days) sets the scored holdout. It is *not* one
  horizon. A holdout of exactly one horizon admits exactly one window, and that is where
  "directional accuracy 100.0%" came from — it was 5/5. At the default 5-day horizon a
  60-day holdout yields `60 − 5 + 1 = 56` windows, or 280 individual predictions.
- The validation set is built with `min_prediction_idx = cutoff + 1` rather than
  `predict=True`, so *every* window after the cutoff is scored, and that same loader drives
  `EarlyStopping` and `ModelCheckpoint`.
- **`purge_horizon` / `embargo_days`** shrink the *training* slice only, never the scored
  window. See {ref}`invariant 10 <invariant-purge-embargo>`.
- **`calibration_days`** carves out a slice between the two, used solely to fit the
  conformal offset — out-of-sample for training and disjoint from what the model is later
  judged on.

`walk_forward_backtest()` repeats the whole thing at rolling origins, training a fresh
model per origin and discarding it. Each origin is scored on *its own* out-of-sample window
starting at `cutoff + 1`; consecutive origins evaluate disjoint windows. That sounds
obvious and was wrong for a long time — see
{ref}`invariant 8 <invariant-walk-forward>`.

A backtest reports **per-window** metrics as well as the aggregate, because the spread
across time is the reason to run more than one window. A single mean over five origins
hides whether the model is consistently mediocre or wildly unstable.

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

## Caveats on the headline numbers

- The reference run was **3 epochs**. It is not a verdict on model quality; the point is
  that the numbers are now measured on enough data to mean something.
- 56 overlapping windows on one symbol is a small, serially-correlated sample. Treat the
  figures as "the model is not obviously working", not as a precise effect size.
- Coverage and skill are reported over the *whole* holdout, not conditioned on regime.
- MAE in price space is denominated in dollars, so it is not comparable across symbols.
  Return-space metrics fix that, which is one more argument for the target change below.

## In flight

Three changes that bear directly on this page are landing separately and are **not**
reflected in the numbers above, which are as of commit `a7a2b5f`:

- **A log-return target** — `TrainingConfig.target` accepts `"log_return"`, making the
  baseline "predict zero return", which is beatable in principle. The result may well be
  that skill stays near zero; that is a legitimate outcome and will be recorded rather than
  tuned away.
- **Conformal calibration of the band** — a distribution-free split-conformal offset fitted
  on the calibration slice above, which directly attacks the 99.3%-on-80% figure without
  retraining. `TrainingConfig.calibration_days` defaults to `0`, i.e. off, precisely because
  switching it on changes every reported coverage number and that has to be a measured,
  deliberate change rather than a silent one.
- **Purged and embargoed splits** — see
  {ref}`invariant 10 <invariant-purge-embargo>`.
  Expect reported performance to get *worse*, not better: purging removes an optimistic
  bias.

When those land with measured before/after numbers, this page's headline table is what
needs updating — and per the project's first non-negotiable, a number here may not improve
without the model improving.
