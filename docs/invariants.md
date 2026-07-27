# Leakage invariants

Seven look-ahead leaks have been found and fixed in this pipeline. Every one of them was
**correct in each individual file and wrong across files**: the vendor module used the
right timestamp, the split arithmetic used the right variable, the plotting code used the
right dates — and the composition was still wrong. That is the signature to look for, and
it is why this page exists as prose rather than only as scattered regression tests.

Each invariant below is stated as a **falsifiable claim**: something you could write a
script to disprove in an afternoon. Each links to the ticket that established it and the
test that now guards it. If you change anything under `data/`, or anything that decides
split geometry, the question to ask out loud is:

> Could a row at time *t* now see information that did not exist at time *t*?

Twice, the whole bug was answered by asking what `decoder_time_idx` actually contains.

:::{note}
This page and PYQ-238's proposed `tests/test_invariants.py` are the same content in two
forms. Until that module lands, the guarding tests are spread across the suite and named
below. If you add an invariant, add it in both places.
:::

## At a glance

| # | Invariant | Ticket | Guarded by |
|---|-----------|--------|------------|
| 1 | A macro value appears only on rows dated at or after its first publication | [PYQ-101][b101] | `test_fetch_macro_uses_the_first_published_cpi_vintage` |
| 2 | Indicator warm-up rows are genuinely NaN and are dropped, never filled | [PYQ-103][b103] | `test_add_technical_indicators_leaves_warmup_rows_genuinely_nan` |
| 3 | `predict=True` decodes timesteps strictly after the last observed bar | [PYQ-115][b115] | `test_prediction_decoder_covers_steps_after_last_observed_bar` |
| 4 | The prediction encoder ends on the last observed bar | [PYQ-115][b115] | `test_prediction_encoder_ends_on_the_last_observed_bar` |
| 5 | Pooled symbols share one calendar: same `Date` ⇒ same `time_idx` | [PYQ-116][b116] | `test_align_time_index_maps_the_same_date_to_the_same_index` |
| 6 | Every validation decoder index exceeds the training cutoff, for every group | [PYQ-116][b116] | `test_pooling_date_aligns_symbols_with_unequal_history` |
| 7 | No leading row is filled from a later value | [PYQ-123][b123] | `test_build_panel_does_not_backfill_a_late_starting_source` |
| 8 | Consecutive walk-forward origins evaluate disjoint windows, each starting at `cutoff + 1` | [PYQ-127][b127] | `test_walk_forward_window_validation_targets_its_own_origin` |
| 9 | A headline is attached to the first session that could still trade on it | [PYQ-129][b129] | `test_build_panel_lands_post_close_news_on_the_next_trading_row` |
| 10 | No training sample's decoder reaches into the scored window | [PYQ-250][f250] | `test_no_training_decoder_overlaps_the_validation_window_at_any_origin` |

Four further invariants are not about leakage but are equally load-bearing; they are in
[Non-leakage invariants](#non-leakage-invariants) at the bottom.

---

## The leak family

### 1. A macro value appears only on rows dated at or after its first publication

**Claim.** For every FRED series and every trading row *t*, the value joined at *t* was
publicly released on or before *t* — and it is the value as *first released*, not a later
revision.

**How it broke.** [PYQ-101][b101] (Critical). `fredapi`'s `get_series()` indexes values by
their economic **reference period**: `CPIAUCSL` dated 2026-06-01 is June's CPI, which BLS
does not publish until mid-July. `fetch_macro()` forward-filled that straight onto a daily
grid and `build_panel()` reindexed it onto the trading calendar with `method="ffill"`. No
shift was applied anywhere, so a row for June 5th saw a number that did not exist until
mid-July. Every run with `use_macro=True` (the default) and a `FRED_API_KEY` set was
affected. Nothing in `macro.py` looked wrong on its own — the leak was the join.

The first fix was a per-series `publication_lag_days`, establishing a convention
([PYQ-305][i305]). PYQ-257 then replaced the constant lag with real ALFRED vintages, which
also removes the *revision* leak: a lag still hands a historical row today's revised
figure rather than the number that was actually on the wire.

**Guarded by** `tests/test_macro.py::test_fetch_macro_uses_the_first_published_cpi_vintage`
— two release rows for the same reference date assert the row is NaN before the release
date and carries the *first* published value afterwards.

### 2. Indicator warm-up rows are genuinely NaN and are dropped, never filled

**Claim.** No panel row contains a rolling-window indicator computed from fewer
observations than that indicator's window, and no such row is imputed from a later value.

**How it broke.** [PYQ-103][b103] (High). `build_panel()` read
`panel.ffill().bfill().dropna()` under a comment claiming it dropped indicator warm-up
rows. It could not: `bfill()` had already back-filled every leading NaN from the first
valid observation, so `dropna()` had nothing left to remove. Measured on a synthetic
49-row warm-up column: **0 rows removed**. The first ~49 rows of `SMA_50` were a constant,
fabricated value silently present in the earliest training sequences.

The fix landed one layer above where the ticket pointed:
{py:func}`~pyquant.data.prices.add_technical_indicators` leaves warm-up rows as real NaN,
and `build_panel()` calls `dropna()` *immediately* after `fetch_prices()` — before any
other source is joined, so no downstream fill logic can launder them.

The same invariant has since caught two more defects. PYQ-132 found EMA/MACD warm-up rows
surviving `dropna()` because `ewm(min_periods=...)` was not set, and PYQ-137 found that
even with `min_periods` the first surviving EMA rows carry ~0.08% seed bias, which is why
the warm-up spans are configurable rather than fixed.

**Guarded by** `tests/test_prices.py::test_add_technical_indicators_leaves_warmup_rows_genuinely_nan`,
`tests/test_dataset.py::test_build_panel_drops_indicator_warmup_rows`, and
`tests/test_prices.py::test_panel_warmup_is_decided_by_the_longest_window_not_by_sma_50`.
That last name is deliberate: hardcoding *which* indicator binds is exactly what let
PYQ-121 and PYQ-132 hide behind `SMA_50`.

### 3. `predict=True` decodes timesteps strictly after the last observed bar

**Claim.** `decoder_time_idx.min() > df["time_idx"].max()` for the observed frame — every
predicted step is a day that has not happened yet.

**How it broke.** [PYQ-115][b115] (Critical — the widest blast radius in the project).
`TimeSeriesDataSet.from_parameters(..., predict=True)` selects, per group, the window whose
*decoder* covers the final `max_prediction_length` timesteps of the frame it is handed. It
does not extrapolate past the end. Handed a frame ending at the last observed bar, it
therefore re-predicted the last five **already-observed** days:

```
decoder_time_idx : [146, 147, 148, 149, 150]
decoder target   : [109.51, 110.36, 111.27, 113.70, 115.23]
actual Close     : [109.51, 110.36, 111.27, 113.70, 115.23]   <- identical
```

So `Expected (5d): +2.73%` was not a forecast; it was a model residual on known prices.
Meanwhile the PNG export labelled its x-axis with genuinely future business dates, so the
chart asserted dates the numbers did not correspond to.

The fix is {py:func}`~pyquant.data.dataset.extend_for_prediction`, which appends `horizon`
future rows per symbol (future `Date`, contiguous `time_idx`, recomputed `dow`/`month_num`,
last observed row carried forward for the unknown reals the decoder never reads) before the
prediction dataset is built. {py:func}`~pyquant.data.dataset.future_business_dates` is the
single source of truth for *which* dates those are.

Live before/after on the same NVO bundle (last bar 2026-07-23): `+2.73%` off medians for
2026-07-17..23 became `−5.31%` off medians for 2026-07-24..30. The sign flip is the point.

**Guarded by** `tests/test_tft.py::test_prediction_decoder_covers_steps_after_last_observed_bar`.

### 4. The prediction encoder ends on the last observed bar

**Claim.** The encoder window handed to `interpret()` ends exactly on the final real
observation — so attention weights line up with the dates they are labelled with.

**How it broke.** Same ticket, [PYQ-115][b115]. `explain` shares the prediction dataset, so
before the fix the importances and attention described a window ending `horizon` days
*before* the last bar, while `attention_to_series()` labelled it with the last *n* panel
dates. Verified off by exactly five days: the encoder ended 2024-09-30 and was labelled as
ending 2024-10-07.

This is stated separately from invariant 3 because they can break independently: appending
too many future rows would satisfy 3 and break 4.

**Guarded by** `tests/test_tft.py::test_prediction_encoder_ends_on_the_last_observed_bar`.

### 5. Pooled symbols share one calendar: same `Date` ⇒ same `time_idx`

**Claim.** In a pooled long frame, `df.groupby("Date")["time_idx"].nunique().max() == 1`.

**How it broke.** [PYQ-116][b116] (Critical). `panel_to_long()` numbers each symbol's rows
from zero independently, so `time_idx = t` meant a *different calendar date* for every
pooled symbol. Two consequences, one obvious and one not:

- Groups were aligned by position rather than by date, so a shared market shock landed at a
  different index in every group and could not be learned cross-sectionally — which is most
  of the stated reason to pool at all.
- `train()` derives `cutoff` from the **global** maximum `time_idx`, so a short-history
  symbol's entire series sat inside the training slice.

{py:func}`~pyquant.data.dataset.align_time_index` re-maps `time_idx` onto the union
calendar of every pooled symbol. `make_dataset()` already sets `allow_missing_timesteps=True`,
which absorbs the per-symbol gaps this creates.

**Guarded by** `tests/test_dataset.py::test_align_time_index_maps_the_same_date_to_the_same_index`
and `tests/test_dataset.py::test_align_time_index_leaves_a_single_symbol_unchanged`.

### 6. Every validation decoder index exceeds the training cutoff, for every group

**Claim.** For *each* symbol in a pooled run, the first index of its validation window is
greater than the training cutoff. Not "on average", not "for the longest symbol".

**How it broke.** Same ticket, [PYQ-116][b116], and this is the half that corrupted the
numbers. With per-symbol indices, `pyquant train AAPL,ARM` at the default `period="5y"`
produced:

```
per-symbol max time_idx: {'LONG': 250, 'SHORT': 90}
global cutoff:           245
  val sample group=LONG  decoder=[246..250]  ALSO IN TRAINING: False
  val sample group=SHORT decoder=[ 86.. 90]  ALSO IN TRAINING: True   <- leak
```

The leaked window drives `val_loss`, which drives `EarlyStopping`, `ModelCheckpoint`
selection, and the validation loss the CLI reports — so the corruption propagates into
which model you actually deploy.

Date alignment fixes symbols that start late. A symbol whose data *stops* early — a
delisting, or a stale feed — still has its validation window inside training and nothing in
the dataset machinery notices, so `_warn_on_stale_symbols()` names it rather than reporting
an optimistic `val_loss` for it.

**Guarded by** `tests/test_tft.py::test_pooling_date_aligns_symbols_with_unequal_history`
and `tests/test_tft.py::test_train_warns_when_a_symbols_history_ends_before_the_cutoff`.

### 7. No leading row is filled from a later value

**Claim.** For every joined source, no panel row dated before that source's first
observation carries a value derived from an observation after it.

**How it broke.** [PYQ-123][b123] (Medium). `panel = panel.ffill().bfill()`. The `ffill()`
is correct and necessary — a joined source's trading calendar need not match the target's.
The trailing `bfill()` can *only* fire on leading NaNs, and fills them from the first
*later* value. That is look-ahead by construction, in a file that had otherwise been
deliberate about exactly this, under a comment that justified only the `ffill` half.

`build_panel()` now does `ffill()` then `dropna()`, the same policy warm-up rows get. A
column with no overlap at all with the price calendar would make `dropna()` empty the
panel, so those are dropped as *columns*, with a warning, first.

**Guarded by** `tests/test_dataset.py::test_build_panel_does_not_backfill_a_late_starting_source`.

(invariant-walk-forward)=
### 8. Consecutive walk-forward origins evaluate disjoint windows, each starting at `cutoff + 1`

**Claim.** In an *n*-window backtest, the *n* evaluated decoder windows are distinct, and
window *i* starts at `cutoff_i + 1`.

**How it broke.** [PYQ-127][b127] (High). Each window built its validation set with
`predict=True` against the **full** frame, and `predict=True` anchors the decoder to the
global last `max_prediction_length` timesteps. So every rolling origin evaluated the
identical final window:

```
current code:
  cutoff=225 -> decoder_time_idx=[246, 247, 248, 249, 250]
  cutoff=230 -> decoder_time_idx=[246, 247, 248, 249, 250]
  cutoff=235 -> decoder_time_idx=[246, 247, 248, 249, 250]
```

`backtest --windows 5` trained five models, scored all five on the same five days, averaged
the result, and labelled it "5 windows". Worse, the earliest cutoff's model was evaluated
20+ days past its training end while the latest was evaluated immediately after — so the
windows were not even comparable. Truncating the frame to `df[df.time_idx <= cutoff + horizon]`
puts each decoder on its own out-of-sample window:

```
after:
  cutoff=225 -> decoder_time_idx=[226, 227, 228, 229, 230]
  cutoff=230 -> decoder_time_idx=[231, 232, 233, 234, 235]
  cutoff=235 -> decoder_time_idx=[236, 237, 238, 239, 240]
```

This is also what makes per-window dispersion meaningful: before the fix, the spread across
"windows" measured only model-initialisation noise.

**Guarded by** `tests/test_tft.py::test_walk_forward_window_validation_targets_its_own_origin`,
which asserts both that each decoder starts at `cutoff + 1` and that three origins produce
three distinct windows.

### 9. A headline is attached to the first session that could still trade on it

**Claim.** For every article, the session it is joined to opens after the article was
published. A headline released at 17:00 ET never touches the row whose target is that day's
close.

**How it broke.** [PYQ-129][b129] (Critical). `fetch_sentiment()` bucketed each article by
its **UTC calendar date** and `build_panel()` joined that onto the trading row of the same
date. A US equity session closes at 20:00 UTC (21:00 during EST), so every headline in the
last 3–4 hours of each UTC day was post-close information attached to a row whose target is
that day's close — roughly 12–17% of each day's news window, and the *most* market-moving
slice, since post-close earnings releases land there almost by definition.

Same class as PYQ-101, in the one source [PYQ-305][i305]'s convention was never extended to,
and arguably worse: the leaked information is event-driven rather than slow-moving. Note
also the distribution of the error — roughly 80% of a default 5-year training window has
structurally-zero sentiment, so contaminated rows were a minority of *training* data but
100% of the *live* prediction path.

`session_date()` is now the single rule: convert the timestamp to `America/New_York` and
assign the headline to the next calendar date if it was published at or after the 16:00
close. Fixing only that would have converted the leak into data loss, because a calendar
date is not necessarily a session — so `align_to_sessions()` maps each dated bucket onto the
first session at or after it, pooling buckets that collide (counts add; sentiment is
averaged *weighted by headline count*). News after the last session is dropped and logged,
never rolled backwards.

Measured on the assembled panel, two headlines on Friday 2022-05-27 at 11:00 and 17:00 ET,
scored +0.6 and −0.9:

```
before:  Friday Sentiment = -0.15   (mean of both -- the 17:00 headline leaked)
         Monday Sentiment =  0.0    (no news)
after:   Friday Sentiment = +0.6    Monday Sentiment = -0.9
```

**Guarded by** `tests/test_dataset.py::test_build_panel_lands_post_close_news_on_the_next_trading_row`
— deliberately the panel-level test, because every one of the six previous leaks was correct
in each individual file — plus
`tests/test_sentiment.py::test_post_close_headline_is_assigned_to_the_next_session` and
`tests/test_sentiment.py::test_align_to_sessions_drops_news_after_the_last_session`.

**Known limitation, recorded rather than hidden:** the rule assumes a 16:00 close and no
half-days, so the ~3 early closes a year assign an afternoon headline to a session it can no
longer trade on. That is the *conservative* direction — it never leaks — so it is left as a
documented limitation.

(invariant-purge-embargo)=
### 10. No training sample's decoder reaches into the scored window

**Claim.** At every walk-forward origin, `max(training decoder index) < min(validation
decoder index)`, and the two are separated by at least `purge_horizon + embargo_days`.

**How it broke.** [PYQ-250][f250]. This one is a *design* gap rather than a regression:
even with invariants 6 and 8 satisfied, the last training samples decode the days
immediately before the split, and a validation sample starting at `cutoff + 1` reads exactly
those days through its own encoder. Training and evaluation therefore share target days
across the boundary, which biases reported out-of-sample performance optimistically.

The standard treatment (López de Prado, *Advances in Financial Machine Learning*) is to
**purge** one label horizon either side of the split and then **embargo** a further buffer,
because serial correlation carries information across the boundary even where no literal
overlap remains. {py:func}`~pyquant.models.tft.purged_training_cutoff` implements both;
`TrainingConfig.purge_horizon` and `TrainingConfig.embargo_days` control them.

Purging must shrink *training* only. If it moved the validation window too, the sample size
PYQ-117 fought for would quietly shrink with it — which is its own test.

**Guarded by** `tests/test_tft.py::test_no_training_decoder_overlaps_the_validation_window_at_any_origin`
(asserted at *every* origin, not just one, because PYQ-127's defect was that the origins
were not actually distinct),
`tests/test_tft.py::test_purge_and_embargo_shrink_the_training_slice_by_exactly_their_sum`,
and `tests/test_tft.py::test_train_still_validates_on_the_full_holdout_after_purging`.

---

## Non-leakage invariants

These are not about look-ahead, but they are the other properties a change is most likely
to break silently.

### 11. Reported metrics come from the *best* checkpoint, not the live post-fit model

`EarlyStopping` does not rewind the live model's weights to the best epoch; `ModelCheckpoint`
saves them and they must be reloaded explicitly. Before [PYQ-109][b109], `train()` and
`walk_forward_backtest()` scored the final, worse, post-early-stopping model — while
`forecast` deployed the best one. The reported number described a model nobody ever ran.

**Guarded by** `tests/test_tft.py::test_train_evaluates_best_checkpoint_not_live_model`.

### 12. Every reported rate carries its denominator

A directional accuracy of 100.0% means something very different from 5 points than from 500.
{py:class}`~pyquant.analysis.metrics.EvaluationMetrics` carries `n_samples`/`n_points`, and
`aggregate_metrics()` *sums* the counts while weighting each rate by its window's point
count — summing the denominator and unweighted-averaging the numerator computes the two
halves of a fraction differently (PYQ-136). See {ref}`Methodology <sample-size>`.

**Guarded by** `tests/test_metrics.py::test_aggregate_metrics_sums_sample_counts_rather_than_averaging_them`
and `tests/test_metrics.py::test_aggregate_metrics_weights_rates_by_each_window_point_count`.

### 13. A `Forecast` cannot exist with a crossed quantile band

`QuantileLoss` does not enforce monotonicity pointwise, so a p90 can land below a p10.
Every consumer — the forecast table, the fan charts, `scan`'s "is the whole band on one side
of zero" guard — assumes monotonic input and misbehaves quietly without it; `scan` could read
an *inverted* band as a confident BUY. {py:meth}`Forecast.__post_init__
<pyquant.analysis.forecast.Forecast>` sorts the band and records how many points had to be
reordered, so no `Forecast` can exist in a crossed state however it was constructed —
including from the planned API layer.

**Guarded by** `tests/test_cli.py::test_forecast_table_renders_a_crossed_band_monotonically`
and `tests/test_metrics.py::test_evaluate_predictions_warns_on_crossing`.

### 14. Forecast dates in the table, the JSON, the PNG and the appended rows are one set

Four consumers previously derived "which days is this for" independently, and PYQ-115 is
what happens when they disagree. {py:func}`~pyquant.data.dataset.future_business_dates` is
now the only implementation, and it skips exchange holidays rather than being Mon–Fri
(PYQ-130) — the `dow` feature the decoder reads is derived from the same dates.

**Guarded by** `tests/test_dataset.py::test_extend_for_prediction_appends_exactly_the_dates_the_forecast_reports`,
`tests/test_forecast.py::test_forecast_dates_match_the_rows_appended_for_prediction`, and
`tests/test_trading_calendar.py::test_next_sessions_returns_exactly_the_requested_count_across_holidays`.

### 15. A cached panel is never replayed across a feature redefinition

Feature definitions change — `RSI_14` was redefined once, and the EMA warm-up twice. The
cache fingerprint therefore includes the code version (package version plus git sha), so a
TTL entry or a named pin can never be reused across an incompatible implementation
(PYQ-133). Secrets never enter the fingerprint: key *presence* is recorded, key values are
not.

**Guarded by** `tests/test_dataset.py::test_cache_fingerprint_changes_with_the_package_version`
and `tests/test_dataset.py::test_cache_fingerprint_records_no_secret_values`.

[b101]: https://github.com/AxelSuu/Pytorch-Quant-Model/blob/main/backlog/bugs.md#pyq-101
[b103]: https://github.com/AxelSuu/Pytorch-Quant-Model/blob/main/backlog/bugs.md#pyq-103
[b109]: https://github.com/AxelSuu/Pytorch-Quant-Model/blob/main/backlog/bugs.md#pyq-109
[b115]: https://github.com/AxelSuu/Pytorch-Quant-Model/blob/main/backlog/bugs.md#pyq-115
[b116]: https://github.com/AxelSuu/Pytorch-Quant-Model/blob/main/backlog/bugs.md#pyq-116
[b123]: https://github.com/AxelSuu/Pytorch-Quant-Model/blob/main/backlog/bugs.md#pyq-123
[b127]: https://github.com/AxelSuu/Pytorch-Quant-Model/blob/main/backlog/bugs.md#pyq-127
[b129]: https://github.com/AxelSuu/Pytorch-Quant-Model/blob/main/backlog/bugs.md#pyq-129
[f250]: https://github.com/AxelSuu/Pytorch-Quant-Model/blob/main/backlog/features.md#pyq-250
[i305]: https://github.com/AxelSuu/Pytorch-Quant-Model/blob/main/backlog/investigations.md#pyq-305
