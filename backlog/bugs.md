# Bugs (PYQ-1xx)

Concrete, reproducible defects — see [`README.md`](README.md) for the format.
Next free ID: **PYQ-129**.

| ID | Priority | Status | Title |
|----|----------|--------|-------|
| [PYQ-101](#pyq-101) | Critical | Resolved | FRED macro series joined by reference date, not publication date (look-ahead leak) |
| [PYQ-102](#pyq-102) | High | Resolved | Target's own sector ETF column not actually dropped |
| [PYQ-103](#pyq-103) | High | Resolved | `dropna()` after `ffill().bfill()` was a no-op |
| [PYQ-104](#pyq-104) | Medium | Resolved | One malformed headline zeroed out the whole sentiment fetch |
| [PYQ-105](#pyq-105) | Medium | Resolved | `explain` fetched its data panel twice, independently |
| [PYQ-106](#pyq-106) | Low | Resolved | `Forecast.median` silently mislabels a non-median quantile |
| [PYQ-107](#pyq-107) | Low | Resolved | Dead `_NON_FEATURE` constants didn't match real column names |
| [PYQ-108](#pyq-108) | Medium | Resolved | CLI leaks third-party `warnings.warn()` output at default verbosity |
| [PYQ-109](#pyq-109) | Critical | Resolved | `train()`/`walk_forward_backtest()` evaluate the wrong (final, not best) checkpoint |
| [PYQ-110](#pyq-110) | Medium | Resolved | One failing FRED series zeroes out the whole macro fetch |
| [PYQ-111](#pyq-111) | Medium | Resolved | Pooled training silently drops a sector ETF's own column for every pooled symbol |
| [PYQ-112](#pyq-112) | Low | Resolved | `start` without `end` (or vice versa) is silently ignored in favor of `period` |
| [PYQ-113](#pyq-113) | Medium | Resolved | `scan` only catches `FileNotFoundError`; any other error crashes the whole comparison |
| [PYQ-114](#pyq-114) | Medium | Resolved | `_finbert()`'s cache permanently poisons on a transient load failure |
| [PYQ-115](#pyq-115) | Critical | Resolved | `forecast`/`explain` predict the last *already-observed* window, not the future |
| [PYQ-116](#pyq-116) | Critical | Resolved | Pooled training leaks a shorter-history symbol's validation window into training |
| [PYQ-117](#pyq-117) | High | Resolved | Every reported metric (and model selection) rests on a single 5-point sample |
| [PYQ-118](#pyq-118) | High | Resolved | A feature column missing at predict time crashes with a bare `KeyError` |
| [PYQ-119](#pyq-119) | High | Resolved | `forecast`/`explain`/`scan` ignore the config the bundle was trained with |
| [PYQ-120](#pyq-120) | Medium | Resolved | Untrained-symbol failure dumps a Rich traceback instead of a clean message |
| [PYQ-121](#pyq-121) | Medium | Resolved | `RSI_14` uses simple-MA smoothing, not Wilder's — not the standard RSI |
| [PYQ-122](#pyq-122) | Medium | Resolved | "Calibration coverage (p10-p90)" label is hardcoded regardless of configured quantiles |
| [PYQ-123](#pyq-123) | Medium | Resolved | `build_panel`'s trailing `bfill()` back-fills future values into leading rows |
| [PYQ-124](#pyq-124) | Medium | Resolved | Quantile crossing is detected but never corrected before display/signalling |
| [PYQ-125](#pyq-125) | Low | Resolved | `TrainingConfig.train_split` is dead; `DataConfig.use_options` is never read |
| [PYQ-126](#pyq-126) | Low | Resolved | `_fmt_bytes` has an unreachable final `return` |
| [PYQ-127](#pyq-127) | High | Resolved | `walk_forward_backtest` evaluates the *same* final window at every rolling origin |
| [PYQ-128](#pyq-128) | Medium | Resolved | A `--config` path that does not exist is silently ignored |

---

## [PYQ-101]
FRED macro series joined by reference date, not publication date — look-ahead leak
Status: Resolved — commit b616184, 2026-07-23
Priority: Critical
Files: `pyquant/data/macro.py` (`_fetch_fred`, `fetch_macro`), `pyquant/data/dataset.py` (`build_panel`)

Problem: fredapi's `get_series()` returns values indexed by the economic
reference period (e.g. CPIAUCSL dated 2026-06-01 is June's CPI), not the date
it was actually published. BLS publishes monthly CPI ~2-4 weeks after
month-end. `fetch_macro()` forward-filled this straight onto a daily grid,
and `build_panel()` reindexed it again onto the trading calendar with
`method="ffill"`. No lag/shift was applied anywhere, so a training row for
e.g. June 5th saw June's CPI value, which in reality wasn't known until
mid-July. DFF (daily fed funds) and T10Y2Y (same-day market data) weren't
meaningfully affected — this was specifically about CPIAUCSL and any future
low-frequency series (GDP, PCE, unemployment, etc.). This was the single most
consequential finding in the original review: it silently inflated
backtested/validation performance in a way that wouldn't hold up live, for
every run with `use_macro=True` (the default) and a `FRED_API_KEY` set.

Resolution: `macro.py`'s `_FredSeriesSpec` now carries a
`publication_lag_days` per series (CPIAUCSL: 21 days) and shifts each
series's index forward by that lag before joining. See PYQ-305 for the
convention this established for future series.

---

## [PYQ-102]
"Drop target's own sector ETF column" was commented but not implemented
Status: Resolved — commit b616184, 2026-07-23
Priority: High
Files: `pyquant/data/dataset.py` (`build_panel`, sectors join)

Problem: the comment read "Drop the target symbol's own ETF column if
present to avoid leakage of itself," but the code directly below it did a
plain join with no filtering. `DataConfig.sector_etfs` defaults to
`["XLK","XLF","XLE","XLV","XLY","SPY"]` — so `pyquant train SPY` (a very
likely first thing to try) silently got a `SEC_SPY` feature duplicating the
target's own same-day return under a different name.

Resolution: `dataset.py` now does
`sectors.drop(columns=[f"SEC_{symbol.upper()}"], errors="ignore")` before the
join.

---

## [PYQ-103]
`dropna()` after `ffill().bfill()` in `build_panel()` was a no-op — warm-up rows weren't dropped
Status: Resolved — commit b616184, 2026-07-23
Priority: High
Files: `pyquant/data/dataset.py` (`build_panel`), `pyquant/data/prices.py` (`add_technical_indicators`)

Problem: comment said "Forward/back fill any remaining gaps from joined
sources, then drop leading rows still NaN from indicator warm-up." But
`bfill()` ran before `dropna()`, and `bfill()` already back-fills any leading
NaN from the first valid observation — so by the time `dropna()` executed,
there was nothing left to drop. Verified empirically in the original review:
a 49-row synthetic warm-up column had 0 rows removed by `dropna()` after
`ffill().bfill()`. Net effect: the first ~49 rows of `SMA_50` (and similarly
other rolling-window indicators) were a constant, fabricated value borrowed
from the first real observation, silently present in the earliest training
sequences.

Resolution: the real fix landed one layer up from where the ticket pointed —
`prices.add_technical_indicators` now leaves warm-up rows as genuine NaN
(rather than filling them), and `build_panel()` calls `panel.dropna()`
immediately after `fetch_prices()`, before any other source is joined, so
warm-up rows are excluded before they can be laundered by any fill logic.

---

## [PYQ-104]
Sentiment pipeline silently discarded ALL headlines if any single article lacked a usable datetime
Status: Resolved — commit b616184, 2026-07-23
Priority: Medium
Files: `pyquant/data/sentiment.py` (`fetch_sentiment`)

Problem: `headlines` was built from every article; `dates` was built only
from articles with a truthy `datetime` field. If any single article was
missing/had a falsy datetime, the two lists ended up different lengths, and
the function returned an empty DataFrame for the whole batch — with no log
message. Indistinguishable from "no API key," "FinBERT unavailable," or "no
news found."

Resolution: `fetch_sentiment()` now filters malformed articles once, up
front (`usable = [a for a in articles if a.get("datetime")]`), derives both
headlines and dates from that same filtered list, and logs a warning with
the drop count.

---

## [PYQ-105]
`explain` CLI command fetched the data panel twice, independently
Status: Resolved — commit b616184, 2026-07-23
Priority: Medium
Files: `pyquant/cli/app.py` (`explain`), `pyquant/analysis/interpret.py` (`explain_forecast`)

Problem: `explain_forecast()` internally called `build_panel()` once to
compute the interpretation. The CLI then called `build_panel()` again,
separately, just to get `panel.index` for `attention_to_series()`. Two
independent live fetches across up to 4 network sources for one command; if
they disagreed in length or content, attention weights would zip to the
wrong dates with no error raised.

Resolution: `Interpretation` now carries `panel_index`, computed from the
single `build_panel()` call inside `explain_forecast()`;
`attention_to_series()` consumes that instead of re-fetching.

---

## [PYQ-106]
`Forecast.median` silently mislabels a non-median quantile when 0.5 isn't configured
Status: Resolved — commit b616184, 2026-07-23
Priority: Low
Files: `pyquant/analysis/forecast.py` (`Forecast.median`)

Problem: `TFTConfig.quantiles` is user-configurable and need not include
0.5. If it doesn't, a fallback of `predictions[:, len(quantiles) // 2]` picks
the middle array position, not the 50th percentile — e.g. for
`quantiles=[0.05,0.25,0.75,0.95]`, "median" would silently return the p75
forecast. This value drives `expected_return_pct()`, the CLI's "Expected"
figure, and `scan`'s BUY/SELL signal.

Resolution: `Forecast.median` (and `evaluate_predictions` in
`analysis/metrics.py`) now raise a clear `ValueError` if 0.5 isn't in the
configured quantiles, instead of guessing.

---

## [PYQ-107]
Dead constants in `dataset._NON_FEATURE` didn't match real column names
Status: Resolved — commit b616184, 2026-07-23
Priority: Low
Files: `pyquant/data/dataset.py`

Problem: `_NON_FEATURE` included `"day_of_week"` and `"month"`, but
`panel_to_long()` actually created columns named `"dow"` and `"month_num"`.
No functional bug — dow/month_num were correctly excluded via the separate
`KNOWN_REALS` check — but stale/confusing, and a future rename could
silently reintroduce calendar columns as "unknown" features.

Resolution: the two dead entries were removed.

---

## [PYQ-108]
CLI leaks third-party `warnings.warn()` output even at default verbosity
Status: Resolved — 2026-07-24 (same session, no commit yet — see git status)
Priority: Medium
Files: `pyquant/cli/app.py` (`_configure_logging`), `tests/test_cli.py`

Problem: `_configure_logging` (PYQ-207) only touched Python's `logging`
module (root level + the `lightning.pytorch` logger). The actual noisy lines
a user saw on every `train`/`forecast`/`explain` call — "Attribute 'loss' is
... already saved during checkpointing", "tensorboardX has been removed as a
dependency", "isinstance(treespec, LeafSpec) is deprecated", "does not have
many workers" — were all emitted via Python's `warnings.warn(...)`
(`UserWarning` / `PossibleUserWarning` / `DeprecationWarning` /
`FutureWarning` subclasses), a channel `logging` never touches. The project
already knew this class of noise needed suppressing — `pytest.ini`'s
`filterwarnings = ["ignore::DeprecationWarning", "ignore::UserWarning"]` and
`tests/test_tft.py`'s own `warnings.filterwarnings("ignore")` existed
specifically to keep the *test suite* clean — but that suppression was never
carried over to the actual `pyquant` CLI entry point.

Resolution: `_configure_logging` now calls
`warnings.filterwarnings("ignore", category=...)` for `UserWarning`,
`DeprecationWarning`, and `FutureWarning` (the last one needed separately —
PyTorch's `LeafSpec` deprecation is a `FutureWarning`, a *sibling* of
`DeprecationWarning`, not a subclass, so it wasn't covered by the first two
filters alone) when not `--debug`; `--debug` calls `warnings.resetwarnings()`
to restore default behavior. Covered by
`test_default_run_suppresses_user_and_deprecation_warnings` and
`test_debug_flag_restores_default_warning_filters` in `tests/test_cli.py`,
built test-first. Verified against a real `pyquant forecast AAPL` /
`pyquant train AAPL` run: clean output, `--debug` restores it.

---

## [PYQ-109]
`train()`/`walk_forward_backtest()` evaluate the live final-epoch model, not the best checkpoint — reported metrics don't match the deployed model
Status: Resolved — 2026-07-24
Priority: Critical
Files: `pyquant/models/tft.py` (`train`, `walk_forward_backtest`, `_evaluate_validation`)

Problem: Lightning's `EarlyStopping` callback stops `trainer.fit()` once
`val_loss` hasn't improved for `patience` (5) epochs, but it does **not**
rewind the live model's weights back to the best epoch — that's what
`ModelCheckpoint` is for, and reloading from it is an explicit extra step
(`Model.load_from_checkpoint(...)`, which `tft.load()` already does
correctly for `forecast`/`explain`). Both `train()` and
`walk_forward_backtest()` call `_evaluate_validation(model, val_loader, ...)`
immediately after `trainer.fit()` on the same in-memory `model` object —
i.e. on whatever epoch training happened to stop at, typically several (up
to `patience`) epochs past the actual best one. Meanwhile `train()`
separately (and correctly) saves `ckpt_cb.best_model_path` as
`bundle_dir/model.ckpt` for later `forecast`/`explain` use — so the model
that gets *deployed* and the model whose metrics get *reported* to the user
are different checkpoints entirely.

Confirmed empirically (2026-07-24) with a controlled comparison — same
trained run, same `val_loader`, only the evaluated checkpoint differs:
training stopped at epoch 25 (best was earlier, `best_model_score=3.364`).

| | Model MAE | Skill vs. baseline | Calibration coverage |
|---|---|---|---|
| FINAL (current behavior) | 9.66 | −30.5% | 20.0% |
| BEST (reloaded checkpoint, i.e. what's actually deployed) | 6.22 | **+15.9%** | **100.0%** |

This is very likely the primary explanation for the bad numbers seen in real
`pyquant train AAPL` / `pyquant backtest AAPL` runs the same session (skill
−128.9% to −504.6%, calibration 0–6.7% — see the now-answered
[investigations.md#pyq-307](investigations.md#pyq-307)): the tool had been
reporting the quality of a worse, already-discarded checkpoint, not the one
it actually hands you. It also means two experiments run while investigating
[features.md#pyq-211](features.md#pyq-211) (more epochs, and a lower
learning rate — both made the *reported* metric worse) are contaminated by
this bug and should be re-run once it's fixed before drawing conclusions
about LR/epoch tuning.

Suggested fix: in both `train()` and `walk_forward_backtest()`, after
`trainer.fit()`, reload the best checkpoint (`ckpt_cb.best_model_path`) the
same way `tft.load()` does, and evaluate/report on *that* model, not the
live post-fit one. `walk_forward_backtest()` currently has no
`ModelCheckpoint` callback at all — it needs one added, even though it
discards the model afterward, purely so each window's evaluation reflects
that window's best epoch instead of its final one.

Acceptance criteria: a regression test — train with `patience` low enough to
guarantee stopping past the best epoch on a synthetic dataset, assert the
reported `EvaluationMetrics` match an independently reloaded
best-checkpoint evaluation, not the live model's.

Resolution: a new `_evaluate_best_checkpoint()` reloads the saved best
checkpoint (as `tft.load()` does) and evaluates *that*, not the live post-fit
model. `train()` calls it on `bundle_dir/model.ckpt`; `walk_forward_backtest()`
now adds a `ModelCheckpoint` per window (into a throwaway `TemporaryDirectory`,
so no artifacts are persisted) and evaluates each window's best epoch.
Regression test `test_train_evaluates_best_checkpoint_not_live_model` asserts
the evaluated model is a distinct object from the fit live model (identical on
the pre-fix code).

---

## [PYQ-110]
One failing FRED series zeroes out the whole macro fetch — sibling of PYQ-104
Status: Resolved — 2026-07-24
Priority: Medium
Files: `pyquant/data/macro.py` (`_fetch_fred`)

Problem: `_fetch_fred()` wraps its entire `for series_id, spec in
FRED_SERIES.items(): ...` loop in one `try/except Exception`. If
`fred.get_series(...)` raises for *any single* series — e.g. CPIAUCSL hits a
transient FRED rate limit *after* DFF and T10Y2Y already fetched
successfully and were added to `cols` — the exception propagates out of the
loop, is caught by the outer `except`, and the function returns `None`
entirely. The already-successfully-fetched series in `cols` are discarded
along with the failing one. This is the exact same bug shape as PYQ-104 (one
bad item invalidates an otherwise-good batch), in a sibling code path that
wasn't caught by the same review. Confirmed via `tests/test_macro.py`:
every FRED test exercises either "all series succeed" or "no key at all" —
the partial-failure path has no test and no handling.

Suggested fix: fetch each series in its own try/except inside the loop (log
a warning and `continue` on failure, matching the pattern already used for
VIX in `_fetch_vix`), so one bad series degrades gracefully instead of
discarding everything.

Acceptance criteria: a test where `get_series` raises for CPIAUCSL but
succeeds for DFF/T10Y2Y — `fetch_macro()` should still return a DataFrame
containing `FedFunds` and `YieldSpread`.

Resolution: `_fetch_fred` now wraps each `get_series` call in its own
try/except (log a warning and `continue` on failure), with the `Fred(...)`
client construction guarded separately. One bad series degrades gracefully
instead of discarding the batch. Covered by
`test_fetch_macro_keeps_series_that_succeed_when_one_fails`.

---

## [PYQ-111]
Pooled training silently drops a sector ETF's own column for every pooled symbol
Status: Resolved — 2026-07-24
Priority: Medium
Files: `pyquant/models/tft.py` (`_build_pooled_long_df`), `pyquant/data/dataset.py` (`build_panel`, sectors join)

Problem: PYQ-102's fix makes `build_panel(symbol, ...)` drop `SEC_<symbol>`
specifically when the target *is* one of the configured sector ETFs (e.g.
building SPY's panel drops `SEC_SPY` to avoid self-leakage). `_build_pooled_long_df`
then intersects columns across every symbol in a pooled `train` call and
silently drops any column not common to all of them, logging a warning.
Combine the two: pooling any sector-ETF ticker with other tickers (e.g.
`pyquant train AAPL,SPY`) means AAPL's panel has 6 `SEC_*` columns but SPY's
has 5 (missing `SEC_SPY`) — so the pool-wide intersection drops `SEC_SPY`
*for AAPL too*, even though AAPL has no self-leakage concern and that
feature was perfectly valid for it. The net effect: adding a sector-ETF
symbol to any pooled training run silently degrades every other symbol's
feature set, beyond what either PYQ-102's fix or the pooling logic intend
individually. Confirmed untested: `test_train_pools_multiple_symbols_into_one_dataset`
in `tests/test_tft.py` pools two symbols built from the *same* schema, so
this interaction never gets exercised.

Suggested fix: decide the intended behavior explicitly rather than let it
fall out of an interaction — e.g. only drop a sector ETF's own column for
*that* symbol's rows rather than the whole pooled column set (fill the
target's own rows with NaN/0 for that one column instead of removing the
column outright), or document this as accepted behavior if pooling
sector-ETF tickers is considered rare enough not to matter.

Acceptance criteria: a regression test pooling one sector-ETF symbol with
one non-ETF symbol, asserting the intended (once decided) behavior for the
non-ETF symbol's sector features.

Resolution (behavior decided): keep the column for the symbols that
legitimately have it. `_build_pooled_long_df` now re-adds each symbol's own
`SEC_<symbol>` column as a neutral `0.0` (only when it was self-leakage-dropped
but present for a sibling) *before* the column-intersection — so the feature
survives for the other symbols while the leaking symbol's own rows carry a
harmless constant instead of the leaked value. Genuinely-flaky missing columns
(PYQ-302) still drop as before. Covered by
`test_pooling_preserves_valid_sector_column_for_other_symbols`.

---

## [PYQ-112]
`start` without `end` (or vice versa) is silently ignored in favor of `period`
Status: Resolved — 2026-07-24
Priority: Low
Files: `pyquant/data/prices.py` (`fetch_prices`), `pyquant/data/macro.py` (`_fetch_vix`), `pyquant/data/sectors.py` (`fetch_sector_returns`)

Problem: all three call sites gate on `if start and end:` (or the sectors.py
equivalent, `period=None if (start and end) else period`) before honoring an
explicit date range, falling back to `period` otherwise. If a caller passes
only `start` (no `end`), the `start` value is silently discarded and
`period` is used instead — not an error, not a warning, just quietly wrong.
Currently unreachable through the CLI (no `--start`/`--end` flags exist
anywhere), so this is a latent bug in the public function signatures
(`tft.train(..., start=, end=)` etc.) rather than something a user can hit
today — but it becomes reachable the moment PYQ-213 (API) or a future CLI
flag exposes date ranges to callers who might reasonably pass just one
bound (e.g. "everything since IPO", passing only `start`).

Suggested fix: treat `start`/`end` independently (`ticker.history(start=start,
end=end)` already accepts either being `None` on its own), or raise a clear
error if exactly one is given and the intent is ambiguous.

Acceptance criteria: a test calling `fetch_prices(..., start="2020-01-01")`
with no `end` returns data starting from `start`, not a `period`-based
range.

Resolution: all three call sites now gate on `start or end` (was `start and
end`), so an explicit range is honored whenever either bound is given — passing
just `start` no longer silently falls back to `period`. Covered by
`test_fetch_prices_honors_start_without_end`.

---

## [PYQ-113]
`scan` only catches `FileNotFoundError`; any other error crashes the whole comparison
Status: Resolved — 2026-07-24
Priority: Medium
Files: `pyquant/cli/app.py` (`scan`)

Problem: `scan`'s loop over tickers wraps `generate_forecast(ticker,
settings)` in `except FileNotFoundError` only (to render "not trained" for
an untrained symbol). Any other exception — a transient yfinance/network
error building that symbol's panel, a misconfigured `quantiles` list
(PYQ-106's `ValueError` path), anything from `build_panel`'s data sources —
propagates uncaught and crashes the entire multi-symbol comparison,
including symbols that would have resolved fine. This directly undermines
`scan`'s purpose: comparing many symbols at once should be robust to one of
them having a bad day.

Suggested fix: broaden the catch to handle expected failure modes
per-symbol (or catch `Exception` generically and render an "error" row with
the exception message), so one flaky symbol doesn't take down the report
for the rest.

Acceptance criteria: a test where one of several symbols raises a
non-`FileNotFoundError` exception from `generate_forecast` — `scan` should
still render rows for the other symbols and exit 0.

Resolution: `scan`'s per-symbol loop now also catches generic `Exception`
(after the existing `FileNotFoundError` "not trained" case), logs a warning,
and renders an `error` row for that symbol — one flaky ticker no longer sinks
the whole comparison. Covered by
`test_scan_survives_one_symbol_raising_non_filenotfound`.

---

## [PYQ-114]
`_finbert()`'s cache permanently poisons on a transient load failure
Status: Resolved — 2026-07-24
Priority: Medium
Files: `pyquant/data/sentiment.py` (`_finbert`)

Problem: `_finbert()` is decorated `@lru_cache(maxsize=1)` and returns
`None` if the `transformers` pipeline fails to construct (missing
dependency, or a transient HuggingFace Hub download hiccup on first use).
`lru_cache` caches the `None` return value just as readily as a successful
pipeline object — so a single transient failure (e.g. a network blip while
downloading `ProsusAI/finbert` the first time it's needed) permanently
disables sentiment scoring for the rest of the process's lifetime, even
though a retry moments later would likely succeed. For today's short-lived
CLI invocations the blast radius is small (worst case: sentiment is
unavailable for that one `train`/`forecast` call; the next CLI invocation is
a fresh process and gets a clean retry). It becomes a much bigger problem
for PYQ-213's proposed long-running API server, which would import
`pyquant` once and serve requests indefinitely — one bad first request would
silently disable sentiment for every request after it, for the life of the
server process.

Suggested fix: only cache successful pipeline construction (e.g. a manual
module-level cache variable set only on success, rather than `lru_cache`
over the whole function including failure), so a later call can retry after
a transient failure.

Acceptance criteria: a test where the first call to `_finbert()` raises but
the second call (same process) would succeed — the second call should
return a working pipeline, not the cached `None`.

Resolution: `_finbert()` no longer uses `@lru_cache`; it caches the pipeline in
a module-level `_FINBERT_PIPELINE` only on *successful* construction. A
transient failure returns `None` without caching it, so a later call retries.
Covered by `test_finbert_retries_after_transient_load_failure`. (This fix is a
prerequisite for the long-running API server in PYQ-213.)

---

## [PYQ-115]
`forecast`/`explain` predict the last *already-observed* window, not the future
Status: Resolved — 2026-07-26
Priority: Critical
Files: `pyquant/models/tft.py` (`_prediction_dataset`, `predict_quantiles`, `interpret`), `pyquant/analysis/forecast.py`, `pyquant/analysis/interpret.py` (`attention_to_series`), `pyquant/cli/charts.py`

Problem: `_prediction_dataset()` builds
`TimeSeriesDataSet.from_parameters(..., predict=True)` against a long df that
ends at the last observed bar. `predict=True` selects, per group, the window
whose *decoder* covers the final `max_prediction_length` timesteps — and those
are real, observed rows. It does not extrapolate past the end of the frame.
Verified on synthetic data (last `time_idx` 150, horizon 5):

```
decoder_time_idx : [146, 147, 148, 149, 150]
decoder target   : [109.51, 110.36, 111.27, 113.70, 115.23]
actual Close     : [109.51, 110.36, 111.27, 113.70, 115.23]   <- identical
dates            : 2024-10-01 .. 2024-10-07  (== "last_date")
```

So the "5-day forecast" is a retro-fit of the last week, and Day N == the last
observed bar. Consequences:

- `Forecast.expected_return_pct()` is `(model's fit of today - today's actual)
  / today's actual` — a residual, rendered as `Expected (5d): +2.73%`.
- The per-day "vs now" column compares a fitted past value against the price
  on the *same* day.
- `charts.export_fan_chart` labels the x-axis with genuinely future business
  dates via `_future_dates()`, so the PNG asserts dates the numbers do not
  correspond to.
- `explain` shares the dataset, so importance/attention describe a window
  ending `horizon` days before the last bar, while `attention_to_series()`
  labels it with the *last* n panel dates. Verified off by exactly 5: encoder
  ends 2024-09-30, labelled as ending 2024-10-07.

This is the defect with the widest blast radius in the project: every number
`forecast`, `scan` and `explain` print is affected, and it is invisible from
inside any single file — it only shows up if you ask what `decoder_time_idx`
actually contains.

Suggested fix: append `max_prediction_length` future rows per symbol to the
long df before building the prediction dataset (future `Date`, contiguous
`time_idx`, recomputed `dow`/`month_num`, last known row carried forward for
the unknown reals — which the decoder never reads). Then `predict=True`
decodes the real future. Derive the display dates from one shared helper so
the table, terminal chart, PNG export and JSON all agree.

Acceptance criteria: a test that trains a bundle, builds the prediction
dataset, and asserts `decoder_time_idx.min() > df["time_idx"].max()` — i.e.
every predicted step is beyond the last observed bar. Plus a test that
`attention_to_series()` ends on the last *observed* date.

Resolution: `dataset.extend_for_prediction()` now appends `horizon` future rows
per symbol (future `Date`, contiguous `time_idx`, recomputed `dow`/`month_num`,
last observed row carried forward for the unknown reals the decoder never reads)
before `_prediction_dataset()` builds the dataset, so `predict=True` decodes the
real future. `dataset.future_business_dates()` is the single source of truth for
which dates those are, and `Forecast.forecast_dates` exposes it -- the CLI table
gained a `Date` column, the PNG export uses it instead of its own private
`_future_dates()`, and `--format json` emits `forecast_dates`, so the three can no
longer disagree. The fix also realigns `explain`: the encoder now genuinely ends
on the last observed bar, which is what `attention_to_series()` already assumed.

Verified end-to-end against live data. Before, `pyquant forecast NVO` (last bar
2026-07-23) reported `Expected (5d) +2.73%` off medians 49.52..49.50 -- the fitted
values for 2026-07-17..23. After, the same bundle reports 2026-07-24..2026-07-30
(weekend correctly skipped) off medians 45.43..45.62, `Expected (5d) -5.31%`. The
sign flip is the whole point: the old figure was a residual on already-known
prices.

Covered by `test_prediction_decoder_covers_steps_after_last_observed_bar`,
`test_prediction_encoder_ends_on_the_last_observed_bar`, four
`extend_for_prediction`/`future_business_dates` unit tests, and two
`Forecast.forecast_dates` tests.

---

## [PYQ-116]
Pooled training leaks a shorter-history symbol's validation window into training
Status: Resolved — 2026-07-26
Priority: Critical
Files: `pyquant/models/tft.py` (`train`, `_build_pooled_long_df`), `pyquant/data/dataset.py` (`panel_to_long`)

Problem: `train()` computes `cutoff = int(df["time_idx"].max()) - horizon`
from the **global** max, but `panel_to_long()` restarts `time_idx` at 0 for
every symbol independently. A symbol with less history therefore has its own
max far below the global cutoff, so its entire series — including the window
`predict=True` later hands back as *validation* — sits inside the training
slice. Verified:

```
per-symbol max time_idx: {'LONG': 250, 'SHORT': 90}
global cutoff:           245
  val sample group=LONG  decoder=[246..250]  ALSO IN TRAINING: False
  val sample group=SHORT decoder=[ 86.. 90]  ALSO IN TRAINING: True   <- leak
```

Reachable with any pooled run mixing histories — e.g. `pyquant train AAPL,ARM`
at the default `period="5y"` (ARM listed 2023). The leak corrupts `val_loss`,
which drives `EarlyStopping`, `ModelCheckpoint` selection, and the "Validation
loss" the CLI reports.

The root cause is worth fixing rather than patching: because `time_idx` is
per-symbol positional, `time_idx = t` is a **different calendar date for each
pooled symbol**. Pooled groups are aligned by position, not by date, so a
shared market shock lands at a different index in each group and the model
cannot learn cross-sectional co-movement from it. This also makes the
README's stated rationale for pooling ("meaningfully more data for the same
architecture") weaker than it should be.

Suggested fix: after pooling, re-map `time_idx` onto a single calendar shared
by every symbol (union of dates -> contiguous index); `allow_missing_timesteps
=True` is already set in `make_dataset()` and absorbs the resulting per-symbol
gaps. That fixes the cutoff arithmetic and the cross-sectional alignment in
one change. Additionally warn when any symbol's last observation falls at or
before the cutoff (a genuinely delisted/stale symbol), since its validation
window would still overlap training.

Acceptance criteria: a test with two symbols of different history lengths
asserting (a) the same `Date` maps to the same `time_idx` across symbols, and
(b) every symbol's last window starts after the training cutoff.

Resolution: `dataset.align_time_index()` re-maps `time_idx` onto the union
calendar of all pooled symbols, and `_build_pooled_long_df()` applies it (as does
`walk_forward_backtest`). The same `Date` now yields the same `time_idx` for every
symbol, so `cutoff = max_idx - validation_days` means the same instant for all of
them and a late-listing symbol's validation window lands after the cutoff like
everyone else's. This also fixes the cross-sectional alignment that made pooling
weaker than advertised: a shared market shock now sits at one index across groups.

`allow_missing_timesteps=True` (already set) absorbs the per-symbol gaps.

Date alignment fixes late *starts*; a symbol whose data *stops* early (delisted,
stale feed) would still have its validation window inside training, so
`_warn_on_stale_symbols()` names any such symbol rather than reporting an
optimistic val_loss for it.

Covered by `test_pooling_date_aligns_symbols_with_unequal_history`,
`test_train_warns_when_a_symbols_history_ends_before_the_cutoff`, and two
`align_time_index` unit tests.

---

## [PYQ-117]
Every reported metric (and model selection) rests on a single 5-point sample
Status: Resolved — 2026-07-26
Priority: High
Files: `pyquant/models/tft.py` (`train`), `pyquant/analysis/metrics.py`, `pyquant/cli/app.py` (`train`, `backtest` tables)

Problem: `train()` builds `validation = TimeSeriesDataSet.from_dataset(
training, df, predict=True, ...)`, and `predict=True` yields exactly **one**
sample per group. Verified:

```
VALIDATION samples used for ALL reported metrics: 1
  -> metric sample size = samples x horizon = 5 points
```

Those 5 points produce `model_mae`, `baseline_mae`, `directional_accuracy`
and `calibration_coverage`, and via `best_model_score` they also drive
`EarlyStopping` and `ModelCheckpoint`. That is why a real run reports
`Directional accuracy 100.0%` and `Calibration coverage 100.0%` — they are
5/5 and 5/5. Printed to one decimal with no denominator, that reads as a
strong result when it is noise. Corroborating evidence from a live NVO run:
the epoch bar showed `val_loss: 3.013` while the table reported best
`0.66954` — a ~4.5x swing on the selection metric, which is what a 5-point
validation set does.

The underlying cause is split *geometry*, not just the `predict=True` flag:
`cutoff = max_idx - horizon` leaves a holdout exactly one horizon long, so
only one full window can ever fit after it. Raising the window count requires
a longer holdout.

Supersedes investigations.md#pyq-303, which asked whether a single 5-day
validation window is statistically reliable. It is not, and the answer does
not need further experimentation — but the numbers ship to the table
regardless, which makes this a defect rather than an open question.

Suggested fix: (a) give the holdout a real span — a `TrainingConfig`
setting (e.g. `validation_days`, default ~60) used as
`cutoff = max_idx - validation_days`, so `validation_days - horizon + 1`
windows per symbol are scored; (b) carry the sample size on
`EvaluationMetrics` (`n_samples`/`n_points`) and print it, so a percentage is
never shown without its denominator; (c) have `aggregate_metrics()` sum the
counts rather than average them.

Acceptance criteria: a test asserting `result.evaluation.n_samples > 1` after
a default-config train, and a unit test that `aggregate_metrics()` sums
sample counts across windows.

Resolution: three changes, since the cause was split geometry rather than one
flag.

1. `TrainingConfig.validation_days` (default 60) replaces the dead `train_split`
   (see PYQ-125). `cutoff = max_idx - validation_days`, so the holdout is a real
   span rather than exactly one horizon.
2. The validation set is built with `min_prediction_idx=cutoff + 1` instead of
   `predict=True`, so *every* window after the cutoff is scored --
   `validation_days - horizon + 1` per symbol -- and that same loader drives
   `EarlyStopping`/`ModelCheckpoint`.
3. `EvaluationMetrics` carries `n_samples`/`n_points`; `aggregate_metrics()` sums
   them rather than averaging; the Rich tables print `Evaluated on N windows (M
   predictions)` and `--format json` emits both.

Verified on live data: `pyquant train NVO` now reports `Evaluated on 56 windows
(280 predictions)` where it previously scored 5 points. The honest picture is much
worse than the old one -- directional accuracy 57.5% (was "100.0%") and skill
-23.5% vs. the persistence baseline (was "+64.9%") -- which is the substance of
investigations.md#pyq-307. Coverage of 99.3% on a nominal 80% band also says the
interval is far too wide, a diagnosis the old 5-point "100.0%" could not support.
(That run was 3 epochs and is not a verdict on model quality; the point is that
the numbers are now measured on enough data to mean something.)

Supersedes investigations.md#pyq-303. Covered by
`test_train_evaluates_many_validation_windows_not_a_single_one`,
`test_evaluate_predictions_records_sample_size`,
`test_aggregate_metrics_sums_sample_counts_rather_than_averaging_them`, and two
CLI tests.

---

## [PYQ-118]
A feature column missing at predict time crashes with a bare `KeyError`
Status: Resolved — 2026-07-26
Priority: High
Files: `pyquant/models/tft.py` (`load`, `predict_quantiles`, `_prediction_dataset`), `pyquant/data/dataset.py` (`feature_columns`)

Problem: this answers investigations.md#pyq-302. Verified both directions of
schema drift between the train-time and predict-time panel:

- **extra** columns at predict time: silently ignored (benign).
- **missing** column at predict time: `KeyError: 'SEC_SPY'` raised from deep
  inside pytorch-forecasting, with no indication of which source vanished or
  why.

`load()` reads `meta["features"]` but nothing ever compares it against the
panel actually built. So a trained bundle becomes uninterpretably broken if
any enrichment present at train time is unavailable later — a rotated-out
`FRED_API_KEY`, a failing sector fetch, Finnhub returning nothing, or a
toggle flipped between runs. The README's "Missing sources are dropped, never
fatal" holds for *training* only; at predict time the same degradation is
fatal and unexplained. This is also the top blocker PYQ-302 identified for
trusting a PYQ-213 API against live data.

Suggested fix: diff `meta["features"]` against `feature_columns(df)` before
building the prediction dataset and raise one clear error naming the missing
columns and the likely cause (which source they came from, which env var
enables it). Extra columns should stay a no-op.

Acceptance criteria: a test training on a rich panel then predicting on a
lean one, asserting the raised error names the missing column and is not a
bare `KeyError`; plus a test that extra columns still predict fine.

Resolution: `_check_feature_schema()` runs inside `_prediction_dataset()` and
diffs `meta["features"]` against `feature_columns(df)`. A missing column now raises
`FeatureSchemaMismatch` naming every absent feature and which source it came from
(`_FEATURE_SOURCE_HINTS` maps `SEC_` to sector ETFs, `VIX`/`CPI`/`FedFunds`/
`YieldSpread` to macro and the key each needs, `Sentiment`/`HeadlineCount` to
Finnhub + the `sentiment` extra), plus what to do about it. Extra columns remain a
deliberate no-op.

The CLI catches it via `EXPECTED_FAILURES` and renders it as a one-line error
(PYQ-120), so a vanished data source reads as a clear instruction rather than a
stack trace.

Covered by `test_predict_raises_a_clear_error_when_a_trained_feature_is_missing`
and `test_predict_ignores_columns_not_seen_during_training`.

---

## [PYQ-119]
`forecast`/`explain`/`scan` ignore the config the bundle was trained with
Status: Resolved — 2026-07-26
Priority: High
Files: `pyquant/cli/app.py` (`forecast`, `explain`, `scan`, `_build_settings`), `pyquant/models/tft.py` (`train` meta.json)

Problem: `train`/`backtest` accept `--config/--period/--no-macro/
--no-sentiment/--no-sectors` and funnel them through `_build_settings()`. The
read-side commands call a bare `load_settings()` and accept none of them. So
`pyquant train AAPL --no-sectors --config configs/x.yaml` followed by
`pyquant forecast AAPL` rebuilds the panel from **defaults** — a different
feature set and a different `period` than the model was trained on. That is
the primary way to trigger PYQ-118.

meta.json records `quantiles`, `max_encoder_length` and
`max_prediction_length`, but not the resolved data toggles or `period`, so
they cannot be recovered from the bundle either.

Suggested fix: persist the resolved `data`/`training` config into meta.json at
train time, and have `forecast`/`explain`/`scan` rebuild the panel from the
bundle's own recorded settings, treating explicit CLI flags as an override
rather than the source of truth. This removes a whole class of drift and is a
stated prerequisite for the PYQ-213 API.

Acceptance criteria: a test that trains with an enrichment disabled and then
forecasts without passing any flags, asserting the panel rebuilt for
prediction used the bundle's recorded toggles (not the defaults).

Resolution: `train()` writes a `config` block into `meta.json` (and therefore
`runs.jsonl`) holding the resolved `data`/`training`/`tft` sub-configs.
`tft.settings_for_bundle()` overlays the recorded schema-relevant `data` fields
(`period`, the four `use_*` toggles, `sector_etfs`) onto the caller's settings, and
`generate_forecast()`/`explain_forecast()` both call it before building their panel.
Bundles predating this simply keep the caller's settings.

Deliberately *not* restored: `cache_dir`, `cache_enabled`, `checkpoint_dir` and the
secrets -- those describe the machine you are on now, not the trained model.

Covered by `test_train_records_the_resolved_data_config`,
`test_settings_for_bundle_restores_the_recorded_data_toggles`,
`test_generate_forecast_rebuilds_the_panel_with_the_bundles_recorded_config` and
its `explain` counterpart.

---

## [PYQ-120]
Untrained-symbol failure dumps a Rich traceback instead of a clean message
Status: Resolved — 2026-07-26
Priority: Medium
Files: `pyquant/cli/app.py` (`app` construction, `forecast`, `explain`), `tests/test_cli.py`

Problem: `pyquant forecast ZZZZNOTTRAINED` prints ~30 lines of framed Typer
traceback ending in `FileNotFoundError: No trained model for ... Run
`pyquant train` first.` The exit code is correctly 1 and the message is
already user-ready — it simply is not caught. `scan` handles this case
(rendering a "not trained" row); `forecast` and `explain` do not.

Related coverage gap: **every** assertion in `tests/test_cli.py` is
`exit_code == 0`. No failure path is exercised anywhere in the suite, which is
why this UX regression went unnoticed.

Suggested fix: `pretty_exceptions_enable=False` on the Typer app plus a small
handler in the commands that prints the message and raises `typer.Exit(1)`.

Acceptance criteria: a test asserting `forecast` on an untrained symbol exits
1, prints the "Run `pyquant train` first" message, and does **not** print a
traceback.

Resolution: the Typer app is constructed with `pretty_exceptions_enable=False`,
and `train`/`backtest`/`forecast`/`explain` wrap their fallible calls in
`except EXPECTED_FAILURES` (`FileNotFoundError`, `FeatureSchemaMismatch`,
`ValueError`) -> `_fail()`, which prints `Error: <message>` and raises
`typer.Exit(1)`. The messages were already user-ready; they just were not caught.

The coverage gap that let this through is closed as part of features.md#pyq-231.

Covered by `test_forecast_on_untrained_symbol_exits_cleanly_without_a_traceback`,
its `explain` counterpart, and `test_train_on_insufficient_history_reports_cleanly`
-- each asserting the exception is a `SystemExit` rather than the original
exception leaking, and that no traceback is printed.

---

## [PYQ-121]
`RSI_14` uses simple-MA smoothing, not Wilder's — not the standard RSI
Status: Resolved — 2026-07-26
Priority: Medium
Files: `pyquant/data/prices.py` (`compute_rsi`)

Problem: `compute_rsi()` smooths average gain/loss with
`rolling(window=period, min_periods=1).mean()` — a simple moving average.
Wilder's RSI, which is what every charting package and every reference
implementation computes, uses Wilder smoothing:
`ewm(alpha=1/period, adjust=False).mean()`. The two differ materially, so the
column labelled `RSI_14` is not comparable to any external RSI, and any
intuition about thresholds (30/70) transfers only loosely.

Secondary issue: `min_periods=1` makes RSI defined from row 2 off a one-row
window, i.e. warm-up garbage that is *not* NaN and so survives
`build_panel()`'s `dropna()`. It is removed only because `SMA_50` happens to
force the first 49 rows out — an accidental dependency between two unrelated
indicators.

Suggested fix: switch to Wilder smoothing and drop `min_periods=1` so the
warm-up rows are genuinely NaN and get dropped on their own merit. Note this
changes a model input, so previously trained bundles are no longer directly
comparable.

Acceptance criteria: a test checking `compute_rsi` against a hand-computed
Wilder RSI on a short known series, and a test asserting the first `period`
rows are NaN.

Resolution: `compute_rsi()` now uses Wilder's smoothing via a `_wilder_average()`
helper -- SMA seed over the first `period` changes, then the recursive
`((n-1)*prev + new) / n`. This is the RSI every charting package plots, so the
column is finally comparable to an external reference and the 30/70 thresholds mean
what they usually mean. `min_periods=1` is gone, so the first `period` rows are
genuinely NaN and are dropped on their own merit rather than depending on `SMA_50`
happening to cut the first 49 rows.

Note this redefines a model input: bundles trained before this are not directly
comparable, which is part of why features.md#pyq-225 now records the code version
in `meta.json`.

Covered by `test_compute_rsi_matches_an_independent_wilder_implementation` (checked
against a deliberately slow, obvious reference implementation written in the test),
plus NaN-warmup, all-gains and all-losses cases.

---

## [PYQ-122]
"Calibration coverage (p10-p90)" label is hardcoded regardless of configured quantiles
Status: Resolved — 2026-07-26
Priority: Medium
Files: `pyquant/cli/app.py` (`train`, `backtest` tables)

Problem: both result tables print the literal string `Calibration coverage
(p10-p90)`. `metrics.py` documents the metric as coverage of the *outermost*
configured quantile band, so with any non-default `quantiles` the label is
simply wrong. The project's own shipped
`configs/wide_quantile_aggressive.yaml` sets `[0.05, 0.25, 0.5, 0.75, 0.95]`,
where the band is p5-p95 — so following the documented example config
produces a mislabelled table.

Suggested fix: derive the label from the configured quantiles
(`quantiles[0]`/`quantiles[-1]`), the same way `_forecast_table` already
derives its column headers.

Acceptance criteria: a test running `train` with non-default quantiles and
asserting the rendered table says `p5-p95`, not `p10-p90`.

Resolution: `_band_label(quantiles)` derives the label from
`quantiles[0]`/`quantiles[-1]`, and both tables render it through the shared
`_add_metric_rows()` helper. `configs/wide_quantile_aggressive.yaml` now correctly
reports `p5-p95`.

Covered by `test_train_table_labels_the_calibration_band_from_configured_quantiles`
and its `backtest` counterpart, both asserting `p10-p90` is *absent*.

---

## [PYQ-123]
`build_panel`'s trailing `bfill()` back-fills future values into leading rows
Status: Resolved — 2026-07-26
Priority: Medium
Files: `pyquant/data/dataset.py` (`build_panel`)

Problem: `panel = panel.ffill().bfill()`. The `ffill()` is correct and
necessary (a joined source's calendar not matching the target's). The trailing
`bfill()` can only ever fire on **leading** NaNs — rows before a joined
source's first observation — and it fills them with the first *later* value.
That is look-ahead: an early training row sees a value that did not exist yet.

Same class of leak as PYQ-101, and much smaller in scope (bounded by how late
a source starts relative to the price history), but it is a leak in a file
that has otherwise been deliberate about exactly this, and the comment above
it only justifies the `ffill` half.

Suggested fix: drop the leading rows that remain NaN after `ffill()` (the same
policy `add_technical_indicators` warm-up rows already get), or fill with an
explicit neutral constant. Either way, state the choice in the comment.

Acceptance criteria: a test where a joined source starts later than the price
history, asserting no pre-source-start row carries a post-source-start value.

Resolution: the trailing `bfill()` is gone. `build_panel()` now does `ffill()`
followed by `dropna()`, so rows before a joined source's first observation are
dropped rather than filled from a future value -- the same policy indicator warm-up
rows already get, and the comment now explains both halves. A column with no
overlap at all with the price calendar would make `dropna()` empty the panel, so
those are dropped as columns (with a warning) first.

Covered by `test_build_panel_does_not_backfill_a_late_starting_source`.

---

## [PYQ-124]
Quantile crossing is detected but never corrected before display/signalling
Status: Resolved — 2026-07-26
Priority: Medium
Files: `pyquant/analysis/metrics.py` (`warn_on_quantile_crossing`), `pyquant/analysis/forecast.py`, `pyquant/cli/app.py` (`_forecast_table`, `scan`), `pyquant/cli/charts.py`

Problem: `warn_on_quantile_crossing()` (added by PYQ-216) correctly detects
non-monotonic quantiles and logs a warning — and it fires in practice; a
1-epoch smoke model produced "1 point(s)". But nothing consumes that signal.
The forecast table, the terminal fan chart, the PNG export and `scan`'s band
logic all render the raw, possibly-crossed values. In `scan` that means
`band_width_pct` can go negative and the `lo_pct > 0` / `hi_pct < 0` guards —
written specifically so a zero-straddling band cannot produce a BUY/SELL — can
be satisfied by an *inverted* band instead.

Also note the warning goes to the logger, which is set to WARNING but formatted
bare, and at default verbosity the user sees a stray line with no context.

Suggested fix: sort the quantile values per timestep before display and before
signalling (monotonic by construction), and surface the crossing count in the
output itself rather than only in the log, so the user knows the model produced
a degenerate band.

Acceptance criteria: a test with deliberately crossed predictions asserting
`_forecast_table` renders monotonically non-decreasing rows and `scan` does not
emit BUY/SELL off an inverted band.

Resolution: `Forecast.__post_init__` sorts `predictions` along the quantile axis
and records how many points were reordered in `n_quantile_crossings`. Putting the
invariant on the dataclass rather than in `generate_forecast()` means no `Forecast`
can exist in a crossed state however it was constructed -- including from the
planned PYQ-213 API layer -- so `_forecast_table`, both charts and `scan`'s
band-direction guards are all safe by construction. `scan` can no longer read an
inverted band as a confident BUY.

The count is surfaced rather than hidden: `forecast` prints a note when it is
non-zero, and `--format json` emits `n_quantile_crossings`.

Covered by `test_forecast_table_renders_a_crossed_band_monotonically`.

---

## [PYQ-125]
`TrainingConfig.train_split` is dead; `DataConfig.use_options` is never read
Status: Resolved — 2026-07-26
Priority: Low
Files: `pyquant/config.py` (`TrainingConfig.train_split`, `DataConfig.use_options`), `pyquant/cli/app.py` (`forecast`)

Problem: two settings in the documented config surface do nothing.

- `TrainingConfig.train_split: float = 0.85` is referenced nowhere. The actual
  split is positional (`max_idx - horizon`). Anyone tuning it — including via
  a YAML experiment config — silently changes nothing.
- `DataConfig.use_options: bool = True` is never read.
  `fetch_options_snapshot()` is called unconditionally in `forecast`, so
  `use_options: false` neither skips the network call nor hides the options
  line.

Suggested fix: delete `train_split` (PYQ-117 introduces a real, honest
holdout setting in its place); honour `use_options` in `forecast` (and in
`scan` if options context is ever added there).

Acceptance criteria: `train_split` gone from the config and from
`tests/test_config.py`; a test asserting `use_options=False` skips the options
fetch and the options line.

Resolution: `train_split` deleted -- PYQ-117 introduced `validation_days` as a
real, honest holdout setting in its place. `use_options` is now honoured: `forecast`
skips `fetch_options_snapshot()` entirely when it is false, so the flag saves the
network call as well as hiding the line.

Covered by `test_training_config_holdout_is_longer_than_one_horizon` (which also
asserts `train_split` is gone) and
`test_forecast_skips_the_options_fetch_when_use_options_is_false`.

---

## [PYQ-126]
`_fmt_bytes` has an unreachable final `return`
Status: Resolved — 2026-07-26
Priority: Low
Files: `pyquant/cli/app.py` (`_fmt_bytes`)

Problem: the loop condition is `if size < 1024 or unit == "GB": return ...`,
so the final iteration (`unit == "GB"`) always returns from inside the loop.
The trailing `return f"{size:.1f} GB"` after the loop can never execute. Ruff
does not flag it (it is not syntactically unreachable), but it is dead code
that implies a fall-through case that does not exist.

Suggested fix: delete the trailing return, or restructure so the GB case is
the explicit fall-through rather than a special case inside the loop.

Acceptance criteria: no unreachable branch remains; existing `cache list`
formatting behaviour unchanged (covered by
`test_cache_list_and_prune_commands`).

Resolution: the loop now iterates `("B", "KB", "MB")` and GB is the explicit
fall-through after it, so there is no unreachable branch and no special case inside
the loop.

Covered by `test_fmt_bytes_formats_every_unit`, including the beyond-GB case that
exercises the fall-through.

---

## [PYQ-127]
`walk_forward_backtest` evaluates the *same* final window at every rolling origin
Status: Resolved — 2026-07-26
Priority: High
Files: `pyquant/models/tft.py` (`walk_forward_backtest`)

Problem: inside the cutoff loop, each window builds
`validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, ...)`
against the **full** `df`. `predict=True` anchors the decoder to the global
last `max_prediction_length` timesteps — so every rolling origin evaluates the
identical final window. Verified across 5 cutoffs:

```
current code:
  cutoff=225 -> decoder_time_idx=[246, 247, 248, 249, 250]
  cutoff=230 -> decoder_time_idx=[246, 247, 248, 249, 250]
  cutoff=235 -> decoder_time_idx=[246, 247, 248, 249, 250]
  cutoff=240 -> decoder_time_idx=[246, 247, 248, 249, 250]
  cutoff=245 -> decoder_time_idx=[246, 247, 248, 249, 250]
```

So `pyquant backtest --windows 5` trains five models at five cutoffs and scores
all five on the same five days, then averages the result and labels it "5
windows". The walk-forward does not walk on the evaluation side, which defeats
the entire purpose of the command (PYQ-202) and of investigations.md#pyq-303's
stability question. Worse, the earliest cutoff's model is evaluated 20+ days
past its training end while the latest is evaluated immediately after — the
windows are not even comparable to each other.

Truncating the frame handed to `from_dataset` fixes it cleanly:

```
df[df.time_idx <= cutoff + horizon]:
  cutoff=225 -> decoder_time_idx=[226, 227, 228, 229, 230]
  cutoff=230 -> decoder_time_idx=[231, 232, 233, 234, 235]
  cutoff=235 -> decoder_time_idx=[236, 237, 238, 239, 240]
  cutoff=240 -> decoder_time_idx=[241, 242, 243, 244, 245]
  cutoff=245 -> decoder_time_idx=[246, 247, 248, 249, 250]
```

Suggested fix: build each window's validation set from
`df[df["time_idx"] <= cutoff + horizon]` so `predict=True` lands the decoder
on exactly that origin's out-of-sample horizon.

Acceptance criteria: a test asserting the evaluated decoder window differs
between consecutive rolling origins, and that each window's decoder starts at
`cutoff + 1`.

Resolution: `_window_validation_dataset()` builds each origin's validation set
from `df[df["time_idx"] <= cutoff + horizon]`, so `predict=True` anchors the decoder
to exactly that origin's out-of-sample window. The walk-forward now genuinely walks:
consecutive origins evaluate consecutive, non-overlapping windows, each starting at
`cutoff + 1`.

This is what makes features.md#pyq-226's per-window spread meaningful -- until now
the five "windows" were five models scored on one identical set of days, so any
dispersion measured only model-init noise.

Covered by `test_walk_forward_window_validation_targets_its_own_origin`, which
asserts both that each decoder starts at `cutoff + 1` and that three origins produce
three distinct windows.

---

## [PYQ-128]
A `--config` path that does not exist is silently ignored
Status: Resolved — 2026-07-26
Priority: Medium
Files: `pyquant/config.py` (`load_settings`, `settings_customise_sources`)

Problem: found while testing failure paths for PYQ-231.
`pyquant train AAPL --config /nonexistent/nope.yaml` prints no warning and
trains happily using the built-in defaults. `YamlConfigSettingsSource` treats a
missing file as "no values to contribute", so a typo'd path degrades to "no
config" rather than an error.

This defeats the purpose of PYQ-209. The whole point of a checked-in YAML
experiment config is that a run is reproducible from a named file; if the name
is wrong you silently get a *different experiment* than the one you asked for,
and the resulting bundle records (via PYQ-119/PYQ-225) a config that was never
the one you intended to run. The failure is worst exactly where it matters most
— comparing two configs, where one silently being "defaults" invalidates the
comparison.

The same applies to `PYQUANT_CONFIG` pointing somewhere stale.

Suggested fix: in `load_settings()`, raise a clear `FileNotFoundError` when an
explicitly-requested config path does not exist. An explicit request is
different from the absent-by-default case, which must stay silent.

Acceptance criteria: a test asserting `load_settings("/nope.yaml")` raises and
names the path, and a CLI test asserting `train --config /nope.yaml` exits
non-zero with a readable message rather than training on defaults.

Resolution: `load_settings()` raises `FileNotFoundError` naming the path when an
explicitly requested config (via `--config` or `PYQUANT_CONFIG`) does not exist. No
config requested at all stays silent, as before. The CLI reports it through
PYQ-120's `_fail()` path, so a typo exits 1 with a readable message instead of
training a different experiment than the one asked for.

Covered by `test_load_settings_rejects_a_config_path_that_does_not_exist`,
`test_load_settings_without_a_config_stays_silent`, and
`test_train_with_a_missing_config_file_fails_instead_of_using_defaults` (which
asserts `tft.train` was never reached).
