# Bugs (PYQ-1xx)

Concrete, reproducible defects — see [`README.md`](README.md) for the format.
Next free ID: **PYQ-115**.

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
