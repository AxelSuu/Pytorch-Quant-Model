# Bugs (PYQ-1xx)

Concrete, reproducible defects — see [`README.md`](README.md) for the format.
Next free ID: **PYQ-171**.

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
| [PYQ-129](#pyq-129) | Critical | Resolved | News sentiment is joined to the session it was published in, including post-close headlines |
| [PYQ-130](#pyq-130) | High | Resolved | `future_business_dates()` names exchange holidays as forecast days |
| [PYQ-131](#pyq-131) | Medium | Resolved | `predict_quantiles` returns `out[0]` without checking which group it is |
| [PYQ-132](#pyq-132) | Medium | Resolved | EMA/MACD warm-up rows survive `dropna()` — the still-live half of PYQ-121 |
| [PYQ-133](#pyq-133) | High | Resolved | Cache entries and pins record no code version, so a pin outlives a feature redefinition |
| [PYQ-134](#pyq-134) | Low | Resolved | `_git_sha()` resolves against the installed package directory, not the working tree |
| [PYQ-135](#pyq-135) | Low | Resolved | `Volume_Change` yields `inf` on a zero-volume session |
| [PYQ-136](#pyq-136) | Medium | Resolved | `aggregate_metrics()` sums sample counts but unweighted-averages the rates they describe |
| [PYQ-137](#pyq-137) | Low | Resolved | EMA seed bias survives `min_periods`: the first surviving panel rows are still ~0.08% off |
| [PYQ-138](#pyq-138) | Low | Resolved | CLI output tests assert on ANSI-coloured stdout, so they pass or fail by ambient terminal |
| [PYQ-139](#pyq-139) | Critical | Resolved | PYQ-257's vintage fetch fails against the live FRED API: every FRED macro feature silently vanished |
| [PYQ-140](#pyq-140) | High | Resolved | Finnhub's free tier serves ~6 days of news, not ~365: `Sentiment` is 99.7% structural zeros |
| [PYQ-141](#pyq-141) | Medium | Open | Headline skill and the per-window skill column beneath it are different estimators |
| [PYQ-142](#pyq-142) | High | Resolved | The displayed log-return forecast band compounds marginal per-step quantiles, ~√h too wide |
| [PYQ-143](#pyq-143) | High | Resolved | `train()`/`walk_forward_backtest()` select checkpoints and report metrics from the same window |
| [PYQ-144](#pyq-144) | High | Resolved | Conformal offset is pooled across horizon steps and fit on an inflated sample size |
| [PYQ-145](#pyq-145) | High | Resolved | API accepts unvalidated `symbol`/`bundle_name`; auth header comparison throws on non-ASCII input |
| [PYQ-146](#pyq-146) | High | Resolved | `load_settings()` mutates a module global; concurrent API requests can read the wrong config |
| [PYQ-147](#pyq-147) | Medium | Resolved | `interpret()` zips variable names to weights with `strict=False` |
| [PYQ-148](#pyq-148) | Medium | Resolved | `use_options` is missing from the panel cache fingerprint |
| [PYQ-149](#pyq-149) | Medium | Open | `backtest --signals` scores an uncalibrated rule while `scan` ships a calibrated one |
| [PYQ-150](#pyq-150) | High | Resolved | Finnhub API key travels in the query string and can reach WARNING logs on retry failure |
| [PYQ-151](#pyq-151) | Low | Open | `with_retry` retries non-retryable errors (401/404) by default |
| [PYQ-152](#pyq-152) | Medium | Resolved | `compute_rsi` returns 100, not 50, on a flat/halted series |
| [PYQ-153](#pyq-153) | Low | Open | PIT values saturate at the outer quantile edges instead of extending into the tails |
| [PYQ-154](#pyq-154) | Low | Open | `next_sessions` is not total for very large `max_prediction_length` |
| [PYQ-155](#pyq-155) | Medium | Resolved | Cache writes (pickle + meta) are two separate, non-atomic operations |
| [PYQ-156](#pyq-156) | Low | Resolved | `cli/app.py` hides a metric row at exactly 0.0; `aggregate_metrics([])` raises `ZeroDivisionError` |
| [PYQ-157](#pyq-157) | Medium | Resolved | `Settings` swallows misspelled nested env keys; `TFTConfig.quantiles` allows a bandless config |
| [PYQ-158](#pyq-158) | Medium | Resolved | `tests/conftest.py`'s `settings` fixture leaves `use_options` pointed at the real repo |
| [PYQ-159](#pyq-159) | Low | Resolved | `TrainRequest.period` is dead; `JobRegistry` never evicts and indexes `_jobs` unguarded |
| [PYQ-160](#pyq-160) | Medium | Resolved | `/docs`, `/redoc`, `/openapi.json` are reachable with no `X-API-Key`, contradicting the documented auth contract |
| [PYQ-161](#pyq-161) | Medium | Resolved | No lock guards concurrent `POST /train` calls for the same bundle name |
| [PYQ-162](#pyq-162) | Low | Resolved | `POST /train`'s documented `failed` job status has zero test coverage |
| [PYQ-163](#pyq-163) | High | Open | `POST /train`/`POST /backtest` background jobs share the request-serving thread pool with every other endpoint |
| [PYQ-164](#pyq-164) | Low | Open | Per-bundle-name lock dicts (`_PredictionLocks`, `BundleCache._load_locks`) never evict |
| [PYQ-165](#pyq-165) | Medium | Resolved | `/healthz` was a sync `def`, sharing the request/background-job thread pool its own liveness check needs to bypass |
| [PYQ-166](#pyq-166) | High | Resolved | `POST /scan`/`POST /train`/`POST /backtest` request fields (`symbols`, `epochs`, `windows`, `period`) had no upper bound |
| [PYQ-167](#pyq-167) | Medium | Resolved | Per-bundle prediction lock blocked indefinitely; a slow request could hold every subsequent caller hostage |
| [PYQ-168](#pyq-168) | Medium | Resolved | `BundleCache.get()` re-deserializes the same checkpoint N times under concurrent cold-start requests |
| [PYQ-169](#pyq-169) | Medium | Resolved | Untrained-bundle `404`s leaked the absolute checkpoint filesystem path to the caller |
| [PYQ-170](#pyq-170) | Low | Resolved | `PYQUANT_API_KEYS` misconfiguration was only caught per-request (`500`), not at process startup |

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

---

## [PYQ-129]
News sentiment is joined to the session it was published in, including post-close headlines
Status: Resolved — 2026-07-26
Priority: Critical
Files: `pyquant/data/sentiment.py` (`fetch_sentiment`), `pyquant/data/dataset.py` (`build_panel`, sentiment join)

Problem: `fetch_sentiment()` buckets each article by its **UTC calendar date**
(`pd.Timestamp(dt.datetime.utcfromtimestamp(a["datetime"]).date())`) and
`build_panel()` joins that daily series straight onto the trading row of the same
date. A US equity session closes at 20:00 UTC (21:00 during EST), so **every headline
published in the last 3–4 hours of each UTC day is post-close information attached to a
row whose target is that day's close.** Roughly 12–17% of each day's news window is
affected, and it is the *most* market-moving slice — post-close earnings releases land
there almost by definition.

This is the same class of leak as PYQ-101, in the one source that PYQ-305's
`publication_lag_days` convention was never extended to. It is arguably worse than
PYQ-101 because the leaked information is event-driven rather than slow-moving: a
same-day earnings-beat headline is nearly a direct readout of the next move.

The leak is currently masked, not absent: PYQ-301 notes ~80% of a default 5-year training
window has structurally-zero sentiment, so the contaminated rows are a minority of
training data — but they are 100% of the *live* prediction path, where sentiment is always
populated. That is the worst possible distribution of the error.

Secondary: `dt.datetime.utcfromtimestamp` is deprecated as of Python 3.12 and emits a
`DeprecationWarning` that PYQ-108's filter silences by default.

Suggested fix: convert each article timestamp to the exchange's local time, and assign it
to the **next** trading session whenever it lands at or after that session's close.
Simplest correct-by-construction variant, if per-exchange close times are more machinery
than is wanted: shift the whole daily sentiment series forward by one trading day before
the join, and say so in the docstring — strictly conservative, never leaks, costs at most
one session of freshness. Either way, record the choice using the same
`publication_lag`-style convention PYQ-305 established for FRED so the two sources are
reasoned about identically. Replace `utcfromtimestamp` with
`dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)`.

Acceptance criteria: a test with two synthetic articles on the same UTC date — one at
14:00 UTC (pre-close) and one at 22:00 UTC (post-close) — asserting the first lands on
that session's row and the second on the next session's. Plus a test asserting no
`DeprecationWarning` is raised by `fetch_sentiment`.

Resolution: took the first (precise) option rather than the blanket one-day shift, because
the blanket shift costs a full session of freshness on *every* headline to fix a problem
that only affects the last few hours of each day — and the live prediction path, where
sentiment is always populated, is exactly where that freshness matters.

`session_date(epoch_seconds)` is now the single rule: convert the timestamp to
`America/New_York`, and assign the headline to the next calendar date if it was published
at or after the 16:00 close, otherwise to that date. `ZoneInfo` keeps the boundary correct
across DST instead of drifting an hour for half the year; it is stdlib, so no new
dependency (per the CLAUDE.md rule on justifying additions). The docstring states the
convention in the same terms `macro.py` uses for `publication_lag_days`, so the two sources
are now reasoned about identically, which is what PYQ-305 established and this source never
inherited.

**Fixing only that would have converted the leak into data loss**, which is why the fix has
a second half. `session_date` returns a calendar date, so Friday-post-close news now lands
on Saturday — and `build_panel` reindexed the daily series straight onto the price index,
which silently drops any date that is not itself a trading day. `align_to_sessions()` maps
each dated bucket onto the first session at or after it, pooling buckets that land on the
same session (counts add; sentiment is averaged *weighted by headline count*, so a
10-headline day does not count the same as a 1-headline day). News after the last session
is dropped and logged — never rolled backwards, which would reintroduce the leak. Weekend
news, which the old code also dropped, now lands on Monday.

Verification, on the assembled panel rather than on either half. Two headlines on Friday
2022-05-27, one at 11:00 ET and one at 17:00 ET, scored +0.6 and −0.9:

```
before:  Friday Sentiment = -0.15   (mean of both -- the 17:00 headline leaked)
         Monday Sentiment =  0.0    (no news)
after:   Friday Sentiment = +0.6    Monday Sentiment = -0.9
```

The −0.15 is the leak in one number: a headline published after the bell moved the feature
on the row whose target is that day's close.

Covered by `test_post_close_headline_is_assigned_to_the_next_session` (the ticket's exact
14:00/22:00 UTC case), `test_fetch_sentiment_raises_no_deprecation_warning` (which promotes
`DeprecationWarning` to an error, since PYQ-108's filter otherwise hides it),
`test_align_to_sessions_rolls_non_trading_dates_onto_the_next_session`,
`test_align_to_sessions_drops_news_after_the_last_session`, and — the one that actually
guards the cross-file behaviour — `test_build_panel_lands_post_close_news_on_the_next_
trading_row`, verified to fail with the −0.15 above when `session_date` is reverted to UTC
bucketing. Every one of the six previous leaks was correct in each individual file and
wrong across files, so this ticket's binding test is deliberately the panel-level one.

`dt.datetime.utcfromtimestamp` is gone, replaced by `dt.datetime.fromtimestamp(ts,
tz=dt.timezone.utc)`.

Invalidates: any bundle trained with `use_sentiment=True` has different feature values on
the rows that carry news, so its metrics are not comparable to a post-fix run. PYQ-301's
question — how much of the training window has non-neutral sentiment — is now also the
question of how much this fix moved, and its answer bounds this ticket's practical impact.
Assumes a 16:00 close and no half-days; the 13:00 early closes (roughly 3 a year) will
assign an afternoon headline to the same session it can no longer trade on. That is the
*conservative* direction — it never leaks — so it is left as a known limitation rather than
pulling in an exchange calendar here; PYQ-130 introduces one for the forecast dates.

---

## [PYQ-130]
`future_business_dates()` names exchange holidays as forecast days
Status: Resolved — 2026-07-26
Priority: High
Files: `pyquant/data/dataset.py` (`future_business_dates`, `extend_for_prediction`), `pyquant/cli/charts.py`

Problem: `pd.bdate_range` returns Mon–Fri and knows nothing about market holidays. So a
forecast made on 2026-07-02 will label a step for 2026-07-03 (the observed Independence
Day holiday), one made before Thanksgiving will label the Friday half-day and the Thursday
holiday identically, and so on. Two distinct consequences:

- **Display**: the table, the PNG export and `--format json` all assert a date on which
  no price will exist, so the forecast cannot be scored against reality without a manual
  correction. PYQ-115 went to real trouble to make one helper the single source of truth
  for these dates; that helper is now the single source of one consistent error.
- **Model input**: `extend_for_prediction()` writes `dow`/`month_num` for those rows, and
  `dow` is a `time_varying_known_real` the decoder genuinely reads. A holiday row supplies
  a (weekday, position-in-horizon) combination that never occurs in training data, because
  training rows only exist for sessions that traded.

The second is the reason this is High rather than cosmetic: it is a silent train/serve
skew in a known-future feature, roughly 9 sessions a year, concentrated around exactly the
dates with unusual volatility.

Suggested fix: derive the future calendar from a real exchange calendar
(`exchange_calendars` is the maintained successor to `trading_calendars`;
`pandas_market_calendars` is the other common choice), defaulting to XNYS and made
configurable on `DataConfig` since the panel already supports non-US tickers in principle.
A dependency-free fallback worth considering first: infer the calendar from the observed
`Date` index — any weekday absent from ~5 years of history is a holiday — which needs no
new dependency and self-corrects per exchange, at the cost of being wrong for a
newly-declared holiday.

Acceptance criteria: a test with `last_date = 2026-07-02` asserting the returned dates
skip 2026-07-03; a test asserting `extend_for_prediction` appends the same dates
`future_business_dates` returns (guarding the PYQ-115 invariant against this change).

Resolution: took **neither** of the two options as stated, and the reasoning is the point.

The dependency-free option the ticket floated — infer the calendar from weekdays absent
from ~5 years of observed history — does not actually work for this use. US market
holidays are *rules* ("fourth Thursday in November", "Good Friday"), not fixed dates, so a
calendar inferred from which dates were historically absent cannot project the next
Thanksgiving into a year it has not seen. Since `future_business_dates` is only ever used
in the forward direction, an inferred calendar is wrong exactly where it is needed.

But `exchange_calendars` is not required either: **pandas already ships the holiday-rule
primitives**, and pandas is already a core dependency. New
`pyquant/data/trading_calendar.py` states the NYSE calendar in ~20 lines of
`AbstractHolidayCalendar` rules and adds no supply chain — which is the disposition
PYQ-310 and PYQ-308 established. `future_business_dates()` now delegates to
`next_sessions()`.

Deliberately *not* `USFederalHolidayCalendar`, which is wrong on three counts: the NYSE
closes on Good Friday (not federal) and trades on Columbus Day and Veterans Day (both
federal). `nearest_workday` encodes the exchange observance rule, with New Year's Day using
`sunday_to_monday` because the NYSE does not close on 31 December when 1 January is a
Saturday.

Verification against the published calendar rather than against itself — the generated 2026
holidays are exactly the ten real NYSE closures, and 2027 (whose observances shift
differently: Juneteenth Sat→Fri 06-18, Independence Sun→Mon 07-05, Christmas Sat→Fri 12-24)
is also exact. The ticket's own example:

```
before:  future_business_dates(2026-07-02, 5) -> 07-03, 07-06, 07-07, 07-08, 07-09
after:   future_business_dates(2026-07-02, 5) -> 07-06, 07-07, 07-08, 07-09, 07-10
```

Half-days are kept: the Friday after Thanksgiving closes early but a price prints, so it is
scoreable and belongs in the horizon. Only full closures are removed.

Both consequences the ticket named are addressed by the one change, because PYQ-115 had
already made this helper the single source of truth: the appended decoder rows get correct
`dow` values, and the table / JSON / PNG follow automatically — `cli/charts.py` reads
`Forecast.forecast_dates`, which delegates here, so it needed no edit. That is the PYQ-115
design paying off, and `test_extend_for_prediction_appends_exactly_the_dates_the_forecast_
reports` now pins it so the two cannot drift.

Covered by `test_future_business_dates_skips_an_observed_exchange_holiday` (the ticket's
2026-07-02 case), `test_future_business_dates_skips_thanksgiving_but_keeps_the_half_day`,
`test_future_business_dates_skips_good_friday`, and `tests/test_trading_calendar.py` —
which asserts the full 2026 list, that Columbus and Veterans Day remain sessions, that
Juneteenth is absent before 2022, and that a requested horizon returns exactly that many
sessions across a holiday run. The first three failed against `pd.bdate_range` with the
holiday present in the returned index.

Changes model inputs: `dow` for decoder rows that previously fell on a holiday now
describes a real session, so a forecast issued within 5 sessions of a holiday differs from
one issued by the old code. Training rows are untouched — they only ever existed for days
that traded, which is precisely why the skew existed.

Known limitations, recorded in the module rather than left implicit: US equities only (a
non-US ticker gets the NYSE calendar — the same assumption `data/sentiment.py` makes about
the 16:00 America/New_York close after PYQ-129, so the two are at least consistent), and
one-off closures such as Hurricane Sandy or a national day of mourning are not rules and
are not modelled. Making the exchange configurable is the point at which
`exchange_calendars` would earn its place; until a second exchange is actually needed, it
does not.

---

## [PYQ-131]
`predict_quantiles` returns `out[0]` without checking which group it is
Status: Resolved — 2026-07-26
Priority: Medium
Files: `pyquant/models/tft.py` (`predict_quantiles`, `interpret`)

Problem: both functions take the first element of the prediction output
(`out[0]`, `raw.output`) with no assertion about which `symbol` group that row belongs
to. Today this is correct **by accident**: `generate_forecast()` and `explain_forecast()`
each build a single-symbol panel via `panel_to_long(panel, symbol)`, so the dataset has
exactly one group and index 0 is unambiguous.

It stops being correct the moment anything hands these functions a multi-symbol frame,
and there are two such paths already contemplated:

- `docs/api-design.md` proposes a `POST /scan` endpoint; batching several symbols into one
  `predict()` call is the obvious optimisation and would silently return the first
  group's forecast for every symbol.
- A pooled-bundle `scan` that reuses one loaded bundle across symbols (an obvious
  performance follow-up to PYQ-204) has the same shape.

The failure is silent and plausible-looking — a wrong-but-reasonable price path — which
is the worst failure mode for a number that drives a BUY/SELL signal.

Suggested fix: have `predict_quantiles`/`interpret` take the symbol they are asked about,
locate its row via the decoder's `groups`/`decoder_cat` output rather than positionally,
and raise if it is absent. Alternatively assert `df["symbol"].nunique() == 1` at entry
with a message naming the constraint — cheaper, and it converts a future silent bug into
an immediate error.

Acceptance criteria: a test passing a two-symbol frame and asserting the returned array
corresponds to the requested symbol (or that a clear error is raised naming the
limitation), not merely that *an* array of the right shape came back.

Resolution (2026-07-26): `_prediction_dataset()` now requires exactly one unique
`symbol` before constructing the prediction dataset. `predict_quantiles` and `interpret`
therefore fail clearly for an accidental batched frame instead of silently selecting output
row zero. `test_prediction_rejects_a_multi_symbol_frame_instead_of_returning_group_zero`
trains a bundle, passes two symbols and asserts the named constraint.

---

## [PYQ-132]
EMA/MACD warm-up rows survive `dropna()` — the still-live half of PYQ-121
Status: Resolved — 2026-07-26
Priority: Medium
Files: `pyquant/data/prices.py` (`add_technical_indicators`)

Problem: PYQ-121 correctly identified that `min_periods=1` produced non-NaN warm-up
garbage which survived `build_panel()`'s `dropna()`, and that it was removed only because
`SMA_50` happened to cut the first 49 rows — "an accidental dependency between two
unrelated indicators." That fix landed for `RSI_14`. The identical problem is still live
for three more columns:

```python
df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
df["EMA_26"] = close.ewm(span=26, adjust=False).mean()
```

`ewm(adjust=False)` with no `min_periods` emits a value from **row 1** — which is just
`close[0]`, not an average of anything. `MACD`, `MACD_Signal` and `MACD_Hist` are all
derived from these. An EMA with span *s* needs roughly 3–4× *s* observations before the
initial-value bias decays below noise, so an EMA-26 is meaningfully biased for ~78 rows.
`SMA_50` cuts only the first 49. **Rows 50–78 of every panel therefore carry four
systematically biased feature columns**, and the bias is largest at the start of the
training window, where a positional split makes them pure training data.

`Realized_Vol_20` and the Bollinger columns are fine (`rolling` defaults to
`min_periods=window`), which is what makes the EMA pair the odd one out rather than a
uniform convention.

Suggested fix: pass `min_periods=span` to both `ewm` calls so the warm-up is genuinely
NaN and gets dropped on its own merit, exactly as PYQ-121 did for RSI. Note this changes
model inputs, so previously trained bundles are not directly comparable — the same caveat
PYQ-121 recorded, and another argument for PYQ-133.

Acceptance criteria: a test asserting `add_technical_indicators` leaves the first 26 rows
of `EMA_26`/`MACD`/`MACD_Signal`/`MACD_Hist` as NaN, extending the existing
`test_add_technical_indicators_leaves_warmup_rows_genuinely_nan` rather than adding a
parallel one; and a test asserting the panel's first surviving row is decided by the
longest warm-up window rather than by `SMA_50` specifically.

Resolution: both `ewm` calls in `add_technical_indicators` and all three in `compute_macd`
now pass `min_periods` equal to their own span, so each column's warm-up is genuinely NaN
and is dropped on its own merit rather than because `SMA_50` happens to cut 49 rows.

The exact warm-up lengths, now pinned by test rather than assumed: `EMA_12` 11 rows,
`EMA_26` 25, `MACD` 25, `MACD_Signal` and `MACD_Hist` 33. Note the ticket's "first 26 rows"
is off by one — `ewm(min_periods=26)` emits its first value at row *index* 25, which is the
26th observation. And the signal line is NaN for 33 rows, not 26: it needs 9 defined `MACD`
values, which only exist from row 25 onward.

**What this changes in a trained model's inputs, measured rather than assumed.** On a
400-row seeded random walk, comparing old vs. new over the rows that actually survive
`build_panel()` (SMA_50 still binds at row 49, so the panel's length is unchanged):

```
MACD         max|Δ| = 0.000000   (identical)
MACD_Signal  max|Δ| = 0.000406   mean|Δ| = 0.000006   mean|MACD_Signal| = 0.764
MACD_Hist    max|Δ| = 0.000406   mean|Δ| = 0.000006
```

`EMA_12`, `EMA_26` and `MACD` are *bit-identical* past their warm-up: `min_periods` only
masks the leading output, it does not alter the `adjust=False` recursion. Only the signal
line moves, because its recursion now seeds on the first defined `MACD` value (row 25)
instead of on row 0 — a 0.05% shift at the first surviving row, decaying to nothing within
a few rows. So unlike PYQ-121, **this does not invalidate previously trained bundles**;
the earlier caveat about comparability does not apply at a material scale here.

Covered by `test_add_technical_indicators_leaves_warmup_rows_genuinely_nan` (extended to
assert the exact warm-up of all nine windowed indicators, not just `SMA_50`) and
`test_panel_warmup_is_decided_by_the_longest_window_not_by_sma_50`, which drops `SMA_50`
and asserts the MACD signal line — not an accident — is what then binds.

Left open: the fix removes the *fabricated* warm-up but not the EMA's residual **seed
bias**, which is what the ticket's "meaningfully biased for ~78 rows" estimate actually
describes. `adjust=False` seeds at `close[0]` regardless of `min_periods`, so 14.6% of the
seed's weight survives at row 25 and 2.3% at row 49. Measured against an unbiased
`adjust=True` reference that is 0.129% of price at row 25 and 0.075% at row 49, falling to
0.012% by row 78. Small, but real, and not something `min_periods` can reach — filed as
PYQ-137 rather than fixed here, because removing it means changing MACD away from the
definition every charting package plots, which is a decision and not a patch.

---

## [PYQ-133]
Cache entries and pins record no code version, so a pin outlives a feature redefinition
Status: Resolved — 2026-07-26
Priority: High
Files: `pyquant/data/cache.py` (`write_cache`, `write_pin`, `read_pin`), `pyquant/data/dataset.py` (`_cache_fingerprint`)

Problem: `_cache_fingerprint()` covers symbol, date range, period, the four `use_*`
toggles, the ETF list, and key *presence*. It does not cover **which code computed the
columns**. `write_pin()` stores a bare pickle with no metadata at all; the TTL entry's
`.meta.json` holds only `cached_at`.

PYQ-121 redefined what `RSI_14` means. PYQ-123 changed which rows survive `build_panel`.
PYQ-132 (above) will change four more columns. Each of those changes the *content* of a
panel without changing anything in the fingerprint. Consequences:

- **TTL cache**: bounded to one hour, so the exposure is a single stale run after an
  upgrade. Annoying, survivable.
- **Pins**: TTL-exempt and permanent by design. A pin created before PYQ-121 replays the
  *old* RSI definition forever, under the same name, into a bundle whose `provenance`
  block faithfully records the **new** git sha. The bundle's own metadata then asserts a
  reproducibility claim that is false, and it is false in the one direction that is
  undetectable — everything looks consistent.

This directly undermines PYQ-225, whose stated thesis is "seed + pinned data + code
version is what actually reproduces a run." Two of those three legs are recorded on the
bundle; the third is recorded on the bundle but *not* on the data it points at, so they
can disagree.

Suggested fix: add `pyquant_version` (and git sha when available) to the cache
fingerprint so a code change invalidates TTL entries automatically. For pins, write a
sibling `<name>.meta.json` recording version, sha, creation time and the column list;
`read_pin()` warns loudly (or refuses without `--force`) when the recorded version differs
from the running one. A column-list comparison is the cheap high-value half — it catches a
*renamed or added* column immediately, even if it cannot catch a silently *redefined* one.

Acceptance criteria: a test asserting two `_cache_fingerprint` calls differing only in
package version produce different keys; a test asserting `read_pin` warns when the pin's
recorded version differs from the current one; a test asserting a pin's metadata records
the panel's column list.

Resolution (2026-07-26): cache fingerprints now include `provenance.code_version()`, so
TTL entries are invalidated when the package version or working-tree revision changes.
Pins now write a sibling metadata file containing the package version, git sha, creation
time, row count and exact panel columns. Replaying a legacy pin with no metadata, a pin
from another version, or a pin whose recorded column list disagrees with its pickle emits
a clear warning; pin removal deletes the metadata too. Verified by the focused PYQ-133
cache/dataset tests (10 passed), including version-key divergence, metadata persistence,
legacy/stale-pin warnings and cleanup.

---

## [PYQ-134]
`_git_sha()` resolves against the installed package directory, not the working tree
Status: Resolved — 2026-07-27
Priority: Low
Files: `pyquant/models/tft.py` (`_git_sha`)

Problem: `_git_sha()` runs `git rev-parse --short HEAD` with
`cwd=Path(__file__).resolve().parent`. Running from a source checkout that is the
intended case and works. But for a `pip install`ed / containerised deployment —
which is the whole point of PYQ-217 and PYQ-213 — `__file__` is inside `site-packages`,
and one of two things happens:

- `site-packages` is not in a repo → `git rev-parse` fails → `None` is recorded. Correct
  outcome, arrived at accidentally.
- `site-packages` *is* inside some git repo (a vendored dependency tree, a Nix/conda env
  under version control, a monorepo with a committed venv) → **the sha of an unrelated
  repository is recorded as PyQuant's provenance.** Silently wrong is worse than absent.

Suggested fix: prefer the installed distribution's recorded version and only consult git
when the package is an editable/source install; or resolve the repo root explicitly and
verify the result actually contains this project (e.g. `git rev-parse --show-toplevel`
matching a known marker) before trusting the sha.

Acceptance criteria: a test running `_git_sha()` with a monkeypatched `__file__` inside a
temporary unrelated git repo, asserting it returns `None` rather than that repo's sha.

Resolution (2026-07-27): reproduced first — with `__file__` pointed at a `site-packages`
tree inside an unrelated repo, `git_sha()` returned that repo's sha (`1422f92`) while the
project's own HEAD was `a7a2b5f`, exactly the silent-mis-stamp the ticket describes.

`provenance.git_sha()` now resolves `git rev-parse --show-toplevel` from the package
directory and **verifies the repo actually contains this file** — `<toplevel>/pyquant/
provenance.py` must resolve to the very module doing the asking — before trusting the sha.
An unrelated enclosing repo therefore yields `None` (absent, which downstream can detect)
rather than a wrong sha (which it cannot). Chosen over the ticket's other suggestion
(prefer the distribution version, consult git only for editable installs) because it keeps
working for the ordinary source checkout without needing to detect install mode, which is
itself unreliable.

`models/tft.py` carried a second copy of this function; it now delegates to
`provenance`, so the same defect cannot be half-fixed. Guarded by
`test_git_sha_returns_none_when_the_package_lives_in_an_unrelated_repo` (builds a real
throwaway repo and asserts both `is None` and `!= that repo's sha`) and
`test_git_sha_still_reports_the_sha_from_a_real_source_checkout`, which pins the case that
already worked.

---

## [PYQ-135]
`Volume_Change` yields `inf` on a zero-volume session
Status: Resolved — 2026-07-27
Priority: Low
Files: `pyquant/data/prices.py` (`add_technical_indicators`)

Problem: `df["Volume_Change"] = volume.pct_change()`. A session with zero reported volume
— a trading halt, a thin ADR, a data-feed gap that yfinance reports as 0 rather than NaN
— makes the *next* row's percent change divide by zero, producing `inf`. `inf` is not NaN,
so it survives `build_panel()`'s `dropna()` and is fed to `GroupNormalizer`, where it
poisons the fitted scale for that group and can propagate NaN through the loss.

Every other indicator in the file guards its denominator (`sma + 1e-10`,
`upper - lower + 1e-10`); this one does not. `Price_Change` has the same shape but is safe
in practice, since a zero close does not occur.

Suggested fix: replace `inf`/`-inf` with NaN after computing the change (letting the
existing row-drop handle it), or use a log-volume difference with a `+1` offset, which is
better behaved and is closer to how volume is usually modelled anyway. Consider a general
`replace([np.inf, -np.inf], np.nan)` at the end of `add_technical_indicators` as a
belt-and-braces guard for the whole indicator block.

Acceptance criteria: a test with a zero-volume row asserting no `inf` survives into the
built panel.

Resolution (2026-07-27): reproduced — a single `Volume=0` session in a 120-row frame put
exactly one `inf` in `Volume_Change`, and it survived `dropna()` into the panel as
predicted.

Took the ticket's belt-and-braces option: `add_technical_indicators` ends with
`df.replace([np.inf, -np.inf], np.nan)`, so the existing row-drop handles it and **the
whole indicator block** is covered rather than just `Volume_Change`. Preferred over a
log-volume difference because that would redefine an existing model input — a PYQ-121-class
change needing its own before/after — where this only removes a value that was never
meaningful. The relative-change definition is unchanged for every non-degenerate row.

Guarded by `test_volume_change_is_nan_not_inf_on_a_zero_volume_session` (asserts the
divide-by-zero row is NaN, that no `inf` remains anywhere in the frame, and that the panel
still has rows) and `test_no_indicator_column_emits_inf_for_a_flat_or_zero_series`, which
holds the invariant for the block as a whole so a future indicator cannot reintroduce the
class silently.

---

## [PYQ-136]
`aggregate_metrics()` sums sample counts but unweighted-averages the rates they describe
Status: Resolved — 2026-07-26
Priority: Medium
Files: `pyquant/analysis/metrics.py` (`aggregate_metrics`)

Problem: PYQ-117 correctly changed `n_samples`/`n_points` to **sum** across windows,
reasoning that "five windows of five points is 25 points of evidence." But the rate and
error metrics are still combined with a plain unweighted `np.mean`:

```python
directional_accuracy=float(np.mean([r.directional_accuracy for r in results])),
n_points=int(sum(r.n_points for r in results)),
```

So the aggregate reports a denominator computed one way and a numerator computed another.
A reader who takes the table at face value — "57.5% over 280 predictions" — is reading a
figure that is only the true pooled rate when every window has an identical sample count.

Today every backtest window does, because `_window_validation_dataset` uses `predict=True`
(one sample per window), so the two definitions coincide numerically and this is latent
rather than live. It becomes reachable as soon as anything produces unequal windows —
PYQ-250's embargo will drop different numbers of samples per origin, a pooled backtest
would give each symbol a different count, and a variable-step walk-forward would too.
Filing it now because the misreading it invites is already possible: the *aggregate row*
already implies a pooling that the arithmetic does not perform.

Suggested fix: weight each rate by that window's `n_points` (a true pooled rate), or keep
the unweighted mean and rename the field to say so (`mean_directional_accuracy`) while
adding a separate pooled figure. The first is almost certainly what a reader expects;
either is defensible, but the current pairing is not.

Acceptance criteria: a unit test aggregating two windows with deliberately unequal
`n_points` and different rates, asserting the aggregate equals the point-weighted pooled
rate rather than the arithmetic mean of the two rates.

Resolution: chose the first of the two options — `aggregate_metrics()` now weights every
rate and error metric by that window's `n_points`, so the aggregate is the true rate *over
the denominator it reports*. The field names keep their meaning ("57.5% over 280
predictions" is now arithmetically that) rather than being renamed to advertise an
unweighted mean, because the pooled figure is what every consumer of the aggregate row —
the Rich table, `--format json`, `meta.json` — is already read as stating.

Windows with no point count at all fall back to an unweighted mean rather than dividing by
zero: `EvaluationMetrics` still has `n_points=0` defaults, so a caller constructing metrics
without PYQ-117's counts must still aggregate. `test_aggregate_metrics_falls_back_to_
unweighted_mean_without_point_counts` pins that path.

**No reported number changes.** Every backtest window today has an identical point count
(`_window_validation_dataset` uses `predict=True`, one sample per window), so pooled and
unweighted means coincide exactly — verified by
`test_aggregate_metrics_sums_sample_counts_rather_than_averaging_them`, which still asserts
`directional_accuracy == 0.5` across five equal windows and passes unchanged. The README's
−23.5% skill / 57.5% directional accuracy figures are therefore unaffected; this fix makes
them *stay* correct once PYQ-250's embargo or a pooled backtest produces unequal windows.

Covered by `test_aggregate_metrics_weights_rates_by_each_window_point_count` (25-point and
5-point windows with rates 1.0 and 0.0, asserting the pooled 25/30 rather than the 0.5
midpoint) and the fallback test above.

---

## [PYQ-137]
EMA seed bias survives `min_periods`: the first surviving panel rows are still ~0.08% off
Status: Resolved — 2026-07-27
Priority: Low
Files: `pyquant/data/prices.py` (`add_technical_indicators`, `compute_macd`)

Problem: found while resolving PYQ-132, whose analysis and whose proposed fix do not quite
meet. PYQ-132 argued — correctly — that "an EMA with span *s* needs roughly 3–4× *s*
observations before the initial-value bias decays below noise, so an EMA-26 is meaningfully
biased for ~78 rows," and then proposed `min_periods=span` as the fix. `min_periods` does
not do that. It masks the first *s* outputs; it does not change the recursion, which
`adjust=False` seeds at `close[0]` no matter what `min_periods` says.

So after PYQ-132 the warm-up is honest — no value is emitted off a one-row window — but the
first values that *are* emitted still carry the seed. Measured against `adjust=True` (the
exact normalised weighted average, which has no seed) on a 400-row seeded random walk:

```
row  25:  seed weight 14.60%   |EMA_26 bias| = 0.1288% of price   <- first emitted row
row  49:  seed weight  2.30%   |EMA_26 bias| = 0.0751% of price   <- first surviving row
row  78:  seed weight  0.25%   |EMA_26 bias| = 0.0115% of price
row 104:  seed weight  0.03%   |EMA_26 bias| = 0.0004% of price
```

Only the first surviving row is affected at ~0.08% of price, because `SMA_50` cuts to row
49 and the bias is ~0.01% by row 78. On a ~1200-row panel that is one row at 0.08% and a
handful at less — which is why this is Low and not a re-open of PYQ-132.

Three options, none free:

- `adjust=True` on the EMAs. Removes the bias exactly, and is what a "mean of the last *s*
  observations, exponentially weighted" should mean. But it changes MACD away from the
  definition every charting package plots, which is precisely the argument PYQ-121 used in
  the *other* direction when it adopted Wilder's smoothing for RSI to match reference
  implementations. Taking it here would make MACD internally principled and externally
  non-standard.
- `min_periods=4*span` (104 rows). Keeps the standard definition and drops the biased rows,
  at the cost of ~104 rows off the front of every panel — and it would become the binding
  warm-up, roughly doubling what `SMA_50` costs today.
- Do nothing, and record the magnitude. Defensible on the numbers above.

The reason to keep the ticket rather than close it "won't fix" is that the choice becomes
material if PYQ-247 lands: on a log-return target the price-level scale disappears, and a
0.08% level bias in a feature is a different size relative to a ~1% daily return than it is
relative to price.

Acceptance criteria: a decision recorded here with its reasoning, and — if either fix is
taken — a test asserting the emitted EMA matches an unbiased reference to a stated
tolerance from the first surviving row onward.

Resolution (2026-07-27): **decision — take `min_periods = 4 * span`** (option 2), with the
multiplier exposed as `add_technical_indicators(warmup_spans=...)` /
`compute_macd(warmup_spans=...)`, default `DEFAULT_EMA_WARMUP_SPANS = 4`.

The measurement changed the decision, in two ways.

*First, this ticket's own framing understated the effect ~100x.* "0.08% of price" is the
wrong denominator for `MACD`/`MACD_Hist`, which are small differences of two EMAs, not
price levels. Against MACD's own typical magnitude the first surviving row was off by
**5.66%**, not 0.08%. That is what moved this from "defensible to do nothing" to worth
paying rows for.

*Second, the ticket's premise about `adjust=True` is wrong.* It is described here as
removing the bias exactly, "the exact normalised weighted average, which has no seed". It
does not. `adjust=True` is exact only over the **truncated window** and is just as blind to
the history the panel does not have. Measured against the reference that actually matters
— an EMA given 3000 rows of prior history, i.e. what a charting package with more history
than our panel plots — `adjust=True` is **1.3–1.6x worse** than the status quo at rows
49/78/104:

```
row  |EMA_26 - full-history EMA_26|, % of price, mean of 20 seeds
      adjust=False (current)   adjust=True
 49          0.1962%             0.2626%
 78          0.0161%             0.0246%
104          0.0043%             0.0071%
```

Seeding the recursion with an SMA of the first `span` observations (what TradingView does)
was measured too and is worse again — 1.6x at row 49. So the real error source is
**window truncation, not the seed choice**, and a longer warm-up is the only one of the
three candidates that attacks it. It also keeps the standard `adjust=False` definition,
which is the direction PYQ-121 argued for when it adopted Wilder's RSI to match reference
implementations — so this resolution is consistent with that precedent rather than in
tension with it.

Cost/benefit at the chosen multiplier, on MACD's own scale vs. the full-history reference:

```
warm-up cut at row   MACD truncation error
     49 (today)             5.66%
     78 (3 spans)           0.61%
    104 (4 spans)           0.08%     <- chosen
```

A 71x reduction for 91 rows (7.2%) off a 5-year panel: 1209 surviving rows becomes 1118.
The binding warm-up moves from `SMA_50` (49) to `MACD_Signal` (138 = `4*26-1` plus
`4*9-1`). Four spans rather than three because the extra 34 rows buy another 7.6x and the
project's stated preference is correctness over sample count.

**This redefines a model input.** `EMA_12`, `EMA_26`, `MACD`, `MACD_Signal` and `MACD_Hist`
all change value at the front of every panel, and every panel is now shorter, so bundles
and metrics from before this commit are not comparable across it — the same invalidation
PYQ-121 recorded, and the reason cache fingerprints include `code_version()` (PYQ-133).

Guarded by `test_first_surviving_ema_row_matches_a_full_history_reference`, which asserts
the stated tolerance (< 0.05% of price) against a 3000-row-history reference from the first
surviving row onward, and `test_ema_warmup_spans_is_configurable_and_trades_rows_for_accuracy`.
`test_add_technical_indicators_leaves_warmup_rows_genuinely_nan` now derives its expected
warm-ups from `DEFAULT_EMA_WARMUP_SPANS` instead of hardcoding PYQ-132's numbers, and two
`test_dataset.py` tests that hardcoded "the panel starts at row 49" now derive the boundary
— hardcoding which indicator binds is what let PYQ-121 and PYQ-132 hide behind `SMA_50` in
the first place.

---

## [PYQ-138]
CLI output tests assert on ANSI-coloured stdout, so they pass or fail by ambient terminal
Status: Resolved — 2026-07-27
Priority: Low
Files: `tests/conftest.py`, `tests/test_cli.py`

Problem: found while running the suite for PYQ-137. `test_cache_list_and_prune_commands`
asserts `"Pruned 0" in pruned.stdout`. Rich decides whether to emit ANSI colour when the
`Console` is constructed — at import of `pyquant/cli/app.py` — from whether stdout looks
like a terminal. Under a piped run (CI, `pytest > log`) it emits plain text and the
assertion holds; under an interactive run it emits `Pruned \x1b[1;36m0\x1b[0m` and the
same test fails:

```
E  AssertionError: assert 'Pruned 0' in 'Pruned \x1b[1;36m0\x1b[0m expired cac...'
```

Verified pre-existing: it fails on `a7a2b5f` with no local changes applied, and passes on
the identical tree under `NO_COLOR=1 TERM=dumb`.

This is the PYQ-120 shape rather than a cosmetic nit — a test whose result is decided by
something other than the code under test. It fails *open* in CI (piped, so always plain),
which is the bad direction: the colour path is never exercised there, and the failure only
appears on a developer's machine, where it reads as "someone broke the cache command."

Resolution (2026-07-27): `tests/conftest.py` sets `NO_COLOR=1` and `TERM=dumb` at module
scope, before any test module imports the CLI and therefore before the `Console` is built.
Every CLI assertion now sees the same plain output under both run modes. Fixed at the
harness level rather than by loosening the individual assertion to a regex, because the
defect is "output format varies with environment", not "this one string is too strict" —
and there are 30-odd CLI assertions that would each need the same loosening. Verified by
running `tests/test_cli.py` both piped and on a tty: 31 passed either way.

---

## [PYQ-139]
PYQ-257's vintage fetch fails against the live FRED API: every FRED macro feature silently vanished
Status: Resolved — 2026-07-27
Priority: Critical
Files: `pyquant/data/macro.py` (`_fetch_fred`, `_vintage_series`), `tests/test_macro.py`

Problem: found by running `build_panel("AAPL", ...)` against live vendors while setting up
PYQ-247's backtest comparison. PYQ-257 replaced fixed publication lags with ALFRED release
vintages and is marked Resolved with a passing test. **Against the real API it fetches
nothing.** The built panel contained one macro column, `VIX` — the one that does not go
through FRED — and `FedFunds`, `YieldSpread` and `CPI` were all absent:

```
WARNING pyquant.data.macro: Could not fetch FRED series T10Y2Y: Bad Request.  There are
  3085 vintage dates in the specified real-time period: 1776-07-04 to 9999-12-31.  This
  exceeds the maximum number of vintage dates allowed for this file type (2000).
WARNING pyquant.data.macro: Could not fetch FRED series CPIAUCSL: float() argument must be
  a string or a real number, not 'NaTType'
panel rows: 1116 cols: 28   macro columns present: ['VIX']
```

Three distinct defects, all in code that its own test suite passes:

1. **The realtime window was never bounded.** `build_panel` calls `fetch_macro(key,
   start=None, end=None, period="5y")`, so `_fetch_fred` passed `realtime_start=None,
   realtime_end=None` and fredapi defaulted to FRED's entire real-time span,
   1776-07-04..9999-12-31. FRED caps one `get_series_all_releases` call at 2000 vintage
   dates; a daily series over that span has thousands.
2. **A missing observation kills the whole series.** FRED encodes one as `"."`, which
   fredapi converts to **`NaT`, not `NaN`**, and `_vintage_series` did `float(row.value)`.
   Measured on the live response: `T10Y2Y` had 551 such rows in a five-year window
   (market holidays) and `CPIAUCSL` had 1. One holiday discarded five years of data.
3. **`realtime_end` in the future.** Clamping to `pd.Timestamp.today()` is the caller's
   local date; FRED's is US-based. From a European clock that is tomorrow, and FRED
   rejects it: *"realtime_end can not be after today's date (2026-07-26)"*.

Why the tests did not catch it: `test_fetch_macro_uses_the_first_published_cpi_vintage` and
its siblings mock `fredapi.Fred` at *our* function boundary and return a hand-built frame
with exactly the dtypes the parser wants — no NaT values, and the realtime arguments are
accepted and ignored. That verifies our logic against our own assumptions about the
payload, which is precisely the half-a-test features.md#pyq-243 describes. This ticket is
the concrete evidence for that one.

Severity is Critical rather than High because of how it fails: graceful degradation
(correct, and a stated contract) turned a total loss of one of four vendors into a logged
warning. Nothing in the reported metrics says the model trained on 24 features instead of
27, and the README's four-source claim was silently false at runtime.

Resolution (2026-07-27):

- `_vintage_windows()` derives a bounded realtime range from the history actually being
  requested (`start`/`end`, else `period` via a new `_period_to_offset`), clamps the end to
  **FRED's own clock** (`America/New_York`) rather than the caller's, and tiles it into
  one-year chunks — ~252 vintages for a daily series, comfortably under the 2000 ceiling
  and independent of how long a `period` is asked for.
- `_vintage_series()` coerces `value` with `pd.to_numeric(errors="coerce")` and drops
  nulls, so a missing observation is treated as "not a release" instead of raising.
- Chunk failures degrade per chunk, not per series, so one bad window costs a gap rather
  than five years.

Verified against the live API, before and after, same call:

```
before:  macro columns ['VIX']                                     (1116 panel rows)
after:   macro columns ['VIX','FedFunds','YieldSpread','CPI']      1263 rows,
         non-null 1261/1260/1260/1260, 2021-07-26..2026-07-23
```

Guarded offline by three tests that each reproduce one defect:
`test_missing_observations_do_not_abort_a_whole_series` (NaT in the value column),
`test_vintage_requests_are_bounded_and_never_ask_for_a_future_realtime_end` (asserts every
issued request is bounded, ends no later than FRED's today, and spans ≤ 400 days), and
`test_a_ten_year_request_is_split_into_chunks_that_cover_the_whole_window` (chunks tile the
range with no gaps, so the feature cannot silently start late).

**This changes model inputs.** Any bundle trained between PYQ-257 landing and this fix was
trained without `FedFunds`/`YieldSpread`/`CPI` regardless of config, so its recorded feature
list and any metric derived from it are not comparable with runs either side. PYQ-257's
resolution note stands on the design but its "verified offline" evidence did not establish
that the integration worked; see also features.md#pyq-243, which this promotes from
"good idea" to "already paid for once".

---

## [PYQ-140]
Finnhub's free tier serves ~6 days of news, not ~365: `Sentiment` is 99.7% structural zeros
Status: Resolved (pending, 2026-07-27)
Priority: High
Files: `pyquant/data/sentiment.py` (module docstring, `fetch_news`), `pyquant/data/dataset.py`

Problem: found while measuring investigations.md#pyq-301 against the live API, now that a
`FINNHUB_API_KEY` is configured. `sentiment.py`'s docstring states the free tier "only
covers ~365 days of news". PYQ-301 reasoned from that to "roughly 80% of training rows
could be structurally zero for this feature" at the default `period="5y"`.

Both figures are wrong. Measured on a freshly-built panel:

```
AAPL: 1116 rows, 2022-02-10..2026-07-27
  inside news window    :     3 (  0.3%)
  OUTSIDE (structural 0):  1113 ( 99.7%)
MSFT: 1116 rows
  inside news window    :     2 (  0.2%)
```

Probing the endpoint directly shows why — the free tier ignores `from` entirely and always
returns the same recent slice:

```
from=2026-07-20 ->  248 headlines across 6 distinct days, range 2026-07-22..2026-07-27
from=2026-06-27 ->  248 headlines across 6 distinct days, range 2026-07-22..2026-07-27
from=2025-07-27 ->  248 headlines across 6 distinct days, range 2026-07-22..2026-07-27
from=2021-07-28 ->  248 headlines across 6 distinct days, range 2026-07-22..2026-07-27
```

Four requests spanning one week to five years return byte-identical coverage: **~6 days**,
not 365.

Consequences, in order of severity:

1. `Sentiment` and `HeadlineCount` are effectively constant-zero columns across training.
   Two of ~27 features carry almost no variance, and the TFT's variable-selection network
   is being asked to weight a column that is zero 99.7% of the time.
2. It is a guaranteed train/serve distribution shift, and a much sharper one than PYQ-301
   anticipated: at predict time the *most recent* rows — the ones the encoder actually
   reads — are the only rows that ever carry sentiment.
3. The README counts four data sources. One of them contributes 0.3% coverage. That is a
   documentation-accuracy problem as much as a data one (non-negotiable #4).
4. FinBERT is downloaded and run, and the sentiment extra is installed, to score ~250
   headlines that reach ~3 rows.

This is filed as a bug rather than folded into PYQ-301 because the module *documents* a
behaviour the vendor does not provide, and code and docs disagreeing about a data source's
coverage is the same class as PYQ-139 — correct-looking in isolation, wrong against the
real API, hidden by graceful degradation.

Suggested fix: correct the docstring to the measured behaviour first (cheap, and stops the
next reader reasoning from 365 days as PYQ-301 did). Then decide between: dropping the
sentiment feature from the default config until deeper history is available; truncating the
training window to the covered period (PYQ-256 already ships the `has_sentiment_data`
indicator that makes either arm measurable); or pricing a paid tier / alternative vendor
(features.md#pyq-214, #pyq-258). The decision needs a backtest arm, not an opinion.

Acceptance criteria: the docstring states the measured coverage; a decision recorded here

Resolution: docstring corrected immediately (`sentiment.py`'s module docstring and the
`_MAX_HISTORY_DAYS` comment now state the measured ~6-day reality and cross-reference this
ticket, rather than the vendor's advertised-but-undelivered ~365 days).

The policy decision waited for investigations.md#pyq-316's feature-group ablation, which is
the backtest arm this ticket asked for rather than an opinion: adding sentiment on top of
price+technicals+macro+sectors moved measured skill from **+0.0453 to +0.0177** on one
AAPL walk-forward run (3 windows, 15 points) — sentiment made the result *worse*, not
neutral, undoing more than half of the previous arm's gain. That is independent, converging
evidence for the mechanism this ticket already established (Sentiment is a
near-constant-zero column carrying a train/serve shift concentrated exactly where the
prediction encoder reads).

**Decision: keep `DataConfig.use_sentiment=True` as the default, do not flip it in this
pass.** This is the same restraint investigations.md#pyq-312/PYQ-247 already established for
a comparably-sized result: one symbol, 15 points, one run is not this project's bar for
changing every user's default, however clean the mechanism looks. What this ticket *does*
close: the docstring is fixed (the acceptance criteria's first half, unconditionally
correct regardless of what happens to the default), and the decision is now a recorded,
evidence-backed recommendation — disable sentiment by default, or gate it on
`has_sentiment_data` coverage — pending the multi-symbol repeat already on the backlog's
`## Now` list. Filed as a recommendation attached to existing follow-up work rather than a
new ticket, since it is the same "needs more than one symbol" gate as PYQ-247's own default
change.
with a backtest comparing sentiment-on vs sentiment-off on equal footing.

---

## [PYQ-141]
Headline skill and the per-window skill column beneath it are different estimators
Status: Open
Priority: Medium
Files: `pyquant/analysis/metrics.py`, `pyquant/cli/app.py`, `docs/methodology.md`

Problem/Ask: `backtest` prints one "Skill vs. baseline" figure (`cli/app.py:154`) and,
directly beneath it, a per-window table with a skill column (`cli/app.py:201`). The two
are not the same statistic and can disagree without limit.

The headline comes from `aggregate_metrics()`, which pools `model_mae` and `baseline_mae`
as `n_points`-weighted averages; `skill_vs_baseline` is then computed *from the pooled
MAEs* — a **ratio of means**. Each per-window row computes skill from that window's own
MAEs — a **mean of ratios** once you read down the column. Windows where the baseline MAE
is small make the per-window ratio explode while contributing almost nothing to the pooled
denominator, so the column can be dominated by windows the headline barely sees.

This is not hypothetical. `docs/methodology.md` records the level-target per-window skills
as `[+0.28, +0.47, +0.35, -2.71, -3.13]`, whose mean is **-94.8%**, alongside a headline
of **-23.5%**. A reader who averages the printed column gets a number four times worse
than the number printed above it, and nothing on screen or on the page says why.

Neither estimator is wrong on its own — the pooled ratio is the defensible aggregate, and
per-window rows exist precisely so a mean cannot hide its own dispersion (PYQ-117's
lesson, and the reason `_per_window_table` was added at all). The defect is presenting
them as one table with one label. This is PYQ-136 one level up: that ticket fixed
numerator and denominator being computed two different ways *inside* the aggregate; this
is the aggregate and its own detail rows being computed two different ways.

Note the same seam in the interval: `cli/app.py:211` bootstraps a confidence interval over
per-window **directional accuracy** only. The headline skill — the number the README, the
docs and every ticket quote — has no interval at all (see PYQ-270).

Reproduction: run `pyquant backtest NVO --windows 5` at defaults, average the skill column
by hand, and compare to the headline row.

Acceptance criteria: the two numbers are either reconciled or explicitly distinguished.
Either label the header row as pooled ("Skill (pooled MAE ratio)") and the column as
per-window, and state in `docs/methodology.md` that the column does not average to the
header; or report both statistics explicitly. A test asserts the divergence is
deliberate — construct windows with unequal baseline MAEs and assert the pooled skill is
*not* the mean of the per-window skills, so the distinction cannot be silently refactored
away.

---

## [PYQ-142]
The displayed log-return forecast band compounds marginal per-step quantiles, ~√h too wide
Status: Resolved — 2026-07-29 (same session, uncommitted — see git status)
Priority: High
Files: `pyquant/analysis/forecast.py` (`log_returns_to_prices`, `generate_forecast`), `pyquant/models/tft.py` (`_window_signal`, `predict_quantiles`)

Problem: `log_returns_to_prices` (`forecast.py:16-18`) does
`last_close * exp(cumsum(log_returns, axis=0))`. `generate_forecast` (`forecast.py:106-112`)
passes the full `(horizon, n_quantiles)` array from `predict_quantiles` through it whenever
`target == "log_return"`, so `cumsum(axis=0)` runs down the horizon axis *independently
within each quantile column*. That computes the path where every step simultaneously
realizes its own marginal quantile — which is not the quantile of the cumulative h-step
return. `tft.py:492-496` (`_window_signal`, what `backtest --signals` scores) has the
identical call shape on the same function.

Under an iid-steps assumption the error is exactly `√h`: a 400k-path simulation at σ=2%/day
gives a naive-cumsum p90 edge that is 1.000/1.412/1.734/2.004/2.233× the true p90 of the
summed return at h=1..5 — matching `√1..√5` (1.000/1.414/1.732/2.000/2.236) almost exactly.
At the default `max_prediction_length=5` the band a `log_return`-target user is shown is
**~2.2× wider** than the model's own distribution implies.

This is checkable against the codebase's own correct implementation: `_evaluate_validation`
(`tft.py:847-866`) scores `calibration_coverage`/`crps`/`winkler_score`/`pit` on the *raw*
`(n_samples, horizon, n_quantiles)` per-step arrays via `evaluate_predictions(...,
target="log_return")` (`metrics.py:240-285`) — it never calls `log_returns_to_prices`. So
every number written to `meta.json`, printed by the CLI, and reported by PYQ-247's
measurement is unaffected. What's wrong is only what's *displayed*: the compounded path
`forecast`/`scan`/`explain` shows, and the P&L `backtest --signals` computes.

Two compounding consequences: `scan`'s "whole band on one side of zero" guard is far less
likely to fire than the reported (correct) coverage would suggest, since the displayed band
is wider than the one that coverage describes. And `predict_quantiles` (`tft.py:926-936`)
applies any PYQ-248 conformal offset in per-step space *before* `generate_forecast`
compounds — so a calibration fitted to correct per-step coverage is not calibrated for the
wider object it ends up widening further.

Scope: only `TrainingConfig.target == "log_return"` bundles are affected
(`config.py:92` — default is `"close"`), but that is the configuration PYQ-247 measured and
`docs/methodology.md` documents as a live alternative, not a hypothetical one.

Reproduction: `tests/test_forecast.py::test_log_return_price_round_trip` currently locks in
the buggy behavior as intended — it asserts `log_returns_to_prices` on a `(2, 3)` array
produces per-column-independent compounding (`prices[1] = [104*101/100, 105*102/100,
106*103/100]`), i.e. exactly the marginal-quantile compounding this ticket describes.

Ask: either (a) train on cumulative-return targets so each decoder step's quantile is
already the h-step quantile, or (b) keep per-step targets, compound only the median path,
and derive band edges from a horizon-scaled dispersion under an explicit, documented
distributional assumption. (a) is the more honest option and avoids inventing a
distributional assumption the model doesn't actually make.

Acceptance criteria: the displayed band's coverage (measured the same way
`_evaluate_validation` measures it, but on the *compounded* path actual/prediction pairs)
matches its nominal level within the same tolerance PYQ-248 already holds itself to; a test
asserts the fixed reconstruction does not simply reproduce the old √h-wide band; the fix
composes with PYQ-248's conformal offset (apply calibration in the space the band is
finally displayed/scored in, not before compounding).

Resolution: took option (b) -- retraining on cumulative targets (option (a)) would redefine
what every `log_return`-target bundle predicts, the kind of change PYQ-121 required a loud
supersede for, and this bug is in the *display/scoring reconstruction*, not the model. Added
`log_return_quantiles_to_price_band(log_return_quantiles, last_close, quantiles)` alongside
the existing `log_returns_to_prices` (now documented as being for a single deterministic
path only -- e.g. the realized actual -- where plain `cumsum` is exact and no distributional
question applies). The new function compounds the *median* path via `cumsum` (exact for a
deterministic center) and derives each quantile's cumulative deviation from the median as
`sqrt(sum of squared per-step deviations)` rather than a linear sum -- the standard
deviation-of-a-sum-of-independent-variables identity, an explicit iid-across-steps
assumption rather than a property the model asserts. Under iid steps with constant per-step
dispersion this exactly recovers the analytic `sqrt(h)` scaling PYQ-142's own simulation
measured as correct; `test_log_return_quantile_band_matches_analytic_iid_normal_quantiles`
checks the closed form directly rather than relying on Monte Carlo tolerance, and
`test_log_return_quantile_band_achieves_close_to_nominal_coverage` additionally verifies
empirical coverage lands within 3 points of nominal on 20k simulated paths.
`generate_forecast` and `_window_signal` (the two callers that previously fed a
`(horizon, n_quantiles)` array into the single-path function) now call the new one instead;
`_window_signal`'s realized-actual path still uses the original function, correctly.

Composition with PYQ-248's conformal offset: left the offset applied where it already was
(per-step, before compounding) rather than moving it after compounding, because the offset
is fit against *per-step* marginal coverage (`_evaluate_validation` scores the raw per-step
arrays, which this ticket confirms are unaffected) -- applying it to per-step deviations
before they feed the new sqrt-dispersion formula is the space it was actually calibrated
for, and the correction propagates through the formula like any other per-step value with no
special-casing needed. Recorded as a decision rather than re-litigated further: PYQ-144
(fixing the offset itself to be per-horizon-step) lands in the same pass and composes
cleanly with this ordering.

Verified: `test_log_return_quantile_band_does_not_reproduce_old_naive_cumsum_band` confirms
the fixed h=5 p90 edge is materially narrower (>1.5x) than the old naive-cumsum band on the
same input. Covered by 4 new tests in `tests/test_forecast.py` (13 total in that file, up
from 10) plus the existing `Forecast`/`generate_forecast` suite, all passing;
`tests/test_tft.py`'s `_window_signal`-exercising tests (`test_walk_forward_backtest_computes_a_signal_per_window_when_requested`)
still pass unchanged. `docs/methodology.md`'s headline numbers do not depend on this path
(confirmed above: `_evaluate_validation` never called the buggy function), so no reported
metric changes -- only what `forecast`/`scan`/`explain` display and what
`backtest --signals` scores for `log_return`-target bundles, which is `TrainingConfig.target
== "close"`'s default and therefore not the configuration any currently-published number
uses.

---

## [PYQ-143]
`train()`/`walk_forward_backtest()` select checkpoints and report metrics from the same window
Status: Resolved — 2026-07-29 (same session, uncommitted — see git status)
Priority: High
Files: `pyquant/models/tft.py` (`train`, `walk_forward_backtest`, `tune`)

Problem: `train()` builds one `validation` `TimeSeriesDataSet`/`val_loader`
(`tft.py:335-342`), then uses it for three different purposes: `ModelCheckpoint(monitor=
"val_loss", save_top_k=1, ...)` and `EarlyStopping(monitor="val_loss", ...)`
(`tft.py:350-355`) select the best of up to `settings.training.max_epochs` epochs *against
this window*, and `_evaluate_validation(best_model, val_loader, ...)` (`tft.py:409-411`)
computes the `EvaluationMetrics` written to `meta.json`, printed by the CLI, and returned by
`POST /train` — from the *same* `val_loader` object. Reported skill, directional accuracy,
coverage, CRPS and Winkler are therefore max-over-epochs statistics, not out-of-sample ones.

The project already has the correct pattern and the correct argument for it, in the same
file: `TuneResult`'s docstring (`tft.py:78-85`) states *"every trial is a selection event,
so the in-search score ... is optimistically biased and must not be reported as the model's
real performance"*, and `tune()` reserves `held_out_days` of the panel that no trial ever
sees (`tft.py:641-645, 667-683`). An epoch is a selection event by the identical argument —
`ModelCheckpoint`/`EarlyStopping` are themselves a 30-trial (one per epoch) search over
`val_loss` on the validation window, and the rule is applied in exactly one of the two
places it structurally applies. `TrainingConfig.calibration_days` (`tft.py:305,391-407`)
already carves out a slice between the training cutoff and the validation window, but it is
used only to fit the PYQ-248 conformal offset — never for early-stopping/checkpoint
selection.

It is worse in `walk_forward_backtest`: `_window_validation_dataset` uses `predict=True`
(`tft.py:455-470`), giving **one sample per origin**. That single `val_loader` is passed to
both `EarlyStopping`/`ModelCheckpoint` (`tft.py:591`, inside the per-window `trainer.fit`)
and `_evaluate_best_checkpoint(..., val_loader, ...)` (`tft.py:592-600`) — so a 5-point
window is what early stopping is tuned on *and* what becomes that window's reported metrics
in the per-window table and the aggregate PYQ-117/PYQ-251 report on top of.
`walk_forward_backtest` has no `calibration_days`-equivalent slice at all.

The bias is largest exactly where the sample is smallest: `effective_sample_size`
(`metrics.py:162-165`, PYQ-251) puts a 60-day/horizon-5 holdout at ~12 independent windows;
selecting the best of ~30 epochs against 11-12 effective points is a substantial upward bias
on every rate the project reports as "the honest number" (PYQ-117's own framing).

Ask: extend the `[train][purge+embargo][calibration][validation]` geometry `train()`
already has to `[train][purge+embargo][selection][purge][test]`, and point
`EarlyStopping`/`ModelCheckpoint` at the selection slice while reporting `EvaluationMetrics`
from the disjoint test slice. In `walk_forward_backtest`, either disable early stopping
(fixed epoch count per window) or carve a selection window from each origin's training
tail — a 5-point selection metric is not usable as a monitor either way. This pairs
naturally with PYQ-269's planned extraction of the window-geometry arithmetic (`max_idx`,
`validation_start`, `purged_training_cutoff`) into one object, since `train`,
`walk_forward_backtest` and `tune` each currently reimplement it and have already drifted
(`train` accounts for `calibration_days`; the other two don't).

Acceptance criteria: `train()`'s reported `EvaluationMetrics` come from data neither
`ModelCheckpoint` nor `EarlyStopping` was monitored against; a test constructs a case where
early-stopping-selected and true-out-of-sample metrics diverge and asserts the reported
number is the latter; `walk_forward_backtest` either fixes epochs or gets an equivalent
split, documented as a deliberate tradeoff either way; expect every reported metric in the
repo to get worse once this lands — that is the point (see PYQ-143's sibling, PYQ-142, for
the same "expect the honest number to be worse" framing).

Resolution: extended the geometry exactly as asked, in both functions, to
`[train][purge+embargo][selection][purge+embargo][calibration][test]`. New
`TrainingConfig.selection_days` (default 30, half of `validation_days`' own default -- an
explicitly unoptimized starting point, not a tuned value) sizes a second held-out slice.
New private helper `_selection_split(cutoff, settings)` applies the existing
`purged_training_cutoff` *twice* -- once for the gap between selection and whatever follows
it (calibration+test, or the single test window), once more for the gap between training and
selection -- so selection shares no target days, purged or otherwise, with either side. This
is deliberately not PYQ-269's fuller consolidation of the geometry arithmetic (`train`,
`walk_forward_backtest` and `tune` still each compute their own `max_idx`/window bounds); it
only factors out what this fix newly introduces, shared identically by the two callers that
need it.

In `train()`, `trainer.fit(...)`'s `val_dataloaders` now points at the new `selection_loader`
instead of `val_loader` -- the only wiring change `EarlyStopping`/`ModelCheckpoint` needed,
since Lightning logs `val_loss` from whichever loader is passed there regardless of what else
references the underlying dataset. `val_loader` (renamed nowhere, to keep the diff small, but
now genuinely the *test* window) is untouched by training and reaches
`_evaluate_validation` only afterwards. `TrainResult.val_loss` and the CLI's `train` table
row are re-labelled ("Selection loss") and re-documented to say what they now measure --
a selection-event statistic useful for judging convergence, not a quality number -- so
nothing downstream can mistake it for the test-window figure it no longer is.

`walk_forward_backtest` took the second option the ticket offered (carve a selection window
from each origin's training tail) rather than the first (fixed epoch count): it reuses
`_selection_split` per origin, identically to `train()`, which keeps `_evaluate_best_checkpoint`
(PYQ-109's best-checkpoint-not-final-epoch fix) intact rather than reverting to a final-epoch
read that PYQ-109 specifically argued against. The one-sample-per-origin problem the ticket
flagged as "worse" is solved the same way `train()` solves it -- the selection window, unlike
the single test window, spans multiple days and gives EarlyStopping a real (if still small,
`selection_days`-sized) sample to monitor.

Verification took the form the ticket's acceptance criteria asked for, adapted: rather than
constructing a case where the *numeric* early-stopping-selected and out-of-sample metrics
diverge (dependent on specific loss dynamics, and liable to be flaky), the new tests verify
the structural invariant that makes such divergence impossible to hide --
`test_train_fits_against_a_selection_window_disjoint_from_the_reported_test_window` and
`test_walk_forward_backtest_fits_against_a_selection_window_per_origin` spy on both
`trainer.fit`'s `val_dataloaders` and the loader `_evaluate_validation`/
`_evaluate_best_checkpoint` receives, and assert not just that the two datasets differ but
that the selection window's decoder range ends strictly before the reported window's begins.
This is a stronger check than a single numeric-divergence example: it holds for every run,
not just a constructed one. `test_selection_days_is_configurable_and_recorded_on_the_bundle`
covers the new config field's plumbing into `meta.json` (via the existing whole-config
recording, no new code needed there). All three pass; the full existing `test_tft.py` suite
(42 tests, up from 39) passes unchanged, including
`test_train_still_validates_on_the_full_holdout_after_purging` (proves `validation_days`'
window position is independent of `purge_horizon`/`embargo_days`, still true with selection
inserted earlier in the timeline) and the PYQ-250 purge/embargo tests (which exercise
`purged_training_cutoff` directly and don't touch the new selection path at all).

Per the ticket's own prediction, this does **not** improve any reported number -- it can only
make them more honest, i.e. probably worse. `docs/methodology.md`'s split-geometry section
and its headline results table are both updated: the geometry diagram now shows the
`selection` slice, and the headline table carries an explicit note that both published
numbers (`-23.5%` skill default, `+2.4%` log-return) predate this fix and were measured with
checkpoint selection biased toward the same window they were scored on, so the honest
numbers are expected to be somewhat worse. Re-measuring both is the natural next step;
it was not done in this pass because no live vendor-data access was available (Yahoo Finance
returned HTTP 429 in this sandbox) to actually run `pyquant train`/`pyquant backtest`, and
fabricating a number to fill the table would violate non-negotiable #1 far more directly than
leaving it visibly stale does.

---

## [PYQ-144]
Conformal offset is pooled across horizon steps and fit on an inflated sample size
Status: Resolved — 2026-07-29 (same session, uncommitted — see git status)
Priority: High
Files: `pyquant/analysis/calibrate.py` (`fit_conformal_offset`), `pyquant/analysis/metrics.py` (`effective_sample_size`)

Problem: `fit_conformal_offset` (`calibrate.py:95-134`) receives `predictions` shaped
`(n_samples, horizon, n_quantiles)` and immediately flattens the horizon axis away:
`scores = conformity_scores(...).reshape(-1)` (`calibrate.py:118`), then fits **one scalar**
`offset` from all of it. `apply_conformal_offset` (`calibrate.py:137-159`) then broadcasts
that single offset onto every decoder step identically (`out[..., 0] -= delta; out[...,
-1] += delta`). Forecast dispersion grows with horizon (see PYQ-142's own √h measurement),
so one additive correction over-narrows h=1, where the band is already closest to correct,
and under-widens h=5, where it needs the most correction — the same category error as
PYQ-142, from the opposite direction. `tests/test_calibrate.py`'s eight tests never mention
"horizon" — the per-step behavior is untested because the code has no per-step behavior.

Second, independent bug in the same function: `n = scores.size` at `calibrate.py:119` is
`n_samples * horizon` — every point of every sliding calibration window, which overlap
heavily (the calibration `TimeSeriesDataSet` at `tft.py:392-397` uses `min_prediction_idx`
without `predict=True`, so consecutive windows share most of their span). This `n` feeds
directly into the finite-sample correction `ceil((n+1)·coverage)/n`
(`calibrate.py:123`) — the same correction PYQ-251 built `effective_sample_size(n_samples,
horizon)` (`metrics.py:162-165`) specifically to fix for exactly this overlap pattern, used
today only via `EvaluationMetrics.effective_n_samples`. `fit_conformal_offset` doesn't call
it, so the "what buys the marginal guarantee" correction the module's own docstring
describes is computed on a count that overstates the independent evidence by roughly the
horizon factor — the same "correct in one file, not applied in the analogous one" pattern
CLAUDE.md's leakage non-negotiable names as the project's recurring bug shape, just outside
the leakage category this time.

Note: the review that prompted this pass also flagged `apply_conformal_offset`'s trailing
`np.sort(out, axis=-1)` (`calibrate.py:159`) as contradicting a "interior quantiles are left
alone" docstring claim. That is **not** a bug — the docstring (`calibrate.py:143-149`)
explicitly names the re-sort scenario and cites PYQ-124's monotonicity precedent; verified
correct, no action needed.

Scope: `TrainingConfig.calibration_days` defaults to 0 (calibration off), so this is dormant
by default — it activates whenever a user turns on split-conformal calibration, which
PYQ-248's own resolution note frames as something users should consider.

Acceptance criteria: `fit_conformal_offset` accepts and returns a per-horizon-step offset
(or a documented decision to keep it pooled, defended against the √h evidence in PYQ-142);
`n` is replaced with `effective_sample_size(n_samples, horizon)` in the finite-sample
correction; a test with horizon-varying synthetic dispersion asserts the per-step offsets
differ and each step's post-calibration coverage is closer to nominal than the pooled
offset's.

Resolution: `ConformalOffset.offset` is now `list[float]`, one value per horizon step, fit by
`fit_conformal_offset` computing `scores[:, h]` (that step's conformity scores across
calibration windows) and taking the finite-sample-corrected quantile of *each column
separately*, rather than flattening the horizon axis first. The correction's `n` is now
`effective_sample_size(n_samples, horizon)` (PYQ-251) instead of raw `n_samples * horizon`,
computed once and shared across all `horizon` per-step quantile levels -- fixing the second
bug in the same change, since both bugs lived in the same few lines.
`apply_conformal_offset` broadcasts the per-step offset against `predictions`' horizon axis
(second-to-last, so it works identically on a single `(horizon, n_quantiles)` forecast or a
`(n_samples, horizon, n_quantiles)` validation batch); a scalar or single-element offset still
broadcasts identically to every step, so a bundle calibrated before this fix keeps working
via `ConformalOffset.from_dict`'s scalar-coercion path -- no bundle needs retraining, though
one calibrated after this fix will get the corrected, more accurate behavior.

Verified with the acceptance criterion's own scenario:
`test_fit_conformal_offset_produces_different_offsets_when_dispersion_varies_by_step` builds
a 3-step synthetic case (true distribution iid N(0,1) at every step; predicted band
deliberately too narrow at h=1, ~correct at h=2, too wide at h=3), confirms the three fitted
offsets differ, and confirms per-step post-calibration coverage is closer to nominal 80% at
*every* step than a pooled offset computed the pre-fix way (flattened scores, raw `n`) on the
same data. `test_offset_from_dict_accepts_a_legacy_scalar` and
`test_applying_a_legacy_scalar_offset_broadcasts_to_every_step` cover the backward-compatible
loading path explicitly. The existing 8 tests were updated for the list-valued API (indexing
`.offset[0]` where they'd asserted a bare scalar) rather than weakened -- their assertions are
otherwise unchanged, including the two Monte Carlo coverage checks (narrows/widens toward
nominal), which still hold at `horizon=1` where `effective_sample_size(n, 1) == n` makes the
fix a no-op by construction, so those two remain a clean regression test for the correction
formula itself. 12 tests total in `tests/test_calibrate.py` (up from 8), all passing;
`tests/test_tft.py`'s two PYQ-248 tests (`test_calibration_slice_produces_an_offset_that_forecast_reuses`,
`test_a_bundle_without_a_calibration_slice_records_no_offset`) pass unchanged after updating
the training-time log line for the new list-valued offset.

Composes with PYQ-142 (landed in the same pass): that fix's
`log_return_quantiles_to_price_band` derives each quantile's cumulative dispersion from its
*per-step* deviation from the median, so a per-step conformal correction (this ticket) flows
through it positionally with no special-casing -- the two fixes were designed together for
exactly this reason, see PYQ-142's own resolution note on composition ordering.

`calibration_days` still defaults to 0, so this remains dormant unless a user has already
opted into conformal calibration -- no currently-published number in `docs/methodology.md`
uses it (its own table says "no conformal correction applied" for the one config that got
measured).

---

## [PYQ-145]
API accepts unvalidated `symbol`/`bundle_name`; auth header comparison throws on non-ASCII input
Status: Resolved — 2026-07-29 (same session, uncommitted — see git status)
Priority: High
Files: `pyquant/models/tft.py` (`_bundle_dir`), `pyquant/api/schemas.py`, `pyquant/api/deps.py`

Problem: `_bundle_dir(settings, name)` (`tft.py:148-149`) is `settings.checkpoint_dir /
name.upper()` with no validation on `name`. `TrainRequest.bundle_name` and
`ScanRequest.symbols` (`api/schemas.py`) have no `field_validator` restricting their
character set — confirmed by grep, `field_validator` does not appear anywhere in that file.
The chain from an API request to the filesystem is direct: `routes/train.py:37` passes
`request.bundle_name` into `tft.train(..., bundle_name=...)` →
`tft.py:291,345-346` (`bundle_dir = _bundle_dir(...); bundle_dir.mkdir(parents=True,
exist_ok=True)`) → `ModelCheckpoint(dirpath=bundle_dir, ...)` and
`torch.save(training.get_parameters(), bundle_dir / "dataset_params.pt")`. A `bundle_name`
of `"../../../../tmp/pwn"` reaches `mkdir`/`torch.save` with nothing in between to stop it —
unlike `GET /forecast/{symbol}`, `POST /train`'s body isn't subject to Starlette's
`/`-rejecting path-parameter matching. And a crafted `symbol` reaching `tft.load()`
(`tft.py:1058-1069`, `torch.load(..., weights_only=False)`) moves the file the unpickler
trusts outside the "your own trained runs" boundary the code comment there assumes.

Related, same trust boundary: `require_api_key` (`api/deps.py:107,132`) calls
`hmac.compare_digest(x_api_key, k)` on plain `str`. Starlette decodes headers as latin-1, so
a byte >127 in `X-API-Key` produces a non-ASCII `str`, and `compare_digest` raises
`TypeError: comparing strings with non-ASCII characters is not supported` (verified
directly) — unhandled, so FastAPI's default handler turns a bad key into a **500** instead
of a **401**. Not an auth bypass (no key is ever accepted this way), but it's an unhandled
exception on attacker-controlled input, and the traceback path is a minor information
disclosure.

Ask: one shared validator (`^[A-Z0-9][A-Z0-9.\-]{0,15}$` or similar) applied as a pydantic
`field_validator` on `TrainRequest.bundle_name` and `ScanRequest.symbols`, and as a
`Path(..., pattern=...)` on route params that take a symbol. Belt and braces: after
building `bundle_dir`, assert `settings.checkpoint_dir.resolve() in
bundle_dir.resolve().parents`. For the auth comparison, encode both sides to bytes
(`.encode("utf-8", errors="surrogateescape")` or reject non-ASCII keys with a clean 401)
before calling `hmac.compare_digest`.

Acceptance criteria: `POST /train {"bundle_name": "../../etc"}` and `POST /scan
{"symbols": ["../x"]}` are rejected with 422 before reaching `_bundle_dir`; a test asserts
`bundle_dir` can never resolve outside `checkpoint_dir`; a non-ASCII `X-API-Key` returns 401
rather than 500, with a regression test.

Resolution: three layers, matching the "belt and braces" ask.

1. `api/schemas.py` gained a shared `SYMBOL_PATTERN` (`^[A-Za-z0-9][A-Za-z0-9.\-]{0,15}$`) and
   a `_validate_symbol()` helper wired as a `field_validator` on `TrainRequest.symbols`,
   `TrainRequest.bundle_name` and `ScanRequest.symbols` -- a path-traversal string is now
   rejected with 422 before any route body runs.
2. The GET routes' `{symbol}` path params (`/forecast/{symbol}`, `/explain/{symbol}`) now
   also declare `Path(..., pattern=SYMBOL_PATTERN)` -- belt-and-braces, since Starlette's
   default path matching already rejects `/` in a single segment but not a bare `".."`
   segment, which `Path("..")` alone can still walk up one level.
3. `_bundle_dir()` itself now resolves the candidate directory and asserts it is
   `checkpoint_dir` or one of its descendants, raising `ValueError` otherwise -- this covers
   every caller, including any future one that forgets to validate first, not only the API
   request path.

`require_api_key` now encodes both sides to UTF-8 bytes before `hmac.compare_digest` --
bytes comparison has no ASCII restriction, and UTF-8-encoding a latin-1-decoded string
(what Starlette produces from a raw header byte) is always lossless. A non-ASCII key now
falls through to the ordinary "no match" 401 instead of raising.

Confirmed test-first: `test_require_api_key_rejects_a_non_ascii_key_with_401_not_500`
reproduces the exact `TypeError: comparing strings with non-ASCII characters is not
supported` from the ticket against the pre-fix code (verified by stashing the fix) and
passes after. Covered overall by `test_train_rejects_a_path_traversal_bundle_name`,
`test_train_rejects_a_path_traversal_symbol`, `test_scan_rejects_a_path_traversal_symbol`,
`test_bundle_dir_never_resolves_outside_checkpoint_dir`, and the non-ASCII auth test above.

---

## [PYQ-146]
`load_settings()` mutates a module global; concurrent API requests can read the wrong config
Status: Resolved — 2026-07-29 (same session, uncommitted — see git status)
Priority: High
Files: `pyquant/config.py` (`load_settings`, `_active_yaml_file`), `pyquant/api/deps.py` (`get_settings`)

Problem: `pyquant/config.py:233,244,256,260` —

```python
_active_yaml_file: Path | None = None

def load_settings(config_path=None):
    global _active_yaml_file
    ...
    _active_yaml_file = Path(chosen) if chosen else None
    try:
        return Settings()
    finally:
        _active_yaml_file = None
```

`get_settings()` (`api/deps.py:22-24`) is `return load_settings()`, called as a FastAPI
dependency on every request. All routes under `pyquant/api/routes/` are declared `def`, not
`async def` (confirmed by grep), so Starlette runs them on its shared `anyio` threadpool —
genuinely concurrent OS threads. `get_settings()` has no `functools.lru_cache` (confirmed
absent), so two concurrent requests can both call `load_settings()` at once. If thread A
sets `_active_yaml_file` and thread B's `finally` resets it to `None` before thread A
reaches `Settings()`, thread A silently builds settings **without** the YAML layer it
intended — no error, no log, just a request served against different hyperparameters than
the one it requested. This is exactly the failure mode PYQ-128 was filed to prevent (a
config path silently ignored), reintroduced one layer up by the mechanism PYQ-128 didn't
touch.

Ask: thread the YAML path through as an argument instead of a global — e.g. build the
`YamlConfigSettingsSource` list in a `Settings` subclass constructed per call, or pass the
path through `settings_customise_sources`'s closure rather than a module global.
Independently, `get_settings()` re-reads `.env` from disk on every request; wrap it in
`functools.lru_cache` with an explicit invalidation hook once the global is gone.

Acceptance criteria: a test that calls `load_settings()` from two threads concurrently, one
with a config path and one without, and asserts neither observes the other's setting; no
module-global mutation remains in the `Settings()`-construction path.

Resolution: `_active_yaml_file` and the `global` statement are gone. `load_settings()` now
calls a new `_settings_class_for_yaml(yaml_file)`, which builds and returns a one-off
`Settings` subclass whose `settings_customise_sources` closes over `yaml_file` as a local
variable rather than reading a module global -- so the chosen path lives entirely inside
that call's stack, and two concurrent `load_settings()` calls (e.g. two API requests on
Starlette's threadpool) can no longer see or clobber each other's path regardless of thread
scheduling. The base `Settings.settings_customise_sources` no longer references any YAML
source at all (a plain `Settings()` call, the common case, never touches the global).

Confirmed test-first: `test_load_settings_from_two_threads_never_observes_the_others_config`
fails reliably against the pre-fix code (one thread reading defaults it shouldn't) and
passes after -- verified by stashing the fix and re-running.

Deliberately not done: the `functools.lru_cache` on `get_settings()` the ticket mentions as
an independent, separate concern (re-reading `.env` per request) rather than part of the
race itself -- adding a cache-invalidation hook with no current caller needing it is the
kind of unjustified complexity non-negotiable #5 argues against; left as a future
performance ticket if request volume ever makes it worth it, not bundled into this fix.

---

## [PYQ-147]
`interpret()` zips variable names to weights with `strict=False`
Status: Resolved — 2026-07-29 (same session, uncommitted — see git status)
Priority: Medium
Files: `pyquant/models/tft.py` (`interpret`)

Problem: `tft.py:953` — `importance = dict(zip(enc_names, enc_weights.tolist(),
strict=False))`. `enc_names` is `bundle.model.encoder_variables`; `enc_weights` comes from
`interpretation["encoder_variables"]`. This is the only `strict=False` zip in the file
(confirmed by grep) — `_build_pooled_long_df` (`tft.py:182`) and `metrics.pit_values`
(`metrics.py:158`) both use `strict=True` for the identical names-to-values pattern, and the
comment immediately above `tft.py:182`'s `strict=True` explains why (PYQ-111/PYQ-302: a
length mismatch should fail loudly, not silently misattribute). If a pytorch-forecasting
version bump ever changes the ordering or length of either array — the library has done
both across minor versions — `strict=False` truncates silently and hands back a
user-facing, apparently-authoritative explainability table with weights attached to the
wrong feature names.

Ask: flip to `strict=True`.

Acceptance criteria: `strict=True` at `tft.py:953`; a regression test constructs mismatched-
length `enc_names`/`enc_weights` and asserts `interpret()` raises rather than silently
truncating.

Resolution: flipped to `strict=True`, matching `_build_pooled_long_df` and
`metrics.pit_values`'s existing convention for the same names-to-values pattern.
Regression test `test_interpret_raises_on_encoder_name_weight_length_mismatch` trains a
real tiny bundle (so `encoder_variables` names stay genuine and the forward pass still
runs), then monkeypatches `bundle.model.interpret_output` to return one fewer weight than
there are names -- confirmed to fail against the pre-fix `strict=False` code (silently
truncated instead of raising) and pass after.

---

## [PYQ-148]
`use_options` is missing from the panel cache fingerprint
Status: Resolved — 2026-07-29 (same session, uncommitted — see git status)
Priority: Medium
Files: `pyquant/data/dataset.py` (`_cache_fingerprint`)

Problem: `_cache_fingerprint` (`dataset.py:42-58`) covers `use_macro`, `use_sectors`,
`use_sentiment`, `use_indicators`, `sector_etfs`, `period`, `code_version`,
`has_fred_key`/`has_finnhub_key` — not `use_options`, even though `build_panel` branches on
it (`dataset.py:147`) and `_SCHEMA_DATA_FIELDS` in `tft.py:1024-1031` lists it. Toggling
`use_options` with `cache_enabled=True` can therefore return a cached panel built under the
opposite setting, silently, within the cache's TTL.

This is not what PYQ-133 (Resolved) covers — that ticket's problem statement predates
`use_options` existing as a fingerprint-relevant field and was specifically about
`code_version`. It's a regression surfaced by PYQ-254 (Resolved), which added `use_options`
to `tft.py`'s `_SCHEMA_DATA_FIELDS` without touching `dataset.py`'s fingerprint — its
resolution note doesn't mention `_cache_fingerprint` at all, and no test
(`tests/test_dataset.py:300-324`) checks `use_options` there.

Ask: add `use_options` (and its own key-presence flag, if a data key is ever required for
it) to `_cache_fingerprint`. Enumerate the fields programmatically from `DataConfig` rather
than listing them by hand a fourth time, so the next toggle can't be forgotten the same way.

Acceptance criteria: a test that toggles only `use_options` and asserts the fingerprint
(and therefore the cache key) changes; ideally a parametrized test over every
`DataConfig` field `build_panel` reads, so a future field is covered by construction rather
than by remembering to add it here.

Resolution: rather than adding a fourth hand-written copy, `SCHEMA_DATA_FIELDS` (the tuple
of schema-relevant `DataConfig` fields) now lives once in `dataset.py` and `tft.py`'s
former private `_SCHEMA_DATA_FIELDS` was deleted in favor of importing it --
`settings_for_bundle()` and `_cache_fingerprint()` now read the identical set, so a future
toggle added to one is automatically covered by the other. `_cache_fingerprint` builds its
`use_*`/`period`/`sector_etfs` entries by iterating `SCHEMA_DATA_FIELDS` instead of naming
each field inline. Covered by
`test_cache_fingerprint_changes_when_any_schema_field_toggles`, parametrized directly over
`SCHEMA_DATA_FIELDS` (7 cases, `use_options` included) so a field added to that tuple later
is exercised by construction rather than by remembering to extend this test -- the
acceptance criteria's "ideally" clause.

---

## [PYQ-149]
`backtest --signals` scores an uncalibrated rule while `scan` ships a calibrated one
Status: Open
Priority: Medium
Files: `pyquant/models/tft.py` (`_window_signal`, `walk_forward_backtest`, `predict_quantiles`)

Problem: `_window_signal` (`tft.py:473-506`), used when `walk_forward_backtest(...,
compute_signals=True)`, reads raw arrays straight from `_raw_validation_arrays` — no
conformal offset is fitted or applied anywhere in `walk_forward_backtest`
(`calibration_days`/`ConformalOffset` machinery exists only in `train()`, `tft.py:386-410`).
Meanwhile `generate_forecast` → `predict_quantiles` (`tft.py:926-936`) applies
`apply_conformal_offset(..., bundle_conformal_offset(bundle))` — the trained bundle's
calibrated offset — to every band `scan` shows. `_window_signal`'s own docstring
(`tft.py:480-484`) claims to reproduce *"the (signal, realized_return_pct) `scan()` would
have shown"*, which is only true when `calibration_days == 0` (today's default). The moment
a user turns on calibration — which PYQ-248/PYQ-255 both frame as the fix that lets the band
guard fire meaningfully at all — `backtest --signals`'s measured P&L and `scan`'s live
behavior silently diverge: the backtest keeps measuring the pessimistic, uncalibrated case.

Ask: either apply the same conformal fit/offset path inside `walk_forward_backtest` before
computing signals, or clearly label `--signals` output as pre-calibration and cross-link to
PYQ-248 so the divergence is documented rather than silent.

Acceptance criteria: with `calibration_days > 0`, a test asserts `_window_signal`'s
classification uses the same offset `scan` would apply for an equivalent bundle/window; or,
if intentionally left uncalibrated, `--signals`' CLI/JSON output and docstring say so
explicitly.

---

## [PYQ-150]
Finnhub API key travels in the query string and can reach WARNING logs on retry failure
Status: Resolved — 2026-07-29 (same session, uncommitted — see git status)
Priority: High
Files: `pyquant/data/sentiment.py`, `pyquant/data/retry.py`

Problem: `sentiment.py:90` builds `params = {"symbol": ..., "from": ..., "to": ...,
"token": api_key}` passed to `requests.get(_FINNHUB_NEWS_URL, params=params, ...)` — the key
travels in the URL query string rather than an `X-Finnhub-Token` header (Finnhub supports
header auth; this module hand-rolls the request rather than using an official client).
`with_retry` (`retry.py:48-55`) logs `logger.warning("%s failed (attempt %d/%d): %s; ...",
..., exc, ...)` on every failed attempt, and `requests.exceptions.HTTPError.__str__()`
embeds the full request URL including its query string. `fetch_news` calls `with_retry`
with the default `exceptions=(Exception,)` (see PYQ-151), so any retryable failure —
including an ordinary transient 5xx, not just a bad key — writes `token=<the actual key>`
into the log stream at WARNING, every attempt, until it either succeeds or exhausts
retries. `FINNHUB_API_KEY` is a configured repository secret (per this project's own
CI setup), so this reaches CI logs too. This is a direct violation of CLAUDE.md's own
non-negotiable: *"Secrets never enter meta.json, runs.jsonl, logs, or cache fingerprints."*

Ask: move the token to a header (`headers={"X-Finnhub-Token": api_key}`, dropped from
`params`), and/or redact `token=...` from the message `with_retry` logs before it's written,
as a second line of defense for any other query-string secret that might be added later.

Acceptance criteria: the key is not present in `params`/the request URL; a test constructs
a failing request and asserts the logged output (at any level) never contains the
configured key; if header-auth isn't viable for some reason, `with_retry`'s log line is
proven to redact `token=`/`api_key=`-shaped query fragments.

Resolution: did both, as the ticket suggested. `fetch_news` now sends `headers=
{"X-Finnhub-Token": api_key}` and no longer puts `token` in `params` at all. As defense in
depth for any other query-string secret (present or future), `with_retry`'s WARNING log now
passes `_redact(str(exc))` instead of the raw exception -- a case-insensitive regex
replacing `token=...`/`api_key=...`-shaped fragments (up to the next `&` or whitespace) with
a redacted placeholder.

Covered by `test_fetch_news_sends_the_api_key_as_a_header_not_a_query_param` (asserts the
key is in `headers`, not `params`, and never appears in `str(params)`),
`test_with_retry_redacts_query_string_secrets_from_the_warning_log` (a retryable failure
whose message embeds a token-bearing URL; asserts the secret is absent and the redacted
placeholder is present in the captured log), and
`test_redact_handles_api_key_and_is_case_insensitive`.

---

## [PYQ-151]
`with_retry` retries non-retryable errors (401/404) by default
Status: Open
Priority: Low
Files: `pyquant/data/retry.py`

Problem: `with_retry`'s default is `exceptions: tuple[type[BaseException], ...] =
(Exception,)` (`retry.py:30`) — every call site (`fetch_prices`, `fetch_news`) uses this
default, so a definitively non-retryable failure (invalid API key → 401, unknown symbol →
404) still costs the full retry budget and every configured sleep, per call, per symbol,
before surfacing. No call site narrows `exceptions` to connection/timeout/5xx/429.

Ask: narrow the default (or each call site's override) to retryable failure modes —
connection errors, timeouts, 429, 5xx — and let 4xx other than 429 fail immediately. Add
jitter to the backoff while touching this, since the current fixed/exponential delay makes
concurrent callers (e.g. the API's threadpool) retry in lockstep.

Acceptance criteria: a test asserts a 401/404-raising callable is not retried (single
attempt, immediate raise); a separate test asserts a 5xx/timeout-raising callable still
retries per the existing backoff schedule.

---

## [PYQ-152]
`compute_rsi` returns 100, not 50, on a flat/halted series
Status: Resolved — 2026-07-29 (same session, uncommitted — see git status)
Priority: Medium
Files: `pyquant/data/prices.py` (`compute_rsi`)

Problem: `prices.py:90-93` —

```python
rsi = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
return rsi.mask((avg_loss == 0) & avg_gain.notna(), 100.0)
```

On a flat series, `avg_gain == avg_loss == 0.0`, so `avg_gain / avg_loss` is `0.0/0.0 =
NaN` and `rsi` is `NaN`. The mask condition `(avg_loss == 0) & avg_gain.notna()` is `True`
even when `avg_gain` is exactly `0.0` — `0.0` is not `NaN`, so `.notna()` passes — and the
mask overwrites with `100.0`. Verified directly: `compute_rsi(pd.Series([100.0]*30),
period=14).iloc[-1] == 100.0`. A halted or pegged instrument therefore reads as maximally
overbought instead of neutral. This is a distinct edge case from PYQ-121 (Resolved), which
fixed simple-MA vs. Wilder's smoothing and doesn't touch this 0/0 branch.

Ask: add `& (avg_gain > 0)` to the mask condition so the 100.0 override only fires on a
genuine "all gains, no losses" run, and emit `50.0` (neutral) for the degenerate
`avg_gain == avg_loss == 0` case.

Acceptance criteria: `compute_rsi` on a constant-price series returns 50.0 (not 100.0), with
a regression test alongside the existing Wilder's-smoothing tests from PYQ-121.

Resolution: split the mask into two — `(avg_loss == 0) & (avg_gain > 0)` still overrides to
`100.0` for a genuine all-gains run, and a new `(avg_loss == 0) & (avg_gain == 0) &
avg_gain.notna()` mask overrides the resulting `NaN` to `50.0` (neutral) for the flat/halted
case. Covered by `test_compute_rsi_is_50_on_a_flat_series`, alongside the existing
all-rises/all-falls tests from PYQ-121.

---

## [PYQ-153]
PIT values saturate at the outer quantile edges instead of extending into the tails
Status: Open
Priority: Low
Files: `pyquant/analysis/metrics.py` (`pit_values`)

Problem: `pit_values` (`metrics.py:152-159`) uses `np.interp(actual, sorted_predictions,
quantiles)` per point, which saturates outside the knot range by construction. With the
default three quantiles (`TFTConfig.quantiles = [0.1, 0.5, 0.9]`, `config.py:62`), every
actual below the predicted p10 maps to exactly `0.1` and every actual above p90 maps to
exactly `0.9` — point masses at the band edges rather than a spread into the tails. The PIT
histogram PYQ-252 added is documented as showing "U-shaped means overconfident," but with
this clamping the histogram is structurally incapable of showing a shape past those two
edge bins — it can only show *how many* points landed outside the band, not how far.

Ask: either extrapolate the tails under an explicit, documented parametric assumption (the
options review flags this needs justification, not just more interpolation), or state the
truncation directly next to wherever the PIT histogram is rendered/described so it isn't
read as a calibration verdict it can't actually support.

Acceptance criteria: either PIT values for out-of-band actuals are no longer clamped to
exactly 0.1/0.9 (with the extrapolation assumption stated in the docstring), or the CLI/docs
output next to the histogram explicitly notes the edge-clamping limitation.

---

## [PYQ-154]
`next_sessions` is not total for very large `max_prediction_length`
Status: Open
Priority: Low
Files: `pyquant/data/trading_calendar.py` (`next_sessions`), `pyquant/config.py` (`TrainingConfig.max_prediction_length`)

Problem: `next_sessions` (`trading_calendar.py:96-102`) over-fetches a **fixed** 15-business-
day margin (`pd.bdate_range(start, periods=count + 15)`) rather than looping until
satisfied. Verified directly: at `count=377` it already returns fewer than `count` dates
(376), and by `count=380` `extend_for_prediction` (`dataset.py:286`) raises `ValueError:
Length of values (379) does not match length of index (380)`. `TrainingConfig
.max_prediction_length` (`config.py:85`) has no upper-bound validator, so an
operator-supplied horizon in that range is accepted at config time and fails later, deep in
the pipeline, with a pandas-internal error rather than a clear one.

Ask: loop `next_sessions` until `len(result) >= count` rather than fetching a fixed margin,
so it is total for any `count`. Low practical severity — realistic horizons are single
digits to low dozens — but cheap to fix and removes a confusing failure mode for anyone
who fat-fingers the config.

Acceptance criteria: `next_sessions(start, count=400)` (or similar) returns exactly `count`
dates; a test pins this at a horizon past the current ~377 threshold.

---

## [PYQ-155]
Cache writes (pickle + meta) are two separate, non-atomic operations
Status: Resolved — 2026-07-29 (same session, uncommitted — see git status)
Priority: Medium
Files: `pyquant/data/cache.py` (`write_cache`, `write_pin`)

Problem: `write_cache` (`cache.py:71-76`) does `panel.to_pickle(path)` then
`_meta_path(path).write_text(json.dumps({"cached_at": ...}))` as two separate,
unlocked filesystem operations; `write_pin` (`cache.py:114-131`) has the identical
pattern. A crash mid-`to_pickle` (or between the two calls) can leave a truncated pickle
that a stale-but-present meta file will happily present as valid, and `pd.read_pickle` will
raise from inside `build_panel` on the next read. Two API worker threads (see PYQ-146's note
on the threadpool) building the same panel concurrently can interleave the two writes.

Ask: write to a temp path (`path.with_suffix(".tmp")`) and `os.replace()` into place for
both the pickle and its meta sidecar, so a reader never observes a partially-written file;
guard reads in a `try/except` that falls back to a refetch on a corrupt cache entry rather
than propagating a raw unpickling error.

Acceptance criteria: a test that kills a simulated write mid-way (or directly writes a
truncated file at the target path) and asserts a subsequent read either succeeds against the
last good state or triggers a clean refetch, never a raw exception from `build_panel`.

Resolution: `_atomic_write_text`/`_atomic_write_pickle` write to a uniquely-named sibling
`.tmp` file (`uuid4`-suffixed, so concurrent writers of the same key never collide on the
temp name itself) then `os.replace()` into place; `write_cache` and `write_pin` both use them
for the pickle and its meta sidecar. `read_cache` now wraps the meta-JSON parse and the
pickle read in `try/except`, logging a warning and returning `None` (a clean cache miss,
triggering `build_panel`'s existing refetch path) on either kind of corruption, rather than
propagating a raw exception.

One deliberate asymmetry: `read_pin` does **not** fall back silently the same way. A pin's
entire purpose is exact reproducibility, so silently refetching live (and possibly
different) data on a corrupt pin would defeat the guarantee it exists to make -- it now
raises a `ValueError` naming the pin instead, which is still a clean, expected error
(`ValueError` is in `cli/app.py`'s `EXPECTED_FAILURES`) rather than a bare unpickling
traceback.

Covered by `test_write_cache_leaves_no_tmp_file_behind`,
`test_write_pin_leaves_no_tmp_file_behind`,
`test_read_cache_treats_a_truncated_pickle_as_a_miss_not_a_crash`,
`test_read_cache_treats_corrupt_meta_json_as_a_miss_not_a_crash`, and
`test_read_pin_raises_a_clear_error_on_a_truncated_pickle`.

---

## [PYQ-156]
`cli/app.py` hides a metric row at exactly 0.0; `aggregate_metrics([])` raises `ZeroDivisionError`
Status: Resolved — 2026-07-29 (same session, uncommitted — see git status)
Priority: Low
Files: `pyquant/cli/app.py`, `pyquant/analysis/metrics.py` (`aggregate_metrics`)

Problem: two small robustness gaps in the same reporting path.

`cli/app.py:171,173` — `if ev.crps:` / `if ev.winkler_score:` are bare truthy checks, not
`is not None`. `EvaluationMetrics.crps`/`winkler_score` (`metrics.py:217-218`) are plain
`float` (not `Optional`), always populated by `evaluate_predictions`. A legitimate value of
exactly `0.0` (e.g. a degenerate/perfect band) silently disappears from the printed summary
table with no indication a row was suppressed.

`metrics.py:307-312` (`aggregate_metrics`) — with `results=[]`, `weights =
np.ones(len(results))` is `np.ones(0)`, and `np.average([], weights=[])` raises
`ZeroDivisionError: Weights sum to zero, can't be normalized` (verified directly).
Reachable: `cli/app.py:279`'s `--windows` option has no minimum-value validator, so
`pyquant backtest SYMBOL --windows 0` reaches `walk_forward_backtest(n_windows=0)`, produces
an empty `cutoffs` list, and calls `aggregate_metrics([])` directly. `ZeroDivisionError` is
not a `ValueError` and isn't in `cli/app.py:39`'s `EXPECTED_FAILURES` tuple, so the CLI
surfaces a raw Python traceback instead of a clean error message.

Ask: `is not None` (or drop the guards — both fields are always populated now, per the
comment already true elsewhere in this file) for the truthiness checks; a guard in
`aggregate_metrics` (or a `--windows` minimum of 1 at the CLI boundary) for the empty case.

Acceptance criteria: a metrics row with `crps == 0.0` or `winkler_score == 0.0` still
prints; `pyquant backtest SYMBOL --windows 0` produces a clean, expected error rather than a
traceback, with a regression test.

Resolution: both truthy checks are now `is not None`. `aggregate_metrics()` raises a clear
`ValueError` on an empty list instead of letting `np.average` raise `ZeroDivisionError` --
`ValueError` is already in `cli/app.py`'s `EXPECTED_FAILURES`, so `--windows 0` now renders
as a clean CLI error rather than a traceback, with no separate `--windows` minimum needed.
Covered by `test_train_table_still_shows_crps_and_winkler_rows_at_exactly_zero`,
`test_aggregate_metrics_raises_a_clear_error_on_an_empty_list`, and
`test_backtest_with_zero_windows_reports_cleanly`.

---

## [PYQ-157]
`Settings` swallows misspelled nested env keys; `TFTConfig.quantiles` allows a bandless config
Status: Resolved — 2026-07-29 (same session, uncommitted — see git status)
Priority: Medium
Files: `pyquant/config.py` (`Settings`, `TFTConfig`), `pyquant/models/tft.py` (`_window_signal`, `permutation_importance`)

Problem: two related validation gaps in `config.py`.

`Settings.model_config` (`config.py:172-177`) sets `extra="ignore"`; the nested
`TFTConfig`/`TrainingConfig`/`DataConfig` `BaseModel`s have no `model_config` override, so
they inherit pydantic v2's equivalent default. Verified directly: `TRAINING__MAX_EPOCH=999`
(missing the trailing "S") leaves `settings.training.max_epochs` at its default with no
error or warning, while the correctly-spelled `TRAINING__MAX_EPOCHS=999` works. This is the
same failure shape PYQ-128 (Resolved) fixed for a missing `--config` *file* — silent
divergence between what an operator intended and what actually ran — just one level down,
at the *key* rather than the file.

Separately, `TFTConfig.quantiles`'s validator (`config.py:64-78`) only checks the list is
sorted ascending — nothing requires `0.5 in quantiles` or `0 < q < 1`. Four call sites do
`quantiles.index(0.5)`: `metrics.py:253-260` (`evaluate_predictions`) raises a clear,
named `ValueError` first; `tft.py:489` (`_window_signal`) and `tft.py:985`
(`permutation_importance`) do not, and raise a bare `ValueError: 0.5 is not in list` from
deep in the stack with no indication which setting caused it.

Ask: consider `extra="forbid"` on the nested config models (at minimum) so a misspelled key
fails at startup the way a missing config file already does. Validate `0.5 in quantiles` and
`0 < q < 1` once, in `TFTConfig`'s validator, so the error names the actual setting instead
of surfacing from three different unguarded call sites.

Acceptance criteria: a misspelled nested env key raises at `Settings()` construction rather
than silently defaulting; `TFTConfig(quantiles=[0.1, 0.9])` (no 0.5) raises a validation
error naming `quantiles`, with a test; `_window_signal`/`permutation_importance` no longer
need their own guard because the invalid state is unreachable by construction.

Resolution: `TFTConfig`, `TrainingConfig` and `DataConfig` all gained
`model_config = ConfigDict(extra="forbid")` -- a misspelled nested env key (or an unknown
kwarg passed directly) now raises `pydantic.ValidationError` at construction instead of
silently keeping the default. `TFTConfig.quantiles` gained a second validator (alongside
the existing ascending-order one) requiring every entry satisfy `0 < q < 1` and that `0.5`
is present, naming `quantiles` in the error either way. `_window_signal`/
`permutation_importance` needed no code change -- their bare `quantiles.index(0.5)` calls
are now unreachable with an invalid config by construction, exactly as the acceptance
criteria describes.

Confirmed test-first: `test_tft_config_rejects_an_unknown_field`,
`test_training_config_rejects_an_unknown_field`, `test_data_config_rejects_an_unknown_field`
and `test_misspelled_nested_env_key_raises_instead_of_silently_defaulting` all fail against
the pre-fix code (verified by stashing the fix) and pass after. Also covered by
`test_tft_quantiles_reject_missing_median`,
`test_tft_quantiles_reject_out_of_range_values`, and
`test_correctly_spelled_nested_env_key_still_works` (the positive case, so `extra="forbid"`
isn't rejecting valid input). Both shipped example configs
(`configs/aapl_baseline.yaml`, `configs/wide_quantile_aggressive.yaml`) verified to still
load cleanly; full suite (including `test_tft.py`/`test_invariants.py`/
`test_learnability.py`) re-run to confirm `extra="forbid"` breaks no existing construction
site.

---

## [PYQ-158]
`tests/conftest.py`'s `settings` fixture leaves `use_options` pointed at the real repo
Status: Resolved — 2026-07-29 (same session, uncommitted — see git status)
Priority: Medium
Files: `tests/conftest.py`

Problem: the `settings` fixture disables `use_macro`/`use_sectors`/`use_sentiment` and
redirects `cache_dir` to `tmp_path`, with a comment stating the intent is *"so tests never
read/write the real project directory"* — but leaves `use_options=True` (the `DataConfig`
default, `config.py:140`) and `options_history_dir` pointed at the real
`data/options_history/` directory (`config.py:199`'s default). `build_panel`
(`dataset.py:147-150`) reads `load_snapshot_history(symbol, settings)` from that path
whenever `use_options` is true, so any test using this fixture reads real accumulated
snapshot data if `pyquant snapshot` has been run locally — a developer who has run it 20+
times gets measurably different test behavior than CI, which never has. (The leak is
read-only: `append_snapshot` is only invoked by the `pyquant snapshot` CLI command, never by
`build_panel`, so tests can't contaminate the real directory, only be silently influenced by
it.)

Ask: add `s.data.use_options = False` and `s.options_history_dir = tmp_path / "options"` to
the fixture, matching the treatment already given to macro/sectors/sentiment/cache.

Acceptance criteria: the `settings` fixture disables `use_options` and redirects
`options_history_dir` into `tmp_path`; a test (or an assertion added to an existing one)
confirms panel-building under the fixture never touches the real `data/options_history/`
directory.

Resolution: added `s.data.use_options = False` and `s.options_history_dir = tmp_path /
"options"` to the fixture, matching the existing macro/sectors/sentiment/cache treatment.
Covered by `test_settings_fixture_isolates_options_history_from_the_real_repo` (asserts the
fixture's own defaults) and
`test_panel_building_with_use_options_reenabled_still_reads_from_tmp_path` (a test that
deliberately opts back into `use_options=True`, as `test_dataset.py`'s existing option tests
already do, still resolves under `tmp_path` rather than the real directory).

---

## [PYQ-159]
`TrainRequest.period` is dead; `JobRegistry` never evicts and indexes `_jobs` unguarded
Status: Resolved — 2026-07-29
Priority: Low
Files: `pyquant/api/schemas.py`, `pyquant/api/jobs.py`

Problem: two small API hygiene gaps.

`TrainRequest.period` (`api/schemas.py:101`) is declared, appears in the generated OpenAPI
schema, and is never read — `_run_train_job` (`api/routes/train.py:25-44`) only consumes
`request.symbols`/`bundle_name`/`epochs`. Grepping `period` across `pyquant/api/` returns
only the declaration. A documented request field that silently does nothing.

`JobRegistry` (`api/jobs.py`) appends to `self._jobs` on every `create()` (`jobs.py:41-45`)
with no eviction, TTL, or size cap anywhere in the file — unlike `BundleCache` in
`api/deps.py`, which has explicit LRU eviction. `mark_running`/`mark_succeeded`/`mark_failed`
(`jobs.py:56-73`) all index `self._jobs[job_id]` directly with no existence check, so a call
with an unregistered `job_id` raises `KeyError`. Under the current design `create()` always
runs synchronously before the job is scheduled, so this `KeyError` path isn't demonstrated
as reachable today — a latent robustness gap rather than a live bug, worth guarding before
the job lifecycle grows more entry points (e.g. cancellation, PYQ-261's noted follow-up).
(Note: `POST /train` sharing Starlette's threadpool with other routes for the duration of a
fit is a separate, already-documented and deliberate v1 tradeoff — see `docs/api-design.md`
and PYQ-261's resolution note — not a new finding here.)

Ask: either thread `period` into `settings.data.period` or delete the field. Add a bound
(size cap and/or TTL-based eviction) to `JobRegistry`, and use `.get(job_id)` with a clear
error (or a no-op with a logged warning) instead of unguarded `__getitem__` in the three
`mark_*` methods.

Acceptance criteria: `period` either does something or is removed from the schema; a test
asserts `JobRegistry` bounds its size under sustained job creation; a test asserts calling
`mark_running`/`mark_succeeded`/`mark_failed` with an unknown `job_id` fails predictably
rather than raising a bare `KeyError` from inside a background task.

Resolution: `TrainRequest.period` now does something — `routes/train.py`'s `start_train`
sets `settings.data.period = request.period` when given, before scheduling the background
fit, the same `if period:` pattern `cli/app.py::_build_settings` already uses for the CLI
flag. `BacktestRequest.period` (new, PYQ-271) gets the identical treatment in
`routes/backtest.py`, so the API doesn't reintroduce the gap for its second job endpoint.

`JobRegistry` (`api/jobs.py`) now takes a `max_jobs` constructor argument (default 500)
and an insertion-order list alongside its `dict`; every `create()`/`try_start_train()` call
evicts the oldest record once the count exceeds `max_jobs`, mirroring `BundleCache`'s LRU
shape. `mark_running`/`mark_succeeded`/`mark_failed` all now do `self._jobs.get(job_id)`
and log a warning + return on a miss instead of an unguarded `self._jobs[job_id]`.

Verification: `test_job_registry_bounds_its_size_under_sustained_job_creation` (`max_jobs=3`,
5 `create()` calls, asserts the oldest two are gone and the newest three remain retrievable)
and `test_job_registry_mark_methods_are_a_no_op_for_an_unknown_job_id` (all three `mark_*`
calls against a nonexistent id, asserts none raises) in `tests/test_api.py`.
`test_train_request_period_overrides_settings_data_period` and
`test_backtest_request_period_overrides_settings_data_period` monkeypatch the underlying
`tft.train`/`tft.walk_forward_backtest` call to capture the `Settings` object it actually
received and assert `settings.data.period` matches the request body. `ruff check .`,
`pytest -q` and `scripts/backlog.py check` all clean.

---

## [PYQ-160]
`/docs`, `/redoc`, `/openapi.json` are reachable with no `X-API-Key`, contradicting the documented auth contract
Status: Resolved — 2026-07-29
Priority: Medium
Files: `pyquant/api/app.py`, `docs/http-api.md`

Problem: `docs/http-api.md`'s Authentication section states "Every endpoint except
`/healthz` requires an `X-API-Key` header" and that an unconfigured service "fails loudly
with `500`, it does not fall open." Both claims describe `require_api_key`
(`api/deps.py:107`), which is wired only via `dependencies=[Depends(require_api_key)]` on
the `forecast`, `explain` and `train` `APIRouter`s (`routes/forecast.py:21`,
`routes/explain.py:21`, `routes/train.py:21`). FastAPI's own `/docs`, `/redoc` and
`/openapi.json` routes are generated by the `FastAPI(...)` constructor (`app.py:13-17`)
independently of any router, and `app.py` never passes a `dependencies=` argument to that
constructor — so these three routes never run `require_api_key` at all.

Verified live: with `PYQUANT_API_KEYS` exported and enforced (confirmed by `GET
/forecast/AAPL` and `GET /train/does-not-exist` both returning `401` with no key), `GET
/openapi.json` and `GET /docs` both returned `200` with no `X-API-Key` header. Since the
exemption is structural (these routes are outside every gated router), the same holds
whether or not `PYQUANT_API_KEYS` is configured at all — the "fails loudly, does not fall
open" contract never engages for them.

Impact: anyone who can reach the host learns the full endpoint list, every request/response
field name and type (`schemas.py`'s complete pydantic models) and validation patterns
(e.g. `SYMBOL_PATTERN`) with no key — smaller than PYQ-150's leaked upstream key, but a real
gap between the stated contract and the shipped behaviour.

Ask: decide this deliberately rather than leave it implicit — either (a) gate
`/docs`/`/redoc`/`/openapi.json` behind `require_api_key` too (`docs_url=None` etc. on the
`FastAPI(...)` constructor plus hand-mounted replacement routes carrying the dependency), or
(b) keep them public and correct `docs/http-api.md`'s "every endpoint except `/healthz`"
claim to name these three as additional, deliberate exceptions with a one-line reason.

Acceptance criteria: a test in `tests/test_api.py` asserts the chosen behaviour for
`/openapi.json` with no `X-API-Key` header — either `401`, or an explicit `200` with a
comment pointing at this ticket so the exemption reads as deliberate. `docs/http-api.md`'s
Authentication section matches whatever the test asserts.

Resolution: chose (a) — gate all three behind `require_api_key`, matching the documented
"every endpoint except `/healthz`" contract rather than carving out three more exceptions
to it. `app.py` now passes `docs_url=None, redoc_url=None, openapi_url=None` to the
`FastAPI(...)` constructor (disabling the unguarded defaults) and hand-mounts
`GET /openapi.json` (via `fastapi.openapi.utils.get_openapi`), `GET /docs` (via
`fastapi.openapi.docs.get_swagger_ui_html`) and `GET /redoc` (via `get_redoc_html`), each
carrying `dependencies=[Depends(require_api_key)]` — the identical dependency the other
routers use, imported directly from `deps.py` rather than reimplemented.

Verification: `test_openapi_json_requires_auth` and `test_docs_and_redoc_require_auth` in
`tests/test_api.py` assert `401` with no key and `200` with a correct one for all three
routes. `docs/http-api.md`'s Authentication section and Endpoints table now state the gate
applies to these three as fact, not as an open question. `ruff check .`, `pytest -q` and
`scripts/backlog.py check` all clean.

---

## [PYQ-161]
No lock guards concurrent `POST /train` calls for the same bundle name
Status: Resolved — 2026-07-29
Priority: Medium
Files: `pyquant/api/routes/train.py`, `pyquant/api/deps.py`, `pyquant/models/tft.py`

Problem: `GET /forecast/{symbol}` and `GET /explain/{symbol}` both serialize against a
per-bundle `threading.Lock` from `get_prediction_lock()` before calling into the model
(`routes/forecast.py:33`, `routes/explain.py:37`), because `docs/api-design.md` #4 documents
that concurrent calls into the same loaded model are unsafe. `POST /train` has no equivalent
guard: `_run_train_job` (`routes/train.py:25-48`) calls `tft.train(...)` directly, and
nothing in `deps.py` or `jobs.py` stops two jobs from running for the same `bundle_name`/
`symbols` at once.

Two `POST /train {"symbols": ["AAPL"]}` requests issued back to back both return `202`
immediately — each gets its own `job_id` from `JobRegistry.create()` — and are then
scheduled as independent `BackgroundTasks`, which Starlette runs concurrently on its
threadpool. Both calls resolve `_bundle_dir(settings, "AAPL")` (`tft.py:205-221`) to the
*same* directory, and both then: `bundle_dir.mkdir(parents=True, exist_ok=True)`
(`tft.py:461-462`), point a Lightning `ModelCheckpoint` at that same `dirpath`
(`tft.py:465-466`), `torch.save(..., bundle_dir / "dataset_params.pt")` (`tft.py:493`), and
write `meta.json` (`tft.py:562`) / append `runs.jsonl` (`tft.py:565`). Whichever job finishes
last silently wins — the other's `job_id` still resolves to `"succeeded"` with a
`TrainResult` describing a run whose on-disk files may since have been partially overwritten
by the other job, and `bundle_cache.invalidate()` fires from both, so a concurrent
`/forecast` mid-write can also hit a half-written `dataset_params.pt`/`model.ckpt` pair.

This needs two real requests close together rather than a race inside one request, so no
existing test exercises it (`test_train_returns_202_and_a_pollable_job_id` uses one job
against one symbol).

Ask: a per-bundle-name lock or an explicit "already training" rejection, mirroring the
per-bundle prediction lock's precedent. Cheapest fix: before scheduling `_run_train_job`,
check (and hold) a per-bundle-name flag — in `JobRegistry` or a small registry beside it —
and reject a second `POST /train` for a bundle already `queued`/`running` with a `409`,
rather than letting two fits race on the same directory.

Acceptance criteria: a test starts two `POST /train` requests for the same `symbols`/
`bundle_name` in quick succession and asserts the second is rejected (`409`) or otherwise
cannot corrupt the first's on-disk bundle — mirroring how
`test_forecast_serializes_concurrent_requests_against_the_same_bundle` proves the prediction
lock. `docs/http-api.md` documents the resulting behaviour under Training.

Resolution: chose the explicit-rejection half of the "Ask" — a per-bundle-name flag in
`JobRegistry` (`api/jobs.py`), not a separate registry. `JobRegistry.try_start_train
(bundle_name)` atomically checks-and-registers under the registry's existing lock: if
`bundle_name` is already a key in a new `_active_bundle_names: dict[str, str]` (bundle_name
-> job_id), it returns `None` and `routes/train.py::start_train` turns that into `409`
before `background_tasks.add_task` is ever called — the second fit is never scheduled, so
it never reaches `_bundle_dir`/`mkdir`/`torch.save` at all, which is stronger than letting
both run and rejecting after the fact. `mark_succeeded`/`mark_failed` release the guard
under the same lock, so a bundle becomes available again the moment its job resolves either
way. `start_train` computes the candidate `bundle_name` as `(request.bundle_name or
"_".join(request.symbols)).upper()` — the same default expression `tft.train()` itself
uses — purely so the lock key matches what `tft.train()` will actually resolve to; that
computation is duplicated, not shared, and `tft.train()` still computes its own copy.

Verification: `test_train_rejects_a_second_request_for_a_bundle_already_in_flight` uses two
real `threading.Thread`s (the same pattern as
`test_forecast_serializes_concurrent_requests_against_the_same_bundle`) against a
monkeypatched `tft.train` that sleeps 0.1s, staggered 0.02s apart, and asserts the pair of
status codes is exactly `[202, 409]` — proving rejection happens for a request that is
genuinely concurrent with an in-flight job, not merely sequential.
`test_train_conflict_names_the_bundle_in_the_error` asserts the `409` body names the actual
bundle. `docs/http-api.md`'s Training section now shows the `409` response and states the
bundle-name default explicitly. `ruff check .`, `pytest -q` and `scripts/backlog.py check`
all clean.

---

## [PYQ-162]
`POST /train`'s documented `failed` job status has zero test coverage
Status: Resolved — 2026-07-29
Priority: Low
Files: `tests/test_api.py`, `pyquant/api/routes/train.py`, `pyquant/api/jobs.py`

Problem: `docs/http-api.md`'s Training section states "Job status moves through `queued` →
`running` → `succeeded` or `failed`; `result` is populated on success and `error` on
failure." `_run_train_job` (`routes/train.py:25-48`) implements the `failed` branch —
`tft.train()` raising is caught, logged, and turned into `registry.mark_failed(job_id,
str(exc))` (`train.py:41-44`) — but grepping `tests/test_api.py` for `failed` returns
nothing. Every existing job test (`test_train_returns_202_and_a_pollable_job_id`,
`test_train_job_404_for_an_unknown_id`, `test_train_rejects_an_empty_symbol_list`, the two
path-traversal tests) only exercises the success path or input validation; none makes
`tft.train()` raise and then asserts `GET /train/{job_id}` actually returns `{"status":
"failed", "error": "..."}`.

This is more than an ordinary coverage gap because PYQ-159 (open, same files) already flags
that `JobRegistry.mark_failed` indexes `self._jobs[job_id]` unguarded — exactly the code
this path exercises. The failure branch is both untested and adjacent to known-fragile code,
so a regression (wrong status string, `error` not populated, an exception inside the
`except` handler itself) could land unnoticed.

Ask: one test that monkeypatches `pyquant.api.routes.train.tft.train` to raise, posts to
`/train`, polls `/train/{job_id}`, and asserts `status == "failed"` and `error` contains the
exception message — the same shape as the existing success-path test, minus the happy path.

Acceptance criteria: the new test fails against a deliberately broken `mark_failed`/`except`
branch (confirmed test-first, per this repo's convention) and passes against the current
code; `docs/http-api.md`'s `failed` claim is now backed by a test rather than only by
reading the source.

Resolution: added `test_train_job_reports_failed_status_and_error`, exactly the shape the
Ask describes — `monkeypatch.setattr("pyquant.api.routes.train.tft.train", raise_error)`,
`POST /train`, poll `GET /train/{job_id}`, assert `status == "failed"`, `result is None`,
and the exception message is in `error`. Confirmed test-first: reverting the `except`
branch in `_run_train_job` to re-raise instead of calling `registry.mark_failed` makes this
test fail with an unhandled `RuntimeError` inside the background task rather than a clean
assertion failure, so it does exercise the branch it claims to. No production code changed
— the `failed` path itself was already correct, only untested, exactly as filed. Added
`_run_backtest_job`'s equivalent `test_backtest_job_reports_failed_status_and_error`
alongside it for the new PYQ-271 endpoint, so the new job type doesn't reintroduce the same
coverage gap on day one.

Verification: both tests pass; `ruff check .`, `pytest -q` and `scripts/backlog.py check`
all clean.

---

## [PYQ-163]
`POST /train`/`POST /backtest` background jobs share the request-serving thread pool with every other endpoint
Status: Open
Priority: High
Files: `pyquant/api/jobs.py`, `pyquant/api/routes/train.py`, `pyquant/api/routes/backtest.py`, `docs/api-design.md`, `docs/http-api.md`

Problem: found by an external production-readiness review of the API (2026-07-30),
independently verified against the installed Starlette/anyio source rather than taken on
the review's word. `docs/api-design.md` #2 already documents that "a request thread must
not block on" a full Lightning fit and that's why `POST /train`/`POST /backtest` run via
FastAPI's `BackgroundTasks`, but it does not name what pool that background work actually
runs on.

Verified: `BackgroundTask.__call__` (`starlette/background.py`), for a sync function, does
`await run_in_threadpool(self.func, ...)`. FastAPI's own sync-`def`-route dispatch
(`fastapi/routing.py`, `run_endpoint_function`) does the identical
`await run_in_threadpool(dependant.call, **values)`. Both go through
`starlette/concurrency.py::run_in_threadpool`, which calls
`anyio.to_thread.run_sync(func)` with no `limiter=` argument — i.e. both paths draw from
`anyio`'s single process-wide default `CapacityLimiter` (`AsyncIOBackend
.current_default_thread_limiter`, 40 slots, lazily created). There is no separate pool for
background jobs; a `POST /train` background task and a concurrent `GET /forecast` request
compete for the same 40 slots, and a training job holds its slot for minutes rather than
milliseconds. `/healthz` was one of the sync handlers sharing this pool — bugs.md#pyq-165
made it `async def` this same pass, which gets the liveness probe itself off the shared
pool entirely, but does not address the pool sharing for `/forecast`/`/explain`/`/scan`,
which still compete with background jobs for the same 40 slots.

At the default limiter size (40) this needs sustained concurrent training/backtest load to
actually exhaust every slot, not "two or three fits" as the review's own phrasing claimed —
that specific number was not reproduced and is not asserted here. The structural problem is
real regardless of the exact threshold: an unbounded number of concurrent `POST
/train`/`POST /backtest` calls (schemas.py's new bounds cap the *cost* of one job, not how
many can run at once) each consume a slot for the job's full duration, and nothing
currently limits how many can be in flight together.

Ask: a dedicated executor for background jobs, so they stop drawing from the same pool
request handling needs. Smallest change with no new dependency: bypass
`BackgroundTasks.add_task` for `/train`/`/backtest` and submit to a module-level
`concurrent.futures.ThreadPoolExecutor(max_workers=N)` directly (`loop.run_in_executor
(dedicated_executor, func, ...)` from an `async def` route), keeping the job registry and
locking exactly as-is. Longer-term, `docs/api-design.md` already names the real fix this
merely delays: a process/queue boundary (arq/Celery + Redis, or a `ProcessPoolExecutor` so
`torch`'s own CPU usage is isolated from the request-serving process) — that is a new
dependency and needs its own justification per this repo's non-negotiable #5, not a
decision to make inside this ticket.

Acceptance criteria: background jobs run on a pool separate from request-handling threads,
verified by a test that starts a slow (mocked) training job and asserts a concurrent
`GET /healthz` (or another cheap endpoint) responds promptly rather than queuing behind it;
`docs/api-design.md` #2 and `docs/http-api.md`'s Concurrency model section state which pool
background jobs run on, not just that they're backgrounded.

---

## [PYQ-164]
Per-bundle-name lock dicts (`_PredictionLocks`, `BundleCache._load_locks`) never evict
Status: Open
Priority: Low
Files: `pyquant/api/deps.py`

Problem: found by the same external review (2026-07-30) that produced bugs.md#pyq-163,
re-verified directly against the current source. `_PredictionLocks._locks`
(`api/deps.py`) grows by one entry per distinct bundle name ever requested and never
shrinks. This pass's own `BundleCache._load_locks` (added to fix the cold-load stampede
this same review flagged) has the identical shape for the same reason.

Both dicts are keyed by input that has already passed `SYMBOL_PATTERN`
(`^[A-Za-z0-9][A-Za-z0-9.\-]{0,15}$`, api/schemas.py) — a large but finite space, not an
arbitrary-string injection point — so this is bounded, slow memory growth under a
long-running process fielding many distinct symbol names, not an unbounded DoS from one
request. Lower severity than bugs.md#pyq-163 for that reason, but real: a server that has
ever been asked about thousands of distinct (even nonexistent) symbols retains a lock
object for every one of them forever.

Ask: bound both dicts. The tempting fix — evict an entry once its lock is released — is
wrong: a lock object referenced by a thread currently waiting on `.acquire()` must not be
replaced out from under it. Safer shapes: an LRU cap mirroring `BundleCache`'s own
(evict the least-recently-used lock *only when unheld*, checking `lock.locked()` before
evicting under the registry lock), or accept the bound growth and cap it with a hard
ceiling past which the oldest unheld lock is dropped and a warning logged.

Acceptance criteria: a test creates locks for more than some cap's worth of distinct names
and asserts the registries stay bounded; the fix does not introduce a use-after-evict race
(a lock currently held or awaited is never evicted out from under its holder/waiter).

---

## [PYQ-165]
`/healthz` was a sync `def`, sharing the request/background-job thread pool its own liveness check needs to bypass
Status: Resolved — 2026-07-30
Priority: Medium
Files: `pyquant/api/routes/health.py`

Problem: found and verified alongside bugs.md#pyq-163 (same external review, same source
audit). `/healthz` was `def healthz()`, so FastAPI dispatched it through
`run_in_threadpool` — the identical shared 40-slot anyio limiter a `POST /train`
background job also draws from (see PYQ-163 for the verified mechanism). A liveness probe
is exactly the request an orchestrator (k8s, Render, etc.) needs answered fastest and most
reliably; queuing it behind a training job's thread is the failure mode that gets a healthy
container killed for a reason unrelated to its actual health.

Ask: make it `async def`. The handler does no I/O (just returns a static
`HealthResponse()`), so nothing is lost — an async handler with no `await` inside still
runs entirely on the event loop, never touching the threadpool at all.

Acceptance criteria: `healthz` is `async def`; a test asserts
`inspect.iscoroutinefunction(healthz)` as a regression guard against it being changed back.

Resolution: exactly the Ask. `routes/health.py::healthz` is now `async def`, with a
docstring explaining why (citing the same verified `run_in_threadpool`/anyio-limiter
mechanism as PYQ-163). This is a *partial* mitigation of PYQ-163's underlying problem — it
gets the liveness probe itself permanently off the shared pool, but `/forecast`,
`/explain`, `/scan` and the other sync routes still share it with background jobs, which is
what PYQ-163 (left Open) tracks.

Verification: `test_healthz_handler_is_async` in `tests/test_api.py` (an `inspect
.iscoroutinefunction` assertion — a direct, load-free regression guard, not a timing-based
proof, since reliably demonstrating threadpool starvation in a unit test would need
saturating all 40 default slots). `ruff check .`, `pytest -q` and `scripts/backlog.py
check` all clean.

---

## [PYQ-166]
`POST /scan`/`POST /train`/`POST /backtest` request fields (`symbols`, `epochs`, `windows`, `period`) had no upper bound
Status: Resolved — 2026-07-30
Priority: High
Files: `pyquant/api/schemas.py`

Problem: found by the same external review as bugs.md#pyq-163, verified directly against
`api/schemas.py`. `ScanRequest.symbols`/`TrainRequest.symbols` were plain `list[str]` with
no `max_length`; `TrainRequest.epochs`/`BacktestRequest.epochs` and
`BacktestRequest.windows` were plain `int`/`int | None` with no bound; `TrainRequest
.period`/`BacktestRequest.period` were plain `str | None`, forwarded unchecked into
`yfinance.history(period=...)` (`data/macro.py`, `data/providers.py`) via `settings.data
.period = request.period` — `DataConfig` doesn't set `validate_assignment=True`, so nothing
re-validates the mutated value, and `period` has no pattern.

Impact: a single authenticated request could demand unbounded vendor-quota spend
(`POST /scan` with hundreds of symbols, each a ~65s cold forecast per
investigations.md#pyq-319) or unbounded training compute (`{"windows": 10000}` on
`POST /backtest`). A bad `period` value fails deep inside the yfinance fetch layer with an
unhelpful vendor-side error rather than a clean `422` at the request boundary.

Ask: bound every caller-controlled numeric/list field; restrict `period` to yfinance's
actual accepted vocabulary.

Acceptance criteria: `ScanRequest.symbols`/`TrainRequest.symbols` capped; `epochs`/`windows`
bounded to a sane positive range; `period` restricted to a known-valid pattern; each bound
covered by a test asserting the boundary rejects with `422`.

Resolution: four new module-level constants in `schemas.py`
(`MAX_SYMBOLS_PER_REQUEST=50`, `MAX_EPOCHS=500`, `MAX_BACKTEST_WINDOWS=50`,
`YFINANCE_PERIOD_PATTERN` — the exact set yfinance accepts: `1d/5d/1mo/3mo/6mo/1y/2y/5y
/10y/ytd/max`), applied via pydantic `Field(max_length=...)`/`Field(gt=0, le=...)`
/`Field(pattern=...)` on `ScanRequest.symbols`, `TrainRequest.symbols`/`epochs`/`period`,
and `BacktestRequest.windows`/`epochs`/`period`. Bounds chosen generously (50 symbols, 500
epochs, 50 windows) to not constrain legitimate use, not tuned to a specific measured
attack cost.

Verification: `test_scan_rejects_more_symbols_than_the_cap`,
`test_train_rejects_more_symbols_than_the_cap`, `test_train_rejects_an_out_of_range_epochs`,
`test_train_rejects_an_invalid_period`, `test_backtest_rejects_an_out_of_range_windows`,
`test_backtest_rejects_an_invalid_period` in `tests/test_api.py`, each asserting `422` at
the boundary. `docs/http-api.md` gained a "Request limits" table under Status codes.
`ruff check .`, `pytest -q` and `scripts/backlog.py check` all clean.

---

## [PYQ-167]
Per-bundle prediction lock blocked indefinitely; a slow request could hold every subsequent caller hostage
Status: Resolved — 2026-07-30
Priority: Medium
Files: `pyquant/api/deps.py`, `pyquant/api/routes/forecast.py`, `pyquant/api/routes/explain.py`

Problem: found by the same external review as bugs.md#pyq-163. `_get_forecast`
(`routes/forecast.py`) and `get_explanation` (`routes/explain.py`) both did
`with get_prediction_lock(symbol): ...` — a bare `threading.Lock()` context manager, which
blocks forever if the lock is held. Combined with bugs.md#pyq-163's shared threadpool, a
slow or hung `predict()` call for one bundle would hold every subsequent request for that
same bundle queuing on its request thread indefinitely, with no way for a caller to know it
was queued rather than just slow.

Ask: a bounded wait with a clean error past the bound, mirroring how `require_api_key`
fails loudly rather than hanging.

Acceptance criteria: a caller queuing behind a busy bundle for longer than some timeout
gets a clean error response (`429`, since this is a retry-appropriate "busy" signal, not a
client mistake) instead of an indefinite hang; covered by a test that holds the lock and
asserts the timeout path.

Resolution: `deps.acquire_prediction_lock(name, timeout=None)`, a `contextmanager` wrapping
`lock.acquire(timeout=...)` and raising `HTTPException(429, ...)` on failure to acquire
within `PREDICTION_LOCK_TIMEOUT_SECONDS` (30s, a module constant, deliberately read inside
the function body rather than bound into the signature's default value so it stays
`monkeypatch`-able for tests). Replaces the bare `with get_prediction_lock(symbol):` in
both `routes/forecast.py` and `routes/explain.py`; `get_prediction_lock` itself is
unchanged and still used internally.

Verification: `test_acquire_prediction_lock_returns_429_when_the_bundle_is_busy` (holds the
lock, asserts a 0.05s-timeout acquire raises `HTTPException(429)`),
`test_acquire_prediction_lock_releases_on_success`, and
`test_forecast_returns_429_when_the_bundle_is_busy` (end-to-end through
`GET /forecast/{symbol}`, `monkeypatch`ing the timeout constant to 0.05s so the test doesn't
take 30s) in `tests/test_api.py`. The existing
`test_forecast_serializes_concurrent_requests_against_the_same_bundle` (0.05s hold, default
30s timeout) still passes unmodified — the timeout is generous enough not to affect normal
serialization. `ruff check .`, `pytest -q` and `scripts/backlog.py check` all clean.

---

## [PYQ-168]
`BundleCache.get()` re-deserializes the same checkpoint N times under concurrent cold-start requests
Status: Resolved — 2026-07-30
Priority: Medium
Files: `pyquant/api/deps.py`

Problem: found by the same external review as bugs.md#pyq-163. `BundleCache.get()`
deliberately calls `tft.load()` *outside* its own `_lock` (a documented, correct choice —
so loading one bundle doesn't block requests for already-cached ones), but that meant N
simultaneous first-requests for the *same* not-yet-cached bundle name each independently
missed the cache, each called `tft.load()`, and each redundantly paid the full checkpoint
deserialization cost before whichever finished last won the cache slot. Not a correctness
bug (every load produces an equivalent bundle), but wasted I/O/CPU exactly during the
highest-latency case this project has measured (investigations.md#pyq-319: a cold path is
dominated by vendor fetch, and checkpoint deserialization adds real milliseconds on top).

Ask: dedupe concurrent misses for the same name without serializing misses for
*different* names (the property the outside-the-lock design was built to preserve).

Acceptance criteria: a test with concurrent `get()` calls for the same name asserts the
underlying load function was called exactly once; a test with concurrent `get()` calls for
*different* names asserts they still overlap in time (proving the fix didn't accidentally
serialize the whole cache).

Resolution: double-checked locking, per-bundle-name. A new `_load_locks: dict[str,
threading.Lock]` (own registry lock, same shape as `_PredictionLocks`) is acquired around
the miss-path load; inside it, the cache is re-checked (in case another thread already
finished loading while this one waited for the load lock) before calling `tft.load()`. The
fast path — an already-cached hit — is unaffected, since it returns before touching any
load lock at all. This introduces the same never-evicts shape bugs.md#pyq-164 tracks for
`_PredictionLocks`; noted there rather than re-filed.

Verification: `test_bundle_cache_dedupes_concurrent_loads_of_the_same_bundle_name` (5
threads, same name, monkeypatched `tft.load` sleeping 0.05s — asserts exactly one call and
that every thread got the identical bundle instance) and
`test_bundle_cache_still_loads_different_names_in_parallel` (2 threads, different names —
asserts their load spans overlap in time) in `tests/test_api.py`. `ruff check .`,
`pytest -q` and `scripts/backlog.py check` all clean.

---

## [PYQ-169]
Untrained-bundle `404`s leaked the absolute checkpoint filesystem path to the caller
Status: Resolved — 2026-07-30
Priority: Medium
Files: `pyquant/api/routes/forecast.py`, `pyquant/api/routes/explain.py`, `docs/http-api.md`

Problem: found by the same external review as bugs.md#pyq-163, with a live example already
sitting in this project's own docs. `tft.py`'s `_load` raises `FileNotFoundError(f"No
trained model for {symbol} at {ckpt}. Run \`pyquant train\` first.")`, where `ckpt` is an
absolute path. `_get_forecast` (`routes/forecast.py`) and `get_explanation`
(`routes/explain.py`) both did `raise HTTPException(404, detail=str(exc))`, forwarding that
absolute path straight into the response body — visible in `docs/http-api.md`'s own
`POST /scan` example (`"... at /.../checkpoints/ZZZZNOPE/model.ckpt. ..."`). Every
non-health endpoint requires a valid `X-API-Key`, which bounds the exposure to authenticated
callers rather than the public internet, but it's still handing any caller the server's
directory layout for no operational benefit.

Ask: sanitize the client-facing message; keep the full detail server-side for operators.

Acceptance criteria: the `404` `detail` for an untrained symbol names the symbol but not an
absolute path; the full original message is still logged; `docs/http-api.md`'s example
matches the actual (now-sanitized) response.

Resolution: both routes now catch the `FileNotFoundError`, log it in full
(`logger.info("Forecast/Explanation requested for untrained bundle: %s", exc)`), and raise
`HTTPException(404, detail=f"No trained model for {symbol}. Run \`pyquant train\` first.")`
— same wording, no path. `tft.py`'s underlying message is deliberately left unchanged,
since the CLI wants the full path (a human operator debugging locally benefits from it) —
only the two API boundary call sites sanitize. `docs/http-api.md`'s `POST /scan` example and
its `404` status-code-table row were updated to match.

Verification: `test_forecast_404_does_not_leak_the_absolute_checkpoint_path` and
`test_explain_404_does_not_leak_the_absolute_checkpoint_path` in `tests/test_api.py`, each
constructing a `FileNotFoundError` with an embedded absolute path and asserting it's absent
from the response while the symbol name is present. `ruff check .`, `pytest -q` and
`scripts/backlog.py check` all clean.

---

## [PYQ-170]
`PYQUANT_API_KEYS` misconfiguration was only caught per-request (`500`), not at process startup
Status: Resolved — 2026-07-30
Priority: Low
Files: `pyquant/api/app.py`, `pyquant/api/deps.py`, `docs/http-api.md`

Problem: found by the same external review as bugs.md#pyq-163. `require_api_key` already
fails loudly with `500` when `PYQUANT_API_KEYS` is unset and unauthenticated access wasn't
explicitly enabled — the right behaviour, but purely per-request. A misconfigured
deployment still starts, still passes a liveness check, and only reveals the problem to
whoever happens to send the first real request — slower and quieter than it needs to be for
something an orchestrator could instead treat as "this deployment never came up."

Ask: check once at startup too, so the process refuses to start rather than accepting
traffic while misconfigured.

Acceptance criteria: an unconfigured app fails to start (verified via a lifespan-hook
exception, not just a documentation claim); a correctly configured app (keys set, or
`PYQUANT_API_ALLOW_UNAUTHENTICATED=1`) starts normally; the existing per-request check is
unchanged, since it's legitimate defense in depth against the env var changing after
startup.

Resolution: `app.py` now takes a `lifespan` async context manager (`_lifespan`) that calls a
new shared helper, `deps.api_auth_is_configured()` (factored out of `require_api_key` so the
two checks cannot disagree about what "configured" means), and raises `RuntimeError` before
`yield` if it returns `False`. Verified empirically before shipping (not assumed): a
`RuntimeError` raised during ASGI lifespan startup propagates cleanly out of
`TestClient.__enter__` in the installed Starlette version, so `uvicorn` (or any real ASGI
server) refuses to start the same way. Confirmed this does **not** affect the existing test
suite: `TestClient(app)` used without a `with` block (every existing test in
`tests/test_api.py`, at module scope) never triggers ASGI lifespan events at all — only
`with TestClient(app) as c:` does, verified against the installed Starlette source before
relying on it.

Verification: `test_app_refuses_to_start_when_api_keys_unconfigured` (asserts
`pytest.raises(RuntimeError, match="PYQUANT_API_KEYS")` around `with TestClient(app):`),
`test_app_starts_cleanly_when_api_keys_configured`, and
`test_app_starts_cleanly_with_the_explicit_dev_opt_out` in `tests/test_api.py` — the one
place in the file that uses `with TestClient(app) as c:` rather than the module-level
`client`. `docs/http-api.md`'s Authentication section states the new boot-time behaviour.
`ruff check .`, `pytest -q` and `scripts/backlog.py check` all clean.
