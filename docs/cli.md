# CLI reference

`pyquant` is one of PyQuant's two front-ends, and deliberately a thin one: each command
resolves settings, calls a single function in `analysis/` or `models/`, and renders the
dataclass it gets back. Nothing in the CLI decides anything a library caller could not
also decide — see {ref}`Architecture <what-each-layer-owns>` for why that split is load
bearing, and the [HTTP API](http-api.md) for the other front-end over the same calls.

```bash
uv run pyquant --help
uv run pyquant <command> --help
```

## Global options

These belong to the application, not to a command, so they go **before** the command name:
`pyquant --format json forecast AAPL`, not `pyquant forecast AAPL --format json`.

| Option | Effect |
|---|---|
| `--verbose`, `-v` | INFO-level logging. Shows what each data source contributed, and what was dropped. |
| `--debug` | DEBUG-level logging, and un-silences Lightning's own output and the warnings the CLI otherwise filters. The first thing to reach for on a NaN training loss or unexpected feature drift. |
| `--format rich\|json` | Output format. `rich` (default) prints tables and charts; `json` prints one JSON document and nothing else. |
| `--quiet`, `-q` | Suppress banners and progress bars. Tables still print. |

`--format json` implies `--quiet`: only the JSON document reaches stdout, so
`pyquant --format json forecast AAPL | jq` works without filtering ANSI escapes out first.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success. |
| `1` | An expected failure — no trained bundle for the symbol, a feature schema the bundle can no longer satisfy ({py:class}`~pyquant.models.tft.FeatureSchemaMismatch`), or an invalid value. Reported as a one-line `Error: …`, not a traceback. Also `doctor`'s code when any bundle is unusable. |
| `2` | A usage error from the argument parser — an unknown flag, a missing argument, `--format` given something other than `rich` or `json`. |

The distinction matters in scripts: `1` means PyQuant ran and something about the *data or
the bundle* was wrong; `2` means the command line itself was wrong. Every command's failure
paths are covered by tests that assert a non-zero exit rather than only asserting success —
that gap is what PYQ-120 was.

## train

```
pyquant train SYMBOLS [OPTIONS]
```

Trains a Temporal Fusion Transformer and writes a bundle to `checkpoints/<name>/`
(`model.ckpt`, `dataset_params.pt`, `meta.json`). Pass several comma-separated symbols to
train one *pooled* model over them.

`--name TEXT`
: Bundle directory name. Defaults to the symbols joined with `_`.

`--pin TEXT`
: Save the assembled panel under this name and reuse it on later runs. Pins are exempt
  from the cache TTL, which is what makes an experiment replayable months later — one of
  the three legs of reproducibility, alongside the seed and the recorded code version.

`--config PATH`
: YAML experiment file. Explicit CLI flags still win over it; see
  [Configuration](api/configuration.md) for the full precedence chain. A path that does not
  exist is an error, not a silent fall back to defaults.

`--epochs INTEGER`
: Override `TrainingConfig.max_epochs`.

`--period TEXT`
: History to pull, e.g. `5y`, `10y`.

`--no-macro` / `--no-sentiment` / `--no-sectors`
: Drop that source from the feature set for this run. Ablations measured with these flags
  are what `investigations.md#pyq-316` reports.

:::{note}
Pooling is *not* currently a free win. Measured on an AAPL+ARM comparison
(`investigations.md#pyq-315`), the pooled model scored **worse** than training each symbol
alone. One comparison, one pair — directional, not settled — but enough that the README no
longer states pooling's benefit as a rationale.
:::

The reported metrics come from the *best* checkpoint, not from the live post-fit model.
That is invariant 11 in [Leakage invariants](invariants.md), and it is a real distinction:
early stopping means the final epoch's weights are usually not the ones being scored.

## backtest

```
pyquant backtest SYMBOL [OPTIONS]
```

Walk-forward evaluation across rolling origins. Each origin trains on everything up to its
own cutoff and scores the window that starts at `cutoff + 1`; consecutive origins evaluate
disjoint windows. This is the honest protocol, and it is the one the headline numbers come
from — see {ref}`split-geometry`.

`--windows INTEGER`
: Number of rolling origins. Default `5`. Each one is a full training run, so this is the
  main cost knob.

`--signals`
: Additionally score the BUY/SELL/HOLD signal `scan` emits: hit rate, turnover, and P&L
  against buy-and-hold.

`--cost-bps FLOAT`
: Per-trade round-trip cost in basis points applied to that P&L. Default `5.0`. A signal
  that is only profitable at zero cost is not a signal.

`--config`, `--epochs`, `--period`, `--no-macro`, `--no-sentiment`, `--no-sectors`
: As for `train`.

:::{warning}
A five-window backtest is 25 predictions with an effective sample size around 5, because
overlapping horizons within a window are not independent draws. Read {ref}`sample-size`
before quoting a number this command produces. The docs deliberately carry a `+36.2%`
backtest result *without* promoting it over the 280-prediction headline for exactly this
reason.
:::

## tune

```
pyquant tune SYMBOL [OPTIONS]
```

Optuna hyperparameter search, writing the winning configuration to `configs/`. Requires
the `tuning` extra (`uv sync --extra tuning`).

Every trial selects on the same data, so the winner is retrained and scored on a held-out
period no trial ever saw. **Report that number, not the in-search validation loss**, which
is optimistically biased by construction.

`--trials INTEGER`
: Number of Optuna trials. Default `15`.

`--held-out-days INTEGER`
: Days reserved for the honest final score. Defaults to `TrainingConfig.validation_days`.

`--epochs INTEGER`
: Max epochs per trial and for the final retrain. Default `5` — trials are meant to be
  cheap and comparative, not converged.

`--config`, `--period`, `--no-macro`, `--no-sentiment`, `--no-sectors`
: As for `train`.

## forecast

```
pyquant forecast SYMBOL [OPTIONS]
```

Loads the trained bundle and emits a p10/p50/p90 forecast for the next
`max_prediction_length` trading days, with a terminal fan chart.

`--bundle TEXT`
: Load a differently-named bundle — how you forecast one symbol from a pooled model.

`--pin TEXT`
: Replay a named dataset snapshot instead of fetching live data.

`--export PATH`
: Also write a PNG fan chart there.

`--no-chart`
: Skip the terminal chart; print the table only.

Every predicted step is strictly beyond the last observed bar, and the dates in the table,
the JSON, the PNG and any appended rows are one set — invariants 3, 4 and 14. If the model
emits a non-monotonic band, the quantiles are reordered for display and the count is
reported rather than hidden, both in the table and as `n_quantile_crossings` in the JSON.

An options snapshot is printed as live market context when `use_options` is on. It is
**display only** and never a model input.

## explain

```
pyquant explain SYMBOL [OPTIONS]
```

Feature importance and temporal attention behind the most recent forecast.

`--bundle TEXT`
: As for `forecast`.

`--top INTEGER`
: Number of top features to show. Default `10`.

`--no-chart`
: Skip the terminal charts.

:::{note}
`explain` prints a caveat when the bundle's own recorded skill is non-positive: attributing
importance within a model that does not beat persistence tells you what the model leaned
on, not what moves the price. Permutation importance and the TFT's own variable-selection
weights agree on the single top feature but only weakly beyond it (Spearman ρ ≈ 0.3) —
`investigations.md#pyq-314`.
:::

## scan

```
pyquant scan SYMBOLS
```

Forecasts each comma-separated symbol from its own trained bundle and prints one
comparison table: current price, median target, expected return, band width, and a
BUY/SELL/HOLD signal.

One symbol failing does not sink the run. An untrained symbol comes back as `not_trained`
and a failing one as `error`, each as its own row, so a five-symbol scan still reports the
four that worked. The signal classifier and the row shape are shared with the HTTP API's
`POST /scan`, so the two front-ends cannot drift apart.

## snapshot

```
pyquant snapshot SYMBOL
```

Appends today's options snapshot to that symbol's accumulated history.

yfinance exposes only a *current* option chain, never history, so running this on a
schedule is the only way this project can ever build a historical options-implied series.
Once enough days accumulate, `build_panel()` picks them up automatically as
`OptionsPutCallRatio`, `OptionsATMIV` and `OptionsIVSkew`.

## precompute

```
pyquant precompute
pyquant precompute --symbols AAPL,MSFT
```

Computes each trained symbol's forecast (or just the given `--symbols`) and writes it to a
local store, tagged with `as_of` (the trading day it was built through) and `computed_at`.
[`GET /forecast/{symbol}`](http-api.md#forecasting) reads from this store instead of
running the pipeline live — this is what makes that a millisecond response instead of the
~65s a cold call otherwise costs (`investigations.md#pyq-319`).

Meant to run on a nightly schedule (a cron-triggered invocation after market close is the
cheapest starting point; no scheduler is bundled). One symbol failing does not sink the
run — a not-yet-trained or otherwise-flaky symbol is reported as its own `error` row, the
same discipline as `scan`. Exits non-zero only if every symbol failed.

`pyquant forecast` is unaffected and still always runs live; this does not replace it.

## doctor

```
pyquant doctor
```

Reports what is switched on — code version, which API keys are present, which optional
extras are installed, torch and accelerator availability — and checks every bundle on disk
against the features currently obtainable.

**Exits non-zero if any bundle's feature schema can no longer be satisfied.** That turns a
broken bundle into something you find by asking, rather than by a forecast failing three
weeks later. Key *presence* is reported; key values never are, anywhere.

## cache

```
pyquant cache list
pyquant cache prune
pyquant cache rm-pin NAME
```

`list`
: Cache directory, entry count, total size, and the names of every saved pin.

`prune`
: Delete TTL-expired entries. **Pins are never touched** — that is the entire point of a
  pin, and the reason it is a separate concept from a cache entry.

`rm-pin NAME`
: Remove one named pin. Reports whether a pin by that name existed rather than failing
  silently.

A cached panel is never replayed across a feature redefinition: the cache key fingerprints
the feature set, so a redefinition misses rather than serving a stale panel under a new
meaning (invariant 15).

## JSON output

Every command supports `--format json`, and every serializer lives in
{py:mod}`pyquant.analysis.serialize` — the same module the HTTP API constructs its
responses from, so a CLI document and an API response are produced by one implementation
rather than two schemas that happen to agree today.

```bash
uv run pyquant --format json forecast AAPL
```

```json
{
  "symbol": "AAPL",
  "last_date": "2026-07-24",
  "current_price": 211.18,
  "horizon": 5,
  "forecast_dates": ["2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30", "2026-07-31"],
  "quantiles": [0.1, 0.5, 0.9],
  "predictions": [[205.1, 211.4, 217.9], ["…"]],
  "n_quantile_crossings": 0,
  "median": [211.4, "…"],
  "expected_return_pct": 0.42
}
```

Two conventions hold across every document:

**Forecast dates are included, never left to be re-derived.** A consumer should not have to
rebuild an exchange calendar — holidays included — to find out which day a prediction is
for.

**Every rate carries its denominator.** `evaluation_to_dict` emits each sample count next
to the rate it describes, so a directional accuracy is never readable without the number of
points behind it. That convention is invariant 12, and it exists because this project once
reported "directional accuracy 100.0%" off five points.

Fields that require configuration are *omitted* rather than raising: `median` and
`expected_return_pct` are absent when `0.5` is not among the configured quantiles.
