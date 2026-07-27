# PyQuant

Probabilistic time series forecasting app with a [Temporal Fusion Transformer](https://arxiv.org/abs/1912.09363) and real time data.

[![CI](https://github.com/AxelSuu/Pytorch-Quant-Model/actions/workflows/ci.yml/badge.svg)](https://github.com/AxelSuu/Pytorch-Quant-Model/actions/workflows/ci.yml)

**Documentation:** [`docs/`](docs/) Build the site locally with
`uv run --group docs sphinx-build -b html docs docs/_build/html`.

PyQuant processes api vendor products such as prices, macro, sector, and news-sentiment signals.
Trains a [Temporal Fusion Transformer](https://arxiv.org/abs/1912.09363), and serves 5-day
p10/p50/p90 forecasts from a Rich terminal UI.

```
                  NVO — 5-day forecast
┏━━━━━┳━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
┃ Day ┃       Date ┃    p10 ┃    p50 ┃    p90 ┃  vs now ┃
┡━━━━━╇━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
│   1 │ 2026-07-24 │ $43.81 │ $45.43 │ $49.14 │ ▼ 5.70% │
│  …  │      …     │    …   │    …   │    …   │    …    │
│   5 │ 2026-07-30 │ $43.66 │ $45.62 │ $49.62 │ ▼ 5.31% │
└─────┴────────────┴────────┴────────┴────────┴─────────┘
Options: put/call 0.54 (bullish), ATM IV 42%, skew +3.12%
```


## Install

PyQuant uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync                       # core install
uv sync --extra sentiment     # + FinBERT news sentiment (downloads the model on first use)
```

## Quickstart

Defaults: 5 years of daily bars, a 60-day lookback, a 5-day horizon, and p10/p50/p90
quantiles, trained for up to 30 epochs with early stopping.

```bash
uv run pyquant train AAPL       # train a TFT  ->  checkpoints/AAPL/
uv run pyquant forecast AAPL    # 5-day p10/p50/p90 forecast + fan chart
uv run pyquant explain AAPL     # which features and which days drove it
```

## How it works

Every enrichment join degrades gracefully: a source that is missing a key, rate-limited,
or disabled is dropped with a logged notice rather than failing the run.

```
yfinance ─┐
FRED ─────┤   pyquant/data/*  ─►  unified daily panel  ─►  TimeSeriesDataSet
Finnhub ──┤   (graceful joins)        (dataset.py)              │
sectors ──┘                                                     ▼
                                          TemporalFusionTransformer (models/tft.py)
                                                                │
                              ┌─────────────────────────────────┤
                              ▼                                 ▼
                    quantile forecast                  feature importance
                    (analysis/forecast.py)             + attention (analysis/interpret.py)
                              └──────────────► Rich CLI (cli/app.py) ◄──────────────┘
```

## Development

Open bugs, planned features, and investigations live in [`backlog/`](backlog/README.md).

```bash
uv run pytest                                # full suite (network-free; external APIs are mocked)
uv run ruff check .                          # lint
uv run python scripts/backlog.py list        # open tickets, priority-sorted
uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html   # docs
```

## API keys (optional)

Copy `.env.example` to `.env` and add any you have. All are optional — PyQuant runs on
Yahoo Finance OHLCV alone, and each key simply lights up another data source:

| Key | Unlocks | Free key |
|-----|---------|----------|
| `FRED_API_KEY` | Macro features (Fed funds, CPI, yield curve) | https://fredaccount.stlouisfed.org/apikeys |
| `FINNHUB_API_KEY` | News headlines for sentiment scoring | https://finnhub.io/register |

## Commands

| Command | What it does |
|---|---|
| `train SYMBOLS` | Train a TFT and save a bundle. Comma-separated symbols pool into one model. |
| `forecast SYMBOL` | Quantile forecast, fan chart, and live options context. |
| `explain SYMBOL` | Feature importances + temporal attention for the forecast. |
| `scan SYMBOLS` | Compare forecasts across several trained symbols. |
| `backtest SYMBOL` | Walk-forward evaluation across rolling origins. |
| `cache list\|prune\|rm-pin` | Inspect and prune the local data-panel cache. |

Common flags: `train --epochs 50 --period 10y`, `train --name my_bundle` to override the
bundle directory, `backtest --windows 10`, `explain --top 20`, `forecast --export
aapl.png` to also write a PNG, and `--no-macro / --no-sentiment / --no-sectors` on
`train`/`backtest` for a leaner feature set. `pyquant <command> --help` has the rest.

**Global flags** go *before* the subcommand — `pyquant --debug train AAPL`:
`--format json`, `--quiet/-q`, `--verbose/-v`, `--debug`.

**Pooling.** `train AAPL,MSFT,NVDA` trains a single model across all three rather than
one model per ticker on a few years of bars each — meaningfully more data for the same
architecture. It saves to `checkpoints/AAPL_MSFT_NVDA/`, and
`forecast AAPL --bundle AAPL_MSFT_NVDA` pulls one symbol's forecast back out of it.

**Caching and pins.** Fetched data panels are cached in `.cache/pyquant/` for an hour so
repeated runs don't re-hit Yahoo/FRED/Finnhub. `train --pin NAME` / `forecast --pin NAME`
save and replay a named, TTL-exempt snapshot, so an experiment can be re-run later against
byte-identical data instead of whatever happens to be live that day.

```bash
uv run pyquant cache list      # size, entry count, saved pins
uv run pyquant cache prune     # drop expired entries (pins are never touched)
uv run pyquant cache rm-pin NAME
```

**Experiment configs.** `train`/`backtest` accept `--config configs/aapl_baseline.yaml`
to load a whole experiment — architecture, windows, epochs, data toggles — from one
checked-in file. Precedence is CLI flags > environment > YAML > built-in defaults. Two
examples ship in [`configs/`](configs/). (`forecast`/`explain` need no config: they read
the one recorded in the bundle they load.)

**Measuring quality.** `train` and `backtest` score the model against a naive persistence
baseline ("predict no change") with directional hit-rate and calibration coverage, rather
than an absolute loss with nothing to compare it to. Every rate prints with the sample
size behind it (`Evaluated on 56 windows (280 predictions)`); tune the holdout with
`TrainingConfig.validation_days`, default 60 days. `backtest` also prints each rolling
origin separately, because the spread across time is the reason to run more than one.

**JSON output.** `--format json` makes any command emit a machine-readable document on
stdout instead of tables — no Rich markup, no ANSI escapes.

```bash
uv run pyquant --format json forecast AAPL | jq '.forecast_dates, .predictions'
```
