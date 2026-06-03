# PyQuant

> Multi-modal market research with a **Temporal Fusion Transformer** — probabilistic
> forecasts you can interrogate, not just a single guessed price.

PyQuant fuses Yahoo Finance prices with macro, options, news-sentiment, and cross-asset
signals, trains a [Temporal Fusion Transformer](https://arxiv.org/abs/1912.09363), and
serves **multi-horizon quantile forecasts** (p10/p50/p90) plus **interpretability** —
which features and which past days drove the prediction — all from a Rich terminal UI.

*(Evolved from PyStock, a single-feature LSTM next-day price predictor.)*

```
            AAPL — 5-day forecast
┏━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃ Day ┃     p10 ┃     p50 ┃     p90 ┃   vs now ┃
┡━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│   1 │ $269.37 │ $278.34 │ $308.07 │ ▼ 10.18% │
│  …  │    …    │    …    │    …    │    …     │
└─────┴─────────┴─────────┴─────────┴──────────┘
Options: put/call 0.48 (bullish), ATM IV 12%, skew +6.25%
```

## What makes it interesting

- **Uncertainty, not point guesses.** Quantile regression gives a p10–p90 fan, so you see
  *confidence*, not just a number.
- **Multi-modal features.** Prices + technical indicators, macro (VIX, rates, CPI, yield
  curve), cross-asset sector returns, and FinBERT-scored news sentiment.
- **Interpretable.** `explain` surfaces TFT feature importances and temporal attention.
- **Graceful degradation.** Runs on pure Yahoo Finance OHLCV out of the box; each API key
  you add lights up another data source. Missing sources are dropped, never fatal.

## Install

PyQuant uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync                       # core install
uv sync --extra sentiment     # + FinBERT news sentiment (downloads the model on first use)
```

## Quickstart

```bash
uv run pyquant train AAPL              # train a TFT (saves to checkpoints/AAPL/)
uv run pyquant forecast AAPL           # quantile forecast + fan chart + options context
uv run pyquant forecast AAPL --export aapl.png   # also write a PNG fan chart
uv run pyquant explain AAPL            # feature importance + temporal attention
uv run pyquant scan AAPL,MSFT,NVDA     # multi-asset comparison table
```

Useful flags: `train --epochs 50 --period 10y`, and `--no-macro / --no-sentiment /
--no-sectors` to train on a leaner feature set.

## API keys (optional)

Copy `.env.example` to `.env` and add any you have. All are optional:

| Key | Unlocks | Free key |
|-----|---------|----------|
| `FRED_API_KEY` | Macro features (Fed funds, CPI, yield curve) | https://fredaccount.stlouisfed.org/apikeys |
| `FINNHUB_API_KEY` | News headlines for sentiment scoring | https://finnhub.io/register |

VIX and sector ETFs come from Yahoo Finance and need no key.

## How it works

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

A note on **options data**: Yahoo Finance only exposes the *current* option chain, not
history, so put/call ratio, ATM IV, and IV skew are shown as live market *context* rather
than fed to the model as a time series. The historical volatility signal the model trains
on is `Realized_Vol_20` (annualised 20-day realized volatility).

## Development

```bash
uv run pytest          # full suite (network-free; external APIs are mocked)
uv run ruff check .    # lint
```

## Disclaimer

For research and education only. Forecasts are model output, not financial advice.
