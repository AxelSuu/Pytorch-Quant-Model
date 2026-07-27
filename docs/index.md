# PyQuant

A probabilistic equity forecasting harness. It assembles a leak-audited daily panel from
four vendors (Yahoo Finance, FRED, Finnhub, sector ETFs), trains a
[Temporal Fusion Transformer](https://arxiv.org/abs/1912.09363) via
pytorch-forecasting / Lightning, and serves p10/p50/p90 forecasts plus feature-importance
interpretation from a Rich terminal UI.

:::{admonition} The forecaster does not currently beat a naive persistence baseline
:class: warning

Measured over 56 walk-forward windows / 280 predictions: **−23.5% skill** against
"predict no change", **57.5%** directional accuracy, and **99.3%** empirical coverage on a
nominal 80% band. This is the project's central open problem, and it is reported here
rather than tuned out of sight. [Methodology](methodology.md) explains how those numbers
are produced and what each one does and does not license you to conclude.
:::

## Quickstart

PyQuant is managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # core install
uv sync --extra sentiment        # + FinBERT news sentiment

uv run pyquant train AAPL        # train a TFT   -> checkpoints/AAPL/
uv run pyquant forecast AAPL     # 5-day p10/p50/p90 forecast + fan chart
uv run pyquant explain AAPL      # which features and which days drove it
uv run pyquant backtest AAPL --windows 5
uv run pyquant --format json forecast AAPL
```

Defaults: 5 years of daily bars, a 60-day encoder, a 5-day horizon, p10/p50/p90, up to 30
epochs with early stopping, and a 60-trading-day validation holdout. Every one of those is
a field on {py:class}`~pyquant.config.Settings` — see the
[configuration reference](api/configuration.md).

All API keys are optional. PyQuant runs on Yahoo Finance OHLCV alone; each key lights up
one more source, and a source that cannot be reached is dropped with a logged notice
rather than failing the run. That contract, and the one place it deliberately stops, is
described in {ref}`Architecture <graceful-degradation>`.

## Where to start reading

[Architecture](architecture.md)
: What each layer owns, and why pytorch-forecasting is confined to two modules.

[Leakage invariants](invariants.md)
: The most useful page in this documentation. Seven look-ahead leaks have been found and
  fixed in this pipeline; every one was correct in each individual file and wrong across
  files. Each is stated here as a falsifiable claim, linked to the ticket that established
  it and the test that now guards it.

[Methodology](methodology.md)
: How the model is evaluated, and an honest reading of the current numbers.

[API reference](api/index.md)
: Autodoc over every public module, plus the full configuration field reference.

## Concepts

```{toctree}
:maxdepth: 2

architecture
invariants
methodology
```

## Reference

```{toctree}
:maxdepth: 2

api/index
api/configuration
```

## Design notes

```{toctree}
:maxdepth: 1

api-design
```

## Backlog

Open bugs, planned features and answered investigations live in
[`backlog/`](https://github.com/AxelSuu/Pytorch-Quant-Model/tree/main/backlog) next to the
code, and are deliberately *not* mirrored here — their value is that a resolution note
sits one directory from the commit that earned it. Ticket IDs are stable and never move
between files, so a reference such as `PYQ-115` stays valid forever.

- `PYQ-1xx` — bugs
- `PYQ-2xx` — features
- `PYQ-3xx` — investigations
