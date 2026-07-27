"""PyQuant — probabilistic equity forecasting over a leak-audited daily panel.

Assembles OHLCV, macro, sector and news-sentiment features into one daily panel,
trains a Temporal Fusion Transformer on it, and serves p10/p50/p90 forecasts plus
feature-importance interpretation.

The package is layered so that the modelling stack stays swappable and the CLI
stays one front-end among several:

- :mod:`pyquant.data` — vendor fetches, indicators, panel assembly, caching.
- :mod:`pyquant.models` — every ``pytorch-forecasting`` / Lightning call.
- :mod:`pyquant.analysis` — forecast/metrics/interpretation objects, which import
  neither the modelling stack nor the terminal UI.
- :mod:`pyquant.cli` — Typer + Rich, a thin caller over the two layers above.
- :mod:`pyquant.api` — FastAPI, a second thin caller over the same two layers
  (``uv sync --extra api``); see ``docs/api-design.md``.
"""
