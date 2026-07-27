"""Model training, checkpointing and inference.

:mod:`pyquant.models.tft` is the *only* module in the project that calls
``pytorch-forecasting`` or Lightning (``pyquant.data.dataset`` aside, which builds
the ``TimeSeriesDataSet``). Keeping those imports confined here is what lets the
analysis layer, the CLI and any future service share one code path and what would
make swapping the modelling backend a contained change rather than a rewrite.
"""
