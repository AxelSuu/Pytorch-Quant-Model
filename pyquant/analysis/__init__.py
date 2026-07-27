"""Library-agnostic result objects and scoring for forecasts.

Everything here operates on plain numpy/pandas values and deliberately imports
neither ``pytorch_forecasting``/Lightning nor Typer/Rich, so the same code backs
the CLI and a future REST layer without a rewrite (see ``docs/api-design.md``).

Modules: :mod:`~pyquant.analysis.forecast` (quantile forecasts),
:mod:`~pyquant.analysis.metrics` (skill, calibration, proper scoring rules),
:mod:`~pyquant.analysis.interpret` (feature importance and attention),
:mod:`~pyquant.analysis.calibrate` (band calibration) and
:mod:`~pyquant.analysis.serialize` (the single JSON mapping for all of them).
"""
