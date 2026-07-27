"""Terminal front-end: the ``pyquant`` command.

Typer for command wiring and Rich for rendering. This package is intentionally
thin — it resolves configuration, calls :mod:`pyquant.models` and
:mod:`pyquant.analysis`, and formats the result. Business logic living here
would be unavailable to any non-terminal front-end, so it does not.
"""
