"""Comparators beyond persistence for judging model skill (PYQ-275).

``persistence_baseline_mae`` (``analysis/metrics.py``) is the only comparator
this project has ever measured skill against, and it is uniquely favourable
to the null: on a near-random-walk level series it is close to optimal by
construction (PYQ-247's own finding), so failing to beat it is weak evidence
about whether the model learned anything. "Does not beat persistence" and
"does not beat anything a competent practitioner would try" are different
statements, and only the second is worth publishing. A negative result is a
claim about the baselines it was measured against.

Every baseline here is a *point* forecaster -- it predicts a single path, the
same shape ``persistence_baseline_mae`` already compares against, not a
quantile band. That keeps this module's job narrow: name additional
comparators for the existing MAE-based skill number, not build a second
model.

Library-agnostic: no torch, no Lightning, no pytorch-forecasting, per the
architecture rule -- these operate on plain arrays, in whatever units the
target already is (price level or log-return; nothing here needs to know
which).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Baseline(Protocol):
    """Anything that can point-forecast a horizon from an encoder history."""

    name: str

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """``history`` is ``(n_samples, encoder_length)`` past target values.

        Returns ``(n_samples, horizon)`` point forecasts.
        """
        ...


@dataclass
class PersistenceBaseline:
    """Last observed value, carried forward flat -- the project's original comparator."""

    name: str = "persistence"

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Carry the last observed value forward, flat."""
        last = np.asarray(history)[:, -1]
        return np.broadcast_to(last[:, None], (last.shape[0], horizon)).copy()


@dataclass
class RandomWalkWithDriftBaseline:
    """Extrapolates the average per-step change observed in ``history``.

    ``forecast(h) = last + h * drift``, ``drift = (last - first) / (n - 1)`` --
    the textbook random-walk-with-drift estimator (Hyndman & Athanasopoulos,
    *Forecasting: Principles and Practice*).
    """

    name: str = "random_walk_drift"

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Extrapolate the observed per-step drift from the last value."""
        history = np.asarray(history, dtype=float)
        n = history.shape[1]
        last = history[:, -1]
        drift = (last - history[:, 0]) / (n - 1) if n > 1 else np.zeros_like(last)
        steps = np.arange(1, horizon + 1)
        return last[:, None] + drift[:, None] * steps[None, :]


@dataclass
class SeasonalNaiveBaseline:
    """Repeats the value observed ``season_length`` steps ago, cycling for ``horizon > season_length``.

    Default ``season_length=5`` -- a trading week -- so the first forecast
    step is "the same day last week", the standard seasonal-naive
    construction for daily data with a weekly pattern. Degrades to using the
    whole history as one season if there isn't enough of it.
    """

    season_length: int = 5
    name: str = "seasonal_naive"

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Repeat the value from one season ago, cycling as needed."""
        history = np.asarray(history)
        n = history.shape[1]
        season = max(1, min(self.season_length, n))
        out = np.empty((history.shape[0], horizon))
        for h in range(horizon):
            col = n - season + (h % season)
            out[:, h] = history[:, col]
        return out


@dataclass
class ClimatologicalBaseline:
    """Predicts the historical mean, flat across the whole horizon.

    The "no information beyond the unconditional average" comparator: it
    ignores recent trend and level entirely, so beating it is close to a
    necessary condition for a forecast to be using its input data at all.
    """

    name: str = "climatological"

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Predict the historical mean, flat across the horizon."""
        mean = np.asarray(history, dtype=float).mean(axis=1)
        return np.broadcast_to(mean[:, None], (mean.shape[0], horizon)).copy()


@dataclass
class AR1Baseline:
    """A hand-rolled AR(1) fit per sample by closed-form OLS, iterated forward.

    Declined ``statsmodels``' ARIMA/ETS -- the ticket's suggested alternative
    -- per CLAUDE.md's non-negotiable #5. ``statsmodels`` is already an
    optional dependency (the ``tuning`` extra), but this module is on
    ``train``/``backtest``'s core path, not just ``tune``'s, and a per-window
    ARIMA fit is materially slower than a closed-form AR(1) for a comparator
    whose entire point is to be a cheap, unglamorous floor -- see the
    resolution note on PYQ-275 for the full reasoning. AR(1) captures the one
    thing seasonal-naive and drift don't (mean-reverting autocorrelation)
    without a new hard dependency.
    """

    name: str = "ar1"

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Fit AR(1) per sample by closed-form OLS and iterate it forward."""
        history = np.asarray(history, dtype=float)
        n_samples, n = history.shape
        out = np.empty((n_samples, horizon))
        for i in range(n_samples):
            y = history[i]
            if n < 2:
                out[i, :] = y[-1] if n else 0.0
                continue
            x_t, x_t1 = y[:-1], y[1:]
            mean_x, mean_x1 = x_t.mean(), x_t1.mean()
            denom = float(np.sum((x_t - mean_x) ** 2))
            phi = float(np.sum((x_t - mean_x) * (x_t1 - mean_x1)) / denom) if denom > 0 else 0.0
            phi = max(-0.999, min(0.999, phi))  # keep the recursion from exploding
            intercept = mean_x1 - phi * mean_x
            value = y[-1]
            for h in range(horizon):
                value = intercept + phi * value
                out[i, h] = value
        return out


DEFAULT_BASELINES: list[Baseline] = [
    PersistenceBaseline(),
    RandomWalkWithDriftBaseline(),
    SeasonalNaiveBaseline(),
    AR1Baseline(),
    ClimatologicalBaseline(),
]


def baseline_maes(
    actuals: np.ndarray,
    history: np.ndarray,
    baselines: list[Baseline] | None = None,
) -> dict[str, float]:
    """MAE of each baseline's point forecast against ``actuals`` (n_samples, horizon).

    Report skill against *each* rather than collapsing to one: the strongest
    baseline (lowest MAE, i.e. hardest for the model to beat) is the honest
    one to headline, and only reporting the weakest available comparator is
    the failure mode this module exists to prevent.
    """
    actuals = np.asarray(actuals)
    horizon = actuals.shape[1]
    chosen = DEFAULT_BASELINES if baselines is None else baselines
    return {b.name: float(np.mean(np.abs(actuals - b.predict(history, horizon)))) for b in chosen}
