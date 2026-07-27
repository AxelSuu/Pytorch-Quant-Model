"""Interpretability: which features and which past days drove the forecast."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pyquant.config import Settings
from pyquant.data.dataset import build_panel, panel_to_long
from pyquant.models import tft


@dataclass
class Interpretation:
    """What the model attended to when producing a forecast.

    ``feature_importance`` is the TFT's variable-selection weight per feature and
    ``attention`` is one weight per encoder timestep, oldest to newest.
    ``panel_index`` records the dates those weights were computed over, so the
    attention vector can be aligned to real calendar days rather than to
    positions. Whether these weights mean what ``explain`` presents them as is
    itself an open question — see investigations.md#pyq-314.
    """

    symbol: str
    feature_importance: dict[str, float]  # feature -> normalised weight
    attention: np.ndarray  # attention weight per past time step (oldest..newest)
    panel_index: pd.DatetimeIndex  # dates of the panel this interpretation was computed from
    # The bundle's own recorded skill_vs_baseline (None if the bundle predates
    # evaluation being recorded). An interpretation of a model that does not
    # outperform persistence describes what the model attends to, not what moves
    # the price (investigations.md#pyq-314) -- carried here so every consumer
    # (CLI, --format json, a future API) can show that caveat next to the
    # numbers it qualifies, rather than presenting them with equal confidence
    # regardless of whether the bundle is any good.
    bundle_skill: float | None = None

    def top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Return the ``n`` highest-weighted features as ``(name, weight)``, descending."""
        return sorted(self.feature_importance.items(), key=lambda kv: kv[1], reverse=True)[:n]


def _bundle_skill(bundle: tft.ModelBundle) -> float | None:
    """Recompute skill_vs_baseline from the bundle's recorded evaluation.

    Not read directly off meta.json: EvaluationMetrics.skill_vs_baseline is a
    @property, not a dataclass field, so it was never serialised into
    meta["evaluation"] in the first place (only vars(evaluation)'s actual fields
    were). Recomputed from the two fields that are recorded, the same formula the
    property itself uses.
    """
    ev = bundle.meta.get("evaluation") or {}
    baseline_mae = ev.get("baseline_mae")
    model_mae = ev.get("model_mae")
    if not baseline_mae:
        return None
    return (baseline_mae - model_mae) / baseline_mae


def explain_forecast(
    symbol: str,
    settings: Settings,
    bundle: tft.ModelBundle | None = None,
) -> Interpretation:
    """Compute feature importance + temporal attention for ``symbol``."""
    symbol = symbol.upper()
    bundle = bundle or tft.load(symbol, settings)
    # Same reasoning as generate_forecast: the panel must match the bundle's own
    # recorded data config, not the live defaults (PYQ-119).
    settings = tft.settings_for_bundle(bundle, settings)
    panel = build_panel(symbol, settings)
    df = panel_to_long(panel, symbol)

    result = tft.interpret(bundle, df)
    attention = np.asarray(result["attention"], dtype=float)
    # Guard against NaNs/negatives from padding.
    attention = np.nan_to_num(attention, nan=0.0)
    return Interpretation(
        symbol=symbol,
        feature_importance=result["encoder_importance"],
        attention=attention,
        panel_index=panel.index,
        bundle_skill=_bundle_skill(bundle),
    )


def attention_to_series(interp: Interpretation) -> pd.Series:
    """Map the attention vector onto the most recent encoder dates for display."""
    n = len(interp.attention)
    dates = interp.panel_index[-n:]
    return pd.Series(interp.attention, index=dates, name="attention")
