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
    symbol: str
    feature_importance: dict[str, float]  # feature -> normalised weight
    attention: np.ndarray  # attention weight per past time step (oldest..newest)

    def top_features(self, n: int = 10) -> list[tuple[str, float]]:
        return sorted(self.feature_importance.items(), key=lambda kv: kv[1], reverse=True)[:n]


def explain_forecast(
    symbol: str,
    settings: Settings,
    bundle: tft.ModelBundle | None = None,
) -> Interpretation:
    """Compute feature importance + temporal attention for ``symbol``."""
    symbol = symbol.upper()
    bundle = bundle or tft.load(symbol, settings)
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
    )


def attention_to_series(interp: Interpretation, panel_index: pd.DatetimeIndex) -> pd.Series:
    """Map the attention vector onto the most recent encoder dates for display."""
    n = len(interp.attention)
    dates = panel_index[-n:]
    return pd.Series(interp.attention, index=dates, name="attention")
