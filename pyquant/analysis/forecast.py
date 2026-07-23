"""High-level forecasting: turn a trained bundle into a structured forecast."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pyquant.config import Settings
from pyquant.data.dataset import build_panel, panel_to_long
from pyquant.models import tft


@dataclass
class Forecast:
    """A multi-horizon quantile forecast plus context for display."""

    symbol: str
    last_date: pd.Timestamp
    current_price: float
    quantiles: list[float]
    predictions: np.ndarray  # shape (horizon, n_quantiles)
    history: pd.Series  # recent close history (date-indexed)

    @property
    def horizon(self) -> int:
        return self.predictions.shape[0]

    def quantile_series(self, q: float) -> np.ndarray:
        """Forecast path for a given quantile (must be one of self.quantiles)."""
        idx = self.quantiles.index(q)
        return self.predictions[:, idx]

    @property
    def median(self) -> np.ndarray:
        if 0.5 not in self.quantiles:
            raise ValueError(
                f"0.5 is not among the configured quantiles {self.quantiles}; "
                "TFTConfig.quantiles must include 0.5 to compute a median."
            )
        return self.quantile_series(0.5)

    def expected_return_pct(self) -> float:
        """Percent change from current price to the final-day median forecast."""
        return float((self.median[-1] - self.current_price) / self.current_price * 100)


def generate_forecast(
    symbol: str,
    settings: Settings,
    bundle: tft.ModelBundle | None = None,
    history_days: int = 90,
    pin: str | None = None,
) -> Forecast:
    """Build a forecast for ``symbol`` using its trained bundle.

    ``pin`` replays a reproducible dataset snapshot instead of live data
    (see pyquant.data.cache) -- useful for re-running a past experiment.
    """
    symbol = symbol.upper()
    bundle = bundle or tft.load(symbol, settings)
    panel = build_panel(symbol, settings, pin=pin)
    df = panel_to_long(panel, symbol)

    predictions = tft.predict_quantiles(bundle, df)
    return Forecast(
        symbol=symbol,
        last_date=panel.index[-1],
        current_price=float(panel["Close"].iloc[-1]),
        quantiles=list(bundle.meta["quantiles"]),
        predictions=predictions,
        history=panel["Close"].tail(history_days),
    )
