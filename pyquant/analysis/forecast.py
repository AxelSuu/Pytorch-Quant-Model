"""High-level forecasting: turn a trained bundle into a structured forecast."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pyquant.analysis.metrics import warn_on_quantile_crossing
from pyquant.config import Settings
from pyquant.data.dataset import build_panel, future_business_dates, panel_to_long
from pyquant.models import tft


def log_returns_to_prices(log_returns: np.ndarray, last_close: float) -> np.ndarray:
    """Reconstruct a price path from per-step log-return quantiles."""
    return float(last_close) * np.exp(np.cumsum(np.asarray(log_returns, dtype=float), axis=0))


@dataclass
class Forecast:
    """A multi-horizon quantile forecast plus context for display."""

    symbol: str
    last_date: pd.Timestamp
    current_price: float
    quantiles: list[float]
    predictions: np.ndarray  # shape (horizon, n_quantiles), monotonic per step
    history: pd.Series  # recent close history (date-indexed)
    # Set by __post_init__: how many points had to be reordered to make the band
    # monotonic. Non-zero means the model produced a degenerate band and what is
    # displayed is a repair of it -- worth surfacing rather than hiding (PYQ-124).
    n_quantile_crossings: int = 0

    def __post_init__(self) -> None:
        """Guarantee a monotonic band, and record whether one had to be imposed.

        QuantileLoss does not enforce monotonicity pointwise, so a p90 can land
        below a p10; PYQ-216 added detection but nothing acted on it. Every
        consumer -- the forecast table, the fan charts, `scan`'s "is the whole
        band on one side of zero" guard -- assumes monotonic input and misbehaves
        quietly without it (`scan` could read an *inverted* band as a confident
        BUY). Enforcing the invariant here rather than in generate_forecast()
        means no Forecast can exist in a crossed state, however it was built --
        including from the planned API layer (PYQ-124).
        """
        predictions = np.asarray(self.predictions, dtype=float)
        self.n_quantile_crossings = warn_on_quantile_crossing(predictions, self.quantiles)
        self.predictions = np.sort(predictions, axis=-1)

    @property
    def horizon(self) -> int:
        return self.predictions.shape[0]

    @property
    def forecast_dates(self) -> pd.DatetimeIndex:
        """The dates each forecast step is for -- the business days after ``last_date``.

        Derived from the same helper that appends the model's prediction rows, so
        the table, charts and JSON cannot drift from what was actually decoded
        (PYQ-115).
        """
        return future_business_dates(self.last_date, self.horizon)

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
    # Rebuild the panel from the toggles the bundle was trained with, not from
    # whatever the current defaults are -- otherwise the feature schema can differ
    # from the model's by construction (PYQ-119).
    settings = tft.settings_for_bundle(bundle, settings)
    panel = build_panel(symbol, settings, pin=pin)
    df = panel_to_long(panel, symbol)

    raw_predictions = tft.predict_quantiles(bundle, df)
    target = (bundle.meta.get("config") or {}).get("training", {}).get("target", "close")
    predictions = (
        log_returns_to_prices(raw_predictions, float(panel["Close"].iloc[-1]))
        if target == "log_return"
        else raw_predictions
    )
    # Forecast.__post_init__ enforces a monotonic band and records any crossing.
    return Forecast(
        symbol=symbol,
        last_date=panel.index[-1],
        current_price=float(panel["Close"].iloc[-1]),
        quantiles=list(bundle.meta["quantiles"]),
        predictions=predictions,
        history=panel["Close"].tail(history_days),
    )
