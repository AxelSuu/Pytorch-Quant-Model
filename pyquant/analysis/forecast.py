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
    """Reconstruct a price path from a *single* sequence of per-step log-returns.

    Exact for one deterministic path (e.g. the realized actual): cumsum-then-exp
    is just the definition of compounding a return sequence. Do not call this on
    a `(horizon, n_quantiles)` array of quantile columns -- that treats each
    quantile as if it were its own path realizing that same marginal quantile at
    every step, which is a different (and wrong) object; see
    `log_return_quantiles_to_price_band` and PYQ-142.
    """
    return float(last_close) * np.exp(np.cumsum(np.asarray(log_returns, dtype=float), axis=0))


def log_return_quantiles_to_price_band(
    log_return_quantiles: np.ndarray, last_close: float, quantiles: list[float]
) -> np.ndarray:
    """Reconstruct an h-step price *quantile band* from per-step log-return quantiles.

    `log_return_quantiles` is `(horizon, n_quantiles)`: one quantile forecast per
    decoder step, e.g. from `predict_quantiles`. Naively cumsum-ing each quantile
    column independently (what this project did before PYQ-142) computes the path
    where every step *simultaneously* realizes its own marginal quantile -- an
    event that requires increasingly strong correlation across steps to occur at
    all, and overstates the true h-step quantile's width by ~sqrt(h) under an iid
    assumption (PYQ-142's 400k-path simulation).

    This instead: compounds the median path via cumsum (exact for a
    deterministic center under the market-standard assumption that log-returns
    are approximately mean/median-symmetric), and treats each quantile's
    per-step deviation from the median as an independent-across-steps dispersion
    contribution, so its cumulative effect scales as
    `sqrt(sum of squared per-step deviations)` -- the textbook identity for the
    standard deviation of a sum of independent variables -- rather than their
    linear sum. Under iid steps with constant per-step dispersion this reduces
    exactly to the analytic `sqrt(h)` scaling of a sum of iid variables.

    This is an explicit, documented distributional assumption (independence
    across horizon steps), not a property the model asserts -- see PYQ-142's
    "(a) retrain on cumulative targets vs (b) an explicit assumption" framing,
    which chose (b) as the non-invalidating fix. If a conformal offset
    (PYQ-248) has already been applied to `log_return_quantiles`, its effect on
    each step's deviation from the median flows through this formula like any
    other per-step correction, so the two compose without special-casing.
    """
    log_return_quantiles = np.asarray(log_return_quantiles, dtype=float)
    if log_return_quantiles.ndim != 2:
        raise ValueError(
            "log_return_quantiles must be (horizon, n_quantiles), got shape "
            f"{log_return_quantiles.shape}"
        )
    quantiles = list(quantiles)
    if 0.5 not in quantiles:
        raise ValueError(f"0.5 must be among quantiles to anchor the band, got {quantiles}")
    median_idx = quantiles.index(0.5)

    median_step = log_return_quantiles[:, median_idx]
    cum_median = np.cumsum(median_step)

    deviations = log_return_quantiles - median_step[:, None]
    cum_dispersion = np.sqrt(np.cumsum(deviations**2, axis=0))
    sign = np.array([1.0 if q > 0.5 else (-1.0 if q < 0.5 else 0.0) for q in quantiles])
    cum_log_return_quantiles = cum_median[:, None] + sign[None, :] * cum_dispersion

    return float(last_close) * np.exp(cum_log_return_quantiles)


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
    end: str | None = None,
) -> Forecast:
    """Build a forecast for ``symbol`` using its trained bundle.

    ``pin`` replays a reproducible dataset snapshot instead of live data
    (see pyquant.data.cache) -- useful for re-running a past experiment.
    ``end`` truncates the panel to simulate forecasting as of a past date
    (PYQ-284); forwarded to ``build_panel`` verbatim, with no shifting applied
    here -- see PYQ-284's ticket for why (vendors disagree on whether their own
    ``end`` is inclusive).
    """
    symbol = symbol.upper()
    bundle = bundle or tft.load(symbol, settings)
    # Rebuild the panel from the toggles the bundle was trained with, not from
    # whatever the current defaults are -- otherwise the feature schema can differ
    # from the model's by construction (PYQ-119).
    settings = tft.settings_for_bundle(bundle, settings)
    panel = build_panel(symbol, settings, end=end, pin=pin)
    df = panel_to_long(panel, symbol)

    raw_predictions = tft.predict_quantiles(bundle, df)
    target = (bundle.meta.get("config") or {}).get("training", {}).get("target", "close")
    predictions = (
        log_return_quantiles_to_price_band(
            raw_predictions, float(panel["Close"].iloc[-1]), list(bundle.meta["quantiles"])
        )
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
