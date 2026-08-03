"""PYQ-255: does `scan`'s BUY/SELL/HOLD signal actually make money?

The project measures forecast *accuracy* carefully (metrics.py) and its *usefulness*
not at all -- a model can have excellent MAE and a useless signal (right about
magnitude, wrong about the sign that matters), or mediocre MAE and a profitable one.
This module scores the second question directly: hit rate conditional on a signal
firing, average return per signal class, turnover, and cumulative P&L against
buy-and-hold with a configurable per-trade cost.

Two caveats worth carrying in your head while reading a report from this module:

1. The threshold below (and the ones ``scan`` uses) are themselves parameters. Tuning
   them against the same data they are evaluated on is a selection event; hold out a
   period before trusting a tuned threshold's P&L.
2. ``classify_signal``'s guard requires the *entire* band on one side of zero. With an
   uncalibrated band as wide as this project's default (99.3% coverage on a nominal
   80%, see docs/methodology.md), that guard rarely fires at all -- this module is
   only informative once PYQ-248's conformal calibration narrows the band, which is
   implemented but off by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Signal = Literal["BUY", "SELL", "HOLD"]

# Percent expected-move threshold below which a forecast is not confident enough to
# act on. Matches cli/app.py's scan() default; both now call the function below
# rather than each hardcoding their own copy of this number.
DEFAULT_THRESHOLD_PCT = 2.0


def classify_signal(
    expected_return_pct: float,
    lower_pct: float,
    upper_pct: float,
    *,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
) -> Signal:
    """BUY/SELL/HOLD from an expected move plus a whole-band-on-one-side guard.

    The guard matters more than the threshold: a wide, zero-straddling band is not
    a real signal even when the median alone looks confident (PYQ-206/PYQ-124) --
    an inverted or merely wide band could otherwise read as a confident BUY.
    """
    if expected_return_pct > threshold_pct and lower_pct > 0:
        return "BUY"
    if expected_return_pct < -threshold_pct and upper_pct < 0:
        return "SELL"
    return "HOLD"


@dataclass
class SignalEvaluation:
    """P&L accounting for a signal series scored against its realized returns."""

    n_buy: int = 0
    n_sell: int = 0
    n_hold: int = 0
    # Conditional on the signal firing -- not on all periods, which is the whole
    # point: a signal that fires rarely but correctly is more useful than one
    # that fires often and randomly, and averaging over every period (including
    # HOLDs) would hide the difference.
    hit_rate_buy: float = 0.0  # fraction of BUY periods with a positive realized return
    hit_rate_sell: float = 0.0  # fraction of SELL periods with a negative realized return
    avg_return_buy_pct: float = 0.0
    avg_return_sell_pct: float = 0.0
    turnover: float = 0.0  # fraction of periods where the position changed
    strategy_pnl_pct: float = 0.0  # cumulative, cost-adjusted
    buy_and_hold_pnl_pct: float = 0.0  # cumulative, same period, no signal/cost
    cost_bps: float = 0.0
    n_periods: int = 0


_POSITION = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}


def _compound(returns_pct: list[float]) -> float:
    """Compound a sequence of percent returns into one cumulative percent."""
    total = 1.0
    for r in returns_pct:
        total *= 1.0 + r / 100.0
    return (total - 1.0) * 100.0


def evaluate_signals(
    signals: list[str],
    realized_returns_pct: list[float],
    *,
    cost_bps: float = 5.0,
) -> SignalEvaluation:
    """Score a signal series against what actually happened.

    ``realized_returns_pct[i]`` is the percent move realized over the period
    ``signals[i]`` was generated for (e.g. one walk-forward window's actual close
    vs. its last observed price). A BUY takes a +1 position for that period, SELL
    a -1, HOLD a 0 (no P&L, no cost); a round-trip cost of ``cost_bps`` basis
    points is charged whenever the position changes from the previous period
    (including going from no position to one on the very first signal).
    """
    if len(signals) != len(realized_returns_pct):
        raise ValueError(
            "signals and realized_returns_pct must be the same length, got "
            f"{len(signals)} and {len(realized_returns_pct)}"
        )
    n = len(signals)
    if n == 0:
        return SignalEvaluation(cost_bps=cost_bps)

    positions = [_POSITION[s] for s in signals]
    buy_returns = [r for s, r in zip(signals, realized_returns_pct, strict=True) if s == "BUY"]
    sell_returns = [r for s, r in zip(signals, realized_returns_pct, strict=True) if s == "SELL"]

    changes = sum(1 for i in range(n) if positions[i] != (positions[i - 1] if i > 0 else 0.0))
    cost_pct = cost_bps / 100.0
    strategy_returns_pct = []
    for i in range(n):
        gross = positions[i] * realized_returns_pct[i]
        changed = positions[i] != (positions[i - 1] if i > 0 else 0.0)
        strategy_returns_pct.append(gross - cost_pct if changed else gross)

    return SignalEvaluation(
        n_buy=len(buy_returns),
        n_sell=len(sell_returns),
        n_hold=n - len(buy_returns) - len(sell_returns),
        hit_rate_buy=(sum(1 for r in buy_returns if r > 0) / len(buy_returns))
        if buy_returns
        else 0.0,
        hit_rate_sell=(sum(1 for r in sell_returns if r < 0) / len(sell_returns))
        if sell_returns
        else 0.0,
        avg_return_buy_pct=(sum(buy_returns) / len(buy_returns)) if buy_returns else 0.0,
        avg_return_sell_pct=(sum(sell_returns) / len(sell_returns)) if sell_returns else 0.0,
        turnover=changes / n,
        strategy_pnl_pct=_compound(strategy_returns_pct),
        buy_and_hold_pnl_pct=_compound(realized_returns_pct),
        cost_bps=cost_bps,
        n_periods=n,
    )
