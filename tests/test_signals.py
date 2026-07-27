"""Tests for PYQ-255's signal P&L accounting on hand-built signal series."""

import pytest

from pyquant.analysis.signals import classify_signal, evaluate_signals

# --- classify_signal --------------------------------------------------------------


def test_classify_signal_buy_requires_the_whole_band_above_zero():
    assert classify_signal(3.0, lower_pct=0.5, upper_pct=6.0) == "BUY"


def test_classify_signal_confident_median_but_zero_straddling_band_is_hold():
    """PYQ-206/PYQ-124: a wide band isn't a real signal even if the median looks
    confident -- an inverted or merely wide band must not read as a confident BUY."""
    assert classify_signal(3.0, lower_pct=-1.0, upper_pct=6.0) == "HOLD"


def test_classify_signal_sell_requires_the_whole_band_below_zero():
    assert classify_signal(-3.0, lower_pct=-6.0, upper_pct=-0.5) == "SELL"


def test_classify_signal_below_threshold_is_hold_even_with_a_one_sided_band():
    assert classify_signal(1.0, lower_pct=0.2, upper_pct=1.8) == "HOLD"


# --- evaluate_signals --------------------------------------------------------------


def test_evaluate_signals_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        evaluate_signals(["BUY", "HOLD"], [1.0])


def test_evaluate_signals_on_empty_series():
    ev = evaluate_signals([], [])
    assert ev.n_periods == 0
    assert ev.strategy_pnl_pct == 0.0


def test_evaluate_signals_hit_rate_is_conditional_on_the_signal_firing():
    """A HOLD-heavy series must not dilute the BUY hit rate -- it's scored only
    over the periods where BUY actually fired, not over every period."""
    signals = ["HOLD", "HOLD", "HOLD", "BUY", "BUY"]
    returns = [-5.0, -5.0, -5.0, 3.0, -1.0]  # 3 HOLD losses would drag down an "over all" rate

    ev = evaluate_signals(signals, returns, cost_bps=0.0)

    assert ev.n_buy == 2
    assert ev.hit_rate_buy == 0.5  # 1 of 2 BUYs was followed by a positive return
    assert ev.avg_return_buy_pct == pytest.approx(1.0)  # mean(3.0, -1.0)


def test_evaluate_signals_sell_hit_rate_counts_negative_realized_returns():
    ev = evaluate_signals(["SELL", "SELL", "SELL"], [-2.0, -2.0, 1.0], cost_bps=0.0)
    assert ev.hit_rate_sell == pytest.approx(2 / 3)


def test_evaluate_signals_all_hold_never_trades_and_matches_buy_and_hold_exactly_offset():
    """All-HOLD: zero strategy P&L (never in the market), buy-and-hold still reports
    what the underlying actually did over the same period."""
    ev = evaluate_signals(["HOLD", "HOLD", "HOLD"], [2.0, -1.0, 3.0], cost_bps=10.0)
    assert ev.strategy_pnl_pct == 0.0
    assert ev.turnover == 0.0
    assert ev.buy_and_hold_pnl_pct == pytest.approx(4.0094, abs=1e-3)  # compounded 2%,-1%,3%


def test_evaluate_signals_charges_cost_only_when_the_position_changes():
    # BUY, BUY, SELL, SELL: one change entering (HOLD->BUY), one change (BUY->SELL).
    # Same-direction repeats (BUY->BUY, SELL->SELL) must not be charged again.
    signals = ["BUY", "BUY", "SELL", "SELL"]
    returns = [1.0, 1.0, 1.0, 1.0]
    cost_bps = 100.0  # 1% per change, deliberately large so it's easy to see in the total

    ev = evaluate_signals(signals, returns, cost_bps=cost_bps)

    assert ev.turnover == 0.5  # 2 changes out of 4 periods
    # Gross per period: BUY,BUY -> +1%,+1%; SELL,SELL -> -1%,-1% (short position).
    # Two of the four periods additionally pay the 1% cost (the two change points).
    uncosted = evaluate_signals(signals, returns, cost_bps=0.0)
    assert ev.strategy_pnl_pct < uncosted.strategy_pnl_pct


def test_evaluate_signals_a_perfectly_timed_strategy_beats_buy_and_hold():
    """BUY only ahead of gains, SELL only ahead of losses (an oracle signal) must
    outperform passively holding through both."""
    signals = ["BUY", "SELL", "BUY", "SELL"]
    returns = [5.0, -5.0, 5.0, -5.0]  # buy-and-hold nets ~ -0.5% compounded

    ev = evaluate_signals(signals, returns, cost_bps=1.0)

    assert ev.strategy_pnl_pct > ev.buy_and_hold_pnl_pct
    assert ev.strategy_pnl_pct > 0
