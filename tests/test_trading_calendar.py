"""Tests for the NYSE trading calendar (PYQ-130).

Runs without the ML stack: the calendar is deliberately in its own module so
``analysis/`` can ask "what dates is this forecast for" without importing
pytorch-forecasting through ``data/dataset.py``.
"""

import pandas as pd

from pyquant.data.trading_calendar import exchange_holidays, next_sessions


def test_nyse_holidays_for_2026_match_the_published_calendar():
    """A real-world anchor: the rules must produce the actual 2026 NYSE closures.

    Includes the three cases that separate the NYSE from the US federal
    calendar and from a naive fixed-date list: Good Friday (Apr 3) is a market
    holiday but not a federal one; Independence Day falls on a Saturday and is
    observed the preceding Friday (Jul 3); Juneteenth is present at all.
    """
    holidays = exchange_holidays(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31"))

    assert [str(d.date()) for d in holidays] == [
        "2026-01-01",  # New Year's Day
        "2026-01-19",  # MLK
        "2026-02-16",  # Presidents Day
        "2026-04-03",  # Good Friday
        "2026-05-25",  # Memorial Day
        "2026-06-19",  # Juneteenth
        "2026-07-03",  # Independence Day observed (Jul 4 is a Saturday)
        "2026-09-07",  # Labor Day
        "2026-11-26",  # Thanksgiving
        "2026-12-25",  # Christmas
    ]


def test_columbus_day_and_veterans_day_are_trading_sessions():
    """Federal holidays the market does *not* observe -- USFederalHolidayCalendar
    would wrongly remove both."""
    sessions = next_sessions(pd.Timestamp("2026-10-09"), 3)
    assert pd.Timestamp("2026-10-12") in sessions  # Columbus Day

    sessions = next_sessions(pd.Timestamp("2026-11-10"), 1)
    assert list(sessions) == [pd.Timestamp("2026-11-11")]  # Veterans Day


def test_juneteenth_is_not_a_holiday_before_the_exchange_observed_it():
    """First observed in 2022; a 2019 panel must not have it removed."""
    holidays = exchange_holidays(pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-31"))
    assert pd.Timestamp("2019-06-19") not in holidays


def test_next_sessions_skips_a_run_of_closed_days():
    """Christmas 2026 is a Friday, so the next session after Thursday 24th is Monday 28th."""
    assert list(next_sessions(pd.Timestamp("2026-12-24"), 2)) == [
        pd.Timestamp("2026-12-28"),
        pd.Timestamp("2026-12-29"),
    ]


def test_next_sessions_returns_exactly_the_requested_count_across_holidays():
    """Over-fetching must not leak extra dates, and holidays must not come up short."""
    for horizon in (1, 5, 10, 20):
        dates = next_sessions(pd.Timestamp("2026-12-23"), horizon)
        assert len(dates) == horizon
        assert dates.is_monotonic_increasing
        assert dates.is_unique


def test_next_sessions_is_empty_for_a_non_positive_horizon():
    assert len(next_sessions(pd.Timestamp("2026-07-02"), 0)) == 0


def test_next_sessions_is_total_past_the_fixed_margin_that_used_to_cap_it():
    """PYQ-154: a fixed 15-business-day over-fetch margin came up short once
    `count` grew large enough that more than 15 holidays fell in the span --
    verified directly at `count=380`, which used to raise instead of return.
    `next_sessions` must return exactly `count` dates for any `count`."""
    dates = next_sessions(pd.Timestamp("2026-01-01"), 380)
    assert len(dates) == 380
    assert dates.is_monotonic_increasing
    assert dates.is_unique
