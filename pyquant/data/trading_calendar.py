"""The exchange calendar the forecast horizon is laid out on (PYQ-130).

``pd.bdate_range`` returns Mon-Fri and knows nothing about market holidays, so
a forecast made on 2026-07-02 labelled a step for 2026-07-03 -- the observed
Independence Day, a date on which no price will ever exist. That is wrong in
two separate ways:

- **Display.** The table, the PNG export and ``--format json`` all assert a
  date that cannot be scored against reality without a manual correction.
- **Model input.** ``extend_for_prediction()`` writes ``dow``/``month_num`` for
  those rows, and ``dow`` is a ``time_varying_known_real`` the decoder actually
  reads. A holiday row supplies a (weekday, position-in-horizon) pair that
  never occurs in training data, because training rows only exist for sessions
  that traded -- a silent train/serve skew in a known-future feature, roughly
  9 sessions a year, concentrated on exactly the dates with unusual volatility.

**No new dependency.** ``exchange_calendars`` and ``pandas_market_calendars``
both solve this, but pandas already ships the holiday-rule primitives needed to
state the NYSE calendar exactly, and pandas is already a core dependency. Per
the standing rule that a new dependency must be justified against doing
nothing (see PYQ-310 and PYQ-308 for the precedent), the rules below are ~20
lines and need no supply chain. Reach for ``exchange_calendars`` when a second
exchange is genuinely needed -- see the limitations at the bottom of this
module.

Rule-based rather than inferred from the observed price index: US market
holidays are mostly *rules* ("fourth Thursday in November"), not fixed dates,
so a calendar inferred from which weekdays were historically absent cannot
project the next Thanksgiving forward into a year it has not seen.
"""

from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """NYSE market holidays.

    Deliberately *not* ``USFederalHolidayCalendar``, which differs on three
    counts: the NYSE closes on Good Friday, which is not a federal holiday, and
    trades normally on Columbus Day and Veterans Day, which are.

    ``nearest_workday`` encodes the exchange's observance rule -- a holiday on
    a Saturday is taken the preceding Friday, on a Sunday the following Monday.
    New Year's Day is the documented exception: the NYSE does not close on
    31 December when 1 January falls on a Saturday, so it uses
    ``sunday_to_monday`` instead.
    """

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        # Federal holiday since 2021; first observed by the NYSE in 2022.
        Holiday("Juneteenth", month=6, day=19, start_date="2022-06-19", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


_CALENDAR = NYSEHolidayCalendar()


def exchange_holidays(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Market holidays falling in ``[start, end]``."""
    return pd.DatetimeIndex(_CALENDAR.holidays(start=start, end=end))


def next_sessions(after: pd.Timestamp, count: int) -> pd.DatetimeIndex:
    """The ``count`` trading sessions strictly after ``after``.

    Weekends and exchange holidays are skipped. Early closes (the half-day
    after Thanksgiving, Christmas Eve) *are* sessions and are kept: a price
    prints on them, so they are scoreable and belong in the horizon.
    """
    if count <= 0:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")

    start = pd.Timestamp(after).normalize() + pd.Timedelta(days=1)
    # Over-fetch so a run of holidays cannot come up short, then take the first
    # `count`. 9 holidays a year means a small horizon can never lose more than
    # a few days, but a fixed margin isn't total for a large `count` -- doubling
    # it until satisfied is (PYQ-154).
    margin = 15
    while True:
        span = pd.bdate_range(start, periods=count + margin)
        holidays = exchange_holidays(span[0], span[-1])
        sessions = span.difference(holidays)
        if len(sessions) >= count:
            return sessions[:count]
        margin *= 2


# --- Known limitations -------------------------------------------------------
# - NYSE/NASDAQ only. A non-US ticker gets the US calendar, which is wrong; the
#   panel is US-equity-shaped throughout (pyquant.data.sentiment makes the same
#   assumption about the 16:00 America/New_York close), so this is consistent
#   rather than newly wrong. Making it configurable is the point at which
#   `exchange_calendars` earns its place.
# - One-off closures -- national days of mourning, Hurricane Sandy in 2012, the
#   9/11 closure -- are not rules and are not modelled. They cannot be
#   predicted forward, which is the only direction this module is used in.
