"""Tests for the nightly-precomputed forecast store (features.md#pyq-282)."""

import pandas as pd

from pyquant.data import forecast_store
from pyquant.data.trading_calendar import latest_session_before


def test_write_then_read_round_trips(settings):
    payload = {"symbol": "AAPL", "last_date": "2026-01-02", "current_price": 100.0}
    forecast_store.write_forecast(
        settings,
        "aapl",
        as_of="2026-01-02",
        computed_at="2026-01-03T01:00:00+00:00",
        payload=payload,
    )

    stored = forecast_store.read_forecast(settings, "AAPL")

    assert stored is not None
    assert stored.symbol == "AAPL"
    assert stored.as_of == "2026-01-02"
    assert stored.computed_at == "2026-01-03T01:00:00+00:00"
    assert stored.payload == payload


def test_read_returns_none_for_a_symbol_never_written(settings):
    assert forecast_store.read_forecast(settings, "NEVERWRITTEN") is None


def test_write_upserts_rather_than_duplicating(settings):
    forecast_store.write_forecast(
        settings,
        "AAPL",
        as_of="2026-01-02",
        computed_at="2026-01-03T01:00:00+00:00",
        payload={"v": 1},
    )
    forecast_store.write_forecast(
        settings,
        "AAPL",
        as_of="2026-01-05",
        computed_at="2026-01-06T01:00:00+00:00",
        payload={"v": 2},
    )

    stored = forecast_store.read_forecast(settings, "AAPL")

    assert stored.as_of == "2026-01-05"
    assert stored.payload == {"v": 2}


def test_symbol_lookup_is_case_insensitive(settings):
    forecast_store.write_forecast(
        settings, "aapl", as_of="2026-01-02", computed_at="2026-01-03T01:00:00+00:00", payload={}
    )
    assert forecast_store.read_forecast(settings, "AAPL") is not None
    assert forecast_store.read_forecast(settings, "aapl") is not None


# --- is_stale -----------------------------------------------------------------


def test_is_stale_true_for_an_as_of_far_in_the_past():
    assert forecast_store.is_stale("2000-01-01", now=pd.Timestamp("2026-01-05"))


def test_is_stale_false_when_as_of_is_the_expected_latest_session():
    expected = latest_session_before(pd.Timestamp("2026-01-05"))
    assert not forecast_store.is_stale(str(expected.date()), now=pd.Timestamp("2026-01-05"))


def test_is_stale_false_over_a_weekend_gap_with_no_new_session():
    """Friday's precompute must not read as stale on Saturday/Sunday -- there is
    no new session for the job to have covered in between."""
    friday = "2026-01-02"
    assert not forecast_store.is_stale(friday, now=pd.Timestamp("2026-01-03"))  # Saturday
    assert not forecast_store.is_stale(friday, now=pd.Timestamp("2026-01-04"))  # Sunday


def test_is_stale_true_once_a_new_session_has_closed_and_the_store_was_not_refreshed():
    """By Tuesday, Monday's session has closed; a store still holding Friday's
    as_of means the nightly job missed a run."""
    friday = "2026-01-02"
    assert forecast_store.is_stale(friday, now=pd.Timestamp("2026-01-06"))  # Tuesday
