"""Tests for panel assembly and TimeSeriesDataSet construction."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from pyquant import provenance
from pyquant.data import cache, dataset, sentiment
from pyquant.data.prices import add_technical_indicators
from pyquant.data.sentiment import _EXCHANGE_TZ


def _patch_prices(monkeypatch, df):
    monkeypatch.setattr(
        dataset, "fetch_prices", lambda *a, **k: add_technical_indicators(df)
    )


def _epoch_et(date: pd.Timestamp, hour: int) -> int:
    """Epoch seconds for ``hour`` exchange-local time on ``date``."""
    local = dt.datetime.combine(date.date(), dt.time(hour), tzinfo=_EXCHANGE_TZ)
    return int(local.timestamp())


def test_build_panel_baseline_ohlcv(monkeypatch, sample_ohlcv_df, settings):
    _patch_prices(monkeypatch, sample_ohlcv_df)
    panel = dataset.build_panel("AAPL", settings)
    assert "Close" in panel.columns
    assert "RSI_14" in panel.columns
    assert not panel.isna().any().any()


def test_build_panel_drops_indicator_warmup_rows(monkeypatch, sample_ohlcv_df, settings):
    """Warm-up rows must be dropped, not silently fabricated via bfill (which
    borrows the first valid value).

    The panel starts where the *longest* indicator warm-up ends -- derived here
    rather than hardcoded, because which indicator binds is not fixed: it was
    SMA_50's 49 rows until PYQ-137 lengthened the EMA warm-up past it. Hardcoding
    the winner is what let PYQ-121/PYQ-132 hide behind SMA_50 in the first place.
    """
    _patch_prices(monkeypatch, sample_ohlcv_df)
    indicators = add_technical_indicators(sample_ohlcv_df)
    expected_start = indicators.dropna().index[0]

    panel = dataset.build_panel("AAPL", settings)

    assert panel.index[0] == expected_start
    assert len(panel) == len(indicators.dropna())
    assert panel.notna().all().all()


def test_build_panel_joins_enabled_sources(monkeypatch, sample_ohlcv_df, settings):
    _patch_prices(monkeypatch, sample_ohlcv_df)
    settings.data.use_macro = True
    settings.data.use_sectors = True

    macro_df = pd.DataFrame({"VIX": 20.0}, index=sample_ohlcv_df.index)
    sec_df = pd.DataFrame({"SEC_XLK": 0.01}, index=sample_ohlcv_df.index)
    monkeypatch.setattr(dataset, "fetch_macro", lambda *a, **k: macro_df)
    monkeypatch.setattr(dataset, "fetch_sector_returns", lambda *a, **k: sec_df)

    panel = dataset.build_panel("AAPL", settings)
    assert "VIX" in panel.columns
    assert "SEC_XLK" in panel.columns


def test_build_panel_drops_target_symbols_own_sector_column(monkeypatch, sample_ohlcv_df, settings):
    """`pyquant train SPY` must not get a SEC_SPY feature that duplicates the target."""
    _patch_prices(monkeypatch, sample_ohlcv_df)
    settings.data.use_sectors = True

    sec_df = pd.DataFrame(
        {"SEC_SPY": 0.02, "SEC_XLK": 0.01}, index=sample_ohlcv_df.index
    )
    monkeypatch.setattr(dataset, "fetch_sector_returns", lambda *a, **k: sec_df)

    panel = dataset.build_panel("SPY", settings)
    assert "SEC_SPY" not in panel.columns
    assert "SEC_XLK" in panel.columns


def test_build_panel_lands_post_close_news_on_the_next_trading_row(
    monkeypatch, sample_ohlcv_df, settings
):
    """The leak PYQ-129 fixed was only visible across two files.

    fetch_sentiment picks the session; build_panel does the join. Each was
    self-consistent while the pair attached post-close headlines to the row
    whose target is that day's close -- so the invariant is asserted here, on
    the assembled panel, not on either half.
    """
    _patch_prices(monkeypatch, sample_ohlcv_df)
    settings.data.use_sentiment = True
    settings.finnhub_api_key = "dummy"

    # Pick the first Friday/Monday pair that survives the indicator warm-up,
    # rather than a fixed index -- the warm-up length moves (PYQ-137).
    sessions = add_technical_indicators(sample_ohlcv_df).dropna().index
    friday, monday = next(
        (a, b)
        for a, b in zip(sessions, sessions[1:], strict=False)
        if a.day_name() == "Friday" and b.day_name() == "Monday"
    )
    monkeypatch.setattr(sentiment, "_finbert", lambda: object())
    monkeypatch.setattr(
        sentiment,
        "fetch_news",
        lambda *a, **k: [
            {"headline": "midday", "datetime": _epoch_et(friday, 11)},
            {"headline": "after the bell", "datetime": _epoch_et(friday, 17)},
        ],
    )
    monkeypatch.setattr(sentiment, "score_headlines", lambda h: [0.6, -0.9])

    panel = dataset.build_panel("AAPL", settings)

    assert panel.loc[friday, "Sentiment"] == pytest.approx(0.6)
    assert panel.loc[friday, "HeadlineCount"] == 1
    assert panel.loc[monday, "Sentiment"] == pytest.approx(-0.9)
    assert panel.loc[monday, "HeadlineCount"] == 1


def test_feature_columns_excludes_identifiers_and_target(monkeypatch, sample_ohlcv_df, settings):
    _patch_prices(monkeypatch, sample_ohlcv_df)
    panel = dataset.build_panel("AAPL", settings)
    long = dataset.panel_to_long(panel, "AAPL")
    feats = dataset.feature_columns(long)
    assert "Close" not in feats  # target excluded
    assert "symbol" not in feats
    assert "time_idx" not in feats
    assert "RSI_14" in feats


def test_panel_to_long_adds_log_returns_and_selects_them_as_the_default_target(
    monkeypatch, sample_ohlcv_df, settings
):
    _patch_prices(monkeypatch, sample_ohlcv_df)
    settings.training.target = "log_return"
    panel = dataset.build_panel("AAPL", settings)
    long = dataset.panel_to_long(panel, "AAPL")

    assert dataset.LOG_RETURN_TARGET in long
    assert long[dataset.LOG_RETURN_TARGET].iloc[0] == pytest.approx(
        np.log(panel["Close"].iloc[1] / panel["Close"].iloc[0])
    )
    assert dataset.target_column(settings) == dataset.LOG_RETURN_TARGET
    assert dataset.LOG_RETURN_TARGET not in dataset.feature_columns(long)


def test_build_panel_uses_cache_on_second_call(monkeypatch, sample_ohlcv_df, settings):
    calls = []

    def fake_fetch_prices(*a, **k):
        calls.append(1)
        return add_technical_indicators(sample_ohlcv_df)

    monkeypatch.setattr(dataset, "fetch_prices", fake_fetch_prices)
    settings.data.cache_enabled = True

    first = dataset.build_panel("AAPL", settings)
    second = dataset.build_panel("AAPL", settings)

    assert len(calls) == 1  # second call served from cache, no re-fetch
    pd.testing.assert_frame_equal(first, second)


def test_build_panel_bypasses_cache_when_disabled(monkeypatch, sample_ohlcv_df, settings):
    calls = []

    def fake_fetch_prices(*a, **k):
        calls.append(1)
        return add_technical_indicators(sample_ohlcv_df)

    monkeypatch.setattr(dataset, "fetch_prices", fake_fetch_prices)
    assert settings.data.cache_enabled is False  # the fixture default

    dataset.build_panel("AAPL", settings)
    dataset.build_panel("AAPL", settings)

    assert len(calls) == 2


def test_build_panel_pin_ignores_later_live_data_changes(monkeypatch, sample_ohlcv_df, settings):
    _patch_prices(monkeypatch, sample_ohlcv_df)

    pinned = dataset.build_panel("AAPL", settings, pin="experiment-1")

    # Live data "changes" after the pin was taken.
    changed = sample_ohlcv_df.copy()
    changed["Close"] = changed["Close"] * 2
    _patch_prices(monkeypatch, changed)

    replayed = dataset.build_panel("AAPL", settings, pin="experiment-1")
    pd.testing.assert_frame_equal(replayed, pinned)


def test_non_feature_constants_match_real_column_names(monkeypatch, sample_ohlcv_df, settings):
    """Every entry in _NON_FEATURE should refer to a column panel_to_long actually
    produces -- stale entries are dead and risk masking a real rename/refactor bug."""
    _patch_prices(monkeypatch, sample_ohlcv_df)
    panel = dataset.build_panel("AAPL", settings)
    long = dataset.panel_to_long(panel, "AAPL")
    for col in dataset._NON_FEATURE:
        assert col in long.columns, f"{col!r} in _NON_FEATURE never appears in panel_to_long output"


def test_make_dataset_builds_timeseries_dataset(monkeypatch, sample_ohlcv_df, settings):
    _patch_prices(monkeypatch, sample_ohlcv_df)
    panel = dataset.build_panel("AAPL", settings)
    long = dataset.panel_to_long(panel, "AAPL")
    ds = dataset.make_dataset(long, settings)
    # The dataset should yield at least one sample.
    assert len(ds) > 0
    params = ds.get_parameters()
    assert params["max_encoder_length"] == settings.training.max_encoder_length


# --- PYQ-133: the fingerprint must cover which code computed the columns -----


def test_cache_fingerprint_changes_with_the_package_version(monkeypatch, settings):
    """PYQ-121 redefined RSI_14 and PYQ-123 changed which rows survive, neither
    of which altered anything the fingerprint covered -- so an upgraded install
    happily served a panel built by the previous definition."""
    monkeypatch.setattr(provenance, "package_version", lambda: "1.0.0")
    before = dataset._cache_fingerprint("AAPL", settings, None, None)

    monkeypatch.setattr(provenance, "package_version", lambda: "1.1.0")
    after = dataset._cache_fingerprint("AAPL", settings, None, None)

    assert before != after
    assert cache.fingerprint_key(before) != cache.fingerprint_key(after)


def test_cache_fingerprint_is_stable_for_identical_inputs(settings):
    a = dataset._cache_fingerprint("AAPL", settings, None, None)
    b = dataset._cache_fingerprint("AAPL", settings, None, None)
    assert cache.fingerprint_key(a) == cache.fingerprint_key(b)


def test_cache_fingerprint_records_no_secret_values(settings):
    """Key *presence* is fingerprinted; key values never are."""
    settings.fred_api_key = "super-secret-fred"
    settings.finnhub_api_key = "super-secret-finnhub"
    fingerprint = dataset._cache_fingerprint("AAPL", settings, None, None)
    assert "super-secret-fred" not in str(fingerprint)
    assert "super-secret-finnhub" not in str(fingerprint)


# --- PYQ-115: prediction rows must cover the future, not observed days -------


def test_future_business_dates_starts_after_last_observed_date():
    dates = dataset.future_business_dates(pd.Timestamp("2024-10-07"), 5)
    assert len(dates) == 5
    assert dates[0] > pd.Timestamp("2024-10-07")
    # Business days only -- no weekend rows.
    assert all(d.dayofweek < 5 for d in dates)


# --- PYQ-130: the forecast dates must be sessions the exchange actually holds -


def test_future_business_dates_skips_an_observed_exchange_holiday():
    """2026-07-04 is a Saturday, so NYSE closes Friday 2026-07-03.

    pd.bdate_range returns Mon-Fri and knows nothing about market holidays, so
    it labelled a forecast step for a day on which no price will ever exist --
    unscoreable against reality, and a decoder row whose `dow` never occurs in
    training data (PYQ-130).
    """
    dates = dataset.future_business_dates(pd.Timestamp("2026-07-02"), 5)

    assert pd.Timestamp("2026-07-03") not in dates
    assert list(dates) == [
        pd.Timestamp("2026-07-06"),
        pd.Timestamp("2026-07-07"),
        pd.Timestamp("2026-07-08"),
        pd.Timestamp("2026-07-09"),
        pd.Timestamp("2026-07-10"),
    ]


def test_future_business_dates_skips_thanksgiving_but_keeps_the_half_day():
    """The Friday after Thanksgiving trades (early close); the Thursday does not."""
    dates = dataset.future_business_dates(pd.Timestamp("2026-11-25"), 3)

    assert pd.Timestamp("2026-11-26") not in dates  # Thanksgiving
    assert list(dates) == [
        pd.Timestamp("2026-11-27"),  # half day, but a session
        pd.Timestamp("2026-11-30"),
        pd.Timestamp("2026-12-01"),
    ]


def test_future_business_dates_skips_good_friday():
    """Good Friday is an NYSE holiday and not a US federal one -- the two calendars differ."""
    dates = dataset.future_business_dates(pd.Timestamp("2026-04-02"), 2)

    assert pd.Timestamp("2026-04-03") not in dates
    assert list(dates) == [pd.Timestamp("2026-04-06"), pd.Timestamp("2026-04-07")]


def test_extend_for_prediction_appends_exactly_the_dates_the_forecast_reports(
    monkeypatch, sample_ohlcv_df, settings
):
    """One set of dates across the appended rows, the table, the JSON and the PNG.

    PYQ-115 made one helper the single source of truth for the forecast dates;
    this asserts the model's decoder rows and that helper cannot drift apart
    when the helper's definition changes (PYQ-130).
    """
    _patch_prices(monkeypatch, sample_ohlcv_df)
    df = dataset.panel_to_long(dataset.build_panel("AAPL", settings), "AAPL")
    last_date = df["Date"].iloc[-1]

    extended = dataset.extend_for_prediction(df, 5)
    appended = list(extended.tail(5)["Date"])

    assert appended == list(dataset.future_business_dates(last_date, 5))


def test_extend_for_prediction_appends_horizon_future_rows(monkeypatch, sample_ohlcv_df, settings):
    _patch_prices(monkeypatch, sample_ohlcv_df)
    df = dataset.panel_to_long(dataset.build_panel("AAPL", settings), "AAPL")
    observed_max = int(df["time_idx"].max())
    last_date = df["Date"].iloc[-1]

    extended = dataset.extend_for_prediction(df, 5)

    assert len(extended) == len(df) + 5
    future = extended[extended["time_idx"] > observed_max]
    assert len(future) == 5
    # Contiguous time_idx continuing from the observed data.
    assert list(future["time_idx"]) == list(range(observed_max + 1, observed_max + 6))
    # Genuinely future calendar dates.
    assert (future["Date"] > last_date).all()


def test_extend_for_prediction_recomputes_calendar_features(monkeypatch, sample_ohlcv_df, settings):
    """dow/month_num are known-in-future reals -- they must describe the future date."""
    _patch_prices(monkeypatch, sample_ohlcv_df)
    df = dataset.panel_to_long(dataset.build_panel("AAPL", settings), "AAPL")
    extended = dataset.extend_for_prediction(df, 5)
    future = extended.tail(5)
    assert list(future["dow"]) == [float(d.dayofweek) for d in future["Date"]]
    assert list(future["month_num"]) == [float(d.month) for d in future["Date"]]


def test_extend_for_prediction_extends_each_symbol_independently(
    monkeypatch, sample_ohlcv_df, settings
):
    _patch_prices(monkeypatch, sample_ohlcv_df)
    panel = dataset.build_panel("AAPL", settings)
    pooled = pd.concat(
        [dataset.panel_to_long(panel, "AAA"), dataset.panel_to_long(panel, "BBB")],
        ignore_index=True,
    )
    extended = dataset.extend_for_prediction(pooled, 3)
    assert len(extended) == len(pooled) + 6
    for symbol in ("AAA", "BBB"):
        group = extended[extended["symbol"] == symbol]
        assert int(group["time_idx"].max()) == int(
            pooled[pooled["symbol"] == symbol]["time_idx"].max()
        ) + 3


# --- PYQ-116: pooled symbols must share one calendar -------------------------


def test_align_time_index_maps_the_same_date_to_the_same_index(
    monkeypatch, sample_ohlcv_df, settings
):
    _patch_prices(monkeypatch, sample_ohlcv_df)
    panel = dataset.build_panel("AAPL", settings)
    # Two symbols with very different history lengths but the same last date.
    pooled = pd.concat(
        [
            dataset.panel_to_long(panel, "LONG"),
            dataset.panel_to_long(panel.tail(80), "SHORT"),
        ],
        ignore_index=True,
    )
    # Before alignment: position-based, so the same date has different indices.
    overlap = pooled.pivot_table(index="Date", columns="symbol", values="time_idx").dropna()
    assert not (overlap["LONG"] == overlap["SHORT"]).all()

    aligned = dataset.align_time_index(pooled)

    overlap = aligned.pivot_table(index="Date", columns="symbol", values="time_idx").dropna()
    assert (overlap["LONG"] == overlap["SHORT"]).all()


def test_align_time_index_leaves_a_single_symbol_unchanged(
    monkeypatch, sample_ohlcv_df, settings
):
    """A lone symbol's dates are already contiguous positions -- alignment is a no-op."""
    _patch_prices(monkeypatch, sample_ohlcv_df)
    df = dataset.panel_to_long(dataset.build_panel("AAPL", settings), "AAPL")
    aligned = dataset.align_time_index(df)
    pd.testing.assert_series_equal(aligned["time_idx"], df["time_idx"], check_dtype=False)


# --- PYQ-123: no back-filling future values into leading rows ----------------


def test_build_panel_does_not_backfill_a_late_starting_source(
    monkeypatch, sample_ohlcv_df, settings
):
    """A joined source that starts late must not have its first value carried
    *backwards* into earlier rows -- that is look-ahead, same class as PYQ-101."""
    _patch_prices(monkeypatch, sample_ohlcv_df)
    settings.data.use_macro = True

    # VIX only exists for the back half of the price history, at a constant 99.
    price_dates = sample_ohlcv_df.index
    late_start = price_dates[300]
    macro = pd.DataFrame({"VIX": 99.0}, index=price_dates[price_dates >= late_start])
    macro.index.name = "Date"
    monkeypatch.setattr(dataset, "fetch_macro", lambda *a, **k: macro)

    panel = dataset.build_panel("AAPL", settings)

    # Every surviving row must be at or after the source's first observation:
    # the pre-source rows are dropped, not fabricated from a future value.
    assert panel.index.min() >= late_start
    assert (panel["VIX"] == 99.0).all()


# --- PYQ-256: has_sentiment_data ---------------------------------------------


def test_has_sentiment_data_separates_no_data_rows_from_genuinely_neutral_ones(
    monkeypatch, sample_ohlcv_df, settings
):
    """Sentiment=0 means two different things -- "no coverage" for most training
    rows and "neutral news" at predict time -- and only the second ever occurs
    live. The indicator column lets the model condition on which (PYQ-256)."""
    _patch_prices(monkeypatch, sample_ohlcv_df)
    settings.data.use_sentiment = True
    settings.finnhub_api_key = "dummy"

    sessions = add_technical_indicators(sample_ohlcv_df).dropna().index
    covered_from = sessions[len(sessions) // 2]
    monkeypatch.setattr(sentiment, "_finbert", lambda: object())
    monkeypatch.setattr(
        sentiment,
        "fetch_news",
        lambda *a, **k: [{"headline": "news", "datetime": _epoch_et(covered_from, 11)}],
    )
    monkeypatch.setattr(sentiment, "score_headlines", lambda h: [0.5])

    panel = dataset.build_panel("AAPL", settings)

    assert "has_sentiment_data" in panel.columns
    before = panel.loc[panel.index < covered_from, "has_sentiment_data"]
    after = panel.loc[panel.index >= covered_from, "has_sentiment_data"]
    assert (before == 0.0).all(), "rows before the news window must be flagged as no-data"
    assert (after == 1.0).all(), "rows inside the news window must be flagged as covered"
    # A quiet day *inside* the window is neutral, not missing -- the distinction
    # the column exists to make.
    assert (panel.loc[panel.index > covered_from, "Sentiment"] == 0.0).all()
    assert (panel.loc[panel.index > covered_from, "has_sentiment_data"] == 1.0).all()


def test_has_sentiment_data_is_absent_when_sentiment_is_disabled(
    monkeypatch, sample_ohlcv_df, settings
):
    """It must not appear when the source is off, or it would break the PYQ-118
    schema check for every bundle trained without sentiment."""
    _patch_prices(monkeypatch, sample_ohlcv_df)
    settings.data.use_sentiment = False

    panel = dataset.build_panel("AAPL", settings)

    assert "has_sentiment_data" not in panel.columns


def test_has_sentiment_data_is_a_model_feature(monkeypatch, sample_ohlcv_df, settings):
    """It is only useful if the model actually sees it."""
    _patch_prices(monkeypatch, sample_ohlcv_df)
    settings.data.use_sentiment = True
    settings.finnhub_api_key = "dummy"
    session = add_technical_indicators(sample_ohlcv_df).dropna().index[100]
    monkeypatch.setattr(sentiment, "_finbert", lambda: object())
    monkeypatch.setattr(
        sentiment,
        "fetch_news",
        lambda *a, **k: [{"headline": "n", "datetime": _epoch_et(session, 11)}],
    )
    monkeypatch.setattr(sentiment, "score_headlines", lambda h: [0.5])

    long = dataset.panel_to_long(dataset.build_panel("AAPL", settings), "AAPL")
    assert "has_sentiment_data" in dataset.feature_columns(long)
