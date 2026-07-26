"""Tests for panel assembly and TimeSeriesDataSet construction."""

import pandas as pd

from pyquant.data import dataset
from pyquant.data.prices import add_technical_indicators


def _patch_prices(monkeypatch, df):
    monkeypatch.setattr(
        dataset, "fetch_prices", lambda *a, **k: add_technical_indicators(df)
    )


def test_build_panel_baseline_ohlcv(monkeypatch, sample_ohlcv_df, settings):
    _patch_prices(monkeypatch, sample_ohlcv_df)
    panel = dataset.build_panel("AAPL", settings)
    assert "Close" in panel.columns
    assert "RSI_14" in panel.columns
    assert not panel.isna().any().any()


def test_build_panel_drops_indicator_warmup_rows(monkeypatch, sample_ohlcv_df, settings):
    """SMA_50 needs 49 real days of history; those rows must be dropped,
    not silently fabricated via bfill (which borrows the first valid value)."""
    _patch_prices(monkeypatch, sample_ohlcv_df)
    panel = dataset.build_panel("AAPL", settings)
    assert panel.index[0] == sample_ohlcv_df.index[49]
    assert len(panel) == len(sample_ohlcv_df) - 49


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


def test_feature_columns_excludes_identifiers_and_target(monkeypatch, sample_ohlcv_df, settings):
    _patch_prices(monkeypatch, sample_ohlcv_df)
    panel = dataset.build_panel("AAPL", settings)
    long = dataset.panel_to_long(panel, "AAPL")
    feats = dataset.feature_columns(long)
    assert "Close" not in feats  # target excluded
    assert "symbol" not in feats
    assert "time_idx" not in feats
    assert "RSI_14" in feats


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


# --- PYQ-115: prediction rows must cover the future, not observed days -------


def test_future_business_dates_starts_after_last_observed_date():
    dates = dataset.future_business_dates(pd.Timestamp("2024-10-07"), 5)
    assert len(dates) == 5
    assert dates[0] > pd.Timestamp("2024-10-07")
    # Business days only -- no weekend rows.
    assert all(d.dayofweek < 5 for d in dates)


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
