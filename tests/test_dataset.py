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
