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


def test_feature_columns_excludes_identifiers_and_target(monkeypatch, sample_ohlcv_df, settings):
    _patch_prices(monkeypatch, sample_ohlcv_df)
    panel = dataset.build_panel("AAPL", settings)
    long = dataset.panel_to_long(panel, "AAPL")
    feats = dataset.feature_columns(long)
    assert "Close" not in feats  # target excluded
    assert "symbol" not in feats
    assert "time_idx" not in feats
    assert "RSI_14" in feats


def test_make_dataset_builds_timeseries_dataset(monkeypatch, sample_ohlcv_df, settings):
    _patch_prices(monkeypatch, sample_ohlcv_df)
    panel = dataset.build_panel("AAPL", settings)
    long = dataset.panel_to_long(panel, "AAPL")
    ds = dataset.make_dataset(long, settings)
    # The dataset should yield at least one sample.
    assert len(ds) > 0
    params = ds.get_parameters()
    assert params["max_encoder_length"] == settings.training.max_encoder_length
