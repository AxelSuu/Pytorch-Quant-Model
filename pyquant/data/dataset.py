"""Assemble enrichment sources into a unified panel and a TimeSeriesDataSet.

Flow:
    build_panel()      -> wide, date-indexed DataFrame (prices + enrichments)
    panel_to_long()    -> long DataFrame with time_idx / symbol / calendar cols
    make_dataset()     -> pytorch_forecasting TimeSeriesDataSet

Every enrichment join is guarded: a source that returns nothing (missing key,
network error, disabled in config) is skipped, and the feature schema is derived
from the columns that actually materialised. This is how graceful degradation
propagates all the way to the model.
"""

from __future__ import annotations

import logging

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

from pyquant.config import Settings
from pyquant.data.macro import fetch_macro
from pyquant.data.prices import fetch_prices
from pyquant.data.sectors import fetch_sector_returns
from pyquant.data.sentiment import fetch_sentiment

logger = logging.getLogger(__name__)

TARGET = "Close"
# Columns that are identifiers/calendar, never model feature reals.
_NON_FEATURE = {"Date", "time_idx", "symbol", "day_of_week", "month"}
# Known-in-future calendar reals.
KNOWN_REALS = ["time_idx", "dow", "month_num"]


def build_panel(
    symbol: str,
    settings: Settings,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch + join all enabled data sources into one date-indexed panel."""
    period = settings.data.period
    panel = fetch_prices(symbol, period=period, start=start, end=end, use_indicators=True)
    price_index = panel.index

    if settings.data.use_macro:
        macro = fetch_macro(settings.fred_api_key, start=start, end=end, period=period)
        if not macro.empty:
            panel = panel.join(macro.reindex(price_index, method="ffill"))
            logger.info("Joined macro features: %s", list(macro.columns))

    if settings.data.use_sectors:
        sectors = fetch_sector_returns(settings.data.sector_etfs, start, end, period)
        # Drop the target symbol's own ETF column if present to avoid leakage of itself.
        if not sectors.empty:
            panel = panel.join(sectors.reindex(price_index))
            logger.info("Joined sector features: %s", list(sectors.columns))

    if settings.data.use_sentiment:
        sentiment = fetch_sentiment(settings.finnhub_api_key, symbol, start=start, end=end)
        if not sentiment.empty:
            panel = panel.join(sentiment.reindex(price_index))
            # Days without news are neutral (0 sentiment, 0 headlines).
            panel[["Sentiment", "HeadlineCount"]] = panel[
                ["Sentiment", "HeadlineCount"]
            ].fillna(0.0)
            logger.info("Joined sentiment features")

    # Forward/back fill any remaining gaps from joined sources, then drop
    # leading rows still NaN from indicator warm-up.
    panel = panel.ffill().bfill()
    panel = panel.dropna()
    return panel


def panel_to_long(panel: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Convert a wide panel to the long format TimeSeriesDataSet expects."""
    df = panel.reset_index().sort_values("Date").reset_index(drop=True)
    df["time_idx"] = range(len(df))
    df["symbol"] = symbol
    # Calendar features (known in the future).
    df["dow"] = df["Date"].dt.dayofweek.astype(float)
    df["month_num"] = df["Date"].dt.month.astype(float)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Dynamic real feature columns present in the long df (excludes target)."""
    return [
        c
        for c in df.columns
        if c not in _NON_FEATURE
        and c not in KNOWN_REALS
        and c != TARGET
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def make_dataset(
    df: pd.DataFrame,
    settings: Settings,
    *,
    training_cutoff: int | None = None,
) -> TimeSeriesDataSet:
    """Build a TimeSeriesDataSet for training from a long df.

    If ``training_cutoff`` is given, only rows with ``time_idx <= cutoff`` are
    used (the rest are held out for validation/prediction).
    """
    unknown_reals = feature_columns(df)
    data = df if training_cutoff is None else df[df["time_idx"] <= training_cutoff]

    return TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target=TARGET,
        group_ids=["symbol"],
        max_encoder_length=settings.training.max_encoder_length,
        max_prediction_length=settings.training.max_prediction_length,
        static_categoricals=["symbol"],
        time_varying_known_reals=KNOWN_REALS,
        time_varying_unknown_reals=[TARGET, *unknown_reals],
        target_normalizer=GroupNormalizer(groups=["symbol"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
