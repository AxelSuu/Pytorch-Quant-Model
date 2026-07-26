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
from pyquant.data import cache
from pyquant.data.macro import fetch_macro
from pyquant.data.prices import fetch_prices
from pyquant.data.sectors import fetch_sector_returns
from pyquant.data.sentiment import fetch_sentiment

logger = logging.getLogger(__name__)

TARGET = "Close"
# Columns that are identifiers, never model feature reals.
_NON_FEATURE = {"Date", "time_idx", "symbol"}
# Known-in-future calendar reals.
KNOWN_REALS = ["time_idx", "dow", "month_num"]


def _cache_fingerprint(symbol: str, settings: Settings, start: str | None, end: str | None) -> dict:
    """What a cached panel's validity depends on -- change any of these, different dataset."""
    data = settings.data
    return {
        "symbol": symbol.upper(),
        "start": start,
        "end": end,
        "period": data.period,
        "use_macro": data.use_macro,
        "use_sectors": data.use_sectors,
        "use_sentiment": data.use_sentiment,
        "sector_etfs": sorted(data.sector_etfs),
        # Key *presence*, not the secret value, is part of what changes the data.
        "has_fred_key": bool(settings.fred_api_key),
        "has_finnhub_key": bool(settings.finnhub_api_key),
    }


def build_panel(
    symbol: str,
    settings: Settings,
    start: str | None = None,
    end: str | None = None,
    pin: str | None = None,
) -> pd.DataFrame:
    """Fetch + join all enabled data sources into one date-indexed panel.

    ``pin``, if given, names a reproducible dataset snapshot: the first call
    fetches and saves it; every later call with the same pin replays that
    exact data, ignoring both live changes and the TTL cache below.
    """
    cache_dir = settings.data.cache_dir
    cache_key = cache.fingerprint_key(_cache_fingerprint(symbol, settings, start, end))
    if pin:
        cached = cache.read_pin(cache_dir, f"{symbol.upper()}_{pin}")
        if cached is not None:
            return cached
    elif settings.data.cache_enabled:
        cached = cache.read_cache(cache_dir, cache_key, ttl_seconds=settings.data.cache_ttl_seconds)
        if cached is not None:
            return cached

    period = settings.data.period
    panel = fetch_prices(symbol, period=period, start=start, end=end, use_indicators=True)
    # Drop leading rows still NaN from indicator warm-up (e.g. SMA_50 needs
    # 49 days of history) before joining other sources, so their own
    # fill/reindex logic never launders these into fabricated values.
    panel = panel.dropna()
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
            sectors = sectors.drop(columns=[f"SEC_{symbol.upper()}"], errors="ignore")
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

    # Forward-fill gaps from joined sources (e.g. a sector ETF's trading calendar
    # not perfectly matching the target's). Rows *before* a joined source's first
    # observation cannot be filled that way, and the bfill that used to follow
    # filled them from the first *later* value -- look-ahead, the same class of
    # leak as PYQ-101. Drop them instead, which is the policy indicator warm-up
    # rows already get (PYQ-123).
    panel = panel.ffill()
    empty_cols = [c for c in panel.columns if panel[c].isna().all()]
    if empty_cols:
        # No overlap at all with the price calendar: dropping the rows would empty
        # the panel, so drop the useless columns instead and say so.
        logger.warning(
            "Dropping column(s) with no data overlapping the price history: %s", empty_cols
        )
        panel = panel.drop(columns=empty_cols)
    panel = panel.dropna()

    if pin:
        cache.write_pin(cache_dir, f"{symbol.upper()}_{pin}", panel)
    elif settings.data.cache_enabled:
        cache.write_cache(cache_dir, cache_key, panel)

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


def align_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """Re-map ``time_idx`` onto one calendar shared by every symbol.

    panel_to_long() numbers each symbol's rows from zero independently, so
    ``time_idx = t`` means a *different calendar date* for each symbol. That has
    two consequences for pooled training (PYQ-116):

    - Groups are aligned by position rather than by date, so a shared market
      shock lands at a different index in every group and cannot be learned
      cross-sectionally -- which is most of the point of pooling.
    - train() derives ``cutoff`` from the global maximum ``time_idx``, so a
      symbol with less history has its entire series -- including the window
      ``predict=True`` later hands back as *validation* -- fall inside the
      training slice, silently corrupting val_loss and therefore early stopping
      and checkpoint selection.

    Mapping every row onto the union calendar fixes both. ``make_dataset`` sets
    ``allow_missing_timesteps=True``, which absorbs the per-symbol gaps this
    creates (a symbol that did not trade on a day another did).
    """
    calendar = pd.DatetimeIndex(sorted(pd.unique(df["Date"])))
    positions = pd.Series(range(len(calendar)), index=calendar)
    df = df.copy()
    df["time_idx"] = df["Date"].map(positions).astype(int)
    return df


def future_business_dates(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    """The ``horizon`` business days that follow ``last_date``.

    Single source of truth for "what dates is the forecast for": the rows
    appended by extend_for_prediction(), the CLI's forecast table, the terminal
    chart and the PNG export all label the same days.
    """
    return pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)


def extend_for_prediction(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Append ``horizon`` future rows per symbol so a prediction decoder covers the future.

    ``predict=True`` anchors a TimeSeriesDataSet's decoder to the *last*
    ``max_prediction_length`` rows of whatever frame it is given. Handed a frame
    that stops at the last observed bar it therefore re-predicts days that
    already happened instead of forecasting (PYQ-115). Appending the future rows
    moves the decoder onto them and leaves the encoder on the last real
    observations.

    The last observed row is carried forward for every column so nothing is NaN;
    only ``time_idx`` and the calendar features -- the known-in-future reals --
    are recomputed. The carried-forward unknown reals, target included, are
    never read by the model: they land in the decoder, which by construction
    sees only ``time_varying_known_reals`` plus statics.
    """
    if horizon <= 0:
        return df

    extended: list[pd.DataFrame] = []
    for _, group in df.groupby("symbol", sort=False):
        group = group.sort_values("time_idx")
        last_idx = int(group["time_idx"].iloc[-1])
        future = pd.concat([group.tail(1)] * horizon, ignore_index=True)
        future["Date"] = future_business_dates(group["Date"].iloc[-1], horizon)
        future["time_idx"] = range(last_idx + 1, last_idx + 1 + horizon)
        future["dow"] = future["Date"].dt.dayofweek.astype(float)
        future["month_num"] = future["Date"].dt.month.astype(float)
        extended.append(pd.concat([group, future], ignore_index=True))
    return pd.concat(extended, ignore_index=True)


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
