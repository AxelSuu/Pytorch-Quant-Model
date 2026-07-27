"""PYQ-238: the pipeline-spanning invariants, asserted directly.

``docs/invariants.md`` states the lesson from the third review pass: every one of six
(now seven) look-ahead leaks (PYQ-101, 103, 115, 116, 123, 127, 129) was correct in each
individual file and wrong across files. Each was then guarded by a regression test scoped
to the file that broke -- which stops that exact recurrence but not the next member of the
family. This module is the consolidated, structural form: one named test per invariant,
built over a synthetic multi-symbol panel with deliberately unequal history, so the
property is checked directly rather than inferred from a single historical bug.

Every test here was verified, while it was being written, to actually fail against the
pre-fix shape of the logic it guards -- not just pass against the current code, which is
the trap PYQ-120's coverage gap illustrates (a test that cannot fail is worse than none).
See the PYQ-238 resolution note in backlog/features.md for the specific verifications.

Numbering matches backlog/features.md#pyq-238's own list of 9 invariants; a couple are
split across two test functions where the ticket's single invariant actually covers two
independently-breakable mechanisms (same convention docs/invariants.md itself already uses
for the leak family).
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from pyquant.analysis import forecast as fc_mod
from pyquant.analysis.serialize import forecast_to_dict
from pyquant.data import dataset as ds_mod
from pyquant.data import sentiment as sentiment_mod
from pyquant.data.dataset import (
    align_time_index,
    build_panel,
    extend_for_prediction,
    make_dataset,
    panel_to_long,
)
from pyquant.data.prices import INDICATOR_COLUMNS, add_technical_indicators
from pyquant.models import tft

# --- shared fixtures -----------------------------------------------------------


@pytest.fixture
def unequal_history_panels(sample_ohlcv_df):
    """Two symbols on the same calendar, one much shorter than the other.

    LONG is the full 400-day synthetic history; SHORT is its last 90 days -- same
    trading calendar, genuinely different start. This is the exact shape PYQ-116 got
    wrong: a shorter symbol's whole series, including what should be its own
    validation window, fell inside the pooled training slice.
    """
    long_panel = add_technical_indicators(sample_ohlcv_df).dropna()
    short_panel = long_panel.tail(90)
    return {"LONG": long_panel, "SHORT": short_panel}


@pytest.fixture
def invariants_settings(tmp_path, settings):
    settings.checkpoint_dir = tmp_path / "checkpoints"
    settings.training.max_encoder_length = 20
    settings.training.max_prediction_length = 5
    settings.tft.hidden_size = 8
    settings.tft.hidden_continuous_size = 4
    return settings


# --- 1. No future information in any training row -------------------------------


def test_a_publication_lagged_source_never_appears_before_its_own_release_date(
    monkeypatch, sample_ohlcv_df, invariants_settings
):
    """Invariant 1, PYQ-101 shape: a joined source's value at row t must have been
    knowable at or before t. A macro-style series releases every 7th day, and its
    value at each release *is* that release day's own ordinal -- so "the value seen
    at row t exceeds t's own ordinal" is a direct, generic leak detector that does
    not depend on knowing macro.py's internals.
    """
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    price_index = panel.index
    release_days = price_index[::7]
    macro = pd.DataFrame(
        {"KnownAsOfOrdinal": [float(d.toordinal()) for d in release_days]}, index=release_days
    )

    monkeypatch.setattr(ds_mod, "fetch_prices", lambda *a, **k: panel)
    monkeypatch.setattr(ds_mod, "fetch_macro", lambda *a, **k: macro)
    invariants_settings.data.use_macro = True

    built = build_panel("TEST", invariants_settings)
    row_ordinals = pd.Series([float(d.toordinal()) for d in built.index], index=built.index)
    assert (built["KnownAsOfOrdinal"] <= row_ordinals).all(), (
        "a row carries a macro value released after that row's own date"
    )


def test_a_late_starting_source_drops_leading_rows_instead_of_backfilling(
    monkeypatch, sample_ohlcv_df, invariants_settings
):
    """Invariant 1, PYQ-123 shape: a source with no data before day N must not leak
    its first value backwards onto the rows before day N -- those rows must be
    absent from the panel entirely, never filled from a later observation.
    """
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    price_index = panel.index
    late_start = price_index[100]
    sectors = pd.DataFrame({"SEC_XLK": np.full(len(price_index) - 100, 7.0)}, index=price_index[100:])

    monkeypatch.setattr(ds_mod, "fetch_prices", lambda *a, **k: panel)
    monkeypatch.setattr(ds_mod, "fetch_sector_returns", lambda *a, **k: sectors)
    invariants_settings.data.use_sectors = True

    built = build_panel("TEST", invariants_settings)
    assert built.index.min() >= late_start, (
        f"panel starts {built.index.min().date()}, before the source's first observation "
        f"{late_start.date()} -- leading rows were backfilled instead of dropped"
    )


def test_a_post_close_headline_never_lands_on_a_session_it_could_not_trade_on(
    monkeypatch, sample_ohlcv_df, invariants_settings
):
    """Invariant 1, PYQ-129 shape: a headline published after the exchange close must
    land on the *next* session's row, never the session that already closed before
    it existed. Exercises the real fetch_sentiment -> session_date -> align_to_sessions
    chain (only the network call and the FinBERT model are stubbed), not a
    re-implementation of the rule.
    """
    panel = add_technical_indicators(sample_ohlcv_df).dropna()
    price_index = panel.index

    i = 100
    while price_index[i].weekday() > 3:  # need D and D+1 both to be trading days
        i += 1
    published_on = price_index[i]
    next_session = price_index[i + 1]
    assert next_session == published_on + pd.Timedelta(days=1)

    tz = ZoneInfo("America/New_York")
    pre_close = dt.datetime(published_on.year, published_on.month, published_on.day, 11, 0, tzinfo=tz)
    post_close = dt.datetime(published_on.year, published_on.month, published_on.day, 17, 0, tzinfo=tz)
    articles = [
        {"datetime": pre_close.timestamp(), "headline": "pre-close headline"},
        {"datetime": post_close.timestamp(), "headline": "post-close headline"},
    ]

    monkeypatch.setattr(ds_mod, "fetch_prices", lambda *a, **k: panel)
    monkeypatch.setattr(sentiment_mod, "fetch_news", lambda *a, **k: articles)
    monkeypatch.setattr(sentiment_mod, "_finbert", lambda: object())
    monkeypatch.setattr(
        sentiment_mod, "score_headlines", lambda hs: [1.0 if "pre" in h else 9.0 for h in hs]
    )
    invariants_settings.data.use_sentiment = True
    invariants_settings.finnhub_api_key = "test-key"

    built = build_panel("TEST", invariants_settings)
    assert built.loc[published_on, "Sentiment"] == 1.0, "pre-close headline did not land on its own session"
    assert built.loc[next_session, "Sentiment"] == 9.0, "post-close headline leaked onto the closed session"


# --- 2. Warm-up rows never carry fabricated values -------------------------------


def test_indicator_warmup_rows_are_dropped_not_fabricated(monkeypatch, sample_ohlcv_df, invariants_settings):
    """Invariant 2, PYQ-103/132 shape: the panel's first surviving row must be
    determined by whichever indicator's window is longest -- not hardcoded to
    SMA_50 by name, since PYQ-121 and PYQ-132 both hid behind exactly that
    hardcoding -- and no indicator may be a constant across the leading rows,
    which is what a fabricated fill would look like.
    """
    raw = add_technical_indicators(sample_ohlcv_df)  # deliberately not dropna'd
    monkeypatch.setattr(ds_mod, "fetch_prices", lambda *a, **k: raw)

    built = build_panel("TEST", invariants_settings)

    first_fully_valid = raw[INDICATOR_COLUMNS].dropna().index[0]
    assert built.index.min() == first_fully_valid

    leading = built.head(10)
    for col in INDICATOR_COLUMNS:
        assert leading[col].nunique() > 1, f"{col} is constant across the leading rows"


# --- 3/4. Prediction decodes the future, from an encoder ending on the last bar --


def test_prediction_decoder_starts_after_and_encoder_ends_on_the_last_observed_bar(
    monkeypatch, unequal_history_panels, invariants_settings
):
    """Invariants 3 and 4, both PYQ-115: predict=True must decode strictly future
    timesteps (3), and the encoder feeding that decode must end exactly on the last
    real observation (4) -- stated as two checks because they can break
    independently: appending too many future rows would satisfy 3 while breaking 4.
    """
    panel = unequal_history_panels["LONG"]
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)
    tft.train("TEST", invariants_settings, max_epochs=1, progress=False)
    bundle = tft.load("TEST", invariants_settings)

    df = panel_to_long(panel, "TEST")
    observed_max = int(df["time_idx"].max())
    ds = tft._prediction_dataset(bundle, df)
    x, _ = next(iter(ds.to_dataloader(train=False, batch_size=1, num_workers=0)))
    decoder_start = int(x["decoder_time_idx"].min())

    assert decoder_start > observed_max, "decoder is not strictly after the last observed bar"
    assert decoder_start == observed_max + 1, "encoder does not end exactly on the last observed bar"


# --- 5/6. One calendar across pooled symbols; validation strictly after cutoff ---


def test_pooled_symbols_share_one_calendar(monkeypatch, unequal_history_panels, invariants_settings):
    """Invariant 5, PYQ-116: the same calendar Date must map to the same time_idx for
    every pooled symbol, or a shared market shock lands at a different position per
    group and cross-sectional learning is impossible by construction.
    """
    monkeypatch.setattr(tft, "build_panel", lambda symbol, *a, **k: unequal_history_panels[symbol])
    df = tft._build_pooled_long_df(["LONG", "SHORT"], invariants_settings, None, None)
    per_date = df.groupby("Date")["time_idx"].nunique()
    assert (per_date == 1).all(), "at least one Date maps to more than one time_idx across symbols"


def test_every_pooled_symbols_validation_window_is_strictly_after_the_training_cutoff(
    monkeypatch, unequal_history_panels, invariants_settings
):
    """Invariant 6, PYQ-116/117: true for *every* group, including the shortest --
    with per-position (not per-date) indices, SHORT's entire series used to sit
    inside the training slice, corrupting val_loss, EarlyStopping and checkpoint
    selection. "True on average" or "true for the longest symbol" is not the claim.
    """
    monkeypatch.setattr(tft, "build_panel", lambda symbol, *a, **k: unequal_history_panels[symbol])
    df = tft._build_pooled_long_df(["LONG", "SHORT"], invariants_settings, None, None)

    horizon = invariants_settings.training.max_prediction_length
    cutoff = int(df["time_idx"].max()) - horizon
    last_per_symbol = df.groupby("symbol")["time_idx"].max()
    assert (last_per_symbol > cutoff).all(), (
        f"cutoff={cutoff}, per-symbol max time_idx={last_per_symbol.to_dict()}"
    )
    shortest = last_per_symbol.idxmin()
    assert last_per_symbol[shortest] > cutoff, f"{shortest} (shortest history) fails the invariant"


# --- 7. The walk-forward walks ---------------------------------------------------


def test_walk_forward_origins_evaluate_disjoint_windows_starting_at_cutoff_plus_one(
    unequal_history_panels, invariants_settings
):
    """Invariant 7, PYQ-127: each rolling origin must decode its own out-of-sample
    window. Before the fix, every origin evaluated the identical final window
    because predict=True always anchors to the frame's last horizon steps.
    """
    panel = unequal_history_panels["LONG"]
    df = align_time_index(panel_to_long(panel, "TEST"))
    horizon = invariants_settings.training.max_prediction_length

    windows = []
    for cutoff in (150, 170, 190):
        training = make_dataset(df, invariants_settings, training_cutoff=cutoff)
        window_ds = tft._window_validation_dataset(training, df, cutoff, horizon)
        x, _ = next(iter(window_ds.to_dataloader(train=False, batch_size=1, num_workers=0)))
        decoded = x["decoder_time_idx"][0].tolist()
        assert decoded[0] == cutoff + 1, f"origin {cutoff} decoded {decoded}"
        windows.append(tuple(decoded))

    assert len(set(windows)) == len(windows), f"origins collapsed onto shared windows: {windows}"


# --- 8. Forecast dates are the same set everywhere they're consumed -------------


def test_forecast_dates_are_the_same_set_in_the_object_json_and_chart(
    monkeypatch, unequal_history_panels, invariants_settings
):
    """Invariant 8, PYQ-115/130: the table, the JSON payload, the chart and the
    appended prediction rows must all be labelled with the same dates -- four
    consumers that used to derive "which days is this for" independently. The chart
    is exercised for real (matplotlib's Agg backend, no display) with a spy on the
    x-values it actually plots, rather than trusting that it merely reads the same
    property by inspection.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.axes

    panel = unequal_history_panels["LONG"]
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)
    tft.train("TEST", invariants_settings, max_epochs=1, progress=False)

    monkeypatch.setattr(fc_mod, "build_panel", lambda *a, **k: panel)
    fc = fc_mod.generate_forecast("TEST", invariants_settings)

    horizon = invariants_settings.training.max_prediction_length
    extended = extend_for_prediction(panel_to_long(panel, "TEST"), horizon)
    appended_dates = list(pd.DatetimeIndex(extended.sort_values("time_idx").tail(horizon)["Date"]))
    assert list(fc.forecast_dates) == appended_dates

    payload = forecast_to_dict(fc)
    assert payload["forecast_dates"] == [d.date().isoformat() for d in fc.forecast_dates]

    seen_x: list[list] = []
    original_fill_between = matplotlib.axes.Axes.fill_between

    def _spy_fill_between(self, x, *a, **k):
        seen_x.append(list(x))
        return original_fill_between(self, x, *a, **k)

    monkeypatch.setattr(matplotlib.axes.Axes, "fill_between", _spy_fill_between)
    from pyquant.cli.charts import export_fan_chart

    export_fan_chart(fc, invariants_settings.checkpoint_dir / "fan.png")
    assert seen_x and seen_x[0] == appended_dates, "chart plotted different dates than the forecast object"


# --- 9. The band is monotone wherever it is consumed -----------------------------


def test_forecast_band_is_monotone_even_when_the_model_output_is_crossed():
    """Invariant 9, PYQ-124: QuantileLoss does not enforce monotonicity pointwise, so
    a p90 can land below a p10. No Forecast may exist in that state however it was
    built -- scan's "whole band on one side of zero" guard would otherwise be able
    to read an inverted band as a confident BUY.
    """
    quantiles = [0.1, 0.5, 0.9]
    horizon = 3
    crossed = np.stack(  # deliberately inverted: p10 > p50 > p90 at every step
        [np.full(horizon, 5.0), np.full(horizon, 3.0), np.full(horizon, 1.0)], axis=-1
    )
    history = pd.Series([100.0, 101.0], index=pd.bdate_range("2024-01-01", periods=2))

    fc = fc_mod.Forecast(
        symbol="TEST",
        last_date=history.index[-1],
        current_price=101.0,
        quantiles=quantiles,
        predictions=crossed,
        history=history,
    )

    assert fc.n_quantile_crossings > 0
    assert (np.diff(fc.predictions, axis=-1) >= 0).all(), "band is not monotone after construction"
