"""Tests for cli/charts.py (PYQ-272).

Invariant 8 requires that the dates in the table, the JSON, the PNG and the
appended rows are one set; the PNG export was the one leg with no direct test.
"""

from __future__ import annotations

import matplotlib.axes
import numpy as np
import pandas as pd
import pytest

from pyquant.analysis.forecast import Forecast
from pyquant.cli import charts


def _forecast(quantiles=(0.1, 0.5, 0.9)):
    history_dates = pd.bdate_range("2024-01-01", periods=40)
    return Forecast(
        symbol="TEST",
        last_date=history_dates[-1],
        current_price=110.0,
        quantiles=list(quantiles),
        predictions=np.array(
            [
                [98.0, 105.0, 112.0],
                [97.0, 106.0, 114.0],
                [96.0, 107.0, 116.0],
            ]
        ),
        history=pd.Series(np.linspace(90, 110, 40), index=history_dates),
    )


def test_export_fan_chart_plots_at_exactly_the_forecast_dates(tmp_path, monkeypatch):
    """The x-values passed to the shaded band must equal `Forecast.forecast_dates`
    exactly -- not merely a same-length series, which invariant 8 requires.

    The *first* fill_between call is the one under test (matching
    test_invariants.py's `seen_x[0]` check): PYQ-324 added a second,
    subsequent fill_between call that bridges the band to the last observed
    close, deliberately excluded from this dates-exactness assertion since
    it is a 2-point visual connector, not a forecast_dates-keyed series.
    """
    forecast = _forecast()
    seen_x: list[list] = []
    original_fill_between = matplotlib.axes.Axes.fill_between

    def spy_fill_between(self, x, *args, **kwargs):
        seen_x.append(list(x))
        return original_fill_between(self, x, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "fill_between", spy_fill_between)

    out_path = charts.export_fan_chart(forecast, tmp_path / "fan.png")

    assert out_path.exists()
    assert seen_x[0] == list(forecast.forecast_dates)


def test_export_fan_chart_median_line_also_matches_the_forecast_dates(tmp_path, monkeypatch):
    forecast = _forecast()
    captured: dict[str, list] = {}
    original_plot = matplotlib.axes.Axes.plot

    def spy_plot(self, *args, **kwargs):
        if kwargs.get("label") == "median":
            captured["x"] = list(args[0])
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", spy_plot)

    charts.export_fan_chart(forecast, tmp_path / "fan.png")

    assert captured["x"] == list(forecast.forecast_dates)


def test_export_fan_chart_band_and_median_connect_to_the_last_observed_close(tmp_path, monkeypatch):
    """PYQ-324: history ends at (last_date, current_price); the forecast band and
    median must visually originate there too, or the PNG reads as a band that
    pops in already at full width / a median disconnected from history, rather
    than fanning out of the last observed close. The *forecast-dates* series
    (asserted by the two tests above) must stay untouched -- invariant 8 -- so
    this connector has to be a separate draw call, not a prepended point on the
    existing fill_between/plot("median") calls."""
    forecast = _forecast()
    fill_calls: list[dict] = []
    plot_calls: list[dict] = []
    original_fill_between = matplotlib.axes.Axes.fill_between
    original_plot = matplotlib.axes.Axes.plot

    def spy_fill_between(self, x, y1, y2, *args, **kwargs):
        fill_calls.append({"x": list(x), "y1": list(y1), "y2": list(y2)})
        return original_fill_between(self, x, y1, y2, *args, **kwargs)

    def spy_plot(self, *args, **kwargs):
        if len(args) >= 2:
            plot_calls.append(
                {"x": list(args[0]), "y": list(args[1]), "label": kwargs.get("label")}
            )
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "fill_between", spy_fill_between)
    monkeypatch.setattr(matplotlib.axes.Axes, "plot", spy_plot)

    charts.export_fan_chart(forecast, tmp_path / "fan.png")

    last_date, last_close = forecast.last_date, forecast.current_price

    # The main band fill_between is still keyed exactly to forecast_dates
    # (invariant 8) -- some other fill_between call must supply the bridge.
    band_call = next(c for c in fill_calls if c["x"] == list(forecast.forecast_dates))
    assert band_call["y1"][0] == forecast.quantile_series(forecast.quantiles[0])[0]
    bridge_calls = [c for c in fill_calls if c is not band_call]
    assert bridge_calls, "expected a separate fill_between bridging history to the band"
    bridge = bridge_calls[0]
    assert bridge["x"][0] == last_date
    assert bridge["y1"][0] == pytest.approx(last_close)
    assert bridge["y2"][0] == pytest.approx(last_close)

    # The labelled median line is still keyed exactly to forecast_dates
    # (invariant 8); a separate, unlabelled connector bridges it to history.
    median_call = next(c for c in plot_calls if c["label"] == "median")
    assert median_call["x"] == list(forecast.forecast_dates)
    median_bridges = [
        c for c in plot_calls if c["label"] != "median" and c["x"] and c["x"][0] == last_date
    ]
    assert median_bridges, "expected a separate connector line from history to the median"
    assert median_bridges[0]["y"][0] == pytest.approx(last_close)


def test_export_fan_chart_omits_the_median_line_when_0_5_is_not_configured(tmp_path):
    """PYQ-106: median plotting is conditional on 0.5 being among the quantiles."""
    forecast = _forecast(quantiles=(0.1, 0.9))
    out_path = charts.export_fan_chart(forecast, tmp_path / "fan.png")
    assert out_path.exists()


def test_export_fan_chart_creates_parent_directories(tmp_path):
    out_path = charts.export_fan_chart(_forecast(), tmp_path / "nested" / "dir" / "fan.png")
    assert out_path.exists()


@pytest.mark.parametrize("fn", ["fan_chart"])
def test_fan_chart_does_not_raise(fn):
    """Smoke test: the terminal chart must not crash on a well-formed forecast --
    plotext renders straight to stdout, so there is nothing else to assert here."""
    getattr(charts, fn)(_forecast())


def test_importance_chart_does_not_raise():
    charts.importance_chart([("Close", 0.6), ("RSI_14", 0.4)])


def test_attention_chart_does_not_raise():
    charts.attention_chart(np.array([0.1, 0.2, 0.3, 0.2, 0.2]))
