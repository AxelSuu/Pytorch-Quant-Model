"""CLI smoke tests using Typer's CliRunner (network-free, mocked forecasts)."""

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from pyquant.analysis.forecast import Forecast
from pyquant.cli import app as app_mod

runner = CliRunner()


def _fake_forecast(symbol="AAPL"):
    dates = pd.bdate_range("2024-01-01", periods=20)
    return Forecast(
        symbol=symbol,
        last_date=dates[-1],
        current_price=100.0,
        quantiles=[0.1, 0.5, 0.9],
        predictions=np.array([[95.0, 105.0, 115.0]] * 5),
        history=pd.Series(np.linspace(90, 100, 20), index=dates),
    )


def test_forecast_command(monkeypatch):
    monkeypatch.setattr(app_mod, "generate_forecast", lambda *a, **k: _fake_forecast())

    class NoOptions:
        put_call_ratio = None

    monkeypatch.setattr(app_mod, "fetch_options_snapshot", lambda s: NoOptions())
    result = runner.invoke(app_mod.app, ["forecast", "AAPL", "--no-chart"])
    assert result.exit_code == 0
    assert "5-day forecast" in result.stdout
    assert "$105.00" in result.stdout  # median day value


def test_scan_handles_untrained(monkeypatch):
    def raise_missing(symbol, settings):
        raise FileNotFoundError("no model")

    monkeypatch.setattr(app_mod, "generate_forecast", raise_missing)
    result = runner.invoke(app_mod.app, ["scan", "AAPL,MSFT"])
    assert result.exit_code == 0
    assert "not trained" in result.stdout


def test_scan_signal_buy(monkeypatch):
    monkeypatch.setattr(app_mod, "generate_forecast", lambda *a, **k: _fake_forecast())
    result = runner.invoke(app_mod.app, ["scan", "AAPL"])
    assert result.exit_code == 0
    # median 105 vs current 100 -> +5% -> BUY
    assert "BUY" in result.stdout
