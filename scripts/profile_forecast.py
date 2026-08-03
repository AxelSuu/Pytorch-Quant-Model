"""Profile a real `forecast`/`explain` call, broken down by phase (PYQ-319).

Answers the question `docs/api-design.md`'s concurrency design needed a number for and
didn't have one: where does the time in one `forecast` call actually go, and is the API's
scaling constraint vendor rate limits/cache hit rate, or GPU/model throughput?

Usage:
    uv run python scripts/profile_forecast.py [SYMBOL]

Runs against live vendors (needs network + FRED_API_KEY/FINNHUB_API_KEY for the macro/
sentiment phases to be representative; both degrade gracefully like the rest of the
pipeline if absent, same as normal operation). Trains one small real bundle first (untimed
setup), then times a cold call (panel cache disabled) and a warm call (cache enabled,
second call reuses it), breaking each into bundle-load / fetch+panel-build / predict --
plus `interpret()`'s own raw-mode predict, the extra cost `explain` pays on top of
`forecast`. Also counts outbound requests per vendor call site.
"""

from __future__ import annotations

import sys
import time
import warnings
from contextlib import contextmanager

warnings.filterwarnings("ignore")


@contextmanager
def _timer(label: str, results: dict):
    start = time.perf_counter()
    yield
    results[label] = time.perf_counter() - start


def _count_calls(monkeypatch, counts: dict) -> None:
    """Wrap every vendor call site so one profiled call's request count is visible."""
    import fredapi
    import requests
    import yfinance as yf

    real_history = yf.Ticker.history
    real_download = yf.download
    real_get_releases = fredapi.Fred.get_series_all_releases
    real_requests_get = requests.get

    def counted_history(self, *a, **k):
        counts["yfinance.Ticker.history"] = counts.get("yfinance.Ticker.history", 0) + 1
        return real_history(self, *a, **k)

    def counted_download(*a, **k):
        counts["yfinance.download"] = counts.get("yfinance.download", 0) + 1
        return real_download(*a, **k)

    def counted_releases(self, *a, **k):
        counts["fredapi.get_series_all_releases"] = (
            counts.get("fredapi.get_series_all_releases", 0) + 1
        )
        return real_get_releases(self, *a, **k)

    def counted_get(*a, **k):
        counts["requests.get"] = counts.get("requests.get", 0) + 1
        return real_requests_get(*a, **k)

    monkeypatch.setattr(yf.Ticker, "history", counted_history)
    monkeypatch.setattr(yf, "download", counted_download)
    monkeypatch.setattr(fredapi.Fred, "get_series_all_releases", counted_releases)
    monkeypatch.setattr(requests, "get", counted_get)


def _profile_call(label: str, symbol: str, settings, monkeypatch) -> dict:
    """One forecast-shaped call, phase-timed. Returns {"timings": ..., "requests": ...}."""
    from pyquant.data.dataset import build_panel, panel_to_long
    from pyquant.models import tft

    counts: dict = {}
    _count_calls(monkeypatch, counts)

    timings: dict = {}
    with _timer("bundle_load", timings):
        bundle = tft.load(symbol, settings)
    with _timer("fetch_and_panel_build", timings):
        panel = build_panel(symbol, settings)
    with _timer("panel_to_long", timings):
        df = panel_to_long(panel, symbol)
    with _timer("predict (forecast)", timings):
        tft.predict_quantiles(bundle, df)
    with _timer("interpret_raw_predict_explain_extra_cost", timings):
        tft.interpret(bundle, df)

    print(f"\n--- {label} ---")
    for phase, seconds in timings.items():
        print(f"  {phase:42s} {seconds * 1000:8.1f} ms")
    print(f"  requests: {counts or '(none -- served from cache)'}")
    return {"timings": timings, "requests": counts}


def main() -> None:
    import pytest

    from pyquant.config import Settings
    from pyquant.models import tft

    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"

    settings = Settings()
    settings.training.max_encoder_length = 20
    settings.training.max_prediction_length = 5
    settings.training.max_epochs = 2
    settings.tft.hidden_size = 8
    settings.tft.hidden_continuous_size = 4

    print(f"Training a small real bundle for {symbol} (untimed setup)...")
    settings.data.cache_enabled = False
    train_start = time.perf_counter()
    tft.train(symbol, settings, progress=False)
    print(f"  done in {time.perf_counter() - train_start:.1f}s")

    with pytest.MonkeyPatch.context() as mp:
        settings.data.cache_enabled = False
        cold = _profile_call("COLD (panel cache disabled)", symbol, settings, mp)

    with pytest.MonkeyPatch.context() as mp:
        settings.data.cache_enabled = True
        _profile_call("(priming the panel cache, not reported)", symbol, settings, mp)
        warm = _profile_call("WARM (panel cache enabled, second call)", symbol, settings, mp)

    print("\n=== Summary ===")
    for label, result in (("cold", cold), ("warm", warm)):
        t = result["timings"]
        forecast_total = t["bundle_load"] + t["fetch_and_panel_build"] + t["predict (forecast)"]
        fetch_share = t["fetch_and_panel_build"] / forecast_total if forecast_total else 0.0
        print(
            f"{label:5s}: forecast total={forecast_total * 1000:7.0f} ms  "
            f"(fetch+panel-build is {fetch_share:.0%} of it)   "
            f"explain adds +{t['interpret_raw_predict_explain_extra_cost'] * 1000:.0f} ms"
        )
    print(f"\nrequests per cold call: {cold['requests']}")
    print(f"requests per warm call: {warm['requests']}")


if __name__ == "__main__":
    main()
