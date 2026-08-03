"""Record real vendor payloads as checked-in fixtures (PYQ-243).

One-off, run manually against live vendors when a fixture needs refreshing (a
vendor changed its response shape and the contract tests below caught it, or
enough time has passed to want a fresh sample). Not part of the test suite --
the tests read the files this script writes, offline.

Usage:
    uv run python scripts/record_fixtures.py

Requires network access and FRED_API_KEY / FINNHUB_API_KEY in the environment
for the FRED and Finnhub fixtures; the yfinance ones need only network. A
source whose key is absent is skipped with a message rather than failing the
whole run.

Finnhub headline/summary/url/image text is replaced with placeholders before
writing -- the contract tests care about the response *shape* (which fields
exist, their types), not the copyrighted article text, so there is no reason
to redistribute the latter in a public fixture.

**Failure-mode fixtures are not recorded by this script, by design (PYQ-273).**
A recorded payload captures the shape a vendor produces when a call succeeds;
the four failure shapes that have actually cost this project (bugs.md#pyq-139's
three FRED cases, bugs.md#pyq-140's Finnhub truncation) only appear under
conditions a fresh recording won't reliably reproduce (a market-holiday NaT, an
unbounded realtime window, a caller's clock running ahead of FRED's). Two
different fixture strategies cover them instead of a third recording mode here:

- The three FRED shapes (tests/test_macro.py, "PYQ-139" section) are hand-built
  inline in the test itself, not as a file in tests/fixtures/. `fredapi.Fred`
  is the boundary those tests mock (see the module docstring on
  `test_fetch_macro_parses_real_recorded_vix_and_fred_payloads`) and it hands
  back a `pandas.DataFrame`, not raw bytes -- a `RecordingFred`/`FakeFred`
  class returning a hand-built frame with a `NaT` value, or an unset/oversized
  realtime window, already *is* the natural fixture for this boundary; a
  parallel on-disk file would just be that same frame moved somewhere less
  readable, not a stronger test.
- The Finnhub truncation shape needs no separate fixture at all: `record_finnhub`
  below only ever asks for the last 7 days (see the request below), so this
  recording does not by itself demonstrate that the vendor *ignores* `from` --
  that evidence is PYQ-140's own live probing across several `from` values, not
  this file. What the recording does give for free is a genuine, non-synthetic
  instance of the *shape* PYQ-140 found: 247 real articles spanning only 6
  distinct days. `test_fetch_sentiment_recorded_payload_reproduces_pyq_140s_truncation`
  (tests/test_sentiment.py) feeds that real payload through `fetch_sentiment`
  with a multi-year `start` and asserts the resulting coverage is exactly that
  sparse, rather than reaching for a hand-built file to fake the same shape.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from pyquant.config import Settings

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
MANIFEST_PATH = FIXTURES_DIR / "MANIFEST.json"


def _record(manifest: dict, name: str, **fields) -> None:
    manifest[name] = {"recorded_at": dt.datetime.now().isoformat(timespec="seconds"), **fields}
    print(f"  recorded {name}: {fields}")


def record_prices(manifest: dict) -> None:
    import yfinance as yf

    print("Fetching yfinance prices (AAPL, 3mo)...")
    df = yf.Ticker("AAPL").history(period="3mo", auto_adjust=True)
    df.to_pickle(FIXTURES_DIR / "yfinance_prices_aapl.pkl")
    _record(
        manifest,
        "yfinance_prices_aapl.pkl",
        vendor="yfinance",
        library_version=yf.__version__,
        symbol="AAPL",
        rows=len(df),
        columns=list(df.columns),
    )


def record_vix(manifest: dict) -> None:
    import yfinance as yf

    print("Fetching yfinance VIX (3mo)...")
    df = yf.Ticker("^VIX").history(period="3mo", auto_adjust=True)
    df.to_pickle(FIXTURES_DIR / "yfinance_vix.pkl")
    _record(
        manifest,
        "yfinance_vix.pkl",
        vendor="yfinance",
        library_version=yf.__version__,
        symbol="^VIX",
        rows=len(df),
        columns=list(df.columns),
    )


def record_sectors(manifest: dict) -> None:
    import yfinance as yf

    print("Fetching yfinance sector ETFs (XLK, SPY, 3mo)...")
    df = yf.download(["XLK", "SPY"], period="3mo", progress=False, auto_adjust=True)
    df.to_pickle(FIXTURES_DIR / "yfinance_sectors.pkl")
    _record(
        manifest,
        "yfinance_sectors.pkl",
        vendor="yfinance",
        library_version=yf.__version__,
        symbols=["XLK", "SPY"],
        rows=len(df),
    )


def record_options(manifest: dict) -> None:
    import yfinance as yf

    print("Fetching yfinance options chain (AAPL, nearest expiry)...")
    ticker = yf.Ticker("AAPL")
    expiries = ticker.options
    if not expiries:
        print("  no options listed for AAPL right now; skipping")
        return
    expiry = expiries[0]
    chain = ticker.option_chain(expiry)
    try:
        fast_last_price = float(ticker.fast_info["lastPrice"])
    except Exception:
        fast_last_price = None
    payload = {
        "expiry": expiry,
        "calls": chain.calls,
        "puts": chain.puts,
        "fast_last_price": fast_last_price,
    }
    pd.to_pickle(payload, FIXTURES_DIR / "yfinance_options_aapl.pkl")
    _record(
        manifest,
        "yfinance_options_aapl.pkl",
        vendor="yfinance",
        library_version=yf.__version__,
        symbol="AAPL",
        expiry=expiry,
        n_calls=len(chain.calls),
        n_puts=len(chain.puts),
    )


def record_fred(manifest: dict, settings: Settings) -> None:
    if not settings.fred_api_key:
        print("No FRED_API_KEY configured; skipping FRED fixture")
        return
    import fredapi

    print("Fetching FRED vintage releases (DFF, last 60 days)...")
    fred = fredapi.Fred(api_key=settings.fred_api_key)
    end = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    start = end - pd.DateOffset(days=60)
    releases = fred.get_series_all_releases(
        "DFF", realtime_start=start.strftime("%Y-%m-%d"), realtime_end=end.strftime("%Y-%m-%d")
    )
    # FRED attaches the *entire* historical series (back to 1954 for DFF) to the
    # earliest vintage boundary in the window, which makes an unfiltered response
    # tens of thousands of rows. Keep only recent reference dates -- this preserves
    # the real multi-vintage/revision structure _vintage_series() parses (many
    # realtime_start groups, each contributing a few rows) while keeping the
    # checked-in fixture small.
    releases["date"] = pd.to_datetime(releases["date"])
    releases["realtime_start"] = pd.to_datetime(releases["realtime_start"])
    cutoff = releases["realtime_start"].max() - pd.DateOffset(days=90)
    releases = releases[releases["date"] >= cutoff].sort_values(["realtime_start", "date"])
    records = json.loads(releases.to_json(orient="records", date_format="iso"))
    (FIXTURES_DIR / "fred_dff.json").write_text(json.dumps(records, indent=2))
    _record(
        manifest,
        "fred_dff.json",
        vendor="FRED (fredapi)",
        library_version=fredapi.__version__,
        series_id="DFF",
        rows=len(records),
    )


def _sanitize_headline(article: dict, i: int) -> dict:
    out = dict(article)
    if "headline" in out:
        out["headline"] = f"Sanitized headline placeholder {i}"
    if "summary" in out:
        out["summary"] = f"Sanitized summary placeholder {i}."
    if "url" in out:
        out["url"] = f"https://example.invalid/article/{i}"
    if "image" in out:
        out["image"] = f"https://example.invalid/image/{i}.png"
    return out


def record_finnhub(manifest: dict, settings: Settings) -> None:
    if not settings.finnhub_api_key:
        print("No FINNHUB_API_KEY configured; skipping Finnhub fixture")
        return
    import requests

    print("Fetching Finnhub company news (AAPL, last 7 days)...")
    end = dt.date.today()
    start = end - dt.timedelta(days=7)
    resp = requests.get(
        "https://finnhub.io/api/v1/company-news",
        params={
            "symbol": "AAPL",
            "from": start.isoformat(),
            "to": end.isoformat(),
            "token": settings.finnhub_api_key,
        },
        timeout=20,
    )
    resp.raise_for_status()
    articles = resp.json()
    sanitized = [_sanitize_headline(a, i) for i, a in enumerate(articles)]
    (FIXTURES_DIR / "finnhub_news_aapl.json").write_text(json.dumps(sanitized, indent=2))
    _record(
        manifest,
        "finnhub_news_aapl.json",
        vendor="Finnhub",
        library_version="REST v1 (requests)",
        symbol="AAPL",
        rows=len(sanitized),
        note="headline/summary/url/image text sanitized",
    )


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    settings = Settings()
    manifest: dict = {}

    record_prices(manifest)
    record_vix(manifest)
    record_sectors(manifest)
    record_options(manifest)
    record_fred(manifest, settings)
    record_finnhub(manifest, settings)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
