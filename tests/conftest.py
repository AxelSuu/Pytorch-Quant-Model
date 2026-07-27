"""Pytest configuration and shared fixtures for PyQuant."""

import os

# Rich decides at Console construction time whether to emit ANSI colour, based on
# whether stdout looks like a terminal. `pyquant.cli.app` builds its Console at
# import, so CLI tests asserting on substrings of stdout passed under a piped
# pytest run and failed under an interactive one -- the same test, two answers,
# decided by the ambient terminal (PYQ-138). Pin it before any test module
# imports the CLI, so output is deterministic either way.
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def sample_ohlcv_df():
    """A realistic, date-indexed OHLCV DataFrame (no network)."""
    np.random.seed(42)
    n = 400
    dates = pd.bdate_range("2022-01-03", periods=n)
    returns = np.random.randn(n) * 0.02
    close = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame(
        {
            "Open": close * (1 + np.random.randn(n) * 0.005),
            "High": close * (1 + np.abs(np.random.randn(n) * 0.01)),
            "Low": close * (1 - np.abs(np.random.randn(n) * 0.01)),
            "Close": close,
            "Volume": np.abs(np.random.randn(n)) * 1_000_000,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


@pytest.fixture
def settings(tmp_path):
    """Default settings with all enrichments off (pure-OHLCV baseline)."""
    from pyquant.config import Settings

    s = Settings()
    s.data.use_macro = False
    s.data.use_sectors = False
    s.data.use_sentiment = False
    # Small windows so tests stay fast.
    s.training.max_encoder_length = 20
    s.training.max_prediction_length = 5
    # Caching is a production concern; keep it off (and isolated to a tmp dir
    # as a backstop) so tests never read/write the real project directory or
    # leak state between tests that happen to share a symbol + settings.
    s.data.cache_enabled = False
    s.data.cache_dir = tmp_path / "cache"
    return s
