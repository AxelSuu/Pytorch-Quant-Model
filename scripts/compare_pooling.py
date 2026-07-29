"""Is pooling actually helping, now that PYQ-116 aligned the calendar? (PYQ-315)

One-off investigation script, not part of the test suite: trains per-symbol models and
one pooled model over the same symbols, then scores the pooled model on *each* symbol's
own validation slice separately (train()'s own reported metric for a pooled bundle is
aggregated across every symbol, which cannot answer "did pooling help THIS symbol").

Usage:
    uv run python scripts/compare_pooling.py SYMBOL1 SYMBOL2 [SYMBOL3 ...]
"""

from __future__ import annotations

import sys
import warnings

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

warnings.filterwarnings("ignore")


def _settings():
    from pyquant.config import Settings

    settings = Settings()
    settings.training.target = "log_return"
    settings.training.max_encoder_length = 30
    settings.training.max_prediction_length = 5
    settings.training.validation_days = 40
    settings.training.max_epochs = 15
    settings.tft.hidden_size = 16
    settings.tft.hidden_continuous_size = 8
    return settings


def per_symbol_skill(symbol: str, settings) -> float:
    from pyquant.models import tft

    result = tft.train(symbol, settings, bundle_name=f"{symbol}_SOLO", progress=False)
    return result.evaluation.skill_vs_baseline


def pooled_skill_per_symbol(symbols: list[str], settings) -> dict[str, float]:
    from pyquant.analysis.metrics import evaluate_predictions
    from pyquant.models import tft

    bundle_name = "_".join(symbols) + "_POOLED"
    tft.train(symbols, settings, bundle_name=bundle_name, progress=False)
    bundle = tft.load(bundle_name, settings)

    df = tft._build_pooled_long_df(symbols, settings, None, None)
    horizon = settings.training.max_prediction_length
    validation_days = max(settings.training.validation_days, horizon)
    max_idx = int(df["time_idx"].max())
    validation_start = max_idx - validation_days + 1

    out: dict[str, float] = {}
    for symbol in symbols:
        sub = df[df["symbol"] == symbol]
        ds = TimeSeriesDataSet.from_parameters(
            bundle.dataset_params,
            sub,
            predict=False,
            stop_randomization=True,
            min_prediction_idx=validation_start,
        )
        dl = ds.to_dataloader(train=False, batch_size=64, num_workers=0)
        predictions, actuals, last_observed, _ = tft._raw_validation_arrays(bundle.model, dl)
        ev = evaluate_predictions(
            predictions,
            actuals,
            last_observed,
            settings.tft.quantiles,
            target="log_return" if settings.training.target == "log_return" else "close",
        )
        out[symbol] = ev.skill_vs_baseline
    return out


def main() -> None:
    symbols = [s.upper() for s in sys.argv[1:]] or ["AAPL", "ARM"]
    settings = _settings()

    print(f"Symbols: {symbols}")
    solo = {s: per_symbol_skill(s, settings) for s in symbols}
    pooled = pooled_skill_per_symbol(symbols, settings)

    print("\n=== skill_vs_baseline, per symbol ===")
    table = pd.DataFrame({"per_symbol": solo, "pooled": pooled})
    table["pooled_minus_solo"] = table["pooled"] - table["per_symbol"]
    print(table.to_string(float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()
