"""Which of the 25+ features earn their place? (PYQ-316)

One-off investigation script: walk-forward backtests one symbol across cumulative
feature groups (price-only / +technicals / +macro / +sectors / +sentiment), plus a
correlation matrix over the technical indicators to name the exactly-redundant columns.
Folds in PYQ-140 (Finnhub coverage): the sentiment arm's skill delta is the evidence
that ticket's default-config decision needs.

Usage:
    uv run python scripts/ablate_features.py [SYMBOL] [--windows N]
"""

from __future__ import annotations

import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

GROUPS: list[tuple[str, dict[str, bool]]] = [
    ("price_only", {"use_indicators": False, "use_macro": False, "use_sectors": False, "use_sentiment": False}),
    ("+technicals", {"use_indicators": True, "use_macro": False, "use_sectors": False, "use_sentiment": False}),
    ("+macro", {"use_indicators": True, "use_macro": True, "use_sectors": False, "use_sentiment": False}),
    ("+sectors", {"use_indicators": True, "use_macro": True, "use_sectors": True, "use_sentiment": False}),
    ("+sentiment", {"use_indicators": True, "use_macro": True, "use_sectors": True, "use_sentiment": True}),
]


def _settings():
    from pyquant.config import Settings

    settings = Settings()
    settings.training.target = "log_return"
    settings.training.max_encoder_length = 20
    settings.training.max_prediction_length = 5
    settings.training.max_epochs = 8
    settings.tft.hidden_size = 8
    settings.tft.hidden_continuous_size = 4
    settings.data.use_options = False  # display-only; never a model feature regardless
    return settings


def run_ablation(symbol: str, n_windows: int) -> pd.DataFrame:
    from pyquant.models import tft

    settings = _settings()
    rows = []
    for label, toggles in GROUPS:
        for field, value in toggles.items():
            setattr(settings.data, field, value)
        result = tft.walk_forward_backtest(symbol, settings, n_windows=n_windows, progress=False)
        ev = result.aggregated
        print(
            f"{label:12s} skill={ev.skill_vs_baseline:+.4f}  dir_acc={ev.directional_accuracy:.3f}  "
            f"coverage={ev.calibration_coverage:.3f}  n_points={ev.n_points}"
        )
        rows.append(
            {
                "group": label,
                "skill_vs_baseline": ev.skill_vs_baseline,
                "directional_accuracy": ev.directional_accuracy,
                "calibration_coverage": ev.calibration_coverage,
                "crps": ev.crps,
                "n_points": ev.n_points,
            }
        )
    return pd.DataFrame(rows)


def indicator_correlations(symbol: str) -> pd.DataFrame:
    from pyquant.config import Settings
    from pyquant.data.dataset import build_panel
    from pyquant.data.prices import INDICATOR_COLUMNS

    settings = Settings()
    settings.data.use_macro = False
    settings.data.use_sectors = False
    settings.data.use_sentiment = False
    panel = build_panel(symbol, settings)
    return panel[INDICATOR_COLUMNS].corr()


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    symbol = args[0].upper() if args else "AAPL"
    n_windows = 3
    if "--windows" in sys.argv:
        n_windows = int(sys.argv[sys.argv.index("--windows") + 1])

    print(f"Ablating feature groups for {symbol}, {n_windows} walk-forward windows each\n")
    results = run_ablation(symbol, n_windows)
    print("\n=== Ablation summary ===")
    print(results.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print("\n=== Indicator correlation matrix (|r| > 0.9) ===")
    corr = indicator_correlations(symbol)
    seen = set()
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) > 0.9:
                print(f"  {a:15s} ~ {b:15s}  r={r:+.3f}")
                seen.add((a, b))
    if not seen:
        print("  (none above the threshold)")


if __name__ == "__main__":
    main()
