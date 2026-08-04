"""Terminal charts (Plotext) and optional Matplotlib PNG export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotext as plt

from pyquant.analysis.forecast import Forecast


def fan_chart(forecast: Forecast, history_tail: int = 60) -> None:
    """Render a price history + forecast fan (p10/p50/p90) in the terminal."""
    plt.clear_figure()
    hist = forecast.history.tail(history_tail)
    hist_x = list(range(len(hist)))
    plt.plot(hist_x, hist.values.tolist(), label="history", color="cyan")

    h = forecast.horizon
    fut_x = list(range(len(hist) - 1, len(hist) - 1 + h + 1))
    last = float(hist.values[-1])

    def path(q):
        """Quantile ``q``'s plot series, anchored to the last observed close.

        Prepending ``last`` is what joins the forecast line to the history line;
        without it the chart shows a visual gap at the forecast origin.
        """
        return [last] + forecast.quantile_series(q).tolist()

    if 0.5 in forecast.quantiles:
        plt.plot(fut_x, path(0.5), label="median", color="yellow")
    lo, hi = forecast.quantiles[0], forecast.quantiles[-1]
    plt.plot(fut_x, path(lo), label=f"p{int(lo * 100)}", color="red")
    plt.plot(fut_x, path(hi), label=f"p{int(hi * 100)}", color="green")

    plt.title(f"{forecast.symbol} — {h}-day forecast with uncertainty band")
    plt.theme("dark")
    plt.plotsize(80, 22)
    plt.show()


def importance_chart(top_features: list[tuple[str, float]]) -> None:
    """Horizontal-ish bar chart of feature importance."""
    plt.clear_figure()
    names = [f for f, _ in top_features][::-1]
    vals = [round(w * 100, 2) for _, w in top_features][::-1]
    plt.bar(names, vals, orientation="horizontal", color="magenta")
    plt.title("Feature importance (%)")
    plt.theme("dark")
    plt.plotsize(80, max(10, len(names) + 4))
    plt.show()


def attention_chart(attention: np.ndarray) -> None:
    """Bar chart of temporal attention over the lookback window."""
    plt.clear_figure()
    x = list(range(-len(attention) + 1, 1))  # days relative to "today" (0)
    plt.bar(x, attention.tolist(), color="orange")
    plt.title("Temporal attention (days ago -> today)")
    plt.theme("dark")
    plt.plotsize(80, 18)
    plt.show()


def export_fan_chart(forecast: Forecast, path: Path) -> Path:
    """Write a Matplotlib PNG of the fan chart. Imported lazily to keep startup fast."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as mpl

    path.parent.mkdir(parents=True, exist_ok=True)
    hist = forecast.history
    # Forecast.forecast_dates is the same calendar the prediction rows were built
    # from, so the axis labels cannot drift from the decoded steps (PYQ-115).
    fut = list(forecast.forecast_dates)

    fig, ax = mpl.subplots(figsize=(11, 5))
    ax.plot(hist.index, hist.values, color="#1f77b4", label="history")

    lo, hi = forecast.quantiles[0], forecast.quantiles[-1]
    last_date, last_close = forecast.last_date, forecast.current_price

    # The main band is drawn first, keyed exactly to `fut` (== forecast_dates)
    # -- invariant 8's test spies on fill_between and asserts its *first* call
    # matches Forecast.forecast_dates exactly, so nothing may be prepended onto
    # this series or drawn ahead of it.
    ax.fill_between(
        fut,
        forecast.quantile_series(lo),
        forecast.quantile_series(hi),
        color="#ff7f0e",
        alpha=0.25,
        label=f"p{int(lo * 100)}–p{int(hi * 100)}",
    )
    if 0.5 in forecast.quantiles:
        ax.plot(fut, forecast.quantile_series(0.5), "--", color="#ff7f0e", label="median")

    # Bridge the visual gap between the last observed close and the forecast's
    # first decoded step (PYQ-324), as a separate draw call after the ones
    # above: plotted alone, the band and the median line both start cold at
    # fut[0] already at their first-step values, which reads as the band
    # popping in offset from history rather than fanning out of it -- an
    # artifact of this rendering function, not of the underlying forecast (the
    # reconstructed price band itself widens monotonically with horizon; see
    # log_return_quantiles_to_price_band). Drawing it after, rather than
    # before or merged into, the calls above keeps their x-values exactly
    # `fut` for invariant 8; the two only touch at the single point x=fut[0],
    # so draw order has no visible effect on the rendered fill/line color.
    ax.fill_between(
        [last_date, fut[0]],
        [last_close, forecast.quantile_series(lo)[0]],
        [last_close, forecast.quantile_series(hi)[0]],
        color="#ff7f0e",
        alpha=0.25,
    )
    if 0.5 in forecast.quantiles:
        ax.plot(
            [last_date, fut[0]],
            [last_close, forecast.median[0]],
            "--",
            color="#ff7f0e",
            alpha=0.6,
        )

    ax.set_title(f"{forecast.symbol} — {forecast.horizon}-day forecast")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    mpl.close(fig)
    return path
