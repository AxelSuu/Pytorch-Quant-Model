"""Rich table/JSON rendering helpers for the CLI (PYQ-330).

Pure formatting: no Typer wiring, no ``console.print`` calls. Callers in
``cli/app.py`` build these tables/strings and print them themselves, so this
module stays a plain function library rather than a second place command
flow lives.
"""

from __future__ import annotations

import json

from rich.table import Table

from pyquant.analysis.forecast import Forecast


def _emit_json(data) -> None:
    """Print plain JSON to stdout -- no Rich, so no ANSI escape codes leak in."""
    print(json.dumps(data, indent=2, default=str))


def _band_label(quantiles: list[float]) -> str:
    """Name the calibration band from the configured quantiles, e.g. "p10-p90".

    `calibration_coverage` measures the *outermost* band, so hardcoding
    "p10-p90" mislabelled any non-default `quantiles` -- including the project's
    own configs/wide_quantile_aggressive.yaml (PYQ-122).
    """
    return f"p{int(quantiles[0] * 100)}-p{int(quantiles[-1] * 100)}"


def _add_metric_rows(table: Table, ev, quantiles: list[float], suffix: str = "") -> None:
    """Add the shared evaluation rows, sample size included.

    The sample size is not optional context: "directional accuracy 100.0%" off 5
    points and off 500 are different claims, and the table used to render them
    identically (PYQ-117).
    """
    table.add_row(f"Model MAE{suffix}", f"{ev.model_mae:.4f}")
    table.add_row(f"Baseline MAE{suffix} (persistence)", f"{ev.baseline_mae:.4f}")
    # "(pooled MAE ratio)" names the estimator explicitly (PYQ-141): this is
    # computed from pooled model/baseline MAE, a *ratio of means*, which is a
    # different statistic from the mean of the per-window skill column
    # `_per_window_table` prints below it and can disagree with it without
    # limit -- see that table's caption in `backtest()`.
    table.add_row("Skill vs. baseline (pooled MAE ratio)", f"{ev.skill_vs_baseline:+.1%}")
    # Baselines beyond persistence (PYQ-275): persistence is uniquely
    # favourable to the null on a near-random-walk level series, so headline
    # skill against it alone is weak evidence. Name the *strongest* one
    # (lowest MAE, hardest to beat) explicitly, rather than only ever
    # reporting skill against the weakest comparator available.
    other_baselines = {k: v for k, v in ev.baseline_maes.items() if k != "persistence"}
    if other_baselines:
        for name, mae in sorted(other_baselines.items()):
            table.add_row(f"Baseline MAE{suffix} ({name})", f"{mae:.4f}")
        strongest_name, strongest_mae = ev.strongest_baseline
        table.add_row(
            f"Skill vs. strongest baseline ({strongest_name}, pooled MAE ratio)",
            f"{ev.skill_vs_strongest_baseline:+.1%}",
        )
    table.add_row(f"Directional accuracy{suffix}", f"{ev.directional_accuracy:.1%}")
    table.add_row(
        f"Calibration coverage{suffix} ({_band_label(quantiles)})",
        f"{ev.calibration_coverage:.1%}",
    )
    if ev.quantile_exceedance:
        for quantile, rate in ev.quantile_exceedance.items():
            table.add_row(f"Empirical p{quantile:.0%}{suffix}", f"{rate:.1%}")
    if ev.pinball_losses:
        table.add_row(
            f"Mean pinball loss{suffix}",
            f"{sum(ev.pinball_losses.values()) / len(ev.pinball_losses):.4f}",
        )
    # CRPS scores the whole predictive distribution; Winkler charges for band
    # width as well as coverage, which is the pathology coverage alone hides --
    # a band can hit nominal by being enormous (PYQ-252). Both: lower is better.
    # `is not None` rather than a bare truthy check (PYQ-156): both fields are
    # plain floats, always populated, and a legitimate 0.0 must still print.
    if ev.crps is not None:
        table.add_row(f"CRPS{suffix} (lower better)", f"{ev.crps:.4f}")
    if ev.winkler_score is not None:
        table.add_row(f"Winkler interval score{suffix} (lower better)", f"{ev.winkler_score:.4f}")
    table.add_row(
        "Evaluated on",
        f"{ev.n_samples} overlapping windows ({ev.n_points} predictions; "
        f"effective n≈{ev.effective_n_samples})",
    )


def _per_window_table(result, quantiles: list[float]) -> Table:
    """Per-window metrics behind the aggregate.

    A mean directional accuracy of 60% built from windows at 100/20 says something
    very different from two windows at 60%, and only the mean was ever shown
    (PYQ-226). The spread is the whole reason to run more than one window.
    """
    table = Table(title="Per-window results", title_style="dim")
    table.add_column("Window", justify="right")
    table.add_column("Model MAE", justify="right")
    table.add_column("Baseline MAE", justify="right")
    # "(per-window)" is deliberate (PYQ-141): this column is a mean-of-ratios
    # once read down, a different estimator from the pooled-MAE-ratio headline
    # above it, and the two can disagree without limit -- see the caption
    # printed after this table in `backtest()`.
    table.add_column("Skill (per-window)", justify="right")
    table.add_column("Directional", justify="right")
    table.add_column(f"Coverage {_band_label(quantiles)}", justify="right")
    for i, ev in enumerate(result.per_window, start=1):
        table.add_row(
            str(i),
            f"{ev.model_mae:.4f}",
            f"{ev.baseline_mae:.4f}",
            f"{ev.skill_vs_baseline:+.1%}",
            f"{ev.directional_accuracy:.1%}",
            f"{ev.calibration_coverage:.1%}",
        )
    return table


def _per_horizon_table(
    evaluation, quantiles: list[float], title: str = "Per-horizon breakdown"
) -> Table:
    """The per-decoder-step profile every other metric averages away (PYQ-267).

    Persistence is hardest to beat at h=1 and progressively less so as h grows,
    so a model that is genuinely learning something should show skill
    *increasing* with horizon while one that only tracks the last close shows
    the opposite -- indistinguishable in a single mean-over-horizon number.
    """
    table = Table(title=title, title_style="dim")
    table.add_column("Step", justify="right")
    table.add_column("Model MAE", justify="right")
    table.add_column("Baseline MAE", justify="right")
    table.add_column("Skill", justify="right")
    table.add_column("Directional", justify="right")
    table.add_column(f"Coverage {_band_label(quantiles)}", justify="right")
    for step in evaluation.per_horizon:
        table.add_row(
            f"h={step.step}",
            f"{step.model_mae:.4f}",
            f"{step.baseline_mae:.4f}",
            f"{step.skill_vs_baseline:+.1%}",
            f"{step.directional_accuracy:.1%}",
            f"{step.calibration_coverage:.1%}",
        )
    return table


def _color_pct(pct: float) -> str:
    """Render a percentage as Rich markup, green/▲ when non-negative and red/▼ below."""
    arrow = "▲" if pct >= 0 else "▼"
    color = "green" if pct >= 0 else "red"
    return f"[{color}]{arrow} {abs(pct):.2f}%[/{color}]"


def _forecast_table(fc: Forecast) -> Table:
    """Build the per-step quantile table, one row per forecast day."""
    table = Table(title=f"{fc.symbol} — {fc.horizon}-day forecast")
    table.add_column("Day", justify="right")
    # Name the actual date each step is for; "Day 1" alone gave no way to notice
    # that the horizon was pointing at the wrong window (PYQ-115).
    table.add_column("Date", justify="right")
    for q in fc.quantiles:
        table.add_column(f"p{int(q * 100)}", justify="right")
    table.add_column("vs now", justify="right")
    dates = fc.forecast_dates
    for d in range(fc.horizon):
        row = [str(d + 1), str(dates[d].date())]
        for q in fc.quantiles:
            row.append(f"${fc.quantile_series(q)[d]:.2f}")
        pct = (fc.median[d] - fc.current_price) / fc.current_price * 100
        row.append(_color_pct(pct))
        table.add_row(*row)
    return table


def _fmt_bytes(n: int) -> str:
    """Format a byte count as B/KB/MB/GB, whole bytes and one decimal above that."""
    size = float(n)
    # GB is the last unit, so it is the explicit fall-through rather than a special
    # case inside the loop with an unreachable return after it (PYQ-126).
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
