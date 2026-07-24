"""PyQuant command-line interface (Typer + Rich)."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pyquant.analysis.forecast import Forecast, generate_forecast
from pyquant.analysis.interpret import attention_to_series, explain_forecast
from pyquant.cli import charts
from pyquant.config import Settings, load_settings
from pyquant.data.options import fetch_options_snapshot
from pyquant.models import tft

app = typer.Typer(
    help="PyQuant — multi-modal market research with a Temporal Fusion Transformer.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
logger = logging.getLogger(__name__)


@app.callback()
def _configure_logging(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable INFO-level logging"),
    debug: bool = typer.Option(
        False, "--debug", help="Enable DEBUG-level logging and un-silence Lightning's own output"
    ),
) -> None:
    """PyQuant — multi-modal market research with a Temporal Fusion Transformer."""
    # Library/runtime chatter is kept out of the pretty CLI output by default;
    # --verbose/--debug turn it back on for troubleshooting (e.g. NaN training
    # loss, unexpected feature drift) without editing source.
    level = logging.DEBUG if debug else logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(message)s", force=True)
    logging.getLogger("lightning.pytorch").setLevel(logging.INFO if debug else logging.ERROR)
    # Lightning/PyTorch emit most of their startup/deprecation chatter via
    # warnings.warn(...), a channel the logging config above never touches.
    if debug:
        warnings.resetwarnings()
    else:
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)


def _build_settings(
    period: str | None,
    no_macro: bool,
    no_sentiment: bool,
    no_sectors: bool,
) -> Settings:
    s = load_settings()
    if period:
        s.data.period = period
    if no_macro:
        s.data.use_macro = False
    if no_sentiment:
        s.data.use_sentiment = False
    if no_sectors:
        s.data.use_sectors = False
    return s


def _color_pct(pct: float) -> str:
    arrow = "▲" if pct >= 0 else "▼"
    color = "green" if pct >= 0 else "red"
    return f"[{color}]{arrow} {abs(pct):.2f}%[/{color}]"


@app.command()
def train(
    symbols: str = typer.Argument(
        ..., help="Ticker symbol, or comma-separated symbols to pool, e.g. AAPL,MSFT,NVDA"
    ),
    name: str = typer.Option(
        None, help="Bundle directory name (defaults to the symbol(s) joined with '_')"
    ),
    pin: str = typer.Option(
        None, help="Name a reproducible dataset snapshot to save/reuse for this experiment"
    ),
    epochs: int = typer.Option(None, help="Override max training epochs"),
    period: str = typer.Option(None, help="History to pull, e.g. 5y, 10y"),
    no_macro: bool = typer.Option(False, "--no-macro", help="Disable macro features"),
    no_sentiment: bool = typer.Option(False, "--no-sentiment", help="Disable news sentiment"),
    no_sectors: bool = typer.Option(False, "--no-sectors", help="Disable sector features"),
):
    """Train a Temporal Fusion Transformer for SYMBOLS (pooled if more than one)."""
    settings = _build_settings(period, no_macro, no_sentiment, no_sectors)
    tickers = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    console.print(f"[bold cyan]Training TFT for {', '.join(tickers)}[/bold cyan]")
    # Lightning renders its own live progress bar during the fit (progress=True);
    # a competing console.status() spinner was only ever masked by it (PYQ-222),
    # so let Lightning's bar be the single live indicator.
    result = tft.train(tickers, settings, bundle_name=name, max_epochs=epochs, progress=True, pin=pin)

    ev = result.evaluation
    table = Table(title=f"Training complete — {result.bundle_dir.name}", show_header=False)
    table.add_row("Bundle", str(result.bundle_dir))
    table.add_row("Symbols", ", ".join(result.symbols))
    table.add_row("Features used", str(result.n_features))
    table.add_row("Epochs run", str(result.epochs_run))
    table.add_row("Validation loss", f"{result.val_loss:.5f}")
    table.add_row("Model MAE", f"{ev.model_mae:.4f}")
    table.add_row("Baseline MAE (persistence)", f"{ev.baseline_mae:.4f}")
    table.add_row("Skill vs. baseline", f"{ev.skill_vs_baseline:+.1%}")
    table.add_row("Directional accuracy", f"{ev.directional_accuracy:.1%}")
    table.add_row("Calibration coverage (p10-p90)", f"{ev.calibration_coverage:.1%}")
    console.print(table)
    console.print("[dim]Next: pyquant forecast " + result.symbols[0] + "[/dim]")


@app.command()
def backtest(
    symbol: str = typer.Argument(..., help="Ticker symbol, e.g. AAPL"),
    windows: int = typer.Option(5, help="Number of rolling walk-forward windows"),
    epochs: int = typer.Option(None, help="Override max training epochs per window"),
    period: str = typer.Option(None, help="History to pull, e.g. 5y, 10y"),
    no_macro: bool = typer.Option(False, "--no-macro", help="Disable macro features"),
    no_sentiment: bool = typer.Option(False, "--no-sentiment", help="Disable news sentiment"),
    no_sectors: bool = typer.Option(False, "--no-sectors", help="Disable sector features"),
):
    """Walk-forward backtest SYMBOL across multiple rolling origins."""
    settings = _build_settings(period, no_macro, no_sentiment, no_sectors)
    console.print(f"[bold cyan]Walk-forward backtesting {symbol.upper()}[/bold cyan]")
    with console.status(f"Training and evaluating {windows} rolling windows..."):
        result = tft.walk_forward_backtest(
            symbol, settings, n_windows=windows, max_epochs=epochs, progress=False
        )

    ev = result.aggregated
    table = Table(
        title=f"Walk-forward backtest — {result.symbol} ({result.n_windows} windows)",
        show_header=False,
    )
    table.add_row("Model MAE (avg)", f"{ev.model_mae:.4f}")
    table.add_row("Baseline MAE (avg, persistence)", f"{ev.baseline_mae:.4f}")
    table.add_row("Skill vs. baseline", f"{ev.skill_vs_baseline:+.1%}")
    table.add_row("Directional accuracy (avg)", f"{ev.directional_accuracy:.1%}")
    table.add_row("Calibration coverage (avg, p10-p90)", f"{ev.calibration_coverage:.1%}")
    console.print(table)


def _forecast_table(fc: Forecast) -> Table:
    table = Table(title=f"{fc.symbol} — {fc.horizon}-day forecast")
    table.add_column("Day", justify="right")
    for q in fc.quantiles:
        table.add_column(f"p{int(q * 100)}", justify="right")
    table.add_column("vs now", justify="right")
    for d in range(fc.horizon):
        row = [str(d + 1)]
        for q in fc.quantiles:
            row.append(f"${fc.quantile_series(q)[d]:.2f}")
        pct = (fc.median[d] - fc.current_price) / fc.current_price * 100
        row.append(_color_pct(pct))
        table.add_row(*row)
    return table


@app.command()
def forecast(
    symbol: str = typer.Argument(..., help="Ticker symbol"),
    bundle: str = typer.Option(
        None, help="Bundle name to load, if different from SYMBOL (e.g. a pooled bundle)"
    ),
    pin: str = typer.Option(
        None, help="Replay a named reproducible dataset snapshot instead of live data"
    ),
    export: Path = typer.Option(None, help="Write a PNG fan chart to this path"),
    no_chart: bool = typer.Option(False, "--no-chart", help="Skip the terminal chart"),
):
    """Forecast SYMBOL with p10/p50/p90 uncertainty bands."""
    settings = load_settings()
    loaded_bundle = tft.load(bundle, settings) if bundle else None
    fc = generate_forecast(symbol, settings, bundle=loaded_bundle, pin=pin)

    console.print(
        Panel(
            f"Current: [bold]${fc.current_price:.2f}[/bold]   "
            f"As of: {fc.last_date.date()}   "
            f"Expected ({fc.horizon}d): {_color_pct(fc.expected_return_pct())}",
            title=f"{fc.symbol}",
        )
    )
    console.print(_forecast_table(fc))

    snap = fetch_options_snapshot(fc.symbol)
    if snap.put_call_ratio is not None:
        console.print(
            f"[dim]Options ({snap.expiry}): put/call {snap.put_call_ratio:.2f} "
            f"({snap.sentiment_label}), ATM IV {snap.atm_iv:.0%}, skew {snap.iv_skew:+.2%}[/dim]"
        )

    if not no_chart:
        charts.fan_chart(fc)
    if export:
        path = charts.export_fan_chart(fc, export)
        console.print(f"[green]Chart written to {path}[/green]")


@app.command()
def explain(
    symbol: str = typer.Argument(..., help="Ticker symbol"),
    bundle_name: str = typer.Option(
        None, "--bundle", help="Bundle name to load, if different from SYMBOL (e.g. a pooled bundle)"
    ),
    top: int = typer.Option(10, help="Number of top features to show"),
    no_chart: bool = typer.Option(False, "--no-chart", help="Skip the terminal charts"),
):
    """Explain SYMBOL's forecast: feature importance + temporal attention."""
    settings = load_settings()
    bundle = tft.load(bundle_name or symbol, settings)
    interp = explain_forecast(symbol, settings, bundle=bundle)

    table = Table(title=f"{interp.symbol} — top {top} feature importances")
    table.add_column("Feature")
    table.add_column("Importance", justify="right")
    for name, weight in interp.top_features(top):
        table.add_row(name, f"{weight * 100:.1f}%")
    console.print(table)

    att = attention_to_series(interp)
    peak = att.idxmax()
    console.print(
        f"[dim]Peak attention on {peak.date()} ({att.max() / att.sum():.0%} of focus)[/dim]"
    )

    if not no_chart:
        charts.importance_chart(interp.top_features(top))
        charts.attention_chart(interp.attention)


@app.command()
def scan(
    symbols: str = typer.Argument(..., help="Comma-separated tickers, e.g. AAPL,MSFT,NVDA"),
):
    """Compare forecasts across multiple trained symbols."""
    settings = load_settings()
    tickers = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    table = Table(title="Multi-asset forecast comparison")
    table.add_column("Symbol")
    table.add_column("Current", justify="right")
    table.add_column("Median target", justify="right")
    table.add_column("Expected", justify="right")
    table.add_column("Band width", justify="right")
    table.add_column("Signal")

    for ticker in tickers:
        try:
            fc = generate_forecast(ticker, settings)
        except FileNotFoundError:
            table.add_row(ticker, "—", "—", "—", "—", "[dim]not trained[/dim]")
            continue
        except Exception as exc:
            # One flaky symbol (a transient data-source error, a bad config,
            # etc.) must not sink the whole multi-symbol comparison (PYQ-113).
            logger.warning("Could not forecast %s: %s", ticker, exc)
            table.add_row(ticker, "—", "—", "—", "—", "[red]error[/red]")
            continue
        pct = fc.expected_return_pct()
        lo = fc.quantile_series(fc.quantiles[0])[-1]
        hi = fc.quantile_series(fc.quantiles[-1])[-1]
        lo_pct = (lo - fc.current_price) / fc.current_price * 100
        hi_pct = (hi - fc.current_price) / fc.current_price * 100
        band = (hi - lo) / fc.current_price * 100
        # Beyond a minimum move, require the *whole* uncertainty band to sit
        # on one side of 0% -- a wide, zero-straddling band isn't a real BUY
        # or SELL signal even if the median alone looks confident.
        if pct > 2 and lo_pct > 0:
            signal = "[green]BUY[/green]"
        elif pct < -2 and hi_pct < 0:
            signal = "[red]SELL[/red]"
        else:
            signal = "HOLD"
        table.add_row(
            ticker,
            f"${fc.current_price:.2f}",
            f"${fc.median[-1]:.2f}",
            _color_pct(pct),
            f"{band:.1f}%",
            signal,
        )
    console.print(table)


if __name__ == "__main__":
    app()
