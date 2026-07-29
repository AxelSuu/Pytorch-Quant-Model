"""PyQuant command-line interface (Typer + Rich)."""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pyquant.analysis import serialize
from pyquant.analysis.forecast import Forecast, generate_forecast
from pyquant.analysis.interpret import attention_to_series, explain_forecast
from pyquant.analysis.metrics import moving_block_bootstrap_interval
from pyquant.analysis.signals import evaluate_signals
from pyquant.cli import charts
from pyquant.config import Settings, load_settings
from pyquant.data import cache as data_cache
from pyquant.data.options import OptionsSnapshot, append_snapshot, fetch_options_snapshot
from pyquant.models import tft

app = typer.Typer(
    help="PyQuant — multi-modal market research with a Temporal Fusion Transformer.",
    no_args_is_help=True,
    add_completion=False,
    # Expected failures (no trained bundle, a vanished data source) carry messages
    # that are already user-ready; a framed traceback around them is noise that
    # obscures the instruction (PYQ-120). _fail() renders them instead.
    pretty_exceptions_enable=False,
)

# Failures that are part of normal operation rather than bugs: report the message,
# exit non-zero, no traceback.
EXPECTED_FAILURES = (FileNotFoundError, tft.FeatureSchemaMismatch, ValueError)
console = Console()
logger = logging.getLogger(__name__)


@dataclass
class _Output:
    """Global output preferences set by the callback (PYQ-212)."""

    fmt: str = "rich"  # "rich" | "json"
    quiet: bool = False

    @property
    def json(self) -> bool:
        """True when ``--format json`` was requested, i.e. Rich output is suppressed."""
        return self.fmt == "json"


_output = _Output()


def _fail(exc: Exception) -> None:
    """Report an expected failure as a clean one-liner and exit non-zero."""
    console.print(f"[red]Error:[/red] {exc}")
    raise typer.Exit(1)


def _emit_json(data) -> None:
    """Print plain JSON to stdout -- no Rich, so no ANSI escape codes leak in."""
    print(json.dumps(data, indent=2, default=str))


@app.callback()
def _configure_logging(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable INFO-level logging"),
    debug: bool = typer.Option(
        False, "--debug", help="Enable DEBUG-level logging and un-silence Lightning's own output"
    ),
    output_format: str = typer.Option(
        "rich", "--format", help="Output format: 'rich' (default) or 'json'", metavar="rich|json"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress banners/progress bars (tables still print)"
    ),
) -> None:
    """PyQuant — multi-modal market research with a Temporal Fusion Transformer."""
    fmt = output_format.strip().lower()
    if fmt not in ("rich", "json"):
        raise typer.BadParameter("--format must be 'rich' or 'json'")
    _output.fmt = fmt
    # JSON output implies quiet: only the JSON document should reach stdout.
    _output.quiet = quiet or fmt == "json"

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
    config: Path | None = None,
) -> Settings:
    """Resolve settings for one command invocation, applying CLI flags last.

    Implements the documented precedence CLI flags > env > ``.env`` > YAML >
    defaults: ``load_settings`` establishes everything up to YAML, and the
    explicit flags here overwrite on top of it (PYQ-209).
    """
    # YAML config (if any) is layered first; the explicit CLI flags below then
    # win over it (PYQ-209).
    s = load_settings(config)
    if period:
        s.data.period = period
    if no_macro:
        s.data.use_macro = False
    if no_sentiment:
        s.data.use_sentiment = False
    if no_sectors:
        s.data.use_sectors = False
    return s


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
    table.add_row("Skill vs. baseline", f"{ev.skill_vs_baseline:+.1%}")
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
            f"Skill vs. strongest baseline ({strongest_name})",
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
    table.add_column("Skill", justify="right")
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


def _per_horizon_table(evaluation, quantiles: list[float], title: str = "Per-horizon breakdown") -> Table:
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


def _directional_interval(result, horizon: int) -> tuple[float, float]:
    """Moving-block interval for backtest window directional accuracy (PYQ-251)."""
    return moving_block_bootstrap_interval(
        [window.directional_accuracy for window in result.per_window], max(1, horizon)
    )


def _color_pct(pct: float) -> str:
    """Render a percentage as Rich markup, green/▲ when non-negative and red/▼ below."""
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
    config: Path = typer.Option(
        None, "--config", help="YAML experiment config (see configs/); CLI flags still win"
    ),
    epochs: int = typer.Option(None, help="Override max training epochs"),
    period: str = typer.Option(None, help="History to pull, e.g. 5y, 10y"),
    no_macro: bool = typer.Option(False, "--no-macro", help="Disable macro features"),
    no_sentiment: bool = typer.Option(False, "--no-sentiment", help="Disable news sentiment"),
    no_sectors: bool = typer.Option(False, "--no-sectors", help="Disable sector features"),
):
    """Train a Temporal Fusion Transformer for SYMBOLS (pooled if more than one)."""
    try:
        settings = _build_settings(period, no_macro, no_sentiment, no_sectors, config=config)
    except EXPECTED_FAILURES as exc:
        _fail(exc)
    tickers = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not _output.quiet:
        console.print(f"[bold cyan]Training TFT for {', '.join(tickers)}[/bold cyan]")
    # Lightning renders its own live progress bar during the fit (progress=True);
    # a competing console.status() spinner was only ever masked by it (PYQ-222),
    # so let Lightning's bar be the single live indicator.
    try:
        result = tft.train(
            tickers, settings, bundle_name=name, max_epochs=epochs, progress=not _output.quiet, pin=pin
        )
    except EXPECTED_FAILURES as exc:
        _fail(exc)

    if _output.json:
        _emit_json(serialize.train_result_to_dict(result))
        return

    table = Table(title=f"Training complete — {result.bundle_dir.name}", show_header=False)
    table.add_row("Bundle", str(result.bundle_dir))
    table.add_row("Symbols", ", ".join(result.symbols))
    table.add_row("Features used", str(result.n_features))
    table.add_row("Epochs run", str(result.epochs_run))
    # The best checkpoint's loss on the *selection* window EarlyStopping/
    # ModelCheckpoint monitored, not the test window the metrics below are
    # computed from -- a selection-event statistic, not a quality number
    # (PYQ-143; see TrainResult.val_loss's docstring).
    table.add_row("Selection loss", f"{result.val_loss:.5f}")
    _add_metric_rows(table, result.evaluation, settings.tft.quantiles)
    console.print(table)
    if len(result.evaluation.per_horizon) > 1:
        console.print(_per_horizon_table(result.evaluation, settings.tft.quantiles))
    if not _output.quiet:
        console.print("[dim]Next: pyquant forecast " + result.symbols[0] + "[/dim]")


@app.command()
def backtest(
    symbol: str = typer.Argument(..., help="Ticker symbol, e.g. AAPL"),
    windows: int = typer.Option(5, help="Number of rolling walk-forward windows"),
    config: Path = typer.Option(
        None, "--config", help="YAML experiment config (see configs/); CLI flags still win"
    ),
    epochs: int = typer.Option(None, help="Override max training epochs per window"),
    period: str = typer.Option(None, help="History to pull, e.g. 5y, 10y"),
    no_macro: bool = typer.Option(False, "--no-macro", help="Disable macro features"),
    no_sentiment: bool = typer.Option(False, "--no-sentiment", help="Disable news sentiment"),
    no_sectors: bool = typer.Option(False, "--no-sectors", help="Disable sector features"),
    signals: bool = typer.Option(
        False,
        "--signals",
        help="Also score scan()'s BUY/SELL/HOLD signal: hit rate, turnover, P&L vs. buy-and-hold",
    ),
    cost_bps: float = typer.Option(5.0, help="Per-trade round-trip cost in basis points, with --signals"),
):
    """Walk-forward backtest SYMBOL across multiple rolling origins."""
    try:
        settings = _build_settings(period, no_macro, no_sentiment, no_sectors, config=config)
    except EXPECTED_FAILURES as exc:
        _fail(exc)

    def _run():
        """Run the backtest; a closure so it can be called with or without the spinner."""
        return tft.walk_forward_backtest(
            symbol,
            settings,
            n_windows=windows,
            max_epochs=epochs,
            progress=False,
            compute_signals=signals,
        )

    try:
        if _output.quiet:
            result = _run()
        else:
            console.print(f"[bold cyan]Walk-forward backtesting {symbol.upper()}[/bold cyan]")
            with console.status(f"Training and evaluating {windows} rolling windows..."):
                result = _run()
    except EXPECTED_FAILURES as exc:
        _fail(exc)

    signal_eval = (
        evaluate_signals(result.signals, result.signal_returns_pct, cost_bps=cost_bps)
        if signals
        else None
    )

    if _output.json:
        data = serialize.backtest_to_dict(result)
        if signal_eval is not None:
            data["signal_evaluation"] = serialize.signal_evaluation_to_dict(signal_eval)
        _emit_json(data)
        return

    table = Table(
        title=f"Walk-forward backtest — {result.symbol} ({result.n_windows} windows)",
        show_header=False,
    )
    _add_metric_rows(table, result.aggregated, settings.tft.quantiles, suffix=" (avg)")
    # A bare "57.5% directional accuracy" invites exactly one question -- is that
    # distinguishable from 50%? -- and the answer depends entirely on how many
    # *independent* windows are behind it. Blocks no shorter than the horizon
    # preserve the overlap the naive bootstrap would destroy (PYQ-251).
    if len(result.per_window) > 1:
        low, high = _directional_interval(result, settings.training.max_prediction_length)
        table.add_row("Directional accuracy 95% CI", f"[{low:.1%}, {high:.1%}]")
    console.print(table)
    if len(result.per_window) > 1:
        console.print(_per_window_table(result, settings.tft.quantiles))
    if len(result.aggregated.per_horizon) > 1:
        console.print(
            _per_horizon_table(result.aggregated, settings.tft.quantiles, title="Per-horizon breakdown (pooled)")
        )

    if signal_eval is not None:
        sig_table = Table(title="Signal evaluation (scan's BUY/SELL/HOLD)", show_header=False)
        sig_table.add_row("Signals", f"{signal_eval.n_buy} BUY / {signal_eval.n_sell} SELL / {signal_eval.n_hold} HOLD")
        sig_table.add_row(
            "Hit rate, conditional on firing",
            f"BUY {signal_eval.hit_rate_buy:.1%}  SELL {signal_eval.hit_rate_sell:.1%}",
        )
        sig_table.add_row(
            "Avg. return when fired",
            f"BUY {signal_eval.avg_return_buy_pct:+.2f}%  SELL {signal_eval.avg_return_sell_pct:+.2f}%",
        )
        sig_table.add_row("Turnover", f"{signal_eval.turnover:.1%}")
        sig_table.add_row(f"Strategy P&L (cost {cost_bps:.0f}bps/trade)", f"{signal_eval.strategy_pnl_pct:+.2f}%")
        sig_table.add_row("Buy-and-hold P&L, same period", f"{signal_eval.buy_and_hold_pnl_pct:+.2f}%")
        console.print(sig_table)
        console.print(
            "[dim]Note: thresholds tuned on this same data are a selection event; "
            "the band guard rarely fires without conformal calibration on "
            "(PYQ-248 default is off).[/dim]"
        )


@app.command()
def tune(
    symbol: str = typer.Argument(..., help="Ticker symbol, e.g. AAPL"),
    trials: int = typer.Option(15, "--trials", help="Number of Optuna trials"),
    held_out_days: int = typer.Option(
        None, help="Days reserved for the honest final score (default: TrainingConfig.validation_days)"
    ),
    epochs: int = typer.Option(5, help="Max epochs per trial and for the final retrain"),
    config: Path = typer.Option(
        None, "--config", help="YAML experiment config (see configs/); CLI flags still win"
    ),
    period: str = typer.Option(None, help="History to pull, e.g. 5y, 10y"),
    no_macro: bool = typer.Option(False, "--no-macro", help="Disable macro features"),
    no_sentiment: bool = typer.Option(False, "--no-sentiment", help="Disable news sentiment"),
    no_sectors: bool = typer.Option(False, "--no-sectors", help="Disable sector features"),
):
    """Optuna hyperparameter search for SYMBOL (PYQ-253); writes the winner to configs/.

    Every trial selects on the same data, so the winning configuration is retrained
    and scored on a held-out period no trial ever saw -- report that number, not
    the in-search validation loss, which is optimistically biased.
    """
    try:
        settings = _build_settings(period, no_macro, no_sentiment, no_sectors, config=config)
    except EXPECTED_FAILURES as exc:
        _fail(exc)

    def _run():
        return tft.tune(
            symbol,
            settings,
            n_trials=trials,
            held_out_days=held_out_days,
            max_epochs=epochs,
            progress=not _output.quiet,
        )

    try:
        if _output.quiet:
            result = _run()
        else:
            console.print(f"[bold cyan]Optuna search for {symbol.upper()}: {trials} trial(s)[/bold cyan]")
            result = _run()
    except (*EXPECTED_FAILURES, ImportError) as exc:
        _fail(exc)

    if _output.json:
        _emit_json(
            {
                "symbol": result.symbol,
                "n_trials": result.n_trials,
                "best_params": result.best_params,
                "best_value": result.best_value,
                "held_out_evaluation": serialize.evaluation_to_dict(result.held_out_evaluation),
                "config_path": str(result.config_path),
            }
        )
        return

    table = Table(title=f"Optuna search complete — {result.symbol}", show_header=False)
    table.add_row("Trials", str(result.n_trials))
    table.add_row("Best in-search value (val_loss)", f"{result.best_value:.5f}")
    for name, value in result.best_params.items():
        table.add_row(f"  {name}", f"{value:.4g}" if isinstance(value, float) else str(value))
    table.add_row("Config written to", str(result.config_path))
    console.print(table)
    console.print("[bold]Held-out evaluation (data no trial saw):[/bold]")
    held_out_table = Table(show_header=False)
    _add_metric_rows(held_out_table, result.held_out_evaluation, settings.tft.quantiles)
    console.print(held_out_table)
    console.print(
        "[dim]Note: the in-search value above is a selection-event score, not this "
        "model's real performance -- the held-out numbers are the ones to trust.[/dim]"
    )


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
    try:
        loaded_bundle = tft.load(bundle, settings) if bundle else None
        fc = generate_forecast(symbol, settings, bundle=loaded_bundle, pin=pin)
    except EXPECTED_FAILURES as exc:
        _fail(exc)
    # An options snapshot is live market context, not a model input, so it is
    # skipped entirely when the config asks for it to be (PYQ-125).
    snap = (
        fetch_options_snapshot(fc.symbol)
        if settings.data.use_options
        else OptionsSnapshot(None, None, None, None)
    )

    if _output.json:
        data = serialize.forecast_to_dict(fc)
        if snap.put_call_ratio is not None:
            data["options"] = {
                "expiry": str(snap.expiry),
                "put_call_ratio": snap.put_call_ratio,
                "sentiment_label": snap.sentiment_label,
                "atm_iv": snap.atm_iv,
                "iv_skew": snap.iv_skew,
            }
        _emit_json(data)
        return

    if not _output.quiet:
        console.print(
            Panel(
                f"Current: [bold]${fc.current_price:.2f}[/bold]   "
                f"As of: {fc.last_date.date()}   "
                f"Expected ({fc.horizon}d): {_color_pct(fc.expected_return_pct())}",
                title=f"{fc.symbol}",
            )
        )
    console.print(_forecast_table(fc))
    if fc.n_quantile_crossings:
        console.print(
            f"[yellow]Note:[/yellow] the model produced a non-monotonic band at "
            f"{fc.n_quantile_crossings} point(s); quantiles were reordered for display. "
            "Treat the interval width with caution."
        )

    if snap.put_call_ratio is not None:
        console.print(
            f"[dim]Options ({snap.expiry}): put/call {snap.put_call_ratio:.2f} "
            f"({snap.sentiment_label}), ATM IV {snap.atm_iv:.0%}, skew {snap.iv_skew:+.2%}[/dim]"
        )

    if not no_chart and not _output.quiet:
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
    try:
        bundle = tft.load(bundle_name or symbol, settings)
        interp = explain_forecast(symbol, settings, bundle=bundle)
    except EXPECTED_FAILURES as exc:
        _fail(exc)

    if _output.json:
        _emit_json(serialize.interpretation_to_dict(interp, top=top))
        return

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
    # An interpretation of a model that does not beat persistence describes what
    # it attends to, not what moves the price (investigations.md#pyq-314) -- and
    # a reader will not naturally draw that distinction from the table alone.
    if interp.bundle_skill is not None and interp.bundle_skill <= 0:
        console.print(
            f"[yellow]Note:[/yellow] this bundle's skill vs. persistence is "
            f"{interp.bundle_skill:+.1%} — at or below the naive baseline. The "
            "importances above describe what the model attends to, not "
            "necessarily what moves the price."
        )

    if not no_chart and not _output.quiet:
        charts.importance_chart(interp.top_features(top))
        charts.attention_chart(interp.attention)


@app.command()
def scan(
    symbols: str = typer.Argument(..., help="Comma-separated tickers, e.g. AAPL,MSFT,NVDA"),
):
    """Compare forecasts across multiple trained symbols."""
    settings = load_settings()
    tickers = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    rows: list[dict] = []
    for ticker in tickers:
        try:
            fc = generate_forecast(ticker, settings)
        except FileNotFoundError:
            rows.append({"symbol": ticker, "status": "not_trained"})
            continue
        except Exception as exc:
            # One flaky symbol (a transient data-source error, a bad config,
            # etc.) must not sink the whole multi-symbol comparison (PYQ-113).
            logger.warning("Could not forecast %s: %s", ticker, exc)
            rows.append({"symbol": ticker, "status": "error", "error": str(exc)})
            continue
        # scan_row_to_dict applies the same threshold+guard classify_signal uses
        # elsewhere (PYQ-255) -- shared with the PYQ-261 API's /scan route so the
        # two front-ends cannot drift.
        rows.append(serialize.scan_row_to_dict(ticker, fc))

    if _output.json:
        _emit_json(rows)
        return

    table = Table(title="Multi-asset forecast comparison")
    table.add_column("Symbol")
    table.add_column("Current", justify="right")
    table.add_column("Median target", justify="right")
    table.add_column("Expected", justify="right")
    table.add_column("Band width", justify="right")
    table.add_column("Signal")

    _signal_markup = {"BUY": "[green]BUY[/green]", "SELL": "[red]SELL[/red]", "HOLD": "HOLD"}
    for r in rows:
        if r["status"] == "not_trained":
            table.add_row(r["symbol"], "—", "—", "—", "—", "[dim]not trained[/dim]")
        elif r["status"] == "error":
            table.add_row(r["symbol"], "—", "—", "—", "—", "[red]error[/red]")
        else:
            table.add_row(
                r["symbol"],
                f"${r['current_price']:.2f}",
                f"${r['median_target']:.2f}",
                _color_pct(r["expected_return_pct"]),
                f"{r['band_width_pct']:.1f}%",
                _signal_markup[r["signal"]],
            )
    console.print(table)


@app.command()
def snapshot(
    symbol: str = typer.Argument(..., help="Ticker symbol"),
):
    """Record today's options snapshot for SYMBOL into its accumulated history.

    yfinance exposes only a *current* option chain, not history, so this is the
    only way this project can ever build a historical options-implied series
    (PYQ-254): run it once a day and the recorded file grows into real training
    data. `build_panel` picks it up automatically as `OptionsPutCallRatio`/
    `OptionsATMIV`/`OptionsIVSkew` once enough days have accumulated.
    """
    settings = load_settings()
    path = append_snapshot(symbol, settings)
    if _output.json:
        _emit_json({"symbol": symbol.upper(), "path": str(path)})
        return
    console.print(f"[green]Recorded options snapshot for {symbol.upper()} to {path}[/green]")


cache_app = typer.Typer(
    help="Inspect and prune the local data-panel cache.", no_args_is_help=True
)
app.add_typer(cache_app, name="cache")


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


@cache_app.command("list")
def cache_list():
    """Show cache size, entry count, and saved pins."""
    settings = load_settings()
    stats = data_cache.cache_stats(settings.data.cache_dir)
    if _output.json:
        _emit_json(stats)
        return
    console.print(
        f"Cache dir: {settings.data.cache_dir}\n"
        f"Entries: {stats['entry_count']}  "
        f"Size: {_fmt_bytes(stats['total_bytes'])}"
    )
    if stats["pins"]:
        console.print("Pins: " + ", ".join(stats["pins"]))
    else:
        console.print("[dim]No pins.[/dim]")


@cache_app.command("prune")
def cache_prune():
    """Delete TTL-expired cache entries (pins are never touched)."""
    settings = load_settings()
    removed = data_cache.prune_expired(
        settings.data.cache_dir, settings.data.cache_ttl_seconds
    )
    if _output.json:
        _emit_json({"removed": removed, "count": len(removed)})
        return
    console.print(f"Pruned {len(removed)} expired cache entr{'y' if len(removed) == 1 else 'ies'}.")


@cache_app.command("rm-pin")
def cache_rm_pin(name: str = typer.Argument(..., help="Pin name to remove")):
    """Remove a named dataset pin."""
    settings = load_settings()
    removed = data_cache.remove_pin(settings.data.cache_dir, name)
    if _output.json:
        _emit_json({"removed": removed, "name": name})
        return
    if removed:
        console.print(f"[green]Removed pin '{name}'.[/green]")
    else:
        console.print(f"[yellow]No pin named '{name}'.[/yellow]")


@app.command()
def doctor():
    """Report what is switched on, and whether every bundle is still usable.

    Exits non-zero if any existing bundle's feature schema can no longer be
    satisfied -- so a broken bundle is found by asking, rather than by a
    forecast failing later (PYQ-263).
    """
    from pyquant.analysis.doctor import run_doctor

    settings = load_settings()
    report = run_doctor(settings)

    if _output.json:
        _emit_json(report.to_dict())
        raise typer.Exit(0 if report.healthy else 1)

    def _tick(ok: bool) -> str:
        return "[green]yes[/green]" if ok else "[yellow]no[/yellow]"

    env = Table(title="Environment", show_header=False)
    env.add_row("PyQuant version", report.code_version)
    for key, present in report.keys.items():
        # Presence only, never the value (see the secrets non-negotiable).
        env.add_row(key, _tick(present))
    for extra, present in report.optional_extras.items():
        env.add_row(extra, _tick(present))
    torch_info = report.torch
    if torch_info.get("available"):
        env.add_row("torch", str(torch_info["version"]))
        env.add_row(
            "accelerator",
            f"{torch_info['accelerator']}"
            + (f" ({torch_info['device_name']})" if torch_info.get("device_name") else ""),
        )
        env.add_row("bf16 supported", _tick(torch_info["bf16_supported"]))
    else:
        env.add_row("torch", f"[red]unavailable: {torch_info.get('error')}[/red]")
    for label, value in report.paths.items():
        env.add_row(label, value)
    env.add_row(
        "cache",
        f"{report.cache.get('entry_count', 0)} entries, "
        f"{_fmt_bytes(report.cache.get('total_bytes', 0))}"
        + (f", pins: {', '.join(report.cache['pins'])}" if report.cache.get("pins") else ""),
    )
    console.print(env)

    if not report.bundles:
        console.print("[dim]No trained bundles yet. Run `pyquant train SYMBOL`.[/dim]")
        return

    bundles = Table(title="Bundles")
    for column in ("Bundle", "Symbols", "Trained", "Target", "Features", "Usable"):
        bundles.add_column(column)
    for bundle in report.bundles:
        bundles.add_row(
            bundle.name,
            ", ".join(str(s) for s in bundle.symbols if s),
            (bundle.trained_at or "?")[:10],
            bundle.target or "?",
            str(bundle.n_features),
            "[green]yes[/green]" if bundle.schema_ok else f"[red]no — {bundle.problem}[/red]",
        )
    console.print(bundles)

    if not report.healthy:
        console.print(
            "[red]At least one bundle cannot be satisfied by the current "
            "configuration.[/red] Restore the source/key it names, or retrain it."
        )
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
