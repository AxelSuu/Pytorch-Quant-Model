"""PyQuant command-line interface (Typer + Rich)."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import get_args

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pyquant.analysis import serialize
from pyquant.analysis.forecast import generate_forecast
from pyquant.analysis.interpret import attention_to_series, explain_forecast
from pyquant.analysis.metrics import (
    directional_accuracy_confidence_interval,
    skill_confidence_interval,
)
from pyquant.analysis.signals import evaluate_signals
from pyquant.api import keystore
from pyquant.cli import charts
from pyquant.cli.render import (
    _add_metric_rows,
    _band_label,
    _color_pct,
    _emit_json,
    _fmt_bytes,
    _forecast_table,
    _per_horizon_table,
    _per_window_table,
)
from pyquant.config import DataConfig, Settings, load_settings
from pyquant.data import cache as data_cache
from pyquant.data.options import OptionsSnapshot, append_snapshot, fetch_options_snapshot
from pyquant.experiments.sweep import Arm, run_sweep
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
    provider: str | None = None,
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
    if provider:
        # Validated here, not left to fail inside build_panel: PriceProviderError
        # is a RuntimeError, not one of EXPECTED_FAILURES below, so an unknown
        # name would otherwise surface as an uncaught traceback instead of the
        # one-line message PYQ-120 established for a bad CLI input.
        available = get_args(DataConfig.model_fields["price_provider"].annotation)
        if provider not in available:
            raise ValueError(f"--provider must be one of {list(available)}, got {provider!r}")
        s.data.price_provider = provider
    return s


def _validate_as_of(as_of: str | None) -> None:
    """Reject a malformed ``--as-of`` early with a one-line message.

    PYQ-120's convention: fail with one clean line rather than let it surface
    as a vendor-call traceback deep inside build_panel.
    """
    if as_of is None:
        return
    try:
        date.fromisoformat(as_of)
    except ValueError:
        raise ValueError(f"--as-of must be an ISO date (YYYY-MM-DD), got {as_of!r}") from None


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
    provider: str = typer.Option(
        None, "--provider", help="Price data provider: yfinance (default) or tiingo (PYQ-277)"
    ),
    as_of: str = typer.Option(
        None,
        "--as-of",
        help="Simulate training as of this date (YYYY-MM-DD): data after it is excluded "
        "(PYQ-284). Pass --name too, or this overwrites the symbol's regular bundle.",
    ),
):
    """Train a Temporal Fusion Transformer for SYMBOLS (pooled if more than one)."""
    try:
        settings = _build_settings(
            period, no_macro, no_sentiment, no_sectors, config=config, provider=provider
        )
        _validate_as_of(as_of)
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
            tickers,
            settings,
            bundle_name=name,
            max_epochs=epochs,
            progress=not _output.quiet,
            pin=pin,
            end=as_of,
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


def _run_seed_sweep(
    symbol: str, settings, windows: int, epochs: int | None, signals: bool, seeds: int
) -> None:
    """The `backtest --seeds N` path (PYQ-265).

    Reports mean +/- sd (min, max) skill across N independent walk-forward
    backtests, rather than the single-seed point estimate every number this
    project has otherwise ever reported.
    """

    def _run_sweep():
        """Run the seed sweep; a closure so it can be called with or without the spinner."""
        return tft.walk_forward_backtest_multi_seed(
            symbol,
            settings,
            seeds=list(range(seeds)),
            n_windows=windows,
            max_epochs=epochs,
            progress=False,
            compute_signals=signals,
        )

    try:
        if _output.quiet:
            sweep = _run_sweep()
        else:
            console.print(
                f"[bold cyan]Walk-forward backtesting {symbol.upper()} across {seeds} seeds[/bold cyan]"
            )
            with console.status(
                f"Training and evaluating {windows} rolling window(s) x {seeds} seed(s)..."
            ):
                sweep = _run_sweep()
    except EXPECTED_FAILURES as exc:
        _fail(exc)

    if _output.json:
        _emit_json(serialize.seed_sweep_to_dict(sweep))
        return

    table = Table(
        title=f"Walk-forward backtest — {sweep.symbol} ({seeds} seeds x {windows} windows)",
        show_header=False,
    )
    table.add_row("Seeds", ", ".join(str(s) for s in sweep.seeds))
    table.add_row(
        "Skill vs. baseline (mean ± sd)",
        f"{sweep.skill_mean:+.1%} ± {sweep.skill_sd:.1%} "
        f"(min {sweep.skill_min:+.1%}, max {sweep.skill_max:+.1%})",
    )
    console.print(table)

    quantiles = settings.tft.quantiles
    seed_table = Table(title="Per-seed results", title_style="dim")
    seed_table.add_column("Seed", justify="right")
    seed_table.add_column("Skill", justify="right")
    seed_table.add_column("Directional", justify="right")
    seed_table.add_column(f"Coverage {_band_label(quantiles)}", justify="right")
    for seed, result in zip(sweep.seeds, sweep.per_seed, strict=True):
        seed_table.add_row(
            str(seed),
            f"{result.aggregated.skill_vs_baseline:+.1%}",
            f"{result.aggregated.directional_accuracy:.1%}",
            f"{result.aggregated.calibration_coverage:.1%}",
        )
    console.print(seed_table)


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
    provider: str = typer.Option(
        None, "--provider", help="Price data provider: yfinance (default) or tiingo (PYQ-277)"
    ),
    signals: bool = typer.Option(
        False,
        "--signals",
        help="Also score scan()'s BUY/SELL/HOLD signal: hit rate, turnover, P&L vs. buy-and-hold",
    ),
    cost_bps: float = typer.Option(
        5.0, help="Per-trade round-trip cost in basis points, with --signals"
    ),
    seeds: int = typer.Option(
        1,
        "--seeds",
        help="Repeat the backtest across this many seeds (0..N-1, PYQ-265) and report mean "
        "+/- sd (min, max) skill instead of a single point. Multiplies runtime by this count.",
    ),
):
    """Walk-forward backtest SYMBOL across multiple rolling origins."""
    try:
        settings = _build_settings(
            period, no_macro, no_sentiment, no_sectors, config=config, provider=provider
        )
    except EXPECTED_FAILURES as exc:
        _fail(exc)

    if seeds > 1:
        _run_seed_sweep(symbol, settings, windows, epochs, signals, seeds)
        return

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
    # windows are behind it (PYQ-251). Bootstrapped at the window level, where
    # each entry is already independent (PYQ-270) -- see
    # `directional_accuracy_confidence_interval`'s docstring for why.
    if len(result.per_window) > 1:
        low, high = directional_accuracy_confidence_interval(result.per_window)
        table.add_row("Directional accuracy 95% CI", f"[{low:.1%}, {high:.1%}]")
        # Skill's own interval (PYQ-270): bootstraps the per-window skill
        # series `_per_window_table` prints below, not the pooled headline
        # row above -- the two are different estimators (PYQ-141), so this CI
        # describes the column, not the point estimate it sits next to.
        skill_ci = skill_confidence_interval(result.per_window)
        if skill_ci is not None:
            skill_low, skill_high = skill_ci
            table.add_row(
                "Skill vs. baseline (per-window) 95% CI", f"[{skill_low:+.1%}, {skill_high:+.1%}]"
            )
    console.print(table)
    if len(result.per_window) > 1:
        console.print(_per_window_table(result, settings.tft.quantiles))
        console.print(
            "[dim]Per-window skill is a mean-of-ratios; the pooled headline above is a "
            "ratio-of-means. They are different estimators and do not have to agree "
            "(PYQ-141).[/dim]"
        )
    if len(result.aggregated.per_horizon) > 1:
        console.print(
            _per_horizon_table(
                result.aggregated, settings.tft.quantiles, title="Per-horizon breakdown (pooled)"
            )
        )

    if signal_eval is not None:
        sig_table = Table(title="Signal evaluation (scan's BUY/SELL/HOLD)", show_header=False)
        sig_table.add_row(
            "Signals",
            f"{signal_eval.n_buy} BUY / {signal_eval.n_sell} SELL / {signal_eval.n_hold} HOLD",
        )
        sig_table.add_row(
            "Hit rate, conditional on firing",
            f"BUY {signal_eval.hit_rate_buy:.1%}  SELL {signal_eval.hit_rate_sell:.1%}",
        )
        sig_table.add_row(
            "Avg. return when fired",
            f"BUY {signal_eval.avg_return_buy_pct:+.2f}%  SELL {signal_eval.avg_return_sell_pct:+.2f}%",
        )
        sig_table.add_row("Turnover", f"{signal_eval.turnover:.1%}")
        sig_table.add_row(
            f"Strategy P&L (cost {cost_bps:.0f}bps/trade)", f"{signal_eval.strategy_pnl_pct:+.2f}%"
        )
        sig_table.add_row(
            "Buy-and-hold P&L, same period", f"{signal_eval.buy_and_hold_pnl_pct:+.2f}%"
        )
        console.print(sig_table)
        console.print(
            "[dim]Note: thresholds tuned on this same data are a selection event; "
            "the band guard rarely fires without conformal calibration on "
            "(PYQ-248 default is off).[/dim]"
        )
        if not result.signals_calibrated:
            note = (
                "[dim]Note: these signals are computed from an uncalibrated band -- "
                "walk_forward_backtest() never fits a conformal offset, unlike scan() "
                "against a deployed bundle (PYQ-149)."
            )
            if settings.training.calibration_days > 0:
                note += (
                    f" This settings.yaml/config has calibration_days={settings.training.calibration_days}"
                    ", so a real scan() call would show a different, calibrated band."
                )
            console.print(note + "[/dim]")


@app.command()
def tune(
    symbol: str = typer.Argument(..., help="Ticker symbol, e.g. AAPL"),
    trials: int = typer.Option(15, "--trials", help="Number of Optuna trials"),
    held_out_days: int = typer.Option(
        None,
        help="Days reserved for the honest final score (default: TrainingConfig.validation_days)",
    ),
    epochs: int = typer.Option(5, help="Max epochs per trial and for the final retrain"),
    config: Path = typer.Option(
        None, "--config", help="YAML experiment config (see configs/); CLI flags still win"
    ),
    period: str = typer.Option(None, help="History to pull, e.g. 5y, 10y"),
    no_macro: bool = typer.Option(False, "--no-macro", help="Disable macro features"),
    no_sentiment: bool = typer.Option(False, "--no-sentiment", help="Disable news sentiment"),
    no_sectors: bool = typer.Option(False, "--no-sectors", help="Disable sector features"),
    provider: str = typer.Option(
        None, "--provider", help="Price data provider: yfinance (default) or tiingo (PYQ-277)"
    ),
):
    """Optuna hyperparameter search for SYMBOL (PYQ-253); writes the winner to configs/.

    Every trial selects on the same data, so the winning configuration is retrained
    and scored on a held-out period no trial ever saw -- report that number, not
    the in-search validation loss, which is optimistically biased.
    """
    try:
        settings = _build_settings(
            period, no_macro, no_sentiment, no_sectors, config=config, provider=provider
        )
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
            console.print(
                f"[bold cyan]Optuna search for {symbol.upper()}: {trials} trial(s)[/bold cyan]"
            )
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


@app.command()
def sweep(
    symbols: str = typer.Option(
        ..., "--symbols", help="Comma-separated symbols, e.g. AAPL,MSFT,NVDA"
    ),
    arm: list[str] = typer.Option(
        ...,
        "--arm",
        help="key=value config override defining one arm, e.g. 'target=log_return'; repeat for more arms",
    ),
    windows: int = typer.Option(5, help="Number of rolling walk-forward windows per cell"),
    config: Path = typer.Option(
        None, "--config", help="YAML experiment config (see configs/); CLI flags still win"
    ),
    epochs: int = typer.Option(None, help="Override max training epochs per window"),
    period: str = typer.Option(None, help="History to pull, e.g. 5y, 10y"),
    no_macro: bool = typer.Option(False, "--no-macro", help="Disable macro features"),
    no_sentiment: bool = typer.Option(False, "--no-sentiment", help="Disable news sentiment"),
    no_sectors: bool = typer.Option(False, "--no-sectors", help="Disable sector features"),
    provider: str = typer.Option(
        None, "--provider", help="Price data provider: yfinance (default) or tiingo (PYQ-277)"
    ),
):
    """Walk-forward backtest every symbol against every arm (PYQ-268).

    A multi-symbol repeat of a configuration comparison -- e.g. `--arm
    target=close --arm target=log_return` -- that used to mean editing
    scripts/ablate_features.py or scripts/compare_pooling.py by hand and
    reconciling the output yourself. Reports per-symbol and pooled skill for
    every arm, plus a paired comparison (PYQ-266) between the first two arms
    on every symbol where both succeeded; a symbol that fails for one arm is
    recorded as a gap rather than taking the whole sweep down.
    """
    try:
        settings = _build_settings(
            period, no_macro, no_sentiment, no_sectors, config=config, provider=provider
        )
    except EXPECTED_FAILURES as exc:
        _fail(exc)

    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    arms = []
    for spec in arm:
        if "=" not in spec:
            _fail(ValueError(f"--arm must be key=value, got {spec!r}"))
        key, value = spec.split("=", 1)
        arms.append(Arm(name=spec, overrides={key: value}))

    def _run():
        try:
            return run_sweep(
                symbol_list, arms, settings, n_windows=windows, max_epochs=epochs, progress=False
            )
        except ValueError as exc:  # an --arm override that doesn't resolve to a real field
            _fail(exc)

    if _output.quiet:
        result = _run()
    else:
        console.print(
            f"[bold cyan]Sweeping {len(symbol_list)} symbol(s) x {len(arms)} arm(s)[/bold cyan]"
        )
        with console.status(f"Running {len(symbol_list) * len(arms)} cell(s)..."):
            result = _run()

    if _output.json:
        _emit_json(serialize.sweep_result_to_dict(result))
        return

    table = Table(title="Sweep — skill vs. baseline")
    table.add_column("Symbol")
    for name in result.arm_names:
        table.add_column(name, justify="right")
    for symbol in result.symbols:
        row = [symbol]
        for name in result.arm_names:
            cell = result.cell(symbol, name)
            row.append(
                f"{cell.result.aggregated.skill_vs_baseline:+.1%}"
                if cell.ok
                else "[red]failed[/red]"
            )
        table.add_row(*row)
    console.print(table)

    pooled_table = Table(title="Pooled (unweighted mean across symbols)", show_header=False)
    for name in result.arm_names:
        pooled = result.pooled_skill(name)
        pooled_table.add_row(
            name, f"{pooled:+.1%}" if pooled is not None else "n/a (every symbol failed)"
        )
    console.print(pooled_table)

    # "Helped 11 of 15 symbols" and "mean skill +0.3%" answer different
    # questions; the pooled table above is the second, this is the first.
    if len(result.arm_names) >= 2:
        base, other = result.arm_names[0], result.arm_names[1]
        helped, total = result.helped_summary(base, other)
        console.print(
            f"[dim]{other!r} scored higher than {base!r} on {helped} of {total} symbol(s)[/dim]"
        )
        for symbol in result.symbols:
            comparison = result.paired_comparison(symbol, base, other)
            if comparison is None:
                continue
            verdict = "excludes zero" if comparison.excludes_zero else "does not exclude zero"
            console.print(
                f"[dim]  {symbol}: {base!r} - {other!r} mean diff "
                f"{comparison.mean_diff:+.1%} [{comparison.ci_low:+.1%}, {comparison.ci_high:+.1%}] "
                f"({verdict})[/dim]"
            )


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
    as_of: str = typer.Option(
        None,
        "--as-of",
        help="Simulate forecasting as of this date (YYYY-MM-DD): data after it is excluded "
        "(PYQ-284). Check the printed 'As of' date -- vendors' own end-date conventions "
        "decide the exact last bar, not this flag directly. Cannot combine with --pin.",
    ),
):
    """Forecast SYMBOL with p10/p50/p90 uncertainty bands."""
    settings = load_settings()
    try:
        _validate_as_of(as_of)
        if as_of and pin:
            raise ValueError(
                "--as-of cannot be combined with --pin: a pin already replays a fixed "
                "dataset snapshot, so --as-of would be silently ignored."
            )
        loaded_bundle = tft.load(bundle, settings) if bundle else None
        fc = generate_forecast(symbol, settings, bundle=loaded_bundle, pin=pin, end=as_of)
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
        None,
        "--bundle",
        help="Bundle name to load, if different from SYMBOL (e.g. a pooled bundle)",
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
            f"{interp.bundle_skill:+.1%} — at or below the naive baseline. This is a "
            "point estimate from one held-out validation split, not a walk-forward "
            "backtest (PYQ-270) — run `pyquant backtest` for a confidence interval. "
            "The importances above describe what the model attends to, not "
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


cache_app = typer.Typer(help="Inspect and prune the local data-panel cache.", no_args_is_help=True)
app.add_typer(cache_app, name="cache")


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
    removed = data_cache.prune_expired(settings.data.cache_dir, settings.data.cache_ttl_seconds)
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


keys_app = typer.Typer(
    help="Issue, list, and revoke API keys for pyquant/api/ (PYQ-281).", no_args_is_help=True
)
app.add_typer(keys_app, name="keys")


@keys_app.command("create")
def keys_create(
    name: str = typer.Option(..., "--name", help="A label identifying who/what this key is for"),
    scopes: str = typer.Option(
        "read", "--scopes", help="Comma-separated scopes, e.g. 'read' or 'read,train'"
    ),
):
    """Issue a new API key. The raw value is shown exactly once -- save it now."""
    try:
        raw_key, record = keystore.create_key(
            keystore.resolve_db_path(), name, scopes.split(",")
        )
    except keystore.InvalidScope as exc:
        _fail(exc)
    if _output.json:
        _emit_json(
            {
                "id": record.id,
                "name": record.name,
                "scopes": sorted(record.scopes),
                "key": raw_key,
            }
        )
        return
    console.print(f"[green]Key created for '{name}'.[/green] This is shown once -- save it now:")
    console.print(f"  [bold]{raw_key}[/bold]")
    console.print(f"id={record.id}  scopes={','.join(sorted(record.scopes))}")


@keys_app.command("list")
def keys_list():
    """List issued keys (id, name, scopes, prefix, timestamps) -- never the raw value."""
    records = keystore.list_keys(keystore.resolve_db_path())
    if _output.json:
        _emit_json(
            [
                {
                    "id": r.id,
                    "name": r.name,
                    "prefix": r.prefix,
                    "scopes": sorted(r.scopes),
                    "created_at": r.created_at,
                    "revoked_at": r.revoked_at,
                    "last_used_at": r.last_used_at,
                }
                for r in records
            ]
        )
        return
    if not records:
        console.print("[dim]No API keys issued yet. Run `pyquant keys create --name X`.[/dim]")
        return
    table = Table(title="API keys")
    for column in ("ID", "Name", "Prefix", "Scopes", "Created", "Revoked", "Last used"):
        table.add_column(column)
    for r in records:
        table.add_row(
            r.id,
            r.name,
            r.prefix,
            ",".join(sorted(r.scopes)),
            r.created_at[:10],
            "[red]" + r.revoked_at[:10] + "[/red]" if r.revoked_at else "[dim]—[/dim]",
            r.last_used_at[:10] if r.last_used_at else "[dim]never[/dim]",
        )
    console.print(table)


@keys_app.command("revoke")
def keys_revoke(key_id: str = typer.Argument(..., help="The key id, from `pyquant keys list`")):
    """Revoke a key. A revoked key is rejected by every subsequent request."""
    removed = keystore.revoke_key(keystore.resolve_db_path(), key_id)
    if _output.json:
        _emit_json({"id": key_id, "revoked": removed})
        return
    if removed:
        console.print(f"[green]Revoked key '{key_id}'.[/green]")
    else:
        console.print(f"[yellow]No active key '{key_id}' to revoke.[/yellow]")


if __name__ == "__main__":
    app()
