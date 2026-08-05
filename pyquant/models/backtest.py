"""Walk-forward backtesting, window geometry, and shared checkpoint/evaluation helpers.

Split out of ``models/tft.py`` (PYQ-269). Window geometry
(``purged_training_cutoff``, ``_selection_split``) and the walk-forward loop
live here per the ticket's proposed layout. The checkpoint-loading and
validation-evaluation helpers (``build_model`` through ``_evaluate_validation``)
are *also* here rather than in ``training.py``, which is a deliberate deviation
from the ticket's suggested split: ``walk_forward_backtest`` needs every one of
them exactly as much as ``training.train`` does (each origin trains, then
evaluates its own best checkpoint), and ``training.py`` already has to import
this module for the window geometry. Putting the checkpoint/eval helpers in
``training.py`` instead would make this module import back from ``training.py``
-- a cycle. Keeping them here means ``training.py`` depends one-directionally
on ``backtest.py``, never the reverse. See ``models/tft.py`` for the
compatibility re-export surface.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

from pyquant.analysis.calibrate import ConformalOffset, apply_conformal_offset
from pyquant.analysis.metrics import EvaluationMetrics, aggregate_metrics, evaluate_predictions
from pyquant.config import Settings
from pyquant.data.dataset import (
    align_time_index,
    build_panel,
    make_dataset,
    panel_to_long,
    target_column,
)
from pyquant.models.bundle import BacktestResult, SeedSweepResult

logger = logging.getLogger(__name__)


def build_model(
    training_dataset: TimeSeriesDataSet, settings: Settings
) -> TemporalFusionTransformer:
    """Construct a TFT sized to the dataset and config."""
    return TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=settings.training.learning_rate,
        hidden_size=settings.tft.hidden_size,
        attention_head_size=settings.tft.attention_head_size,
        dropout=settings.tft.dropout,
        hidden_continuous_size=settings.tft.hidden_continuous_size,
        loss=QuantileLoss(quantiles=settings.tft.quantiles),
        log_interval=-1,
        optimizer="adam",
    )


def purged_training_cutoff(cutoff: int, settings: Settings) -> int:
    """Last ``time_idx`` a *training* sample may decode, given purge + embargo.

    ``cutoff`` is where the held-out period begins minus one. Left alone, the
    last training samples decode the days immediately before it -- and a
    validation sample starting at ``cutoff + 1`` reads exactly those days
    through its own encoder. Training and evaluation therefore share target
    days across the boundary, which biases reported out-of-sample performance
    optimistically. The financial-ML standard treatment (López de Prado,
    *Advances in Financial Machine Learning*) is to **purge** one label horizon
    either side of the split and then **embargo** a further buffer, because
    serial correlation carries information across the boundary even where no
    literal overlap remains (PYQ-250).

    This is the last known member of the leak family PYQ-101/103/115/116/123/127
    belong to, and the one the literature considers table stakes.

    Returns the reduced cutoff; callers keep using the original for the
    validation window, which must not move.
    """
    horizon = settings.training.max_prediction_length
    purge = settings.training.purge_horizon
    purge = horizon if purge is None else int(purge)
    return cutoff - max(0, purge) - max(0, settings.training.embargo_days)


def _selection_split(cutoff: int, settings: Settings) -> tuple[int, int, int]:
    """Carve a purged selection window out of the training tail (PYQ-143).

    ``cutoff`` plays the same role it plays in `purged_training_cutoff`: the
    last index before whatever comes after selection (calibration + the test
    window in `train()`; the single test window in `walk_forward_backtest`).
    Returns ``(train_cutoff, selection_start, selection_end)``.

    Applies `purged_training_cutoff` twice -- once to leave the existing
    purge+embargo gap between selection and what follows it, once more to
    leave an equal gap between training and selection -- so
    EarlyStopping/ModelCheckpoint select against a window that shares no
    target days, even purged ones, with either the training data or the
    window `EvaluationMetrics` is finally reported from. Before this, both
    used the same window: a selection-event bias identical in kind to the one
    `TuneResult`'s own docstring names for Optuna trials, just applied to
    epochs instead of trials.
    """
    selection_end = purged_training_cutoff(cutoff, settings)
    selection_start = selection_end - settings.training.selection_days + 1
    train_cutoff = purged_training_cutoff(selection_start - 1, settings)
    return train_cutoff, selection_start, selection_end


def _load_best_checkpoint(
    best_model_path: str | Path | None, live_model: TemporalFusionTransformer
) -> TemporalFusionTransformer:
    """The best-epoch checkpoint, falling back to the live model.

    EarlyStopping does not rewind the live model's weights to the best epoch --
    ModelCheckpoint does, and it must be reloaded explicitly (as tft.load() does
    for forecast/explain). Reload the saved best checkpoint so reported metrics
    reflect the model that actually gets deployed, not the worse final one
    (PYQ-109). Falls back to the live model only if no checkpoint was written.
    """
    best_model_path_p = Path(best_model_path) if best_model_path else None
    if best_model_path_p and best_model_path_p.exists():
        return TemporalFusionTransformer.load_from_checkpoint(
            str(best_model_path_p), map_location="cpu"
        )
    logger.warning("No best checkpoint found; evaluating the live post-fit model instead.")
    return live_model


def _evaluate_best_checkpoint(
    best_model_path: str | Path | None,
    live_model: TemporalFusionTransformer,
    val_loader,
    quantiles: list[float],
    target: str = "Close",
    conformal: ConformalOffset | None = None,
) -> EvaluationMetrics:
    """Evaluate the best-epoch checkpoint (PYQ-109), applying any conformal offset."""
    model = _load_best_checkpoint(best_model_path, live_model)
    return _evaluate_validation(model, val_loader, quantiles, target, conformal=conformal)


def _raw_validation_arrays(model: TemporalFusionTransformer, loader):
    """(predictions, actuals, last_observed) for a loader, all in target units.

    PYQ-313 verified against pytorch-forecasting 1.7.0 that these three come back
    in the target's own space rather than the normalizer's; PYQ-240's test pins
    it. Extracted so the calibration slice and the validation window are read
    exactly the same way.
    """
    result = model.predict(
        loader,
        mode="quantiles",
        return_x=True,
        return_y=True,
        trainer_kwargs={"enable_progress_bar": False, "logger": False},
    )
    return (
        result.output.cpu().numpy(),
        result.y[0].cpu().numpy(),
        result.x["encoder_target"][:, -1].cpu().numpy(),
        # The full encoder window, not just its last value -- what
        # analysis.baselines' comparators beyond persistence need (PYQ-275).
        # Already computed as part of the same forward pass, so this is free.
        result.x["encoder_target"].cpu().numpy(),
    )


def _evaluate_validation(
    model: TemporalFusionTransformer,
    val_loader,
    quantiles: list[float],
    target: str = "Close",
    conformal: ConformalOffset | None = None,
) -> EvaluationMetrics:
    """Score the held-out validation window vs. baselines beyond persistence (PYQ-275)."""
    predictions, actuals, last_observed, history = _raw_validation_arrays(model, val_loader)
    # Score the band the user will actually be shown. Reporting coverage for an
    # uncalibrated band while `forecast` prints a calibrated one would make the
    # published number describe something nobody sees (PYQ-248).
    predictions = apply_conformal_offset(predictions, conformal)
    return evaluate_predictions(
        predictions,
        actuals,
        last_observed,
        quantiles,
        target="log_return" if target == "LogReturn" else "close",
        history=history,
    )


def _window_validation_dataset(
    training: TimeSeriesDataSet, df: pd.DataFrame, cutoff: int, horizon: int
) -> TimeSeriesDataSet:
    """Validation set for one walk-forward origin: exactly the horizon after ``cutoff``.

    ``predict=True`` anchors the decoder to the last ``horizon`` timesteps of
    whatever frame it is handed -- so passing the *full* df made every rolling
    origin evaluate the identical final window, and a 5-window backtest was five
    differently-trained models scored on the same five days (PYQ-127). Truncating
    the frame at ``cutoff + horizon`` puts the decoder on that origin's own
    out-of-sample window instead.
    """
    window = df[df["time_idx"] <= cutoff + horizon]
    return TimeSeriesDataSet.from_dataset(training, window, predict=True, stop_randomization=True)


def _window_signal(
    predictions: np.ndarray,
    actuals: np.ndarray,
    last_observed: np.ndarray,
    quantiles: list[float],
    target: str,
) -> tuple[str, float]:
    """The (signal, realized_return_pct) scan() would have shown for one window.

    Derived from that walk-forward window's raw prediction/actual arrays
    (PYQ-255). walk_forward_backtest is single-symbol, so each window's arrays
    hold exactly one sample.

    True only when ``TrainingConfig.calibration_days == 0`` (PYQ-149):
    ``predictions`` here is never conformal-offset the way `predict_quantiles`
    offsets a deployed bundle's band (PYQ-248) -- there is no calibration fit
    anywhere in `walk_forward_backtest`. With calibration on, this reads a
    systematically different (uncalibrated) band than a real `scan()` call
    against an equivalent bundle would show; see `BacktestResult
    .signals_calibrated`.
    """
    from pyquant.analysis.forecast import (
        log_return_quantiles_to_price_band,
        log_returns_to_prices,
    )
    from pyquant.analysis.signals import classify_signal

    median_idx = quantiles.index(0.5)
    pred, actual, last_obs = predictions[0], actuals[0], float(last_observed[0])

    if target == "LogReturn":
        # actual is a single realized path -- plain compounding is exact for it.
        # pred is a (horizon, n_quantiles) band -- needs the PYQ-142 reconstruction.
        band = log_return_quantiles_to_price_band(pred, last_obs, quantiles)
        median_path, lower_path, upper_path = band[:, median_idx], band[:, 0], band[:, -1]
        actual_path = log_returns_to_prices(actual, last_obs)
    else:
        median_path, lower_path, upper_path = pred[:, median_idx], pred[:, 0], pred[:, -1]
        actual_path = actual

    expected_pct = float((median_path[-1] - last_obs) / last_obs * 100)
    lower_pct = float((lower_path[-1] - last_obs) / last_obs * 100)
    upper_pct = float((upper_path[-1] - last_obs) / last_obs * 100)
    realized_pct = float((actual_path[-1] - last_obs) / last_obs * 100)

    return classify_signal(expected_pct, lower_pct, upper_pct), realized_pct


def walk_forward_backtest(
    symbol: str,
    settings: Settings,
    *,
    n_windows: int = 5,
    step: int | None = None,
    start: str | None = None,
    end: str | None = None,
    max_epochs: int | None = None,
    progress: bool = False,
    compute_signals: bool = False,
) -> BacktestResult:
    """Train/evaluate across many rolling origins (walk-forward validation).

    Unlike train(), each window's model is discarded after evaluation -- this
    measures how stable the metrics are across time (see PYQ-303), it does not
    produce a deployable bundle. ``compute_signals`` additionally records, per
    window, the BUY/SELL/HOLD signal scan() would have shown and the realized
    return -- one extra forward pass per window, so it defaults off (PYQ-255).
    See ``BacktestResult.signals_calibrated`` (PYQ-149): this never applies a
    conformal offset, so with ``calibration_days > 0`` the signals measured
    here diverge from what a calibrated bundle's ``scan()`` would show.
    """
    if compute_signals and settings.training.calibration_days > 0:
        logger.warning(
            "walk_forward_backtest(compute_signals=True) with calibration_days=%d: "
            "signals are computed from an uncalibrated band (PYQ-149) -- this measures "
            "a systematically different case than scan() would show for a bundle "
            "trained with calibration on. See BacktestResult.signals_calibrated.",
            settings.training.calibration_days,
        )
    symbol = symbol.upper()
    seed_everything(settings.training.seed, workers=True)
    panel = build_panel(symbol, settings, start, end)
    df = align_time_index(panel_to_long(panel, symbol))

    horizon = settings.training.max_prediction_length
    encoder_len = settings.training.max_encoder_length
    step = step if step is not None else horizon
    if compute_signals and step < horizon:
        # analysis.signals.evaluate_signals/_compound treats each window's realized
        # return as one sequential, non-overlapping "trade" and compounds them
        # multiplicatively -- correct only when consecutive windows don't share
        # calendar days. step < horizon means they do, so the resulting
        # strategy_pnl_pct/buy_and_hold_pnl_pct double-counts the overlapping days
        # and doesn't correspond to any real trading history (bugs.md#pyq-328).
        # Currently unreachable from the CLI (no --step flag), but reachable from
        # this function's own Python API -- fail loudly rather than let a future
        # caller silently get a wrong number.
        raise ValueError(
            f"walk_forward_backtest(compute_signals=True) requires step >= horizon "
            f"to avoid double-counting overlapping windows in the P&L accounting "
            f"(got step={step}, horizon={horizon})"
        )
    max_idx = int(df["time_idx"].max())

    latest_cutoff = max_idx - horizon
    earliest_cutoff = latest_cutoff - (n_windows - 1) * step
    # Purge + embargo eat into the *earliest* origin's training slice, so the
    # history check has to account for them or the first window fails mid-run
    # rather than up front (PYQ-250). Each origin also carves a selection
    # window (PYQ-143) out of its own training tail, purged on both sides --
    # account for that too.
    earliest_train_cutoff, _, earliest_selection_end = _selection_split(earliest_cutoff, settings)
    if earliest_train_cutoff <= encoder_len:
        gap = earliest_cutoff - earliest_selection_end  # width of one purge+embargo gap
        selection_days = settings.training.selection_days
        raise ValueError(
            f"Not enough history for {n_windows} walk-forward window(s) of {symbol}: "
            f"need more than {encoder_len + horizon + (n_windows - 1) * step + 2 * gap + selection_days} "
            f"rows (two {gap}-day purged/embargoed gaps around a {selection_days}-day selection "
            f"window, before each origin), got {len(df)}."
        )

    cutoffs = sorted(latest_cutoff - i * step for i in range(n_windows))
    epochs = max_epochs if max_epochs is not None else settings.training.max_epochs
    batch_size = settings.training.batch_size

    num_workers = settings.training.num_workers
    per_window: list[EvaluationMetrics] = []
    signals: list[str] = []
    signal_returns_pct: list[float] = []
    for cutoff in cutoffs:
        # This origin's own selection window (PYQ-143): carved from its
        # training tail, purged from both training and the single test
        # window below, the same [train][purge][selection][purge][test]
        # shape train() uses.
        train_cutoff, selection_start, selection_end = _selection_split(cutoff, settings)
        training = make_dataset(df, settings, training_cutoff=train_cutoff)
        validation = _window_validation_dataset(training, df, cutoff, horizon)
        selection_df = df[df["time_idx"] <= selection_end + horizon - 1]
        selection = TimeSeriesDataSet.from_dataset(
            training, selection_df, min_prediction_idx=selection_start, stop_randomization=True
        )
        train_loader = training.to_dataloader(
            train=True, batch_size=batch_size, num_workers=num_workers
        )
        val_loader = validation.to_dataloader(
            train=False, batch_size=batch_size, num_workers=num_workers
        )
        selection_loader = selection.to_dataloader(
            train=False, batch_size=batch_size, num_workers=num_workers
        )

        model = build_model(training, settings)
        # Checkpoint into a throwaway dir: the backtest discards each window's
        # model, but it must still evaluate that window's *best* epoch rather
        # than its final (post-EarlyStopping) one, same as train() (PYQ-109).
        with tempfile.TemporaryDirectory() as tmp_ckpt_dir:
            ckpt_cb = ModelCheckpoint(
                dirpath=tmp_ckpt_dir, filename="best", monitor="val_loss", save_top_k=1, mode="min"
            )
            trainer = Trainer(
                max_epochs=epochs,
                accelerator="auto",
                precision=settings.training.precision,
                gradient_clip_val=settings.training.gradient_clip_val,
                callbacks=[
                    ckpt_cb,
                    EarlyStopping(
                        monitor="val_loss",
                        patience=settings.training.early_stopping_patience,
                        mode="min",
                    ),
                ],
                logger=False,
                enable_progress_bar=progress,
                enable_model_summary=False,
            )
            # val_dataloaders drives EarlyStopping/ModelCheckpoint --
            # selection_loader, never val_loader, so checkpoint choice is not
            # a selection event scored against this origin's reported window
            # (PYQ-143).
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=selection_loader)
            per_window.append(
                _evaluate_best_checkpoint(
                    ckpt_cb.best_model_path,
                    model,
                    val_loader,
                    settings.tft.quantiles,
                    target_column(settings),
                )
            )
            if compute_signals:
                # A second forward pass rather than reusing _evaluate_best_checkpoint's:
                # that helper returns only the aggregated EvaluationMetrics, not the raw
                # arrays a signal needs, and this stays opt-in specifically to keep the
                # default backtest path (no signals) at its current one-pass cost.
                best_model = _load_best_checkpoint(ckpt_cb.best_model_path, model)
                predictions, actuals, last_observed, _ = _raw_validation_arrays(
                    best_model, val_loader
                )
                signal, realized_pct = _window_signal(
                    predictions,
                    actuals,
                    last_observed,
                    settings.tft.quantiles,
                    target_column(settings),
                )
                signals.append(signal)
                signal_returns_pct.append(realized_pct)

    return BacktestResult(
        symbol=symbol,
        n_windows=len(cutoffs),
        per_window=per_window,
        aggregated=aggregate_metrics(per_window),
        signals=signals,
        signal_returns_pct=signal_returns_pct,
        origins=list(cutoffs),
    )


def walk_forward_backtest_multi_seed(
    symbol: str,
    settings: Settings,
    *,
    seeds: list[int] | None = None,
    n_windows: int = 5,
    step: int | None = None,
    start: str | None = None,
    end: str | None = None,
    max_epochs: int | None = None,
    progress: bool = False,
    compute_signals: bool = False,
) -> SeedSweepResult:
    """Repeat `walk_forward_backtest` once per seed (PYQ-265).

    ``seeds`` defaults to ``settings.training.seeds`` (itself defaulting to a
    single-element list, so nothing changes unless a caller opts in -- the
    same shape PYQ-248 shipped conformal calibration defaulted off). Every
    other argument means exactly what it means on `walk_forward_backtest`,
    applied identically across seeds.

    Each seed gets its own deep-copied `Settings` (only `training.seed`
    differs) rather than mutating the caller's object, and its own call to
    `build_panel` inside `walk_forward_backtest` -- the panel is identical
    across seeds, so this re-fetches/re-builds it `len(seeds)` times. Left
    as-is rather than threading a pre-built panel through
    `walk_forward_backtest`'s signature: that is a real inefficiency, but
    fixing it means changing a function three other callers already share,
    for a cost that is a caching problem (`DataConfig.cache_enabled`,
    PYQ-205) rather than a correctness one.

    Cost is the obvious objection and should be stated rather than hidden:
    this multiplies training time by ``len(seeds)``. That is the correct
    price for the claim investigations.md#pyq-321 needs answered.
    """
    chosen_seeds = list(seeds) if seeds is not None else list(settings.training.seeds)
    per_seed: list[BacktestResult] = []
    for seed in chosen_seeds:
        seed_settings = settings.model_copy(deep=True)
        seed_settings.training.seed = seed
        per_seed.append(
            walk_forward_backtest(
                symbol,
                seed_settings,
                n_windows=n_windows,
                step=step,
                start=start,
                end=end,
                max_epochs=max_epochs,
                progress=progress,
                compute_signals=compute_signals,
            )
        )
    return SeedSweepResult(symbol=symbol.upper(), seeds=chosen_seeds, per_seed=per_seed)
