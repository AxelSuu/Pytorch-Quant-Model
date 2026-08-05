"""Optuna hyperparameter search (PYQ-253).

Split out of ``models/tft.py`` (PYQ-269). Depends on ``models/backtest.py``
(window geometry) and ``models/training.py`` (the final retrain). See
``models/tft.py`` for the compatibility re-export surface.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from lightning.pytorch import seed_everything
from pytorch_forecasting import TimeSeriesDataSet

from pyquant.config import Settings
from pyquant.data.dataset import align_time_index, build_panel, make_dataset, panel_to_long
from pyquant.models.backtest import purged_training_cutoff
from pyquant.models.bundle import TuneResult, _bundle_dir
from pyquant.models.training import train

logger = logging.getLogger(__name__)


def tune(
    symbol: str,
    settings: Settings,
    *,
    n_trials: int = 15,
    held_out_days: int | None = None,
    max_epochs: int = 5,
    progress: bool = False,
) -> TuneResult:
    """Optuna hyperparameter search over the coupled TFT/training knobs (PYQ-253).

    Absorbs PYQ-211's narrower scope (learning-rate-only tuning via
    ``Tuner.lr_find``): learning rate is one of at least six coupled knobs
    (``hidden_size``, ``attention_head_size``, ``dropout``,
    ``hidden_continuous_size``, ``learning_rate``, ``gradient_clip_val``), and
    tuning one in isolation is close to uninformative.

    The search trains and selects entirely within ``df[time_idx < held_out_start]``
    -- the final ``held_out_days`` of the panel are never seen by any trial. The
    winning configuration is then retrained via :func:`train` with
    ``validation_days=held_out_days`` on the *full* panel, so
    ``TuneResult.held_out_evaluation`` is scored on data the search never touched.
    Report that number, not ``best_value`` (the winning trial's own in-search
    validation loss) -- every trial is a selection event, so the in-search score is
    optimistically biased by construction.
    """
    try:
        import optuna
        from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
            optimize_hyperparameters,
        )
    except ImportError as exc:
        raise ImportError("pyquant tune needs the 'tuning' extra: uv sync --extra tuning") from exc

    symbol = symbol.upper()
    seed_everything(settings.training.seed, workers=True)
    panel = build_panel(symbol, settings)
    df = align_time_index(panel_to_long(panel, symbol))

    horizon = settings.training.max_prediction_length
    encoder_len = settings.training.max_encoder_length
    held_out_days = max(
        held_out_days if held_out_days is not None else settings.training.validation_days, horizon
    )
    max_idx = int(df["time_idx"].max())
    held_out_start = max_idx - held_out_days + 1

    search_df = df[df["time_idx"] < held_out_start]
    search_validation_days = max(settings.training.validation_days, horizon)
    search_max_idx = int(search_df["time_idx"].max()) if len(search_df) else -1
    search_validation_start = search_max_idx - search_validation_days + 1
    search_train_cutoff = purged_training_cutoff(search_validation_start - 1, settings)
    if search_train_cutoff <= encoder_len:
        raise ValueError(
            f"Not enough history for {symbol} to reserve {held_out_days} held-out day(s) "
            f"AND run a search-region validation split of {search_validation_days} day(s): "
            f"need more history, or a smaller held_out_days/validation_days."
        )

    training = make_dataset(search_df, settings, training_cutoff=search_train_cutoff)
    validation = TimeSeriesDataSet.from_dataset(
        training, search_df, min_prediction_idx=search_validation_start, stop_randomization=True
    )
    batch_size = settings.training.batch_size
    num_workers = settings.training.num_workers
    train_loader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=num_workers
    )
    val_loader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=num_workers
    )

    bundle_name = f"{symbol}_TUNED"
    bundle_dir = _bundle_dir(settings, bundle_name)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    study_path = bundle_dir / "optuna_study.db"
    study = optuna.create_study(
        study_name=bundle_name,
        storage=f"sqlite:///{study_path}",
        direction="minimize",
        load_if_exists=True,
    )

    with tempfile.TemporaryDirectory() as tmp_model_dir:
        # optimize_hyperparameters() unconditionally adds a LearningRateMonitor
        # callback, which raises unless the Trainer has a logger -- so unlike every
        # other Trainer in this file, `logger` must NOT be forced off here. Its
        # own TensorBoardLogger writes under log_dir, contained to the same
        # temporary directory as the throwaway per-trial checkpoints.
        optimize_hyperparameters(
            train_loader,
            val_loader,
            model_path=tmp_model_dir,
            max_epochs=max_epochs,
            n_trials=n_trials,
            use_learning_rate_finder=False,
            trainer_kwargs={
                "enable_progress_bar": progress,
                "accelerator": "auto",
                "gradient_clip_val": settings.training.gradient_clip_val,
            },
            log_dir=str(Path(tmp_model_dir) / "tb_logs"),
            study=study,
            verbose=1 if progress else 0,
        )

    best_params = dict(study.best_trial.params)
    logger.info(
        "Optuna search for %s: %d trials, best value %.6g", symbol, n_trials, study.best_value
    )

    tuned = settings.model_copy(deep=True)
    for tft_field in ("hidden_size", "hidden_continuous_size", "attention_head_size", "dropout"):
        if tft_field in best_params:
            setattr(tuned.tft, tft_field, best_params[tft_field])
    if "learning_rate" in best_params:
        tuned.training.learning_rate = best_params["learning_rate"]
    if "gradient_clip_val" in best_params:
        tuned.training.gradient_clip_val = best_params["gradient_clip_val"]
    # The final retrain's validation slice IS the honest held-out evaluation: same
    # size as what the search excluded, so train()'s own validation_days-from-the-
    # end logic lands on exactly the region no trial ever saw.
    tuned.training.validation_days = held_out_days
    tuned.training.max_epochs = max_epochs

    result = train(symbol, tuned, bundle_name=bundle_name, progress=progress)

    config_path = _write_tuned_config(bundle_name, tuned, best_params)

    return TuneResult(
        symbol=symbol,
        n_trials=n_trials,
        best_params=best_params,
        best_value=float(study.best_value),
        held_out_evaluation=result.evaluation,
        bundle_dir=bundle_dir,
        config_path=config_path,
    )


def _write_tuned_config(bundle_name: str, tuned: Settings, best_params: dict) -> Path:
    """Write the winning configuration as a YAML file in configs/ (PYQ-209's format)."""
    import yaml

    from pyquant.config import project_root

    configs_dir = project_root() / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    config_path = configs_dir / f"{bundle_name.lower()}_tuned.yaml"
    payload = {
        "tft": {
            "hidden_size": tuned.tft.hidden_size,
            "hidden_continuous_size": tuned.tft.hidden_continuous_size,
            "attention_head_size": tuned.tft.attention_head_size,
            "dropout": tuned.tft.dropout,
        },
        "training": {
            "learning_rate": tuned.training.learning_rate,
            "gradient_clip_val": tuned.training.gradient_clip_val,
        },
    }
    header = (
        f"# PyQuant experiment config -- Optuna search winner for {bundle_name} (PYQ-253).\n"
        f"# Trial params: {best_params}\n"
        "# The in-search validation loss that selected this configuration is optimistically\n"
        "# biased (every trial is a selection event) -- see the bundle's meta.json for its\n"
        "# score on data the search never saw, which is the number to trust.\n"
    )
    config_path.write_text(header + yaml.dump(payload, sort_keys=False))
    return config_path
