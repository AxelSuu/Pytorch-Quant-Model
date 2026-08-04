"""Temporal Fusion Transformer wrapper -- compatibility re-export surface.

All pytorch-forecasting / Lightning calls are isolated to the ``models/``
package so the rest of the codebase stays library-agnostic. This module used
to hold the whole implementation; PYQ-269 split it into five focused
submodules (below) because it had grown to 1075+ lines as the only place
anything Lightning-touching could go. This file now just re-exports every
name external code and the test suite reference, so ``from pyquant.models
import tft`` and ``from pyquant.models.tft import train, load, ...`` keep
resolving unchanged and no bundle produced by a prior version is invalidated.

    models/bundle.py     dataclasses, bundle load/save, provenance, schema check
    models/backtest.py   walk-forward, window geometry, purged cutoff, and the
                          checkpoint/evaluation helpers shared with training.py
    models/training.py   train()
    models/tuning.py     tune() (Optuna)
    models/inference.py  predict_quantiles(), interpret(), permutation_importance()

A caller reaching into ``pyquant.models.tft`` for an *internal* helper (e.g. a
test patching ``tft.build_panel`` before calling ``tft.train(...)``) should
patch the submodule that actually calls it instead -- ``pyquant.models
.training.build_panel`` in that example -- since that is where the name is
looked up at call time now. Patching ``some_module.tft.<public function>``
from outside the ``models`` package (e.g. ``pyquant.api.deps.tft.load``) is
unaffected by the split and continues to work exactly as before.
"""

from __future__ import annotations

from pyquant.models.backtest import (
    _evaluate_best_checkpoint,
    _evaluate_validation,
    _load_best_checkpoint,
    _raw_validation_arrays,
    _selection_split,
    _window_signal,
    _window_validation_dataset,
    build_model,
    purged_training_cutoff,
    walk_forward_backtest,
    walk_forward_backtest_multi_seed,
)
from pyquant.models.bundle import (
    _FEATURE_SOURCE_HINTS,
    BacktestResult,
    FeatureSchemaMismatch,
    ModelBundle,
    SeedSweepResult,
    TrainResult,
    TuneResult,
    _bundle_dir,
    _check_feature_schema,
    _git_sha,
    _package_version,
    _provenance,
    _source_hint,
    bundle_conformal_offset,
    list_bundles,
    load,
    load_meta,
    settings_for_bundle,
)
from pyquant.models.inference import (
    _prediction_dataset,
    interpret,
    permutation_importance,
    predict_quantiles,
)
from pyquant.models.training import (
    _build_pooled_long_df,
    _warn_on_stale_symbols,
    train,
)
from pyquant.models.tuning import _write_tuned_config, tune

__all__ = [
    "_FEATURE_SOURCE_HINTS",
    "BacktestResult",
    "FeatureSchemaMismatch",
    "ModelBundle",
    "SeedSweepResult",
    "TrainResult",
    "TuneResult",
    "_build_pooled_long_df",
    "_bundle_dir",
    "_check_feature_schema",
    "_evaluate_best_checkpoint",
    "_evaluate_validation",
    "_git_sha",
    "_load_best_checkpoint",
    "_package_version",
    "_prediction_dataset",
    "_provenance",
    "_raw_validation_arrays",
    "_selection_split",
    "_source_hint",
    "_warn_on_stale_symbols",
    "_window_signal",
    "_window_validation_dataset",
    "_write_tuned_config",
    "build_model",
    "bundle_conformal_offset",
    "interpret",
    "list_bundles",
    "load",
    "load_meta",
    "permutation_importance",
    "predict_quantiles",
    "purged_training_cutoff",
    "settings_for_bundle",
    "train",
    "tune",
    "walk_forward_backtest",
    "walk_forward_backtest_multi_seed",
]
