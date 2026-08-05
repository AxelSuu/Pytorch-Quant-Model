"""Bundle dataclasses, persistence, provenance, and feature-schema checking.

Split out of ``models/tft.py`` (PYQ-269) as the dependency-free base of the
``models/`` package: every other submodule (``backtest``, ``training``,
``tuning``, ``inference``) imports from here, and this module imports from
none of them, so there is no import cycle to manage. See ``models/tft.py``
for the compatibility re-export surface and each submodule's own docstring
for the rest of the split.

A trained model is persisted as a bundle directory under ``checkpoints/<bundle_name>/``
(``<bundle_name>`` is the symbol, or the joined symbol list for pooled training):
    model.ckpt          Lightning checkpoint (architecture + weights)
    dataset_params.pt   TimeSeriesDataSet parameters (encoders/normalizers)
    meta.json           symbol, feature names, metrics, training timestamp
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer

from pyquant import provenance
from pyquant.analysis.calibrate import ConformalOffset
from pyquant.analysis.metrics import EvaluationMetrics
from pyquant.config import Settings
from pyquant.data.dataset import SCHEMA_DATA_FIELDS, feature_columns

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    symbols: list[str]
    bundle_dir: Path
    # The best checkpoint's loss on the *selection* window (PYQ-143), not on
    # the window `evaluation` below is computed from -- it is what
    # EarlyStopping/ModelCheckpoint monitored, so it is a selection-event
    # statistic and optimistically biased by construction (the same reason
    # `TuneResult.best_value` below is not `held_out_evaluation`). Useful for
    # judging whether training converged, not as a quality number.
    val_loss: float
    n_features: int
    epochs_run: int
    evaluation: EvaluationMetrics


@dataclass
class BacktestResult:
    symbol: str
    n_windows: int
    per_window: list[EvaluationMetrics]
    aggregated: EvaluationMetrics
    # Populated only when walk_forward_backtest(..., compute_signals=True): the
    # BUY/SELL/HOLD scan() would have emitted at each origin, and the realized
    # percent move over that origin's horizon (PYQ-255). Kept optional rather
    # than always computed -- it costs one extra forward pass per window.
    signals: list[str] = field(default_factory=list)
    signal_returns_pct: list[float] = field(default_factory=list)
    # Always False (PYQ-149): `_window_signal` reads straight from
    # `_raw_validation_arrays`, with no conformal fit/offset step anywhere in
    # `walk_forward_backtest` -- unlike `scan()`, which applies the trained
    # bundle's calibrated offset (PYQ-248) to every band it shows. With
    # `TrainingConfig.calibration_days > 0` this evaluation measures a
    # different (pessimistic, uncalibrated) band than a deployed bundle's
    # signals would show; a caller must not read `signals`/`signal_returns_pct`
    # as reproducing `scan()`'s behaviour when that's set. Kept as an explicit
    # field rather than only a docstring/log warning so a JSON/programmatic
    # consumer doesn't have to know to go looking for the caveat in prose.
    signals_calibrated: bool = False
    # This backtest's window origins (each window's `cutoff`, i.e. its decoder
    # starts at `cutoff + 1`), in the same order as `per_window` (PYQ-266). Lets
    # `analysis.metrics.compare_backtests` verify two results were scored on
    # literally the same walk-forward windows before treating their per-window
    # differences as paired -- comparing unlike windows is the failure mode a
    # paired test exists to rule out, not something to trust by convention.
    origins: list[int] = field(default_factory=list)


@dataclass
class SeedSweepResult:
    """Multiple independent walk-forward backtests of the same configuration, one per seed.

    Every number this project has ever reported is one draw from one seed
    (`TrainingConfig.seed`, fixed at 42); this is the tooling half of
    investigations.md#pyq-321's question -- how much of that number is
    initialisation noise rather than signal (PYQ-265). `per_seed` retains
    every individual `BacktestResult`, not just the summary below, so
    `analysis.metrics.compare_backtests` can consume them pairwise -- e.g.
    comparing seed-by-seed across two different configurations, once such a
    comparison is wanted.
    """

    symbol: str
    seeds: list[int]
    per_seed: list[BacktestResult]

    @property
    def _skills(self) -> list[float]:
        return [result.aggregated.skill_vs_baseline for result in self.per_seed]

    @property
    def skill_mean(self) -> float:
        """Mean skill vs. baseline across seeds."""
        return float(np.mean(self._skills))

    @property
    def skill_sd(self) -> float:
        """Sample standard deviation (ddof=1); 0.0 for a single seed."""
        return float(np.std(self._skills, ddof=1)) if len(self._skills) > 1 else 0.0

    @property
    def skill_min(self) -> float:
        """Minimum skill vs. baseline across seeds."""
        return float(np.min(self._skills))

    @property
    def skill_max(self) -> float:
        """Maximum skill vs. baseline across seeds."""
        return float(np.max(self._skills))


@dataclass
class TuneResult:
    """An Optuna hyperparameter search (PYQ-253), plus its winner's honest score.

    ``held_out_evaluation`` comes from data the search never trained or selected
    on -- every trial is a selection event, so the in-search score
    (``best_value``, the pruned/selected trial's own validation loss) is
    optimistically biased and must not be reported as the model's real
    performance.
    """

    symbol: str
    n_trials: int
    best_params: dict
    best_value: float
    held_out_evaluation: EvaluationMetrics
    bundle_dir: Path
    config_path: Path


class FeatureSchemaMismatch(RuntimeError):
    """A bundle's trained features are not all present in the freshly built panel.

    Raised instead of letting pytorch-forecasting fail with a bare ``KeyError``
    naming one column and nothing else (PYQ-118).
    """


# Which source each feature family comes from, so a missing column can say what
# went away rather than just which key was absent.
_FEATURE_SOURCE_HINTS: tuple[tuple[str, str], ...] = (
    ("SEC_", "sector ETF returns (DataConfig.use_sectors; Yahoo Finance)"),
    ("VIX", "macro context (DataConfig.use_macro; Yahoo Finance, no key needed)"),
    ("FedFunds", "FRED macro series (DataConfig.use_macro + FRED_API_KEY)"),
    ("YieldSpread", "FRED macro series (DataConfig.use_macro + FRED_API_KEY)"),
    ("CPI", "FRED macro series (DataConfig.use_macro + FRED_API_KEY)"),
    (
        "Sentiment",
        "news sentiment (DataConfig.use_sentiment + FINNHUB_API_KEY + 'sentiment' extra)",
    ),
    (
        "HeadlineCount",
        "news sentiment (DataConfig.use_sentiment + FINNHUB_API_KEY + 'sentiment' extra)",
    ),
)


def _source_hint(column: str) -> str:
    for prefix, hint in _FEATURE_SOURCE_HINTS:
        if column.startswith(prefix):
            return hint
    return "price data / technical indicators"


@dataclass
class ModelBundle:
    """A loaded model plus everything needed to forecast/interpret with it."""

    model: TemporalFusionTransformer
    dataset_params: dict
    meta: dict


def _bundle_dir(settings: Settings, name: str) -> Path:
    """The on-disk directory for bundle ``name``, guaranteed inside ``checkpoint_dir``.

    Belt and braces beneath the API layer's request-schema validation
    (PYQ-145): a ``name`` like ``"../../etc"`` reaches this function directly
    from a POST body (unlike a GET route's ``{symbol}``, a JSON body field
    isn't subject to Starlette's `/`-rejecting path-parameter matching), and
    from here flows straight into ``mkdir``/``torch.save``. Checked here too so
    every caller is covered, not only the ones that remembered to validate
    first.
    """
    bundle_dir = settings.checkpoint_dir / name.upper()
    checkpoint_root = settings.checkpoint_dir.resolve()
    resolved = bundle_dir.resolve()
    if resolved != checkpoint_root and checkpoint_root not in resolved.parents:
        raise ValueError(f"Invalid bundle name {name!r}: resolves outside checkpoint_dir")
    return bundle_dir


# These were duplicated here so data/cache.py could stamp a pin without importing
# the ML stack. Two copies meant PYQ-134's "an unrelated repo's sha is recorded as
# PyQuant's provenance" had to be fixed twice; delegate instead so it cannot drift.
_package_version = provenance.package_version
_git_sha = provenance.git_sha


def _provenance(pin: str | None) -> dict:
    """What is needed to reproduce this run, beyond the seed and the data.

    PYQ-210 recorded the seed and PYQ-205 added pinned datasets, but neither
    captured *which code* ran -- and feature definitions do change (PYQ-121
    redefined RSI_14). Version + sha + pin is the set that actually reproduces a
    bundle (PYQ-225).
    """
    return {"pyquant_version": _package_version(), "git_sha": _git_sha(), "pin": pin}


def _check_feature_schema(bundle: ModelBundle, df: pd.DataFrame) -> None:
    """Fail clearly if the panel is missing features the bundle was trained on.

    build_panel()'s per-source graceful degradation means the column set depends
    on which optional sources succeeded on that particular call. Extra columns are
    harmless (from_parameters ignores them), but a *missing* one used to surface as
    a bare ``KeyError`` from deep inside pytorch-forecasting, with no hint that a
    data source had gone away (PYQ-118).
    """
    expected = list(bundle.meta.get("features") or [])
    if not expected:
        return
    available = set(feature_columns(df))
    missing = [c for c in expected if c not in available]
    if not missing:
        return
    details = "\n".join(f"  - {c}  (from: {_source_hint(c)})" for c in missing)
    raise FeatureSchemaMismatch(
        f"The data panel is missing {len(missing)} of the {len(expected)} feature(s) "
        f"this bundle was trained on:\n{details}\n"
        "The model cannot be used without them. Either restore the source (check the "
        "API key / network, or the matching DataConfig toggle) or retrain the bundle "
        "against the feature set you have now."
    )


def bundle_conformal_offset(bundle: ModelBundle) -> ConformalOffset | None:
    """The conformal band correction recorded at train time, if any (PYQ-248)."""
    recorded = bundle.meta.get("conformal")
    return ConformalOffset.from_dict(recorded) if recorded else None


def settings_for_bundle(bundle: ModelBundle, settings: Settings) -> Settings:
    """Return a copy of ``settings`` using the data toggles ``bundle`` was trained with.

    A bundle's feature schema is decided by which sources were enabled at train
    time. Rebuilding the prediction panel from whatever the current defaults happen
    to be is precisely how the PYQ-118 mismatch gets triggered: train with
    ``--no-sectors`` and forecast without it, and the schemas differ by
    construction rather than by bad luck (PYQ-119).

    Bundles trained before this was recorded simply keep the caller's settings.
    """
    recorded = (bundle.meta.get("config") or {}).get("data") or {}
    if not recorded:
        return settings
    restored = settings.model_copy(deep=True)
    for field_name in SCHEMA_DATA_FIELDS:
        if field_name in recorded:
            setattr(restored.data, field_name, recorded[field_name])
    return restored


def load(symbol: str, settings: Settings) -> ModelBundle:
    """Load a trained bundle for ``symbol``."""
    bundle_dir = _bundle_dir(settings, symbol)
    ckpt = bundle_dir / "model.ckpt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No trained model for {symbol.upper()} at {ckpt}. Run `pyquant train` first."
        )
    model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt), map_location="cpu")
    # weights_only=False is required: get_parameters() serializes
    # pytorch-forecasting normalizers/encoders that are not on PyTorch's
    # safe-unpickling allowlist, so weights_only=True raises UnpicklingError
    # (verified: PYQ-306). This deserialization can execute arbitrary code, so
    # only ever load bundles from your own trusted training runs -- the same
    # trust boundary relied on by the pickle panel cache in pyquant.data.cache.
    dataset_params = torch.load(bundle_dir / "dataset_params.pt", weights_only=False)
    meta = json.loads((bundle_dir / "meta.json").read_text())
    return ModelBundle(model=model, dataset_params=dataset_params, meta=meta)


def load_meta(symbol: str, settings: Settings) -> dict:
    """Read a bundle's ``meta.json`` without loading the model/dataset (PYQ-283).

    Same not-trained convention as ``load()`` (``FileNotFoundError``, checked
    against ``meta.json`` rather than ``model.ckpt`` since that's the only file
    this needs), but skips ``TemporalFusionTransformer.load_from_checkpoint``
    and ``torch.load`` -- both real costs for a caller that only wants the
    recorded symbol/timestamp/evaluation, e.g. an API discovery endpoint asked
    to answer for every trained bundle at once.
    """
    bundle_dir = _bundle_dir(settings, symbol)
    meta_path = bundle_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No trained model for {symbol.upper()} at {meta_path}. Run `pyquant train` first."
        )
    return json.loads(meta_path.read_text())


def list_bundles(settings: Settings) -> list[dict]:
    """Every trained bundle's ``meta.json`` under ``checkpoint_dir`` (PYQ-283).

    Sorted by ``trained_at`` descending (most recently trained first) so a
    caller listing "what's trained" sees the freshest bundle first without
    re-sorting. A bundle directory without a readable ``meta.json`` (e.g. a
    training run that crashed before writing one) is skipped rather than
    failing the whole listing -- one broken bundle must not hide every other
    one, the same discipline ``POST /scan`` applies per-symbol.
    """
    checkpoint_dir = settings.checkpoint_dir
    if not checkpoint_dir.is_dir():
        return []
    metas = []
    for bundle_dir in checkpoint_dir.iterdir():
        meta_path = bundle_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            metas.append(json.loads(meta_path.read_text()))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping unreadable bundle meta.json at %s: %s", meta_path, exc)
    metas.sort(key=lambda m: m.get("trained_at", ""), reverse=True)
    return metas
