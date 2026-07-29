"""Typed configuration for PyQuant.

Settings load from environment variables and an optional ``.env`` file.
API keys are read from the environment; everything else has sensible defaults
so the tool runs out of the box on pure OHLCV data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


def project_root() -> Path:
    """The directory bundles and caches are resolved against.

    ``checkpoint_dir`` and ``cache_dir`` default to relative paths, which used to
    be resolved against whatever the ambient working directory happened to be.
    For the CLI that is a paper cut -- `pyquant train AAPL` from the repo root
    then `pyquant forecast AAPL` from elsewhere fails to find the bundle. For a
    long-running server (PYQ-213/PYQ-261) it is worse: the process's working
    directory is not guaranteed, so checkpoints can land somewhere unexpected,
    and a different cwd per restart means the service cannot find bundles it
    created itself (PYQ-220).

    Anchoring is deliberately *not* XDG by default. Repo-local `checkpoints/`
    and `.cache/pyquant` are what the README, `.gitignore` and every existing
    install already expect, so switching to `platformdirs` would strand them.
    ``PYQUANT_HOME`` overrides the anchor for a deployment that wants bundles
    outside the source tree, and an absolute path in config still wins outright.
    """
    override = os.environ.get("PYQUANT_HOME")
    if override:
        return Path(override).expanduser().resolve()
    # pyquant/config.py -> pyquant/ -> project root
    return Path(__file__).resolve().parent.parent


def _anchor(path: Path) -> Path:
    """Resolve ``path`` against the project root unless it is already absolute."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else (project_root() / path).resolve()


class TFTConfig(BaseModel):
    """Temporal Fusion Transformer architecture hyperparameters."""

    # extra="forbid" so a misspelled nested env key (e.g. TFT__QUANTILE, missing
    # the trailing "S") fails at Settings() construction instead of silently
    # keeping the default with no error -- the same failure shape PYQ-128 fixed
    # for a missing --config *file*, one level down at the *key* (PYQ-157).
    model_config = ConfigDict(extra="forbid")

    hidden_size: int = 32
    attention_head_size: int = 4
    dropout: float = 0.1
    hidden_continuous_size: int = 16
    # Quantiles for the prediction intervals (p10 / p50 / p90).
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])

    @field_validator("quantiles")
    @classmethod
    def _quantiles_sorted_ascending(cls, v: list[float]) -> list[float]:
        """Reject an unsorted quantile list.

        evaluate_predictions() treats the first configured quantile as the
        lower calibration bound and the last as the upper -- an unsorted list
        (e.g. [0.9, 0.1, 0.5]) would silently invert the band with no error.
        """
        if list(v) != sorted(v):
            raise ValueError(
                f"quantiles must be sorted ascending, got {v}; the first/last "
                "entries are used as the lower/upper calibration bounds."
            )
        return v

    @field_validator("quantiles")
    @classmethod
    def _quantiles_valid_and_include_median(cls, v: list[float]) -> list[float]:
        """Reject a bandless config before it reaches an unguarded call site.

        `_window_signal`/`permutation_importance` (models/tft.py) both do
        `quantiles.index(0.5)` with no guard of their own, so a config missing
        0.5 previously surfaced as a bare `ValueError: 0.5 is not in list` from
        deep in the stack, naming neither the setting nor the cause (PYQ-157) --
        `evaluate_predictions` already raised a clear error for the same case,
        so this makes that the *only* behavior, everywhere, by construction.
        """
        if not all(0.0 < q < 1.0 for q in v):
            raise ValueError(f"quantiles must all satisfy 0 < q < 1, got {v}")
        if 0.5 not in v:
            raise ValueError(
                f"quantiles must include 0.5 (the median), got {v}; "
                "used as the median prediction throughout the pipeline."
            )
        return v


class TrainingConfig(BaseModel):
    """Training and windowing settings."""

    model_config = ConfigDict(extra="forbid")

    max_encoder_length: int = 60  # lookback window (days)
    max_prediction_length: int = 5  # forecast horizon (days)
    batch_size: int = 64
    max_epochs: int = 30
    learning_rate: float = 0.01
    # Price levels are non-stationary, making persistence nearly optimal by
    # construction. Keep the established target until PYQ-247 records a
    # controlled backtest; ``log_return`` is available for that comparison.
    target: Literal["close", "log_return"] = "close"
    # Length of the held-out *test* tail, in trading days -- what every
    # reported metric (EvaluationMetrics, meta.json, the CLI) is computed
    # from. This must be comfortably longer than max_prediction_length: the
    # number of windows scored is (validation_days - max_prediction_length +
    # 1), so a holdout of exactly one horizon yields a single window -- 5
    # points driving every reported metric (PYQ-117). 60 days gives ~56
    # windows at the default 5-day horizon. Disjoint from `selection_days`
    # below (PYQ-143): nothing that selects a checkpoint ever sees this slice.
    validation_days: int = 60
    # Length of a *second*, earlier held-out slice EarlyStopping/ModelCheckpoint
    # select against, purged on both sides from training and from the test
    # slice above (PYQ-143). Before this existed, `train()`/
    # `walk_forward_backtest()` picked the best of many epochs against the
    # exact window later reported as "the" metrics -- a selection-event bias
    # identical in kind to the one `TuneResult`'s own docstring names for
    # Optuna trials, just applied to epochs instead of trials. 30 days gives
    # ~26 windows at the default 5-day horizon -- half of validation_days'
    # default, a deliberately unoptimized starting point (see PYQ-143's
    # resolution note), not a tuned value.
    selection_days: int = 30
    # Trading days held out *between* the (purged) selection slice and the
    # test window, used only to fit the PYQ-248 conformal offset. Zero
    # disables conformal calibration entirely, which is the default because
    # switching it on changes every reported coverage figure -- that has to be
    # a measured, deliberate change, not a silent one. See docs/methodology.md.
    calibration_days: int = 0
    # Look-ahead control around every split (PYQ-250). A training sample whose
    # decoder reaches into the horizon immediately before the validation window
    # shares target days with the period about to be scored; `purge_horizon`
    # drops those. `embargo_days` drops a further buffer, because serial
    # correlation leaks across the boundary even without literal overlap.
    # Defaults follow López de Prado: purge one horizon, embargo a few days.
    purge_horizon: int | None = None  # None -> max_prediction_length
    embargo_days: int = 2
    # Epochs without val_loss improvement before EarlyStopping fires. Worth tuning
    # alongside validation_days: the noisier the selection metric, the less a small
    # patience means (PYQ-224).
    early_stopping_patience: int = 5
    gradient_clip_val: float = 0.1
    # Reproducibility: seed_everything() is called with this before each fit and
    # the value is recorded in meta.json so a run can be reproduced.
    seed: int = 42
    # DataLoader worker processes. 0 = single-process loading (safe default);
    # a non-zero value parallelises data loading during training.
    num_workers: int = 0
    # Lightning precision string (e.g. "32-true", "bf16-mixed", "16-mixed").
    # Defaults to full fp32 so nothing changes unless explicitly opted in.
    precision: str = "32-true"


class DataConfig(BaseModel):
    """Data sourcing and enrichment toggles.

    Each enrichment flag is a *request*. A source only activates if it is both
    enabled here and has the credentials/data it needs; otherwise it degrades
    gracefully (the features are simply dropped with a logged notice).
    """

    model_config = ConfigDict(extra="forbid")

    period: str = "5y"  # history pulled from yfinance
    use_macro: bool = True
    use_options: bool = True
    use_sentiment: bool = True
    use_sectors: bool = True
    # Technical indicators (SMA/EMA/RSI/MACD/Bollinger/...) computed from the
    # target's own OHLCV. Unlike the other toggles this defaults on with no
    # graceful-degradation path -- there is no vendor to lose -- but a feature
    # ablation needs a "price-only" arm to compare against (PYQ-316), and there
    # was previously no way to ask for one short of hand-editing prices.py.
    use_indicators: bool = True
    # Sector ETFs used for cross-asset features.
    sector_etfs: list[str] = Field(
        default_factory=lambda: ["XLK", "XLF", "XLE", "XLV", "XLY", "SPY"]
    )

    # Local panel cache: avoids re-fetching identical data across repeated
    # train/forecast/explain runs and eases pressure on informal rate limits.
    cache_enabled: bool = True
    cache_ttl_seconds: float = 3600.0  # 1 hour
    # validate_default so the anchoring below runs for the default too, not only
    # when a value is supplied -- DataConfig is built from its default_factory on
    # every Settings(), so without it the default stays relative (PYQ-220).
    cache_dir: Path = Field(default=Path(".cache/pyquant"), validate_default=True)

    @field_validator("cache_dir")
    @classmethod
    def _anchor_cache_dir(cls, v: Path) -> Path:
        return _anchor(v)


class Settings(BaseSettings):
    """Top-level settings, composed from sub-configs plus secrets/paths."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Secrets (optional — absence disables the corresponding enrichment).
    fred_api_key: str | None = None
    finnhub_api_key: str | None = None

    # Paths. Relative values are anchored to the project root rather than the
    # ambient cwd, so `train` here and `forecast` there find the same bundle
    # (PYQ-220). Set PYQUANT_HOME, or give an absolute path, to move them.
    checkpoint_dir: Path = Field(default=Path("checkpoints"), validate_default=True)

    @field_validator("checkpoint_dir")
    @classmethod
    def _anchor_checkpoint_dir(cls, v: Path) -> Path:
        return _anchor(v)

    # An append-only accumulated options-snapshot history (PYQ-254 route 1), one
    # JSONL file per symbol. Deliberately not under `data.cache_dir`: the panel
    # cache is a TTL-pruned, rebuildable convenience, while this is meant to be
    # a permanent, slowly-growing dataset -- useless on day one, the only source
    # of historical options-implied data this project can have at all once
    # enough days accumulate, since yfinance exposes only a current chain.
    options_history_dir: Path = Field(default=Path("data/options_history"), validate_default=True)

    @field_validator("options_history_dir")
    @classmethod
    def _anchor_options_history_dir(cls, v: Path) -> Path:
        return _anchor(v)

    # Nested config sections
    tft: TFTConfig = Field(default_factory=TFTConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority (earlier wins): init kwargs > env > .env > YAML config > secrets.
        # The YAML source sits *below* env vars (PYQ-209), so a checked-in
        # experiment config is overridable by the environment, and explicit CLI
        # flags -- applied after load_settings() -- still win over everything.
        #
        # No YAML layer by default -- load_settings() builds a per-call subclass
        # with its own override of this method when a config path is given
        # (PYQ-146), rather than this base class reading a module-global path.
        return (init_settings, env_settings, dotenv_settings, file_secret_settings)


def _settings_class_for_yaml(yaml_file: Path) -> type[Settings]:
    """A ``Settings`` subclass whose YAML source is fixed at class-creation time.

    Building one of these per ``load_settings()`` call (rather than stashing the
    path in a module global that ``Settings.settings_customise_sources`` reads
    back) means the YAML path lives in a closure private to this call, so two
    concurrent ``load_settings()`` calls -- e.g. two API requests on Starlette's
    threadpool, since routes under ``pyquant/api/routes/`` are sync `def`s --
    can never observe each other's config path (PYQ-146: the previous global
    could be reset by one thread's `finally` before another thread reached
    `Settings()`, silently building settings without the YAML layer it asked for).
    """

    class _SettingsWithYaml(Settings):
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            return (
                init_settings,
                env_settings,
                dotenv_settings,
                YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file),
                file_secret_settings,
            )

    return _SettingsWithYaml


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load settings from environment + .env, optionally layering a YAML config.

    ``config_path`` (or the ``PYQUANT_CONFIG`` env var) names a YAML experiment
    file whose values sit below env vars but above the built-in defaults, so a
    full experiment (hidden_size, quantiles, epochs, data toggles, ...) can be
    checked into version control as one file.
    """
    chosen = config_path or os.environ.get("PYQUANT_CONFIG")
    # An explicitly requested config that isn't there is an error, not a silent
    # fallback to defaults: YamlConfigSettingsSource treats a missing file as
    # "no values to contribute", so a typo'd path used to train a completely
    # different experiment than the one asked for -- and record it as such
    # (PYQ-128). No config requested at all stays silent, as before.
    if chosen is not None and not Path(chosen).is_file():
        raise FileNotFoundError(
            f"Config file not found: {chosen}. Remove the --config/PYQUANT_CONFIG "
            "setting to run with the built-in defaults."
        )
    if chosen is None:
        return Settings()
    return _settings_class_for_yaml(Path(chosen))()
