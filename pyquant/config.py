"""Typed configuration for PyQuant.

Settings load from environment variables and an optional ``.env`` file.
API keys are read from the environment; everything else has sensible defaults
so the tool runs out of the box on pure OHLCV data.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class TFTConfig(BaseModel):
    """Temporal Fusion Transformer architecture hyperparameters."""

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


class TrainingConfig(BaseModel):
    """Training and windowing settings."""

    max_encoder_length: int = 60  # lookback window (days)
    max_prediction_length: int = 5  # forecast horizon (days)
    batch_size: int = 64
    max_epochs: int = 30
    learning_rate: float = 0.01
    train_split: float = 0.85
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

    period: str = "5y"  # history pulled from yfinance
    use_macro: bool = True
    use_options: bool = True
    use_sentiment: bool = True
    use_sectors: bool = True
    # Sector ETFs used for cross-asset features.
    sector_etfs: list[str] = Field(
        default_factory=lambda: ["XLK", "XLF", "XLE", "XLV", "XLY", "SPY"]
    )

    # Local panel cache: avoids re-fetching identical data across repeated
    # train/forecast/explain runs and eases pressure on informal rate limits.
    cache_enabled: bool = True
    cache_ttl_seconds: float = 3600.0  # 1 hour
    cache_dir: Path = Path(".cache/pyquant")


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

    # Paths
    checkpoint_dir: Path = Path("checkpoints")

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
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings, dotenv_settings]
        if _active_yaml_file is not None:
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=_active_yaml_file))
        sources.append(file_secret_settings)
        return tuple(sources)


# Set transiently by load_settings() so the classmethod above can see the chosen
# YAML path without it becoming part of the frozen model_config.
_active_yaml_file: Path | None = None


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load settings from environment + .env, optionally layering a YAML config.

    ``config_path`` (or the ``PYQUANT_CONFIG`` env var) names a YAML experiment
    file whose values sit below env vars but above the built-in defaults, so a
    full experiment (hidden_size, quantiles, epochs, data toggles, ...) can be
    checked into version control as one file.
    """
    global _active_yaml_file
    chosen = config_path or os.environ.get("PYQUANT_CONFIG")
    _active_yaml_file = Path(chosen) if chosen else None
    try:
        return Settings()
    finally:
        _active_yaml_file = None
