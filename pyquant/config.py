"""Typed configuration for PyQuant.

Settings load from environment variables and an optional ``.env`` file.
API keys are read from the environment; everything else has sensible defaults
so the tool runs out of the box on pure OHLCV data.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TFTConfig(BaseModel):
    """Temporal Fusion Transformer architecture hyperparameters."""

    hidden_size: int = 32
    attention_head_size: int = 4
    dropout: float = 0.1
    hidden_continuous_size: int = 16
    # Quantiles for the prediction intervals (p10 / p50 / p90).
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])


class TrainingConfig(BaseModel):
    """Training and windowing settings."""

    max_encoder_length: int = 60  # lookback window (days)
    max_prediction_length: int = 5  # forecast horizon (days)
    batch_size: int = 64
    max_epochs: int = 30
    learning_rate: float = 0.01
    train_split: float = 0.85
    gradient_clip_val: float = 0.1


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


def load_settings() -> Settings:
    """Load settings from environment + .env file."""
    return Settings()
