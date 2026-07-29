"""Pydantic response models, built directly from analysis/serialize.py's dicts.

Per docs/api-design.md's recommended path: the CLI's `--format json` and this API both
call the exact same serializer functions, and these models are a thin pydantic wrapper
around their output (`ModelName(**serialize.x_to_dict(obj))`) purely for OpenAPI schema
generation and response validation. This is what makes "response schemas match the CLI's
--format json payloads field-for-field" true by construction rather than by convention --
the two front-ends cannot drift because they share the conversion code, not just its shape.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, field_validator

# A ticker symbol or bundle name becomes a filesystem path segment
# (`_bundle_dir`: `settings.checkpoint_dir / name.upper()`). Unlike a GET
# route's `{symbol}` path parameter, a request-body field isn't subject to
# Starlette's `/`-rejecting path matching, so `"../../etc"` would otherwise
# reach `mkdir`/`torch.save` unfiltered (PYQ-145). Real tickers/bundle names
# are short, uppercase alphanumerics with an occasional `.`/`-`
# (e.g. "BRK.B", "AAPL_MSFT").
SYMBOL_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9.\-]{0,15}$"
_SYMBOL_RE = re.compile(SYMBOL_PATTERN)


def _validate_symbol(value: str) -> str:
    normalized = value.strip().upper()
    if not _SYMBOL_RE.match(normalized):
        raise ValueError(
            f"Invalid symbol/bundle name {value!r}: must match {SYMBOL_PATTERN}"
        )
    return normalized


class EvaluationResponse(BaseModel):
    """Mirrors analysis.serialize.evaluation_to_dict()."""

    model_mae: float
    baseline_mae: float
    skill_vs_baseline: float
    directional_accuracy: float
    calibration_coverage: float
    quantile_exceedance: dict[str, float]
    pinball_losses: dict[str, float]
    crps: float
    winkler_score: float
    pit: list[float]
    n_samples: int
    n_points: int
    effective_n_samples: int


class ForecastResponse(BaseModel):
    """Mirrors analysis.serialize.forecast_to_dict(); GET /forecast/{symbol}."""

    symbol: str
    last_date: str
    current_price: float
    horizon: int
    forecast_dates: list[str]
    quantiles: list[float]
    predictions: list[list[float]]
    n_quantile_crossings: int
    median: list[float] | None = None
    expected_return_pct: float | None = None


class FeatureImportance(BaseModel):
    """One (feature, weight) entry of an InterpretationResponse."""

    feature: str
    weight: float


class InterpretationResponse(BaseModel):
    """Mirrors analysis.serialize.interpretation_to_dict(); GET /explain/{symbol}."""

    symbol: str
    feature_importance: list[FeatureImportance]
    attention: list[float]
    bundle_skill: float | None = None


class ScanRow(BaseModel):
    """Mirrors analysis.serialize.scan_row_to_dict(); one POST /scan result row."""

    symbol: str
    status: str
    error: str | None = None
    current_price: float | None = None
    median_target: float | None = None
    expected_return_pct: float | None = None
    band_width_pct: float | None = None
    signal: str | None = None


class ScanRequest(BaseModel):
    """Request body for POST /scan."""

    symbols: list[str]

    @field_validator("symbols")
    @classmethod
    def _validate_symbols(cls, values: list[str]) -> list[str]:
        return [_validate_symbol(v) for v in values]


class TrainResultResponse(BaseModel):
    """Mirrors analysis.serialize.train_result_to_dict()."""

    symbols: list[str]
    bundle_dir: str
    val_loss: float
    n_features: int
    epochs_run: int
    evaluation: EvaluationResponse


class TrainRequest(BaseModel):
    """Request body for POST /train."""

    symbols: list[str]
    bundle_name: str | None = None
    epochs: int | None = None
    period: str | None = None

    @field_validator("symbols")
    @classmethod
    def _validate_symbols(cls, values: list[str]) -> list[str]:
        return [_validate_symbol(v) for v in values]

    @field_validator("bundle_name")
    @classmethod
    def _validate_bundle_name(cls, value: str | None) -> str | None:
        return _validate_symbol(value) if value is not None else None


class TrainJobResponse(BaseModel):
    """202 response to POST /train: the job to poll via GET /train/{job_id}."""

    job_id: str
    status: str


class TrainJobStatusResponse(BaseModel):
    """Response for GET /train/{job_id}."""

    job_id: str
    status: str
    result: TrainResultResponse | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Response for GET /healthz."""

    status: str = "ok"
