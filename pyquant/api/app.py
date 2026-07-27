"""FastAPI app instance: router mounting (PYQ-261, docs/api-design.md).

uv sync --extra api
uv run uvicorn pyquant.api.app:app
"""

from __future__ import annotations

from fastapi import FastAPI

from pyquant.api.routes import explain, forecast, health, train

app = FastAPI(
    title="PyQuant API",
    description="Probabilistic equity forecasts from a Temporal Fusion Transformer.",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(forecast.router)
app.include_router(explain.router)
app.include_router(train.router)
