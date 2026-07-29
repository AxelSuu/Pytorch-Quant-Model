"""FastAPI app instance: router mounting (PYQ-261, docs/api-design.md).

uv sync --extra api
uv run uvicorn pyquant.api.app:app
"""

from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse

from pyquant.api.deps import require_api_key
from pyquant.api.routes import backtest, explain, forecast, health, train

app = FastAPI(
    title="PyQuant API",
    description="Probabilistic equity forecasts from a Temporal Fusion Transformer.",
    version="0.1.0",
    # Replaced below by key-gated equivalents (bugs.md#pyq-160): FastAPI's default
    # /docs, /redoc and /openapi.json are mounted independently of any APIRouter's
    # `dependencies=`, so they bypassed require_api_key entirely -- anyone who could
    # reach the host learned the full endpoint/schema surface with no key, contradicting
    # docs/http-api.md's "every endpoint except /healthz requires X-API-Key". Disabling
    # the built-ins and hand-mounting the same routes with the dependency attached makes
    # that claim true rather than aspirational.
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.include_router(health.router)
app.include_router(forecast.router)
app.include_router(explain.router)
app.include_router(train.router)
app.include_router(backtest.router)


@app.get("/openapi.json", include_in_schema=False, dependencies=[Depends(require_api_key)])
def _openapi_schema() -> dict:
    """Key-gated replacement for FastAPI's default /openapi.json (bugs.md#pyq-160)."""
    return get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )


@app.get("/docs", include_in_schema=False, dependencies=[Depends(require_api_key)])
def _swagger_ui() -> HTMLResponse:
    """Key-gated replacement for FastAPI's default /docs (bugs.md#pyq-160)."""
    return get_swagger_ui_html(openapi_url="/openapi.json", title=f"{app.title} - Swagger UI")


@app.get("/redoc", include_in_schema=False, dependencies=[Depends(require_api_key)])
def _redoc_ui() -> HTMLResponse:
    """Key-gated replacement for FastAPI's default /redoc (bugs.md#pyq-160)."""
    return get_redoc_html(openapi_url="/openapi.json", title=f"{app.title} - ReDoc")
