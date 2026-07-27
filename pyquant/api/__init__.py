"""PyQuant FastAPI service layer (PYQ-261), per docs/api-design.md's design note.

A second front-end beside the CLI, over the same plain functions and dataclasses in
``analysis/``/``models/`` -- nothing in the core moved to build this. Install with the
``api`` extra (``uv sync --extra api``) and run with:

    uv run uvicorn pyquant.api.app:app
"""
