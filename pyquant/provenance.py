"""Which code produced an artifact — version and git sha.

PYQ-225's thesis is that **seed + pinned data + code version** is what actually
reproduces a run. The seed (PYQ-210) and the pinned data (PYQ-205) were already
recorded; this module is the third leg, and it lives here rather than in
``models/tft.py`` so that ``data/cache.py`` can stamp a pin with it without
importing the ML stack (PYQ-133). ``models/tft.py`` keeps the same functions
under its private names.

Deliberately free of torch, Typer and Rich: it is imported by ``data/`` and by
``analysis/``, both of which have layering rules about what they may pull in.
"""

from __future__ import annotations

import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def package_version() -> str:
    """Installed distribution version, or ``"unknown"`` from a bare source tree."""
    try:
        return version("pyquant")
    except PackageNotFoundError:  # running from a source tree without an install
        return "unknown"


def git_sha() -> str | None:
    """Best-effort short git sha of the working tree, or None outside a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def code_version() -> str:
    """A single string identifying the running code, for fingerprints.

    Combines the distribution version with the git sha when one is available.
    The version alone is too coarse during development -- it does not move
    between commits, and PYQ-121-style feature redefinitions land *between*
    releases, which is exactly the case the cache fingerprint has to catch.
    """
    sha = git_sha()
    base = package_version()
    return f"{base}+{sha}" if sha else base
