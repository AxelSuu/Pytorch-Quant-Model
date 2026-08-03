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
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def package_version() -> str:
    """Installed distribution version, or ``"unknown"`` from a bare source tree."""
    try:
        return version("pyquant")
    except PackageNotFoundError:  # running from a source tree without an install
        return "unknown"


def _git(args: list[str], cwd: Path) -> str | None:
    """Run a git command, returning its stdout or ``None`` if it is unusable.

    ``None`` covers every "no git answer" case alike — git absent, not a
    repository, timed out, or a non-zero exit — because provenance capture must
    degrade to "unknown" rather than fail a training run.
    """
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5, cwd=cwd)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


@cache
def _resolve_git_sha(here: Path) -> str | None:
    toplevel = _git(["rev-parse", "--show-toplevel"], cwd=here.parent)
    if toplevel is None:
        return None
    if (Path(toplevel) / "pyquant" / here.name).resolve() != here:
        return None
    return _git(["rev-parse", "--short", "HEAD"], cwd=here.parent)


def git_sha() -> str | None:
    """Short git sha of *this project's* working tree, or None.

    Resolving ``git rev-parse`` from the package directory is correct for a
    source checkout and accidentally correct for a wheel installed outside any
    repo (git fails, None is recorded). It is silently *wrong* for the case in
    between: ``site-packages`` sitting inside some unrelated repository -- a
    vendored dependency tree, a conda env under version control, a monorepo with
    a committed venv -- where it stamps that repo's sha onto PyQuant's
    provenance (PYQ-134). A wrong provenance is worse than a missing one,
    because nothing downstream can tell it is wrong.

    So the repo is verified to actually contain this file before its sha is
    trusted: ``<toplevel>/pyquant/provenance.py`` must be this very module.

    The git calls are memoized by resolved ``__file__`` (PYQ-279): a working
    tree's sha can't change mid-process, so nothing in-process ever needs it
    re-read after the first call for a given path. Keying on the path rather
    than caching the bare result keeps ``__file__``-monkeypatching tests (see
    ``test_provenance.py``) correctly cache-missing on a different path
    instead of replaying another test's cached answer.
    """
    here = Path(__file__).resolve()
    return _resolve_git_sha(here)


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
