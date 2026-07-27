"""Sphinx configuration for the PyQuant documentation site (PYQ-232).

Build:
    uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html

The ``-W`` is not decoration. A docs site that is not built with
warnings-as-errors rots within two refactors: a renamed module, a dead
``:func:`` reference or an autodoc import failure all degrade silently
otherwise. PYQ-233 gates this exact command in CI, for the same reason PYQ-311
gates ``scripts/backlog.py check``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DOCS = Path(__file__).parent.resolve()
_ROOT = _DOCS.parent

# The site documents the working tree, not an installed copy.
sys.path.insert(0, str(_ROOT))
# Local extensions (see docs/_ext/).
sys.path.insert(0, str(_DOCS / "_ext"))

# -- Project ---------------------------------------------------------------

project = "PyQuant"
author = "Axel"
copyright = "2026, Axel"  # noqa: A001

# Read the version from the installed package metadata rather than restating
# it, so the docs cannot disagree with pyproject.toml (the same reasoning
# behind provenance.package_version()).
try:  # pragma: no cover - trivial
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("pyquant")
except Exception:  # pragma: no cover - source checkout without an install
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

# -- General ---------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    # Local: renders pydantic-settings models with their defaults and the
    # source comments that explain them (see docs/_ext/pyquant_config.py).
    "pyquant_config",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]
nitpicky = False  # see docs/api/index.md for why this is not on yet (PYQ-233)

# -- MyST ------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    # `[15 invariants]{.pq-card__count}` on the landing page's cards: the count is
    # content, not decoration, so it stays in the Markdown rather than becoming a
    # block of raw HTML that search and translation cannot see.
    "attrs_inline",
]
# Narrative pages cross-link into each other's sections (invariants.md ->
# methodology.md#..., architecture.md -> invariants.md#...).
myst_heading_anchors = 3

# -- autodoc / autosummary -------------------------------------------------

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    # Undocumented members are still part of the public surface; hiding them
    # would make the site look more complete than the docstrings are (79%
    # coverage at the time of writing, PYQ-236 tracks the rest).
    "undoc-members": True,
}
# Keep annotations in the signature: PYQ-232's acceptance criterion is that an
# upstream type in a *signature* resolves to an upstream doc page.
autodoc_typehints = "signature"
autodoc_preserve_defaults = True

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

# -- intersphinx -----------------------------------------------------------

# The reason PYQ-232 chose Sphinx over MkDocs: this codebase is a thin layer
# over pandas / torch / pytorch-forecasting / pydantic, so resolving those
# types into upstream documentation carries real explanatory weight.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "pytorch_forecasting": ("https://pytorch-forecasting.readthedocs.io/en/stable/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}
intersphinx_timeout = 30

# Runtime type objects stringify to their *defining* module, which is an
# implementation detail upstream does not document: pandas documents
# ``pandas.DataFrame``, not ``pandas.core.frame.DataFrame``. Without this map
# every such annotation renders as plain text and the whole intersphinx
# argument above evaporates. Keys are what autodoc emits; values are what the
# upstream inventory actually publishes.
_CANONICAL_TARGETS = {
    "pandas.core.frame.DataFrame": "pandas.DataFrame",
    "pandas.core.series.Series": "pandas.Series",
    "pandas.core.indexes.base.Index": "pandas.Index",
    "pandas.core.indexes.datetimes.DatetimeIndex": "pandas.DatetimeIndex",
    "pandas._libs.tslibs.timestamps.Timestamp": "pandas.Timestamp",
    "numpy.ndarray": "numpy.ndarray",
    "pytorch_forecasting.data.timeseries.TimeSeriesDataSet": (
        "pytorch_forecasting.data.timeseries.TimeSeriesDataSet"
    ),
}


def _resolve_canonical(app, env, node, contnode):
    """Retry an unresolved Python reference under its documented public name."""
    from sphinx.ext.intersphinx import resolve_reference_in_inventory

    target = node.get("reftarget")
    canonical = _CANONICAL_TARGETS.get(target)
    if node.get("refdomain") != "py" or canonical is None:
        return None
    node = node.deepcopy()
    node["reftarget"] = canonical
    for inventory in intersphinx_mapping:
        resolved = resolve_reference_in_inventory(env, inventory, node, contnode)
        if resolved is not None:
            return resolved
    return None


# -- Docstring conventions -------------------------------------------------

_LITERAL_BLOCK_SAFE = ("..", "-", "*", "+", ":", ">")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _colon_led_listings(app, what, name, obj, options, lines):
    """Treat ``text:`` followed by an indented block as an RST literal block.

    Several module docstrings introduce a column-aligned listing this way —
    ``models/tft.py``'s bundle layout and ``data/dataset.py``'s pipeline flow
    are the two clearest. RST has no such convention: a *one-line* lead-in is
    read as a definition-list term, and a lead-in of two or more lines is an
    outright ``Unexpected indentation`` error, which ``-W`` turns into a build
    failure. Promoting the lead-in to ``::`` renders the listing monospaced and
    column-aligned, which is what it is for.

    Deliberately narrow: it fires only when a line ends in a single ``:`` and
    the very next line is non-blank and more indented, and never on a directive,
    field, list-item or block-quote marker. Runs after napoleon (priority 800)
    so it sees real RST rather than a Google-style section header.
    """
    index = 0
    while index < len(lines) - 1:
        current, following = lines[index], lines[index + 1]
        stripped = current.strip()
        starts_markup = stripped.startswith(_LITERAL_BLOCK_SAFE)
        if (
            stripped.endswith(":")
            and not stripped.endswith("::")
            and not starts_markup
            and following.strip()
            and _indent(following) > _indent(current)
        ):
            lines[index] = current + ":"
            lines.insert(index + 1, "")
            index += 1
        index += 1


def setup(app):
    app.connect("missing-reference", _resolve_canonical)
    app.connect("autodoc-process-docstring", _colon_led_listings, priority=800)
    return {"parallel_read_safe": True, "parallel_write_safe": True}


# -- HTML ------------------------------------------------------------------

html_theme = "furo"
html_title = f"PyQuant {version}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# The fan mark: a known history to the left of the last observed bar, a p10/p90
# band that only widens to the right of it. It is the shape the product actually
# emits, and the dashed rule in it is invariants 3 and 4 (docs/invariants.md).
html_favicon = "_static/favicon.svg"

# Colour tokens live here rather than in custom.css so each is defined once per
# theme instead of once per rule; docs/_static/custom.css carries structure and
# typography only. Furo emits these verbatim as CSS custom properties, so the
# project's own --pq-* tokens can ride along with the theme's.
_LIGHT = {
    "color-brand-primary": "#0f6b73",
    "color-brand-content": "#0d5f66",
    "color-brand-visited": "#0d5f66",
    "color-background-primary": "#fbfcfc",
    "color-background-secondary": "#f2f5f5",
    "color-background-border": "#dde4e5",
    "color-foreground-primary": "#12191c",
    "color-foreground-secondary": "#3f4c51",
    "color-foreground-muted": "#68767c",
    # The p10/p90 band, faded at its edges the way a fan chart is.
    "pq-band-edge": "rgba(15, 107, 115, 0.12)",
    # Reserved for caveats about sample size -- the recurring qualifier on every
    # number this project reports.
    "pq-accent": "#9a5b12",
}
_DARK = {
    "color-brand-primary": "#5fbcc4",
    "color-brand-content": "#74ccd4",
    "color-brand-visited": "#74ccd4",
    "color-background-primary": "#0e1417",
    "color-background-secondary": "#141c20",
    "color-background-border": "#26333a",
    "color-foreground-primary": "#dde5e7",
    "color-foreground-secondary": "#a8b6ba",
    "color-foreground-muted": "#7d8f95",
    "pq-band-edge": "rgba(95, 188, 196, 0.14)",
    "pq-accent": "#d9994a",
}

html_theme_options = {
    "source_repository": "https://github.com/AxelSuu/Pytorch-Quant-Model/",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_logo": "logo.svg",
    "dark_logo": "logo-dark.svg",
    # Furo adds the leading `--` itself; the keys here are bare names.
    "light_css_variables": _LIGHT,
    "dark_css_variables": _DARK,
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/AxelSuu/Pytorch-Quant-Model",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 '
                "8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49"
                "-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01"
                "-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07"
                "-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12"
                "0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82"
                " 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65"
                " 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012"
                ' 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}
