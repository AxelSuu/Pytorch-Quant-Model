"""Render a pydantic settings model as a field reference (PYQ-232).

Why this exists rather than an off-the-shelf extension: PYQ-232's acceptance
criterion is that the ``Settings``/``TFTConfig``/``TrainingConfig``/``DataConfig``
reference shows *every field with its default and description*. Plain
``sphinx.ext.autodoc`` cannot do that for pydantic v2 — model fields are moved
off the class into ``model_fields`` at class-creation time, so autodoc finds
almost nothing to document — and this project's convention (CLAUDE.md, "Config
over constants") records each default's rationale in a ``#`` comment above or
beside the field, which Sphinx only picks up in its ``#:`` form.

``autodoc-pydantic`` would solve the first half but not the second, and adding a
dependency needs a recorded reason (CLAUDE.md non-negotiable 5). Reading the
model's own ``model_fields`` plus the source comments that already exist costs
one local file and no new dependency, so that is what this does.

Usage (MyST)::

    ```{pyquant-config-model} pyquant.config.TrainingConfig
    ```

Description precedence, highest first:

1. an explicit ``Field(description=...)``;
2. a trailing ``#`` comment on the field's own line;
3. a ``#`` comment block directly above the field;
4. a *section* comment — a block above the field that is itself preceded by a
   blank line — which applies to the whole run of adjacent fields beneath it;
5. the preceding field's comment block, but only when it names this field
   explicitly, which is direct evidence that it describes it.

Anything left over renders with an empty description. That is deliberate: the
page reports what the source actually documents rather than inventing prose, so
a gap in ``config.py`` shows up as a gap here.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import io
import re
import textwrap
import tokenize

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective


def _class_source(cls: type) -> tuple[str, list[str]]:
    source = textwrap.dedent(inspect.getsource(cls))
    return source, source.splitlines()


def _comments_by_line(source: str) -> dict[int, tuple[bool, str]]:
    """Map 1-based line number -> (comment is alone on its line, comment text)."""
    found: dict[int, tuple[bool, str]] = {}
    readline = io.StringIO(source).readline
    for token in tokenize.generate_tokens(readline):
        if token.type != tokenize.COMMENT:
            continue
        alone = token.line.lstrip().startswith("#")
        found[token.start[0]] = (alone, token.string.lstrip("#").strip())
    return found


def _block_above(lineno: int, comments: dict[int, tuple[bool, str]]) -> tuple[int, str] | None:
    """The contiguous own-line comment block ending just above ``lineno``."""
    parts: list[str] = []
    cursor = lineno - 1
    while cursor >= 1 and cursor in comments and comments[cursor][0]:
        parts.append(comments[cursor][1])
        cursor -= 1
    if not parts:
        return None
    return cursor + 1, " ".join(reversed(parts))


def _default_expression(value: ast.expr | None) -> str:
    """A readable source-faithful default, unwrapping ``Field(...)``."""
    if value is None:
        return ""
    if isinstance(value, ast.Call) and getattr(value.func, "id", "") == "Field":
        for keyword in value.keywords:
            if keyword.arg == "default":
                return ast.unparse(keyword.value)
            if keyword.arg == "default_factory":
                factory = keyword.value
                if isinstance(factory, ast.Lambda):
                    return ast.unparse(factory.body)
                return f"{ast.unparse(factory)}()"
        return ""
    return ast.unparse(value)


def _describe_fields(cls: type) -> list[dict[str, str]]:
    """One row per pydantic field: name, annotation, default, description."""
    source, lines = _class_source(cls)
    comments = _comments_by_line(source)
    class_def = ast.parse(source).body[0]
    assert isinstance(class_def, ast.ClassDef)

    model_fields = dict(getattr(cls, "model_fields", {}))

    # Pass 1: locate each field's assignment and its own comments.
    entries: list[dict] = []
    for node in class_def.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, annotation = node.target.id, ast.unparse(node.annotation)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name, annotation = node.targets[0].id, ""
        else:
            continue
        if name not in model_fields:
            continue  # model_config and friends are not fields

        trailing = comments.get(node.end_lineno or node.lineno)
        block = _block_above(node.lineno, comments)
        entries.append(
            {
                "name": name,
                "annotation": annotation,
                "default": _default_expression(node.value),
                "lineno": node.lineno,
                "trailing": trailing[1] if trailing and not trailing[0] else "",
                "block": block[1] if block else "",
                "block_start": block[0] if block else node.lineno,
            }
        )

    # Pass 2: promote a block that is itself preceded by a blank line to a
    # *section* heading covering the adjacent run of fields beneath it.
    for index, entry in enumerate(entries):
        if not entry["block"]:
            continue
        above = entry["block_start"] - 1
        preceded_by_blank = above < 1 or not lines[above - 1].strip()
        if not preceded_by_blank:
            continue
        entry["section"] = entry["block"]
        entry["block"] = ""
        for offset in range(index + 1, len(entries)):
            follower, previous = entries[offset], entries[offset - 1]
            gap = range(previous["lineno"], follower["lineno"])
            if follower["block"] or any(not lines[i - 1].strip() for i in gap):
                break
            follower.setdefault("section", entry["section"])

    # Pass 3: a comment block that names a later field describes it too.
    for index, entry in enumerate(entries):
        text = entry.get("block") or entry.get("section") or ""
        if not text or index + 1 >= len(entries):
            continue
        follower = entries[index + 1]
        if follower["trailing"] or follower["block"] or follower.get("section"):
            continue
        if re.search(rf"\b{re.escape(follower['name'])}\b", text):
            follower["section"] = text

    rows = []
    for entry in entries:
        field_info = model_fields[entry["name"]]
        if entry["block"] and entry["trailing"]:
            # A trailing comment annotates the *default* (``# 1 hour``,
            # ``# None -> max_prediction_length``); a block above explains the
            # field. Keep both rather than letting the terser one win.
            own = f"{entry['block']} ({entry['trailing']})"
        else:
            own = entry["block"] or entry["trailing"]
        description = (field_info.description or "") or own or entry.get("section", "")
        rows.append(
            {
                "name": entry["name"],
                "annotation": entry["annotation"],
                "default": entry["default"],
                "description": description,
                "required": field_info.is_required(),
            }
        )
    return rows


def _cell(text: str) -> list[str]:
    """A list-table cell body, blank-safe and continuation-indented."""
    if not text:
        return ["        -"]
    first, *rest = text.splitlines()
    return [f"        - {first}", *[f"          {line}" for line in rest]]


class PyQuantConfigModel(SphinxDirective):
    """``pyquant-config-model <dotted.path.To.Model>``."""

    has_content = False
    required_arguments = 1
    optional_arguments = 0

    def run(self) -> list[nodes.Node]:
        dotted = self.arguments[0].strip()
        module_name, _, class_name = dotted.rpartition(".")
        cls = getattr(importlib.import_module(module_name), class_name)
        rows = _describe_fields(cls)

        out: list[str] = [f".. py:class:: {class_name}", f"   :module: {module_name}", ""]
        docstring = inspect.getdoc(cls) or ""
        for line in docstring.splitlines():
            out.append(f"   {line}" if line else "")
        out += [
            "",
            "   .. list-table::",
            "      :header-rows: 1",
            "      :widths: 22 22 18 38",
            "      :class: pyquant-config-table",
            "",
            "      * - Field",
            "        - Type",
            "        - Default",
            "        - Description",
        ]
        for row in rows:
            default = "*required*" if row["required"] else f"``{row['default']}``"
            out += [
                f"      * - ``{row['name']}``",
                *_cell(f"``{row['annotation']}``" if row["annotation"] else ""),
                *_cell(default),
                *_cell(row["description"]),
            ]
        out.append("")

        container = nodes.container()
        self.state.nested_parse(StringList(out, source=dotted), self.content_offset, container)
        return container.children


def setup(app):
    app.add_directive("pyquant-config-model", PyQuantConfigModel)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
