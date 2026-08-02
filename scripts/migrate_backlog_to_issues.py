#!/usr/bin/env python3
"""One-time migration: backlog/*.md tickets -> GitHub Issues.

Companion to docs/autonomous-loop-plan.md section 2.4. Reuses the same table/detail
parsing conventions as scripts/backlog.py (see backlog/README.md for the format), then
additionally captures each ticket's full detail body so it can become an issue body.

WHY THIS EXISTS AS A SEPARATE SCRIPT rather than extending scripts/backlog.py: this is a
one-time migration tool, not a standing consistency checker. It gets moved to
backlog/_archive/ alongside the old markdown files once the migration is verified and
done (docs/autonomous-loop-plan.md #2.4 step 3) -- scripts/backlog.py's own fate is the
same, but it stays load-bearing (CI's `backlog.py check`) until that day.

Requirements: the GitHub CLI (`gh`), authenticated (`gh auth status`), run from a clone
of this repo with `backlog/*.md` present. Cannot be run from a sandboxed session with no
`gh` auth -- run it from a local terminal or an interactive Claude Code session.

Usage:
    # ALWAYS dry-run first and read the output before doing anything else.
    python3 scripts/migrate_backlog_to_issues.py --repo AxelSuu/Pytorch-Quant-Model

    # Smoke-test on a handful of tickets before running the full migration.
    python3 scripts/migrate_backlog_to_issues.py --repo AxelSuu/Pytorch-Quant-Model \\
        --execute --only PYQ-101,PYQ-141,PYQ-149

    # Full run, once the smoke test above looks right.
    python3 scripts/migrate_backlog_to_issues.py --repo AxelSuu/Pytorch-Quant-Model --execute

Safe to interrupt and re-run with --only / --skip-existing once gh issue search is wired
up for your account -- see the NOTE in main() about idempotency before doing a full run
twice.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

BACKLOG_DIR = Path(__file__).resolve().parent.parent / "backlog"
FILES = {
    "bug": (BACKLOG_DIR / "bugs.md", "type:bug"),
    "feature": (BACKLOG_DIR / "features.md", "type:feature"),
    "investigation": (BACKLOG_DIR / "investigations.md", "type:investigation"),
}

PRIORITY_LABEL = {"critical": "P0", "high": "P1", "medium": "P2", "low": "P3"}
STATUS_RE = re.compile(r"^(Open|Resolved|Answered|Superseded)", re.IGNORECASE)
PRIORITY_RE = re.compile(r"^(Critical|High|Medium|Low)", re.IGNORECASE)

TABLE_ROW_RE = re.compile(
    r"^\|\s*\[PYQ-(\d+)\]\([^)]*\)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$"
)
DETAIL_HEADING_RE = re.compile(r"^##\s*\[PYQ-(\d+)\]\s*$")

# Heuristic only -- flags tickets whose body mentions a known human-decision blocker, so
# the migrated issue starts with `needs-human` for triage rather than silently losing that
# context. Not authoritative; review the flagged issues, don't trust the label blindly.
NEEDS_HUMAN_HINTS = re.compile(
    r"\b(GPU|API key|product decision|Docker(?:\s+CLI)?|deliberately deferred|"
    r"needs? a human|blocked on)\b",
    re.IGNORECASE,
)


@dataclass
class Ticket:
    id: int
    type: str
    title: str
    table_priority: str
    table_status: str
    body: str = ""
    detail_status: str | None = None
    detail_priority: str | None = None


def _leading_keyword(text: str, pattern: re.Pattern) -> str | None:
    m = pattern.match(text.strip())
    return m.group(1) if m else None


def parse_file(ttype: str, path: Path) -> dict[int, Ticket]:
    lines = path.read_text().splitlines()
    tickets: dict[int, Ticket] = {}

    for line in lines:
        m = TABLE_ROW_RE.match(line)
        if not m:
            continue
        tid, priority, status, title = m.groups()
        if title.lower() == "title":
            continue
        tickets[int(tid)] = Ticket(
            id=int(tid), type=ttype, title=title, table_priority=priority, table_status=status
        )

    current: Ticket | None = None
    body_lines: list[str] = []

    def flush() -> None:
        if current is not None:
            current.body = "\n".join(body_lines).strip()

    for line in lines:
        m = DETAIL_HEADING_RE.match(line)
        if m:
            flush()
            tid = int(m.group(1))
            current = tickets.get(tid)
            body_lines = []
            continue
        if current is None:
            continue
        stripped = line.strip()
        if stripped.startswith("---"):
            flush()
            current = None
            continue
        if stripped.startswith("Status:") and current.detail_status is None:
            current.detail_status = _leading_keyword(stripped[len("Status:") :], STATUS_RE)
        elif stripped.startswith("Priority:") and current.detail_priority is None:
            current.detail_priority = _leading_keyword(stripped[len("Priority:") :], PRIORITY_RE)
        body_lines.append(line)
    flush()

    return tickets


def load_all() -> dict[str, tuple[Ticket, str]]:
    """Returns {"PYQ-101": (Ticket, type_label), ...}."""
    out: dict[str, tuple[Ticket, str]] = {}
    for ttype, (path, type_label) in FILES.items():
        if not path.exists():
            print(f"WARNING: {path} not found, skipping", file=sys.stderr)
            continue
        for tid, ticket in parse_file(ttype, path).items():
            out[f"PYQ-{tid}"] = (ticket, type_label)
    return out


def build_labels(ticket: Ticket, type_label: str) -> list[str]:
    labels = [type_label]
    priority_kw = (ticket.detail_priority or _leading_keyword(ticket.table_priority, PRIORITY_RE) or "").lower()
    if priority_kw in PRIORITY_LABEL:
        labels.append(PRIORITY_LABEL[priority_kw])
    status_kw = (ticket.detail_status or _leading_keyword(ticket.table_status, STATUS_RE) or "").lower()
    if status_kw == "open":
        labels.append("status:backlog")
    if NEEDS_HUMAN_HINTS.search(ticket.body):
        labels.append("needs-human")
    return labels


def build_body(pyq_id: str, source_file: str, ticket: Ticket) -> str:
    return (
        f"{ticket.body}\n\n"
        f"---\n"
        f"Migrated from: `{source_file}#{pyq_id.lower()}`. This issue is a 1:1 migration "
        f"of that ticket's detail block, verbatim. See docs/autonomous-loop-plan.md section "
        f"2.4 for the migration this came from.\n"
    )


def is_closed(ticket: Ticket) -> bool:
    status_kw = (ticket.detail_status or _leading_keyword(ticket.table_status, STATUS_RE) or "").lower()
    return status_kw in {"resolved", "answered", "superseded"}


def run_gh(args: list[str], execute: bool) -> str:
    if not execute:
        print(f"  [dry-run] gh {' '.join(args)}")
        return ""
    result = subprocess.run(["gh", *args], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: gh {' '.join(args)}\n{result.stderr}", file=sys.stderr)
        return ""
    return result.stdout.strip()


def migrate_one(repo: str, pyq_id: str, ticket: Ticket, type_label: str, source_file: str, execute: bool) -> None:
    labels = build_labels(ticket, type_label)
    body = build_body(pyq_id, source_file, ticket)
    title = f"[{pyq_id}] {ticket.title}"

    print(f"{pyq_id} -> title={title!r} labels={labels} closed={is_closed(ticket)}")

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(body)
        body_path = f.name

    args = ["issue", "create", "--repo", repo, "--title", title, "--body-file", body_path]
    for label in labels:
        args += ["--label", label]
    output = run_gh(args, execute)

    if is_closed(ticket) and execute and output:
        # output is the issue URL; gh issue close accepts a URL directly.
        status_kw = ticket.detail_status or "Resolved"
        run_gh(
            ["issue", "close", output, "--comment", f"Migrated as already {status_kw} -- see body for detail."],
            execute,
        )
    elif is_closed(ticket) and not execute:
        print(f"  [dry-run] gh issue close <new-issue-url> --comment 'Migrated as already {ticket.detail_status or 'Resolved'}'")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="owner/repo, e.g. AxelSuu/Pytorch-Quant-Model")
    parser.add_argument("--execute", action="store_true", help="Actually call gh. Default is dry-run.")
    parser.add_argument("--only", help="Comma-separated PYQ-NNN IDs to migrate (for smoke-testing).")
    args = parser.parse_args()

    if not args.execute:
        print("=== DRY RUN (pass --execute to actually create issues) ===\n")

    all_tickets = load_all()
    if not all_tickets:
        print("No tickets parsed -- check backlog/*.md exist and match the expected format.", file=sys.stderr)
        return 1

    only = {s.strip().upper() for s in args.only.split(",")} if args.only else None

    source_file_by_type = {"bug": "bugs.md", "feature": "features.md", "investigation": "investigations.md"}

    count = 0
    for pyq_id, (ticket, type_label) in sorted(all_tickets.items(), key=lambda kv: kv[1][0].id):
        if only and pyq_id not in only:
            continue
        migrate_one(
            args.repo, pyq_id, ticket, type_label, source_file_by_type[ticket.type], args.execute
        )
        count += 1

    print(f"\n{count} ticket(s) processed.")
    if not args.execute:
        print("This was a dry run -- nothing was created. Re-run with --execute once this looks right.")
        print("Recommended: smoke-test first with --execute --only PYQ-101,PYQ-141 (pick a couple of IDs).")
    return 0


if __name__ == "__main__":
    # NOTE on idempotency: this script does not check for an existing migrated issue
    # before creating one. Running it twice with --execute will duplicate every issue.
    # If you need to re-run after a partial failure, use --only with the IDs that didn't
    # make it through the first pass (their titles/labels are printed as each one runs).
    raise SystemExit(main())
