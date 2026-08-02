#!/usr/bin/env python3
"""Lightweight consistency checker + query tool for backlog/*.md.

No external dependencies -- this is meant to be run directly from a clone
with nothing installed (`uv run python scripts/backlog.py ...` or plain
`python3 scripts/backlog.py ...`).

Format assumed (see backlog/README.md):
  - A markdown table near the top of each file with rows
    "| [PYQ-NNN](#pyq-nnn) | <priority> | <status> | <title> |".
  - One "## [PYQ-NNN]" heading per ticket below the table, followed by a
    title line, then "Status: ..." and "Priority: ..." lines (in either
    order, possibly with trailing commentary after the leading keyword).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

BACKLOG_DIR = Path(__file__).resolve().parent.parent / "backlog"
FILES = {
    "bug": (BACKLOG_DIR / "bugs.md", range(100, 200)),
    "feature": (BACKLOG_DIR / "features.md", range(200, 300)),
    "investigation": (BACKLOG_DIR / "investigations.md", range(300, 400)),
}

PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
PRIORITY_RE = re.compile(r"^(Critical|High|Medium|Low)", re.IGNORECASE)
STATUS_RE = re.compile(r"^(Open|Resolved|Answered|Superseded)", re.IGNORECASE)

TABLE_ROW_RE = re.compile(
    r"^\|\s*\[PYQ-(\d+)\]\([^)]*\)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$"
)
DETAIL_HEADING_RE = re.compile(r"^##\s*\[PYQ-(\d+)\]\s*$")


@dataclass
class Ticket:
    id: int
    type: str
    table_priority: str
    table_status: str
    title: str
    detail_priority: str | None = None
    detail_status: str | None = None
    has_detail: bool = False


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
        if title.lower() == "title":  # header row
            continue
        tickets[int(tid)] = Ticket(
            id=int(tid), type=ttype, table_priority=priority, table_status=status, title=title
        )

    current: Ticket | None = None
    for line in lines:
        m = DETAIL_HEADING_RE.match(line)
        if m:
            tid = int(m.group(1))
            current = tickets.get(tid)
            if current is not None:
                current.has_detail = True
            continue
        if current is None:
            continue
        stripped = line.strip()
        if stripped.startswith("Status:") and current.detail_status is None:
            current.detail_status = _leading_keyword(stripped[len("Status:") :], STATUS_RE)
        elif stripped.startswith("Priority:") and current.detail_priority is None:
            current.detail_priority = _leading_keyword(stripped[len("Priority:") :], PRIORITY_RE)
        elif stripped.startswith("---"):
            current = None  # end of this ticket's detail block

    return tickets


def load_all() -> dict[int, Ticket]:
    all_tickets: dict[int, Ticket] = {}
    for ttype, (path, _) in FILES.items():
        for tid, ticket in parse_file(ttype, path).items():
            if tid in all_tickets:
                print(
                    f"ERROR: PYQ-{tid} appears in both {all_tickets[tid].type} and {ttype}",
                    file=sys.stderr,
                )
            all_tickets[tid] = ticket
    return all_tickets


def cmd_check(_args: argparse.Namespace) -> int:
    problems: list[str] = []

    for ttype, (path, id_range) in FILES.items():
        if not path.exists():
            problems.append(f"{path} does not exist")
            continue
        tickets = parse_file(ttype, path)
        if not tickets:
            problems.append(f"{path}: no tickets found -- table format regex may be stale")
            continue
        for tid, t in sorted(tickets.items()):
            if tid not in id_range:
                problems.append(f"PYQ-{tid} in {path.name} is outside its expected ID range")
            if not t.has_detail:
                problems.append(f"PYQ-{tid}: table row with no matching '## [PYQ-{tid}]' detail section")
            if t.detail_status is None:
                problems.append(f"PYQ-{tid}: detail section missing a recognizable 'Status:' line")
            if t.detail_priority is None:
                problems.append(f"PYQ-{tid}: detail section missing a recognizable 'Priority:' line")

            table_status_kw = _leading_keyword(t.table_status, STATUS_RE)
            if table_status_kw and t.detail_status and table_status_kw.lower() != t.detail_status.lower():
                problems.append(
                    f"PYQ-{tid}: table says Status={t.table_status!r} but detail says {t.detail_status!r}"
                )
            table_priority_kw = _leading_keyword(t.table_priority, PRIORITY_RE)
            if (
                table_priority_kw
                and t.detail_priority
                and table_priority_kw.lower() != t.detail_priority.lower()
            ):
                problems.append(
                    f"PYQ-{tid}: table says Priority={t.table_priority!r} but detail says {t.detail_priority!r}"
                )

    # Cross-file duplicate check (also emitted inline by load_all(), but
    # collect here too so `check` is self-contained and returns a real exit code).
    seen: dict[int, str] = {}
    for ttype, (path, _) in FILES.items():
        for tid in parse_file(ttype, path):
            if tid in seen:
                problems.append(f"PYQ-{tid} is duplicated across {seen[tid]} and {ttype}")
            seen[tid] = ttype

    if problems:
        print(f"{len(problems)} problem(s) found:\n")
        for p in problems:
            print(f"  - {p}")
        return 1

    total = sum(len(parse_file(t, p)) for t, (p, _) in FILES.items())
    print(f"OK -- {total} tickets across {len(FILES)} files, no inconsistencies found.")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    tickets = load_all()

    types = {t.strip().lower() for t in args.type.split(",")} if args.type else None
    priorities = {p.strip().lower() for p in args.priority.split(",")} if args.priority else None
    statuses = {s.strip().lower() for s in args.status.split(",")} if args.status else {"open"}

    rows = []
    for t in tickets.values():
        status_kw = (t.detail_status or _leading_keyword(t.table_status, STATUS_RE) or "").lower()
        priority_kw = (t.detail_priority or _leading_keyword(t.table_priority, PRIORITY_RE) or "").lower()
        if "all" not in statuses and status_kw not in statuses:
            continue
        if types and t.type not in types:
            continue
        if priorities and priority_kw not in priorities:
            continue
        rows.append((PRIORITY_ORDER.get(priority_kw, 99), t))

    rows.sort(key=lambda r: (r[0], r[1].id))

    if not rows:
        print("(no matching tickets)")
        return 0

    id_w = max(len(f"PYQ-{t.id}") for _, t in rows)
    type_w = max(len(t.type) for _, t in rows)
    pri_w = max(len(t.table_priority) for _, t in rows)
    for _, t in rows:
        print(
            f"PYQ-{t.id:<{id_w - 4}} {t.type:<{type_w}} {t.table_priority:<{pri_w}} "
            f"{t.table_status:<10} {t.title}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List tickets (default: open only), sorted by priority.")
    p_list.add_argument("--type", help="Comma-separated: bug,feature,investigation")
    p_list.add_argument("--priority", help="Comma-separated: critical,high,medium,low")
    p_list.add_argument(
        "--status", default="open", help="Comma-separated status filter, or 'all' (default: open)"
    )
    p_list.set_defaults(func=cmd_list)

    p_check = sub.add_parser(
        "check", help="Validate table/detail consistency and ID ranges across backlog files."
    )
    p_check.set_defaults(func=cmd_check)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
