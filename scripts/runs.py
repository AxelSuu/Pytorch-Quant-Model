"""Compare training runs recorded in runs.jsonl, across every bundle (PYQ-259).

`runs.jsonl` (PYQ-203) plus provenance (PYQ-225) plus dataset pins (PYQ-205) already
record everything one run needs to be reproduced. What they cannot do is answer "which
of my last 30 runs had the best skill, and what did they have in common" -- that means
reading every bundle's `runs.jsonl` and lining the fields up, which is what this does.

This is the ~100-line alternative `backlog/features.md#pyq-259` evaluates against adding
`mlflow` as a dependency; see that ticket's resolution note for the reasoning behind
shipping this instead.

Usage:
    uv run python scripts/runs.py compare
    uv run python scripts/runs.py compare --symbol AAPL --sort-by val_loss --top 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_runs(checkpoint_dir: Path) -> list[dict[str, Any]]:
    """Every recorded run across every bundle directory under checkpoint_dir."""
    runs: list[dict[str, Any]] = []
    if not checkpoint_dir.is_dir():
        return runs
    for runs_file in sorted(checkpoint_dir.glob("*/runs.jsonl")):
        bundle_dir = runs_file.parent.name
        for line in runs_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record["_bundle_dir"] = bundle_dir
            runs.append(record)
    return runs


def _skill(record: dict[str, Any]) -> float:
    ev = record.get("evaluation") or {}
    baseline_mae, model_mae = ev.get("baseline_mae"), ev.get("model_mae")
    if not baseline_mae:
        return float("-inf")
    return (baseline_mae - model_mae) / baseline_mae


_SORT_KEYS = {
    "skill": _skill,
    "val_loss": lambda r: r.get("val_loss", float("inf")),
    "trained_at": lambda r: r.get("trained_at", ""),
}


def cmd_compare(args: argparse.Namespace) -> int:
    from pyquant.config import Settings

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Settings().checkpoint_dir
    runs = load_runs(checkpoint_dir)
    if args.symbol:
        runs = [r for r in runs if r.get("symbol", "").upper() == args.symbol.upper()]
    if not runs:
        print(f"(no runs found under {checkpoint_dir})")
        return 0

    key = _SORT_KEYS.get(args.sort_by)
    if key is None:
        print(f"Unknown --sort-by {args.sort_by!r}; choose from {sorted(_SORT_KEYS)}")
        return 1
    runs.sort(key=key, reverse=(args.sort_by != "val_loss"))
    if args.top:
        runs = runs[: args.top]

    header = (
        f"{'bundle':<16} {'trained_at':<20} {'target':<10} {'seed':>5} {'epochs':>6} "
        f"{'val_loss':>10} {'skill':>8} {'dir_acc':>8} {'coverage':>9} {'sha':<8}"
    )
    print(header)
    print("-" * len(header))
    for r in runs:
        ev = r.get("evaluation") or {}
        sha = (r.get("provenance") or {}).get("git_sha") or "?"
        print(
            f"{r.get('_bundle_dir', '?'):<16} {r.get('trained_at', '?'):<20} "
            f"{r.get('target', '?'):<10} {r.get('seed', '?'):>5} {r.get('epochs_run', '?'):>6} "
            f"{r.get('val_loss', float('nan')):>10.5f} {_skill(r):>+8.3f} "
            f"{ev.get('directional_accuracy', float('nan')):>8.3f} "
            f"{ev.get('calibration_coverage', float('nan')):>9.3f} {sha[:8]:<8}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_compare = sub.add_parser("compare", help="Table of runs across every bundle, sorted by skill.")
    p_compare.add_argument("--checkpoint-dir", help="Override the checkpoint directory (default: Settings())")
    p_compare.add_argument("--symbol", help="Restrict to one bundle's runs.jsonl")
    p_compare.add_argument("--sort-by", default="skill", choices=sorted(_SORT_KEYS))
    p_compare.add_argument("--top", type=int, default=30, help="Max rows to show (default 30)")
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
