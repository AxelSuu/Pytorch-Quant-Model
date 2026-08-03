"""The multi-symbol, multi-configuration sweep runner (PYQ-268).

Repeats a walk-forward backtest across every (symbol, arm) cell of a matrix and
returns a tidy result set -- the instrument the project's own
``backlog/README.md`` ``## Now`` list has named as the prerequisite for three
pending repeats (PYQ-247's target comparison, investigations.md#pyq-315's
pooling result, #pyq-316's feature ablation) across two review passes without
it existing.

Explicitly *not* in scope: running any of those repeats. This module delivers
the instrument; the results those runs would produce are separate, later work,
and what investigations.md#pyq-322's decision rule consumes once they exist.

Calls into ``models.tft.walk_forward_backtest`` (the only Lightning-touching
dependency) rather than reimplementing it, but does not itself import torch,
Lightning or pytorch-forecasting -- consistent with how ``cli/app.py`` already
calls into ``models/tft.py`` without importing those directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyquant.analysis.metrics import PairedComparison, ScoredWindows, compare_backtests
from pyquant.config import Settings
from pyquant.models.tft import BacktestResult, walk_forward_backtest

# Sub-configs checked, in order, when an override key has no dotted prefix.
_SUB_CONFIGS = ("training", "data", "tft")


@dataclass
class Arm:
    """A named configuration variant: ``overrides`` applied on top of a base ``Settings``."""

    name: str
    overrides: dict[str, Any]


def _resolve_override_target(settings: Any, key: str) -> tuple[Any, str]:
    """Find which sub-config an override key belongs to; return (sub_config, attr)."""
    if "." in key:
        sub_name, attr = key.split(".", 1)
        sub = getattr(settings, sub_name, None)
        if sub is None or not hasattr(sub, attr):
            raise ValueError(f"unknown override key {key!r}")
        return sub, attr
    matches = [name for name in _SUB_CONFIGS if hasattr(getattr(settings, name, None), key)]
    if not matches:
        raise ValueError(
            f"unknown override key {key!r}: not found on any of {_SUB_CONFIGS} "
            "(use a dotted path, e.g. 'training.target', if it belongs elsewhere)"
        )
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous override key {key!r}: present on {matches}, use a dotted path "
            "(e.g. 'training.use_indicators') to disambiguate"
        )
    return getattr(settings, matches[0]), key


def _coerce(current: Any, raw: str) -> Any:
    """Coerce a CLI override string into the type the current field value already has."""
    if isinstance(current, bool):
        lowered = raw.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        raise ValueError(f"cannot parse {raw!r} as a bool")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def apply_overrides(settings: Settings, overrides: dict[str, Any]) -> Settings:
    """A deep copy of ``settings`` with each ``overrides`` entry applied.

    Values may already be the right type (an ``Arm`` built directly in Python)
    or a raw CLI string (``pyquant sweep --arm key=value``) -- coerced against
    whatever type the field already holds. The caller's own ``settings`` is
    never mutated.
    """
    out = settings.model_copy(deep=True)
    for key, value in overrides.items():
        sub, attr = _resolve_override_target(out, key)
        current = getattr(sub, attr)
        setattr(sub, attr, _coerce(current, value) if isinstance(value, str) else value)
    return out


@dataclass
class SweepCell:
    """One (symbol, arm) cell's outcome -- a result, or a recorded gap.

    A failing cell (insufficient history, a data-fetch error, an override that
    doesn't apply to this symbol) does not take the whole sweep down; it
    degrades to ``error`` instead, and is excluded from every ``SweepResult``
    aggregate rather than silently coercing to zero.
    """

    symbol: str
    arm: str
    result: BacktestResult | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True if this cell has a result rather than a recorded error."""
        return self.result is not None


@dataclass
class SweepResult:
    """The full symbol x arm cell matrix from one sweep (PYQ-268).

    "Helped 11 of 15 symbols" and "mean skill +0.3%" answer different
    questions, and only the pair is honest -- both ``helped_summary`` and
    ``pooled_skill`` are exposed rather than collapsing to one number.
    """

    symbols: list[str]
    arm_names: list[str]
    cells: list[SweepCell]

    def cell(self, symbol: str, arm: str) -> SweepCell:
        """The cell for (symbol, arm); raises KeyError if it isn't in this sweep."""
        for c in self.cells:
            if c.symbol == symbol and c.arm == arm:
                return c
        raise KeyError((symbol, arm))

    def skill_by_symbol(self, arm: str) -> dict[str, float]:
        """{symbol: skill_vs_baseline} for every symbol where ``arm`` succeeded."""
        return {
            c.symbol: c.result.aggregated.skill_vs_baseline
            for c in self.cells
            if c.arm == arm and c.ok
        }

    def pooled_skill(self, arm: str) -> float | None:
        """Unweighted mean skill across symbols where ``arm`` succeeded, or None if none did."""
        skills = list(self.skill_by_symbol(arm).values())
        return float(sum(skills) / len(skills)) if skills else None

    def helped_summary(self, base_arm: str, other_arm: str) -> tuple[int, int]:
        """(symbols where ``other_arm`` scored higher than ``base_arm``, symbols where both succeeded)."""
        base = self.skill_by_symbol(base_arm)
        other = self.skill_by_symbol(other_arm)
        common = sorted(set(base) & set(other))
        helped = sum(1 for s in common if other[s] > base[s])
        return helped, len(common)

    def paired_comparison(
        self, symbol: str, base_arm: str, other_arm: str
    ) -> PairedComparison | None:
        """A window-paired comparison for one symbol between two arms (PYQ-266).

        None if either cell failed, or if their windows don't verifiably align
        (an override changed the window geometry between arms) -- refusing to
        compare rather than raising, the same degrade-gracefully shape a sweep
        already applies to a failing cell.
        """
        base_cell, other_cell = self.cell(symbol, base_arm), self.cell(symbol, other_arm)
        if not base_cell.ok or not other_cell.ok:
            return None
        try:
            return compare_backtests(
                ScoredWindows(base_cell.result.per_window, base_cell.result.origins),
                ScoredWindows(other_cell.result.per_window, other_cell.result.origins),
            )
        except ValueError:
            return None


def run_sweep(
    symbols: list[str],
    arms: list[Arm],
    settings: Settings,
    *,
    n_windows: int = 5,
    step: int | None = None,
    max_epochs: int | None = None,
    progress: bool = False,
) -> SweepResult:
    """Run ``walk_forward_backtest`` for every (symbol, arm) cell.

    A cell that raises degrades to a recorded ``SweepCell.error`` rather than
    aborting the sweep -- fifteen symbols should not all fail because one has
    too little history for one arm.
    """
    cells: list[SweepCell] = []
    for arm in arms:
        arm_settings = apply_overrides(settings, arm.overrides)
        for symbol in symbols:
            try:
                result = walk_forward_backtest(
                    symbol,
                    arm_settings,
                    n_windows=n_windows,
                    step=step,
                    max_epochs=max_epochs,
                    progress=progress,
                )
                cells.append(SweepCell(symbol=symbol, arm=arm.name, result=result))
            except Exception as exc:
                cells.append(SweepCell(symbol=symbol, arm=arm.name, error=str(exc)))
    return SweepResult(symbols=list(symbols), arm_names=[a.name for a in arms], cells=cells)
