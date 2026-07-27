"""Environment and bundle health check (PYQ-263).

The project has a lot of optional, silently-degrading surface: two API keys, an
optional extra, a TTL cache, named pins, bundles that record a config and a
feature schema, and an accelerator that may or may not be there. Every one of
those degrades gracefully by design, which is correct -- and which also means a
user cannot easily tell what is actually switched on. PYQ-139 is the cautionary
case: an entire vendor dropped out and the only trace was one log line.

The genuinely useful part is the last check: whether each existing bundle's
recorded feature schema *can still be satisfied right now*. That turns "your
bundle is broken" from a runtime error at forecast time into a proactive one.

Kept out of ``cli/`` so the same report can back a ``/healthz`` endpoint
(PYQ-213/PYQ-261) without going through Typer, and out of ``models/`` so
importing it does not drag in torch until it is actually needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from pyquant import provenance
from pyquant.config import Settings, project_root
from pyquant.data import cache as data_cache


@dataclass
class BundleHealth:
    """One trained bundle's identity and whether it is still usable."""

    name: str
    symbols: list[str]
    trained_at: str | None
    n_features: int
    target: str | None
    schema_ok: bool
    problem: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form for ``--format json`` and a future /healthz endpoint."""
        return {
            "name": self.name,
            "symbols": self.symbols,
            "trained_at": self.trained_at,
            "n_features": self.n_features,
            "target": self.target,
            "schema_ok": self.schema_ok,
            "problem": self.problem,
        }


@dataclass
class DoctorReport:
    """Everything ``pyquant doctor`` knows, in one serializable object."""

    keys: dict[str, bool]
    optional_extras: dict[str, bool]
    torch: dict[str, Any]
    paths: dict[str, str]
    cache: dict[str, Any]
    bundles: list[BundleHealth] = field(default_factory=list)
    code_version: str = ""

    @property
    def healthy(self) -> bool:
        """False if any existing bundle can no longer be satisfied."""
        return all(b.schema_ok for b in self.bundles)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able form for ``--format json`` and a future /healthz endpoint."""
        return {
            "healthy": self.healthy,
            "code_version": self.code_version,
            "keys": self.keys,
            "optional_extras": self.optional_extras,
            "torch": self.torch,
            "paths": self.paths,
            "cache": self.cache,
            "bundles": [b.to_dict() for b in self.bundles],
        }


def _torch_info() -> dict[str, Any]:
    """Accelerator and precision support, without hard-failing if torch is absent."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is a hard dependency
        return {"available": False, "error": str(exc)}

    cuda = bool(torch.cuda.is_available())
    mps = bool(getattr(getattr(torch.backends, "mps", None), "is_available", bool)())
    info: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "accelerator": "cuda" if cuda else "mps" if mps else "cpu",
        "cuda": cuda,
        "mps": mps,
    }
    if cuda:
        info["device_name"] = torch.cuda.get_device_name(0)
        # bf16 needs Ampere or newer; reporting it stops a user setting
        # TrainingConfig.precision to something the card silently emulates.
        info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    else:
        info["bf16_supported"] = False
    return info


def _bundle_health(bundle_dir: Path, settings: Settings) -> BundleHealth:
    """Read one bundle's meta.json and check its features are still buildable.

    The check is deliberately *offline*: it compares the recorded feature list
    against the columns the currently-enabled sources would produce, rather than
    fetching. A network check would make `doctor` slow, rate-limited and
    non-deterministic -- and the failure it is looking for (a source switched
    off, a feature renamed by a code change) is visible without one.
    """
    name = bundle_dir.name
    meta_path = bundle_dir / "meta.json"
    if not meta_path.exists():
        return BundleHealth(name, [], None, 0, None, False, "no meta.json")
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return BundleHealth(name, [], None, 0, None, False, f"unreadable meta.json: {exc}")

    features = list(meta.get("features") or [])
    health = BundleHealth(
        name=name,
        symbols=list(meta.get("symbols") or [meta.get("symbol")]),
        trained_at=meta.get("trained_at"),
        n_features=len(features),
        target=meta.get("target"),
        schema_ok=True,
    )

    # Which sources the bundle needs, versus which are switched on now. A bundle
    # trained with sentiment cannot be served with `use_sentiment=False`: that is
    # exactly the PYQ-118 mismatch, and PYQ-119 made it detectable by recording
    # the config.
    needs = {
        "use_macro": any(f.startswith(("VIX", "FedFunds", "YieldSpread", "CPI")) for f in features),
        "use_sectors": any(f.startswith("SEC_") for f in features),
        "use_sentiment": any(f in ("Sentiment", "HeadlineCount") for f in features),
    }
    disabled = [
        flag for flag, needed in needs.items() if needed and not getattr(settings.data, flag)
    ]
    if disabled:
        health.schema_ok = False
        health.problem = (
            "trained with " + ", ".join(sorted(disabled)) + " but they are disabled now"
        )
        return health

    missing_keys = []
    if needs["use_macro"] and any(f.startswith(("FedFunds", "YieldSpread", "CPI")) for f in features):
        if not settings.fred_api_key:
            missing_keys.append("FRED_API_KEY")
    if needs["use_sentiment"] and not settings.finnhub_api_key:
        missing_keys.append("FINNHUB_API_KEY")
    if needs["use_sentiment"] and find_spec("transformers") is None:
        missing_keys.append("the 'sentiment' extra (transformers)")
    if missing_keys:
        health.schema_ok = False
        health.problem = "needs " + ", ".join(missing_keys)
    return health


def run_doctor(settings: Settings) -> DoctorReport:
    """Collect the full health report. Never raises on a degraded environment."""
    checkpoint_dir = settings.checkpoint_dir
    bundle_dirs = (
        sorted(d for d in checkpoint_dir.iterdir() if d.is_dir())
        if checkpoint_dir.exists()
        else []
    )

    try:
        cache_stats = data_cache.cache_stats(settings.data.cache_dir)
    except OSError as exc:
        cache_stats = {"error": str(exc), "entry_count": 0, "total_bytes": 0, "pins": []}

    return DoctorReport(
        # Presence only -- never the value. Secrets do not enter meta.json,
        # runs.jsonl, logs or cache fingerprints, and this is one of those.
        keys={
            "FRED_API_KEY": bool(settings.fred_api_key),
            "FINNHUB_API_KEY": bool(settings.finnhub_api_key),
        },
        optional_extras={"transformers (sentiment)": find_spec("transformers") is not None},
        torch=_torch_info(),
        paths={
            "project_root": str(project_root()),
            "checkpoint_dir": str(checkpoint_dir),
            "cache_dir": str(settings.data.cache_dir),
        },
        cache=cache_stats,
        bundles=[_bundle_health(d, settings) for d in bundle_dirs],
        code_version=provenance.code_version(),
    )
