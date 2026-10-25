"""Dataset registry for Phase 6.

The registry is a YAML file, not a Python constant, so that every dataset's
licence, timebase and permitted use are visible in review.  Missing files are a
normal condition: the registry reports availability rather than crashing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from neurobranch.config import PROJECT_ROOT, resolve_path
from neurobranch.data.schemas import DatasetSpec, SchemaError

DEFAULT_REGISTRY = "configs/datasets.yaml"


def load_registry(path: str | Path | None = None, *, root: Path | None = None) -> list[DatasetSpec]:
    p = resolve_path(path or DEFAULT_REGISTRY, root)
    if not p.exists():
        raise FileNotFoundError(f"dataset registry not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    rows = raw.get("datasets")
    if not isinstance(rows, list):
        raise SchemaError(f"{p}: expected a 'datasets' list")
    specs: list[DatasetSpec] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise SchemaError(f"{p}: each dataset must be a mapping")
        try:
            spec = DatasetSpec(
                key=str(row["key"]),
                citation=str(row["citation"]),
                license=None if row.get("license") is None else str(row["license"]),
                raw_glob=str(row["raw_glob"]),
                loader=str(row["loader"]),
                timebase_s=float(row["timebase_s"]),
                modalities=tuple(row["modalities"]),
                allowed_uses=tuple(row["allowed_uses"]),
                sha256=None if row.get("sha256") is None else str(row["sha256"]),
                notes=str(row.get("notes", "")),
            )
        except KeyError as exc:
            raise SchemaError(f"{p}: dataset missing field {exc.args[0]!r}") from exc
        if spec.key in seen:
            raise SchemaError(f"{p}: duplicate dataset key {spec.key!r}")
        seen.add(spec.key)
        specs.append(spec)
    return specs


def availability(spec: DatasetSpec, *, root: Path | None = None) -> dict[str, Any]:
    base = root or PROJECT_ROOT
    pattern = spec.raw_glob
    # A glob with no wildcard is treated as a literal path.
    matches = sorted(base.glob(pattern)) if any(ch in pattern for ch in "*?[") else [base / pattern]
    existing = [p for p in matches if p.exists()]
    return {
        "key": spec.key,
        "available": bool(existing),
        "n_matches": len(existing),
        "examples": [str(p.relative_to(base)) for p in existing[:5]],
        "license": spec.license,
        "allowed_uses": list(spec.allowed_uses),
    }


def registry_report(path: str | Path | None = None, *, root: Path | None = None) -> dict[str, Any]:
    specs = load_registry(path, root=root)
    return {
        "registry": str(resolve_path(path or DEFAULT_REGISTRY, root)),
        "n_datasets": len(specs),
        "datasets": [availability(s, root=root) for s in specs],
    }
