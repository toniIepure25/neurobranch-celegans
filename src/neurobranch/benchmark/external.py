"""Contract for an external stimulus-response transfer cohort.

No external cohort with the same trial-level optogenetic task is present in this
checkout.  This module validates a candidate table and reports incompatibilities;
it does not manufacture a transfer result from a dataset with a different task.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, resolve_path

REQUIRED_COLUMNS = (
    "dataset_id",
    "event_id",
    "target",
    "neuron",
    "responses",
)
MIN_COLUMNS = ("dataset_id", "event_id", "target", "neuron", "response_mean")


@dataclass(frozen=True)
class ExternalAudit:
    path: str
    n_rows: int
    n_animals: int
    n_events: int
    n_targets: int
    n_responders: int
    columns: tuple[str, ...]
    missing_required: tuple[str, ...]
    compatible: bool
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "n_rows": self.n_rows,
            "n_animals": self.n_animals,
            "n_events": self.n_events,
            "n_targets": self.n_targets,
            "n_responders": self.n_responders,
            "columns": list(self.columns),
            "missing_required": list(self.missing_required),
            "compatible": self.compatible,
            "notes": list(self.notes),
        }


def audit_external_table(path: str | Path, *, root: str | Path | None = None) -> ExternalAudit:
    base = Path(root) if root is not None else PROJECT_ROOT
    p = resolve_path(path, base)
    if not p.exists():
        raise FileNotFoundError(p)
    frame = (
        pd.read_parquet(p)
        if p.suffix.lower() in {".parquet", ".pq"}
        else pd.read_csv(p)
    )
    missing = tuple(c for c in REQUIRED_COLUMNS if c not in frame.columns)
    # response_mean is accepted because the Phase 6 endpoint uses it.
    missing = tuple(c for c in missing if c != "responses" or "response_mean" not in frame.columns)
    notes = []
    if missing:
        notes.append("missing required columns: " + ", ".join(missing))
    for col in ("dataset_id", "event_id", "target", "neuron"):
        if col in frame.columns and frame[col].isna().any():
            notes.append(f"{col} contains missing values")
    if "response_mean" in frame.columns and not np.isfinite(frame["response_mean"]).all():
        notes.append("response_mean contains non-finite values")
    if "event_id" in frame.columns and "target" in frame.columns:
        per_target = frame.groupby("target")["event_id"].nunique()
        if (per_target < 2).any():
            notes.append("some targets have fewer than two events")
    return ExternalAudit(
        path=str(p),
        n_rows=int(len(frame)),
        n_animals=int(frame["dataset_id"].nunique()) if "dataset_id" in frame else 0,
        n_events=int(frame["event_id"].nunique()) if "event_id" in frame else 0,
        n_targets=int(frame["target"].nunique()) if "target" in frame else 0,
        n_responders=int(frame["neuron"].nunique()) if "neuron" in frame else 0,
        columns=tuple(str(c) for c in frame.columns),
        missing_required=missing,
        compatible=not missing,
        notes=tuple(notes),
    )
