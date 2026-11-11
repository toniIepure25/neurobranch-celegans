"""Typed records and validation for Phase 6 datasets.

The Phase 6 benchmark deliberately keeps the schema small.  A dataset is a
collection of animals, each with labelled traces, stimulation events and
downstream responses.  Nothing is imputed at this layer: ambiguous labels are
represented as ``None`` and are excluded by the task code.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_UNITS = {"animal", "recording", "session", "trial"}
_CLAIM_LEVELS = {"none", "external_only", "descriptive", "causal"}
_USE_LEVELS = {"train", "eval", "external_only", "demo"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SchemaError(ValueError):
    """Raised when a Phase 6 record violates the documented schema."""


@dataclass(frozen=True)
class CohortManifest:
    """Provenance record for one incoming cohort.

    The manifest is intentionally separate from the dataset registry: it travels
    with a delivery, records what was actually received, and can be rejected
    before any loader sees the data.
    """

    cohort_id: str
    source: str
    license: str | None
    sha256: str
    unit: str
    n_units: int
    timebase_s: float
    files: tuple[str, ...]
    behaviors: tuple[str, ...] = ()
    intervention: bool = False
    allowed_uses: tuple[str, ...] = ("external_only",)
    owner: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.cohort_id:
            raise SchemaError("cohort_id must be non-empty")
        if not self.source:
            raise SchemaError(f"{self.cohort_id}: source must be non-empty")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise SchemaError(f"{self.cohort_id}: sha256 must be 64 lowercase hex characters")
        if self.unit not in _UNITS:
            raise SchemaError(f"{self.cohort_id}: unknown unit {self.unit!r}")
        if self.n_units <= 0:
            raise SchemaError(f"{self.cohort_id}: n_units must be positive")
        if self.timebase_s <= 0:
            raise SchemaError(f"{self.cohort_id}: timebase_s must be positive")
        if not self.files:
            raise SchemaError(f"{self.cohort_id}: at least one file is required")
        bad_paths = [
            str(path)
            for path in self.files
            if Path(path).is_absolute() or ".." in Path(path).parts
        ]
        if bad_paths:
            raise SchemaError(
                f"{self.cohort_id}: files must be relative and stay inside the cohort "
                f"directory; rejected {bad_paths}"
            )
        if len(set(self.files)) != len(self.files):
            raise SchemaError(f"{self.cohort_id}: duplicate files are not allowed")
        bad_uses = set(self.allowed_uses) - _USE_LEVELS
        if bad_uses:
            raise SchemaError(f"{self.cohort_id}: unknown allowed_uses {sorted(bad_uses)}")
        if self.license is None and "train" in self.allowed_uses:
            raise SchemaError(
                f"{self.cohort_id}: training requires an explicit licence; "
                "no licence means external_only"
            )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["files"] = list(self.files)
        out["behaviors"] = list(self.behaviors)
        out["allowed_uses"] = list(self.allowed_uses)
        return out

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CohortManifest:
        data = dict(payload)
        for key in ("files", "behaviors", "allowed_uses"):
            if key not in data:
                continue
            if not isinstance(data[key], (list, tuple)):
                raise SchemaError(f"{data.get('cohort_id', 'cohort')}: {key} must be a list")
            if any(not isinstance(item, str) for item in data[key]):
                raise SchemaError(f"{data.get('cohort_id', 'cohort')}: {key} must contain strings")
            data[key] = tuple(data[key])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_path(cls, path: str | Path) -> CohortManifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SchemaError(f"{path}: manifest must be a JSON object")
        return cls.from_dict(payload)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    citation: str
    license: str | None
    raw_glob: str
    loader: str
    timebase_s: float
    modalities: tuple[str, ...]
    allowed_uses: tuple[str, ...]
    sha256: str | None = None
    notes: str = ""
    required_columns: tuple[str, ...] = ()
    expected_units: str | None = None
    claim_level: str = "external_only"
    owner: str | None = None

    def __post_init__(self) -> None:
        if not self.key:
            raise SchemaError("dataset key must be non-empty")
        if self.timebase_s <= 0:
            raise SchemaError(f"{self.key}: timebase_s must be positive")
        if not self.modalities:
            raise SchemaError(f"{self.key}: at least one modality is required")
        allowed = _USE_LEVELS
        bad = set(self.allowed_uses) - allowed
        if bad:
            raise SchemaError(f"{self.key}: unknown allowed_uses {sorted(bad)}")
        if self.claim_level not in _CLAIM_LEVELS:
            raise SchemaError(
                f"{self.key}: unknown claim_level {self.claim_level!r}; "
                f"expected one of {sorted(_CLAIM_LEVELS)}"
            )
        if self.expected_units is not None and self.expected_units not in _UNITS:
            raise SchemaError(f"{self.key}: unknown expected_units {self.expected_units!r}")
        if any(not name for name in self.required_columns):
            raise SchemaError(f"{self.key}: required_columns must be non-empty strings")
        if self.license is None and "train" not in self.allowed_uses:
            # No explicit licence: local training is allowed only when the user
            # obtained the data themselves; redistribution is never implied.
            return

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnimalRecord:
    animal_id: str
    dataset_key: str
    n_frames: int
    dt: float
    source: str = ""

    def __post_init__(self) -> None:
        if self.n_frames <= 0:
            raise SchemaError(f"{self.animal_id}: n_frames must be positive")
        if self.dt <= 0:
            raise SchemaError(f"{self.animal_id}: dt must be positive")


@dataclass(frozen=True)
class NeuronRecord:
    dataset_key: str
    raw_label: str
    canonical_label: str | None
    trace_index: int
    ambiguous: bool = False

    def __post_init__(self) -> None:
        if self.trace_index < 0:
            raise SchemaError("trace_index must be non-negative")
        if self.ambiguous and self.canonical_label is not None:
            raise SchemaError("an ambiguous label cannot have a canonical identity")


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    animal_id: str
    target: str
    stim_frame: int
    time_seconds: float
    context: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stim_frame < 0:
            raise SchemaError(f"{self.event_id}: stim_frame must be non-negative")
        if not self.target:
            raise SchemaError(f"{self.event_id}: target is required")


@dataclass(frozen=True)
class ResponseRecord:
    event_id: str
    responder: str
    value: float
    metric: str = "response_mean"

    def __post_init__(self) -> None:
        if not self.responder:
            raise SchemaError("responder is required")
        if self.metric not in {"response", "response_mean", "response_integral"}:
            raise SchemaError(f"unknown response metric {self.metric!r}")


@dataclass(frozen=True)
class SplitManifest:
    name: str
    fold: int
    train_rows: tuple[int, ...]
    val_rows: tuple[int, ...]
    test_rows: tuple[int, ...]
    held_out: tuple[str, ...]
    seed: int

    def __post_init__(self) -> None:
        sets = [set(self.train_rows), set(self.val_rows), set(self.test_rows)]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise SchemaError(f"{self.name}: split rows overlap")
        if not self.test_rows:
            raise SchemaError(f"{self.name}: empty test split")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["train_rows"] = list(self.train_rows)
        out["val_rows"] = list(self.val_rows)
        out["test_rows"] = list(self.test_rows)
        out["held_out"] = list(self.held_out)
        return out
