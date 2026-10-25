"""Typed records and validation for Phase 6 datasets.

The Phase 6 benchmark deliberately keeps the schema small.  A dataset is a
collection of animals, each with labelled traces, stimulation events and
downstream responses.  Nothing is imputed at this layer: ambiguous labels are
represented as ``None`` and are excluded by the task code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class SchemaError(ValueError):
    """Raised when a Phase 6 record violates the documented schema."""


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

    def __post_init__(self) -> None:
        if not self.key:
            raise SchemaError("dataset key must be non-empty")
        if self.timebase_s <= 0:
            raise SchemaError(f"{self.key}: timebase_s must be positive")
        if not self.modalities:
            raise SchemaError(f"{self.key}: at least one modality is required")
        allowed = {"train", "eval", "external_only", "demo"}
        bad = set(self.allowed_uses) - allowed
        if bad:
            raise SchemaError(f"{self.key}: unknown allowed_uses {sorted(bad)}")
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
