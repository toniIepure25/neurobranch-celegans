"""Adapter for the Baskoylu & Flavell (2026) freely-moving C. elegans data.

Each animal is one HDF5 file.  The export stores traces as ``(time, neuron)``
(despite the Julia README describing the transposed column-major view), and the
behaviour arrays are already resampled to the 1,600 confocal frames.  Reversal
events use **0-based start and exclusive end** (verified against
``behavior/reversal_vec``).

The published ``gcamp/trace_array`` is z-scored.  The adapter therefore exposes
the original traces separately and never uses the published z-score for the
primary benchmark.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from neurobranch.config import sha256_file

DATASET_DOI = "doi:10.5061/dryad.w9ghx3g4v"
DATASET_LICENSE = "CC0-1.0"
DEFAULT_RAW = "data/raw/baskoylu2026/files"
DEFAULT_LABELS = "data/raw/baskoylu2026/neuropal_labels.tsv"


@dataclass
class FreelyMovingRecording:
    animal_id: str
    source_file: str
    source_sha256: str
    traces_raw: np.ndarray  # (time, neuron), original GCaMP
    traces_published_z: np.ndarray  # (time, neuron), whole-recording z-score
    time_seconds: np.ndarray  # relative to first confocal frame
    velocity: np.ndarray
    reversal_vec: np.ndarray
    reversal_onsets: np.ndarray  # 0-based frame where reversal begins
    reversal_offsets: np.ndarray  # 0-based exclusive end
    head_angle: np.ndarray
    angular_velocity: np.ndarray
    distance_from_center: np.ndarray
    labels: tuple[str | None, ...] = ()
    label_confidence: tuple[int | None, ...] = ()
    label_ambiguous: tuple[bool, ...] = ()
    quality: dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.traces_raw.shape[0])

    @property
    def n_neurons(self) -> int:
        return int(self.traces_raw.shape[1])

    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.time_seconds)))

    @property
    def duration_s(self) -> float:
        return float(self.time_seconds[-1] - self.time_seconds[0])

    @property
    def event_rate_per_min(self) -> float:
        return float(len(self.reversal_onsets) / max(self.duration_s / 60.0, 1e-9))

    @property
    def identified_indices(self) -> dict[str, int]:
        return {
            name: i
            for i, (name, amb) in enumerate(zip(self.labels, self.label_ambiguous))
            if name is not None and not amb
        }


def discover_recordings(data_dir: str | Path = DEFAULT_RAW) -> list[Path]:
    return sorted(Path(data_dir).glob("*cudata.h5"))


def _animal_id(path: Path) -> str:
    return path.name[: -len("-cudata.h5")]


def load_labels(path: str | Path = DEFAULT_LABELS) -> pd.DataFrame:
    """Load the exported NeuroPAL TSV; trace indices are 0-based."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["animal_id", "trace_index", "label", "neuron_class", "confidence"])
    d = pd.read_csv(p, sep="\t")
    d["trace_index"] = d["trace_index"].astype(int)
    d["confidence"] = d["confidence"].astype(int)
    return d


def _labels_for(
    animal_id: str, n_neurons: int, labels: pd.DataFrame
) -> tuple[tuple[str | None, ...], tuple[int | None, ...], tuple[bool, ...]]:
    names: list[str | None] = [None] * n_neurons
    conf: list[int | None] = [None] * n_neurons
    if labels.empty:
        return tuple(names), tuple(conf), tuple(False for _ in range(n_neurons))
    sub = labels[labels["animal_id"] == animal_id]
    for _, row in sub.iterrows():
        i = int(row["trace_index"])
        if 0 <= i < n_neurons:
            names[i] = str(row["label"])
            conf[i] = int(row["confidence"])
    counts: dict[str, int] = {}
    for name in names:
        if name is not None:
            counts[name] = counts.get(name, 0) + 1
    ambiguous = tuple(name is not None and counts[name] > 1 for name in names)
    return tuple(names), tuple(conf), ambiguous


def _validate_events(
    animal_id: str, reversal_vec: np.ndarray, events: np.ndarray, n_frames: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if events.ndim != 2 or events.shape[1] != 2:
        raise ValueError(f"{animal_id}: reversal_events must have two columns, got {events.shape}")
    # The export uses 0-based start and exclusive end.  This was verified against
    # reversal_vec: e.g. event [13, 30] matches reversal_vec[13:30] == 1.
    starts = events[:, 0].astype(int)
    ends = events[:, 1].astype(int)
    if np.any(starts < 0) or np.any(ends > n_frames) or np.any(ends <= starts):
        raise ValueError(f"{animal_id}: reversal event bounds are invalid")
    if np.any(np.diff(starts) < 0):
        raise ValueError(f"{animal_id}: reversal events are not sorted")
    if len(starts) > 1 and np.any(starts[1:] < ends[:-1]):
        raise ValueError(f"{animal_id}: overlapping reversal events")
    occupied = np.zeros(n_frames, dtype=bool)
    for s, e in zip(starts, ends):
        occupied[s:e] = True
        if not np.all(reversal_vec[s:e] == 1):
            raise ValueError(f"{animal_id}: event [{s}, {e}) contains non-reversal frames")
    mismatch = np.flatnonzero(occupied.astype(np.int8) != reversal_vec.astype(np.int8))
    # A recording can start in the middle of a reversal; the event table omits
    # that initial partial segment.  Only frame 0 may differ.
    if len(mismatch) and not (len(mismatch) == 1 and mismatch[0] == 0):
        raise ValueError(
            f"{animal_id}: reversal_events disagree with reversal_vec in {len(mismatch)} frames"
        )
    return starts, ends, {
        "event_vec_mismatch_frames": int(len(mismatch)),
        "initial_partial_reversal": bool(len(mismatch) == 1),
        "events_validated": int(len(starts)),
    }


def load_recording(
    path: str | Path,
    *,
    labels: pd.DataFrame | None = None,
    labels_path: str | Path = DEFAULT_LABELS,
) -> FreelyMovingRecording:
    p = Path(path)
    animal_id = _animal_id(p)
    lab = labels if labels is not None else load_labels(labels_path)
    with h5py.File(p, "r") as f:
        raw = np.asarray(f["gcamp/trace_array_original"][:], dtype=np.float64)
        published = np.asarray(f["gcamp/trace_array"][:], dtype=np.float64)
        t = np.asarray(f["timing/timestamp_confocal"][:], dtype=np.float64).ravel()
        velocity = np.asarray(f["behavior/velocity"][:], dtype=np.float64).ravel()
        rv = np.asarray(f["behavior/reversal_vec"][:], dtype=np.int64).ravel()
        events = np.asarray(f["behavior/reversal_events"][:], dtype=np.int64)
        head = np.asarray(f["behavior/head_angle"][:], dtype=np.float64).ravel()
        angular = np.asarray(f["behavior/angular_velocity"][:], dtype=np.float64).ravel()
        dist = np.asarray(f["behavior/dist_from_cuCenter"][:], dtype=np.float64).ravel()
    if events.shape[0] == 2:
        events = events.T
    if raw.ndim != 2:
        raise ValueError(f"{animal_id}: trace array must be 2-D")
    if raw.shape[0] != len(t):
        raise ValueError(f"{animal_id}: {raw.shape[0]} trace frames but {len(t)} timestamps")
    for name, arr in [
        ("velocity", velocity),
        ("reversal_vec", rv),
        ("head_angle", head),
        ("angular_velocity", angular),
        ("distance", dist),
    ]:
        if arr.shape != (len(t),):
            raise ValueError(f"{animal_id}: {name} has shape {arr.shape}, expected ({len(t)},)")
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError(f"{animal_id}: confocal timestamps are not strictly increasing")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{animal_id}: raw traces contain non-finite values")
    starts, ends, event_qc = _validate_events(animal_id, rv, events, len(t))
    names, conf, ambiguous = _labels_for(animal_id, raw.shape[1], lab)
    time_s = t - t[0]
    quality = {
        "n_frames": int(len(t)),
        "n_neurons": int(raw.shape[1]),
        "duration_s": float(time_s[-1]),
        "median_dt_s": float(np.median(np.diff(time_s))),
        "n_reversal_events": int(len(starts)),
        "reversal_fraction": float(np.mean(rv)),
        "event_rate_per_min": float(len(starts) / max(time_s[-1] / 60.0, 1e-9)),
        "n_labeled_traces": int(sum(x is not None for x in names)),
        "n_identified_traces": int(sum(x is not None and not a for x, a in zip(names, ambiguous))),
        "n_ambiguous_traces": int(sum(ambiguous)),
        **event_qc,
    }
    return FreelyMovingRecording(
        animal_id=animal_id,
        source_file=str(p),
        source_sha256=sha256_file(p),
        traces_raw=raw,
        traces_published_z=published,
        time_seconds=time_s,
        velocity=velocity,
        reversal_vec=rv,
        reversal_onsets=starts,
        reversal_offsets=ends,
        head_angle=head,
        angular_velocity=angular,
        distance_from_center=dist,
        labels=names,
        label_confidence=conf,
        label_ambiguous=ambiguous,
        quality=quality,
    )


def load_all(
    data_dir: str | Path = DEFAULT_RAW,
    *,
    labels_path: str | Path = DEFAULT_LABELS,
    min_events: int = 5,
    min_frames: int = 300,
) -> tuple[list[FreelyMovingRecording], list[dict[str, Any]]]:
    """Load all recordings and return usable ones plus a coverage audit.

    Exclusions are structural (unreadable, too short, too few events), not based
    on model performance.
    """
    lab = load_labels(labels_path)
    recs: list[FreelyMovingRecording] = []
    rows: list[dict[str, Any]] = []
    for p in discover_recordings(data_dir):
        animal = _animal_id(p)
        try:
            rec = load_recording(p, labels=lab)
        except Exception as exc:  # noqa: BLE001 - audit must record every failure
            rows.append({"animal_id": animal, "usable": False, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        usable = rec.n_frames >= min_frames and len(rec.reversal_onsets) >= min_events
        reason = ""
        if not usable:
            reason = f"frames {rec.n_frames} < {min_frames} or events {len(rec.reversal_onsets)} < {min_events}"
        rows.append({**rec.quality, "animal_id": animal, "usable": usable, "reason": reason})
        if usable:
            recs.append(rec)
    return recs, rows


def write_audit(
    recs: list[FreelyMovingRecording],
    rows: list[dict[str, Any]],
    *,
    out_dir: str | Path = "reports/phase3",
    labels_path: str | Path = DEFAULT_LABELS,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "dataset_coverage.csv", index=False)
    label_rows = []
    all_labels = load_labels(labels_path)
    for rec in recs:
        for i, (name, conf, amb) in enumerate(zip(rec.labels, rec.label_confidence, rec.label_ambiguous)):
            if name is not None:
                label_rows.append(
                    {
                        "animal_id": rec.animal_id,
                        "trace_index": i,
                        "label": name,
                        "confidence": conf,
                        "ambiguous": bool(amb),
                    }
                )
    pd.DataFrame(label_rows).to_csv(out / "neuron_identity_coverage.csv", index=False)
    summary = {
        "dataset_doi": DATASET_DOI,
        "license": DATASET_LICENSE,
        "n_animals_total": len(rows),
        "n_animals_usable": len(recs),
        "n_frames_total": int(sum(r.n_frames for r in recs)),
        "n_reversal_events": int(sum(len(r.reversal_onsets) for r in recs)),
        "n_neurons_min": int(min(r.n_neurons for r in recs)) if recs else 0,
        "n_neurons_max": int(max(r.n_neurons for r in recs)) if recs else 0,
        "n_unique_labels": int(all_labels["label"].nunique()) if not all_labels.empty else 0,
        "n_labeled_traces": int(
            sum(1 for r in recs for n in r.labels if n is not None)
        ),
        "n_ambiguous_label_traces": int(sum(r.quality["n_ambiguous_traces"] for r in recs)),
        "median_dt_s": float(np.median([r.dt for r in recs])) if recs else None,
        "animals": [r.animal_id for r in recs],
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
