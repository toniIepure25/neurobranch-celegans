"""Loader and provenance-aware audit utilities for the Kato et al. (2015) data.

The public OSF archive stores MATLAB v7.3 (HDF5) files.  Three details matter
for scientific correctness and are handled explicitly here:

* ``NeuronNames`` entries are arrays of Unicode code points, not Python
  strings; one entry in worm 3 is an extra level of cell nesting.
* ``States`` contains eight mutually exclusive one-hot fields, including
  ``dt`` (dorsal turn).  Dropping ``dt`` silently relabels turns as forward.
* Behaviour labels in this dataset were derived using AVA, SMD and RIB
  neurons.  Those neurons must be excluded from predictors used to evaluate
  behaviour decoding, otherwise the task is circular.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, resolve_path, sha256_file

STATE_ORDER: tuple[str, ...] = ("dt", "fwd", "nostate", "rev1", "rev2", "revsus", "slow", "vt")
STATE_DISPLAY = {
    "dt": "Dorsal turn",
    "fwd": "Forward",
    "nostate": "No state",
    "rev1": "Reverse 1",
    "rev2": "Reverse 2",
    "revsus": "Sustained reverse",
    "slow": "Forward slowing",
    "vt": "Ventral turn",
}
MACRO_ORDER: tuple[str, ...] = ("FWD", "REV", "TURN", "NOSTATE")
MACRO_DISPLAY = {
    "FWD": "Forward / slowing",
    "REV": "Reverse",
    "TURN": "Post-reversal turn",
    "NOSTATE": "Unlabelled / ambiguous",
}
STATE_TO_MACRO = {
    "fwd": "FWD",
    "slow": "FWD",
    "rev1": "REV",
    "rev2": "REV",
    "revsus": "REV",
    "dt": "TURN",
    "vt": "TURN",
    "nostate": "NOSTATE",
}

# Neurons that Kato et al. used to construct the behavioural labels.
BEHAVIOR_LABEL_NEURONS: tuple[str, ...] = (
    "AVAL",
    "AVAR",
    "SMDDR",
    "SMDDL",
    "SMDVR",
    "SMDVL",
    "RIBR",
    "RIBL",
)


def _deref(fh: h5py.File, value: Any) -> Any:
    if isinstance(value, h5py.Reference) and value:
        return fh[value]
    return value


def _decode_name(fh: h5py.File, value: Any, _depth: int = 0) -> str:
    """Decode MATLAB char arrays (possibly cell-nested) to a neuron name."""
    if _depth > 5:
        raise ValueError("Neuron name nesting exceeds five levels")
    obj = _deref(fh, value)
    arr = np.asarray(obj)
    if arr.size == 0:
        return ""
    first = arr.ravel()[0]
    if isinstance(first, h5py.Reference):
        return _decode_name(fh, first, _depth + 1)
    if isinstance(first, (bytes, np.bytes_)):
        return first.decode("utf-8", "replace")
    if isinstance(first, str):
        return first
    try:
        return "".join(chr(int(v)) for v in arr.ravel() if int(v) > 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot decode neuron name from {type(first)}") from exc


def _scalar(obj: Any) -> float:
    return float(np.asarray(obj).ravel()[0])


def _dataset_string(fh: h5py.File, value: Any) -> str:
    arr = np.asarray(_deref(fh, value)).ravel()
    return "".join(chr(int(v)) for v in arr if int(v) > 0)


@dataclass(frozen=True)
class WormRecording:
    worm_id: int
    dataset_name: str
    neurons: tuple[str, ...]
    traces: np.ndarray  # (time, neuron) corrected deltaF/F
    raw_traces: np.ndarray
    states: tuple[str, ...]
    fps: float
    time_seconds: np.ndarray
    source_file: str
    source_sha256: str
    state_onehot: np.ndarray | None = None

    @property
    def n_frames(self) -> int:
        return int(self.traces.shape[0])

    @property
    def n_neurons(self) -> int:
        return int(self.traces.shape[1])

    @property
    def identified_mask(self) -> np.ndarray:
        return np.array([any(ch.isalpha() for ch in n) for n in self.neurons], dtype=bool)

    @property
    def behavior_neuron_mask(self) -> np.ndarray:
        named = set(BEHAVIOR_LABEL_NEURONS)
        return np.array([n in named for n in self.neurons], dtype=bool)

    def subset(self, indices: Iterable[int]) -> WormRecording:
        idx = np.asarray(list(indices), dtype=int)
        return WormRecording(
            worm_id=self.worm_id,
            dataset_name=self.dataset_name,
            neurons=tuple(self.neurons[i] for i in idx),
            traces=self.traces[:, idx],
            raw_traces=self.raw_traces[:, idx],
            states=self.states,
            fps=self.fps,
            time_seconds=self.time_seconds,
            source_file=self.source_file,
            source_sha256=self.source_sha256,
            state_onehot=self.state_onehot,
        )

    def neuron_indices(self, names: Iterable[str]) -> list[int]:
        wanted = list(names)
        idx = []
        for name in wanted:
            if name not in self.neurons:
                raise KeyError(f"Neuron {name!r} is not present in worm {self.worm_id}")
            idx.append(self.neurons.index(name))
        return idx

    def macro_labels(self) -> np.ndarray:
        return np.array([STATE_TO_MACRO[s] for s in self.states], dtype="<U7")


@dataclass
class KatoDataset:
    recordings: list[WormRecording]
    source_file: Path
    source_sha256: str
    readme: str = ""

    def __iter__(self):
        return iter(self.recordings)

    def __len__(self) -> int:
        return len(self.recordings)

    def __getitem__(self, item: int) -> WormRecording:
        return self.recordings[item]


def load_kato(path: str | Path | None = None, *, verify: bool = True) -> KatoDataset:
    """Load all five WT_NoStim recordings without modifying the source file."""
    rel = path or "data/raw/kato2015/WT_NoStim.mat"
    p = resolve_path(rel)
    if not p.exists():
        raise FileNotFoundError(
            f"Kato dataset not found at {p}. Run `neurobranch data download` or place WT_NoStim.mat there."
        )
    digest = sha256_file(p) if verify else "unverified"
    readme_path = p.parent / "readme_Kato2015.txt"
    readme = readme_path.read_text(encoding="utf-8", errors="replace") if readme_path.exists() else ""

    recordings: list[WormRecording] = []
    with h5py.File(p, "r") as fh:
        group = fh["WT_NoStim"]
        n = int(group["fps"].shape[0])
        for i in range(n):
            names_ds = _deref(fh, group["NeuronNames"][i, 0])
            names = tuple(_decode_name(fh, x) for x in np.asarray(names_ds).ravel())
            traces = np.asarray(_deref(fh, group["deltaFOverF_bc"][i, 0]), dtype=np.float32)
            raw = np.asarray(_deref(fh, group["deltaFOverF"][i, 0]), dtype=np.float32)
            if traces.shape[0] == len(names) and traces.shape[1] != len(names):
                traces = traces.T
                raw = raw.T
            if traces.shape[1] != len(names):
                raise ValueError(f"Worm {i}: {traces.shape=} inconsistent with {len(names)} names")
            state_group = _deref(fh, group["States"][i, 0])
            onehot = np.zeros((traces.shape[0], len(STATE_ORDER)), dtype=np.uint8)
            for j, state in enumerate(STATE_ORDER):
                if state in state_group:
                    onehot[:, j] = np.asarray(state_group[state]).ravel().astype(np.uint8)
            row_sums = onehot.sum(axis=1)
            if not np.all(row_sums == 1):
                bad = np.flatnonzero(row_sums != 1)
                raise ValueError(f"Worm {i}: states are not exactly one-hot at {bad[:5]}")
            states = tuple(STATE_ORDER[j] for j in onehot.argmax(axis=1))
            fps = _scalar(_deref(fh, group["fps"][i, 0]))
            tv = np.asarray(_deref(fh, group["tv"][i, 0]), dtype=np.float64).ravel()
            dataset_name = _dataset_string(fh, group["dataset"][i, 0])
            recordings.append(
                WormRecording(
                    worm_id=i,
                    dataset_name=dataset_name,
                    neurons=names,
                    traces=traces,
                    raw_traces=raw,
                    states=states,
                    fps=fps,
                    time_seconds=tv,
                    source_file=str(p),
                    source_sha256=digest,
                    state_onehot=onehot,
                )
            )
    return KatoDataset(recordings=recordings, source_file=p, source_sha256=digest, readme=readme)


def predictor_indices(rec: WormRecording, *, exclude_behavior: bool = True) -> np.ndarray:
    """Indices of neurons permitted as model inputs.

    Numeric labels are retained: they are unidentified cells but still carry
    genuine calcium signal.  They are never treated as named cells in
    intervention experiments.
    """
    keep = np.ones(rec.n_neurons, dtype=bool)
    if exclude_behavior:
        keep &= ~rec.behavior_neuron_mask
    return np.flatnonzero(keep)


def recording_summary(rec: WormRecording) -> dict[str, Any]:
    identified = rec.identified_mask
    macro = rec.macro_labels()
    vals, counts = np.unique(macro, return_counts=True)
    return {
        "worm_id": rec.worm_id,
        "dataset_name": rec.dataset_name,
        "n_frames": rec.n_frames,
        "n_neurons_total": rec.n_neurons,
        "n_identified": int(identified.sum()),
        "n_unidentified_numeric": int((~identified).sum()),
        "n_behavior_label_neurons_present": int(rec.behavior_neuron_mask.sum()),
        "duration_seconds": float(rec.n_frames / rec.fps),
        "fps": float(rec.fps),
        "start_time": float(rec.time_seconds[0]),
        "end_time": float(rec.time_seconds[-1]),
        "missing_named_label_neurons": sorted(set(BEHAVIOR_LABEL_NEURONS) - set(rec.neurons)),
        "macro_counts": {str(v): int(c) for v, c in zip(vals, counts)},
        "state_counts": {s: int(rec.states.count(s)) for s in STATE_ORDER},
        "nan_fraction": float(np.isnan(rec.traces).mean()),
        "source_sha256": rec.source_sha256,
    }


def neuron_table(dataset: KatoDataset) -> pd.DataFrame:
    rows = []
    for rec in dataset:
        for idx, name in enumerate(rec.neurons):
            rows.append(
                {
                    "worm_id": rec.worm_id,
                    "neuron_index": idx,
                    "neuron_name": name,
                    "identified": any(ch.isalpha() for ch in name),
                    "behavior_label_neuron": name in BEHAVIOR_LABEL_NEURONS,
                    "trace_mean": float(np.nanmean(rec.traces[:, idx])),
                    "trace_std": float(np.nanstd(rec.traces[:, idx])),
                }
            )
    return pd.DataFrame(rows)


def transition_table(dataset: KatoDataset, *, macro: bool = True) -> pd.DataFrame:
    rows = []
    for rec in dataset:
        labels = rec.macro_labels() if macro else np.asarray(rec.states)
        for t in range(len(labels) - 1):
            if labels[t] != labels[t + 1]:
                rows.append(
                    {
                        "worm_id": rec.worm_id,
                        "frame": t,
                        "time_seconds": float(rec.time_seconds[t]),
                        "from_state": labels[t],
                        "to_state": labels[t + 1],
                    }
                )
    return pd.DataFrame(rows)


def write_audit(dataset: KatoDataset, out_dir: str | Path = "reports") -> dict[str, Path]:
    """Write machine-readable audit tables plus a compact JSON manifest."""
    out = resolve_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries = [recording_summary(r) for r in dataset]
    pd.DataFrame(summaries).to_csv(out / "recording_summary.csv", index=False)
    neurons = neuron_table(dataset)
    neurons.to_csv(out / "neuron_identity_table.csv", index=False)
    transitions = transition_table(dataset, macro=True)
    transitions.to_csv(out / "macro_transitions.csv", index=False)
    state_transitions = transition_table(dataset, macro=False)
    state_transitions.to_csv(out / "fine_state_transitions.csv", index=False)
    payload = {
        "dataset": "Kato et al. 2015 WT_NoStim, OSF project 2395t",
        "source_file": str(dataset.source_file),
        "source_sha256": dataset.source_sha256,
        "n_worms": len(dataset),
        "n_frames_total": int(sum(r.n_frames for r in dataset)),
        "n_named_cells_total": int(sum(r.identified_mask.sum() for r in dataset)),
        "state_order": list(STATE_ORDER),
        "macro_order": list(MACRO_ORDER),
        "behavior_label_neurons": list(BEHAVIOR_LABEL_NEURONS),
        "recordings": summaries,
        "provenance": {
            "osf_project": "https://osf.io/2395t/",
            "download_url": "https://osf.io/download/na4f9/",
            "paper_doi": "10.1016/j.cell.2015.09.034",
        },
        "caveats": [
            "Behaviour labels are inferred from calcium signals in AVA/SMD/RIB and are not independent observations.",
            "The 109-135 segmented traces per worm are not the full 302-neuron connectome.",
            "No intervention was performed in WT_NoStim.",
            "Unidentified numeric traces are not assigned cell identities.",
        ],
    }
    with open(out / "dataset_audit.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return {
        "summary_csv": out / "recording_summary.csv",
        "neuron_csv": out / "neuron_identity_table.csv",
        "transitions_csv": out / "macro_transitions.csv",
        "manifest": out / "dataset_audit.json",
    }


def default_raw_dir() -> Path:
    return PROJECT_ROOT / "data" / "raw" / "kato2015"
