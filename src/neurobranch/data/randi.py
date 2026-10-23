"""Adapter for the Randi et al. (2023) optogenetic propagation atlas.

The public OSF archive (project ``e2syt``) distributes *processed text exports*
from ``pumpprobe``, not raw microscope movies.  Each dataset contains:

* ``<id>_gcamp.txt``: frames x traces, processed GCaMP signal;
* ``<id>_labels.txt``: one trace label per column (blank/numeric/ambiguous
  entries are preserved, never silently imputed);
* ``<id>_t.txt``: frame times in seconds;
* ``<id>_stim_neurons.txt``: trace index of the stimulated neuron per trial,
  with negative sentinels for discarded/ambiguous trials;
* ``<id>_stim_volume_i.txt``: frame index of each stimulation;
* ``<id>_ds_name.txt``: original acquisition path.

The adapter computes a simple, auditable event-triggered response: the signed
peak deviation from a pre-stimulus baseline in a fixed post-stimulus window.
This is a benchmark target, not a claim that the preprocessing reproduces the
authors' response detector.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_INVALID_LABELS = {
    "",
    "smthng else",
    "unknown",
    "none",
    "nan",
    "merge",
    "rig",
    "check",
    "mah",
    "target",
    # Unnumbered ventral-cord/AS classes are ambiguous without an index.
    "da",
    "db",
    "dd",
    "va",
    "vb",
    "vc",
    "vd",
    "as",
}


def canonical_neuron_name(raw: str | None) -> str | None:
    """Normalize a trace label without inventing an identity.

    Ventral-cord names are zero-padded (``DB1`` -> ``DB01``).  Blank, numeric,
    placeholder, and ambiguous labels return ``None``.  ``RIG`` is ambiguous
    between RIGL/RIGR and is deliberately not imputed.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or not any(ch.isalpha() for ch in s):
        return None
    if s.lower() in _INVALID_LABELS:
        return None
    m = re.match(r"^(VA|VB|DA|DB|DD|VC|AS)(\d{1,2})$", s)
    if m:
        return f"{m.group(1)}{int(m.group(2)):02d}"
    return s


@dataclass
class RandiRecording:
    dataset_id: int
    source_path: str
    traces: np.ndarray
    time_seconds: np.ndarray
    raw_labels: tuple[str, ...]
    labels: tuple[str | None, ...]
    stim_frames: np.ndarray
    stim_trace_indices: np.ndarray
    stimulated_labels: tuple[str | None, ...]
    ambiguous: tuple[bool, ...] = ()

    @property
    def n_frames(self) -> int:
        return int(self.traces.shape[0])

    @property
    def n_traces(self) -> int:
        return int(self.traces.shape[1])

    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.time_seconds)))


def discover_datasets(data_dir: str | Path) -> list[int]:
    d = Path(data_dir)
    ids = []
    for p in d.glob("*_labels.txt"):
        try:
            ids.append(int(p.name.split("_")[0]))
        except ValueError:
            continue
    return sorted(ids)


def load_recording(data_dir: str | Path, dataset_id: int) -> RandiRecording:
    d = Path(data_dir)
    prefix = d / str(dataset_id)
    missing = [
        suffix
        for suffix in ("_gcamp.txt", "_labels.txt", "_t.txt", "_stim_neurons.txt", "_stim_volume_i.txt")
        if not Path(f"{prefix}{suffix}").exists()
    ]
    if missing:
        raise FileNotFoundError(f"dataset {dataset_id} missing {missing}")
    traces = np.loadtxt(f"{prefix}_gcamp.txt")
    if traces.ndim == 1:
        traces = traces[:, None]
    t = np.loadtxt(f"{prefix}_t.txt").ravel()
    raw_labels = tuple(
        line.rstrip("\n")
        for line in Path(f"{prefix}_labels.txt").read_text(encoding="utf-8").splitlines()
    )
    labels = tuple(canonical_neuron_name(x) for x in raw_labels)
    stim = np.loadtxt(f"{prefix}_stim_neurons.txt", dtype=int).ravel()
    frames = np.loadtxt(f"{prefix}_stim_volume_i.txt", dtype=int).ravel()
    if len(stim) != len(frames):
        raise ValueError(f"dataset {dataset_id}: {len(stim)} stim labels but {len(frames)} volumes")
    counts: dict[str, int] = {}
    for name in labels:
        if name is not None:
            counts[name] = counts.get(name, 0) + 1
    ambiguous = tuple(name is not None and counts.get(name, 0) > 1 for name in labels)
    stimulated = tuple(
        labels[i]
        if 0 <= i < len(labels) and i >= 0 and not ambiguous[i]
        else None
        for i in stim
    )
    source = ""
    name_path = Path(f"{prefix}_ds_name.txt")
    if name_path.exists():
        source = name_path.read_text(encoding="utf-8").strip()
    if traces.shape[0] != len(t):
        raise ValueError(f"dataset {dataset_id}: {traces.shape[0]} frames but {len(t)} times")
    if len(raw_labels) > traces.shape[1]:
        # A few exports contain trailing blank lines after the final trace.
        extra = raw_labels[traces.shape[1] :]
        if any(x.strip() for x in extra):
            raise ValueError(
                f"dataset {dataset_id}: {len(raw_labels)} labels for {traces.shape[1]} traces and extras are non-blank"
            )
        raw_labels = raw_labels[: traces.shape[1]]
        labels = labels[: traces.shape[1]]
    if traces.shape[1] != len(labels):
        raise ValueError(f"dataset {dataset_id}: {traces.shape[1]} traces but {len(labels)} labels")
    # Missing samples are preserved.  The trial extractor skips traces whose
    # baseline or post window is not finite rather than imputing a response.
    return RandiRecording(
        dataset_id=dataset_id,
        source_path=source,
        traces=traces.astype(np.float64),
        time_seconds=t.astype(np.float64),
        raw_labels=raw_labels,
        labels=labels,
        stim_frames=frames,
        stim_trace_indices=stim,
        stimulated_labels=stimulated,
        ambiguous=ambiguous,
    )


@dataclass(frozen=True)
class ResponseConfig:
    pre_frames: int = 8
    post_frames: int = 12
    post_start: int = 1
    noise_floor: float = 1e-6


def extract_trials(
    rec: RandiRecording,
    *,
    config: ResponseConfig | None = None,
) -> pd.DataFrame:
    """Return long-form downstream responses for every valid stimulation.

    The stimulated neuron's own trace is excluded from response targets: its
    signal is dominated by the direct optogenetic/fluorescence effect and is
    not a downstream propagation measurement.
    """
    cfg = config or ResponseConfig()
    rows: list[dict[str, Any]] = []
    # Duplicate canonical labels are retained in the recording but excluded
    # from the benchmark: the export does not say which duplicate is which cell.
    named = [
        i
        for i, n in enumerate(rec.labels)
        if n is not None and not (i < len(rec.ambiguous) and rec.ambiguous[i])
    ]
    if not named:
        return pd.DataFrame()
    for event_index, (i0, target_idx, target) in enumerate(
        zip(rec.stim_frames, rec.stim_trace_indices, rec.stimulated_labels)
    ):
        if target is None or i0 < cfg.pre_frames or i0 + cfg.post_start + cfg.post_frames >= rec.n_frames:
            continue
        if target_idx < 0:
            continue
        pre = rec.traces[i0 - cfg.pre_frames : i0]
        post = rec.traces[i0 + cfg.post_start : i0 + cfg.post_start + cfg.post_frames]
        if pre.size == 0 or post.size == 0:
            continue
        if not np.isfinite(pre).all() or not np.isfinite(post).all():
            # Keep the trial only if a finite subset of traces is large enough
            # to support a response estimate; no values are imputed.
            finite_traces = np.isfinite(pre).all(axis=0) & np.isfinite(post).all(axis=0)
            if finite_traces.sum() < 2:
                continue
        else:
            finite_traces = np.ones(pre.shape[1], dtype=bool)
        baseline = np.full(pre.shape[1], np.nan)
        baseline[finite_traces] = np.median(pre[:, finite_traces], axis=0)
        delta = post - baseline
        # Signed value at the maximum absolute excursion.
        peak_idx = np.argmax(np.abs(delta), axis=0)
        peak = delta[peak_idx, np.arange(delta.shape[1])]
        noise = np.full(pre.shape[1], np.nan)
        noise[finite_traces] = (
            1.4826 * np.median(np.abs(pre[:, finite_traces] - baseline[finite_traces]), axis=0)
            + cfg.noise_floor
        )
        z = np.abs(peak) / noise
        global_idx = [i for i in named if i != target_idx and finite_traces[i] and np.isfinite(peak[i])]
        if not global_idx:
            continue
        global_delta = float(np.median(peak[global_idx]))
        for j in named:
            if j == target_idx or not finite_traces[j] or not np.isfinite(peak[j]):
                continue
            rows.append(
                {
                    "dataset_id": int(rec.dataset_id),
                    "event_index": int(event_index),
                    "event_id": f"{rec.dataset_id}:{event_index}",
                    "target": target,
                    "target_trace_index": int(target_idx),
                    "neuron": rec.labels[j],
                    "neuron_trace_index": int(j),
                    "stim_frame": int(i0),
                    "time_seconds": float(rec.time_seconds[i0]),
                    "response": float(peak[j]),
                    "response_z": float(z[j]),
                    "response_sign": int(np.sign(peak[j])),
                    "pre_activity": float(baseline[j]),
                    "global_delta": global_delta,
                    "response_residual": float(peak[j] - global_delta),
                }
            )
    return pd.DataFrame(rows)


def build_trial_table(
    data_dir: str | Path,
    *,
    dataset_ids: list[int] | None = None,
    config: ResponseConfig | None = None,
) -> pd.DataFrame:
    """Load all (or selected) datasets and return one long-form table."""
    ids = dataset_ids if dataset_ids is not None else discover_datasets(data_dir)
    frames = []
    for ds in ids:
        rec = load_recording(data_dir, ds)
        table = extract_trials(rec, config=config)
        if len(table):
            frames.append(table)
    if not frames:
        raise ValueError("no valid Randi trials extracted")
    return pd.concat(frames, ignore_index=True)


def coverage_summary(table: pd.DataFrame, data_dir: str | Path) -> dict[str, Any]:
    """Dataset/target/neuron coverage, with missing labels counted explicitly."""
    ids = discover_datasets(data_dir)
    return {
        "n_source_datasets": len(ids),
        "dataset_ids": ids,
        "n_events": int(table["event_id"].nunique()),
        "n_rows": int(len(table)),
        "n_animals": int(table["dataset_id"].nunique()),
        "n_stimulated_neurons": int(table["target"].nunique()),
        "n_recorded_neurons": int(table["neuron"].nunique()),
        "n_target_neuron_pairs": int(table[["target", "neuron"]].drop_duplicates().shape[0]),
        "events_per_animal": {
            str(k): int(v) for k, v in table.groupby("dataset_id")["event_id"].nunique().items()
        },
        "targets": sorted(table["target"].unique().tolist()),
        "recorded_neurons": sorted(table["neuron"].unique().tolist()),
        "response_window": {"pre_frames": 8, "post_frames": 12, "post_start": 1},
        "unit": "processed GCaMP signal, baseline-subtracted peak",
    }


def load_connectome(path: str | Path) -> pd.DataFrame:
    """Load a Witvliet-style edge table with canonicalized neuron names."""
    c = pd.read_csv(path, sep="\t")
    for col in ("pre", "post"):
        c[col] = c[col].map(canonical_neuron_name)
    c = c.dropna(subset=["pre", "post"])
    return c


def connectome_feature_table(connectome: pd.DataFrame) -> pd.DataFrame:
    """Directed and electrical connection counts for each target->neuron pair."""
    chem = (
        connectome[connectome["type"] == "chemical"]
        .groupby(["pre", "post"])["synapses"]
        .sum()
        .rename("chemical_synapses")
    )
    elec = (
        connectome[connectome["type"] == "electrical"]
        .groupby(["pre", "post"])["synapses"]
        .sum()
        .rename("electrical_synapses")
    )
    out = pd.concat([chem, elec], axis=1).reset_index().rename(columns={"pre": "target", "post": "neuron"})
    out[["chemical_synapses", "electrical_synapses"]] = (
        out[["chemical_synapses", "electrical_synapses"]].fillna(0).astype(float)
    )
    return out
