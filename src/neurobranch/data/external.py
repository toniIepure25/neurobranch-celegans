"""Metadata and analysis adapters for external C. elegans recordings.

Only two public sources are handled here:

* Kato et al. (2015) ``WT_Stim.mat`` from OSF project ``2395t``.
* Nichols et al. (2017) ``npr1_prelet.mat``, ``npr1_let.mat``,
  ``n2_let.mat`` and ``n2_prelet.mat`` from OSF project ``kbf38``.

The loaders preserve trace-level labels exactly as stored.  A trace can carry
zero, one or several neuron labels; labels are never silently deduplicated or
mapped onto another animal.  Oxygen is stored as an environmental stimulus and
is never treated as targeted neuronal stimulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from neurobranch.config import SplitConfig, sha256_file
from neurobranch.data.kato import BEHAVIOR_LABEL_NEURONS, WormRecording
from neurobranch.preprocessing import WormData, make_worm_data

KATO_STIM_GROUP = "WT_Stim"
NICHOLS_GROUP_MARKERS = ("FiveStates", "IDs", "traces")

KATO_STIM_STATE_NAMES = ("fwd", "rev", "revsus", "turn")
KATO_STIM_STATE_MAP = {
    1: "fwd",
    2: "rev",
    3: "revsus",
    4: "turn",
}


def _resolve(fh: h5py.File, value: Any, depth: int = 0) -> Any:
    """Follow MATLAB v7.3 object references until a dataset or group remains."""
    if isinstance(value, h5py.Reference):
        value = fh[value]
    while isinstance(value, h5py.Dataset) and value.dtype == object and depth < 12:
        arr = value[()]
        if arr.size == 0:
            return value
        value = fh[arr.flat[0]]
        depth += 1
    return value


def _char_text(fh: h5py.File, value: Any) -> str:
    """Decode a MATLAB char array, including nested cell wrappers."""
    obj = _resolve(fh, value)
    if isinstance(obj, h5py.Dataset):
        arr = obj[()]
        if arr.dtype == object:
            return "".join(_char_text(fh, ref) for ref in arr.ravel())
        if arr.dtype.kind in "uif":
            return "".join(chr(int(v)) for v in arr.ravel() if int(v) > 0)
        return str(arr)
    return ""


def _label_tuple(fh: h5py.File, value: Any) -> tuple[str, ...]:
    """Decode one trace's label entry without collapsing multi-neuron labels."""
    obj = fh[value] if isinstance(value, h5py.Reference) else value
    if not isinstance(obj, h5py.Dataset):
        return ()
    arr = obj[()]
    if arr.dtype != object:
        text = _char_text(fh, obj)
        return (text,) if text else ()
    labels: list[str] = []
    for ref in arr.ravel():
        labels.extend(_label_tuple(fh, ref))
    return tuple(labels)


def _text_list(fh: h5py.File, value: Any) -> tuple[str, ...]:
    """Decode every element of a MATLAB cell-of-strings (or a single string)."""
    obj = fh[value] if isinstance(value, h5py.Reference) else value
    if not isinstance(obj, h5py.Dataset):
        return ()
    arr = obj[()]
    if arr.dtype != object:
        text = _char_text(fh, obj)
        return (text,) if text else ()
    return tuple(text for ref in arr.ravel() if (text := _char_text(fh, ref)))


def _labels(fh: h5py.File, field: h5py.Dataset, index: int) -> tuple[tuple[str, ...], ...]:
    ref = field[index, 0] if field.ndim > 1 else field[index]
    obj = fh[ref]
    if not isinstance(obj, h5py.Dataset):
        return ()
    arr = obj[()]
    if arr.dtype != object:
        text = _char_text(fh, obj)
        return ((text,),) if text else ((),)
    return tuple(_label_tuple(fh, ref) for ref in arr.ravel())


def _scalar(fh: h5py.File, value: Any) -> float:
    return float(np.asarray(_resolve(fh, value)).ravel()[0])


def _vector(fh: h5py.File, value: Any) -> np.ndarray:
    return np.asarray(_resolve(fh, value), dtype=np.float64).ravel()


def piecewise_oxygen(
    time_seconds: np.ndarray,
    switchtimes: np.ndarray,
    concentrations: np.ndarray,
    initialstate: int,
) -> np.ndarray:
    """Piecewise-constant oxygen percentage from the Kato/Nichols metadata.

    ``initialstate`` is MATLAB's 1-based index into ``concentrations``.  Each
    switch toggles between the supplied concentration levels.  This is a
    measured environmental input, not a neuronal perturbation.
    """
    t = np.asarray(time_seconds, dtype=np.float64).ravel()
    levels = np.asarray(concentrations, dtype=np.float64).ravel()
    switches = np.sort(np.asarray(switchtimes, dtype=np.float64).ravel())
    if levels.size < 2:
        raise ValueError("oxygen stimulus needs at least two concentration levels")
    if not 1 <= int(initialstate) <= levels.size:
        raise ValueError("initialstate must be a 1-based index into concentrations")
    state = int(initialstate) - 1
    idx = np.full(t.shape, state, dtype=np.int64)
    for switch in switches:
        after = t >= switch
        idx[after] = 1 - idx[after]
    return levels[idx]


def _stimulus(fh: h5py.File, group: h5py.Group) -> dict[str, Any]:
    if "stimulus" not in group:
        return {}
    stim = _resolve(fh, group["stimulus"])
    out: dict[str, Any] = {}
    for key in stim:
        field = _resolve(fh, stim[key])
        arr = np.asarray(field)
        if key in {"identity", "type", "concunits"}:
            out[key] = _char_text(fh, stim[key])
        else:
            out[key] = arr.ravel().astype(float).tolist()
    return out


@dataclass(frozen=True)
class ExternalRecording:
    """One recording session with trace-level neuron labels preserved."""

    source: str
    session_id: str
    worm_index: int
    dataset_name: str
    labels: tuple[tuple[str, ...], ...]
    traces: np.ndarray
    raw_traces: np.ndarray | None
    fps: float
    time_seconds: np.ndarray
    states: tuple[str, ...]
    state_names: tuple[str, ...]
    stimulus: dict[str, Any]
    source_file: str
    source_sha256: str

    @property
    def n_frames(self) -> int:
        return int(self.traces.shape[0])

    @property
    def n_traces(self) -> int:
        return int(self.traces.shape[1])

    @property
    def duration_seconds(self) -> float:
        if self.time_seconds.size < 2:
            return float(self.n_frames / self.fps)
        return float(self.time_seconds[-1] - self.time_seconds[0])

    @property
    def neurons(self) -> tuple[str, ...]:
        return tuple("|".join(labels) if labels else f"trace_{i}" for i, labels in enumerate(self.labels))

    @property
    def named_mask(self) -> np.ndarray:
        return np.array([bool(set(labels)) for labels in self.labels], dtype=bool)

    @property
    def behavior_mask(self) -> np.ndarray:
        named = set(BEHAVIOR_LABEL_NEURONS)
        return np.array([bool(named.intersection(labels)) for labels in self.labels], dtype=bool)

    @property
    def named_labels(self) -> tuple[str, ...]:
        return tuple(sorted({label for labels in self.labels for label in labels}))

    @property
    def duplicate_label_traces(self) -> int:
        seen: set[str] = set()
        duplicated = 0
        for labels in self.labels:
            if len(labels) > 1 or any(label in seen for label in labels):
                duplicated += 1
            seen.update(labels)
        return duplicated

    @property
    def nan_fraction(self) -> float:
        return float(np.isnan(self.traces).mean())

    @property
    def oxygen_percent(self) -> np.ndarray | None:
        if not self.stimulus or "switchtimes" not in self.stimulus:
            return None
        return piecewise_oxygen(
            self.time_seconds,
            np.asarray(self.stimulus["switchtimes"], dtype=float),
            np.asarray(self.stimulus["conc"], dtype=float),
            int(self.stimulus["initialstate"][0]),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "session_id": self.session_id,
            "worm_index": int(self.worm_index),
            "dataset_name": self.dataset_name,
            "n_frames": self.n_frames,
            "n_traces": self.n_traces,
            "n_labeled_traces": int(self.named_mask.sum()),
            "n_unique_labels": len(self.named_labels),
            "n_behavior_label_traces": int(self.behavior_mask.sum()),
            "n_duplicate_or_multi_label_traces": self.duplicate_label_traces,
            "duration_seconds": self.duration_seconds,
            "fps": float(self.fps),
            "state_names": list(self.state_names),
            "state_counts": {name: int(self.states.count(name)) for name in self.state_names},
            "nan_fraction": self.nan_fraction,
            "stimulus": self.stimulus,
            "source_sha256": self.source_sha256,
        }


def load_kato_stim(path: str | Path, *, verify: bool = True) -> list[ExternalRecording]:
    """Load all seven Kato 2015 WT_Stim recordings."""
    p = Path(path)
    digest = sha256_file(p) if verify else "unverified"
    recordings: list[ExternalRecording] = []
    with h5py.File(p, "r") as fh:
        if KATO_STIM_GROUP not in fh:
            raise ValueError(f"{p} has no {KATO_STIM_GROUP} group")
        group = fh[KATO_STIM_GROUP]
        n = group["fps"].shape[0]
        for i in range(n):
            labels = _labels(fh, group["IDs"], i)
            traces = np.asarray(_resolve(fh, group["traces"][i, 0]), dtype=np.float64)
            raw = np.asarray(_resolve(fh, group["traces_raw"][i, 0]), dtype=np.float64)
            if traces.shape[0] == len(labels) and traces.shape[1] != len(labels):
                traces = traces.T
                raw = raw.T
            if traces.shape[1] != len(labels):
                raise ValueError(
                    f"Kato Stim recording {i}: traces {traces.shape} inconsistent with {len(labels)} labels"
                )
            time = _vector(fh, group["timeVectorSeconds"][i, 0])
            if time.size != traces.shape[0]:
                raise ValueError(f"Kato Stim recording {i}: time/traces length mismatch")
            states_num = _vector(fh, group["States"][i, 0]).astype(int)
            if states_num.size != traces.shape[0]:
                raise ValueError(f"Kato Stim recording {i}: states/traces length mismatch")
            try:
                states = tuple(KATO_STIM_STATE_MAP[int(v)] for v in states_num)
            except KeyError as exc:
                raise ValueError(f"Kato Stim recording {i}: unknown state {exc}") from exc
            stimulus = _stimulus(fh, group)
            recordings.append(
                ExternalRecording(
                    source="kato2015_wt_stim",
                    session_id=f"kato_stim_{i}",
                    worm_index=i,
                    dataset_name=_char_text(fh, group["dataset"][i, 0]),
                    labels=labels,
                    traces=traces,
                    raw_traces=raw,
                    fps=_scalar(fh, group["fps"][i, 0]),
                    time_seconds=time,
                    states=states,
                    state_names=KATO_STIM_STATE_NAMES,
                    stimulus=stimulus,
                    source_file=str(p),
                    source_sha256=digest,
                )
            )
    return recordings


def load_nichols(path: str | Path, *, verify: bool = True) -> list[ExternalRecording]:
    """Load all sessions in one Nichols 2017 OSF file."""
    p = Path(path)
    digest = sha256_file(p) if verify else "unverified"
    recordings: list[ExternalRecording] = []
    with h5py.File(p, "r") as fh:
        candidates = [key for key in fh if not key.startswith("#")]
        if len(candidates) != 1:
            raise ValueError(f"{p}: expected one top-level recording group, got {candidates}")
        group_name = candidates[0]
        group = fh[group_name]
        if not all(marker in group for marker in NICHOLS_GROUP_MARKERS):
            raise ValueError(f"{p}: not a Nichols recording file")
        n = group["fps"].shape[0]
        for i in range(n):
            labels = _labels(fh, group["IDs"], i)
            traces = np.asarray(_resolve(fh, group["traces"][i, 0]), dtype=np.float64)
            if traces.shape[0] == len(labels) and traces.shape[1] != len(labels):
                traces = traces.T
            if traces.shape[1] != len(labels):
                raise ValueError(
                    f"Nichols {p.name} recording {i}: traces {traces.shape} inconsistent with {len(labels)} labels"
                )
            time = _vector(fh, group["timeVectorSeconds"][i, 0])
            if time.size != traces.shape[0]:
                raise ValueError(f"Nichols {p.name} recording {i}: time/traces length mismatch")
            five = _resolve(fh, group["FiveStates"][i, 0])
            names = tuple(name.lower() for name in _text_list(fh, five["fiveStateNames"]) if name)
            states_num = _vector(fh, five["fiveStates"]).astype(int)
            if states_num.size != traces.shape[0]:
                raise ValueError(f"Nichols {p.name} recording {i}: state/traces length mismatch")
            if states_num.min() < 0 or states_num.max() >= len(names):
                raise ValueError(f"Nichols {p.name} recording {i}: state index outside names")
            states = tuple(names[int(v)] for v in states_num)
            recordings.append(
                ExternalRecording(
                    source="nichols2017",
                    session_id=f"{p.stem}_{i}",
                    worm_index=i,
                    dataset_name=_char_text(fh, group["dataset"][i, 0]),
                    labels=labels,
                    traces=traces,
                    raw_traces=None,
                    fps=_scalar(fh, group["fps"][i, 0]),
                    time_seconds=time,
                    states=states,
                    state_names=names,
                    stimulus=_stimulus(fh, group),
                    source_file=str(p),
                    source_sha256=digest,
                )
            )
    return recordings


def load_external_file(path: str | Path, *, verify: bool = True) -> list[ExternalRecording]:
    """Dispatch a local OSF file by its top-level MATLAB group."""
    p = Path(path)
    with h5py.File(p, "r") as fh:
        keys = [key for key in fh if not key.startswith("#")]
        if len(keys) == 1 and "FiveStates" in fh[keys[0]]:
            return load_nichols(p, verify=verify)
        if KATO_STIM_GROUP in keys:
            return load_kato_stim(p, verify=verify)
    raise ValueError(f"{p}: unsupported external recording file")


def build_kato_stim_worm_data(
    rec: ExternalRecording,
    *,
    max_seconds: float = 360.0,
    train_fraction: float = 2.0 / 3.0,
    val_fraction: float = 1.0 / 6.0,
    purge_frames: int = 12,
) -> WormData:
    """Build a leakage-safe WormData view of the constant-oxygen first segment.

    The first 360 s of WT_Stim are 21% oxygen; the oxygen switches start at
    360 s.  This function therefore provides an autonomous-dynamics view, not
    an oxygen-response view.  Behaviour-label traces are excluded exactly as in
    Phase 6.  The returned data keep the animal's own trace order and never
    pool numeric labels with another animal.
    """
    if rec.source != "kato2015_wt_stim":
        raise ValueError("build_kato_stim_worm_data expects a Kato WT_Stim recording")
    if max_seconds > 360.0:
        raise ValueError("max_seconds must stay inside the constant 21% oxygen segment (<= 360 s)")
    n = int(np.searchsorted(rec.time_seconds, float(max_seconds), side="left"))
    if n < 120:
        raise ValueError(f"segment too short: {n} frames")
    keep = ~rec.behavior_mask
    if int(keep.sum()) < 5:
        raise ValueError("fewer than five non-behaviour traces remain")
    traces = rec.traces[:n, keep]
    raw = rec.raw_traces[:n, keep] if rec.raw_traces is not None else traces.copy()
    names = tuple(rec.neurons[j] for j in np.flatnonzero(keep))
    state_map = {"fwd": "fwd", "rev": "rev1", "revsus": "revsus", "turn": "dt"}
    states = tuple(state_map[state] for state in rec.states[:n])
    worm = WormRecording(
        worm_id=rec.worm_index,
        dataset_name=rec.dataset_name,
        neurons=names,
        traces=traces.astype(np.float32),
        raw_traces=raw.astype(np.float32),
        states=states,
        fps=rec.fps,
        time_seconds=rec.time_seconds[:n].copy(),
        source_file=rec.source_file,
        source_sha256=rec.source_sha256,
    )
    split = SplitConfig(
        mode="temporal",
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        purge_frames=int(purge_frames),
    )
    return make_worm_data(worm, split)
