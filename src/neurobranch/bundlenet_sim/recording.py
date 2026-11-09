"""Load and prepare the published *C. elegans* premotor recordings.

The Morrison & Young release distributes JSON recordings in
``references/external/Celegans_premotor/data/Atanas_data``.  Each file stores a
``trace_array`` (one row per imaging trace), a ``labeled`` mapping from trace
index to NeuroPAL label, and the mean frame interval.

This module deliberately reproduces the preprocessing used by the published
Julia notebook:

* Savitzky-Golay smoothing with window 11 and polynomial order 5;
* the two edge frames are discarded on each side;
* missing neurons are replaced only by the explicit twin/partner rules in
  ``functions_regression.jl``.  No value is invented for a neuron without a
  documented replacement.

The output is a :class:`ClampedRecording` whose rows follow the model's
205-neuron order, restricted to the 84 neurons with measured signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.signal import savgol_filter

INTRINSIC_NEURONS: tuple[str, ...] = (
    "AVBL",
    "AVBR",
    "RIBL",
    "RIBR",
    "RID",
    "AVDL",
    "AVDR",
    "AVAL",
    "AVAR",
    "RIML",
    "RIMR",
    "AVEL",
    "AVER",
    "AIBL",
    "AIBR",
)

_FORWARD = ("AVAL", "AVAR", "AVEL", "AVER")
_BACKWARD = ("AVBL", "AVBR", "RIBL", "RIBR")


class ClampedDataError(ValueError):
    """Raised when a recording cannot be mapped to the published core."""


@dataclass(frozen=True)
class AtanasRecording:
    recording_id: str
    dt: float
    traces: np.ndarray
    labels: tuple[str | None, ...]
    source: str

    @property
    def n_traces(self) -> int:
        return int(self.traces.shape[0])

    @property
    def n_frames(self) -> int:
        return int(self.traces.shape[1])


@dataclass(frozen=True)
class ClampedRecording:
    """A recording cropped to the published 84-neuron core.

    ``traces`` has shape ``(n_core, n_frames)`` and rows follow
    ``model.neuron_names``.  ``behavior`` is derived from the published
    forward/backward readout and is in ``{-1, 0, 1}``.
    """

    recording_id: str
    dt: float
    time: np.ndarray
    traces: np.ndarray
    core_names: tuple[str, ...]
    intrinsic_names: tuple[str, ...]
    signal_names: tuple[str, ...]
    behavior: np.ndarray
    replacement_map: dict[str, str]
    source: str

    @property
    def n_core(self) -> int:
        return len(self.core_names)

    @property
    def n_intrinsic(self) -> int:
        return len(self.intrinsic_names)

    @property
    def n_signal(self) -> int:
        return len(self.signal_names)

    @property
    def n_frames(self) -> int:
        return int(self.traces.shape[1])

    def rows(self, names: Iterable[str]) -> np.ndarray:
        index = {name: i for i, name in enumerate(self.core_names)}
        try:
            return np.array([index[name] for name in names], dtype=np.int64)
        except KeyError as exc:  # pragma: no cover - defensive
            raise ClampedDataError(f"neuron not in core: {exc.args[0]}") from exc


def load_atanas_recording(path: Path | str) -> AtanasRecording:
    """Read one JSON recording without imputing missing labels."""
    import json

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    traces = np.asarray(payload["trace_array"], dtype=np.float64)
    if traces.ndim != 2:
        raise ClampedDataError(f"{path.name}: trace_array must be 2-D, got {traces.shape}")
    labels_by_index: dict[int, str | None] = {}
    for key, meta in payload.get("labeled", {}).items():
        label = meta.get("label") if isinstance(meta, dict) else None
        labels_by_index[int(key)] = str(label) if label else None
    labels = tuple(labels_by_index.get(i) for i in range(traces.shape[0]))
    return AtanasRecording(
        recording_id=path.stem,
        dt=float(payload["avg_timestep"]),
        traces=traces,
        labels=labels,
        source=str(path),
    )


def smooth_traces(traces: np.ndarray, *, window: int = 11, polyorder: int = 5) -> np.ndarray:
    """Apply the published Savitzky-Golay preprocessing.

    The Julia notebook smooths each trace, then removes two frames on each side
    (``ts_tmp[2:end-1]`` in 1-based Julia).  The result is the same length as the
    input minus four frames.
    """
    if traces.ndim != 2:
        raise ClampedDataError(f"traces must be 2-D, got {traces.shape}")
    if traces.shape[1] <= window:
        raise ClampedDataError(
            f"recording has {traces.shape[1]} frames; need more than {window} for smoothing"
        )
    smoothed = np.stack(
        [savgol_filter(row, window_length=window, polyorder=polyorder) for row in traces],
        axis=0,
    )
    return smoothed[:, 2:-2]


def paper_replacement(name: str, available: set[str]) -> str | None:
    """Return the documented donor for a missing neuron, or ``None``.

    These are the explicit rules in the release's ``functions_regression.jl``.
    We intentionally do not apply a generic nearest-neighbour or contralateral
    fill: that would add neurons the published core does not contain and make
    the simulation look more complete than it is.
    """
    if name in {"AVBR", "RIBR", "AVBL", "RIBL"}:
        partners = {
            "AVBR": ("RIBR", "AVBL"),
            "RIBR": ("AVBR", "AVBL"),
            "AVBL": ("RIBL", "AVBR"),
            "RIBL": ("AVBL", "AVBR"),
        }[name]
        for candidate in partners:
            if candidate in available:
                return candidate
    if name in {"AIBL", "AIBR"}:
        partners = ("RIML", "RIMR") if name == "AIBL" else ("RIMR", "RIML")
        for candidate in partners:
            if candidate in available:
                return candidate
    if name == "RID":
        for candidate in ("AVBR", "AVBL"):
            if candidate in available:
                return candidate
    # The release's length-4/5 twin rule is applied to the ``switchout_list``
    # of highly correlated neurons.  We only apply it to the 15 intrinsic
    # neurons, where the twin substitution is explicit and auditable; applying
    # it to every missing neuron would add dozens of non-core rows.
    if name in INTRINSIC_NEURONS and len(name) in {4, 5} and name[-1] in {"L", "R"}:
        twin = name[:-1] + ("R" if name[-1] == "L" else "L")
        if twin in available:
            return twin
    return None


def behavior_readout(traces: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    """Published forward/backward readout with a 0.5 threshold.

    ``traces`` must be in the same row order as ``names``.  The function is
    deliberately explicit about missing readout neurons; silently averaging a
    different set would change the behavioural target.
    """
    index = {name: i for i, name in enumerate(names)}
    missing = [name for name in (*_FORWARD, *_BACKWARD) if name not in index]
    if missing:
        raise ClampedDataError(f"behaviour readout missing neurons: {missing}")
    forward = traces[[index[name] for name in _FORWARD]].mean(axis=0)
    backward = traces[[index[name] for name in _BACKWARD]].mean(axis=0)
    difference = forward - backward
    return np.where(difference > 0.5, 1, np.where(difference < -0.5, -1, 0)).astype(np.int64)


def build_clamped_recording(
    model,
    atanas: AtanasRecording,
    *,
    t_start: int = 5,
    n_steps: int = 1400,
    expected_core: int | None = 84,
    expected_intrinsic: int = 15,
) -> ClampedRecording:
    """Map one recording onto the published core.

    ``expected_core=None`` accepts a recording whose measured core is not the
    canonical 84-neuron set (the release has one canonical recording and many
    shorter/partial ones).  All 15 intrinsic neurons must still be present.
    """
    if model.neuron_names is None:
        raise ClampedDataError("premotor model has no neuron-name order")
    names = tuple(model.neuron_names)
    if len(names) != model.n_nodes:
        raise ClampedDataError("model neuron names do not match matrix size")

    by_label: dict[str, int] = {}
    for row, label in enumerate(atanas.labels):
        if label is not None:
            by_label[label] = row
    available = set(by_label)

    traces = np.zeros((model.n_nodes, atanas.n_frames), dtype=np.float64)
    replacement_map: dict[str, str] = {}
    for i, name in enumerate(names):
        if name in by_label:
            traces[i] = atanas.traces[by_label[name]]
            continue
        donor = paper_replacement(name, available)
        if donor is not None:
            traces[i] = atanas.traces[by_label[donor]]
            replacement_map[name] = donor

    core = [i for i, name in enumerate(names) if np.any(traces[i] != 0.0)]
    core_names = tuple(names[i] for i in core)
    if expected_core is not None and len(core_names) != expected_core:
        missing = [name for name in names if name not in core_names]
        raise ClampedDataError(
            f"{atanas.recording_id}: expected {expected_core} core neurons, got {len(core_names)}; "
            f"missing={missing[:12]}"
        )

    smoothed = smooth_traces(traces[core])
    if smoothed.shape[1] < t_start + n_steps + 1:
        raise ClampedDataError(
            f"{atanas.recording_id}: only {smoothed.shape[1]} usable frames; "
            f"need {t_start + n_steps + 1}"
        )
    cropped = smoothed[:, t_start : t_start + n_steps + 1]

    missing_intrinsic = [name for name in INTRINSIC_NEURONS if name not in core_names]
    if len(missing_intrinsic) != 0:
        raise ClampedDataError(
            f"{atanas.recording_id}: intrinsic neurons missing after replacement: {missing_intrinsic}"
        )
    signal_names = tuple(name for name in core_names if name not in INTRINSIC_NEURONS)
    if len(INTRINSIC_NEURONS) != expected_intrinsic:
        raise AssertionError("INTRINSIC_NEURONS changed without updating validation")

    return ClampedRecording(
        recording_id=atanas.recording_id,
        dt=atanas.dt,
        time=atanas.dt * np.arange(cropped.shape[1], dtype=np.float64),
        traces=cropped,
        core_names=core_names,
        intrinsic_names=INTRINSIC_NEURONS,
        signal_names=signal_names,
        behavior=behavior_readout(cropped, core_names),
        replacement_map=replacement_map,
        source=atanas.source,
    )


def list_atanas_recordings(directory: Path | str) -> list[Path]:
    return sorted(Path(directory).glob("*.json"))
