"""Observed neural-activity access for the Studio.

Only the Kato 2015 ``WT_NoStim`` recordings are loaded.  The loader is lazy and
cached because the raw MATLAB archive is large; the activity view never changes
the scientific arrays.  Rendering downsampling is explicitly separate from the
full-resolution arrays used by the models.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from neurobranch.diagnostics.common import load_nostim
from neurobranch.studio.metadata import ArtifactUnavailable


@lru_cache(maxsize=1)
def _all_recordings() -> tuple[Any, ...]:
    _, data, _ = load_nostim()
    return tuple(data)


def load_activity(session: int) -> dict[str, Any]:
    data = _all_recordings()
    sid = int(session)
    if sid < 0 or sid >= len(data):
        raise ArtifactUnavailable(f"Recording {sid} is not available")
    d = data[sid]
    names = [str(name) for name in d.predictor_names]
    return {
        "session": sid,
        "dataset_name": d.recording.dataset_name,
        "names": names,
        "named_mask": np.asarray([any(ch.isalpha() for ch in name) for name in names]),
        "z": np.asarray(d.X, dtype=np.float64),
        "raw": np.asarray(d.X_raw, dtype=np.float64),
        "time": np.asarray(d.recording.time_seconds, dtype=np.float64),
        "fps": float(d.recording.fps),
        "split_bounds": {k: [int(v[0]), int(v[1])] for k, v in d.split_bounds.items()},
        "states": tuple(d.recording.states),
        "n_frames": int(d.n_frames),
        "n_traces": int(d.recording.n_neurons),
        "n_predictors": int(d.X.shape[1]),
    }


def rendering_downsample(
    matrix: np.ndarray, time: np.ndarray, *, max_columns: int = 1200
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return a display-only downsample and the stride used.

    The stride is displayed next to the figure.  Scores and model inputs always
    use the full-resolution arrays.
    """
    arr = np.asarray(matrix)
    stride = max(1, int(np.ceil(arr.shape[0] / max(1, int(max_columns)))))
    return arr[::stride], np.asarray(time)[::stride], stride


def neuron_panel(activity: dict[str, Any], selected: list[str], *, raw: bool = False) -> np.ndarray:
    index = {name: i for i, name in enumerate(activity["names"])}
    missing = [name for name in selected if name not in index]
    if missing:
        raise ArtifactUnavailable(f"Neuron labels are not present: {missing}")
    source = activity["raw"] if raw else activity["z"]
    return np.column_stack([source[:, index[name]] for name in selected]) if selected else source[:, :0]
