"""Read-only access to the projected trajectory artifact.

The artifact stores a fixed training-fitted 10-D PCA projection.  PC1--PC3 are
neural-state coordinates, not anatomical coordinates.  Observed and predicted
arrays share the same projection, origin, and frame-to-second axis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from neurobranch.studio.metadata import ArtifactUnavailable
from neurobranch.studio.paths import StudioPaths


@dataclass(frozen=True)
class TrajectoryStore:
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]

    @property
    def model_order(self) -> tuple[str, ...]:
        return tuple(self.metadata["model_order"])

    @property
    def model_labels(self) -> dict[str, str]:
        return dict(self.metadata["model_labels"])

    def session(self, session: int) -> dict[str, Any]:
        for row in self.metadata["sessions"]:
            if int(row["session"]) == int(session):
                return row
        raise ArtifactUnavailable(f"Session {session} is not in the trajectory artifact")

    def landmark_index(self, session: int, landmark: int) -> int:
        landmarks = self.session(session)["landmarks"]
        try:
            return int(np.flatnonzero(np.asarray(landmarks, dtype=int) == int(landmark))[0])
        except IndexError as exc:
            raise ArtifactUnavailable(
                f"Landmark {landmark} is not a saved H20 landmark for session {session}"
            ) from exc

    def observed(self, session: int) -> np.ndarray:
        return np.asarray(self.arrays[f"observed_s{int(session)}"], dtype=np.float64)

    def time_seconds(self, session: int) -> np.ndarray:
        return np.asarray(self.arrays[f"tv_s{int(session)}"], dtype=np.float64)

    def case(self, session: int, landmark: int, horizon: int = 20) -> dict[str, Any]:
        """Return a matched observed/predicted case.

        The observed continuation and every model forecast begin at
        ``landmark + 1``.  No future observation is supplied to any model input.
        """
        sid = int(session)
        t = int(landmark)
        if horizon not in (1, 5, 10, 20):
            raise ValueError("Studio supports the frozen H1/H5/H10/H20 profiles")
        idx = self.landmark_index(sid, t)
        meta = self.session(sid)
        tv = self.time_seconds(sid)
        observed = self.observed(sid)
        context_start = max(0, t - 180)
        context = observed[context_start : t + 1]
        truth = np.asarray(self.arrays[f"truth_s{sid}"], dtype=np.float64)[idx][:horizon]
        mean_pack = np.asarray(self.arrays[f"means_s{sid}"], dtype=np.float64)[idx][
            :, :horizon
        ]
        means = {
            model: mean_pack[j]
            for j, model in enumerate(self.model_order)
        }
        ensemble_pack = np.asarray(self.arrays[f"ens_s{sid}"], dtype=np.float64)[idx][
            :, :, :horizon
        ]
        ensembles: dict[str, np.ndarray] = {
            "B3_population": ensemble_pack[0],
            "B5_common_k10": ensemble_pack[1],
        }
        for model, key in (
            ("B3_population", f"ens100_s{sid}_{t}_B3"),
            ("B5_common_k10", f"ens100_s{sid}_{t}_B5"),
        ):
            if key in self.arrays:
                ensembles[model] = np.asarray(self.arrays[key], dtype=np.float64)[
                    :, :horizon
                ]
        return {
            "session": sid,
            "landmark": t,
            "horizon": int(horizon),
            "origin_time": float(tv[t]),
            "context_time": tv[context_start : t + 1],
            "future_time": tv[t + 1 : t + horizon + 1],
            "context": context,
            "origin": observed[t],
            "truth": truth,
            "means": means,
            "ensembles": ensembles,
            "explained_variance_ratio": np.asarray(meta["explained_variance_ratio"]),
            "explained_3d": float(meta["explained_variance_3d"]),
            "dataset_name": meta["dataset_name"],
            "model_labels": self.model_labels,
            "fps": float(meta["fps"]),
            "split_bounds": meta["split_bounds"],
            "scores": {
                model: self._score(meta, t, model) for model in self.model_order
            },
        }

    def _score(self, meta: dict[str, Any], landmark: int, model: str) -> float | None:
        for row in meta["scores"]:
            if int(row["landmark"]) == int(landmark):
                value = row.get(model)
                return None if value is None else float(value)
        return None

    def preview_observed(self, session: int, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
        """Downsampled observed path for rendering only.

        This is never used for a score.  The selected landmarks are evenly
        spaced and the origin is added by :meth:`case` at full resolution.
        """
        observed = self.observed(session)
        tv = self.time_seconds(session)
        stride = max(1, int(np.ceil(len(observed) / max(1, int(n)))))
        return observed[::stride], tv[::stride]


@lru_cache(maxsize=2)
def load_trajectory_store(root=None) -> TrajectoryStore:
    paths = StudioPaths() if root is None else StudioPaths.at(Path(root))
    if not paths.trajectory_json.exists() or not paths.trajectory_npz.exists():
        raise ArtifactUnavailable(
            "Trajectory artifact is missing. Run "
            "`python scripts/studio_build_trajectories.py` from the repository root."
        )
    metadata = json.loads(paths.trajectory_json.read_text(encoding="utf-8"))
    with np.load(paths.trajectory_npz, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return TrajectoryStore(metadata=metadata, arrays=arrays)
