"""Load the published BunDLe-Net/NC-MCM latent trajectories.

Source artifacts
----------------
``references/external/nc-mcm/precomputed_results/bundlenet_consistent_embedding_worm_*.npz``
from the NC-MCM reference release.  Each file contains ``x`` (T, 3) and ``b``
(T, 1).  The coordinates are the "consistent" BunDLe-Net embeddings, i.e. the
per-worm embeddings after the alignment used in the NC-MCM paper.  They are
copied into ``data/processed/bundlenet_sim`` by
``scripts/fetch_vienna_artifacts.py``; the copy is not committed because the
upstream artifacts have separate licenses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Order used by BunDLe-Net's demo (main.py) and by the NC-MCM release.
BEHAVIOR_NAMES: tuple[str, ...] = (
    "Dorsal turn",
    "Forward",
    "No state",
    "Reverse-1",
    "Reverse-2",
    "Sustained reversal",
    "Slowing",
    "Ventral turn",
)

WORM_IDS: tuple[int, ...] = (0, 1, 2, 3, 4)


def available_worm_ids(data_dir: Path | str | None = None) -> tuple[int, ...]:
    """Discover released embedding files, including a future sixth worm.

    The five published worms remain the default evidence base; discovery is
    deliberately pattern-based so a new ``bundlenet_consistent_embedding_worm_5.npz``
    can be dropped in without editing code.  A missing directory returns an
    empty tuple; the loader raises rather than silently returning an empty set.
    """
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    pattern = "bundlenet_consistent_embedding_worm_*.npz"
    found: list[int] = []
    for path in sorted(root.glob(pattern)):
        suffix = path.stem.rsplit("_", 1)[-1]
        if suffix.isdigit():
            found.append(int(suffix))
    return tuple(sorted(set(found)))


def default_data_dir() -> Path:
    """Return the in-repo directory holding the five embedding files."""
    return Path(__file__).resolve().parents[3] / "data" / "processed" / "bundlenet_sim"


@dataclass(frozen=True)
class WormTrajectory:
    """One worm's aligned 3-D BunDLe-Net trajectory and behaviour labels."""

    worm_id: int
    latent: np.ndarray
    behavior: np.ndarray
    source: str

    def __post_init__(self) -> None:
        if self.latent.ndim != 2 or self.latent.shape[1] != 3:
            raise ValueError(f"worm {self.worm_id}: latent must be (T, 3), got {self.latent.shape}")
        if self.behavior.shape != (self.latent.shape[0],):
            raise ValueError(
                f"worm {self.worm_id}: behaviour must be ({self.latent.shape[0]},), got {self.behavior.shape}"
            )
        if not np.isfinite(self.latent).all():
            raise ValueError(f"worm {self.worm_id}: latent contains non-finite values")
        if self.behavior.min() < 0 or self.behavior.max() >= len(BEHAVIOR_NAMES):
            raise ValueError(f"worm {self.worm_id}: behaviour label outside 0..{len(BEHAVIOR_NAMES) - 1}")


def load_trajectories(
    data_dir: Path | str | None = None,
    worm_ids: tuple[int, ...] | list[int] | None = None,
) -> list[WormTrajectory]:
    """Load released worms, sorted by worm id.

    By default every matching embedding in ``data_dir`` is loaded, so a sixth
    held-out worm can be added without code changes.  Pass ``worm_ids`` to
    freeze a subset for a pre-registered comparison.

    Raises
    ------
    FileNotFoundError
        If no embedding files exist, or a requested file is missing.  We prefer
        a loud failure over silently training on a subset.
    """
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    selected = tuple(worm_ids) if worm_ids is not None else available_worm_ids(root)
    if not selected:
        raise FileNotFoundError(
            f"no BunDLe-Net embeddings found in {root}; run scripts/fetch_vienna_artifacts.py"
        )
    trajectories: list[WormTrajectory] = []
    for worm_id in selected:
        path = root / f"bundlenet_consistent_embedding_worm_{worm_id}.npz"
        if not path.exists():
            raise FileNotFoundError(f"missing BunDLe-Net embedding: {path}")
        with np.load(path) as payload:
            latent = np.asarray(payload["x"], dtype=np.float64)
            behavior = np.asarray(payload["b"], dtype=np.int64).reshape(-1)
        trajectories.append(
            WormTrajectory(
                worm_id=worm_id,
                latent=latent,
                behavior=behavior,
                source=str(path),
            )
        )
    return trajectories


def transition_arrays(
    trajectories: list[WormTrajectory],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack one-step transitions from all supplied worms.

    Returns
    -------
    x : (N, 3)
    y : (N, 3)
    worm : (N,)
        Worm id for each transition, useful for grouped diagnostics.
    """
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ws: list[np.ndarray] = []
    for traj in trajectories:
        if len(traj.latent) < 2:
            continue
        xs.append(traj.latent[:-1])
        ys.append(traj.latent[1:])
        ws.append(np.full(len(traj.latent) - 1, traj.worm_id, dtype=np.int64))
    if not xs:
        raise ValueError("no transitions available")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
