"""Reader and descriptive analytics for the published NC-MCM artifacts.

The NC-MCM release stores, per worm:
  arr_0  x       (T, neurons)
  arr_1  c       (T, 20)  cluster assignments for K = 1..20
  arr_2  b       (T,)      behaviour labels, 1-based, ordered as arr_3
  arr_3  names   (8,)      NC-MCM behaviour names
  arr_4  p       (20,)     Markov-chain p-values for each K
  arr_5  bprob   (T, 28)   pairwise logistic probabilities (8 choose 2)

We expose the descriptive causal-model objects (clusters, transitions, p-values)
without asserting that a cluster intervention is an identified causal effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import BEHAVIOR_NAMES

# NC-MCM uses a different label order from the BunDLe-Net demo.  Map the
# 1-based NC-MCM labels onto the canonical 0-based BunDLe-Net order.
NCMCM_TO_CANONICAL = {
    "rev2": 4,      # Reverse-2
    "rev1": 3,      # Reverse-1
    "slow": 6,      # Slowing
    "vt": 7,        # Ventral turn
    "nostate": 2,   # No state
    "revsus": 5,    # Sustained reversal
    "fwd": 1,       # Forward
    "dt": 0,        # Dorsal turn
}


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "processed" / "ncmcm"


@dataclass(frozen=True)
class NCMCMWorm:
    worm_id: int
    x: np.ndarray
    clusters: np.ndarray
    behavior: np.ndarray
    behavior_names: tuple[str, ...]
    p_values: np.ndarray
    pairwise_probabilities: np.ndarray

    def cluster_sequence(self, k: int) -> np.ndarray:
        if not 1 <= k <= self.clusters.shape[1]:
            raise ValueError(f"k must be in 1..{self.clusters.shape[1]}")
        return self.clusters[:, k - 1].astype(np.int64)

    def transition_counts(self, k: int) -> np.ndarray:
        seq = self.cluster_sequence(k)
        counts = np.zeros((k, k), dtype=np.int64)
        for a, b in zip(seq[:-1], seq[1:]):
            counts[int(a) - 1, int(b) - 1] += 1
        return counts

    def transition_matrix(self, k: int, smoothing: float = 1e-9) -> np.ndarray:
        counts = self.transition_counts(k).astype(np.float64)
        counts += smoothing
        return counts / counts.sum(axis=1, keepdims=True)

    def markov_p(self, k: int) -> float:
        return float(self.p_values[k - 1])

    def behavior_counts_by_cluster(self, k: int) -> np.ndarray:
        seq = self.cluster_sequence(k)
        table = np.zeros((k, len(BEHAVIOR_NAMES)), dtype=np.int64)
        for cluster, label in zip(seq, self.behavior):
            table[int(cluster) - 1, int(label)] += 1
        return table

    def behavior_distribution_by_cluster(self, k: int, smoothing: float = 1e-9) -> np.ndarray:
        table = self.behavior_counts_by_cluster(k).astype(np.float64)
        table += smoothing
        return table / table.sum(axis=1, keepdims=True)

    def stationary_distribution(self, k: int) -> np.ndarray:
        P = self.transition_matrix(k)
        eigvals, eigvecs = np.linalg.eig(P.T)
        idx = int(np.argmin(np.abs(eigvals - 1.0)))
        vec = np.real(eigvecs[:, idx])
        vec = np.maximum(vec, 0)
        return vec / vec.sum()

    def behavior_name(self, canonical_index: int) -> str:
        return BEHAVIOR_NAMES[canonical_index]


def load_ncmcm(worm_id: int, data_dir: Path | str | None = None) -> NCMCMWorm:
    """Load one worm; ``worm_id`` is the 0-based BunDLe-Net id (0..4)."""
    if worm_id not in range(5):
        raise ValueError("worm_id must be 0..4")
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    path = root / f"nc_mcm_model_worm_{worm_id + 1}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing NC-MCM artifact: {path}")
    with np.load(path, allow_pickle=True) as payload:
        x = np.asarray(payload["arr_0"], dtype=np.float64)
        clusters = np.asarray(payload["arr_1"], dtype=np.int64)
        raw_b = np.asarray(payload["arr_2"], dtype=np.int64).reshape(-1)
        names = tuple(str(n) for n in payload["arr_3"])
        p_values = np.asarray(payload["arr_4"], dtype=np.float64).reshape(-1)
        bprob = np.asarray(payload["arr_5"], dtype=np.float64)
    if len(names) != 8:
        raise ValueError(f"expected 8 behaviour names, got {names}")
    mapping = np.asarray([NCMCM_TO_CANONICAL[name] for name in names], dtype=np.int64)
    behavior = mapping[raw_b - 1]  # raw labels are 1-based
    return NCMCMWorm(
        worm_id=worm_id,
        x=x,
        clusters=clusters,
        behavior=behavior,
        behavior_names=names,
        p_values=p_values,
        pairwise_probabilities=bprob,
    )
