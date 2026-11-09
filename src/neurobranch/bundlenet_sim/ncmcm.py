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

    def bootstrap_transition(
        self, k: int, n_boot: int = 200, seed: int = 0
    ) -> dict[str, np.ndarray]:
        """Dirichlet posterior over rows of the transition matrix.

        This quantifies sampling uncertainty of the *fitted* Markov chain.  It
        does not turn the chain into a causal model.
        """
        counts = self.transition_counts(k).astype(np.float64) + 0.5
        rng = np.random.default_rng(seed)
        samples = np.empty((n_boot, k, k), dtype=np.float64)
        for i in range(k):
            samples[:, i, :] = rng.dirichlet(counts[i], size=n_boot)
        return {
            "mean": samples.mean(axis=0),
            "low": np.quantile(samples, 0.025, axis=0),
            "high": np.quantile(samples, 0.975, axis=0),
        }

    def intervention_horizon(
        self, k: int, start_state: int, horizon: int = 20
    ) -> dict[str, np.ndarray]:
        """Propagate a forced state through the fitted Markov chain.

        ``do(S_t = start_state)`` at t=0, then follow the fitted transition
        matrix.  This is a model-based counterfactual under the Markov
        assumption; it is not an experimentally identified causal effect.
        """
        if not 1 <= start_state <= k:
            raise ValueError(f"start_state must be in 1..{k}")
        P = self.transition_matrix(k)
        B = self.behavior_distribution_by_cluster(k)
        state_dist = np.zeros((horizon + 1, k), dtype=np.float64)
        state_dist[0, start_state - 1] = 1.0
        for t in range(horizon):
            state_dist[t + 1] = state_dist[t] @ P
        behavior_dist = state_dist @ B
        return {"state_distribution": state_dist, "behavior_distribution": behavior_dist}

    def cross_worm_js(self, other: NCMCMWorm) -> dict[str, float]:
        """Jensen-Shannon divergence of *behaviour* statistics across worms.

        Cluster indices are not semantically aligned across independently fitted
        worms, so comparing state labels or state-transition matrices directly
        would be meaningless.  Behaviour labels are aligned, so only those are
        compared here.
        """
        from .benchmark import js_divergence

        p_beh = np.bincount(self.behavior, minlength=len(BEHAVIOR_NAMES)).astype(float)
        q_beh = np.bincount(other.behavior, minlength=len(BEHAVIOR_NAMES)).astype(float)
        p_trans = np.full((len(BEHAVIOR_NAMES), len(BEHAVIOR_NAMES)), 1e-9)
        q_trans = np.full_like(p_trans, 1e-9)
        for a, b in zip(self.behavior[:-1], self.behavior[1:]):
            p_trans[int(a), int(b)] += 1
        for a, b in zip(other.behavior[:-1], other.behavior[1:]):
            q_trans[int(a), int(b)] += 1
        p_trans /= p_trans.sum(axis=1, keepdims=True)
        q_trans /= q_trans.sum(axis=1, keepdims=True)
        return {
            "behavior_js": js_divergence(p_beh, q_beh),
            "behavior_transition_js": js_divergence(p_trans.reshape(-1), q_trans.reshape(-1)),
        }


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
