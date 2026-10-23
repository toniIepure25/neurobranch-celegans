"""Generate a documented synthetic branching system for smoke tests.

This is synthetic data.  It is used to verify that the analysis code can
recover a known branch point; it is never presented as experimental evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticBranchConfig:
    n_steps: int = 2400
    n_neurons: int = 24
    latent_dim: int = 3
    noise: float = 0.04
    branch_bias: float = 0.65
    seed: int = 7


@dataclass
class SyntheticDataset:
    traces: np.ndarray
    latent: np.ndarray
    labels: np.ndarray
    branch_frame: int
    ground_truth: dict


def generate_branching_system(cfg: SyntheticBranchConfig | None = None) -> SyntheticDataset:
    """Two stochastic futures from a shared metastable state.

    The latent state has a slow reversal axis and a turn-direction axis.  Around
    the designated branch point the sign of the turn axis is sampled, producing
    two behaviourally distinct futures.  Observations are a random linear
    mixture plus independent noise.
    """
    cfg = cfg or SyntheticBranchConfig()
    rng = np.random.default_rng(cfg.seed)
    T, N, D = cfg.n_steps, cfg.n_neurons, cfg.latent_dim
    z = np.zeros((T, D), dtype=np.float64)
    labels = np.empty(T, dtype="<U7")
    z[0] = [1.0, 0.0, 0.0]
    labels[0] = "FWD"
    branch_frame = T // 3
    chosen = 1
    for t in range(1, T):
        prev = z[t - 1].copy()
        if t < branch_frame:
            prev[0] = 0.97 * prev[0] + 0.04 * rng.normal()
            prev[1] = 0.90 * prev[1] + 0.03 * rng.normal()
            prev[2] = 0.92 * prev[2] + 0.03 * rng.normal()
            labels[t] = "FWD" if prev[0] > 0.15 else "REV"
        elif t == branch_frame:
            chosen = 1 if rng.random() < cfg.branch_bias else -1
            prev[:] = [0.8, 0.0, 0.0]
            labels[t] = "REV"
        elif t < branch_frame + 260:
            phase = (t - branch_frame) / 260.0
            prev[0] = 0.94 * prev[0] - 0.02
            prev[1] = 0.88 * prev[1] + 0.16 * chosen * (1 - phase) + 0.03 * rng.normal()
            prev[2] = 0.90 * prev[2] + 0.02 * rng.normal()
            labels[t] = "TURN" if phase > 0.45 else "REV"
        else:
            prev *= 0.95
            prev[0] += 0.05
            prev[1] += 0.01 * rng.normal()
            prev[2] += 0.01 * rng.normal()
            labels[t] = "FWD"
        z[t] = prev + cfg.noise * rng.normal(size=D)
    mixing = rng.normal(size=(D, N)) / np.sqrt(D)
    traces = z @ mixing + cfg.noise * 0.5 * rng.normal(size=(T, N))
    return SyntheticDataset(
        traces=traces.astype(np.float32),
        latent=z.astype(np.float32),
        labels=labels,
        branch_frame=branch_frame,
        ground_truth={
            "chosen_turn_direction": chosen,
            "branch_frame": branch_frame,
            "mixing_matrix": mixing,
            "note": "synthetic; not biological evidence",
        },
    )
