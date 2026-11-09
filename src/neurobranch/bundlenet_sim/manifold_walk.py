"""Data-driven manifold walk on the published BunDLe-Net embeddings.

This is deliberately *not* a mechanistic dynamics model.  It is a kernel
transition sampler: from the current latent state, sample one of the k nearest
observed states and take its recorded successor.  It stays on the observed
manifold, which makes it useful for interactive exploration, while making no
claim to extrapolate beyond the five recorded worms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .dataset import WormTrajectory, transition_arrays


@dataclass
class ManifoldWalk:
    n_neighbors: int = 25
    noise: float = 0.01
    states_: np.ndarray | None = None
    successors_: np.ndarray | None = None
    behaviors_: np.ndarray | None = None
    nn_: NearestNeighbors | None = None

    def fit(self, trajectories: list[WormTrajectory]) -> ManifoldWalk:
        xs, ys, _ = transition_arrays(trajectories)
        behaviors = np.concatenate([t.behavior[:-1] for t in trajectories])
        self.states_ = xs
        self.successors_ = ys
        self.behaviors_ = behaviors
        self.nn_ = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(xs))).fit(xs)
        return self

    def step(
        self,
        state: np.ndarray,
        rng: np.random.Generator,
        target_behavior: int | None = None,
        temperature: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        if self.nn_ is None or self.states_ is None or self.successors_ is None:
            raise RuntimeError("model is not fitted")
        state = np.asarray(state, dtype=np.float64).reshape(1, 3)
        k = self.nn_.n_neighbors
        distances, indices = self.nn_.kneighbors(state, n_neighbors=k)
        distances = distances[0]
        indices = indices[0]
        if target_behavior is not None:
            keep = self.behaviors_[indices] == target_behavior
            if keep.any():
                distances = distances[keep]
                indices = indices[keep]
        # Softmax over negative squared distance; temperature controls diversity.
        logits = -(distances**2) / max(temperature, 1e-6)
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()
        choice = int(rng.choice(len(indices), p=weights))
        nxt = self.successors_[indices[choice]].copy()
        if self.noise > 0:
            nxt += rng.normal(0.0, self.noise, size=3)
        return nxt, int(self.behaviors_[indices[choice]])

    def rollout_with_behavior(
        self,
        y0: np.ndarray,
        n_steps: int,
        seed: int | None = None,
        target_behavior: int | None = None,
        temperature: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        out = np.empty((n_steps, 3), dtype=np.float64)
        labels = np.empty(n_steps, dtype=np.int64)
        state = np.asarray(y0, dtype=np.float64).reshape(3)
        for t in range(n_steps):
            state, label = self.step(
                state, rng, target_behavior=target_behavior, temperature=temperature
            )
            out[t] = state
            labels[t] = label
        return out, labels

    def rollout(
        self,
        y0: np.ndarray,
        n_steps: int,
        seed: int | None = None,
        target_behavior: int | None = None,
        temperature: float = 1.0,
    ) -> np.ndarray:
        return self.rollout_with_behavior(
            y0,
            n_steps,
            seed=seed,
            target_behavior=target_behavior,
            temperature=temperature,
        )[0]
