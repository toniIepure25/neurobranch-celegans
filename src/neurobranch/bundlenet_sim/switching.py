"""Behaviour-conditioned switching linear dynamics on BunDLe-Net latents.

This is a deliberately small, inspectable generative model:

    b_{t+1} ~ P(b_{t+1} | b_t)
    y_{t+1} = A_{b_t} y_t + c_{b_t} + eps_t

It is not a mechanistic model and it is not a hidden-state SLDS: the behaviour
label is observed at training time and is used as the switch.  The point is to
provide a stable, behaviour-coherent alternative to a single global quadratic
map, with an explicit Markov behaviour process.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dataset import BEHAVIOR_NAMES, WormTrajectory, transition_arrays


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Fit ``y = [x, 1] @ W`` with an unpenalised intercept-free design."""
    design = np.column_stack([x, np.ones(len(x))])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[-1, -1] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def _stable(matrix: np.ndarray, max_radius: float = 0.995) -> np.ndarray:
    radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    if not np.isfinite(radius) or radius <= max_radius or radius == 0:
        return matrix
    return matrix * (max_radius / radius)


@dataclass
class SwitchingLinearDynamics:
    alpha: float = 10.0
    min_behavior_samples: int = 30
    max_radius: float = 0.995
    global_coef_: np.ndarray | None = field(default=None, init=False, repr=False)
    coef_: np.ndarray | None = field(default=None, init=False, repr=False)
    behavior_transition_: np.ndarray | None = field(default=None, init=False, repr=False)
    behavior_counts_: np.ndarray | None = field(default=None, init=False, repr=False)
    clip_min_: np.ndarray | None = field(default=None, init=False, repr=False)
    clip_max_: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit_transitions(
        self,
        source: np.ndarray,
        target: np.ndarray,
        behavior: np.ndarray,
    ) -> SwitchingLinearDynamics:
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        behavior = np.asarray(behavior, dtype=np.int64).reshape(-1)
        if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
            raise ValueError(f"expected matching (N, 3), got {source.shape} and {target.shape}")
        if len(behavior) != len(source):
            raise ValueError("behaviour must align with transitions")

        self.global_coef_ = _ridge_fit(source, target, self.alpha)
        coef = np.repeat(self.global_coef_[None, :, :], len(BEHAVIOR_NAMES), axis=0)
        for label in range(len(BEHAVIOR_NAMES)):
            mask = behavior == label
            if int(mask.sum()) < self.min_behavior_samples:
                continue
            fitted = _ridge_fit(source[mask], target[mask], self.alpha)
            fitted[:3, :3] = _stable(fitted[:3, :3], self.max_radius)
            coef[label] = fitted
        self.coef_ = coef

        counts = np.full((len(BEHAVIOR_NAMES), len(BEHAVIOR_NAMES)), 0.5, dtype=np.float64)
        np.add.at(counts, (behavior[:-1], behavior[1:]), 1.0)
        self.behavior_counts_ = counts
        self.behavior_transition_ = counts / counts.sum(axis=1, keepdims=True)
        observed = np.concatenate([source, target], axis=0)
        self.clip_min_ = observed.min(axis=0)
        self.clip_max_ = observed.max(axis=0)
        return self

    def predict(self, state: np.ndarray, behavior: int) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model is not fitted")
        state = np.asarray(state, dtype=np.float64).reshape(1, 3)
        design = np.column_stack([state, np.ones(1)])
        out = design @ self.coef_[int(behavior)]
        return np.clip(out[0], self.clip_min_, self.clip_max_)

    def sample_next_behavior(self, behavior: int, rng: np.random.Generator) -> int:
        if self.behavior_transition_ is None:
            raise RuntimeError("model is not fitted")
        return int(rng.choice(len(BEHAVIOR_NAMES), p=self.behavior_transition_[int(behavior)]))

    def rollout(
        self,
        y0: np.ndarray,
        behavior0: int,
        n_steps: int,
        noise: float = 0.0,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        rng = np.random.default_rng(seed)
        out = np.empty((n_steps, 3), dtype=np.float64)
        labels = np.empty(n_steps, dtype=np.int64)
        state = np.asarray(y0, dtype=np.float64).reshape(3)
        behavior = int(behavior0)
        for t in range(n_steps):
            state = self.predict(state, behavior)
            if noise > 0:
                state = np.clip(
                    state + rng.normal(0.0, noise, size=3), self.clip_min_, self.clip_max_
                )
            behavior = self.sample_next_behavior(behavior, rng)
            out[t] = state
            labels[t] = behavior
        return out, labels


@dataclass
class SwitchingFit:
    dynamics: SwitchingLinearDynamics
    behavior: object
    train_worm_ids: tuple[int, ...]

    def simulate(
        self,
        y0: np.ndarray,
        behavior0: int,
        n_steps: int,
        noise: float = 0.0,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        latents, labels = self.dynamics.rollout(y0, behavior0, n_steps, noise=noise, seed=seed)
        proba = self.behavior.predict_proba(latents)
        return latents, labels, proba


def fit_switching(
    trajectories: list[WormTrajectory],
    alpha: float = 10.0,
) -> SwitchingFit:
    from .dynamics import BehaviorReadout

    x, y, _ = transition_arrays(trajectories)
    transition_behavior = np.concatenate([t.behavior[:-1] for t in trajectories])
    dynamics = SwitchingLinearDynamics(alpha=alpha).fit_transitions(x, y, transition_behavior)
    latents = np.concatenate([t.latent for t in trajectories], axis=0)
    behavior = np.concatenate([t.behavior for t in trajectories], axis=0)
    readout = BehaviorReadout().fit(latents, behavior)
    return SwitchingFit(
        dynamics=dynamics,
        behavior=readout,
        train_worm_ids=tuple(sorted(t.worm_id for t in trajectories)),
    )
