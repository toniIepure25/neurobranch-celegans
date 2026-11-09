"""Small, auditable dynamics and behaviour models on top of BunDLe-Net latents.

The point is not to claim that these polynomials are the worm's mechanism.  The
point is to provide a *generative* latent simulator with an explicit, inspectable
model class and a held-out-worm evaluation.  A quadratic map is the smallest
model that can represent curvature and multiple fixed points in a 3-D manifold;
it is also easy to beat with a persistence baseline, which keeps us honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

from .dataset import BEHAVIOR_NAMES, WormTrajectory, transition_arrays


def _quadratic_features(y: np.ndarray) -> np.ndarray:
    """Explicit, tested degree-2 feature map for 3-D inputs."""
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    if y.shape[1] != 3:
        raise ValueError(f"expected 3-D latents, got {y.shape}")
    y0, y1, y2 = y[:, 0], y[:, 1], y[:, 2]
    return np.column_stack(
        [
            np.ones(len(y)),
            y0,
            y1,
            y2,
            y0 * y0,
            y1 * y1,
            y2 * y2,
            y0 * y1,
            y0 * y2,
            y1 * y2,
        ]
    )


@dataclass
class PolynomialRidgeDynamics:
    """Ridge-regularised quadratic map ``y_{t+1} = W phi(y_t) + b``.

    Features and targets are standardised with training statistics.  The model
    is fitted in closed form; no GPU, no neural network, no hidden state.
    """

    alpha: float = 1.0
    feature_mean_: np.ndarray | None = field(default=None, init=False, repr=False)
    feature_scale_: np.ndarray | None = field(default=None, init=False, repr=False)
    target_mean_: np.ndarray | None = field(default=None, init=False, repr=False)
    target_scale_: np.ndarray | None = field(default=None, init=False, repr=False)
    coef_: np.ndarray | None = field(default=None, init=False, repr=False)
    clip_min_: np.ndarray | None = field(default=None, init=False, repr=False)
    clip_max_: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, latents: np.ndarray) -> PolynomialRidgeDynamics:
        latents = np.asarray(latents, dtype=np.float64)
        if latents.ndim != 2 or latents.shape[1] != 3 or len(latents) < 3:
            raise ValueError(f"expected (T, 3) with T >= 3, got {latents.shape}")
        return self.fit_transitions(latents[:-1], latents[1:])

    def fit_transitions(self, source: np.ndarray, target: np.ndarray) -> PolynomialRidgeDynamics:
        """Fit from explicitly paired one-step transitions.

        Passing transitions rather than trajectories lets callers concatenate
        worms without accidentally treating a worm boundary as a real step.
        """
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if source.ndim != 2 or source.shape[1] != 3:
            raise ValueError(f"source must be (N, 3), got {source.shape}")
        if target.shape != source.shape:
            raise ValueError(f"target must match source, got {target.shape} vs {source.shape}")
        x = _quadratic_features(source)
        y = target
        self.feature_mean_ = x.mean(axis=0)
        self.feature_scale_ = x.std(axis=0)
        self.feature_scale_[self.feature_scale_ < 1e-8] = 1.0
        self.target_mean_ = y.mean(axis=0)
        self.target_scale_ = y.std(axis=0)
        self.target_scale_[self.target_scale_ < 1e-8] = 1.0
        xs = (x - self.feature_mean_) / self.feature_scale_
        ys = (y - self.target_mean_) / self.target_scale_
        # The bias feature is centred to zero by standardisation, so it carries no
        # information; regularise it too to keep the normal equations invertible.
        penalty = np.eye(xs.shape[1]) * self.alpha
        self.coef_ = np.linalg.solve(xs.T @ xs + penalty, xs.T @ ys)
        observed = np.concatenate([source, target], axis=0)
        self.clip_min_ = observed.min(axis=0)
        self.clip_max_ = observed.max(axis=0)
        return self

    def predict(self, latents: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("model is not fitted")
        x = _quadratic_features(latents)
        xs = (x - self.feature_mean_) / self.feature_scale_
        ys = xs @ self.coef_
        out = ys * self.target_scale_ + self.target_mean_
        return np.clip(out, self.clip_min_, self.clip_max_)

    def rollout(self, y0: np.ndarray, n_steps: int, noise: float = 0.0, seed: int | None = None) -> np.ndarray:
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        rng = np.random.default_rng(seed)
        out = np.empty((n_steps, 3), dtype=np.float64)
        state = np.asarray(y0, dtype=np.float64).reshape(1, 3)
        for t in range(n_steps):
            state = self.predict(state)
            if noise > 0:
                state = state + rng.normal(0.0, noise, size=state.shape)
            out[t] = state[0]
        return out


@dataclass
class BehaviorReadout:
    """Logistic read-out from a 3-D latent state to the eight behaviour labels."""

    max_iter: int = 3000
    coef_: np.ndarray | None = field(default=None, init=False, repr=False)
    intercept_: np.ndarray | None = field(default=None, init=False, repr=False)
    classes_: np.ndarray | None = field(default=None, init=False, repr=False)
    class_counts_: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, latents: np.ndarray, behavior: np.ndarray) -> BehaviorReadout:
        latents = np.asarray(latents, dtype=np.float64)
        behavior = np.asarray(behavior, dtype=np.int64).reshape(-1)
        if len(latents) != len(behavior):
            raise ValueError("latents and behaviour must have the same length")
        self.class_counts_ = np.bincount(behavior, minlength=len(BEHAVIOR_NAMES))
        model = LogisticRegression(max_iter=self.max_iter, class_weight="balanced")
        model.fit(latents, behavior)
        self.coef_ = np.asarray(model.coef_, dtype=np.float64)
        self.intercept_ = np.asarray(model.intercept_, dtype=np.float64)
        self.classes_ = np.asarray(model.classes_, dtype=np.int64)
        return self

    @property
    def majority_class_(self) -> int:
        if self.class_counts_ is None:
            raise RuntimeError("model is not fitted")
        return int(np.argmax(self.class_counts_))

    def predict_proba(self, latents: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None or self.classes_ is None:
            raise RuntimeError("model is not fitted")
        x = np.atleast_2d(np.asarray(latents, dtype=np.float64))
        logits = x @ self.coef_.T + self.intercept_
        if logits.shape[1] == 1:  # binary fallback
            p1 = 1.0 / (1.0 + np.exp(-logits[:, 0]))
            proba = np.column_stack([1.0 - p1, p1])
        else:
            logits = logits - logits.max(axis=1, keepdims=True)
            proba = np.exp(logits)
            proba /= proba.sum(axis=1, keepdims=True)
        # The classifier may not have seen every class in a small training split;
        # expand to the full eight-class label space so the UI is stable.
        full = np.zeros((proba.shape[0], len(BEHAVIOR_NAMES)), dtype=np.float64)
        for column, cls in enumerate(self.classes_):
            full[:, int(cls)] = proba[:, column]
        return full

    def predict(self, latents: np.ndarray) -> np.ndarray:
        return self.predict_proba(latents).argmax(axis=1)


@dataclass
class FittedModels:
    dynamics: PolynomialRidgeDynamics
    behavior: BehaviorReadout
    train_worm_ids: tuple[int, ...]

    def simulate(
        self,
        y0: np.ndarray,
        n_steps: int,
        noise: float = 0.0,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        trajectory = self.dynamics.rollout(y0, n_steps=n_steps, noise=noise, seed=seed)
        proba = self.behavior.predict_proba(trajectory)
        return trajectory, proba


def fit_models(
    trajectories: list[WormTrajectory],
    alpha: float = 1.0,
    train_worm_ids: tuple[int, ...] | None = None,
) -> FittedModels:
    """Fit dynamics and behaviour read-out on the requested worms."""
    if train_worm_ids is not None:
        wanted = set(train_worm_ids)
        selected = [t for t in trajectories if t.worm_id in wanted]
        missing = wanted - {t.worm_id for t in selected}
        if missing:
            raise ValueError(f"requested worms not present: {sorted(missing)}")
    else:
        selected = list(trajectories)
    if not selected:
        raise ValueError("no trajectories selected")

    x, y, _ = transition_arrays(selected)
    dynamics = PolynomialRidgeDynamics(alpha=alpha).fit_transitions(x, y)
    latents = np.concatenate([t.latent for t in selected], axis=0)
    behavior = np.concatenate([t.behavior for t in selected], axis=0)
    readout = BehaviorReadout().fit(latents, behavior)
    return FittedModels(
        dynamics=dynamics,
        behavior=readout,
        train_worm_ids=tuple(sorted(t.worm_id for t in selected)),
    )
