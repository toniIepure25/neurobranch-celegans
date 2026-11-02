"""Small, prespecified probabilistic models for neural forecast uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge

from neurobranch.probabilistic.residual import ResidualSeries


def diagonal_scale(values: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("need at least two residual rows")
    return np.sqrt(np.maximum(values.var(axis=0, ddof=1), floor))


@dataclass
class ScaleModel:
    """A frozen diagonal innovation scale."""

    scale: np.ndarray
    kind: str
    metadata: dict


def state_features(X: np.ndarray, t: int, window: int = 10) -> np.ndarray:
    """Features available at time ``t`` only.

    Absolute level, absolute one-step change, recent standard deviation and
    recent mean absolute level. No value after ``t`` is used.
    """
    X = np.asarray(X, dtype=np.float64)
    t = int(t)
    prev = max(0, t - 1)
    lo = max(0, t - int(window) + 1)
    recent = X[lo : t + 1]
    return np.column_stack(
        [
            np.abs(X[t]),
            np.abs(X[t] - X[prev]),
            recent.std(axis=0),
            np.abs(recent).mean(axis=0),
        ]
    )


class StateVarianceModel:
    """Per-neuron log-variance regression on pre-landmark state features.

    The model is fit only on residuals whose targets are at or before
    ``train_end``. Features use the state immediately before each target.
    A smearing correction is applied because exponentiated log-variance
    predictions are biased downward.
    """

    feature_names = ("abs_level", "abs_change", "recent_std", "recent_abs_mean")

    def __init__(self, models, feature_mean, feature_std, bias, train_end):
        self.models = models
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.bias = bias
        self.train_end = int(train_end)

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        residuals: ResidualSeries,
        train_end: int,
        *,
        ridge: float = 1.0,
        window: int = 10,
        eps: float = 1e-6,
        max_samples: int = 1200,
    ) -> StateVarianceModel:
        train_end = int(train_end)
        mask = residuals.targets <= train_end
        targets = residuals.targets[mask]
        values = residuals.values[mask]
        if len(targets) < 30:
            raise ValueError("too few training residuals")
        # Keep the most recent residuals; state features are aligned to k - 1.
        targets = targets[-int(max_samples) :]
        values = values[-int(max_samples) :]
        feats = np.stack([state_features(X, int(k) - 1, window=window) for k in targets])
        n, d, p = feats.shape
        flat = feats.reshape(n * d, p)
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std = np.where(std > 1e-12, std, 1.0)
        z = ((flat - mean) / std).reshape(n, d, p)
        y = np.log(np.maximum(values**2, eps))
        models, biases = [], []
        for neuron in range(d):
            model = Ridge(alpha=float(ridge), fit_intercept=True)
            model.fit(z[:, neuron, :], y[:, neuron])
            pred = model.predict(z[:, neuron, :])
            ratio = np.mean(values[:, neuron] ** 2) / max(
                np.mean(np.exp(np.clip(pred, -20, 20))), 1e-12
            )
            models.append(model)
            biases.append(float(np.clip(ratio, 0.1, 10.0)))
        return cls(models, mean, std, np.asarray(biases), train_end)

    def predict(self, X: np.ndarray, t: int, window: int = 10) -> np.ndarray:
        feats = state_features(X, int(t), window=window)
        z = (feats - self.feature_mean) / self.feature_std
        out = np.empty(z.shape[0], dtype=np.float64)
        for neuron, model in enumerate(self.models):
            out[neuron] = np.exp(np.clip(float(model.predict(z[neuron][None, :])[0]), -20, 20))
        return np.sqrt(np.maximum(out * self.bias, 1e-8))


def low_rank_plus_diagonal_covariance(
    eigenvalues: np.ndarray, eigenvectors: np.ndarray, residual_variance: np.ndarray
) -> np.ndarray:
    low = (eigenvectors * eigenvalues[None, :]) @ eigenvectors.T
    return low + np.diag(np.maximum(residual_variance, 1e-10))


def fit_low_rank_plus_diagonal(values: np.ndarray, rank: int) -> tuple[np.ndarray, dict]:
    """Fit a rank-r factor covariance plus a positive diagonal remainder."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 5:
        raise ValueError("need a residual matrix")
    rank = int(rank)
    if rank < 1 or rank >= values.shape[1]:
        raise ValueError("rank must be between 1 and d-1")
    cov = np.cov(values, rowvar=False, ddof=1)
    eigval, eigvec = np.linalg.eigh((cov + cov.T) / 2.0)
    order = np.argsort(eigval)[::-1][:rank]
    vals = np.clip(eigval[order], 1e-10, None)
    vecs = eigvec[:, order]
    low = (vecs * vals[None, :]) @ vecs.T
    remainder = np.maximum(np.diag(cov - low), 1e-10)
    full = low + np.diag(remainder)
    return full, {
        "rank": rank,
        "retained_eigenvalue_fraction": float(vals.sum() / max(np.trace(cov), 1e-12)),
        "residual_diagonal_fraction": float(remainder.sum() / max(np.trace(full), 1e-12)),
    }


def diagonal_covariance(values: np.ndarray) -> np.ndarray:
    return np.diag(np.maximum(np.asarray(values).var(axis=0, ddof=1), 1e-10))
