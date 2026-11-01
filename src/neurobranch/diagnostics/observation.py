"""Observation-aware diagnostic models.

These are deliberately small linear models.  They test whether latent memory or
a causal calcium-filter view improves real forecasting; they do not claim to
recover membrane potential or spike trains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from neurobranch.dynamics.stability import stabilize_companion


def _ridge(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    mu = design.mean(axis=0)
    sd = design.std(axis=0)
    sd[sd < 1e-8] = 1.0
    xs = (design - mu) / sd
    y = target - target.mean(axis=0)
    coef = np.linalg.solve(xs.T @ xs + ridge * np.eye(xs.shape[1]), xs.T @ y)
    intercept = target.mean(axis=0) - (mu / sd) @ coef
    return np.vstack([coef / sd[:, None], intercept])


@dataclass
class SecondOrderLatentLDS:
    """Latent VAR(2) with a train-only PCA observation map.

    The second latent lag is the minimal additional memory needed for a
    calcium-like filtered observation.  It is not a biophysical model.
    """

    pca: PCA
    A1: np.ndarray
    A2: np.ndarray
    b: np.ndarray
    n_neurons: int
    latent_dim: int
    stability: dict[str, Any] = field(default_factory=dict)
    name: str = "second_order_latent_lds"
    history_len: int = 2
    native_stochastic: bool = False
    supports_input: bool = False

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        train_mask: np.ndarray,
        starts: np.ndarray,
        *,
        latent_dim: int = 10,
        ridge: float = 1e-2,
        spectral_cap: float = 0.995,
        seed: int = 17,
    ) -> SecondOrderLatentLDS:
        X = np.asarray(X, dtype=np.float64)
        train_mask = np.asarray(train_mask, dtype=bool)
        starts = np.asarray(starts, dtype=np.int64)
        if latent_dim < 1 or latent_dim > X.shape[1]:
            raise ValueError("invalid latent_dim")
        pca = PCA(n_components=int(latent_dim), random_state=int(seed)).fit(X[train_mask])
        Z = pca.transform(X)
        design = np.column_stack([Z[starts], Z[starts - 1], np.ones(len(starts))])
        target = Z[starts + 1]
        coef = _ridge(design, target, ridge)
        A1 = coef[:latent_dim].T.copy()
        A2 = coef[latent_dim : 2 * latent_dim].T.copy()
        b = coef[-1].copy()
        A, record = stabilize_companion(
            np.column_stack([A1, A2]),
            n_neurons=latent_dim,
            history=2,
            cap=spectral_cap,
        )
        return cls(
            pca=pca,
            A1=A[:, :latent_dim],
            A2=A[:, latent_dim:],
            b=b,
            n_neurons=int(X.shape[1]),
            latent_dim=int(latent_dim),
            stability=record,
        )

    @property
    def observation_matrix(self) -> np.ndarray:
        return np.asarray(self.pca.components_, dtype=np.float64).T

    @property
    def observation_mean(self) -> np.ndarray:
        return np.asarray(self.pca.mean_, dtype=np.float64)

    def encode(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr[None]
        out = np.asarray(self.pca.transform(arr), dtype=np.float64)
        return out[0] if squeeze else out

    def decode(self, Z: np.ndarray) -> np.ndarray:
        arr = np.asarray(Z, dtype=np.float64)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr[None]
        out = np.asarray(self.pca.inverse_transform(arr), dtype=np.float64)
        return out[0] if squeeze else out

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.encode(X)

    def _next_latent(self, z_t: np.ndarray, z_prev: np.ndarray) -> np.ndarray:
        return z_t @ self.A1.T + z_prev @ self.A2.T + self.b

    def predict_next(self, history: np.ndarray) -> np.ndarray:
        hist = np.asarray(history, dtype=np.float64)
        if hist.ndim != 3 or hist.shape[1] < 2:
            raise ValueError("second-order model needs (B, >=2, N) history")
        z = self.encode(hist[:, -2:, :].transpose(1, 0, 2).reshape(-1, self.n_neurons))
        z = z.reshape(2, len(hist), self.latent_dim)
        z_next = self._next_latent(z[1], z[0])
        return np.asarray(self.decode(z_next), dtype=np.float64)

    def behavior_proba(self, history: np.ndarray) -> np.ndarray:
        hist = np.asarray(history)
        return np.full((hist.shape[0], 4), 0.25, dtype=np.float64)

    def latent(self, history: np.ndarray) -> np.ndarray:
        hist = np.asarray(history, dtype=np.float64)
        return self.encode(hist[:, -1, :])

    def rollout(
        self,
        x0: np.ndarray,
        steps: int,
        *,
        seed: int = 0,
        noise_std: float = 0.0,
        perturb=None,
        initial_history: np.ndarray | None = None,
    ) -> np.ndarray:
        if initial_history is None:
            raise ValueError("second-order rollout requires initial_history")
        hist = np.asarray(initial_history, dtype=np.float64)
        if hist.ndim != 2 or hist.shape[0] < 2:
            raise ValueError("initial_history must be (>=2, N)")
        z = self.encode(hist[-2:])
        z_prev, z_t = z[0], z[1]
        states = np.empty((steps + 1, self.n_neurons), dtype=np.float64)
        states[0] = hist[-1]
        rng = np.random.default_rng(seed)
        for s in range(steps):
            z_next = self._next_latent(z_t, z_prev)
            if noise_std > 0:
                z_next = z_next + rng.normal(scale=noise_std, size=self.latent_dim)
            x_next = self.decode(z_next[None])[0]
            if perturb is not None:
                x_next = np.asarray(perturb(x_next, s), dtype=np.float64)
            states[s + 1] = x_next
            z_prev, z_t = z_t, self.encode(x_next[None])[0]
        return states


def _causal_calcium_transform(X: np.ndarray, alpha: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    prev = np.vstack([X[:1], X[:-1]])
    return X + ((1.0 - alpha) / alpha) * (X - prev)


@dataclass
class CausalCalciumLDS:
    """Causal high-pass latent LDS followed by a first-order calcium filter.

    The transform is an explicit, testable approximation, not deconvolution.
    Alpha is selected on training one-step error only.
    """

    pca: PCA
    A: np.ndarray
    b: np.ndarray
    alpha: float
    n_neurons: int
    latent_dim: int
    stability: dict[str, Any] = field(default_factory=dict)
    alpha_table: list[dict[str, float]] = field(default_factory=list)
    name: str = "causal_calcium_lds"
    history_len: int = 2
    native_stochastic: bool = False
    supports_input: bool = False

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        train_mask: np.ndarray,
        starts: np.ndarray,
        *,
        latent_dim: int = 10,
        ridge: float = 1e-2,
        spectral_cap: float = 0.995,
        alpha_grid: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9),
        seed: int = 17,
    ) -> CausalCalciumLDS:
        X = np.asarray(X, dtype=np.float64)
        train_mask = np.asarray(train_mask, dtype=bool)
        starts = np.asarray(starts, dtype=np.int64)
        best: dict[str, Any] | None = None
        table = []
        for alpha in alpha_grid:
            drive = _causal_calcium_transform(X, alpha)
            pca = PCA(n_components=int(latent_dim), random_state=int(seed)).fit(drive[train_mask])
            Z = pca.transform(drive)
            design = np.column_stack([Z[starts], np.ones(len(starts))])
            coef = _ridge(design, Z[starts + 1], ridge)
            A = coef[:latent_dim].T.copy()
            b = coef[-1].copy()
            A, record = stabilize_companion(
                np.column_stack([A, np.zeros_like(A)]),
                n_neurons=latent_dim,
                history=2,
                cap=spectral_cap,
            )
            A = A[:, :latent_dim]
            z_next = Z[starts] @ A.T + b
            drive_next = pca.inverse_transform(z_next)
            yhat = (1.0 - alpha) * X[starts] + alpha * drive_next
            mse = float(np.mean((yhat - X[starts + 1]) ** 2))
            table.append({"alpha": float(alpha), "train_one_step_mse": mse})
            if best is None or mse < best["mse"]:
                best = {"alpha": float(alpha), "mse": mse, "pca": pca, "A": A, "b": b, "record": record}
        assert best is not None
        return cls(
            pca=best["pca"],
            A=best["A"],
            b=best["b"],
            alpha=best["alpha"],
            n_neurons=int(X.shape[1]),
            latent_dim=int(latent_dim),
            stability=best["record"],
            alpha_table=table,
        )

    @property
    def observation_matrix(self) -> np.ndarray:
        return np.asarray(self.pca.components_, dtype=np.float64).T

    def encode(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr[None]
        drive = _causal_calcium_transform(arr, self.alpha)
        out = np.asarray(self.pca.transform(drive), dtype=np.float64)
        return out[0] if squeeze else out

    def decode_drive(self, Z: np.ndarray) -> np.ndarray:
        arr = np.asarray(Z, dtype=np.float64)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr[None]
        out = np.asarray(self.pca.inverse_transform(arr), dtype=np.float64)
        return out[0] if squeeze else out

    def predict_next(self, history: np.ndarray) -> np.ndarray:
        hist = np.asarray(history, dtype=np.float64)
        if hist.ndim != 3 or hist.shape[1] < 2:
            raise ValueError("calcium model needs (B, >=2, N) history")
        z_t = np.stack([self.encode(hist[i, -2:, :])[-1] for i in range(len(hist))])
        z_next = z_t @ self.A.T + self.b
        drive_next = self.decode_drive(z_next)
        return (1.0 - self.alpha) * hist[:, -1, :] + self.alpha * drive_next

    def behavior_proba(self, history: np.ndarray) -> np.ndarray:
        hist = np.asarray(history)
        return np.full((hist.shape[0], 4), 0.25, dtype=np.float64)

    def latent(self, history: np.ndarray) -> np.ndarray:
        hist = np.asarray(history, dtype=np.float64)
        return np.stack([self.encode(hist[i, -2:, :])[-1] for i in range(len(hist))])

    def rollout(
        self,
        x0: np.ndarray,
        steps: int,
        *,
        seed: int = 0,
        noise_std: float = 0.0,
        perturb=None,
        initial_history: np.ndarray | None = None,
    ) -> np.ndarray:
        if initial_history is None:
            raise ValueError("calcium rollout requires initial_history")
        hist = np.asarray(initial_history, dtype=np.float64)
        if hist.ndim != 2 or hist.shape[0] < 2:
            raise ValueError("initial_history must be (>=2, N)")
        z = self.encode(hist[-2:])
        z_t = z[-1]
        c = hist[-1].copy()
        states = np.empty((steps + 1, self.n_neurons), dtype=np.float64)
        states[0] = c
        rng = np.random.default_rng(seed)
        for s in range(steps):
            z_next = z_t @ self.A.T + self.b
            if noise_std > 0:
                z_next = z_next + rng.normal(scale=noise_std, size=self.latent_dim)
            drive_next = self.decode_drive(z_next[None])[0]
            c_next = (1.0 - self.alpha) * c + self.alpha * drive_next
            if perturb is not None:
                c_next = np.asarray(perturb(c_next, s), dtype=np.float64)
            states[s + 1] = c_next
            z_t = self.encode(c_next[None] if False else np.vstack([hist[-1], c_next]))[-1]
            c = c_next
        return states
