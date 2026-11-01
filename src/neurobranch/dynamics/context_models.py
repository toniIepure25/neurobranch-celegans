"""Causally filtered, train-only PCA latent linear dynamics.

This is deliberately a small, auditable model.  A PCA encoder is fitted only on
the training mask; the latent transition is a stable linear map with an optional
known external input; and every rollout starts from a posterior inferred from
observed history.  No future observation is used by the filter or the rollout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import nnls
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from neurobranch.dynamics.models import BehaviorProbe
from neurobranch.dynamics.stability import stabilize_matrix


def _ridge_solve(phi: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    p = phi.shape[1]
    gram = phi.T @ phi + float(ridge) * np.eye(p)
    return np.linalg.solve(gram, phi.T @ target)


@dataclass
class StableLatentLDS:
    """Stable latent LDS with a causal Kalman-style filter."""

    pca: PCA
    A: np.ndarray
    b: np.ndarray
    B: np.ndarray
    process_cov: np.ndarray
    obs_cov: np.ndarray
    initial_cov: np.ndarray
    probe: BehaviorProbe
    n_neurons: int
    latent_dim: int
    input_dim: int = 0
    input_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    input_std: np.ndarray = field(default_factory=lambda: np.ones(0))
    spectral_cap: float = 0.995
    stability: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    name: str = "latent_lds"
    history_len: int = 1
    native_stochastic: bool = False
    supports_input: bool = True

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        starts: np.ndarray,
        *,
        train_mask: np.ndarray,
        latent_dim: int = 10,
        ridge: float = 1e-2,
        spectral_cap: float = 0.995,
        input_values: np.ndarray | None = None,
        preserve_mean: bool = True,
        seed: int = 17,
    ) -> StableLatentLDS:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        train_mask = np.asarray(train_mask, dtype=bool)
        starts = np.asarray(starts, dtype=np.int64)
        if X.ndim != 2 or len(X) != len(train_mask) or len(y) != len(X):
            raise ValueError("X, y and train_mask have inconsistent shapes")
        if starts.size == 0:
            raise ValueError("at least one transition start is required")
        if not (train_mask[starts].all() and train_mask[starts + 1].all()):
            raise ValueError("all transition starts must lie wholly inside the training mask")
        if int(train_mask.sum()) < max(20, latent_dim + 2):
            raise ValueError("training mask is too small for the requested latent dimension")
        if latent_dim < 1 or latent_dim > min(X.shape[1], int(train_mask.sum()) - 2):
            raise ValueError("latent_dim is outside the identifiable range")

        pca = PCA(n_components=int(latent_dim), random_state=int(seed))
        pca.fit(X[train_mask])
        Z = pca.transform(X)

        input_dim = 0
        input_mean = np.zeros(0, dtype=np.float64)
        input_std = np.ones(0, dtype=np.float64)
        if input_values is not None:
            U = np.asarray(input_values, dtype=np.float64)
            if U.ndim == 1:
                U = U[:, None]
            if U.ndim != 2 or len(U) != len(X):
                raise ValueError("input_values must be (T,) or (T, n_inputs)")
            if not np.isfinite(U).all():
                raise ValueError("input_values contains non-finite values")
            input_dim = int(U.shape[1])
            input_mean = U[train_mask].mean(axis=0)
            input_std = U[train_mask].std(axis=0)
            input_std[input_std < 1e-9] = 1.0
            U = (U - input_mean) / input_std
        else:
            U = np.zeros((len(X), 0), dtype=np.float64)

        phi = np.column_stack([Z[starts], U[starts], np.ones(len(starts))])
        target = Z[starts + 1]
        coef = _ridge_solve(phi, target, ridge)
        A = coef[:latent_dim].T.copy()
        B = coef[latent_dim : latent_dim + input_dim].T.copy()
        b = coef[-1].copy()
        A, st = stabilize_matrix(A, spectral_cap)
        if preserve_mean:
            # Stabilising A changes its stationary mean if the fitted intercept
            # is retained.  Anchor the intercept to the training latent mean so
            # the autonomous rollout does not acquire an artificial drift.
            mu = Z[train_mask].mean(axis=0)
            b = mu - A @ mu
        residuals = target - (Z[starts] @ A.T + U[starts] @ B.T + b)
        process_cov = np.asarray(LedoitWolf().fit(residuals).covariance_, dtype=np.float64)
        process_cov = 0.5 * (process_cov + process_cov.T) + 1e-10 * np.eye(latent_dim)

        reconstruction = pca.inverse_transform(Z)
        obs_residual = X[train_mask] - reconstruction[train_mask]
        obs_cov = np.diag(np.var(obs_residual, axis=0) + 1e-8)

        probe = BehaviorProbe(seed=int(seed))
        probe.fit(X[train_mask], y[train_mask])
        initial_cov = np.diag(np.var(Z[train_mask], axis=0) + 1e-8)
        one_step = np.sqrt(np.mean(residuals**2))
        return cls(
            pca=pca,
            A=A,
            b=b,
            B=B,
            process_cov=process_cov,
            obs_cov=obs_cov,
            initial_cov=initial_cov,
            probe=probe,
            n_neurons=int(X.shape[1]),
            latent_dim=int(latent_dim),
            input_dim=input_dim,
            input_mean=input_mean,
            input_std=input_std,
            spectral_cap=float(spectral_cap),
            stability=dict(st.__dict__),
            diagnostics={
                "n_train_frames": int(train_mask.sum()),
                "n_transitions": int(len(starts)),
                "latent_ridge": float(ridge),
                "latent_train_rmse": float(one_step),
                "explained_variance_ratio": float(np.sum(pca.explained_variance_ratio_)),
                "input_identifiable": bool(input_dim > 0),
                "preserve_training_mean": bool(preserve_mean),
            },
        )

    # ------------------------------------------------------------------ encoding
    def encode(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr[None]
        if arr.ndim != 2 or arr.shape[1] != self.n_neurons:
            raise ValueError(f"expected (T, {self.n_neurons}) observations")
        z = self.pca.transform(arr)
        return z[0] if squeeze else z

    def transform(self, X: np.ndarray) -> np.ndarray:
        """PCAEncoder-compatible alias used by branching diagnostics."""
        return self.encode(X)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        arr = np.asarray(Z, dtype=np.float64)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr[None]
        x = self.pca.inverse_transform(arr)
        return x[0] if squeeze else x

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Retrospective reconstruction; not a forecast."""
        return self.decode(self.encode(X))

    def _normalise_inputs(self, inputs: np.ndarray | None, length: int) -> np.ndarray:
        if self.input_dim == 0:
            if inputs is not None and np.asarray(inputs).size:
                raise ValueError("this model has no fitted input channel")
            return np.zeros((length, 0), dtype=np.float64)
        if inputs is None:
            return np.zeros((length, self.input_dim), dtype=np.float64)
        U = np.asarray(inputs, dtype=np.float64)
        if U.ndim == 1:
            U = U[:, None]
        if U.shape != (length, self.input_dim):
            raise ValueError(f"inputs must have shape ({length}, {self.input_dim}), got {U.shape}")
        return (U - self.input_mean) / self.input_std

    # ------------------------------------------------------------------ filtering
    @property
    def observation_matrix(self) -> np.ndarray:
        # sklearn PCA stores (latent, neuron); the observation equation is
        # neuron = D latent + mean, so D must be (neuron, latent).
        return np.asarray(self.pca.components_, dtype=np.float64).T

    @property
    def observation_mean(self) -> np.ndarray:
        return np.asarray(self.pca.mean_, dtype=np.float64)

    def filter(
        self,
        X: np.ndarray,
        *,
        inputs: np.ndarray | None = None,
        initial_cov: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Causal posterior means/covariances given observations up to each t."""
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.n_neurons or len(arr) == 0:
            raise ValueError("filter expects a non-empty (T, n_neurons) array")
        U = self._normalise_inputs(inputs, len(arr))
        D = self.observation_matrix
        R = np.asarray(self.obs_cov, dtype=np.float64)
        P = (
            np.asarray(self.initial_cov, dtype=np.float64).copy()
            if initial_cov is None
            else np.asarray(initial_cov, dtype=np.float64).copy()
        )
        z = self.encode(arr[0])
        Z = np.zeros((len(arr), self.latent_dim), dtype=np.float64)
        covs = np.zeros((len(arr), self.latent_dim, self.latent_dim), dtype=np.float64)
        Z[0] = z
        covs[0] = P
        eye = np.eye(self.latent_dim)
        for t in range(1, len(arr)):
            z_pred = self._transition(Z[t - 1], U[t - 1])
            P_pred = self._transition_matrix(Z[t - 1], U[t - 1]) @ P @ self._transition_matrix(Z[t - 1], U[t - 1]).T + self.process_cov
            innovation = arr[t] - (D @ z_pred + self.observation_mean)
            S = D @ P_pred @ D.T + R
            S = 0.5 * (S + S.T) + 1e-12 * np.eye(len(S))
            K = P_pred @ D.T @ np.linalg.solve(S, np.eye(len(S)))
            Z[t] = z_pred + K @ innovation
            P = (eye - K @ D) @ P_pred
            P = 0.5 * (P + P.T)
            covs[t] = P
        return {"latent": Z, "covariance": covs, "inputs_used": U}

    def latent(self, history: np.ndarray) -> np.ndarray | None:
        """Posterior latent path for the supplied observed history."""
        return self.filter(history)["latent"]

    # ------------------------------------------------------------------ prediction
    def _transition(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        """One latent transition; subclasses may make this context dependent."""
        return self.A @ z + self.b + self.B @ u

    def _transition_matrix(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Effective linear map at the current context (for covariance propagation)."""
        return self.A

    def _coefficients(self, z: np.ndarray, u: np.ndarray) -> np.ndarray | None:
        """Operator coefficients for diagnostics; fixed LDS has none."""
        return None

    def predict_next(
        self,
        history: np.ndarray,
        *,
        inputs: np.ndarray | None = None,
    ) -> np.ndarray:
        arr = np.asarray(history, dtype=np.float64)
        if arr.ndim == 3:
            if arr.shape[2] != self.n_neurons:
                raise ValueError("batched history must be (batch, time, n_neurons)")
            return np.stack([self.predict_next(h, inputs=inputs) for h in arr], axis=0)
        if arr.ndim == 1:
            arr = arr[None]
        if arr.ndim != 2 or arr.shape[1] != self.n_neurons:
            raise ValueError("history must be (T, n_neurons)")
        if len(arr) == 0:
            raise ValueError("history is empty")
        z = self.encode(arr[-1])
        if inputs is None:
            u = np.zeros((1, self.input_dim), dtype=np.float64)
        else:
            u = self._normalise_inputs(np.asarray(inputs).reshape(1, -1), 1)
        return self.decode(self._transition(z, u[0]))

    def behavior_proba(self, history: np.ndarray) -> np.ndarray:
        arr = np.asarray(history, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr[:, -1, :]
        elif arr.ndim == 1:
            arr = arr[None]
        return self.probe.predict_proba(arr)

    # ------------------------------------------------------------------ rollout
    def rollout(
        self,
        x0: np.ndarray,
        steps: int,
        *,
        seed: int = 0,
        noise_std: float = 0.0,
        perturb=None,
        initial_history: np.ndarray | None = None,
        stochastic: bool = False,
        inputs: np.ndarray | None = None,
    ) -> np.ndarray:
        if steps < 1:
            raise ValueError("steps must be at least 1")
        if noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        U = self._normalise_inputs(inputs, int(steps))
        if initial_history is not None:
            hist = np.asarray(initial_history, dtype=np.float64)
            if hist.ndim != 2 or hist.shape[1] != self.n_neurons:
                raise ValueError("initial_history must be (history, n_neurons)")
            z = self.filter(hist)["latent"][-1].copy()
            x = hist[-1].copy()
        else:
            x = np.asarray(x0, dtype=np.float64).reshape(self.n_neurons).copy()
            z = self.encode(x)
        states = np.empty((steps + 1, self.n_neurons), dtype=np.float64)
        states[0] = x
        rng = np.random.default_rng(int(seed))
        for s in range(steps):
            z_pred = self._transition(z, U[s])
            if stochastic or noise_std > 0:
                z_pred = z_pred + rng.multivariate_normal(
                    np.zeros(self.latent_dim),
                    self.process_cov + (noise_std**2) * np.eye(self.latent_dim),
                )
            x_next = self.decode(z_pred)
            if perturb is not None:
                x_next = np.asarray(perturb(x_next, s), dtype=np.float64)
            if x_next.shape != (self.n_neurons,):
                raise ValueError("perturb returned the wrong state shape")
            states[s + 1] = x_next
            # The intervention is part of the state at the next causal step.
            z = self.encode(x_next)
        return states

    def input_vector_for(self, frame: int, steps: int) -> np.ndarray:
        raise NotImplementedError("input schedules are supplied by the evaluation adapter")


@dataclass
class CompositionalLatentLDS(StableLatentLDS):
    """Compact compositional latent model inspired by dLDS, not its reproduction.

    The transition is a convex combination of ``M`` stable linear operators,
    ``z_{t+1} = sum_m c_{m,t} F_m z_t + b + B u_t``.  Coefficients are generated
    causally from the current latent state and the current known input through a
    fitted linear policy; no future coefficient or future observation is used.
    The model is fitted by alternating non-negative coefficient estimation and
    weighted operator regression followed by exact operator stabilisation.
    """

    F: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    coefficient_weights: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    operator_names: tuple[str, ...] = ()
    coefficient_diagnostics: dict[str, Any] = field(default_factory=dict)
    n_operators: int = 0
    name: str = "compositional_lds"

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        starts: np.ndarray,
        *,
        train_mask: np.ndarray,
        n_operators: int = 3,
        operator_ridge: float = 1e-2,
        coefficient_ridge: float = 1e-2,
        alternations: int = 5,
        seed: int = 17,
        **base_kwargs: Any,
    ) -> CompositionalLatentLDS:
        if n_operators < 1:
            raise ValueError("n_operators must be >= 1")
        base = StableLatentLDS.fit(
            X,
            y,
            starts,
            train_mask=train_mask,
            seed=seed,
            **base_kwargs,
        )
        p = base.latent_dim
        Z = base.encode(X)
        U = base._normalise_inputs(
            base_kwargs.get("input_values"), len(X)
        )
        zs = Z[starts]
        target = Z[starts + 1] - base.b
        rng = np.random.default_rng(int(seed))
        F = np.empty((n_operators, p, p), dtype=np.float64)
        F[0] = base.A
        for m in range(1, n_operators):
            F[m], _ = stabilize_matrix(
                base.A + 0.15 * rng.normal(size=(p, p)) / np.sqrt(max(1, p)),
                float(base.spectral_cap),
            )
        F[0], _ = stabilize_matrix(F[0], float(base.spectral_cap))

        def estimate_coefficients(ops: np.ndarray) -> np.ndarray:
            C = np.zeros((len(starts), len(ops)), dtype=np.float64)
            aug = np.sqrt(max(1e-8, coefficient_ridge)) * np.eye(len(ops))
            for i in range(len(starts)):
                G = np.column_stack([ops[m] @ zs[i] for m in range(len(ops))])
                sol, _ = nnls(
                    np.vstack([G, aug]),
                    np.r_[target[i], np.zeros(len(ops))],
                )
                total = float(sol.sum())
                C[i] = sol / total if total > 1e-12 else np.full(len(ops), 1.0 / len(ops))
            return C

        C = np.full((len(starts), n_operators), 1.0 / n_operators)
        for _ in range(max(1, int(alternations))):
            C = estimate_coefficients(F)
            for m in range(n_operators):
                w = np.sqrt(np.maximum(C[:, m], 1e-8))
                Zw = zs * w[:, None]
                Yw = target * w[:, None]
                F[m] = _ridge_solve(Zw, Yw, operator_ridge)
                F[m], _ = stabilize_matrix(F[m], float(base.spectral_cap))
        C = estimate_coefficients(F)

        phi = np.column_stack([zs, U[starts], np.ones(len(starts))])
        weights = _ridge_solve(phi, C, coefficient_ridge)
        policy_c = np.maximum(phi @ weights, 0.0)
        denom = policy_c.sum(axis=1, keepdims=True)
        policy_c = np.divide(
            policy_c,
            denom,
            out=np.full_like(policy_c, 1.0 / n_operators),
            where=denom > 1e-12,
        )
        policy_rmse = float(np.sqrt(np.mean((policy_c - C) ** 2)))

        # Re-estimate process noise under the realized operator mixture.
        effective = np.einsum("nm,mpq->npq", C, F)
        pred = np.einsum("npq,nq->np", effective, zs) + base.b + U[starts] @ base.B.T
        residuals = Z[starts + 1] - pred
        process_cov = np.asarray(LedoitWolf().fit(residuals).covariance_, dtype=np.float64)
        process_cov = 0.5 * (process_cov + process_cov.T) + 1e-10 * np.eye(p)

        entropy = -np.sum(C * np.log(np.maximum(C, 1e-12)), axis=1)
        radii = [float(np.max(np.abs(np.linalg.eigvals(F[m])))) for m in range(n_operators)]
        diag = {
            "n_operators": int(n_operators),
            "operator_spectral_radii": radii,
            "max_operator_radius": float(max(radii)),
            "mean_coefficients": C.mean(axis=0).tolist(),
            "sd_coefficients": C.std(axis=0).tolist(),
            "mean_coefficient_entropy": float(entropy.mean()),
            "max_coefficient_entropy": float(np.log(n_operators)),
            "coefficient_policy_rmse": policy_rmse,
            "n_transitions": int(len(starts)),
            "alternations": int(alternations),
            "identification_warning": (
                "Operator signs/ordering are not identifiable; coefficients are model-internal, "
                "not biological states. Policy is fitted to retrospective coefficients and can collapse."
            ),
            "reference_relation": (
                "Inspired by the dLDS form but independently implemented; not a reproduction of "
                "Yezerets et al. (2025) and not evidence for synaptic mechanisms."
            ),
        }
        payload = dict(base.__dict__)
        payload.pop("name", None)
        payload.pop("process_cov", None)
        return cls(
            **payload,
            F=F,
            coefficient_weights=weights,
            operator_names=tuple(f"F{m + 1}" for m in range(n_operators)),
            coefficient_diagnostics=diag,
            n_operators=int(n_operators),
            process_cov=process_cov,
            name="compositional_lds",
        )

    def coefficients_for(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        phi = np.r_[np.asarray(z, dtype=np.float64), np.asarray(u, dtype=np.float64), 1.0]
        c = np.maximum(self.coefficient_weights.T @ phi, 0.0)
        total = float(c.sum())
        return c / total if total > 1e-12 else np.full(self.n_operators, 1.0 / self.n_operators)

    def _coefficients(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.coefficients_for(z, u)

    def _transition_matrix(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        c = self.coefficients_for(z, u)
        return np.tensordot(c, self.F, axes=(0, 0))

    def _transition(self, z: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self._transition_matrix(z, u) @ z + self.b + self.B @ u


@dataclass
class SharedLatentModel(StableLatentLDS):
    """A shared transition expressed through a session-specific observation map.

    ``D`` maps the shared latent coordinate to the session's neurons and ``R``
    maps the session's calibration PCA coordinate into the shared coordinate.
    ``R=I`` is the zero-shot case.  The class is intentionally small: it does not
    claim that independently fitted PCA bases are biologically aligned.
    """

    D: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    R: np.ndarray = field(default_factory=lambda: np.eye(1))
    name: str = "shared_latent_adapted"

    def encode(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr[None]
        if arr.ndim != 2 or arr.shape[1] != self.n_neurons:
            raise ValueError(f"expected (T, {self.n_neurons}) observations")
        z = (arr - self.mean) @ self.D @ self.R
        return z[0] if squeeze else z

    def decode(self, Z: np.ndarray) -> np.ndarray:
        arr = np.asarray(Z, dtype=np.float64)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr[None]
        base = arr @ np.linalg.pinv(self.R)
        x = self.mean + base @ self.D.T
        return x[0] if squeeze else x

    @property
    def observation_matrix(self) -> np.ndarray:
        return self.D @ np.linalg.pinv(self.R)

    @property
    def observation_mean(self) -> np.ndarray:
        return self.mean

    @classmethod
    def from_shared(
        cls,
        *,
        A: np.ndarray,
        b: np.ndarray,
        process_cov: np.ndarray,
        D: np.ndarray,
        mean: np.ndarray,
        R: np.ndarray,
        probe: BehaviorProbe,
        n_neurons: int,
        latent_dim: int,
        obs_cov: np.ndarray,
        initial_cov: np.ndarray,
        spectral_cap: float,
        name: str = "shared_latent_adapted",
        metadata: dict[str, Any] | None = None,
    ) -> SharedLatentModel:
        return cls(
            pca=None,  # type: ignore[arg-type]
            A=A,
            b=b,
            B=np.zeros((latent_dim, 0)),
            process_cov=process_cov,
            obs_cov=obs_cov,
            initial_cov=initial_cov,
            probe=probe,
            n_neurons=int(n_neurons),
            latent_dim=int(latent_dim),
            input_dim=0,
            spectral_cap=float(spectral_cap),
            stability={},
            diagnostics=dict(metadata or {}),
            D=np.asarray(D, dtype=np.float64),
            mean=np.asarray(mean, dtype=np.float64),
            R=np.asarray(R, dtype=np.float64),
            name=name,
            supports_input=False,
            native_stochastic=False,
        )
