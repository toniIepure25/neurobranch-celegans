"""Small stochastic components for the joint-dynamics experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurobranch.probabilistic.residual import center_residuals, regularized_covariance


@dataclass
class LowRankTemporalInnovation:
    """Low-rank AR(1) residual state plus isotropic residual noise."""

    factors: np.ndarray
    phi: np.ndarray
    eta_scale: np.ndarray
    diag_scale: np.ndarray
    metadata: dict

    @classmethod
    def fit(cls, residuals: np.ndarray, *, rank: int = 8, phi_cap: float = 0.95) -> LowRankTemporalInnovation:
        centered, _ = center_residuals(residuals)
        cov, meta = regularized_covariance(centered, method="ledoit")
        eigval, eigvec = np.linalg.eigh(cov)
        order = np.argsort(eigval)[::-1]
        rank = int(min(rank, centered.shape[1], len(centered) // 10))
        factors = eigvec[:, order[:rank]]
        scores = centered @ factors
        prev, nxt = scores[:-1], scores[1:]
        denom = np.sum(prev**2, axis=0) + 1e-8
        phi = np.clip(np.sum(prev * nxt, axis=0) / denom, -phi_cap, phi_cap)
        eta = nxt - prev * phi
        eta_scale = np.sqrt(np.maximum(eta.var(axis=0, ddof=1), 1e-10))
        reconstruction = scores @ factors.T
        diag_scale = np.sqrt(np.maximum((centered - reconstruction).var(axis=0, ddof=1), 1e-10))
        return cls(
            factors=factors,
            phi=phi,
            eta_scale=eta_scale,
            diag_scale=diag_scale,
            metadata={
                **meta,
                "rank": rank,
                "phi_median": float(np.median(phi)),
                "phi_abs_max": float(np.max(np.abs(phi))),
                "captured_variance": float(eigval[order[:rank]].sum() / max(eigval.sum(), 1e-12)),
            },
        )

    def sample(self, n_members: int, horizon: int, rng: np.random.Generator) -> np.ndarray:
        n_members, horizon = int(n_members), int(horizon)
        z = np.empty((n_members, horizon, self.factors.shape[1]), dtype=np.float64)
        # Stationary initialization, not a zero transient.
        stationary = self.eta_scale / np.sqrt(np.maximum(1.0 - self.phi**2, 1e-3))
        z[:, 0] = rng.standard_normal((n_members, self.factors.shape[1])) * stationary
        for step in range(1, horizon):
            z[:, step] = z[:, step - 1] * self.phi + rng.standard_normal(z[:, step].shape) * self.eta_scale
        eps = z @ self.factors.T
        eps += rng.standard_normal((n_members, horizon, self.diag_scale.size)) * self.diag_scale
        return eps


def sample_student_t(
    n_members: int,
    horizon: int,
    covariance: np.ndarray,
    rng: np.random.Generator,
    *,
    df: float = 5.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Multivariate Student-t innovations with covariance ``scale**2 * cov``."""
    cov = np.asarray(covariance, dtype=np.float64)
    eigval, eigvec = np.linalg.eigh((cov + cov.T) / 2)
    root = eigvec @ np.diag(np.sqrt(np.clip(eigval, 1e-10, None)))
    z = rng.standard_normal((int(n_members) * int(horizon), cov.shape[0])) @ root.T
    chi = rng.chisquare(float(df), size=(int(n_members) * int(horizon), 1))
    eps = z * np.sqrt(float(df) / chi) * float(scale)
    # A multivariate t with df>2 has covariance df/(df-2) * cov; normalize it.
    eps *= np.sqrt(max(float(df) - 2.0, 1e-6) / float(df))
    return eps.reshape(int(n_members), int(horizon), -1)


def block_bootstrap_starts(starts: np.ndarray, *, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block resample of causal transition indices."""
    starts = np.asarray(starts, dtype=np.int64)
    block = max(1, min(int(block), len(starts)))
    n_blocks = int(np.ceil(len(starts) / block))
    pieces = []
    for _ in range(n_blocks):
        lo = int(rng.integers(0, len(starts) - block + 1))
        pieces.append(starts[lo : lo + block])
    return np.concatenate(pieces)[: len(starts)]
