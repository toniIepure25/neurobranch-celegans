"""Analytical and Monte Carlo uncertainty propagation for linear dynamics."""

from __future__ import annotations

import numpy as np


def forecast_covariance(A: np.ndarray, Q: np.ndarray, horizon: int) -> np.ndarray:
    """Covariance of x_{t+h} for independent zero-mean innovations.

    Returns an array of shape ``(horizon + 1, d, d)``; index 0 is the known
    initial state covariance (zero).
    """
    A = np.asarray(A, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    horizon = int(horizon)
    d = A.shape[0]
    out = np.zeros((horizon + 1, d, d), dtype=np.float64)
    P = np.zeros((d, d), dtype=np.float64)
    for h in range(1, horizon + 1):
        P = A @ P @ A.T + Q
        out[h] = (P + P.T) / 2.0
    return out


def monte_carlo_covariance(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    Q: np.ndarray,
    horizon: int,
    n_members: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a Gaussian VAR and return sample means and covariances.

    Shapes are ``(horizon, d)`` and ``(horizon, d, d)``.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x0 = np.asarray(x0, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    horizon = int(horizon)
    n_members = int(n_members)
    chol = np.linalg.cholesky((Q + Q.T) / 2.0 + 1e-12 * np.eye(Q.shape[0]))
    x = np.repeat(x0[None, :], n_members, axis=0)
    values = np.empty((n_members, horizon, x0.size), dtype=np.float64)
    for step in range(horizon):
        eps = rng.standard_normal((n_members, x0.size)) @ chol.T
        x = x @ A.T + b + eps
        values[:, step] = x
    means = values.mean(axis=0)
    covs = np.empty((horizon, x0.size, x0.size), dtype=np.float64)
    for step in range(horizon):
        covs[step] = np.cov(values[:, step, :], rowvar=False, ddof=1)
    return means, covs


def transient_amplification(A: np.ndarray, horizon: int) -> dict:
    """Singular-value growth and non-normality of a transition operator."""
    A = np.asarray(A, dtype=np.float64)
    horizon = int(horizon)
    radius = float(np.max(np.abs(np.linalg.eigvals(A))))
    power = np.eye(A.shape[0])
    norms = []
    for _ in range(horizon):
        power = A @ power
        norms.append(float(np.linalg.norm(power, 2)))
    commute = A @ A.T - A.T @ A
    denom = float(np.linalg.norm(A @ A.T) + np.linalg.norm(A.T @ A))
    return {
        "spectral_radius": radius,
        "max_singular_value_h": float(np.max(norms)),
        "max_amplification_over_radius": float(
            np.max(norms) / max(radius ** np.arange(1, horizon + 1))
        ),
        "nonnormality": float(np.linalg.norm(commute) / denom) if denom > 0 else float("nan"),
        "final_power_norm": norms[-1],
        "radius_to_horizon": radius**horizon,
    }


def nonnormality(A: np.ndarray) -> float:
    A = np.asarray(A, dtype=np.float64)
    commute = A @ A.T - A.T @ A
    denom = float(np.linalg.norm(A @ A.T) + np.linalg.norm(A.T @ A))
    return float(np.linalg.norm(commute) / denom) if denom > 0 else float("nan")


def covariance_trace_error(predicted: np.ndarray, observed: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    denom = float(np.trace(observed))
    return float(np.trace(predicted) / denom) if denom > 0 else float("nan")
