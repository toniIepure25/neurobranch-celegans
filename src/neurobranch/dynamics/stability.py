"""Numerical stability utilities for autoregressive and latent dynamics.

The Phase 6 audit found that scaling the coefficient block of a lag-p VAR does
not scale every companion-matrix eigenvalue by the same factor.  The helpers
here operate on the companion matrix itself and verify the realized radius after
every transformation.  They are additive: no frozen Phase 6 checkpoint is
modified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StabilityResult:
    """Diagnostics for one stabilisation attempt."""

    requested_cap: float
    initial_radius: float
    realized_radius: float
    iterations: int
    tolerance: float
    converged: bool

    @property
    def exceeded(self) -> bool:
        return self.realized_radius > self.requested_cap * (1.0 + self.tolerance)


def companion_matrix(A: np.ndarray, n_neurons: int, history: int) -> np.ndarray:
    """Companion matrix for ``x_t = A [x_{t-1}; ...; x_{t-p}] + b``.

    ``A`` has shape ``(n_neurons, n_neurons * history)``.  The state ordering is
    ``[x_t, x_{t-1}, ..., x_{t-p+1}]``.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape != (n_neurons, n_neurons * history):
        raise ValueError(
            f"A must have shape ({n_neurons}, {n_neurons * history}), got {A.shape}"
        )
    if history < 1:
        raise ValueError("history must be >= 1")
    size = n_neurons * history
    C = np.zeros((size, size), dtype=np.float64)
    C[:n_neurons, :] = A
    if history > 1:
        C[n_neurons:, :-n_neurons] = np.eye(n_neurons * (history - 1))
    return C


def spectral_radius(A: np.ndarray, n_neurons: int, history: int) -> float:
    """Largest absolute eigenvalue of the companion matrix."""
    return float(np.max(np.abs(np.linalg.eigvals(companion_matrix(A, n_neurons, history)))))


def stabilize_companion(
    A: np.ndarray,
    n_neurons: int,
    history: int,
    cap: float,
    *,
    tolerance: float = 1e-10,
    max_iter: int = 100,
) -> tuple[np.ndarray, StabilityResult]:
    """Scale the AR coefficient block until the companion radius respects ``cap``.

    A one-shot ``A *= cap / rho`` is not exact for ``history > 1`` because the
    companion matrix also contains an identity shift block.  This routine
    iterates the scaling and verifies the radius of the actual companion matrix.
    It never rescales the identity block, so the lag structure is preserved.
    """
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("cap must be finite and positive")
    A = np.asarray(A, dtype=np.float64).copy()
    rho0 = spectral_radius(A, n_neurons, history)
    if not np.isfinite(rho0):
        raise ValueError("companion spectral radius is not finite")
    if rho0 <= cap:
        return A, StabilityResult(cap, rho0, rho0, 0, tolerance, True)

    scale = 1.0
    rho = rho0
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):  # noqa: B007 - used in result
        # The coefficient block is linear in the scale.  Iterate because the
        # identity block is fixed and the radius is not exactly linear in scale.
        scale *= cap / rho
        A_scaled = A * scale
        rho = spectral_radius(A_scaled, n_neurons, history)
        if rho <= cap * (1.0 + tolerance):
            A = A_scaled
            converged = True
            break
        A = A_scaled
    if not converged:
        # Conservative final fallback.  It may shrink slightly more than needed,
        # but it guarantees the requested cap rather than silently exceeding it.
        A *= cap / rho
        rho = spectral_radius(A, n_neurons, history)
        converged = rho <= cap * (1.0 + tolerance)
    return A, StabilityResult(cap, rho0, rho, iterations, tolerance, converged)


def stabilize_matrix(
    A: np.ndarray,
    cap: float,
    *,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, StabilityResult]:
    """Scale a square latent transition so its spectral radius respects ``cap``."""
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("cap must be finite and positive")
    A = np.asarray(A, dtype=np.float64).copy()
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("latent transition must be square")
    rho0 = float(np.max(np.abs(np.linalg.eigvals(A))))
    if rho0 <= cap:
        return A, StabilityResult(cap, rho0, rho0, 0, tolerance, True)
    A *= cap / rho0
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    return A, StabilityResult(cap, rho0, rho, 1, tolerance, rho <= cap * (1 + tolerance))
