"""Regression tests for companion-matrix stabilisation."""

from __future__ import annotations

import numpy as np

from neurobranch.dynamics.stability import (
    companion_matrix,
    spectral_radius,
    stabilize_companion,
    stabilize_matrix,
)
from neurobranch.dynamics.stable_var import StableVARModel


def _lag2_system(n=2, a1=None, a2=None, T=1200, seed=0):
    rng = np.random.default_rng(seed)
    if a1 is None:
        a1 = np.array([[1.12, 0.05], [-0.03, 0.98]])
    if a2 is None:
        a2 = np.diag([-0.18, -0.12])
    X = np.zeros((T, n))
    for t in range(2, T):
        X[t] = a1 @ X[t - 1] + a2 @ X[t - 2] + rng.normal(scale=0.05, size=n)
    return X


def test_companion_matrix_has_shift_block():
    A = np.array([[0.4, 0.1, 0.2, -0.1], [0.0, 0.5, 0.2, 0.3]])
    C = companion_matrix(A, n_neurons=2, history=2)
    assert C.shape == (4, 4)
    np.testing.assert_allclose(C[:2, :], A)
    np.testing.assert_allclose(C[2:, :2], np.eye(2))
    np.testing.assert_allclose(C[2:, 2:], 0.0)


def test_iterative_stabilisation_respects_cap_for_lag_two():
    X = _lag2_system()
    starts = np.arange(2, len(X) - 1)
    model = StableVARModel.fit(
        X,
        (np.arange(len(X)) % 3).astype(int),
        starts,
        history=2,
        ridge=1e-6,
        spectral_cap=0.95,
    )
    assert model.realized_spectral_radius <= 0.95 * (1 + 1e-8)
    assert model.stabilization["initial_radius"] > 0.95
    assert model.stabilization["converged"]


def test_stable_system_is_not_modified():
    A = np.array([[0.2, 0.0, 0.1, 0.0], [0.0, 0.3, 0.0, 0.1]])
    out, result = stabilize_companion(A, 2, 2, 0.995)
    np.testing.assert_allclose(out, A)
    assert result.iterations == 0
    assert result.realized_radius == result.initial_radius


def test_latent_matrix_stabilisation():
    A = np.array([[1.1, 0.2], [0.0, 0.99]])
    out, result = stabilize_matrix(A, 0.95)
    assert result.realized_radius <= 0.95 * (1 + 1e-8)
    assert np.max(np.abs(np.linalg.eigvals(out))) <= 0.95 * (1 + 1e-8)


def test_spectral_radius_is_finite_for_noise_driven_system():
    assert np.isfinite(spectral_radius(np.zeros((2, 4)), 2, 2))


def test_old_frozen_checkpoint_style_one_shot_radius_can_exceed_cap():
    # This is the documented failure mode, not a target behavior.
    A = np.array([[0.99, 0.2, 0.0, 0.0], [0.0, 0.99, 0.0, 0.0]])
    C = companion_matrix(A, 2, 2)
    rho = float(np.max(np.abs(np.linalg.eigvals(C))))
    scaled = A * (0.995 / rho)
    assert spectral_radius(scaled, 2, 2) != 0.995
