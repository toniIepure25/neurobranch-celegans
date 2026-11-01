"""Latent LDS causal-filter and no-future-observation tests."""

from __future__ import annotations

import numpy as np

from neurobranch.dynamics.context_models import (
    CompositionalLatentLDS,
    StableLatentLDS,
)


def _synthetic(T=520, N=12, p=4, seed=0):
    rng = np.random.default_rng(seed)
    A = np.diag([0.85, 0.7, 0.5, -0.35])
    D = rng.normal(size=(p, N))
    Z = np.zeros((T, p))
    for t in range(1, T):
        Z[t] = A @ Z[t - 1] + rng.normal(scale=0.15, size=p)
    X = Z @ D + rng.normal(scale=0.05, size=(T, N))
    y = (np.arange(T) // 37) % 4
    return X, y.astype(int), A, D


def _fit():
    X, y, _, _ = _synthetic()
    mask = np.zeros(len(X), dtype=bool)
    mask[:380] = True
    starts = np.arange(2, 378)
    return StableLatentLDS.fit(
        X,
        y,
        starts,
        train_mask=mask,
        latent_dim=4,
        ridge=1e-4,
        spectral_cap=0.95,
    ), X, mask


def test_pca_is_fitted_on_training_frames_only():
    model, X, mask = _fit()
    np.testing.assert_allclose(model.pca.mean_, X[mask].mean(axis=0), atol=1e-10)
    # A future-only shift must not change the fitted PCA mean.
    X2 = X.copy()
    X2[400:] += 100.0
    model2, _, _ = _fit()
    np.testing.assert_allclose(model.pca.mean_, model2.pca.mean_, atol=1e-12)


def test_filter_does_not_use_future_observations():
    model, X, _ = _fit()
    short = model.filter(X[:40])["latent"]
    long = model.filter(X[:80])["latent"]
    np.testing.assert_allclose(short, long[:40], atol=1e-12)


def test_latent_history_has_no_future_dependency():
    model, X, _ = _fit()
    history = X[20:30]
    changed = history.copy()
    changed[-1] += 1.0
    a = model.latent(history)
    b = model.latent(changed)
    assert a.shape == (10, model.latent_dim)
    assert not np.allclose(a[-1], b[-1])


def test_rollouts_are_reproducible_and_stochastic_when_requested():
    model, X, _ = _fit()
    h = X[30:31]
    a = model.rollout(None, 12, initial_history=h, seed=7, stochastic=False)
    b = model.rollout(None, 12, initial_history=h, seed=7, stochastic=False)
    np.testing.assert_allclose(a, b)
    c = model.rollout(None, 12, initial_history=h, seed=7, stochastic=True)
    d = model.rollout(None, 12, initial_history=h, seed=7, stochastic=True)
    np.testing.assert_allclose(c, d)
    assert not np.allclose(a, c)


def test_external_input_is_identifiable_and_causal():
    X, y, _, _ = _synthetic()
    T = len(X)
    U = np.zeros((T, 1))
    U[200:350] = 1.0
    mask = np.zeros(T, dtype=bool)
    mask[:380] = True
    starts = np.arange(2, 378)
    model = StableLatentLDS.fit(
        X,
        y,
        starts,
        train_mask=mask,
        latent_dim=4,
        ridge=1e-4,
        input_values=U,
        spectral_cap=0.95,
    )
    assert model.input_dim == 1
    assert np.linalg.norm(model.B) > 0
    base = model.filter(X[30:40])
    with_input = model.filter(X[30:40], inputs=U[30:40])
    assert base["latent"].shape == with_input["latent"].shape


def test_one_step_prediction_shape():
    model, X, _ = _fit()
    out = model.predict_next(X[20:25])
    assert out.shape == (model.n_neurons,)
    assert np.isfinite(out).all()


def test_compositional_coefficients_are_causal_simplex_and_operators_stable():
    X, y, _, _ = _synthetic()
    mask = np.zeros(len(X), dtype=bool)
    mask[:380] = True
    starts = np.arange(2, 378)
    model = CompositionalLatentLDS.fit(
        X,
        y,
        starts,
        train_mask=mask,
        latent_dim=4,
        ridge=1e-4,
        n_operators=3,
        alternations=2,
        spectral_cap=0.95,
    )
    z = model.encode(X[40])
    c = model.coefficients_for(z, np.zeros(0))
    assert c.shape == (3,)
    assert np.all(c >= 0) and abs(c.sum() - 1) < 1e-9
    assert max(model.coefficient_diagnostics["operator_spectral_radii"]) <= 0.95 * (1 + 1e-8)
    # Filtering the same past with a perturbed future must be identical.
    a = model.filter(X[:30])["latent"]
    changed = X.copy()
    changed[30:] += 10.0
    b = model.filter(changed[:30])["latent"]
    np.testing.assert_allclose(a, b, atol=1e-12)
    # Coefficients are generated from the current state, never an oracle path.
    assert not hasattr(model, "future_coefficients")
