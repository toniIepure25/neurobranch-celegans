from __future__ import annotations

import numpy as np

from neurobranch.simulator.latent import (
    BunDLeEncoder,
    PCAEncoder,
    latent_comparison,
    linear_cka,
    path_geometry,
)


def test_pca_encoder_is_train_fitted_and_invertible(toy_data):
    enc = PCAEncoder.fit(toy_data, n_components=3)
    Z = enc.transform(toy_data.X[:20])
    X = enc.inverse(Z)
    assert Z.shape == (20, 3)
    assert X.shape == (20, toy_data.n_neurons)
    assert np.mean((X - toy_data.X[:20]) ** 2) < 0.9 * np.var(toy_data.X[:20])


def test_bundlenet_encoder_is_train_only_and_shaped(toy_data):
    enc = BunDLeEncoder.fit(toy_data, epochs=2, seed=17)
    Z = enc.transform(toy_data.X[:40])
    assert Z.shape == (40, 3)
    assert np.isfinite(Z).all()
    assert enc.metadata["train_only"] is True
    assert enc.metadata["causal_lowpass"] is True


def test_latent_comparison_reports_geometry():
    rng = np.random.default_rng(0)
    real = rng.normal(size=(21, 8))
    sim = real + rng.normal(scale=0.1, size=real.shape)
    enc = type("E", (), {"transform": lambda self, X: np.asarray(X)[:, :3]})()
    out = latent_comparison(real, sim, enc)
    assert np.isfinite(out["latent_energy_distance"])
    assert out["real_path_length"] > 0
    assert out["path_length_ratio"] > 0


def test_path_geometry_handles_short_sequence():
    out = path_geometry(np.zeros((2, 3)))
    assert np.isnan(out["curvature"])


def test_cka_identity_and_rotation():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3))
    assert np.isclose(linear_cka(X, X), 1.0)
    Y = X @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    assert linear_cka(X, Y) > 0.99
