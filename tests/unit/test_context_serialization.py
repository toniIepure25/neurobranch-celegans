"""Round-trip serialization for the new context models."""

from __future__ import annotations

import numpy as np

from neurobranch.dynamics.checkpointing import load_model, save_model
from neurobranch.dynamics.context_models import (
    CompositionalLatentLDS,
    SharedLatentModel,
    StableLatentLDS,
)
from neurobranch.dynamics.models import BehaviorProbe
from neurobranch.dynamics.stable_var import StableVARModel


def _data(T=240, N=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, N))
    y = (np.arange(T) // 20) % 4
    starts = np.arange(2, 198)
    mask = np.zeros(T, dtype=bool)
    mask[:200] = True
    return X, y, starts, mask


def test_latent_and_compositional_roundtrip(tmp_path):
    X, y, starts, mask = _data()
    latent = StableLatentLDS.fit(X, y, starts, train_mask=mask, latent_dim=4)
    comp = CompositionalLatentLDS.fit(
        X, y, starts, train_mask=mask, latent_dim=4, n_operators=2, alternations=1
    )
    for model, name in [(latent, "latent"), (comp, "comp")]:
        path = save_model(model, tmp_path / f"{name}.pt")
        loaded, _ = load_model(path)
        assert loaded.name == model.name
        np.testing.assert_allclose(loaded.A, model.A, atol=1e-10)
        np.testing.assert_allclose(loaded.b, model.b, atol=1e-10)
        h = X[20:25]
        np.testing.assert_allclose(loaded.predict_next(h), model.predict_next(h), atol=1e-10)


def test_stable_var_roundtrip(tmp_path):
    X, y, starts, _ = _data()
    model = StableVARModel.fit(X, y, starts, history=2, ridge=1e-2, spectral_cap=0.95)
    path = save_model(model, tmp_path / "var.pt")
    loaded, _ = load_model(path)
    assert loaded.name == "var_stable"
    assert loaded.realized_spectral_radius <= 0.95 * (1 + 1e-8)
    np.testing.assert_allclose(loaded.A, model.A)


def test_shared_model_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    N, p = 8, 3
    D = np.linalg.qr(rng.normal(size=(N, p)))[0]
    mean = rng.normal(size=N)
    R = np.eye(p)
    A = 0.8 * np.eye(p)
    probe = BehaviorProbe()
    X = rng.normal(size=(100, N))
    probe.fit(X, (np.arange(100) // 10) % 4)
    model = SharedLatentModel.from_shared(
        A=A,
        b=np.zeros(p),
        process_cov=np.eye(p) * 0.1,
        D=D,
        mean=mean,
        R=R,
        probe=probe,
        n_neurons=N,
        latent_dim=p,
        obs_cov=np.eye(N),
        initial_cov=np.eye(p),
        spectral_cap=0.995,
    )
    path = save_model(model, tmp_path / "shared.pt")
    loaded, _ = load_model(path)
    z = model.encode(X[0])
    np.testing.assert_allclose(loaded.decode(z), model.decode(z), atol=1e-10)
    assert loaded.name == "shared_latent_adapted"
