"""Contract tests for the three-tier Vienna workbench."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neurobranch.bundlenet_sim.dataset import BEHAVIOR_NAMES, load_trajectories
from neurobranch.bundlenet_sim.dynamics import fit_models
from neurobranch.bundlenet_sim.manifold_walk import ManifoldWalk
from neurobranch.bundlenet_sim.ncmcm import load_ncmcm
from neurobranch.bundlenet_sim.persistence import load_models, save_models
from neurobranch.bundlenet_sim.premotor import PremotorParameters, load_premotor, simulate

pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data" / "processed" / "ncmcm").exists(),
    reason="run scripts/fetch_vienna_artifacts.py to restore third-party artifacts",
)


def test_bundlenet_trajectories_are_aligned() -> None:
    trajectories = load_trajectories()
    assert [t.worm_id for t in trajectories] == [0, 1, 2, 3, 4]
    for trajectory in trajectories:
        assert trajectory.latent.shape[1] == 3
        assert len(trajectory.latent) == len(trajectory.behavior)
        assert set(np.unique(trajectory.behavior)).issubset(set(range(len(BEHAVIOR_NAMES))))


def test_manifold_walk_stays_finite_and_on_scale() -> None:
    trajectories = load_trajectories()
    walk = ManifoldWalk(n_neighbors=10, noise=0.0).fit(trajectories)
    start = trajectories[0].latent[0]
    sim = walk.rollout(start, n_steps=50, seed=3)
    assert sim.shape == (50, 3)
    assert np.isfinite(sim).all()
    all_latents = np.concatenate([t.latent for t in trajectories])
    assert sim.min() >= all_latents.min() - 1.0
    assert sim.max() <= all_latents.max() + 1.0


def test_model_roundtrip(tmp_path) -> None:
    models = fit_models(load_trajectories(), alpha=1.0)
    path = save_models(models, tmp_path / "model.npz")
    loaded = load_models(path)
    probe = np.array([[0.0, 0.0, 0.0], [0.1, -0.2, 0.3]])
    np.testing.assert_allclose(models.dynamics.predict(probe), loaded.dynamics.predict(probe))
    np.testing.assert_allclose(models.behavior.predict_proba(probe), loaded.behavior.predict_proba(probe))


def test_ncmcm_transition_matrix_and_behaviour_mapping() -> None:
    model = load_ncmcm(0)
    assert model.behavior.min() == 0
    assert model.behavior.max() == len(BEHAVIOR_NAMES) - 1
    for k in (2, 6, 20):
        P = model.transition_matrix(k)
        assert P.shape == (k, k)
        np.testing.assert_allclose(P.sum(axis=1), 1.0)
        assert 0.0 <= model.markov_p(k) <= 1.0
        dist = model.behavior_distribution_by_cluster(k)
        assert dist.shape == (k, len(BEHAVIOR_NAMES))
        np.testing.assert_allclose(dist.sum(axis=1), 1.0)


def test_premotor_sandbox_runs_and_responds_to_stimulation() -> None:
    model = load_premotor()
    base = PremotorParameters(duration=2.0, n_samples=40)
    t, y = simulate(model, base, seed=0)
    assert y.shape == (model.n_nodes, len(t))
    assert np.isfinite(y).all()

    stimulated = PremotorParameters(
        duration=2.0,
        n_samples=40,
        stim_node=0,
        stim_amplitude=5.0,
        stim_start=0.5,
        stim_end=1.0,
    )
    _, y_stim = simulate(model, stimulated, seed=0)
    assert not np.allclose(y, y_stim)
