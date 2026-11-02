"""Unit tests for the joint-dynamics package and experiment contracts."""

from __future__ import annotations

import numpy as np

from neurobranch.joint_dynamics.evaluation import (
    autocorrelation_error,
    neural_covariance_error,
    pooled_neural_covariance,
)
from neurobranch.joint_dynamics.models import (
    LowRankTemporalInnovation,
    block_bootstrap_starts,
    sample_student_t,
)
from neurobranch.joint_dynamics.residuals import acf_profile, lead1_innovations, residual_structure
from neurobranch.probabilistic.metrics import coverage_summary, energy_score
from neurobranch.probabilistic.residual import ResidualSeries
from neurobranch.uncertainty_calibration.trajectory import trajectory_energy_score


def _series() -> ResidualSeries:
    rng = np.random.default_rng(4)
    values = rng.standard_normal((30, 3))
    targets = np.arange(1, 31)
    fits = targets - 1
    leads = np.tile(np.arange(1, 6), 6)
    states = rng.standard_normal((30, 3))
    return ResidualSeries("synthetic", values, targets, fits, leads, states)


def test_lead1_innovations_excludes_multistep_errors() -> None:
    s = _series()
    sample = lead1_innovations(s)
    assert len(sample.values) == 6
    assert all(s.lead[s.lead == 1] == 1)


def test_lead1_innovations_respects_causal_cutoff() -> None:
    s = _series()
    sample = lead1_innovations(s, t=5)
    assert sample.targets.max() <= 5


def test_acf_profile_detects_ar_structure() -> None:
    rng = np.random.default_rng(3)
    x = np.zeros(2000)
    for t in range(1, len(x)):
        x[t] = 0.8 * x[t - 1] + rng.standard_normal()
    profile = acf_profile(x[:, None], max_lag=5)
    assert profile.loc[profile.lag == 1, "acf_median"].iloc[0] > 0.5


def test_residual_structure_returns_finite_core_fields() -> None:
    rng = np.random.default_rng(7)
    s = ResidualSeries("x", rng.standard_normal((100, 4)), np.arange(1, 101), np.arange(100), np.ones(100, int), rng.standard_normal((100, 4)))
    info = residual_structure(s)
    assert info["n"] == 100
    assert np.isfinite(info["trace_cov"])
    assert 0 <= info["top_eigenvalue_fraction"] <= 1


def test_low_rank_innovation_is_stable_and_reproducible() -> None:
    rng = np.random.default_rng(5)
    residuals = rng.standard_normal((300, 5))
    m1 = LowRankTemporalInnovation.fit(residuals, rank=3)
    m2 = LowRankTemporalInnovation.fit(residuals, rank=3)
    assert np.allclose(m1.phi, m2.phi)
    a = m1.sample(20, 10, np.random.default_rng(11))
    b = m1.sample(20, 10, np.random.default_rng(11))
    assert a.shape == (20, 10, 5)
    assert np.allclose(a, b)
    assert np.isfinite(a).all()


def test_student_t_sampling_has_requested_covariance_scale() -> None:
    cov = np.array([[1.0, 0.4], [0.4, 2.0]])
    eps = sample_student_t(20000, 1, cov, np.random.default_rng(13), df=8.0, scale=1.0)
    empirical = np.cov(eps[:, 0], rowvar=False)
    assert np.allclose(empirical, cov, rtol=0.12, atol=0.12)


def test_block_bootstrap_keeps_indices_inside_causal_window() -> None:
    starts = np.arange(100, 200)
    out = block_bootstrap_starts(starts, block=10, rng=np.random.default_rng(2))
    assert len(out) == len(starts)
    assert out.min() >= 100
    assert out.max() <= 199


def test_energy_score_prefers_correct_ensemble() -> None:
    rng = np.random.default_rng(8)
    obs = np.array([0.0, 0.0])
    good = rng.normal(0, 1, (500, 2))
    bad = rng.normal(8, 1, (500, 2))
    assert energy_score(good, obs) < energy_score(bad, obs)


def test_trajectory_energy_penalises_wrong_temporal_path() -> None:
    rng = np.random.default_rng(9)
    obs = np.zeros((5, 2))
    good = rng.normal(0, 0.5, (200, 5, 2))
    bad = np.repeat(rng.normal(5, 0.5, (200, 1, 2)), 5, axis=1)
    assert trajectory_energy_score(good, obs) < trajectory_energy_score(bad, obs)


def test_coverage_is_nominal_for_calibrated_gaussian() -> None:
    rng = np.random.default_rng(10)
    ens = rng.normal(0, 1, (5000, 1))
    obs = rng.normal(0, 1, 2000)
    cov = []
    for y in obs:
        cov.append(coverage_summary(ens, np.array([y]))["coverage_80"])
    assert abs(float(np.mean(cov)) - 0.8) < 0.02


def test_autocorrelation_and_covariance_errors_vanish_for_identical_paths() -> None:
    rng = np.random.default_rng(12)
    path = rng.standard_normal((100, 20, 4))
    assert autocorrelation_error(path, path) < 1e-10
    assert neural_covariance_error(path, path) < 1e-10


def test_pooled_covariance_is_symmetric_psd() -> None:
    rng = np.random.default_rng(14)
    cov = pooled_neural_covariance(rng.standard_normal((200, 5, 6)))
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > -1e-10


def test_no_future_data_in_parameter_bootstrap() -> None:
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "jd_parameter.py"
    spec = importlib.util.spec_from_file_location("jd_parameter_test", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    rng = np.random.default_rng(15)
    X = rng.standard_normal((700, 3))
    t = 600
    a = module._bootstrap_models(X, t, draws=3)
    X2 = X.copy()
    X2[t + 1 :] += 1000.0
    b = module._bootstrap_models(X2, t, draws=3)
    assert len(a) == len(b) == 3
    for (A1, b1, _), (A2, b2, _) in zip(a, b):
        assert np.allclose(A1, A2)
        assert np.allclose(b1, b2)
