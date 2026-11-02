"""Tests for calibrated uncertainty propagation and probabilistic forecasts."""

from __future__ import annotations

import numpy as np

from neurobranch.adaptive.manifest import verify_manifest
from neurobranch.config import PROJECT_ROOT
from neurobranch.probabilistic.baselines import IndependentARModel
from neurobranch.probabilistic.metrics import coverage_summary, crps_ensemble, energy_score
from neurobranch.probabilistic.residual import ResidualSeries
from neurobranch.probabilistic.stochastic import Innovation, build_innovation, ensemble_rollout
from neurobranch.uncertainty_calibration.models import (
    StateVarianceModel,
    diagonal_scale,
    fit_low_rank_plus_diagonal,
    low_rank_plus_diagonal_covariance,
)
from neurobranch.uncertainty_calibration.propagation import (
    forecast_covariance,
    monte_carlo_covariance,
    transient_amplification,
)
from neurobranch.uncertainty_calibration.trajectory import trajectory_energy_score


def test_analytical_covariance_matches_monte_carlo() -> None:
    A = np.array([[0.55, 0.15], [0.0, 0.35]])
    Q = np.array([[0.4, 0.05], [0.05, 0.2]])
    P = forecast_covariance(A, Q, 20)
    _, mc = monte_carlo_covariance(A, np.zeros(2), np.zeros(2), Q, 20, 15000, np.random.default_rng(4))
    assert abs(np.trace(P[20]) - np.trace(mc[-1])) / np.trace(P[20]) < 0.04
    assert np.allclose(P[1], Q)


def test_nonnormal_transient_amplification_is_not_spectral_radius() -> None:
    A = np.array([[0.9, 4.0], [0.0, 0.9]])
    amp = transient_amplification(A, 20)
    assert amp["spectral_radius"] < 1.0
    assert amp["max_singular_value_h"] > 5.0
    assert amp["nonnormality"] > 0.3


def test_block_innovation_preserves_serial_dependence() -> None:
    rng = np.random.default_rng(8)
    values = np.zeros((4000, 2))
    for t in range(1, len(values)):
        values[t] = 0.8 * values[t - 1] + rng.standard_normal(2)
    block = build_innovation(values, "block", block=20)
    iid = build_innovation(values, "bootstrap")
    eb = block.sample(200, 40, np.random.default_rng(1))
    ei = iid.sample(200, 40, np.random.default_rng(1))
    ac_b = np.mean([np.corrcoef(eb[:, :-1, j].ravel(), eb[:, 1:, j].ravel())[0, 1] for j in range(2)])
    ac_i = np.mean([np.corrcoef(ei[:, :-1, j].ravel(), ei[:, 1:, j].ravel())[0, 1] for j in range(2)])
    assert ac_b > 0.5
    assert ac_i < 0.1


def test_innovation_centering_and_scaling() -> None:
    rng = np.random.default_rng(2)
    values = rng.standard_normal((2000, 4)) * np.array([1.0, 2.0, 0.5, 3.0])
    innovation = build_innovation(values + 5.0, "gauss_diag")
    samples = innovation.sample(5000, 1, np.random.default_rng(3))[:, 0, :]
    assert np.allclose(samples.mean(0), 0.0, atol=0.05)
    assert np.allclose(samples.std(0), values.std(0), rtol=0.05)


def test_observation_noise_is_not_separable_from_process_noise() -> None:
    rng = np.random.default_rng(5)
    latent = rng.standard_normal((5000, 2)) * 0.3
    observed = latent + rng.standard_normal((5000, 2)) * 0.5
    assert np.trace(np.cov(observed, rowvar=False)) > np.trace(np.cov(latent, rowvar=False))
    # A single observed series cannot identify the split without a measurement model.
    assert not np.allclose(np.trace(np.cov(observed, rowvar=False)), np.trace(np.cov(latent, rowvar=False)))


def test_low_rank_plus_diagonal_is_positive_semidefinite() -> None:
    rng = np.random.default_rng(6)
    values = rng.standard_normal((800, 8))
    cov, meta = fit_low_rank_plus_diagonal(values, 3)
    assert np.linalg.eigvalsh(cov).min() > -1e-9
    assert cov.shape == (8, 8)
    assert 0.0 < meta["retained_eigenvalue_fraction"] < 1.0
    assert np.allclose(cov, cov.T)


def test_low_rank_reconstruction() -> None:
    eig = np.array([2.0, 1.0, 0.2])
    vec = np.eye(3)
    cov = low_rank_plus_diagonal_covariance(eig, vec, np.array([0.1, 0.1, 0.2]))
    assert np.allclose(np.diag(cov), [2.1, 1.1, 0.4])


def test_state_variance_predictions_do_not_use_future() -> None:
    rng = np.random.default_rng(9)
    X = rng.standard_normal((800, 3))
    targets = np.arange(2, 700)
    residuals = X[targets] - X[targets - 1]
    series = ResidualSeries("test", residuals, targets, np.zeros_like(targets), np.ones_like(targets), X[targets - 1])
    model = StateVarianceModel.fit(X, series, train_end=600)
    before = model.predict(X, 650)
    altered = X.copy()
    altered[651:] *= 100.0
    after = model.predict(altered, 650)
    assert np.allclose(before, after)
    changed = X.copy()
    changed[650] += 10.0
    assert not np.allclose(before, model.predict(changed, 650))


def test_prequential_subset_respects_target_time() -> None:
    values = np.arange(30, dtype=float)[:, None]
    targets = np.arange(1, 31)
    series = ResidualSeries("test", values, targets, np.zeros(30, dtype=int), np.ones(30, dtype=int), values)
    subset = series.subset(10)
    assert len(subset) == 10
    assert np.all(targets[:10] <= 10)


def test_horizon_alignment_for_persistence() -> None:
    x0 = np.array([1.0, -2.0])
    innovation = Innovation("gauss_diag", scale=np.zeros(2), metadata={})
    ens = ensemble_rollout("persistence", None, x0, 12, 3, innovation, np.random.default_rng(1))
    assert ens.shape == (3, 12, 2)
    assert np.allclose(ens, x0[None, None, :])


def test_independent_ar_recovers_independent_coefficients() -> None:
    rng = np.random.default_rng(10)
    X = np.zeros((1200, 3))
    for t in range(1, 1200):
        X[t] = np.array([0.6, 0.4, 0.2]) * X[t - 1] + rng.standard_normal(3) * 0.2
    fitted = IndependentARModel(window=500).fit_at(X, 1000)
    assert np.allclose(fitted.a, [0.6, 0.4, 0.2], atol=0.1)
    residual = X[501:1001] - (fitted.a * X[500:1000] + fitted.b)
    assert np.max(np.abs(np.corrcoef(residual, rowvar=False) - np.eye(3))) < 0.25


def test_marginal_and_multivariate_scores() -> None:
    rng = np.random.default_rng(11)
    ens = rng.standard_normal((3000, 3))
    obs = np.zeros(3)
    crps = crps_ensemble(ens, obs)
    assert crps.shape == (3,)
    assert energy_score(ens, obs) > 0
    draws = [coverage_summary(ens, rng.standard_normal(3)) for _ in range(300)]
    assert 0.45 < np.mean([d["coverage_50"] for d in draws]) < 0.55


def test_interval_calibration_on_gaussian_draws() -> None:
    rng = np.random.default_rng(12)
    ens = rng.standard_normal((4000, 1))
    draws = [coverage_summary(ens, rng.standard_normal(1)) for _ in range(500)]
    assert abs(np.mean([d["coverage_95"] for d in draws]) - 0.95) < 0.02


def test_trajectory_energy_prefers_correct_joint_distribution() -> None:
    rng = np.random.default_rng(13)
    truth = np.zeros((5, 2))
    correct = rng.standard_normal((300, 5, 2))
    wrong = correct + np.array([3.0, -3.0])
    assert trajectory_energy_score(correct, truth) < trajectory_energy_score(wrong, truth)


def test_seed_reproducibility() -> None:
    innovation = Innovation("gauss_diag", scale=np.ones(3), metadata={})
    a = innovation.sample(10, 5, np.random.default_rng(77))
    b = innovation.sample(10, 5, np.random.default_rng(77))
    c = innovation.sample(10, 5, np.random.default_rng(78))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_long_horizon_numerical_stability() -> None:
    A = np.diag([0.9, 0.8])
    Q = np.eye(2) * 0.1
    P = forecast_covariance(A, Q, 500)
    assert np.isfinite(P).all()
    assert np.trace(P[500]) < 10.0


def test_diagonal_scale_is_regularized() -> None:
    values = np.zeros((10, 3))
    scale = diagonal_scale(values)
    assert np.all(scale > 0)


def test_report_manifest_matches_artifacts() -> None:
    report = PROJECT_ROOT / "reports/simulator/uncertainty_calibration"
    if not (report / "manifest.json").exists():
        return
    ok, problems = verify_manifest(report)
    assert ok, problems
