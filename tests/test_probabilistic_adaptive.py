"""Tests for calibrated probabilistic adaptive dynamics."""

from __future__ import annotations

import numpy as np
import pytest

from neurobranch.adaptive.var import FrozenVAR, RollingVAR, fit_ridge_var
from neurobranch.probabilistic.baselines import IndependentARModel, LocalMeanModel
from neurobranch.probabilistic.metrics import (
    coverage_summary,
    crps_ensemble,
    energy_score,
    interval_coverage,
    pit_histogram,
)
from neurobranch.probabilistic.residual import (
    ResidualSeries,
    center_residuals,
    regularized_covariance,
    residual_diagnostics,
)
from neurobranch.probabilistic.stochastic import build_innovation, ensemble_rollout


def _var(A, b, d):
    return FrozenVAR(A=np.asarray(A, float), b=np.asarray(b, float), ridge=1.0, spectral_cap=0.995, fit_record={})


def test_energy_score_prefers_calibrated_distribution():
    rng = np.random.default_rng(0)
    y = np.zeros(5)
    narrow = rng.normal(scale=0.9, size=(3000, 5))
    wide = rng.normal(scale=4.0, size=(3000, 5))
    assert energy_score(narrow, y) < energy_score(wide, y)


def test_crps_matches_known_gaussian_order():
    rng = np.random.default_rng(2)
    y = np.ones(4)
    good = rng.normal(loc=1.0, scale=0.5, size=(4000, 4))
    shifted = rng.normal(loc=3.0, scale=0.5, size=(4000, 4))
    assert crps_ensemble(good, y).mean() < crps_ensemble(shifted, y).mean()


def test_coverage_and_width_are_calibrated():
    rng = np.random.default_rng(3)
    ens = rng.normal(size=(5000, 8))
    obs = rng.normal(size=8)
    cov = coverage_summary(ens, obs)
    assert abs(cov["coverage_80"] - 0.8) < 0.08
    assert abs(cov["coverage_95"] - 0.95) < 0.06
    assert cov["width_95"] > cov["width_50"] > 0


def test_pit_is_uniform_for_calibrated_ensemble():
    rng = np.random.default_rng(4)
    ens = rng.normal(size=(1000, 200))
    y = rng.normal(size=200)
    pit = (ens <= y[None, :]).mean(axis=0)
    density, err = pit_histogram(pit, bins=10)
    assert density.shape == (10,)
    assert err < 0.08


def test_covariance_is_positive_semidefinite():
    rng = np.random.default_rng(5)
    values = rng.normal(size=(40, 12)) @ rng.normal(size=(12, 12))
    cov, meta = regularized_covariance(values)
    assert meta["method"] == "ledoit"
    assert np.linalg.eigvalsh(cov).min() > 0
    assert np.allclose(cov, cov.T)


def test_centered_innovations_have_zero_mean():
    rng = np.random.default_rng(6)
    values = rng.normal(loc=5.0, size=(100, 4))
    centered, mean = center_residuals(values)
    assert np.allclose(centered.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(mean, 5.0, atol=0.2)


def test_bootstrap_and_block_reproducibility():
    rng = np.random.default_rng(7)
    pool = rng.normal(size=(200, 3))
    inn = build_innovation(pool, "block", block=5)
    a = inn.sample(20, 10, np.random.default_rng(11))
    b = inn.sample(20, 10, np.random.default_rng(11))
    assert np.allclose(a, b)
    assert a.shape == (20, 10, 3)
    assert not np.allclose(a, inn.sample(20, 10, np.random.default_rng(12)))


def test_block_bootstrap_preserves_temporal_dependence_better_than_iid():
    rng = np.random.default_rng(8)
    x = np.zeros(500)
    for i in range(1, len(x)):
        x[i] = 0.8 * x[i - 1] + rng.normal()
    pool = x[:, None]
    block = build_innovation(pool, "block", block=20).sample(1, 100, np.random.default_rng(1))[0, :, 0]
    iid = build_innovation(pool, "bootstrap").sample(1, 100, np.random.default_rng(1))[0, :, 0]

    def acf(a):
        return np.corrcoef(a[:-1], a[1:])[0, 1]

    assert acf(block) > acf(iid)


def test_monte_carlo_recovers_conditional_mean():
    A = np.array([[0.8, 0.1], [0.0, 0.6]])
    b = np.array([0.2, -0.1])
    fitted = _var(A, b, 2)
    rng = np.random.default_rng(9)
    innovation = build_innovation(np.zeros((50, 2)), "gauss")
    ens = ensemble_rollout("var", fitted, np.array([1.0, -1.0]), 10, 3000, innovation, rng)
    mean = ens.mean(axis=0)
    truth = fitted.forecast(np.array([1.0, -1.0]), 10)
    assert np.max(np.abs(mean - truth)) < 0.03


def test_recursive_rollout_horizon_alignment():
    fitted = _var([[0.5, 0], [0, 0.5]], [0, 0], 2)
    inn = build_innovation(np.zeros((20, 2)), "gauss")
    ens = ensemble_rollout("var", fitted, np.ones(2), 17, 5, inn, np.random.default_rng(1))
    assert ens.shape == (5, 17, 2)
    assert np.allclose(ens[:, 0, :], 0.5 * np.ones((5, 2)), atol=1e-3)


def test_long_horizon_ensemble_is_finite():
    fitted = _var(np.eye(4) * 0.99, np.zeros(4), 4)
    inn = build_innovation(np.random.default_rng(3).normal(size=(100, 4)), "bootstrap")
    ens = ensemble_rollout("var", fitted, np.ones(4), 300, 20, inn, np.random.default_rng(4))
    assert np.isfinite(ens).all()


def test_rolling_boundary_has_no_future_target():
    X = np.arange(100, dtype=float)[:, None]
    fitted = RollingVAR(window=20, min_transitions=10).fit_at(X, 50)
    assert fitted.fit_record["last_transition"] <= 49
    assert fitted.fit_record["first_transition"] >= 30


def test_independent_ar_forecast_and_cap():
    rng = np.random.default_rng(10)
    X = np.zeros((300, 4))
    for i in range(1, 300):
        X[i] = 0.5 * X[i - 1] + rng.normal(scale=0.3, size=4)
    fitted = IndependentARModel(window=200).fit_at(X, 250)
    assert fitted is not None
    assert fitted.spectral_radius <= 0.995 + 1e-12
    pred = fitted.forecast(X[250], 10)
    assert pred.shape == (10, 4)


def test_local_mean_uses_only_history():
    X = np.arange(100, dtype=float)[:, None]
    fitted = LocalMeanModel(window=10, min_history=10).fit_at(X, 20)
    assert fitted.value[0] == pytest.approx(np.mean(X[11:21]))


def test_residual_indexing_and_diagnostics():
    rng = np.random.default_rng(11)
    values = rng.normal(size=(100, 3))
    series = ResidualSeries(
        "test",
        values,
        np.arange(10, 110),
        np.arange(9, 109),
        np.ones(100, dtype=int),
        rng.normal(size=(100, 3)),
    )
    subset = series.subset(50)
    assert len(subset) == 41
    diag = residual_diagnostics(series)
    assert diag["n"] == 100
    assert np.isfinite(diag["mean_abs_cross_corr"])


def test_cross_neuron_dynamics_can_add_information():
    rng = np.random.default_rng(12)
    A = np.array([[0.7, 0.0], [0.8, 0.7]])
    X = np.zeros((400, 2))
    for i in range(1, 400):
        X[i] = A @ X[i - 1] + rng.normal(scale=0.1, size=2)
    starts = np.arange(1, 300)
    fitted = fit_ridge_var(X, starts, ridge=0.01, spectral_cap=None)
    assert fitted is not None
    Afit, bfit, _ = fitted
    Adiag = np.diag(np.diag(Afit))
    full_err, diag_err = [], []
    for t in range(250, 340, 5):
        xf = X[t].copy()
        xd = X[t].copy()
        full_pred, diag_pred = [], []
        for _ in range(5):
            xf = Afit @ xf + bfit
            xd = Adiag @ xd + bfit
            full_pred.append(xf.copy())
            diag_pred.append(xd.copy())
        truth = X[t + 1 : t + 6]
        full_err.append(np.mean((np.asarray(full_pred) - truth) ** 2))
        diag_err.append(np.mean((np.asarray(diag_pred) - truth) ** 2))
    assert np.mean(full_err) < np.mean(diag_err)


def test_interval_width_penalizes_wide_forecasts():
    rng = np.random.default_rng(13)
    obs = np.zeros(3)
    narrow = rng.normal(scale=0.5, size=(500, 3))
    wide = rng.normal(scale=5.0, size=(500, 3))
    assert interval_coverage(narrow, obs, 0.95)[1].mean() < interval_coverage(wide, obs, 0.95)[1].mean()


def test_reproduction_matches_historical_artifact():
    import pandas as pd

    root = pytest.importorskip("neurobranch.config").PROJECT_ROOT
    new = pd.read_csv(root / "reports/simulator/probabilistic_adaptive/baseline_per_landmark.csv")
    old = pd.read_csv(root / "reports/simulator/adaptive_dynamics/model_per_landmark.csv")
    new = new[new.horizon == 20]
    old = old[old.horizon == 20]
    pairs = [("B1_frozen_var", "frozen_var"), ("B5_rolling_var_w500", "rolling_w500")]
    for new_name, old_name in pairs:
        a = new[new.model == new_name].set_index(["worm_id", "landmark"]).mse
        b = old[old.model == old_name].set_index(["worm_id", "landmark"]).mse
        assert len(a) == len(b) == 175
        assert np.max(np.abs(a.sort_index().to_numpy() - b.sort_index().to_numpy())) < 1e-9


def test_gate_statuses_and_manifest_are_consistent():
    import json

    from neurobranch.adaptive.manifest import verify_manifest

    root = pytest.importorskip("neurobranch.config").PROJECT_ROOT
    out = root / "reports/simulator/probabilistic_adaptive"
    gates = json.loads((out / "prob_gates.json").read_text(encoding="utf-8"))
    assert gates["gates"]["D_probabilistic_score_advantage"]["status"] == "FAIL"
    assert gates["gates"]["E_calibration"]["status"] == "FAIL"
    assert gates["gates"]["F_trajectory_fidelity"]["status"] == "FAIL"
    assert gates["gates"]["H_biological_validation"]["status"] == "NOT_EVALUABLE"
    if (out / "manifest.json").exists():
        ok, problems = verify_manifest(out)
        assert ok, problems


def test_primary_model_energy_is_worse_than_local_mean():
    import pandas as pd

    root = pytest.importorskip("neurobranch.config").PROJECT_ROOT
    summary = pd.read_csv(root / "reports/simulator/probabilistic_adaptive/prob_summary.csv")
    energy = summary[summary.metric == "energy_h20"].set_index("model")["mean"]
    assert energy["P6_adaptive_block"] > energy["P1_local_mean"]
    assert energy["P6_adaptive_block"] > energy["P2_independent_ar"]
