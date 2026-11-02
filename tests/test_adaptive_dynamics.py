"""Tests for causal adaptive dynamics and the stability experiment."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from neurobranch.adaptive.calcium import CorrectedCalciumForecast
from neurobranch.adaptive.evaluation import block_bootstrap_ci, evaluation_landmarks, paired_summary
from neurobranch.adaptive.manifest import sha256_file, verify_manifest, write_manifest
from neurobranch.adaptive.nonstationarity import (
    principal_angles_deg,
    synthetic_observed,
    window_stats,
)
from neurobranch.adaptive.var import EWMAVAR, FrozenVAR, RollingVAR, spectral_radius, stabilize_matrix
from neurobranch.config import PROJECT_ROOT
from neurobranch.diagnostics.observation import CausalCalciumLDS


def _stationary_var(seed: int = 3, n: int = 6, t: int = 400):
    rng = np.random.default_rng(seed)
    A = np.diag(rng.uniform(0.5, 0.9, size=n))
    X = np.zeros((t, n))
    for i in range(1, t):
        X[i] = A @ X[i - 1] + rng.normal(scale=0.2, size=n)
    return X, A


def test_spectral_radius_and_cap():
    A = np.diag([0.5, 1.2, -0.8])
    assert spectral_radius(A) == pytest.approx(1.2)
    capped, record = stabilize_matrix(A, cap=0.9)
    assert spectral_radius(capped) == pytest.approx(0.9)
    assert record["clipped"] is True


def test_chronological_isolation_and_window_boundaries():
    X, _ = _stationary_var()
    t = 260
    model = RollingVAR(window=50, min_transitions=10)
    fitted = model.fit_at(X, t)
    assert fitted is not None
    assert fitted.fit_record["last_transition"] == t - 1
    assert fitted.fit_record["first_transition"] == t - 49
    assert fitted.fit_record["n_transitions"] == 49

    X_future = X.copy()
    X_future[t + 1 :] += 1e6
    fitted_future = model.fit_at(X_future, t)
    assert np.allclose(fitted.A, fitted_future.A)
    x0 = X[t].copy()
    assert np.allclose(fitted.forecast(x0, 5), fitted_future.forecast(x0, 5))


def test_adaptive_parameters_update_with_time():
    X, _ = _stationary_var()
    early = RollingVAR(window=80).fit_at(X, 200)
    late = RollingVAR(window=80).fit_at(X, 260)
    assert early is not None and late is not None
    assert not np.allclose(early.A, late.A)


def test_ewma_recursive_stability_and_horizon_integrity():
    X, _ = _stationary_var()
    model = EWMAVAR(halflife=40).fit_at(X, 260)
    assert model is not None
    assert model.spectral_radius <= 0.995 + 1e-12
    out = model.forecast(X[260], 25)
    assert out.shape == (25, X.shape[1])
    assert np.isfinite(out).all()
    # Forecast output cannot depend on future observations by construction.
    assert out.shape[0] == 25


def test_rolling_fit_matches_frozen_estimator():
    X, _ = _stationary_var()
    starts = np.arange(1, 300)
    frozen = FrozenVAR.fit(X, starts, ridge=1.0, spectral_cap=0.995)
    rolled = RollingVAR(window=300, min_transitions=10).fit_at(X, 300)
    assert rolled is not None
    assert np.allclose(frozen.A, rolled.A, atol=1e-10)
    assert np.allclose(frozen.b, rolled.b, atol=1e-10)


def test_principal_angles_are_basis_invariant():
    rng = np.random.default_rng(7)
    Q, _ = np.linalg.qr(rng.normal(size=(10, 4)))
    R, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    same = principal_angles_deg(Q, Q @ R)
    assert np.allclose(same, 0.0, atol=1e-5)
    complement = principal_angles_deg(Q, np.linalg.qr(rng.normal(size=(10, 2)))[0])
    assert len(complement) == 2


def test_synthetic_stationary_versus_regime_switch():
    stationary = synthetic_observed(n_frames=800, latent_dim=6, n_neurons=20, regime=False)
    mask = np.zeros(len(stationary), dtype=bool)
    mask[:300] = True
    s_stats = window_stats(stationary, reference_mask=mask, window=120, step=60, k=4)
    regime = synthetic_observed(n_frames=800, latent_dim=6, n_neurons=20, regime=True)
    r_stats = window_stats(regime, reference_mask=mask, window=120, step=60, k=4)
    assert r_stats.reference_angle_mean_deg.tail(3).mean() > s_stats.reference_angle_mean_deg.tail(3).mean()


def test_evaluation_landmarks_leave_full_horizon():
    T = 500
    masks = {"test": np.zeros(T, dtype=bool)}
    masks["test"][300:] = True
    data = SimpleNamespace(n_frames=T, masks=masks)
    landmarks = evaluation_landmarks(data, horizon_max=100, stride=10)
    assert len(landmarks) > 0
    assert landmarks.max() + 100 <= T - 1
    assert np.all(np.diff(landmarks) == 10)


def test_block_bootstrap_and_paired_metric():
    values = np.linspace(-0.5, 1.5, 40)
    mean, lo, hi = block_bootstrap_ci(values, block=5, n_boot=200, seed=1)
    assert lo <= mean <= hi
    frame = pd.DataFrame(
        {
            "worm_id": [0, 0, 1, 1],
            "landmark": [10, 20, 10, 20],
            "model": ["frozen_var", "frozen_var", "frozen_var", "frozen_var"],
            "horizon": [20, 20, 20, 20],
            "mse": [2.0, 2.0, 2.0, 2.0],
        }
    )
    adaptive = frame.copy()
    adaptive["model"] = "adaptive"
    adaptive["mse"] = [1.0, 1.5, 1.0, 1.0]
    summary = paired_summary(pd.concat([frame, adaptive], ignore_index=True))
    row = summary[summary.model == "adaptive"].iloc[0]
    assert row["mean_mse_improvement"] == pytest.approx(0.875)
    assert row["fraction_landmarks_better"] == pytest.approx(1.0)


def test_corrected_calcium_is_finite_on_stable_system():
    rng = np.random.default_rng(5)
    T, n, k = 500, 8, 3
    A = np.diag([0.85, 0.8, 0.9])
    D = rng.normal(size=(k, n))
    z = rng.normal(size=k)
    c = np.zeros(n)
    X = np.empty((T, n))
    for t in range(T):
        z = A @ z + rng.normal(scale=0.05, size=k)
        c = 0.7 * c + 0.3 * (D.T @ z)
        X[t] = c + rng.normal(scale=0.02, size=n)
    mask = np.zeros(T, dtype=bool)
    mask[:350] = True
    starts = np.arange(2, 348)
    base = CausalCalciumLDS.fit(X, mask, starts, latent_dim=3, ridge=1e-2)
    corrected = CorrectedCalciumForecast(base)
    out = corrected.forecast(X[349:450], 100)
    assert out.shape == (100, n)
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) < 1e3


def test_dashboard_artifacts_and_manifest(tmp_path: Path):
    root = PROJECT_ROOT / "reports/simulator/adaptive_dynamics"
    required = ["model_paired_overall.csv", "calcium_diagnostics.json", "nonstationarity_summary.csv"]
    for name in required:
        assert (root / name).exists(), name
    frame = pd.read_csv(root / "model_paired_overall.csv")
    assert {"model", "mean_mse_improvement", "ci_low", "ci_high"}.issubset(frame.columns)
    payload = json.loads((root / "calcium_diagnostics.json").read_text(encoding="utf-8"))
    assert "diagnostics" in payload and payload["stage"] == "B_calcium_stability"

    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    write_manifest(tmp_path)
    ok, problems = verify_manifest(tmp_path)
    assert ok and not problems
    assert sha256_file(tmp_path / "a.txt")
    (tmp_path / "a.txt").write_text("changed", encoding="utf-8")
    ok, problems = verify_manifest(tmp_path)
    assert not ok and problems
