"""Unit and artifact checks for the Kato common-state controls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurobranch.kato_common_state.evaluation import score_ensemble
from neurobranch.kato_common_state.factors import fit_factor_model
from neurobranch.kato_common_state.models import (
    coefficient_ablation,
    fit_common_factor_ar,
    fit_matched_own_ar,
    fit_trend_ar,
    rollout_common_factor,
    rollout_matched,
)
from neurobranch.kato_common_state.surrogates import fit_common_state_surrogate
from neurobranch.kato_common_state.synthetic import evaluate_system, simulate_system


def _ar_data(n: int = 500, seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 4), dtype=np.float64)
    for t in range(1, n):
        X[t] = 0.65 * X[t - 1] + rng.normal(scale=0.2, size=4)
    return X


def test_matched_own_ar_recovers_diagonal_dynamics() -> None:
    X = _ar_data()
    model = fit_matched_own_ar(X, np.arange(len(X) - 1), window=400, ridge=1.0)
    assert np.allclose(model.a, 0.65, atol=0.05)
    assert model.n_parameters == 8
    rollout = rollout_matched(model, X[-1], np.zeros((3, 5, 4)))
    assert rollout.shape == (3, 5, 4)


def test_trend_ar_recovers_linear_drift() -> None:
    n = 500
    time = np.arange(n, dtype=np.float64)
    X = 0.5 * time[:, None] + np.random.default_rng(2).normal(scale=0.05, size=(n, 2))
    model = fit_trend_ar(X, np.arange(n - 1), window=400, ridge=1.0)
    assert np.all(model.trend > 0.0)
    assert model.n_parameters == 6


def test_common_factor_model_is_train_only_and_invertible() -> None:
    rng = np.random.default_rng(7)
    train = rng.normal(size=(400, 8))
    model = fit_factor_model(train, n_components=4, seed=11)
    scores = model.scores(train, 3)
    residual = model.residual(train, 3)
    assert scores.shape == (400, 3)
    assert np.max(np.abs((residual - residual.mean(axis=0)) @ model.pca.components_[:3].T)) < 1e-10
    assert model.fit_record["fit_split"] == "train"


def test_common_factor_ar_shapes_and_rollout() -> None:
    rng = np.random.default_rng(8)
    n = 700
    X = rng.normal(size=(n, 6))
    factors = fit_factor_model(X[:500], n_components=3, seed=3)
    F = factors.scores(X, 3)
    model = fit_common_factor_ar(X, F, np.arange(400, 600), k=3, window=500, ridge=1.0)
    assert model.own_coef.shape == (6, 4)
    assert model.n_parameters == 6 * 4 + 3 * 3 + 3 + 6
    ens = rollout_common_factor(model, X[600], F[600], np.zeros((2, 3, 6)))
    assert ens.shape == (2, 3, 6)


def test_coefficient_ablation_removes_offdiagonal() -> None:
    X = _ar_data()
    from neurobranch.kato_interactions.models import fit_var

    full = fit_var(X, np.arange(len(X) - 1), spectral_cap=None)
    assert full is not None
    ablated = coefficient_ablation(full, X, len(X) - 1)
    assert np.allclose(ablated.A - np.diag(np.diag(ablated.A)), 0.0)


def test_common_preserving_surrogate_keeps_common_trajectory() -> None:
    rng = np.random.default_rng(12)
    X = rng.normal(size=(700, 6))
    factors = fit_factor_model(X[:400], n_components=3, seed=5)
    model, diag = fit_common_state_surrogate(
        X,
        650,
        factors=factors,
        k=3,
        method="common_preserved_residual_shift",
        window=300,
        seed=4,
    )
    assert model is not None
    assert diag["common_factor_max_change"] < 1e-10
    assert diag["residual_max_change"] > 0.0


def test_factor_preserving_surrogate_keeps_residual() -> None:
    rng = np.random.default_rng(13)
    X = rng.normal(size=(700, 5))
    factors = fit_factor_model(X[:400], n_components=3, seed=6)
    _, diag = fit_common_state_surrogate(
        X,
        650,
        factors=factors,
        k=3,
        method="residual_preserved_factor_shift",
        window=300,
        seed=4,
    )
    assert diag["residual_max_change"] < 1e-10
    assert diag["common_factor_max_change"] > 0.0


def test_surrogate_never_reads_future_frames() -> None:
    rng = np.random.default_rng(14)
    X = rng.normal(size=(900, 4))
    X_future = X.copy()
    X_future[601:] += 100.0
    factors = fit_factor_model(X[:400], n_components=2, seed=7)
    a, _ = fit_common_state_surrogate(
        X, 600, factors=factors, k=2, method="circular_full", window=300, seed=9
    )
    b, _ = fit_common_state_surrogate(
        X_future, 600, factors=factors, k=2, method="circular_full", window=300, seed=9
    )
    assert a is not None and b is not None
    assert np.allclose(a.A, b.A)
    assert np.allclose(a.b, b.b)


def test_synthetic_controls_separate_common_from_direct_coupling() -> None:
    common = evaluate_system(simulate_system("common", 11), factor_ks=(3,))
    coupled = evaluate_system(simulate_system("coupled", 11), factor_ks=(3,))
    assert common["b3_minus_b5_3_mse_gain"] < 0.0
    assert coupled["b3_minus_b5_3_mse_gain"] > 0.0


def test_evaluation_returns_factor_and_residual_spaces() -> None:
    rng = np.random.default_rng(15)
    train = rng.normal(size=(300, 5))
    factors = fit_factor_model(train, n_components=3, seed=1)
    ens = rng.normal(size=(10, 20, 5))
    truth = rng.normal(size=(20, 5))
    metrics = score_ensemble(ens, truth, factors.pca, factor_k=3)
    assert "pca10_energy_h20" in metrics
    assert "neuron_energy_h20" in metrics
    assert "factor_energy_h20" in metrics
    assert "residual_energy_h20" in metrics


def test_artifact_contract_if_present() -> None:
    from pathlib import Path

    out = Path(__file__).resolve().parents[1] / "reports" / "simulator" / "kato_common_state"
    if not (out / "model_per_landmark.csv").exists():
        pytest.skip("common-state artifacts have not been generated")
    frame = pd.read_csv(out / "model_per_landmark.csv")
    assert {"B3_population", "B5_common_k10", "B1_matched_own"} <= set(frame["model"])
    assert frame["pca10_energy_h20"].notna().all()


def test_residual_advantage_sign_and_decomposition_if_present() -> None:
    from pathlib import Path

    out = Path(__file__).resolve().parents[1] / "reports" / "simulator" / "kato_common_state"
    if not (out / "residual_cross_neuron.csv").exists():
        pytest.skip("common-state artifacts have not been generated")
    residual = pd.read_csv(out / "residual_cross_neuron.csv")
    primary = residual[residual["block"] == 5].iloc[0]
    assert primary["model"] == "B3_population"
    assert primary["reference"] == "B5_common_k10"
    assert primary["mean_improvement"] > 0
    decomp = pd.read_csv(out / "decomposition.csv")
    common = float(decomp.loc[decomp["contrast"] == "common_factors_B5_vs_B4m", "mean_improvement"].iloc[0])
    residual_gain = float(
        decomp.loc[decomp["contrast"] == "residual_cross_B3_vs_B5", "mean_improvement"].iloc[0]
    )
    population = float(
        decomp.loc[decomp["contrast"] == "population_B3_vs_B4m", "mean_improvement"].iloc[0]
    )
    assert common > 0 and residual_gain > 0
    assert abs((common + residual_gain) - population) < 1e-9


def test_surrogate_direction_and_gates_if_present() -> None:
    import json
    from pathlib import Path

    out = Path(__file__).resolve().parents[1] / "reports" / "simulator" / "kato_common_state"
    if not (out / "surrogate_paired.csv").exists():
        pytest.skip("common-state artifacts have not been generated")
    paired = pd.read_csv(out / "surrogate_paired.csv")
    primary = paired[(paired["block"] == 5) & (paired["reference"] == "surrogate_common_preserved_residual_shift")]
    assert not primary.empty
    assert primary.iloc[0]["mean_improvement"] > 0
    gates = json.loads((out / "gates.json").read_text(encoding="utf-8"))
    assert gates["G4_residual_cross_neuron_information"]["status"] == "PASS"
    assert gates["G7_structural_identifiability"]["status"] == "FAIL"
    assert gates["G8_biological_intervention_validity"]["status"] == "NOT_EVALUABLE"
