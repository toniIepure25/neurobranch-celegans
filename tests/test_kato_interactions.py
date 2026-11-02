"""Unit and artifact checks for the Kato-only interaction audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from neurobranch.kato_interactions.metrics import paired_row, session_block_ci
from neurobranch.kato_interactions.models import (
    FrozenLinearModel,
    diagonal_copy,
    fit_var,
    fit_window,
)
from neurobranch.kato_interactions.sensitivity import (
    finite_difference_error,
    matched_control_indices,
    pairwise_disagreement,
    response_cosine,
    response_metrics,
    support_fraction,
    unit_neuron_delta,
)
from neurobranch.kato_interactions.spectra import (
    leading_decay_frames,
    mode_table,
    nonnormality,
    transient_amplification,
)
from neurobranch.kato_interactions.states import (
    assign_states,
    dwell_times,
    fit_states,
    occupancy,
    row_normalize,
    simulate_markov,
    state_metrics,
    transition_counts,
    weighted_transition_tv,
)
from neurobranch.kato_interactions.surrogates import fit_shuffled_window, shuffle_window
from neurobranch.kato_interactions.synthetic import audit, evaluate_system

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "simulator" / "kato_interactions"


def _var_data(n: int = 5000, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    A = np.array([[0.55, 0.10], [0.02, 0.40]], dtype=np.float64)
    X = np.zeros((n, 2), dtype=np.float64)
    for t in range(1, n):
        X[t] = A @ X[t - 1] + rng.normal(scale=0.2, size=2)
    return X, A


def test_var_recovers_known_operator() -> None:
    X, A = _var_data()
    model = fit_var(X, np.arange(0, len(X) - 1), ridge=1.0, spectral_cap=None)
    assert model is not None
    assert np.max(np.abs(model.A - A)) < 0.05
    assert abs(float(model.b.mean())) < 0.02


def test_diagonal_fit_has_zero_offdiagonal() -> None:
    X, _ = _var_data()
    model = fit_var(X, np.arange(0, len(X) - 1), diagonal=True)
    assert model is not None
    assert np.allclose(model.A - np.diag(np.diag(model.A)), 0.0)
    assert model.fit_record["diagonal"] is True


def test_diagonal_copy_keeps_diagonal_and_removes_cross_terms() -> None:
    X, _ = _var_data()
    full = fit_var(X, np.arange(0, len(X) - 1), spectral_cap=None)
    assert full is not None
    copied = diagonal_copy(full, X, len(X) - 1)
    assert np.allclose(copied.A, np.diag(np.diag(full.A)))
    assert copied.offdiagonal_energy == pytest.approx(0.0)
    assert copied.fit_record["diagonalized_from"] == "fitted_population_operator"


def test_fit_window_is_causal_and_matches_explicit_window() -> None:
    X, _ = _var_data(800)
    t = 500
    model = fit_window(X, t, window=100, min_transitions=10)
    explicit = fit_var(X, np.arange(t - 99, t), spectral_cap=0.995)
    assert model is not None and explicit is not None
    assert np.allclose(model.A, explicit.A, atol=1e-12)
    assert model.fit_record["landmark"] == t
    assert model.fit_record["split"] == "causal_window"


def test_fit_window_requires_enough_transitions() -> None:
    X, _ = _var_data(80)
    assert fit_window(X, 20, window=100, min_transitions=50) is None


def test_shuffle_window_preserves_marginals_without_future_values() -> None:
    rng = np.random.default_rng(2)
    window = rng.normal(size=(120, 4))
    for method in ("circular", "block"):
        shifted, diagnostics = shuffle_window(window, method=method, block=20, seed=4)
        assert shifted.shape == window.shape
        assert diagnostics["n"] == 120
        for j in range(window.shape[1]):
            assert np.allclose(np.sort(shifted[:, j]), np.sort(window[:, j]))
        assert np.all(np.isin(shifted, window))


def test_shuffled_fit_uses_only_the_causal_window() -> None:
    rng = np.random.default_rng(5)
    X = rng.normal(size=(700, 3))
    t = 600
    model = fit_shuffled_window(X, t, window=200, method="circular", block=20, seed=9)
    assert model is not None
    assert model.fit_record["landmark"] == t
    assert model.fit_record["split"] == "causal_window_surrogate"
    assert model.fit_record["surrogate_seed"] == 9


def test_operator_rotation_mode_and_decay() -> None:
    theta = 2.0 * np.pi / 20.0
    A = 0.9 * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    table = mode_table(A, fps=2.0)
    complex_modes = table[table["is_complex"]]
    assert len(complex_modes) == 2
    assert complex_modes["period_frames"].iloc[0] == pytest.approx(20.0, rel=1e-10)
    assert leading_decay_frames(A) == pytest.approx(-1.0 / np.log(0.9))


def test_transient_amplification_exceeds_radius_for_nonnormal_matrix() -> None:
    A = np.array([[0.9, 5.0], [0.0, 0.9]])
    peak, step = transient_amplification(A, horizon=20)
    assert peak > 5.0
    assert 1 <= step <= 20
    assert nonnormality(A) > 0.0


def test_state_geometry_and_transition_math() -> None:
    rng = np.random.default_rng(7)
    train = np.vstack(
        [
            rng.normal(loc=-3.0, scale=0.2, size=(40, 2)),
            rng.normal(loc=0.0, scale=0.2, size=(40, 2)),
            rng.normal(loc=3.0, scale=0.2, size=(40, 2)),
        ]
    )
    model = fit_states(train, n_states=3, seed=1)
    labels = assign_states(train, model.cluster_centers_)
    assert labels.shape == (120,)
    counts = transition_counts(labels, 3)
    T = row_normalize(counts)
    assert np.allclose(T.sum(axis=1), 1.0)
    assert occupancy(labels, 3).sum() == pytest.approx(1.0)
    assert len(dwell_times(labels, 3)) == 3
    assert weighted_transition_tv(T, T, counts) == pytest.approx(0.0)


def test_markov_simulation_follows_deterministic_cycle() -> None:
    transition = np.array([[0.0, 1.0], [1.0, 0.0]])
    paths = simulate_markov(
        transition,
        n_members=10,
        horizon=8,
        initial=np.zeros(10, dtype=np.int64),
        rng=np.random.default_rng(0),
    )
    assert np.array_equal(paths[:, 0], np.zeros(10, dtype=np.int64))
    assert np.array_equal(paths[:, 1], np.ones(10, dtype=np.int64))
    assert np.array_equal(paths[:, 2], np.zeros(10, dtype=np.int64))


def test_state_metrics_exact_match_and_mismatch() -> None:
    truth = np.tile(np.arange(6), 20)
    metrics, detail = state_metrics(truth[None, :], truth, train_labels=truth, n_states=6)
    assert metrics["transition_tv_observed"] == pytest.approx(0.0)
    assert metrics["occupancy_tv_observed"] == pytest.approx(0.0)
    assert detail["model_transition"].shape == (6, 6)
    bad = np.zeros_like(truth)
    bad_metrics, _ = state_metrics(bad[None, :], truth, train_labels=truth, n_states=6)
    assert bad_metrics["occupancy_tv_observed"] > 0.0


def test_sensitivity_impulse_matches_finite_difference_and_analytic_jacobian() -> None:
    A = np.array([[0.7, 0.1, 0.0], [0.0, 0.5, 0.2], [0.1, 0.0, 0.4]])
    model = FrozenLinearModel(A=A, b=np.zeros(3))
    x0 = np.array([0.2, -0.1, 0.4])
    delta = unit_neuron_delta(1, 1.0, 3)
    assert finite_difference_error(model, x0, delta) == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(model.impulse(delta, 5)[-1], np.linalg.matrix_power(A, 5) @ delta)


def test_response_metrics_and_support_fraction() -> None:
    rng = np.random.default_rng(12)
    train = rng.normal(size=(200, 4))
    pca = PCA(n_components=2, random_state=0).fit(train)
    model = FrozenLinearModel(A=0.5 * np.eye(4), b=np.zeros(4))
    metrics, neuron, latent = response_metrics(
        model, pca, np.zeros(4), 2, amplitude=1.0, horizon=20
    )
    assert neuron.shape == (20, 4)
    assert latent.shape == (20, 2)
    assert metrics["finite_difference_error"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["target_component_h20"] > 0.0
    inside = support_fraction(train, train_reference=train)
    outside = support_fraction(np.full((3, 4), 100.0), train_reference=train)
    assert inside < 0.05
    assert outside == pytest.approx(1.0)


def test_matched_controls_exclude_targets_and_choose_closest_variance() -> None:
    variance = np.array([1.0, 1.01, 5.0, 1.02])
    controls = matched_control_indices(variance, 0, exclude={2}, n_controls=2)
    assert controls == [1, 3]


def test_pairwise_disagreement_identical_and_opposite() -> None:
    base = {"a": np.array([1.0, 0.0]), "b": np.array([1.0, 0.0])}
    same = {"a": np.array([1.0, 0.0]), "b": np.array([1.0, 0.0])}
    result = pairwise_disagreement(base, same)
    assert result["mean_response_cosine"] == pytest.approx(1.0)
    assert result["prediction_relative_spread"] == pytest.approx(0.0)
    opposite = {"a": np.array([1.0, 0.0]), "b": np.array([-1.0, 0.0])}
    assert pairwise_disagreement(base, opposite)["mean_response_cosine"] == pytest.approx(-1.0)


def test_response_cosine_handles_zero_vector() -> None:
    assert np.isnan(response_cosine(np.zeros(3), np.ones(3)))


def test_session_block_ci_and_paired_row() -> None:
    rows = []
    for session in range(2):
        for landmark in range(20):
            rows.append({"worm_id": session, "landmark": landmark, "model": "a", "metric": 2.0})
            rows.append({"worm_id": session, "landmark": landmark, "model": "b", "metric": 1.0})
    frame = pd.DataFrame(rows)
    ci = session_block_ci(frame[frame.model == "a"], "metric", block=5, n_boot=100)
    assert ci["mean"] == pytest.approx(2.0)
    paired = paired_row(frame, reference="a", model="b", metric="metric", block=5, n_boot=100)
    assert paired["mean_improvement"] == pytest.approx(1.0)
    assert paired["ci_low"] == pytest.approx(1.0)
    assert paired["fraction_better"] == pytest.approx(1.0)


def test_synthetic_coupled_recovery_and_common_input_confusion() -> None:
    coupled = evaluate_system("coupled", seed=23, n_frames=1200)
    independent = evaluate_system("independent", seed=23, n_frames=1200)
    common = evaluate_system("common", seed=23, n_frames=1200)
    assert coupled["offdiag_recovery_corr"] > 0.8
    assert abs(independent["offdiag_recovery_corr"]) < 0.2 or np.isnan(independent["offdiag_recovery_corr"])
    assert common["fit_offdiag_norm"] > 0.1
    assert common["forecast_ratio_full_over_diag"] > 1.0


def test_synthetic_audit_has_all_scenarios() -> None:
    table = audit(seeds=(11,))
    assert set(table["scenario"]) == {"independent", "coupled", "common", "coupled_common"}
    assert len(table) == 4


@pytest.mark.parametrize(
    "artifact",
    [
        "baseline_reproduction.csv",
        "reproduction_summary.json",
        "cross_neuron_per_landmark.csv",
        "cross_neuron_paired.csv",
        "surrogate_summary.csv",
        "operator_fits.csv",
        "state_transition_scores.csv",
        "sensitivity_responses.csv",
        "sensitivity_identifiability.csv",
        "synthetic_audit.csv",
        "gates.json",
    ],
)
def test_required_artifacts_exist(artifact: str) -> None:
    path = OUT / artifact
    if not path.exists():
        pytest.skip("run scripts/kato_interactions_main.py first")
    assert path.stat().st_size > 0


def test_reproduction_artifact_is_exact_and_j2_j5_are_mean_equivalent() -> None:
    path = OUT / "reproduction_summary.json"
    if not path.exists():
        pytest.skip("run scripts/kato_interactions_main.py first")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if "j2_j5_mean_signed_energy_difference" not in summary:
        pytest.skip("stale artifact; rerun scripts/kato_interactions_main.py")
    assert summary["status"] == "PASS"
    assert summary["max_abs_reproduction_difference"] < 1e-6
    assert abs(summary["j2_j5_mean_signed_energy_difference"]) < 0.01
    assert summary["j2_j5_energy_ci_low"] < 0.0 < summary["j2_j5_energy_ci_high"]


def test_gate_statuses_are_declared() -> None:
    path = OUT / "gates.json"
    if not path.exists():
        pytest.skip("run scripts/kato_interactions_main.py first")
    gates = json.loads(path.read_text(encoding="utf-8"))
    assert set(gates) == {
        "G1_reproduction",
        "G2_cross_neuron_information",
        "G3_operator_reliability",
        "G4_state_transition_fidelity",
        "G5_sensitivity_reliability",
        "G6_empirical_support",
        "G7_generative_validity",
        "G8_biological_validation",
    }
    allowed = {"PASS", "FAIL", "PARTIAL", "INCONCLUSIVE", "NOT_EVALUABLE"}
    assert all(gate["status"] in allowed for gate in gates.values())
    assert gates["G8_biological_validation"]["status"] == "NOT_EVALUABLE"


def test_named_presence_artifact_does_not_impute() -> None:
    path = OUT / "named_neuron_presence.csv"
    if not path.exists():
        pytest.skip("run scripts/kato_interactions_main.py first")
    frame = pd.read_csv(path)
    assert set(frame["neuron"]) == {"AVBL", "AVBR", "AVEL", "AVER", "RIVL", "RIVR"}
    assert (frame.loc[frame.present, "index"] >= 0).all()
    assert (frame.loc[~frame.present, "index"] == -1).all()
