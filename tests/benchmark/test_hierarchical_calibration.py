"""Small contract tests for Phase 12 calibration helpers."""

from __future__ import annotations

from neurobranch.benchmark.hierarchical_calibration import (
    heldout_predictive_check,
    negative_control,
    prepare_templates,
    simulation_based_calibration,
)
from neurobranch.design.calibration import simulate_atlas


def test_sbc_reports_rank_and_coverage() -> None:
    frame = simulate_atlas(n_animals=12, n_targets=2, n_responders=8, n_events=3, seed=2)
    templates = prepare_templates(frame, coverage=0.0)
    out = simulation_based_calibration(
        templates,
        n_reps=10,
        targets_per_rep=1,
        draws=300,
        grid_resolution=30,
        seed=4,
    )
    assert out["n_fits"] == 10
    assert "n22" in out["parameters"]
    assert 0.0 <= out["parameters"]["n22"]["coverage"] <= 1.0
    assert 0.0 <= out["parameters"]["n22"]["mean_rank"] <= 1.0


def test_negative_control_is_near_zero_on_permuted_profiles() -> None:
    frame = simulate_atlas(n_animals=20, n_targets=2, n_responders=10, n_events=3, signal_sd=2.0, seed=3)
    out = negative_control(frame, n_targets=2, draws=300, grid_resolution=30, seed=4)
    assert out["status"] == "ok"
    assert out["median_reliability_n22"] < 0.2


def test_heldout_ppc_returns_records() -> None:
    frame = simulate_atlas(n_animals=16, n_targets=2, n_responders=8, n_events=3, seed=6)
    out = heldout_predictive_check(
        frame,
        n_splits=2,
        max_targets=2,
        draws=200,
        grid_resolution=30,
        seed=7,
    )
    assert out["status"] == "ok"
    assert out["n"] > 0
    assert 0.0 <= out["coverage"] <= 1.0
