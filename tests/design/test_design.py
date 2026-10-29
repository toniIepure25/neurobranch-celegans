"""Tests for the design calculator and calibration study."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from neurobranch.design.calibration import coverage_study, simulate_atlas, true_reliability
from neurobranch.design.ceiling import (
    animals_for_reliability,
    design_table,
    required_animals_by_target,
)


def _components() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": ["A", "B", "C"],
            "signal": [1.0, 1.0, 0.1],
            "noise": [1.0, 0.5, 1.0],
            "interaction": [1.0, 0.5, 1.0],
            "n_animals": [20, 20, 20],
            "n_responders": [8, 8, 8],
        }
    )


def test_required_animals_inverts_the_reliability_formula() -> None:
    required = animals_for_reliability(1.0, 1.0, 1.0, 0.6)
    assert required == pytest.approx(3.0)
    signal, noise, interaction = 1.0, 2.0, 0.5
    for goal in (0.4, 0.7, 0.85):
        n = animals_for_reliability(signal, noise, interaction, goal)
        achieved = signal / (signal + (noise + interaction) / n)
        assert achieved == pytest.approx(goal)


def test_required_animals_handles_unattainable_goals() -> None:
    assert math.isinf(animals_for_reliability(0.0, 1.0, 0.0, 0.5))
    assert math.isinf(animals_for_reliability(1.0, 1.0, 0.0, 1.0))
    assert animals_for_reliability(1.0, 0.0, 0.0, 0.5) == 1.0
    assert animals_for_reliability(1.0, 1.0, 0.0, 0.0) == 0.0


def test_design_table_reports_budget_growth() -> None:
    table = design_table(_components(), goals=(0.5, 0.8))
    assert list(table["goal"]) == [0.5, 0.8]
    assert table.loc[1, "median_required"] > table.loc[0, "median_required"]
    assert 0.0 <= table.loc[0, "fraction_feasible"] <= 1.0


def test_required_animals_by_target_has_one_entry_per_target() -> None:
    required = required_animals_by_target(_components(), goal=0.7)
    assert list(required.index) == ["A", "B", "C"]
    assert required["C"] > required["A"]


def test_true_reliability_matches_the_formula() -> None:
    truth = true_reliability(signal_sd=2.0, noise_sd=1.0, interaction_sd=1.0, fold_size=10)
    assert truth == pytest.approx(4.0 / (4.0 + (1.0 / 3.0 + 1.0) / 10.0))


def test_simulated_atlas_has_expected_shape_and_columns() -> None:
    frame = simulate_atlas(n_animals=6, n_targets=2, n_responders=4, n_events=2)
    assert set(frame.columns) == {"dataset_id", "target", "neuron", "event_id", "response_mean"}
    assert frame["dataset_id"].nunique() == 6
    assert frame["target"].nunique() == 2


def test_coverage_study_returns_a_bounded_result() -> None:
    result = coverage_study(n_reps=8, n_animals=12, n_targets=3, seed=5)
    assert result["n_reps"] == 8
    assert 0.0 <= result["coverage"] <= 1.0
    assert result["mean_width"] > 0.0
    assert abs(result["bias"]) < 0.2
