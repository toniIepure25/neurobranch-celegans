"""Tests for the Phase 15 estimand and uncertainty audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurobranch.benchmark.reliability_audit import (
    audit_estimands,
    bayesian_bootstrap,
    calibration_study,
    null_distributions,
    ranking_stability,
)
from neurobranch.benchmark.robustness import _build_target_matrices, animal_jackknife
from neurobranch.benchmark.variance import estimand_table, summarise_reliability_estimands


def _components() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": ["T1", "T2", "T3"],
            "n_animals": [10, 22, 30],
            "n_responders": [5, 5, 5],
            "signal": [1.0, 2.0, 0.5],
            "noise": [1.0, 1.0, 0.5],
            "interaction": [0.0, 0.0, 0.0],
        }
    )


def _simulate(
    *,
    n_animals: int = 10,
    n_targets: int = 3,
    n_responders: int = 6,
    n_events: int = 3,
    seed: int = 15,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    profiles = rng.normal(0.0, 1.0, size=(n_targets, n_responders))
    rows = []
    for target_index in range(n_targets):
        for animal in range(n_animals):
            for responder in range(n_responders):
                for event in range(n_events):
                    rows.append(
                        {
                            "dataset_id": f"a{animal}",
                            "target": f"T{target_index}",
                            "neuron": f"N{responder}",
                            "event_id": f"{animal}:{target_index}:{event}",
                            "response_mean": float(profiles[target_index, responder] + rng.normal()),
                        }
                    )
    return pd.DataFrame(rows)


def test_estimand_table_has_three_populations() -> None:
    table = estimand_table(_components())
    assert table.loc[0, "reliability_obs"] == pytest.approx(10.0 / 11.0)
    assert table.loc[1, "reliability_obs"] == pytest.approx(2.0 / (2.0 + 1.0 / 22.0))
    assert bool(table.loc[0, "extrapolated_22"]) is True
    assert bool(table.loc[1, "in_identified_subset"]) is True
    assert np.isnan(table.loc[0, "reliability_id"])


def test_summary_reports_all_estimands_and_counts() -> None:
    summary = summarise_reliability_estimands(_components())
    assert summary["R_obs"]["n_targets"] == 3
    assert summary["R_22"]["n_targets"] == 3
    assert summary["R_22"]["n_extrapolated"] == 1
    assert summary["R_id"]["n_targets"] == 2
    assert summary["R_id"]["estimate"] > summary["R_obs"]["estimate"]


def test_coverage_aware_jackknife_is_used() -> None:
    frame = _simulate(n_animals=12, n_targets=2, n_responders=5)
    unrestricted = animal_jackknife(frame, fold_size=6, min_animals=4, min_responder_coverage=0.0)
    restricted = animal_jackknife(frame, fold_size=6, min_animals=4, min_responder_coverage=0.8)
    assert unrestricted["status"] == "ok"
    assert restricted["status"] == "ok"
    assert restricted["valid_fraction"] >= 0.95
    assert restricted["n_targets"] == 2


def test_observed_and_fold_jackknife_have_matching_estimates() -> None:
    frame = _simulate(n_animals=12, n_targets=2, n_responders=5)
    matrices, _ = _build_target_matrices(
        frame,
        response="response_mean",
        animal="dataset_id",
        min_events=1,
        min_animals=4,
        min_responder_coverage=0.0,
        fold_size=6,
    )
    combined = audit_estimands(matrices, fold_size=6)
    for key, estimand in (("R_obs", "obs"), ("R_22", "fold"), ("R_id", "id")):
        assert combined["summary"][key]["estimate"] == pytest.approx(
            combined["jackknife"][estimand]["estimate"], abs=1e-12
        )
        assert combined["summary"][key]["n_targets"] == combined["jackknife"][estimand]["n_targets"]


def test_bayesian_bootstrap_is_bounded_and_reproducible() -> None:
    frame = _simulate(n_animals=10, n_targets=2, n_responders=5)
    matrices, _ = _build_target_matrices(
        frame,
        response="response_mean",
        animal="dataset_id",
        min_events=1,
        min_animals=4,
        min_responder_coverage=0.0,
        fold_size=5,
    )
    first = bayesian_bootstrap(matrices, estimand="fold", fold_size=5, n_draws=30, seed=7)
    second = bayesian_bootstrap(matrices, estimand="fold", fold_size=5, n_draws=30, seed=7)
    np.testing.assert_allclose(first, second, equal_nan=True)
    finite = first[np.isfinite(first)]
    assert finite.size >= 20
    assert np.all((finite >= 0.0) & (finite <= 1.0))


def test_null_controls_return_both_distributions() -> None:
    frame = _simulate(n_animals=8, n_targets=2, n_responders=5)
    matrices, _ = _build_target_matrices(
        frame,
        response="response_mean",
        animal="dataset_id",
        min_events=1,
        min_animals=4,
        min_responder_coverage=0.0,
        fold_size=4,
    )
    result = null_distributions(matrices, estimand="fold", fold_size=4, n_draws=20, seed=7)
    assert set(result) == {"permutation", "signflip"}
    for entry in result.values():
        assert np.isfinite(entry["median"])
        assert np.isfinite(entry["q95"])


def test_calibration_study_has_all_estimands() -> None:
    frame = _simulate(n_animals=8, n_targets=2, n_responders=5)
    matrices, _ = _build_target_matrices(
        frame,
        response="response_mean",
        animal="dataset_id",
        min_events=1,
        min_animals=4,
        min_responder_coverage=0.0,
        fold_size=4,
    )
    result = calibration_study(matrices, fold_size=4, n_reps=20, seed=7)
    assert set(result["estimands"]) == {"obs", "fold", "id"}
    assert result["estimands"]["fold"]["n_total"] == 20


def test_ranking_stability_is_fail_closed() -> None:
    frame = _simulate(n_animals=12, n_targets=5, n_responders=5)
    result = ranking_stability(frame, n_splits=5, min_animals=3, seed=7)
    assert result["status"] in {"ok", "no_ranking", "insufficient_splits"}
    assert -1.0 <= result["median_spearman"] <= 1.0
