"""Tests for the Phase 10 robustness checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurobranch.benchmark.robustness import (
    animal_half_replication,
    animal_jackknife,
    summary_sensitivity,
)
from neurobranch.benchmark.variance import profile_components_by_target


def _simulate_multi(
    *,
    n_animals: int = 8,
    n_targets: int = 3,
    n_responders: int = 6,
    n_events: int = 3,
    pair_sd: float = 1.0,
    noise_sd: float = 1.0,
    seed: int = 11,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    profiles = rng.normal(0.0, pair_sd, size=(n_targets, n_responders))
    rows = []
    for target_index in range(n_targets):
        for animal in range(n_animals):
            for responder in range(n_responders):
                for event in range(n_events):
                    value = float(profiles[target_index, responder] + rng.normal(0.0, noise_sd))
                    rows.append(
                        {
                            "dataset_id": f"a{animal}",
                            "target": f"T{target_index}",
                            "neuron": f"N{responder}",
                            "event_id": f"{animal}:{target_index}:{event}",
                            "response_mean": value,
                            "response_integral": 2.0 * value,
                            "response_residual": value + rng.normal(0.0, 0.1),
                        }
                    )
    return pd.DataFrame(rows)


def _mean_reliability(frame: pd.DataFrame, *, fold_size: int, min_animals: int) -> float:
    components = profile_components_by_target(frame, min_animals=min_animals)
    if components.empty:
        return float("nan")
    signal = components["signal"].to_numpy(dtype=float)
    error = (components["noise"] + components["interaction"]).to_numpy(dtype=float)
    reliability = np.clip(signal / (signal + error / float(fold_size)), 0.0, 1.0)
    return float(np.nanmean(reliability))


def test_jackknife_se_matches_brute_force_leave_one_animal_out() -> None:
    frame = _simulate_multi(n_animals=8, n_targets=2)
    result = animal_jackknife(frame, fold_size=4, min_animals=4)
    animals = sorted(frame["dataset_id"].unique())
    values = [_mean_reliability(frame[frame["dataset_id"] != animal], fold_size=4, min_animals=4) for animal in animals]
    values = np.asarray(values, dtype=float)
    expected_se = np.sqrt((values.size - 1) / values.size * np.sum((values - values.mean()) ** 2))
    assert result["n_animals"] == values.size
    assert result["se"] == pytest.approx(expected_se, rel=1e-6, abs=1e-9)
    assert result["ci_low"] <= result["estimate"] <= result["ci_high"]


def test_jackknife_handles_too_few_animals() -> None:
    frame = _simulate_multi(n_animals=2, n_targets=1)
    result = animal_jackknife(frame, fold_size=2, min_animals=1)
    assert np.isnan(result["se"])


def test_summary_sensitivity_is_scale_invariant_for_a_scaled_summary() -> None:
    frame = _simulate_multi(n_animals=10, n_targets=4)
    table = summary_sensitivity(
        frame,
        summaries=("response_mean", "response_integral"),
        fold_size=5,
        min_animals=4,
    )
    assert set(table["summary"]) == {"response_mean", "response_integral"}
    reference = table.loc[table["summary"] == "response_mean", "mean_reliability_at_fold"].iloc[0]
    scaled = table.loc[table["summary"] == "response_integral", "mean_reliability_at_fold"].iloc[0]
    assert scaled == pytest.approx(reference, rel=0.02)
    assert table.loc[table["summary"] == "response_integral", "spearman_vs_reference"].iloc[0] > 0.99


def test_half_replication_returns_a_bounded_correlation() -> None:
    frame = _simulate_multi(n_animals=12, n_targets=4, n_events=3)
    result = animal_half_replication(frame, n_splits=5, min_animals=3)
    assert result["n_splits"] == 5
    assert -1.0 <= result["median_spearman"] <= 1.0
    assert result["ci_low"] <= result["median_spearman"] <= result["ci_high"]
