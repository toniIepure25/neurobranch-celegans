"""Synthetic validation of the variance-corrected reliability estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurobranch.benchmark.variance import (
    animal_reliability_by_target,
    decompose_variance,
    repeat_reliability_by_target,
    summarise_reliability,
)


def _simulate(
    *,
    n_animals: int = 12,
    n_responders: int = 8,
    n_events: int = 4,
    pair_sd: float = 1.0,
    noise_sd: float = 1.0,
    animal_pair_sd: float = 0.0,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pair = rng.normal(0.0, pair_sd, size=(1, n_responders))
    rows = []
    for animal in range(n_animals):
        animal_offset = rng.normal(0.0, animal_pair_sd, size=(1, n_responders))
        for responder in range(n_responders):
            for event in range(n_events):
                rows.append(
                    {
                        "dataset_id": f"a{animal}",
                        "target": "T1",
                        "neuron": f"N{responder}",
                        "event_id": f"{animal}:{event}",
                        "response_mean": float(pair[0, responder] + animal_offset[0, responder] + rng.normal(0, noise_sd)),
                    }
                )
    return pd.DataFrame(rows)


def test_repeat_reliability_recovers_known_ratio() -> None:
    # tau^2 = 1, sigma^2 = 1, n = 4  ->  reliability = 1 / (1 + 1/4) = 0.8
    frame = _simulate(n_events=4, pair_sd=1.0, noise_sd=1.0)
    table = repeat_reliability_by_target(frame)
    assert len(table) == 1
    value = float(table.loc[0, "repeat_reliability"])
    assert 0.65 <= value <= 0.92
    assert table.loc[0, "repeat_ceiling"] == pytest.approx(np.sqrt(value))


def test_repeat_reliability_increases_with_events() -> None:
    few = repeat_reliability_by_target(_simulate(n_events=2))["repeat_reliability"].iloc[0]
    many = repeat_reliability_by_target(_simulate(n_events=12))["repeat_reliability"].iloc[0]
    assert many > few


def test_animal_mean_reliability_is_high_when_profiles_are_shared() -> None:
    frame = _simulate(n_animals=16, animal_pair_sd=0.0)
    table = animal_reliability_by_target(frame, min_animals=6)
    assert not table.empty
    assert float(table["animal_mean_reliability"].iloc[0]) > 0.85


def test_animal_mean_reliability_improves_with_more_animals() -> None:
    few = animal_reliability_by_target(_simulate(n_animals=6, noise_sd=2.0))
    many = animal_reliability_by_target(_simulate(n_animals=30, noise_sd=2.0))
    assert float(many["animal_mean_reliability"].iloc[0]) > float(few["animal_mean_reliability"].iloc[0])


def test_single_animal_reliability_tracks_signal_to_noise() -> None:
    strong = animal_reliability_by_target(_simulate(pair_sd=5.0, noise_sd=1.0, n_events=3))
    weak = animal_reliability_by_target(_simulate(pair_sd=0.5, noise_sd=1.0, n_events=3))
    assert float(strong["single_animal_reliability"].iloc[0]) > float(weak["single_animal_reliability"].iloc[0])


def test_summary_reports_target_level_interval() -> None:
    table = pd.concat([repeat_reliability_by_target(_simulate(n_animals=6, n_events=3, seed=s)) for s in (1, 2, 3)], ignore_index=True)
    summary = summarise_reliability(table)
    assert summary.repeat_ci_low <= summary.mean_repeat_reliability <= summary.repeat_ci_high
    assert 0.0 <= summary.repeat_ci_low <= summary.repeat_ci_high <= 1.0


def test_variance_decomposition_returns_finite_components() -> None:
    components = decompose_variance(_simulate(n_animals=10, n_events=4))
    payload = components.as_dict()
    assert payload["n_pairs"] == 10 * 8
    for key in ("within_event", "pair", "target", "animal"):
        assert np.isfinite(payload[key])
    assert payload["pair"] >= 0.0


def test_summary_handles_empty_table() -> None:
    summary = summarise_reliability(pd.DataFrame())
    assert summary.n_targets == 0
    assert np.isnan(summary.mean_repeat_reliability)


def test_missing_columns_raise() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        repeat_reliability_by_target(pd.DataFrame({"target": ["A"]}))


