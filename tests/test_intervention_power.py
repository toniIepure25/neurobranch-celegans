"""Cluster-randomized intervention power tests."""

from __future__ import annotations

import pytest

from neurobranch.interventions.power import (
    build_power_report,
    minimum_detectable_effect,
    simulate_power,
)


def test_large_effect_has_high_power() -> None:
    power = simulate_power(
        animals_per_arm=6,
        trials_per_animal=30,
        p_sham=0.30,
        p_target=0.75,
        icc=0.1,
        reps=200,
        seed=7,
    )
    assert power > 0.8


def test_null_effect_is_near_alpha() -> None:
    power = simulate_power(
        animals_per_arm=6,
        trials_per_animal=30,
        p_sham=0.30,
        p_target=0.30,
        icc=0.1,
        reps=400,
        seed=7,
    )
    assert 0.0 <= power <= 0.12


def test_mde_is_deterministic_and_in_range() -> None:
    first = minimum_detectable_effect(
        animals_per_arm=6,
        trials_per_animal=30,
        p_sham=0.30,
        icc=0.2,
        target_power=0.8,
        reps=100,
        seed=11,
    )
    second = minimum_detectable_effect(
        animals_per_arm=6,
        trials_per_animal=30,
        p_sham=0.30,
        icc=0.2,
        target_power=0.8,
        reps=100,
        seed=11,
    )
    assert first == second
    assert 0.0 < first <= 0.7


def test_report_shape() -> None:
    report = build_power_report(
        animals_per_arm=(6,),
        iccs=(0.05, 0.2),
        reps=50,
        seed=3,
    )
    assert report["format"] == "vienna-intervention-power-v1"
    assert report["seed"] == 3
    assert len(report["designs"]) == 1
    assert len(report["sensitivity"]) == 2
    assert {"power", "mde", "animals_per_arm"} <= set(report["designs"][0])


def test_invalid_probability_raises() -> None:
    with pytest.raises(ValueError, match="probability"):
        simulate_power(
            animals_per_arm=6,
            trials_per_animal=30,
            p_sham=1.2,
            p_target=0.5,
            icc=0.2,
            reps=10,
        )
