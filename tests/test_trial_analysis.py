"""Animal-level intervention analysis tests."""

from __future__ import annotations

import pytest

from neurobranch.interventions.trial_analysis import analyze_trial_outcomes


def _rows(target: list[float], sham: list[float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, value in enumerate(target):
        rows.append({"animal_id": f"t{index}", "arm": "target", "outcome": value})
    for index, value in enumerate(sham):
        rows.append({"animal_id": f"s{index}", "arm": "sham", "outcome": value})
    return rows


def test_clear_effect_is_detected() -> None:
    report = analyze_trial_outcomes(
        _rows([1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]),
        n_bootstrap=500,
        n_permutations=500,
        seed=5,
    )
    assert report["effect"] == pytest.approx(1.0)
    assert report["ci_low"] == pytest.approx(1.0)
    assert report["ci_high"] == pytest.approx(1.0)
    assert report["p_value"] < 0.05


def test_no_effect_is_not_significant() -> None:
    report = analyze_trial_outcomes(
        _rows([0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1]),
        n_bootstrap=200,
        n_permutations=200,
        seed=5,
    )
    assert report["effect"] == pytest.approx(0.0)
    assert report["p_value"] > 0.05


def test_missing_arm_raises() -> None:
    with pytest.raises(ValueError, match="target and sham"):
        analyze_trial_outcomes(_rows([1, 1], []))


def test_non_binary_outcome_raises() -> None:
    with pytest.raises(ValueError, match="binary"):
        analyze_trial_outcomes(_rows([1, 2], [0, 0]))


def test_single_animal_per_arm_raises() -> None:
    with pytest.raises(ValueError, match="two animals"):
        analyze_trial_outcomes(_rows([1], [0]))
