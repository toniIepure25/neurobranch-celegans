"""Animal-level intervention analysis tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from neurobranch.interventions.trial_analysis import (
    analyze_trial_outcomes,
    read_trial_outcomes,
)

ROOT = Path(__file__).resolve().parents[1]


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


def test_read_trial_outcomes_from_csv(tmp_path: Path) -> None:
    path = tmp_path / "outcomes.csv"
    path.write_text(
        "animal_id,arm,outcome\n"
        "t1,target,1\n"
        "t1,target,0\n"
        "t2,target,1\n"
        "s1,sham,0\n"
        "s2,sham,0\n",
        encoding="utf-8",
    )
    rows = read_trial_outcomes(path)
    assert len(rows) == 5
    assert rows[0] == {"animal_id": "t1", "arm": "target", "outcome": 1}
    report = analyze_trial_outcomes(rows, n_bootstrap=100, n_permutations=100)
    assert report["n_target_animals"] == 2
    assert report["n_sham_animals"] == 2


def test_read_trial_outcomes_requires_outcome_column(tmp_path: Path) -> None:
    path = tmp_path / "outcomes.csv"
    path.write_text("animal_id,arm\nt1,target\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outcome"):
        read_trial_outcomes(path)


def test_read_trial_outcomes_rejects_non_binary_and_unknown_arm(tmp_path: Path) -> None:
    bad_outcome = tmp_path / "bad_outcome.csv"
    bad_outcome.write_text("animal_id,arm,outcome\nt1,target,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="binary"):
        read_trial_outcomes(bad_outcome)

    bad_arm = tmp_path / "bad_arm.csv"
    bad_arm.write_text("animal_id,arm,outcome\nt1,random,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="target or sham"):
        read_trial_outcomes(bad_arm)


def test_analysis_cli_writes_report(tmp_path: Path) -> None:
    trials = tmp_path / "outcomes.csv"
    trials.write_text(
        "animal_id,arm,outcome\n"
        "t1,target,1\nt1,target,1\n"
        "t2,target,1\nt2,target,1\n"
        "s1,sham,0\ns1,sham,0\n"
        "s2,sham,0\ns2,sham,0\n",
        encoding="utf-8",
    )
    out = tmp_path / "analysis.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_intervention_trials.py"),
            "--trials",
            str(trials),
            "--out",
            str(out),
            "--n-bootstrap",
            "100",
            "--n-permutations",
            "100",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["format"] == "vienna-intervention-analysis-v1"
    assert report["effect"] == pytest.approx(1.0)
    assert report["n_target_animals"] == 2
