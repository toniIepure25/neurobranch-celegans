"""Run the frozen IC1-v2 finite-synthetic acceptance grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neurobranch.abstraction.ic1_v2_contract import InterventionDesign, protocol_fingerprint
from neurobranch.abstraction.ic1_v2_oracle import OracleResult, enumerate_response_laws
from neurobranch.abstraction.ic1_v2_sampling import (
    IntervalEstimate,
    SampleResult,
    estimate_occupancy_weighted,
    estimate_support_diameter,
)
from neurobranch.abstraction.systems import DiscreteFixture

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "research/ic1_v2/IC1_V2_ESTIMAND_SPEC.md"
PINNED_PROTOCOL = "b52c929c6be758b1b47304f238b7ea574e4ebbc689d74193fa2bf94faa45ef92"
SAMPLE_SIZES = (128, 512, 8192)
N_REPLICATES = 256
ENTROPY = 25112026
TOLERANCE = 1e-12
MODES = ("support_diameter", "occupancy_weighted")
CONVERGENCE_CASES = {"ring_l1", "ring_asym", "coincident_null", "coincident_multicategory"}


@dataclass(frozen=True)
class Case:
    name: str
    expectation: str
    design: InterventionDesign
    expected_diameter: float
    expected_weighted: float
    deterministic_diameter: bool = False


def _design(
    name: str,
    weights: tuple[int, ...],
    cells: tuple[int, ...],
    emission: NDArray[np.float64],
    intervention: tuple[str, ...],
    kernels: dict[str, NDArray[np.float64]],
) -> InterventionDesign:
    n_states = len(weights)
    passive = np.tile(np.asarray(weights, dtype=float) / sum(weights), (n_states, 1))
    operators = {"passive": passive, **kernels}
    fixture = DiscreteFixture(
        name=name,
        state_labels=tuple(f"x{state}" for state in range(n_states)),
        partition=np.asarray(cells),
        behavior=emission,
        kernels=operators,
        operators=tuple(operators),
        identifiable_operators=("passive",),
    )
    return InterventionDesign(fixture, weights, intervention, emission)


def _one_hot(labels: tuple[int, ...]) -> NDArray[np.float64]:
    return np.eye(max(labels) + 1)[list(labels)]


def _destinations(*states: int) -> NDArray[np.float64]:
    return np.eye(len(states))[list(states)]


def _ring(name: str, history: int, asym: bool) -> InterventionDesign:
    contexts = (0, 1) if asym else (0,)
    states = list(product(range(3), (1, -1), contexts))
    index = {state: i for i, state in enumerate(states)}
    cells = tuple(position if history == 1 else 3 * ((position - velocity) % 3) + position
                  for position, velocity, _ in states)
    kernels = {"passive": np.zeros((len(states), len(states)))}
    if asym:
        kernels["asym"] = np.zeros_like(kernels["passive"])
    for operator, kernel in kernels.items():
        for row, (position, velocity, context) in enumerate(states):
            kick = context if operator == "asym" else 0
            for next_velocity, probability in ((velocity, 0.85), (-velocity, 0.15)):
                next_position = (position + next_velocity + kick) % 3
                kernel[row, index[next_position, next_velocity, context]] += probability
    return _design(name, (1,) * len(states), cells, _one_hot(tuple(state[0] for state in states)),
                   ("asym" if asym else "passive",), kernels)


def calibration_cases() -> tuple[Case, ...]:
    """Fixed order is part of the seed contract."""
    four_state = _design("four_state", (1, 1, 0, 0), (0, 0, 1, 1), _one_hot((0, 0, 1, 1)),
                         ("probe",), {"probe": _destinations(2, 0, 2, 2)})
    rare_weights = (99_999_999, 1, 0, 0)
    rare = _design("rare", rare_weights, (0, 0, 1, 1), four_state.emission, four_state.operators,
                   {"probe": _destinations(2, 0, 2, 2)})
    impossible = _design("impossible", (1, 1, 0), (0, 0, 0), _one_hot((0, 0, 1)),
                         ("probe",), {"probe": _destinations(0, 1, 2)})
    post_only = _design("post_only", (1, 1, 0, 0), (0, 0, 1, 1), _one_hot((0, 0, 0, 1)),
                        ("probe",), {"probe": _destinations(3, 1, 2, 3)})
    multi = _design("multi_step_exit", (1, 1, 0, 0), (0, 0, 1, 1), _one_hot((0, 0, 0, 1)),
                     ("probe", "follow"),
                     {"probe": _destinations(2, 1, 1, 3), "follow": _destinations(0, 1, 3, 3)})
    repeated = replace(multi, operators=("probe", "probe"))
    identity = {"probe": np.eye(2)}
    null = _design("coincident_null", (1, 1), (0, 0), np.array([[0.20, 0.30, 0.50]] * 2),
                   ("probe",), identity)
    alternative = _design("coincident_multicategory", (1, 1), (0, 0),
                          np.array([[0.65, 0.10, 0.25], [0.15, 0.60, 0.25]]), ("probe",), identity)
    constant = replace(four_state, fixture=replace(four_state.fixture, partition=np.zeros(4, dtype=int)))
    collapsed = replace(constant, emission=np.ones((4, 1)))
    return (
        Case("ring_l1", "Opposite velocities from one position produce different next positions.",
             _ring("ring_l1", 1, False), 0.70, 0.35),
        Case("ring_l2", "Two passive positions determine velocity and each start cell is a singleton.",
             _ring("ring_l2", 2, False), 0.0, 0.0, True),
        Case("ring_asym", "Velocity is known and two passive-compatible contexts disagree after the probe.",
             _ring("ring_asym", 2, True), 0.85, 0.425),
        Case("four_state", "The intervention shrinks equilibrium support but the passive starts still disagree.",
             four_state, 1.0, 0.5, True),
        Case("rare", "A positive state of mass 1e-8 controls the full diameter.",
             rare, 1.0, 0.0000000199999998, True),
        Case("impossible", "An unreachably labelled state aliases with the start cell but is inadmissible.",
             impossible, 0.0, 0.0, True),
        Case("post_only", "A zero-passive-mass terminal state supplies an admissible intervention outcome.",
             post_only, 1.0, 0.5, True),
        Case("multi_step_exit", "The first transition enters a zero-mass bridge and the second uses it.",
             multi, 1.0, 0.5, True),
        Case("persistent", "Repeating the first operator instead leads both starts to the same outcome.",
             repeated, 0.0, 0.0, True),
        Case("coincident_null", "Common support and identical stochastic terminal laws expose plug-in TV bias.",
             null, 0.0, 0.0),
        Case("coincident_multicategory", "Common support with different overlapping three-category emissions.",
             alternative, 0.5, 0.25),
        Case("constant_cell", "A constant start abstraction still aliases responses with a fixed external outcome.",
             constant, 1.0, 0.5, True),
        Case("constant_response", "Replacing the external outcome with a constant makes distances vanish.",
             collapsed, 0.0, 0.0, True),
    )


def _wilson(successes: int, total: int) -> dict[str, float | int]:
    z = 1.959963984540054
    rate = successes / total
    scale = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / scale
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / scale
    return {"hits": successes, "total": total, "rate": rate, "lower": max(0.0, centre - half),
            "upper": min(1.0, centre + half)}


def _covers(estimate: IntervalEstimate, truth: float) -> bool:
    return estimate.lower - TOLERANCE <= truth <= estimate.upper + TOLERANCE


def _summary(estimates: list[IntervalEstimate], truth: float) -> dict:
    if not estimates:
        return {"oracle": truth, "usable": 0}
    values = np.asarray([estimate.value for estimate in estimates])
    error = values - truth
    return {
        "oracle": truth, "usable": len(estimates), "mean": float(values.mean()),
        "bias": float(error.mean()), "rmse": float(np.sqrt((error ** 2).mean())),
        "error_quantiles_2.5_50_97.5": np.quantile(error, [.025, .50, .975]).tolist(),
        "mean_interval_width": float(np.mean([estimate.upper - estimate.lower for estimate in estimates])),
        "coverage": _wilson(sum(_covers(estimate, truth) for estimate in estimates), N_REPLICATES),
    }


def _one_grid_point(
    case: Case, oracle: OracleResult, indices: tuple[int, int, int], sample_size: int
) -> dict:
    case_id, grid_id, mode_id = indices
    use_diameter = mode_id == 0
    overall_truth = oracle.diameter if use_diameter else oracle.weighted
    cell_truth = {label: cell.diameter if use_diameter else cell.weighted for label, cell in oracle.cells.items()}
    samples: list[SampleResult] = []
    failures = []
    omissions: Counter[int] = Counter()
    for replicate in range(N_REPLICATES):
        seed = np.random.SeedSequence(ENTROPY, spawn_key=(case_id, grid_id, replicate, mode_id))
        rng = np.random.default_rng(seed)
        try:
            if use_diameter:
                result = estimate_support_diameter(case.design, trials_per_start=sample_size, rng=rng)
            else:
                result = estimate_occupancy_weighted(
                    case.design, pairs_per_cell=sample_size, trials_per_start=sample_size, rng=rng
                )
            estimates = (result.overall, *result.cells.values())
            if result.cells.keys() != cell_truth.keys() or any(
                not (0 <= e.lower <= e.value <= e.upper <= 1 + TOLERANCE) for e in estimates
            ):
                raise ValueError("missing cell, nonfinite score or invalid interval")
            omissions.update(result.unobserved_passive_starts)
            samples.append(result)
        except (ArithmeticError, RuntimeError, ValueError) as error:
            failures.append({"replicate": replicate, "error": str(error)})

    hits = sum(
        _covers(sample.overall, overall_truth)
        and all(_covers(sample.cells[cell], truth) for cell, truth in cell_truth.items())
        for sample in samples
    )
    deterministic_ok = not (use_diameter and case.deterministic_diameter) or all(
        abs(sample.overall.value - overall_truth) <= TOLERANCE
        and all(abs(sample.cells[cell].value - truth) <= TOLERANCE for cell, truth in cell_truth.items())
        for sample in samples
    )
    overall = _summary([sample.overall for sample in samples], overall_truth)
    checks = {
        "no_invalid_or_missing": len(samples) == N_REPLICATES,
        "deterministic_diameter": deterministic_ok,
        "simultaneous_coverage_at_least_0.92": hits / N_REPLICATES >= .92,
    }
    if sample_size == SAMPLE_SIZES[-1]:
        checks.update({
            "absolute_bias_at_most_0.025": abs(overall.get("bias", math.inf)) <= .025,
            "rmse_at_most_0.03": overall.get("rmse", math.inf) <= .03,
            "interval_width_at_most_0.12": overall.get("mean_interval_width", math.inf) <= .12,
        })
    return {
        "n_per_start": sample_size, "pairs_per_occupied_cell": None if use_diameter else sample_size,
        "overall": overall,
        "cells": {cell: _summary([sample.cells[cell] for sample in samples], truth)
                  for cell, truth in cell_truth.items()},
        "simultaneous_coverage": _wilson(hits, N_REPLICATES),
        "finite_sample_omissions": dict(omissions), "failures": failures,
        "validity_checks": checks, "pass": all(checks.values()),
        "example_actual_trajectory_counts_and_intervals": asdict(samples[0]) if samples else None,
    }


def run_calibration() -> dict:
    protocol_bytes = SPEC.read_bytes()
    identity = protocol_fingerprint(protocol_bytes)
    if identity != PINNED_PROTOCOL:
        raise ValueError("the calibration specification differs from the pinned preimplementation version")
    cases = calibration_cases()
    results = {}
    for case_id, case in enumerate(cases):
        oracle = enumerate_response_laws(case.design)
        targets_ok = (
            abs(oracle.diameter - case.expected_diameter) <= TOLERANCE
            and abs(oracle.weighted - case.expected_weighted) <= TOLERANCE
        )
        runs = {}
        convergence = {}
        for mode_id, mode in enumerate(MODES):
            grid = [_one_grid_point(case, oracle, (case_id, grid_id, mode_id), n)
                    for grid_id, n in enumerate(SAMPLE_SIZES)]
            runs[mode] = grid
            if case.name in CONVERGENCE_CASES:
                small = grid[0]["overall"].get("rmse")
                large = grid[-1]["overall"].get("rmse")
                convergence[mode] = small is not None and large is not None and large <= .40 * small
        results[case.name] = {
            "expectation": case.expectation, "operators_in_order": case.design.operators,
            "passive_weights": case.design.passive_weights, "cell_by_start": case.design.fixture.partition.tolist(),
            "oracle": asdict(oracle), "analytic_expected": [case.expected_diameter, case.expected_weighted],
            "analytic_targets_pass": targets_ok, "convergence_checks": convergence, "grid": runs,
            "pass": targets_ok and all(convergence.values()) and all(
                entry["pass"] for grid in runs.values() for entry in grid
            ),
        }

    rare = cases[4].design
    peak = max(rare.passive_weights)  # The rare fixture has one occupied cell.
    trimmed_weights = tuple(weight if weight * 1_000_000 >= peak else 0 for weight in rare.passive_weights)
    trimmed = enumerate_response_laws(replace(rare, passive_weights=trimmed_weights))
    excluded = Fraction(sum(rare.passive_weights) - sum(trimmed_weights), sum(rare.passive_weights))
    lf = protocol_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    checks = {
        "rare_full_support_preserved": results["rare"]["oracle"]["diameter"] == 1.0,
        "rare_floor_changes_estimand": trimmed.diameter == 0.0,
        "lf_crlf_equivalence": protocol_fingerprint(lf.replace(b"\n", b"\r\n")) == identity,
        "change_changes_identity": protocol_fingerprint(protocol_bytes + b"\nProtocol amendment.\n") != identity,
    }
    source_paths = ("scripts/ic1_v2_acceptance.py", "src/neurobranch/abstraction/systems.py",
                    "src/neurobranch/abstraction/ic1_v2_contract.py", "src/neurobranch/abstraction/ic1_v2_oracle.py",
                    "src/neurobranch/abstraction/ic1_v2_sampling.py", "tests/abstraction/test_ic1_v2.py")
    return {
        "schema": "ic1-v2-acceptance-1",
        "scope": "Known finite synthetic fixtures and declared passive starts",
        "calibration_pass": all(checks.values()) and all(result["pass"] for result in results.values()),
        "additional_checks": checks, "fixtures": results, "fixture_order": list(results),
        "rare_state_sensitivity": {"support_policy": "strictly positive passive mass",
                                   "sensitivity_relative_floor": 1e-6, "trimmed_diameter": trimmed.diameter,
                                   "excluded_passive_mass_exact": str(excluded)},
        "protocol": {"canonical_sha256": identity, "input_bytes_sha256": hashlib.sha256(protocol_bytes).hexdigest()},
        "reproducibility": {
            "seed_entropy": ENTROPY, "seed_key_order": ["fixture", "grid", "replicate", "mode"],
            "replicates_per_grid_point": N_REPLICATES, "sample_sizes": SAMPLE_SIZES, "mode_order": MODES,
            "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in source_paths},
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "system": platform.system()},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="write a new JSON artifact")
    args = parser.parse_args()
    if args.output.exists():
        parser.exit(2, "output already exists; preserve the previous run\n")
    result = run_calibration()
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    passed = sum(fixture["pass"] for fixture in result["fixtures"].values())
    print(f"Calibration {'PASS' if result['calibration_pass'] else 'FAIL'}: {passed}/{len(result['fixtures'])}")
    return 0 if result["calibration_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
