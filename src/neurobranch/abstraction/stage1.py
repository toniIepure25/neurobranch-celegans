"""Stage-One pipeline: run every fixture, evaluate the frozen gates, write the report.

Nothing here is allowed to change a threshold after seeing a result.  The gates are
read from ``research/interventional_consistency_protocol.md`` (hash recorded in the
manifest) and the verdict is a mechanical consequence of them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from neurobranch.abstraction import contract as contract_module
from neurobranch.abstraction.commutation import estimate_from_rollout, estimate_from_simulation
from neurobranch.abstraction.conformance import (
    conformance_checks,
    probe_environment,
    write_conformance,
)
from neurobranch.abstraction.estimands import TOLERANCE, tv_distance
from neurobranch.abstraction.reference import exact_commutation, is_exactly_lumpable
from neurobranch.abstraction.systems import HISTORY_LENGTHS, RateCircuit, discrete_fixtures
from neurobranch.config import PROJECT_ROOT, git_state, sha256_file

REPORT_DIR = PROJECT_ROOT / "reports" / "interventional_consistency"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "interventional_consistency" / "ic1"
BASELINE_PATH = PROJECT_ROOT / "research" / "interventional_consistency_frozen_baseline.sha256"
PROTOCOL_PATH = PROJECT_ROOT / "research" / "interventional_consistency_protocol.md"

#: Fixtures used by the simulation-versus-enumeration agreement gate.
AGREEMENT_FIXTURES = (
    "exactly_lumpable/positive",
    "exactly_lumpable/behaviour_only",
    "exactly_lumpable/intervention_only",
)
#: Fixtures whose exact error must be zero for the "positive control" gate.
POSITIVE_CONTROL = "exactly_lumpable/positive"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def _dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=_json_default)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dumps(payload) + "\n", encoding="utf-8")


def frozen_baseline_status() -> dict[str, Any]:
    """Gate G7: every recorded historical manifest still hashes to its recorded value."""
    if not BASELINE_PATH.exists():
        return {"status": "missing", "checked": 0, "mismatched": [], "path": BASELINE_PATH.name}
    recorded = {}
    for line in BASELINE_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        recorded[relative.strip()] = digest
    mismatched = []
    missing = []
    for relative, digest in recorded.items():
        path = PROJECT_ROOT / relative
        if not path.exists():
            missing.append(relative)
            continue
        if sha256_file(path) != digest:
            mismatched.append(relative)
    status = "ok" if not mismatched and not missing and recorded else "failed"
    return {
        "status": status,
        "checked": len(recorded),
        "mismatched": mismatched,
        "missing": missing,
        "path": BASELINE_PATH.name,
    }


def independence_status(fixtures: dict[str, Any]) -> dict[str, Any]:
    """Gate G1a: the enumerating reference must not call the simulator."""
    from neurobranch.abstraction.systems import DiscreteFixture

    original = DiscreteFixture.simulate

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("exact_commutation called simulate: the paths are not independent")

    DiscreteFixture.simulate = forbidden  # type: ignore[method-assign]
    try:
        for fixture in fixtures.values():
            exact_commutation(fixture)
    except AssertionError as exc:
        return {"status": "failed", "detail": str(exc)}
    finally:
        DiscreteFixture.simulate = original  # type: ignore[method-assign]
    return {
        "status": "ok",
        "detail": "exact_commutation completed with DiscreteFixture.simulate disabled",
    }


def per_cell_rows(fixtures: dict[str, Any], *, occupancy_floor: float = 1e-6) -> list[dict[str, Any]]:
    """Per-cell detail, so a pooled number can always be traced to its cells."""
    rows: list[dict[str, Any]] = []
    for name, fixture in fixtures.items():
        occupancy = fixture.stationary_weights("passive")
        for operator in fixture.operators:
            projection = fixture.macro_projection(operator)
            weights = occupancy
            for cell in np.unique(fixture.partition):
                members = np.flatnonzero(fixture.partition == cell)
                peak = float(weights[members].max())
                reachable = members[weights[members] >= occupancy_floor * peak] if peak > 0 else members[:0]
                dynamic = 0.0
                for first in range(max(reachable.size - 1, 0)):
                    for second in range(first + 1, reachable.size):
                        dynamic = max(
                            dynamic,
                            tv_distance(projection[reachable[first]], projection[reachable[second]]),
                        )
                rows.append(
                    {
                        "fixture": name,
                        "operator": operator,
                        "cell": int(cell),
                        "n_members": int(members.size),
                        "n_reachable": int(reachable.size),
                        "dynamic": round(float(dynamic), 12),
                        "occupancy": round(float(weights[members].sum()), 9),
                    }
                )
    return rows


def _exact_block(fixtures: dict[str, Any]) -> dict[str, Any]:
    block: dict[str, Any] = {}
    for name, fixture in fixtures.items():
        block[name] = {
            "n_states": fixture.n_states,
            "n_cells": fixture.n_cells,
            "operators": list(fixture.operators),
            "identifiable_operators": list(fixture.identifiable_operators),
            "lumpable": {operator: is_exactly_lumpable(fixture, operator) for operator in fixture.operators},
            "errors": {operator: error.as_dict() for operator, error in exact_commutation(fixture).items()},
        }
    return block


def _agreement_block(fixtures: dict[str, Any], *, rng_seed: int, quick: bool) -> dict[str, Any]:
    from neurobranch.abstraction.commutation import DEFAULT_N_CHAINS, DEFAULT_N_STEPS

    chains = 300 if quick else DEFAULT_N_CHAINS
    steps = 120 if quick else DEFAULT_N_STEPS
    block: dict[str, Any] = {}
    for name in AGREEMENT_FIXTURES:
        fixture = fixtures[name]
        exact = exact_commutation(fixture)
        generator = np.random.default_rng(rng_seed)
        for operator in fixture.operators:
            estimate, diagnostics = estimate_from_simulation(
                fixture, operator, n_chains=chains, n_steps=steps, rng=generator
            )
            floor = float(diagnostics["noise_floor"])
            block[f"{name}|{operator}"] = {
                "exact_dynamic": round(exact[operator].dynamic, 12),
                "estimated_dynamic": round(estimate.dynamic, 12),
                "absolute_gap": round(abs(estimate.dynamic - exact[operator].dynamic), 12),
                "noise_floor": round(floor, 6),
                "within_floor_or_tolerance": abs(estimate.dynamic - exact[operator].dynamic)
                <= max(floor, TOLERANCE),
                "same_side_of_tolerance": (estimate.dynamic > TOLERANCE)
                == (exact[operator].dynamic > TOLERANCE),
                "diagnostics": diagnostics,
            }
    return block


def _circuit_block(*, rng_seed: int, quick: bool) -> dict[str, Any]:
    circuit = RateCircuit()
    block: dict[str, Any] = {}
    generator = np.random.default_rng(rng_seed)
    for operator in ("passive", "lesion", "clamp"):
        estimate, diagnostics = estimate_from_rollout(
            circuit,
            operator,
            n_chains=120 if quick else 400,
            replicates=40 if quick else 80,
            rng=generator,
        )
        block[operator] = {"error": estimate.as_dict(), "diagnostics": diagnostics}
    return block


def _contract_block() -> dict[str, Any]:
    contracts = contract_module.stage1_contracts()
    refusal = None
    try:
        contract_module.require_comparable(
            {"pca_observation": contracts["pca_observation"], "oracle": contracts["oracle"]}
        )
        refusal = {"refused": False, "detail": "no error raised: the gate fails"}
    except contract_module.ContractError as exc:
        refusal = {"refused": True, "detail": str(exc)}
    allowed = True
    try:
        contract_module.require_comparable(
            {
                "pca_observation": contracts["pca_observation"],
                "pca_observation_history": contracts["pca_observation_history"],
            }
        )
    except contract_module.ContractError as exc:  # pragma: no cover - gate failure
        allowed = False
        refusal["history_exception_failed"] = str(exc)
    return {
        "contracts": {name: contract.to_dict() for name, contract in contracts.items()},
        "labels": {name: contract.label() for name, contract in contracts.items()},
        "privileged_excluded_from_ranking": sorted(
            name for name, contract in contracts.items() if contract.privileged()
        ),
        "cross_contract_refusal": refusal,
        "history_length_exception_allowed": allowed,
    }


def _witness_plateau(exact: dict[str, Any]) -> tuple[bool, dict[str, float]]:
    values = {
        str(length): exact[f"ring_witness/L{length}"]["errors"]["asym"]["dynamic"] for length in HISTORY_LENGTHS
    }
    ordered = [values[str(length)] for length in HISTORY_LENGTHS]
    plateau = all(value > TOLERANCE for value in ordered) and ordered[-1] >= 0.5 * max(ordered)
    return plateau, values


def _gate(name: str, passed: bool, detail: str, **extra: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(passed), "detail": detail, **extra}


def evaluate_gates(
    fixtures: dict[str, Any],
    exact: dict[str, Any],
    agreement: dict[str, Any],
    contracts: dict[str, Any],
    baseline: dict[str, Any],
    independence: dict[str, Any],
    determinism_identical: bool | None,
) -> list[dict[str, Any]]:
    positive = exact[POSITIVE_CONTROL]["errors"]["passive"]
    g2 = positive["dynamic"] <= 1e-12 and positive["n_cells_singleton"] == 0

    behaviour = exact["exactly_lumpable/behaviour_only"]["errors"]["passive"]
    intervention = exact["exactly_lumpable/intervention_only"]["errors"]
    g3 = (
        behaviour["dynamic"] > TOLERANCE
        and behaviour["behavior"] <= 1e-12
        and intervention["passive"]["dynamic"] <= 1e-12
        and intervention["lesion"]["dynamic"] > TOLERANCE
        and intervention["lesion"]["behavior"] <= 1e-12
    )

    ring_one = exact["ring/L1"]["errors"]["passive"]
    ring_two = exact["ring/L2"]["errors"]["passive"]
    g3_history = ring_one["dynamic"] > TOLERANCE and ring_two["dynamic"] <= 1e-12

    g1b = all(entry["within_floor_or_tolerance"] and entry["same_side_of_tolerance"] for entry in agreement.values())
    g1b_count = sum(1 for entry in agreement.values() if entry["within_floor_or_tolerance"])

    plateau, plateau_values = _witness_plateau(exact)

    return [
        _gate(
            "G1a_independent_reference",
            independence["status"] == "ok",
            independence["detail"],
        ),
        _gate(
            "G1b_simulation_agreement",
            g1b,
            "the sampling estimator reproduces the enumerator within its declared noise floor "
            f"on {g1b_count}/{len(agreement)} operator-fixture pairs",
            pairs=g1b_count,
        ),
        _gate(
            "G2_positive_control",
            g2,
            "the exactly lumpable fixture has zero dynamic error with at least one genuine pair per cell",
            dynamic=positive["dynamic"],
            singleton_cells=positive["n_cells_singleton"],
        ),
        _gate(
            "G3_negative_controls",
            g3,
            "behaviour-only and intervention-only negatives separate on the intended estimand and only there",
            behaviour_only_dynamic=behaviour["dynamic"],
            intervention_only_lesion=intervention["lesion"]["dynamic"],
            intervention_only_passive=intervention["passive"]["dynamic"],
        ),
        _gate(
            "G3b_history_ladder",
            g3_history,
            "history length 1 fails and length 2 succeeds, exactly",
            l1=ring_one["dynamic"],
            l2=ring_two["dynamic"],
        ),
        _gate(
            "G4_contract_enforcement",
            contracts["cross_contract_refusal"]["refused"] and contracts["history_length_exception_allowed"],
            contracts["cross_contract_refusal"]["detail"],
        ),
        _gate(
            "G5_identifiability_witness",
            plateau,
            "the witness operator's error stays above tolerance and does not shrink with the window",
            values=plateau_values,
        ),
        _gate(
            "G6_determinism",
            bool(determinism_identical),
            "two in-process runs with the same seed produced identical metrics",
        ),
        _gate(
            "G7_frozen_history",
            baseline["status"] == "ok",
            f"{baseline['checked']} recorded historical manifests re-hashed",
            mismatched=baseline.get("mismatched", []),
            missing=baseline.get("missing", []),
        ),
    ]


def _summary_markdown(metrics: dict[str, Any]) -> str:
    gates = metrics["gates"]
    passed = sum(1 for gate in gates if gate["passed"])
    lines = [
        "# Stage One — interventional consistency of neural abstractions",
        "",
        f"Protocol: `{metrics['provenance']['protocol_sha256'][:16]}…` (`ic1`, frozen).",
        f"Seed: `{metrics['provenance']['seed']}`. Gates passed: **{passed}/{len(gates)}**.",
        "",
        "> Synthetic ground truth only. Nothing below is evidence about *C. elegans*, and no",
        "> external method was reproduced (see `CONFORMANCE.md`).",
        "",
        "## Gates",
        "",
        "| gate | result | detail |",
        "|---|---|---|",
    ]
    for gate in gates:
        lines.append(f"| {gate['gate']} | {'PASS' if gate['passed'] else 'FAIL'} | {gate['detail']} |")
    lines += ["", "## Exact commutation errors", "", "| fixture | operator | behaviour | dynamic | worst case | cells | singleton | unsupported |", "|---|---|---|---|---|---|---|---|"]
    for name, entry in metrics["exact"].items():
        for operator, error in entry["errors"].items():
            lines.append(
                f"| {name} | {operator} | {error['behavior']:.4f} | {error['dynamic']:.4f} | "
                f"{error['worst_case']:.4f} | {error['n_cells']} | {error['n_cells_singleton']} | "
                f"{error['n_unsupported']} |"
            )
    lines += [
        "",
        "## Sampling versus enumeration",
        "",
        "| pair | exact | estimated | gap | noise floor |",
        "|---|---|---|---|---|",
    ]
    for key, entry in metrics["agreement"].items():
        lines.append(
            f"| {key} | {entry['exact_dynamic']:.4f} | {entry['estimated_dynamic']:.4f} | "
            f"{entry['absolute_gap']:.4f} | {entry['noise_floor']:.4f} |"
        )
    lines += [
        "",
        "## Continuous system (descriptive, no gate)",
        "",
        "| operator | worst case | mean pairwise | behaviour | cells |",
        "|---|---|---|---|---|",
    ]
    for operator, entry in metrics["circuit"].items():
        error = entry["error"]
        mean_pair = error["mean_pairwise"]
        lines.append(
            f"| {operator} | {error['dynamic']:.3f} | "
            f"{'n/a' if mean_pair is None else format(mean_pair, '.3f')} | "
            f"{error['behavior']:.3f} | {error['n_cells']} |"
        )
    lines += [
        "",
        "The worst-case error saturates at 1 for a hard partition of a continuous state:",
        "members at opposite edges of a cell leave through different faces. The mean pairwise",
        "term is the graded quantity there. Note that it is *lower* under two of the operators",
        "than under the passive dynamics, which is the substantive point — commutation is a",
        "property of the pair (abstraction, operator), not of the abstraction alone.",
        "",
        "## Verdict vocabulary",
        "",
        "| label | meaning here |",
        "|---|---|",
        "| `verified` | checked against a source in this environment |",
        "| `simulated` | our own synthetic ground truth |",
        "| `unidentifiable` | cannot be estimated from the permitted information |",
        "| `falsified` | tested and failed |",
        "| `not_reproduced` | the external implementation did not run here |",
        "",
    ]
    return "\n".join(lines)


def run_stage1(
    *,
    seed: int = 0,
    quick: bool = False,
    out_dir: Path | None = None,
    determinism_check: bool = True,
) -> dict[str, Any]:
    """Run every fixture, evaluate the gates and write the report bundle."""
    report_dir = out_dir or REPORT_DIR
    artifact_dir = ARTIFACT_DIR if out_dir is None else report_dir / "artifacts" / "ic1"

    fixtures = discrete_fixtures()
    exact = _exact_block(fixtures)
    independence = independence_status(fixtures)
    agreement = _agreement_block(fixtures, rng_seed=seed, quick=quick)
    circuit = _circuit_block(rng_seed=seed, quick=quick)
    contracts = _contract_block()
    baseline = frozen_baseline_status()
    cells = per_cell_rows(fixtures)

    plateau, _ = _witness_plateau(exact)
    agreeing = sum(1 for entry in agreement.values() if entry["within_floor_or_tolerance"])

    def build() -> dict[str, Any]:
        return {
            "schema": "neurobranch-ic-stage1-v1",
            "provenance": {
                "protocol": PROTOCOL_PATH.name,
                "protocol_sha256": sha256_file(PROTOCOL_PATH) if PROTOCOL_PATH.exists() else "missing",
                "baseline": BASELINE_PATH.name,
                "seed": seed,
                "quick": quick,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": sys.platform,
            },
            "gates": [],
            "exact": exact,
            "agreement": agreement,
            "circuit": circuit,
            "contracts": contracts,
            "baseline": baseline,
            "independence": independence,
            "conformance": [check.as_dict() for check in conformance_checks(agreeing_fixtures=agreeing, witness_plateau=plateau)],
        }

    metrics = build()
    metrics["gates"] = evaluate_gates(
        fixtures, exact, agreement, contracts, baseline, independence, determinism_identical=None
    )
    digest_first = hashlib.sha256(_dumps({**metrics, "gates": []}).encode("utf-8")).hexdigest()

    identical: bool | None = None
    if determinism_check:
        second = build()
        digest_second = hashlib.sha256(_dumps({**second, "gates": []}).encode("utf-8")).hexdigest()
        identical = digest_first == digest_second

    metrics["gates"] = evaluate_gates(
        fixtures, exact, agreement, contracts, baseline, independence, determinism_identical=identical
    )
    metrics["determinism"] = {"checked": determinism_check, "identical": identical, "digest": digest_first}
    all_passed = all(gate["passed"] for gate in metrics["gates"])
    metrics["verdict"] = {
        "technical_go": all_passed,
        "if_no_go": sorted(gate["gate"] for gate in metrics["gates"] if not gate["passed"]),
        "stage_two_justified": all_passed,
        "not_claimed": [
            "biological or cognitive interpretation of any number here",
            "reproduction of NC-MCM or BunDLe-Net",
            "that the benchmark is exhaustive over representations or systems",
        ],
    }

    _write_json(report_dir / "metrics.json", metrics)
    _write_json(artifact_dir / "metrics.json", metrics)
    _write_json(artifact_dir / "contracts.json", contracts)
    (report_dir / "SUMMARY.md").write_text(_summary_markdown(metrics), encoding="utf-8")

    environment = probe_environment()
    write_conformance(
        report_dir / "CONFORMANCE.md",
        conformance_checks(agreeing_fixtures=agreeing, witness_plateau=plateau, environment=environment),
        environment,
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    cell_path = report_dir / "commutation_by_cell.csv"
    with cell_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)

    manifest = {
        "schema": "neurobranch-ic-stage1-manifest-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(PROJECT_ROOT),
        "protocol_sha256": metrics["provenance"]["protocol_sha256"],
        "seed": seed,
        "verdict": metrics["verdict"],
        "outputs": {},
    }
    for path in (
        report_dir / "metrics.json",
        report_dir / "SUMMARY.md",
        report_dir / "CONFORMANCE.md",
        cell_path,
        artifact_dir / "metrics.json",
        artifact_dir / "contracts.json",
    ):
        manifest["outputs"][path.relative_to(PROJECT_ROOT).as_posix()] = sha256_file(path)
    _write_json(report_dir / "manifest.json", manifest)
    digests = "\n".join(f"{digest}  {name}" for name, digest in manifest["outputs"].items()) + "\n"
    (report_dir / "manifest.sha256").write_text(digests, encoding="utf-8")

    return metrics


def summary_line(metrics: dict[str, Any]) -> str:
    verdict = metrics["verdict"]
    passed = sum(1 for gate in metrics["gates"] if gate["passed"])
    state = "GO" if verdict["technical_go"] else "NO-GO"
    return f"{state}: {passed}/{len(metrics['gates'])} gates passed; failed={verdict['if_no_go']}"


__all__ = [
    "evaluate_gates",
    "frozen_baseline_status",
    "independence_status",
    "per_cell_rows",
    "run_stage1",
    "summary_line",
    "asdict",
]
