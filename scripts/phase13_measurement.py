"""Phase 13: projection-consistent reliability and event-level variance."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.benchmark.measurement import fit_measurement_reliability
from neurobranch.benchmark.measurement_calibration import (
    permutation_control,
    prepare_measurement_templates,
    prior_sensitivity,
    simulation_based_calibration,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "reports" / "phase3" / "randi_trial_table_phase3.csv.gz"
DEFAULT_OUT = ROOT / "reports" / "phase13"
PHASE11_REFERENCE = 0.314
PRIMARY_MODEL = "m2_event_variance"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _runner_config(quick: bool) -> dict[str, Any]:
    if quick:
        return {
            "draws": 500,
            "grid": 40,
            "event_grid": 80,
            "sbc_m1_reps": 40,
            "sbc_m2_reps": 30,
            "targets_per_rep": 2,
            "negative_targets": 30,
            "prior_draws": 500,
            "prior_grid": 40,
            "coverage_draws": 500,
        }
    return {
        "draws": 2500,
        "grid": 80,
        "event_grid": 200,
        "sbc_m1_reps": 1000,
        "sbc_m2_reps": 600,
        "targets_per_rep": 3,
        "negative_targets": 100,
        "prior_draws": 1200,
        "prior_grid": 50,
        "coverage_draws": 1200,
    }


def _gate(name: str, requirement: str, passed: bool, observed: Any) -> dict[str, Any]:
    return {
        "gate": name,
        "requirement": requirement,
        "observed": _clean(observed),
        "result": "pass" if passed else "fail",
    }


def _coverage_row(result: Any, coverage: float) -> dict[str, Any]:
    summary = result.summary
    return {
        "coverage": float(coverage),
        "n_targets": summary.get("n_targets"),
        "n_ok": summary.get("n_ok"),
        "pooled_reliability_n22": summary.get("pooled_reliability_median"),
        "pooled_ceiling_n22": summary.get("pooled_ceiling_median"),
        "variance_floor_fraction": summary.get("variance_floor_fraction"),
    }


def run(*, quick: bool = False, out: Path = DEFAULT_OUT) -> dict[str, Any]:
    cfg = _runner_config(quick)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(
        DEFAULT_INPUT,
        usecols=["dataset_id", "event_id", "target", "neuron", "response_mean"],
    )
    templates = prepare_measurement_templates(frame, coverage=0.8)
    if not templates:
        raise RuntimeError("no eligible templates")

    real: dict[str, Any] = {}
    for model in ("m0_reference", "m1_projection_scaled", PRIMARY_MODEL):
        real[model] = fit_measurement_reliability(
            frame,
            model=model,
            coverage=0.8,
            draws=cfg["draws"],
            grid_resolution=cfg["grid"],
            event_grid=cfg["event_grid"],
        )
        real[model].by_target.to_csv(out / f"hierarchical_by_target_{model}.csv", index=False)

    coverage_results = {}
    for coverage in (0.5, 0.8, 1.0):
        result = fit_measurement_reliability(
            frame,
            model=PRIMARY_MODEL,
            coverage=coverage,
            draws=cfg["coverage_draws"],
            grid_resolution=cfg["grid"],
            event_grid=cfg["event_grid"],
        )
        coverage_results[coverage] = _coverage_row(result, coverage)
    pd.DataFrame(coverage_results.values()).to_csv(out / "coverage_sensitivity.csv", index=False)

    min_events_results = []
    for min_events in (1, 2):
        result = fit_measurement_reliability(
            frame,
            model=PRIMARY_MODEL,
            coverage=0.8,
            min_events=min_events,
            draws=cfg["coverage_draws"],
            grid_resolution=cfg["grid"],
            event_grid=cfg["event_grid"],
        )
        row = _coverage_row(result, 0.8)
        row["min_events"] = min_events
        min_events_results.append(row)
    pd.DataFrame(min_events_results).to_csv(out / "min_events_sensitivity.csv", index=False)

    sbc_m1 = simulation_based_calibration(
        templates,
        model="m1_projection_scaled",
        n_reps=cfg["sbc_m1_reps"],
        targets_per_rep=cfg["targets_per_rep"],
        draws=cfg["draws"],
        grid_resolution=cfg["grid"],
        event_grid=cfg["event_grid"],
        seed=31,
    )
    sbc_m2 = simulation_based_calibration(
        templates,
        model=PRIMARY_MODEL,
        n_reps=cfg["sbc_m2_reps"],
        targets_per_rep=cfg["targets_per_rep"],
        draws=cfg["draws"],
        grid_resolution=cfg["grid"],
        event_grid=cfg["event_grid"],
        seed=32,
    )
    _write_json(out / "sbc_m1.json", sbc_m1)
    _write_json(out / "sbc.json", sbc_m2)

    negative = permutation_control(
        templates,
        model=PRIMARY_MODEL,
        n_targets=cfg["negative_targets"],
        draws=cfg["coverage_draws"],
        grid_resolution=cfg["grid"],
        event_grid=cfg["event_grid"],
        seed=43,
    )
    signflip = permutation_control(
        templates,
        model=PRIMARY_MODEL,
        n_targets=cfg["negative_targets"],
        draws=cfg["coverage_draws"],
        grid_resolution=cfg["grid"],
        event_grid=cfg["event_grid"],
        seed=44,
        sign_flip=True,
    )
    _write_json(out / "negative_control.json", negative)
    _write_json(out / "signflip_control.json", signflip)

    prior = prior_sensitivity(
        frame,
        model=PRIMARY_MODEL,
        draws=cfg["prior_draws"],
        grid_resolution=cfg["prior_grid"],
        seed=47,
    )
    _write_json(out / "prior_sensitivity.json", prior)

    primary = real[PRIMARY_MODEL].summary
    sbc_n22 = sbc_m2["parameters"].get("n22", {})
    one_event = sbc_m2.get("one_event_stratum", {})
    gates = [
        _gate(
            "G1a_sbc_coverage",
            "22-animal coverage in [0.93, 0.97]",
            bool(0.93 <= float(sbc_n22.get("coverage", np.nan)) <= 0.97)
            if np.isfinite(sbc_n22.get("coverage", np.nan))
            else False,
            sbc_n22.get("coverage"),
        ),
        _gate(
            "G1b_sbc_rank",
            "KS rank-uniformity p > 0.01",
            bool(float(sbc_n22.get("ks_pvalue", 0.0)) > 0.01),
            sbc_n22.get("ks_pvalue"),
        ),
        _gate(
            "G1c_grid_edge",
            "maximum edge posterior mass < 0.05",
            bool(float(sbc_m2.get("max_edge_mass") or 1.0) < 0.05),
            sbc_m2.get("max_edge_mass"),
        ),
        _gate(
            "G1d_fit_integrity",
            "failed or edge-dominated fits <= 5%",
            bool(
                (int(sbc_m2["n_failed_fits"]) + int(sbc_m2["n_edge_failures"]))
                / max(int(sbc_m2["n_fits"]), 1)
                <= 0.05
            ),
            {
                "failed": sbc_m2["n_failed_fits"],
                "edge_failures": sbc_m2["n_edge_failures"],
                "n_fits": sbc_m2["n_fits"],
            },
        ),
        _gate(
            "G1e_one_event_stratum",
            "coverage in [0.90, 0.98] when n=1 fraction >= 0.5",
            bool(0.90 <= float(one_event.get("coverage", np.nan)) <= 0.98)
            if np.isfinite(one_event.get("coverage", np.nan))
            else False,
            one_event.get("coverage"),
        ),
        _gate(
            "G2_heldout_ppc",
            "coverage >= 0.90 and MAE <= 0.10",
            False,
            {"status": "not_run", "reason": "G3 failed before a valid M2 PPC was implemented"},
        ),
        _gate(
            "G3_negative_control",
            "median <= 0.05 and q95 <= 0.15",
            bool(
                float(negative["median_reliability_n22"]) <= 0.05
                and float(negative["q95_reliability_n22"]) <= 0.15
            ),
            {
                "median": negative["median_reliability_n22"],
                "q95": negative["q95_reliability_n22"],
            },
        ),
        _gate(
            "G3b_signflip_control",
            "median <= 0.05 and q95 <= 0.15",
            bool(
                float(signflip["median_reliability_n22"]) <= 0.05
                and float(signflip["q95_reliability_n22"]) <= 0.15
            ),
            {
                "median": signflip["median_reliability_n22"],
                "q95": signflip["q95_reliability_n22"],
            },
        ),
        _gate(
            "G4_prior_sensitivity",
            "pooled R(22) range <= 0.05",
            bool(float(prior["range"]) <= 0.05),
            prior["range"],
        ),
        _gate(
            "G5_event_variance_recovery",
            "posterior coverage of omega^2 in [0.90, 0.98]",
            bool(0.90 <= float(sbc_m2.get("event_variance_coverage", np.nan)) <= 0.98)
            if np.isfinite(sbc_m2.get("event_variance_coverage", np.nan))
            else False,
            sbc_m2.get("event_variance_coverage"),
        ),
    ]
    reference_difference = abs(float(primary["pooled_reliability_median"]) - PHASE11_REFERENCE)
    gates.append(
        _gate(
            "G6_phase11_agreement",
            "pooled R(22) within 0.05 of 0.314",
            bool(reference_difference <= 0.05),
            {
                "phase11_reference": PHASE11_REFERENCE,
                "observed": primary["pooled_reliability_median"],
                "absolute_difference": reference_difference,
            },
        )
    )
    decision = "adopted" if all(gate["result"] == "pass" for gate in gates) else "not_adopted_gate_failures"
    summary = {
        "protocol": "research/phase13_protocol.md",
        "deviations": "research/phase13_deviations.md",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "quick": bool(quick),
        "input": str(DEFAULT_INPUT.relative_to(ROOT)),
        "input_sha256": _sha256(DEFAULT_INPUT),
        "primary_model": PRIMARY_MODEL,
        "phase11_reference": PHASE11_REFERENCE,
        "decision": decision,
        "real": {model: result.summary for model, result in real.items()},
        "coverage": coverage_results,
        "min_events": min_events_results,
        "sbc_m1": sbc_m1,
        "sbc_m2": sbc_m2,
        "negative_control": negative,
        "signflip_control": signflip,
        "prior_sensitivity": prior,
        "gates": gates,
        "n_gates_failed": int(sum(gate["result"] == "fail" for gate in gates)),
        "claim_boundary": (
            "If not adopted, 0.560 remains the only descriptive coverage-aware ceiling; "
            "no calibrated interval, target ranking, or animal-budget statement is published."
        ),
    }
    _write_json(out / "summary.json", summary)
    real[PRIMARY_MODEL].by_target.to_csv(out / "hierarchical_by_target.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small pilot run")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    summary = run(quick=args.quick, out=args.out)
    print(
        json.dumps({"decision": summary["decision"], "n_gates_failed": summary["n_gates_failed"]}, indent=2)
    )


if __name__ == "__main__":
    main()
