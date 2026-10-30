"""Phase 12R: aligned calibration rerun with frozen gates.

The script is fail-closed.  A hierarchical interval is only described as
calibrated when every gate in ``research/phase12r_protocol.md`` passes and the
point estimate is within 0.05 of the Phase 11 reference.  ``--quick`` is a
smoke test only and never produces an adoption decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from neurobranch.benchmark.hierarchical import fit_hierarchical_reliability
from neurobranch.benchmark.hierarchical_calibration import (
    heldout_predictive_check,
    negative_control,
    prepare_templates,
    prior_sensitivity,
    simulation_based_calibration,
)
from neurobranch.design.posterior_design import design_summary, design_table

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports" / "phase3" / "randi_trial_table_phase3.csv.gz"
PHASE11_SUMMARY = ROOT / "reports" / "phase11" / "summary.json"
DEFAULT_OUT = ROOT / "reports" / "phase12r"

PRIMARY_COVERAGE = 0.8
N_EVAL = (1, 22)
SESOI_RELIABILITY = 0.05
PHASE11_REFERENCE_FALLBACK = 0.3136077301426884


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _phase11_reference() -> float:
    if not PHASE11_SUMMARY.is_file():
        return PHASE11_REFERENCE_FALLBACK
    payload = json.loads(PHASE11_SUMMARY.read_text(encoding="utf-8"))
    return float(
        payload.get("headline", {}).get("coverage_0_8_pooled_reliability", PHASE11_REFERENCE_FALLBACK)
    )


def _coverage_row(result: Any, coverage: float) -> dict[str, Any]:
    return {
        "coverage": coverage,
        "n_targets": int(len(result.by_target)),
        "n_ok": int((result.by_target["status"] == "ok").sum()),
        "pooled_reliability_median": result.pooled_reliability_median,
        "pooled_ceiling_median": result.pooled_ceiling_median,
        "single_animal_reliability_median": result.single_animal_reliability_median,
        "max_grid_edge_mass": result.summary.get("max_grid_edge_mass", float("nan")),
        "n_variance_floored": result.summary.get("n_variance_floored"),
        "variance_floor_fraction": result.summary.get("variance_floor_fraction"),
        "status": result.status,
    }


def _runner_config(quick: bool) -> dict[str, Any]:
    if quick:
        return {
            "primary_draws": 400,
            "primary_grid": 40,
            "sbc_reps": 20,
            "sbc_targets": 1,
            "sbc_draws": 400,
            "sbc_grid": 30,
            "ppc_splits": 3,
            "ppc_targets": 5,
            "ppc_draws": 400,
            "ppc_grid": 30,
            "negative_targets": 10,
            "negative_draws": 400,
            "negative_grid": 30,
            "prior_draws": 400,
            "prior_grid": 30,
        }
    return {
        "primary_draws": 2500,
        "primary_grid": 80,
        "sbc_reps": 1000,
        "sbc_targets": 3,
        "sbc_draws": 2000,
        "sbc_grid": 80,
        "ppc_splits": 20,
        "ppc_targets": 40,
        "ppc_draws": 1200,
        "ppc_grid": 60,
        "negative_targets": 100,
        "negative_draws": 1200,
        "negative_grid": 50,
        "prior_draws": 1200,
        "prior_grid": 50,
    }


def run(*, quick: bool = False, out: Path = DEFAULT_OUT) -> dict[str, Any]:
    started = time.time()
    if not INPUT.is_file():
        raise FileNotFoundError(f"missing input table: {INPUT}")
    out.mkdir(parents=True, exist_ok=True)
    config = _runner_config(quick)
    frame = pd.read_csv(INPUT)

    primary = fit_hierarchical_reliability(
        frame,
        coverage=PRIMARY_COVERAGE,
        n_eval=N_EVAL,
        draws=int(config["primary_draws"]),
        grid_resolution=int(config["primary_grid"]),
        seed=12,
    )
    primary.by_target.to_csv(out / "hierarchical_by_target.csv", index=False)

    coverage_results = {
        0.5: fit_hierarchical_reliability(
            frame,
            coverage=0.5,
            n_eval=N_EVAL,
            draws=max(300, int(config["primary_draws"]) // 2),
            grid_resolution=max(30, int(config["primary_grid"]) - 20),
            seed=12,
        ),
        0.8: primary,
        1.0: fit_hierarchical_reliability(
            frame,
            coverage=1.0,
            n_eval=N_EVAL,
            draws=max(300, int(config["primary_draws"]) // 2),
            grid_resolution=max(30, int(config["primary_grid"]) - 20),
            seed=12,
        ),
    }
    pd.DataFrame([_coverage_row(result, coverage) for coverage, result in coverage_results.items()]).to_csv(
        out / "coverage_sensitivity.csv", index=False
    )

    min_events_2 = fit_hierarchical_reliability(
        frame,
        coverage=PRIMARY_COVERAGE,
        n_eval=N_EVAL,
        draws=int(config["primary_draws"]),
        grid_resolution=int(config["primary_grid"]),
        seed=12,
        min_events=2,
    )
    pd.DataFrame(
        [
            _coverage_row(primary, PRIMARY_COVERAGE),
            _coverage_row(min_events_2, PRIMARY_COVERAGE),
        ]
    ).assign(min_events=[1, 2]).to_csv(out / "min_events_sensitivity.csv", index=False)

    templates = prepare_templates(
        frame,
        coverage=PRIMARY_COVERAGE,
        min_events=1,
        min_animals=6,
    )
    sbc = simulation_based_calibration(
        templates,
        n_reps=int(config["sbc_reps"]),
        targets_per_rep=int(config["sbc_targets"]),
        n_eval=N_EVAL,
        draws=int(config["sbc_draws"]),
        grid_resolution=int(config["sbc_grid"]),
        seed=31,
    )
    _write_json(out / "sbc.json", sbc)

    ppc = heldout_predictive_check(
        frame,
        coverage=PRIMARY_COVERAGE,
        n_splits=int(config["ppc_splits"]),
        max_targets=int(config["ppc_targets"]),
        draws=int(config["ppc_draws"]),
        grid_resolution=int(config["ppc_grid"]),
        seed=41,
    )
    _write_json(out / "heldout_ppc.json", ppc)

    negative = negative_control(
        frame,
        n_targets=int(config["negative_targets"]),
        coverage=PRIMARY_COVERAGE,
        draws=int(config["negative_draws"]),
        grid_resolution=int(config["negative_grid"]),
        seed=43,
    )
    _write_json(out / "negative_control.json", negative)

    sensitivity = prior_sensitivity(
        frame,
        coverage=PRIMARY_COVERAGE,
        draws=int(config["prior_draws"]),
        grid_resolution=int(config["prior_grid"]),
        seed=47,
    )
    _write_json(out / "prior_sensitivity.json", sensitivity)

    design = design_table(primary, target_reliability=0.5, probability=0.8, max_animals=60)
    design.to_csv(out / "design_posterior.csv", index=False)
    design_info = design_summary(design)

    sbc_n22 = sbc.get("parameters", {}).get("n22", {})
    sbc_coverage = float(sbc_n22.get("coverage", float("nan")))
    rank_p = float(sbc_n22.get("ks_pvalue", float("nan")))
    primary_edge = float(cast(float, primary.summary.get("max_grid_edge_mass", float("nan"))))
    sbc_edge = float(sbc.get("max_edge_mass") or float("nan"))
    edge = (
        max(value for value in (primary_edge, sbc_edge) if np.isfinite(value))
        if (np.isfinite(primary_edge) or np.isfinite(sbc_edge))
        else float("nan")
    )
    attempted = int(sbc.get("n_fits", 0))
    bad_fits = int(sbc.get("n_failed_fits", 0)) + int(sbc.get("n_edge_failures", 0))
    fit_integrity = (bad_fits / attempted) if attempted > 0 else float("inf")

    gate_sbc_coverage = bool(np.isfinite(sbc_coverage) and 0.93 <= sbc_coverage <= 0.97)
    gate_sbc_rank = bool(np.isfinite(rank_p) and rank_p > 0.01)
    gate_edge = bool(np.isfinite(edge) and edge < 0.05)
    gate_integrity = bool(attempted > 0 and fit_integrity <= 0.05)
    gate_sbc = bool(gate_sbc_coverage and gate_sbc_rank and gate_edge and gate_integrity)
    gate_ppc = bool(
        ppc.get("status") == "ok"
        and float(ppc.get("coverage", 0.0)) >= 0.90
        and float(ppc.get("median_abs_error", 99.0)) <= 0.10
    )
    gate_negative = bool(
        negative.get("status") == "ok"
        and float(negative.get("median_reliability_n22", 99.0)) <= 0.05
        and float(negative.get("q95_reliability_n22", 99.0)) <= 0.15
    )
    prior_values = [
        float(item["pooled_reliability_median"])
        for item in sensitivity["fits"].values()
        if item.get("pooled_reliability_median") is not None
    ]
    prior_range = (max(prior_values) - min(prior_values)) if prior_values else float("nan")
    gate_prior = bool(np.isfinite(prior_range) and prior_range <= 0.05)

    reference = _phase11_reference()
    difference = float(primary.pooled_reliability_median - reference)
    estimate_close = bool(abs(difference) <= SESOI_RELIABILITY)
    all_gates = bool(gate_sbc and gate_ppc and gate_negative and gate_prior)
    adopted = bool(all_gates and estimate_close and not quick)
    if quick:
        status = "quick_mode_no_decision"
    elif adopted:
        status = "adopted"
    elif all_gates and not estimate_close:
        status = "not_adopted_estimate_mismatch"
    else:
        status = "not_adopted_gate_failures"

    reasons = (
        primary.by_target.loc[primary.by_target["status"] != "ok", "reason"]
        .fillna("unspecified")
        .value_counts()
        .to_dict()
    )
    summary = {
        "phase": "12R",
        "mode": "quick" if quick else "full",
        "status": status,
        "primary_coverage": PRIMARY_COVERAGE,
        "n_eval": list(N_EVAL),
        "config": config,
        "model": "hierarchical additive model with animal centering and known cell sampling variance",
        "prior": "sigma, tau ~ HalfNormal(1) on the standardized scale",
        "input": {
            "path": str(INPUT.relative_to(ROOT)),
            "sha256": _sha256(INPUT),
            "n_rows": int(len(frame)),
        },
        "hierarchical": {
            "pooled_reliability_median": primary.pooled_reliability_median,
            "pooled_ceiling_median": primary.pooled_ceiling_median,
            "single_animal_reliability_median": primary.single_animal_reliability_median,
            "n_ok": int(cast(int, primary.summary.get("n_ok", 0))),
            "n_ineligible": int(cast(int, primary.summary.get("n_ineligible", 0))),
            "n_variance_floored": primary.summary.get("n_variance_floored"),
            "variance_floor_fraction": primary.summary.get("variance_floor_fraction"),
            "exclusion_reasons": reasons,
        },
        "phase11_reference": {
            "pooled_reliability": reference,
            "pooled_ceiling": 0.5600069018705827,
        },
        "difference_vs_phase11": difference,
        "estimate_within_0.05": estimate_close,
        "design": design_info,
        "gates": {
            "g1a_sbc_coverage": gate_sbc_coverage,
            "g1b_sbc_rank": gate_sbc_rank,
            "g1c_grid_edge": gate_edge,
            "g1d_fit_integrity": gate_integrity,
            "g1_sbc_all": gate_sbc,
            "g2_heldout_ppc": gate_ppc,
            "g3_negative_control": gate_negative,
            "g4_prior_sensitivity": gate_prior,
            "all_gates": all_gates,
            "sbc_coverage_n22": sbc_coverage,
            "sbc_coverage_ci_low": sbc_n22.get("coverage_ci_low"),
            "sbc_coverage_ci_high": sbc_n22.get("coverage_ci_high"),
            "sbc_rank_ks_pvalue": rank_p,
            "sbc_n_fits": attempted,
            "sbc_n_failed_fits": sbc.get("n_failed_fits"),
            "sbc_n_edge_failures": sbc.get("n_edge_failures"),
            "sbc_fit_integrity_failure_fraction": fit_integrity,
            "max_grid_edge_mass": edge,
            "ppc_coverage": ppc.get("coverage"),
            "ppc_median_abs_error": ppc.get("median_abs_error"),
            "ppc_n": ppc.get("n"),
            "negative_median_n22": negative.get("median_reliability_n22"),
            "negative_q95_n22": negative.get("q95_reliability_n22"),
            "negative_n_requested": negative.get("n_requested"),
            "negative_n_eligible": negative.get("n_targets"),
            "negative_n_excluded": negative.get("n_excluded"),
            "negative_exclusion_reasons": negative.get("exclusion_reasons"),
            "prior_range": prior_range,
        },
        "decision": (
            "Quick mode only; no adoption decision."
            if quick
            else (
                "Hierarchical posterior adopted under the frozen Phase 12R protocol; "
                "Phase 11 0.560 remains the descriptive reference."
                if adopted
                else "Gates or estimate agreement failed; retain Phase 11 0.560 and "
                "do not call a hierarchical interval calibrated."
            )
        ),
        "runtime_seconds": float(time.time() - started),
        "software": {
            "python": ".".join(map(str, __import__("sys").version_info[:3])),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "commands": [
            "python scripts/phase12r_hierarchy.py",
            "python scripts/phase12r_manifest.py",
            "python scripts/audit_claims.py",
        ],
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps(_clean(summary), indent=2, allow_nan=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smoke test only")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="artifact directory")
    args = parser.parse_args()
    run(quick=args.quick, out=args.out)


if __name__ == "__main__":
    main()
