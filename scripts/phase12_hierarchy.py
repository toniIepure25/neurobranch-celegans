"""Phase 12: hierarchical reliability, calibration gates and real-data checks.

The script is intentionally fail-closed.  A hierarchical estimate is only
marked adopted when every pre-declared gate passes; otherwise the Phase 11
coverage-aware bound remains the primary descriptive result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
OUT = ROOT / "reports" / "phase12"

PRIMARY_COVERAGE = 0.8
N_EVAL = (1, 22)
SESOI_RELIABILITY = 0.05


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


def _coverage_row(result: Any, coverage: float) -> dict[str, Any]:
    return {
        "coverage": coverage,
        "n_targets": int(len(result.by_target)),
        "n_ok": int((result.by_target["status"] == "ok").sum()),
        "pooled_reliability_median": result.pooled_reliability_median,
        "pooled_ceiling_median": result.pooled_ceiling_median,
        "single_animal_reliability_median": result.single_animal_reliability_median,
        "max_grid_edge_mass": result.summary.get("max_grid_edge_mass", float("nan")),
        "status": result.status,
    }


def _moment_row() -> dict[str, Any]:
    path = ROOT / "reports" / "phase11" / "coverage_sensitivity.csv"
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    column = "min_responder_coverage" if "min_responder_coverage" in frame.columns else "coverage"
    row = frame[np.isclose(frame[column].astype(float), PRIMARY_COVERAGE)]
    if row.empty:
        return {}
    record = row.iloc[0]
    return {
        "mean_pooled_reliability": float(record["pooled_reliability_at_fold"]),
        "mean_pooled_ceiling": float(record["pooled_ceiling_at_fold"]),
        "mean_single_animal_reliability": float(record["mean_single_animal_reliability"]),
    }


def main() -> None:
    if not INPUT.is_file():
        raise FileNotFoundError(f"missing input table: {INPUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT)

    fits: dict[float, Any] = {}
    for coverage in (0.5, 0.8, 1.0):
        fits[coverage] = fit_hierarchical_reliability(
            frame,
            coverage=coverage,
            n_eval=N_EVAL,
            draws=2500 if coverage == PRIMARY_COVERAGE else 1500,
            grid_resolution=80 if coverage == PRIMARY_COVERAGE else 60,
            seed=12,
        )
    primary = fits[PRIMARY_COVERAGE]
    primary.by_target.to_csv(OUT / "hierarchical_by_target.csv", index=False)
    pd.DataFrame([_coverage_row(result, coverage) for coverage, result in fits.items()]).to_csv(
        OUT / "coverage_sensitivity.csv", index=False
    )

    templates = prepare_templates(frame, coverage=PRIMARY_COVERAGE)
    sbc = simulation_based_calibration(
        templates,
        n_reps=200,
        targets_per_rep=3,
        draws=1200,
        grid_resolution=50,
        seed=31,
    )
    _write_json(OUT / "sbc.json", sbc)

    ppc = heldout_predictive_check(
        frame,
        coverage=PRIMARY_COVERAGE,
        n_splits=5,
        max_targets=40,
        draws=600,
        grid_resolution=40,
        seed=41,
    )
    _write_json(OUT / "heldout_ppc.json", ppc)

    negative = negative_control(
        frame,
        n_targets=40,
        coverage=PRIMARY_COVERAGE,
        draws=1200,
        grid_resolution=50,
        seed=43,
    )
    _write_json(OUT / "negative_control.json", negative)

    sensitivity = prior_sensitivity(
        frame,
        coverage=PRIMARY_COVERAGE,
        draws=1200,
        grid_resolution=50,
        seed=47,
    )
    _write_json(OUT / "prior_sensitivity.json", sensitivity)

    design = design_table(primary, target_reliability=0.5, probability=0.8, max_animals=60)
    design.to_csv(OUT / "design_posterior.csv", index=False)
    design_info = design_summary(design)

    moment = _moment_row()
    moment_reliability = float(moment.get("mean_pooled_reliability", float("nan")))
    moment_ceiling = float(moment.get("mean_pooled_ceiling", float("nan")))
    sbc_n22 = sbc["parameters"].get("n22", {})
    rank_p = float(sbc_n22.get("ks_pvalue", float("nan")))
    sbc_coverage = float(sbc_n22.get("coverage", float("nan")))
    edge = float(primary.summary.get("max_grid_edge_mass", float("nan")))
    gate_sbc = (
        np.isfinite(sbc_coverage)
        and 0.93 <= sbc_coverage <= 0.97
        and np.isfinite(rank_p)
        and rank_p > 0.01
        and np.isfinite(edge)
        and edge < 0.05
    )
    gate_ppc = (
        ppc.get("status") == "ok"
        and float(ppc.get("coverage", 0.0)) >= 0.90
        and float(ppc.get("median_abs_error", 99.0)) <= 0.10
    )
    gate_negative = (
        negative.get("status") == "ok"
        and float(negative.get("median_reliability_n22", 99.0)) <= 0.05
        and float(negative.get("q95_reliability_n22", 99.0)) <= 0.15
    )
    prior_values = [
        float(item["pooled_reliability_median"])
        for item in sensitivity["fits"].values()
        if item.get("pooled_reliability_median") is not None
    ]
    gate_prior = bool(prior_values) and max(prior_values) - min(prior_values) <= 0.05
    adopted = bool(gate_sbc and gate_ppc and gate_negative and gate_prior)
    difference = float(primary.pooled_reliability_median - moment_reliability)

    summary = {
        "phase": 12,
        "status": "adopted" if adopted else "not_adopted_gate_failures",
        "primary_coverage": PRIMARY_COVERAGE,
        "n_eval": list(N_EVAL),
        "model": "hierarchical additive model with animal centering and known cell sampling variance",
        "prior": "sigma, tau ~ HalfNormal(1) on the standardized scale",
        "hierarchical": {
            "pooled_reliability_median": primary.pooled_reliability_median,
            "pooled_ceiling_median": primary.pooled_ceiling_median,
            "single_animal_reliability_median": primary.single_animal_reliability_median,
            "n_ok": int(primary.summary.get("n_ok", 0)),
            "n_ineligible": int(primary.summary.get("n_ineligible", 0)),
        },
        "moment_phase11": {
            "pooled_reliability": moment_reliability,
            "pooled_ceiling": moment_ceiling,
        },
        "design": design_info,
        "difference_vs_moment": difference,
        "reproduction_within_0.05": bool(abs(difference) <= SESOI_RELIABILITY),
        "gates": {
            "g1_sbc": bool(gate_sbc),
            "g2_heldout_ppc": bool(gate_ppc),
            "g3_negative_control": bool(gate_negative),
            "g4_prior_sensitivity": bool(gate_prior),
            "sbc_coverage_n22": sbc_coverage,
            "sbc_rank_ks_pvalue": rank_p,
            "max_grid_edge_mass": edge,
            "ppc_coverage": ppc.get("coverage"),
            "ppc_median_abs_error": ppc.get("median_abs_error"),
            "negative_median_n22": negative.get("median_reliability_n22"),
            "negative_q95_n22": negative.get("q95_reliability_n22"),
            "prior_range": (max(prior_values) - min(prior_values)) if prior_values else None,
        },
        "decision": (
            "Hierarchical posterior adopted as primary inferential model."
            if adopted
            else "Gates failed; retain Phase 11 coverage-aware 0.560 and do not call a hierarchical interval calibrated."
        ),
        "commands": [
            "python scripts/phase12_hierarchy.py",
            "python scripts/phase12_benchmark.py",
            "python scripts/phase12_manifest.py",
        ],
    }
    _write_json(OUT / "summary.json", summary)
    print(json.dumps(_clean(summary), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
