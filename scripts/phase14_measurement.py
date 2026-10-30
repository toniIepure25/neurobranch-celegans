"""Phase 14B: fit the zero-inflated reliability model and evaluate frozen gates.

The script is fail-closed.  A missing or malformed calibration artifact is a
failed gate; it is never silently omitted.  The full run uses the thresholds in
`research/phase14_protocol.md`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.benchmark.measurement import fit_measurement_reliability
from neurobranch.benchmark.measurement_calibration import (
    heldout_ppc_measurement,
    permutation_control,
    prepare_measurement_templates,
    prior_sensitivity,
    simulation_based_calibration,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRIALS = ROOT / "reports" / "phase3" / "randi_trial_table_phase3.csv.gz"
DEFAULT_OUT = ROOT / "reports" / "phase14"
MODEL = "m4_spike_slab"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _gate(
    gates: dict[str, dict[str, Any]],
    name: str,
    *,
    passed: bool,
    value: Any,
    requirement: str,
    note: str = "",
) -> None:
    gates[name] = {
        "passed": bool(passed),
        "value": value,
        "requirement": requirement,
        "note": note,
    }


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, default=DEFAULT_TRIALS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--draws", type=int, default=2500)
    parser.add_argument("--grid", type=int, default=80)
    parser.add_argument("--event-grid", type=int, default=200)
    parser.add_argument("--sbc-reps", type=int, default=1000)
    parser.add_argument("--sbc-targets", type=int, default=3)
    parser.add_argument("--ppc-splits", type=int, default=20)
    parser.add_argument("--ppc-targets", type=int, default=40)
    parser.add_argument("--null-targets", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args(argv)
    if not args.trials.is_file():
        print(f"trials table not found: {args.trials}", file=sys.stderr)
        return 2
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    trials = pd.read_csv(args.trials)
    templates = prepare_measurement_templates(trials, coverage=0.8)
    if not templates:
        print("no eligible templates", file=sys.stderr)
        return 2

    fit = fit_measurement_reliability(
        trials,
        model=MODEL,
        coverage=0.8,
        draws=args.draws,
        grid_resolution=args.grid,
        seed=args.seed,
        event_grid=args.event_grid,
        pi0=0.5,
        slab_scale=1.0,
    )
    eligible = fit.by_target[fit.by_target["status"] == "ok"]
    if eligible.empty:
        print("no eligible targets in real fit", file=sys.stderr)
        return 2
    real_r = float(fit.pooled_reliability_median)
    real_row = {
        "status": fit.status,
        "model": MODEL,
        "n_targets": int(len(fit.by_target)),
        "n_ok": int(len(eligible)),
        "pooled_reliability_median_n22": real_r,
        "single_animal_reliability_median_n1": float(fit.single_animal_reliability_median),
        "pooled_ceiling_median_n22": float(fit.pooled_ceiling_median),
        "p_zero_median": float(eligible["p_zero"].median()),
        "p_zero_mean": float(eligible["p_zero"].mean()),
        "spike_fraction": float((eligible["p_zero"] >= 0.5).mean()),
        "event_variance_median": float(eligible["event_variance_median"].median()),
        "max_grid_edge_mass": float(eligible["grid_edge_mass"].max()),
        "min_grid_n_eff": float(eligible["grid_n_eff"].min()),
    }
    fit.by_target.to_csv(out / "hierarchical_by_target.csv", index=False)
    _write_json(
        out / "real_fit.json",
        {
            "summary": real_row,
            "config": {
                "trials": str(args.trials),
                "out": str(args.out),
                "draws": args.draws,
                "grid": args.grid,
                "event_grid": args.event_grid,
                "seed": args.seed,
            },
        },
    )

    reference_path = out / "reference_audit.json"
    if not reference_path.is_file():
        print("reference audit missing; run scripts/phase14_reference_audit.py first", file=sys.stderr)
        return 2
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    v1 = float(reference["treatments"]["v1_primary"]["pooled_reliability_at_fold"])

    sbc = simulation_based_calibration(
        templates,
        model=MODEL,
        n_reps=args.sbc_reps,
        targets_per_rep=args.sbc_targets,
        draws=args.draws,
        grid_resolution=args.grid,
        event_grid=args.event_grid,
        seed=31,
        pi0=0.5,
        slab_scale=1.0,
    )
    _write_json(out / "sbc.json", sbc)

    ppc = heldout_ppc_measurement(
        trials,
        model=MODEL,
        n_splits=args.ppc_splits,
        max_targets=args.ppc_targets,
        draws=1200,
        grid_resolution=60,
        event_grid=120,
        seed=41,
    )
    _write_json(out / "heldout_ppc.json", ppc)

    negative = permutation_control(
        templates,
        model=MODEL,
        n_targets=args.null_targets,
        draws=1200,
        grid_resolution=50,
        event_grid=100,
        seed=43,
    )
    _write_json(out / "negative_control.json", negative)
    signflip = permutation_control(
        templates,
        model=MODEL,
        n_targets=args.null_targets,
        draws=1200,
        grid_resolution=50,
        event_grid=100,
        seed=44,
        sign_flip=True,
    )
    _write_json(out / "signflip_control.json", signflip)

    priors = prior_sensitivity(
        trials,
        model=MODEL,
        draws=1200,
        grid_resolution=50,
        seed=47,
        prior_scales=(0.5, 1.0, 2.0),
        pi0_values=(0.25, 0.5, 0.75),
    )
    _write_json(out / "prior_sensitivity.json", priors)

    min_events = fit_measurement_reliability(
        trials,
        model=MODEL,
        coverage=0.8,
        min_events=2,
        draws=1200,
        grid_resolution=50,
        seed=48,
        event_grid=100,
    )
    min_events.summary["method"] = "m4 min_events=2 sensitivity"
    _write_json(out / "min_events_sensitivity.json", min_events.summary)

    coverage_rows: list[dict[str, Any]] = []
    for coverage in (0.5, 0.8, 1.0):
        result = fit_measurement_reliability(
            trials,
            model=MODEL,
            coverage=coverage,
            draws=1200,
            grid_resolution=50,
            seed=49,
            event_grid=100,
        )
        coverage_rows.append(
            {
                "min_responder_coverage": coverage,
                "status": result.status,
                "n_targets": int(result.summary.get("n_targets", 0)),
                "n_ok": int(result.summary.get("n_ok", 0)),
                "pooled_reliability_median_n22": float(result.pooled_reliability_median),
                "pooled_ceiling_median_n22": float(result.pooled_ceiling_median),
            }
        )
    pd.DataFrame(coverage_rows).to_csv(out / "coverage_sensitivity.csv", index=False)

    transport = {
        "status": "not_evaluable",
        "reason": "No independent repeated-stimulation atlas is available locally. "
        "Kato WT_NoStim is spontaneous imaging; WT_Stim is absent; behavioural "
        "datasets are a different estimand.",
        "checked_paths": [
            "data/raw/kato2015/WT_NoStim.mat",
            "data/raw/kato2015/WT_Stim.mat",
            "data/raw/baskoylu2026",
            "data/raw/kramer2026",
            "data/raw/randi2023",
        ],
    }
    _write_json(out / "transport_status.json", transport)

    gates: dict[str, dict[str, Any]] = {}
    sbc_n22 = sbc.get("parameters", {}).get("n22", {})
    n_fits = int(sbc.get("n_fits", 0))
    n_failed = int(sbc.get("n_failed_fits", 0))
    coverage_22 = float(sbc_n22.get("coverage", float("nan")))
    ks_p = float(sbc_n22.get("ks_pvalue", float("nan")))
    _gate(
        gates,
        "G1a_sbc_coverage",
        passed=_finite(coverage_22) and 0.93 <= coverage_22 <= 0.97,
        value=coverage_22,
        requirement="[0.93, 0.97]",
    )
    _gate(
        gates,
        "G1b_sbc_rank",
        passed=_finite(ks_p) and ks_p > 0.01,
        value=ks_p,
        requirement="KS p > 0.01",
    )
    edge = float(sbc.get("max_edge_mass", float("nan")))
    _gate(
        gates,
        "G1c_grid_edge",
        passed=_finite(edge) and edge < 0.05,
        value=edge,
        requirement="< 0.05",
    )
    failure_fraction = (n_failed / n_fits) if n_fits else 1.0
    _gate(
        gates,
        "G1d_fit_integrity",
        passed=failure_fraction <= 0.05,
        value=failure_fraction,
        requirement="<= 0.05",
        note=f"{n_failed}/{n_fits} failed fits",
    )
    one_event = sbc.get("one_event_stratum", {})
    one_cov = float(one_event.get("coverage", float("nan")))
    one_n = int(one_event.get("n", 0))
    _gate(
        gates,
        "G1e_one_event_stratum",
        passed=one_n >= 30 and _finite(one_cov) and 0.90 <= one_cov <= 0.98,
        value=one_cov,
        requirement="[0.90, 0.98] with n >= 30",
        note=f"n={one_n}",
    )
    ppc_cov = float(ppc.get("coverage", float("nan")))
    ppc_mae = float(ppc.get("median_abs_error", float("nan")))
    _gate(
        gates,
        "G2_heldout_ppc",
        passed=ppc.get("status") == "ok"
        and _finite(ppc_cov)
        and ppc_cov >= 0.90
        and _finite(ppc_mae)
        and ppc_mae <= 0.10,
        value={"coverage": ppc_cov, "median_abs_error": ppc_mae, "n": ppc.get("n", 0)},
        requirement="coverage >= 0.90 and MAE <= 0.10; must be run",
    )
    null_median = float(negative.get("median_reliability_n22", float("nan")))
    null_q95 = float(negative.get("q95_reliability_n22", float("nan")))
    _gate(
        gates,
        "G3_negative_control",
        passed=_finite(null_median) and null_median <= 0.05 and _finite(null_q95) and null_q95 <= 0.15,
        value={"median": null_median, "q95": null_q95, "n": negative.get("n_eligible", 0)},
        requirement="median <= 0.05 and q95 <= 0.15",
    )
    flip_median = float(signflip.get("median_reliability_n22", float("nan")))
    flip_q95 = float(signflip.get("q95_reliability_n22", float("nan")))
    _gate(
        gates,
        "G3b_signflip_control",
        passed=_finite(flip_median)
        and flip_median <= 0.05
        and _finite(flip_q95)
        and flip_q95 <= 0.15,
        value={"median": flip_median, "q95": flip_q95, "n": signflip.get("n_eligible", 0)},
        requirement="median <= 0.05 and q95 <= 0.15",
    )
    prior_range = float(priors.get("range", float("nan")))
    _gate(
        gates,
        "G4_prior_sensitivity",
        passed=_finite(prior_range) and prior_range <= 0.05,
        value=prior_range,
        requirement="range <= 0.05",
    )
    omega_cov = float(sbc.get("event_variance_coverage", float("nan")))
    omega_n = int(sbc.get("event_variance_n", 0))
    _gate(
        gates,
        "G5_event_variance_recovery",
        passed=omega_n >= 100 and _finite(omega_cov) and 0.90 <= omega_cov <= 0.98,
        value=omega_cov,
        requirement="[0.90, 0.98] with n >= 100",
        note=f"n={omega_n}",
    )
    reference_delta = abs(real_r - v1)
    _gate(
        gates,
        "G6_reference_agreement",
        passed=reference_delta <= 0.05,
        value=reference_delta,
        requirement="|R(22) - R_ref(V1)| <= 0.05",
        note=f"model={real_r!r}, reference={v1!r}",
    )
    spike = sbc.get("spike_calibration", {})
    spike_zero = float(spike.get("median_p_zero_true_zero", float("nan")))
    spike_n = int(spike.get("n_true_zero", 0))
    _gate(
        gates,
        "G7_spike_calibration",
        passed=spike_n >= 100 and _finite(spike_zero) and spike_zero >= 0.80,
        value=spike_zero,
        requirement="median P(z=0 | true z=0) >= 0.80 with n >= 100",
        note=f"n={spike_n}",
    )

    failed = [name for name, gate in gates.items() if not gate["passed"]]
    adopted = not failed
    summary = {
        "schema_version": 1,
        "phase": 14,
        "model": MODEL,
        "input": {
            "path": str(args.trials.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha256(args.trials),
        },
        "config": {
            "coverage": 0.80,
            "fold_size": 22,
            "n_eval": [1, 22],
            "draws": args.draws,
            "grid_resolution": args.grid,
            "event_grid": args.event_grid,
            "pi0": 0.5,
            "slab_scale": 1.0,
            "seed": args.seed,
            "sbc_reps": args.sbc_reps,
            "sbc_targets_per_rep": args.sbc_targets,
            "ppc_splits": args.ppc_splits,
            "ppc_targets": args.ppc_targets,
            "null_targets": args.null_targets,
        },
        "real_fit": real_row,
        "reference": {
            "v1_pooled_reliability_at_fold": v1,
            "v1_pooled_ceiling_at_fold": float(
                reference["treatments"]["v1_primary"]["pooled_ceiling_at_fold"]
            ),
            "v0_pooled_reliability_at_fold": float(
                reference["treatments"]["v0_historical"]["pooled_reliability_at_fold"]
            ),
            "audit_decision": reference["decisions"],
        },
        "gates": gates,
        "decision": "adopted" if adopted else "not_adopted_gate_failures",
        "n_gates": len(gates),
        "n_gates_failed": len(failed),
        "failed_gates": failed,
        "transport_status": "not_evaluable",
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    _write_json(out / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("decision", "n_gates_failed", "failed_gates")}, indent=2))
    print(json.dumps(gates, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
