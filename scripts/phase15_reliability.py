"""Phase 15: reliability without extrapolation.

Computes the three pre-specified estimands, delete-one-animal jackknife
intervals, a Bayesian-bootstrap cross-check, null controls, extrapolation and
ranking diagnostics, and the Phase 15 decision.  It does not fit a measurement
model.

Run from the repository root::

    python scripts/phase15_reliability.py
    python scripts/phase15_reliability.py --skip-calibration --skip-nulls
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.benchmark.reliability_audit import (
    ESTIMANDS,
    audit_estimands,
    bayesian_bootstrap,
    calibration_study,
    null_distributions,
    ranking_stability,
)
from neurobranch.benchmark.reporting import ROOT
from neurobranch.benchmark.robustness import (
    _build_target_matrices,
    extrapolation_check,
)
from neurobranch.benchmark.variance import cell_summaries

DEFAULT_TRIALS = ROOT / "reports" / "phase3" / "randi_trial_table_phase3.csv.gz"
DEFAULT_OUT = ROOT / "reports" / "phase15"
FOLD_SIZE = 22
COVERAGE = 0.8
MIN_EVENTS = 1
MIN_ANIMALS = 6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _compact_jackknife(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"loo_animals", "loo_values"}
    }


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _event_variance(trials: pd.DataFrame) -> tuple[float, int]:
    cells = cell_summaries(trials)
    eligible = cells[(cells["n"] >= 2) & (cells["var"] > 0) & np.isfinite(cells["var"])]
    if eligible.empty:
        raise RuntimeError("no eligible cells for the one-event variance")
    return float(np.median(eligible["var"].to_numpy(dtype=float))), int(len(eligible))


def _build_variants(trials: pd.DataFrame, omega2: float) -> dict[str, dict[str, Any]]:
    configurations = {
        "v0_historical": {},
        "v1_primary": {"one_event_variance": omega2},
        "v1h_homogeneous": {"event_variance": omega2},
        "v2_min_events_2": {"min_events": 2},
    }
    variants: dict[str, dict[str, Any]] = {}
    for name, options in configurations.items():
        matrices, _ = _build_target_matrices(
            trials,
            response="response_mean",
            animal="dataset_id",
            min_events=int(options.get("min_events", MIN_EVENTS)),
            min_animals=MIN_ANIMALS,
            min_responder_coverage=COVERAGE,
            fold_size=FOLD_SIZE,
            one_event_variance=options.get("one_event_variance"),
            event_variance=options.get("event_variance"),
        )
        audit = audit_estimands(matrices, fold_size=FOLD_SIZE)
        variants[name] = {
            "summary": audit["summary"],
            "jackknife": {key: _compact_jackknife(value) for key, value in audit["jackknife"].items()},
            "jackknife_full": audit["jackknife"],
            "table": audit["table"],
            "matrices": matrices,
            "parameters": {
                "min_events": int(options.get("min_events", MIN_EVENTS)),
                "min_animals": MIN_ANIMALS,
                "min_responder_coverage": COVERAGE,
                "fold_size": FOLD_SIZE,
                "one_event_variance": options.get("one_event_variance"),
                "event_variance": options.get("event_variance"),
            },
        }
    return variants


def _gate(status: str, requirement: str, observed: Any) -> dict[str, Any]:
    return {"status": status, "requirement": requirement, "observed": observed}


def _compute_gates(
    primary: dict[str, Any],
    variants: dict[str, dict[str, Any]],
    nulls: dict[str, Any],
    calibration: dict[str, Any],
    extrapolation: dict[str, Any],
    ranking: dict[str, Any],
) -> dict[str, Any]:
    summary = primary["summary"]
    gates: dict[str, Any] = {}
    estimands_present = all(key in summary for key in ("R_obs", "R_22", "R_id"))
    gates["G1_estimand_completeness"] = _gate(
        "pass" if estimands_present and summary["R_obs"]["n_targets"] > 0 else "fail",
        "all three estimands with target counts and extrapolation flags",
        {
            "n_targets": summary["R_obs"]["n_targets"],
            "n_extrapolated_22": summary["R_22"]["n_extrapolated"],
            "estimands_present": estimands_present,
        },
    )
    jackknife = primary["jackknife"]
    jack_ok = all(
        jackknife[estimand].get("status") == "ok" and jackknife[estimand].get("valid_fraction", 0.0) >= 0.95
        for estimand in ESTIMANDS
    )
    gates["G2_jackknife_integrity"] = _gate(
        "pass" if jack_ok else "fail",
        ">=95% valid LOO replicates, status ok",
        {estimand: jackknife[estimand].get("valid_fraction") for estimand in ESTIMANDS},
    )
    calibration_statuses = {
        estimand: calibration.get("estimands", {}).get(estimand, {}).get("status", "not_run")
        for estimand in ESTIMANDS
    }
    gates["G3_calibration"] = _gate(
        "pass" if all(value == "calibrated" for value in calibration_statuses.values()) else "fail",
        "coverage 0.93--0.97 for each estimand",
        calibration_statuses,
    )
    null_ok = all(
        nulls.get(estimand, {}).get(mode, {}).get("median", float("inf")) <= 0.05
        and nulls.get(estimand, {}).get(mode, {}).get("q95", float("inf")) <= 0.15
        for estimand in ESTIMANDS
        for mode in ("permutation", "signflip")
    )
    gates["G4_null_control"] = _gate(
        "pass" if null_ok else "fail",
        "median <= 0.05 and q95 <= 0.15 for both controls",
        nulls,
    )
    extrapolation_ok = bool(extrapolation.get("gate_passed", 0.0))
    gates["G5_extrapolation"] = _gate(
        "pass" if extrapolation_ok else "fail",
        "half-split MAE <= 0.10 and abs bias <= 0.05",
        {
            "median_abs_error": extrapolation.get("median_abs_error"),
            "bias": extrapolation.get("bias"),
            "median_targets": extrapolation.get("median_targets"),
        },
    )
    ranking_ok = ranking.get("status") == "ok"
    gates["G6_ranking_stability"] = _gate(
        "pass" if ranking_ok else "fail",
        "Spearman lower CI > 0.50 and top-10 Jaccard > 0.60",
        ranking,
    )
    v1 = variants["v1_primary"]["summary"]
    v0 = variants["v0_historical"]["summary"]
    deltas = {
        estimand: abs(v1[estimand]["estimate"] - v0[estimand]["estimate"])
        for estimand in ("R_obs", "R_22", "R_id")
        if np.isfinite(v1[estimand]["estimate"]) and np.isfinite(v0[estimand]["estimate"])
    }
    gates["G7_variance_robustness"] = _gate(
        "pass" if all(delta <= 0.05 for delta in deltas.values()) else "fail",
        "abs(V1 - V0) <= 0.05 for each estimand",
        deltas,
    )
    gates["G8_reproducibility"] = _gate(
        "pending",
        "manifest, hashes, validator, clean worktree",
        "checked by phase15_validate.py",
    )
    return gates


def _compute_decision(gates: dict[str, Any], variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    for name, gate in gates.items():
        if gate["status"] == "fail":
            reasons.append(name)
    calibrated = gates["G3_calibration"]["status"] == "pass" and gates["G4_null_control"]["status"] == "pass"
    if not calibrated:
        label = "not_calibrated"
    elif gates["G5_extrapolation"]["status"] != "pass":
        label = "counterfactual_only"
    elif gates["G6_ranking_stability"]["status"] != "pass":
        label = "calibrated_descriptive_no_ranking"
    else:
        label = "calibrated_descriptive"
    return {
        "label": label,
        "calibrated_interval": bool(calibrated),
        "ceiling_claim": bool(calibrated and gates["G4_null_control"]["status"] == "pass"),
        "ranking_claim": bool(gates["G6_ranking_stability"]["status"] == "pass"),
        "extrapolation_status": "supported" if gates["G5_extrapolation"]["status"] == "pass" else "counterfactual_only",
        "r22_estimate": variants["v1_primary"]["summary"]["R_22"]["estimate"],
        "r22_extrapolated_targets": variants["v1_primary"]["summary"]["R_22"]["n_extrapolated"],
        "reasons": reasons,
        "note": "A calibrated descriptive interval is not a causal, transport, ranking or animal-budget claim.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, default=DEFAULT_TRIALS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-bootstrap", type=int, default=2_000)
    parser.add_argument("--n-null", type=int, default=1_000)
    parser.add_argument("--n-calibration", type=int, default=1_000)
    parser.add_argument("--n-splits", type=int, default=15)
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-nulls", action="store_true")
    args = parser.parse_args(argv)

    if not args.trials.exists():
        print(f"trials table not found: {args.trials}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    trials = pd.read_csv(args.trials)
    omega2, n_variance_cells = _event_variance(trials)

    variants = _build_variants(trials, omega2)
    primary = variants["v1_primary"]
    matrices = primary["matrices"]
    primary["table"].to_csv(args.out / "estimands.csv", index=False)

    jackknife_rows: list[dict[str, Any]] = []
    for estimand, result in primary["jackknife_full"].items():
        for animal, value in zip(result["loo_animals"], result["loo_values"], strict=False):
            jackknife_rows.append({"estimand": estimand, "animal": animal, "reliability": value})
    pd.DataFrame(jackknife_rows).to_csv(args.out / "jackknife.csv", index=False)

    bootstrap: dict[str, dict[str, Any]] = {}
    bootstrap_rows: list[dict[str, Any]] = []
    for estimand in ESTIMANDS:
        draws = bayesian_bootstrap(
            matrices,
            estimand=estimand,
            fold_size=FOLD_SIZE,
            n_draws=args.n_bootstrap,
            seed=args.seed,
        )
        finite = draws[np.isfinite(draws)]
        bootstrap[estimand] = {
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "ci_low": float(np.quantile(finite, 0.025)) if finite.size else float("nan"),
            "ci_high": float(np.quantile(finite, 0.975)) if finite.size else float("nan"),
            "n_draws": int(args.n_bootstrap),
            "n_valid": int(finite.size),
        }
        for draw_index, value in enumerate(draws):
            bootstrap_rows.append(
                {
                    "estimand": estimand,
                    "draw": draw_index,
                    "reliability": float(value) if np.isfinite(value) else None,
                }
            )
    pd.DataFrame(bootstrap_rows).to_csv(args.out / "bayesian_bootstrap.csv", index=False)

    nulls: dict[str, Any] = {}
    if not args.skip_nulls:
        for estimand in ESTIMANDS:
            nulls[estimand] = null_distributions(
                matrices,
                estimand=estimand,
                fold_size=FOLD_SIZE,
                n_draws=args.n_null,
                seed=args.seed + 1,
            )
    else:
        nulls = {estimand: {"status": "not_run"} for estimand in ESTIMANDS}
    (args.out / "nulls.json").write_text(
        json.dumps(nulls, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    extrapolation = extrapolation_check(
        trials,
        n_splits=args.n_splits,
        min_animals=MIN_ANIMALS,
        seed=args.seed,
        min_responder_coverage=COVERAGE,
        one_event_variance=omega2,
    )
    extrapolation["coverage"] = COVERAGE
    extrapolation["variance_treatment"] = "v1_one_event"
    (args.out / "extrapolation.json").write_text(
        json.dumps(extrapolation, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )

    ranking = ranking_stability(
        trials,
        n_splits=args.n_splits,
        min_animals=3,
        min_responder_coverage=COVERAGE,
        seed=args.seed,
        one_event_variance=omega2,
    )
    (args.out / "ranking.json").write_text(
        json.dumps(ranking, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )

    if args.skip_calibration:
        calibration: dict[str, Any] = {
            "status": "not_run",
            "n_reps": 0.0,
            "note": "run without --skip-calibration for the final artifact",
        }
    else:
        calibration = calibration_study(
            matrices,
            fold_size=FOLD_SIZE,
            n_reps=args.n_calibration,
            seed=args.seed,
        )
        calibration["status"] = (
            "calibrated"
            if all(entry["status"] == "calibrated" for entry in calibration["estimands"].values())
            else "not_calibrated"
        )
    (args.out / "calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )

    gates = _compute_gates(primary, variants, nulls, calibration, extrapolation, ranking)
    decision = _compute_decision(gates, variants)

    summary_estimands: dict[str, Any] = {}
    for key, estimand in (("R_obs", "obs"), ("R_22", "fold"), ("R_id", "id")):
        block = dict(primary["summary"][key])
        block["jackknife"] = _compact_jackknife(primary["jackknife"][estimand])
        block["bootstrap"] = bootstrap[estimand]
        block["null"] = nulls[estimand]
        summary_estimands[key] = block

    variant_summary: dict[str, Any] = {}
    for name, variant in variants.items():
        variant_summary[name] = {
            "estimands": variant["summary"],
            "jackknife": variant["jackknife"],
            "parameters": variant["parameters"],
        }

    summary = {
        "schema_version": 1,
        "phase": "phase15",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(args.trials.relative_to(ROOT)) if args.trials.is_relative_to(ROOT) else str(args.trials),
            "sha256": _sha256(args.trials),
            "n_rows": int(len(trials)),
        },
        "parameters": {
            "fold_size": FOLD_SIZE,
            "min_events": MIN_EVENTS,
            "min_animals": MIN_ANIMALS,
            "min_responder_coverage": COVERAGE,
            "n_bootstrap": int(args.n_bootstrap),
            "n_null": int(args.n_null),
            "n_calibration": int(args.n_calibration),
            "n_splits": int(args.n_splits),
            "seed": int(args.seed),
            "variance_treatment": "v1_one_event",
        },
        "event_variance": {"omega2": omega2, "n_cells": n_variance_cells},
        "estimands": summary_estimands,
        "variance_sensitivity": variant_summary,
        "extrapolation": extrapolation,
        "ranking": ranking,
        "calibration": calibration,
        "gates": gates,
        "decision": decision,
        "provenance": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "protocol": "research/phase15_protocol.md",
        },
    }
    _write_summary(args.out / "summary.json", summary)
    print(json.dumps({"decision": decision, "gates": {k: v["status"] for k, v in gates.items()}}, indent=2))
    print(f"runtime_seconds: {time.perf_counter() - started:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
