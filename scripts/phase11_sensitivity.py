"""Phase 11: coverage sensitivity, extrapolation and two-way uncertainty.

The Phase 10 estimator used every responder, including neurons observed in only
one animal.  Sparse coverage can manufacture apparent shared signal because a
responder measured once contributes to the pooled profile but has no sampling
variance estimate from the other animals.  This script quantifies that
sensitivity, runs the pre-declared extrapolation gate and computes a two-way
bootstrap at the coverage-aware setting.

Run from the repository root::

    python scripts/phase11_sensitivity.py
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

from neurobranch.benchmark.reporting import ROOT
from neurobranch.benchmark.robustness import animal_half_replication, extrapolation_check
from neurobranch.benchmark.variance import (
    animal_reliability_by_target,
    profile_components_by_target,
    summarise_components,
)
from neurobranch.design.calibration import two_way_coverage_study
from neurobranch.uncertainty import two_way_bootstrap

DEFAULT_TRIALS = ROOT / "reports" / "phase3" / "randi_trial_table_phase3.csv.gz"
DEFAULT_OUT = ROOT / "reports" / "phase11"
COVERAGES = (0.0, 0.3, 0.5, 0.8, 1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, default=DEFAULT_TRIALS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fold-size", type=int, default=22)
    parser.add_argument("--n-boot", type=int, default=300)
    parser.add_argument("--n-splits", type=int, default=15)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)

    if not args.trials.exists():
        print(f"trials table not found: {args.trials}", file=sys.stderr)
        return 2
    if args.fold_size < 1:
        print("--fold-size must be >= 1", file=sys.stderr)
        return 2
    trials = pd.read_csv(args.trials)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    sensitivity: list[dict[str, Any]] = []
    for coverage in COVERAGES:
        components = profile_components_by_target(
            trials,
            min_events=1,
            min_animals=6,
            min_responder_coverage=coverage,
        )
        summary = summarise_components(components, fold_size=args.fold_size)
        single = animal_reliability_by_target(
            trials,
            min_events=1,
            min_animals=6,
            min_responder_coverage=coverage,
        )
        sensitivity.append(
            {
                "min_responder_coverage": coverage,
                "n_targets": int(len(components)),
                "mean_n_responders": float(components["n_responders"].mean()) if len(components) else 0.0,
                "pooled_reliability_at_fold": summary["pooled_reliability_at_fold"],
                "pooled_ceiling_at_fold": summary["pooled_ceiling_at_fold"],
                "mean_single_animal_reliability": float(single["single_animal_reliability"].mean())
                if len(single)
                else float("nan"),
            }
        )
        print(
            "coverage {:.1f}: n={} pooled={:.3f} ceiling={:.3f}".format(
                coverage,
                len(components),
                summary["pooled_reliability_at_fold"],
                summary["pooled_ceiling_at_fold"],
            )
        )
    sensitivity_frame = pd.DataFrame(sensitivity)
    sensitivity_frame.to_csv(args.out / "coverage_sensitivity.csv", index=False)

    raw_half = animal_half_replication(
        trials,
        n_splits=args.n_splits,
        min_animals=6,
        seed=args.seed,
        min_responder_coverage=0.8,
    )
    shrunk_half = animal_half_replication(
        trials,
        n_splits=args.n_splits,
        min_animals=6,
        seed=args.seed,
        shrink=True,
        min_responder_coverage=0.8,
    )
    half_replication = {
        "coverage": 0.8,
        "raw": raw_half,
        "shrunk": shrunk_half,
        "shrinkage_gate_passed": bool(
            np.isfinite(shrunk_half["median_spearman"])
            and shrunk_half["median_spearman"] >= 0.30
            and shrunk_half["ci_low"] > 0.10
        ),
    }
    (args.out / "half_replication.json").write_text(
        json.dumps(half_replication, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    extrapolation = extrapolation_check(
        trials,
        n_splits=args.n_splits,
        min_animals=6,
        seed=args.seed,
        min_responder_coverage=0.8,
    )
    extrapolation["coverage"] = 0.8
    (args.out / "extrapolation.json").write_text(
        json.dumps(extrapolation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    bootstrap = two_way_bootstrap(
        trials,
        fold_size=args.fold_size,
        n_boot=args.n_boot,
        seed=args.seed,
        min_animals=6,
        min_responder_coverage=0.8,
    )
    bootstrap.by_target.to_csv(args.out / "uncertainty_by_target.csv", index=False)
    uncertainty_summary = dict(bootstrap.summary)
    uncertainty_summary["fold_size"] = args.fold_size
    (args.out / "uncertainty_summary.json").write_text(
        json.dumps(uncertainty_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    coverage_study = two_way_coverage_study(n_reps=100, n_boot=100)
    (args.out / "coverage_study.json").write_text(
        json.dumps(coverage_study, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary = {
        "phase": "phase11",
        "purpose": "coverage-aware sensitivity of the reliability ceiling",
        "input": {
            "path": str(args.trials.relative_to(ROOT)) if args.trials.is_relative_to(ROOT) else str(args.trials),
            "sha256": _sha256(args.trials),
            "n_rows": int(len(trials)),
        },
        "parameters": {
            "fold_size": args.fold_size,
            "coverages": list(COVERAGES),
            "n_boot": args.n_boot,
            "n_splits": args.n_splits,
            "seed": args.seed,
        },
        "headline": {
            "phase10_pooled_reliability": float(
                sensitivity_frame.loc[
                    sensitivity_frame["min_responder_coverage"] == 0.0, "pooled_reliability_at_fold"
                ].iloc[0]
            ),
            "coverage_0_8_pooled_reliability": float(
                sensitivity_frame.loc[
                    sensitivity_frame["min_responder_coverage"] == 0.8, "pooled_reliability_at_fold"
                ].iloc[0]
            ),
            "extrapolation_gate_passed": bool(extrapolation["gate_passed"]),
            "shrinkage_gate_passed": bool(half_replication["shrinkage_gate_passed"]),
        },
        "coverage_study": coverage_study,
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["headline"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
