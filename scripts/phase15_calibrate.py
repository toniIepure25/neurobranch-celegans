"""Run a standalone Phase 15 calibration study.

The primary 1,000-replicate study is embedded in ``phase15_reliability.py`` so
the summary and its gate are produced in one process.  This script reruns the
same simulation, or the pre-specified heavy-tailed misspecification check, and
writes a separate artifact for audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from neurobranch.benchmark.reliability_audit import calibration_study
from neurobranch.benchmark.reporting import ROOT
from neurobranch.benchmark.robustness import _build_target_matrices
from neurobranch.benchmark.variance import cell_summaries

DEFAULT_TRIALS = ROOT / "reports" / "phase3" / "randi_trial_table_phase3.csv.gz"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, default=DEFAULT_TRIALS)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "phase15" / "calibration_heavy.json")
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--heavy", action="store_true")
    parser.add_argument("--seed", type=int, default=115)
    args = parser.parse_args(argv)
    if not args.trials.exists():
        print(f"trials table not found: {args.trials}", file=sys.stderr)
        return 2
    trials = pd.read_csv(args.trials)
    cells = cell_summaries(trials)
    eligible = cells[(cells["n"] >= 2) & (cells["var"] > 0) & np.isfinite(cells["var"])]
    omega2 = float(np.median(eligible["var"].to_numpy(dtype=float)))
    matrices, _ = _build_target_matrices(
        trials,
        response="response_mean",
        animal="dataset_id",
        min_events=1,
        min_animals=6,
        min_responder_coverage=0.8,
        fold_size=22,
        one_event_variance=omega2,
    )
    result = calibration_study(
        matrices,
        fold_size=22,
        n_reps=args.reps,
        seed=args.seed,
        heavy_tailed=args.heavy,
    )
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    result["omega2"] = omega2
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result["estimands"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
