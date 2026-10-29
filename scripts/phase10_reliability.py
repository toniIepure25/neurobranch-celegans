"""Phase 10: metric-aligned reliability ceilings for the Randi atlas.

Thin wrapper around :func:`neurobranch.benchmark.reporting.write_phase10_report`.
Produces the per-target components, the reliability curve and a schema-v2
summary under ``reports/phase10``.  The pooled ceiling is computed at the
Phase 6-8 fold size; the single-animal ceiling is kept as a separate quantity.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from neurobranch.benchmark.reporting import ROOT, write_phase10_report

DEFAULT_TRIALS = ROOT / "reports" / "phase3" / "randi_trial_table_phase3.csv.gz"
DEFAULT_OUT = ROOT / "reports" / "phase10"
REQUIRED = {"dataset_id", "target", "neuron", "event_id", "response_mean"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, default=DEFAULT_TRIALS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fold-size", type=int, default=22)
    args = parser.parse_args(argv)

    if not args.trials.exists():
        print(f"trials table not found: {args.trials}", file=sys.stderr)
        return 2
    if args.fold_size < 1:
        print("--fold-size must be >= 1", file=sys.stderr)
        return 2
    trials = pd.read_csv(args.trials)
    missing = REQUIRED - set(trials.columns)
    if missing:
        print(f"trials table missing columns: {sorted(missing)}", file=sys.stderr)
        return 2

    summary = write_phase10_report(
        trials,
        args.out,
        input_path=args.trials,
        fold_size=args.fold_size,
    )
    curve = pd.read_csv(args.out / "ceiling_curve.csv")
    print(f"targets with pooled components: {int(summary['n_pooled_targets'])}")
    print(
        "single-animal: mean={:.3f} [{:.3f}, {:.3f}] ceiling={:.3f}".format(
            summary["mean_animal_reliability"],
            summary["animal_ci_low"],
            summary["animal_ci_high"],
            np.sqrt(summary["mean_animal_reliability"]),
        )
    )
    print(
        "pooled at fold size {}: mean={:.3f} ceiling={:.3f} mean-target-ceiling={:.3f}".format(
            args.fold_size,
            summary["pooled_reliability_at_fold"],
            summary["pooled_ceiling_at_fold"],
            summary["mean_target_ceiling"],
        )
    )
    print(curve.to_string(index=False))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
