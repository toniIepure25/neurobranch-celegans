"""Phase 9: measurement ceiling audit for the Randi optogenetic atlas.

Thin wrapper around :func:`neurobranch.benchmark.reporting.write_reliability_report`.
Produces ``reliability_by_target.csv``, ``variance_components.json``,
``reliability_summary.json`` and ``manifest.json`` under ``reports/phase9``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from neurobranch.benchmark.reporting import ROOT, write_reliability_report

DEFAULT_TRIALS = ROOT / "reports" / "phase3" / "randi_trial_table_phase3.csv.gz"
DEFAULT_OUT = ROOT / "reports" / "phase9"
REQUIRED = {"dataset_id", "target", "neuron", "event_id", "response_mean"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, default=DEFAULT_TRIALS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if not args.trials.exists():
        print(f"trials table not found: {args.trials}", file=sys.stderr)
        return 2
    trials = pd.read_csv(args.trials)
    missing = REQUIRED - set(trials.columns)
    if missing:
        print(f"trials table missing columns: {sorted(missing)}", file=sys.stderr)
        return 2

    summary = write_reliability_report(trials, args.out, input_path=args.trials)
    table = pd.read_csv(args.out / "reliability_by_target.csv")
    print(f"targets with repeat ceiling: {int(np.isfinite(table['repeat_reliability']).sum())}")
    print(f"targets with animal ceiling: {int(np.isfinite(table['single_animal_reliability']).sum())}")
    print(
        "repeat reliability: mean={:.3f} [{:.3f}, {:.3f}] median={:.3f}".format(
            summary["mean_repeat_reliability"],
            summary["repeat_ci_low"],
            summary["repeat_ci_high"],
            summary["median_repeat_reliability"],
        )
    )
    print(
        "single-animal reliability: mean={:.3f} [{:.3f}, {:.3f}] median={:.3f} p90={:.3f}".format(
            summary["mean_animal_reliability"],
            summary["animal_ci_low"],
            summary["animal_ci_high"],
            summary["median_animal_reliability"],
            summary["p90_animal_reliability"],
        )
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
