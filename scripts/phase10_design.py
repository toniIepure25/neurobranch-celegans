"""Phase 10 design calculator: animal budget and coverage diagnostic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from neurobranch.benchmark.reporting import ROOT, write_json
from neurobranch.design.calibration import coverage_study
from neurobranch.design.ceiling import design_table

DEFAULT_COMPONENTS = ROOT / "reports" / "phase10" / "profile_components.csv"
DEFAULT_OUT = ROOT / "reports" / "phase10"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=Path, default=DEFAULT_COMPONENTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reps", type=int, default=30)
    args = parser.parse_args(argv)

    if not args.components.exists():
        print(f"components not found: {args.components}", file=sys.stderr)
        return 2
    components = pd.read_csv(args.components)
    table = design_table(components)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "design_budget.csv", index=False)
    print(table.round(2).to_string(index=False))

    coverage = coverage_study(n_reps=args.reps)
    write_json(args.out / "coverage_study.json", coverage)
    print(
        "coverage diagnostic: nominal=0.95 observed={:.3f} width={:.3f} bias={:.3f}".format(
            coverage["coverage"],
            coverage["mean_width"],
            coverage["bias"],
        )
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
