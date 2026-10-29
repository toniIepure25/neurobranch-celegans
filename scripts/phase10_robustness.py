"""Phase 10 robustness artefacts: summary sensitivity, half-split, jackknife."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from neurobranch.benchmark.reporting import ROOT, clean_json, write_json
from neurobranch.benchmark.robustness import (
    animal_half_replication,
    animal_jackknife,
    summary_sensitivity,
)

DEFAULT_TRIALS = ROOT / "reports" / "phase3" / "randi_trial_table_phase3.csv.gz"
DEFAULT_OUT = ROOT / "reports" / "phase10"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, default=DEFAULT_TRIALS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fold-size", type=int, default=22)
    parser.add_argument("--splits", type=int, default=25)
    args = parser.parse_args(argv)

    if not args.trials.exists():
        print(f"trials table not found: {args.trials}", file=sys.stderr)
        return 2
    trials = pd.read_csv(args.trials)
    args.out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    sensitivity = summary_sensitivity(trials, fold_size=args.fold_size)
    sensitivity.to_csv(args.out / "summary_sensitivity.csv", index=False)
    print(sensitivity.to_string(index=False))

    half = animal_half_replication(trials, n_splits=args.splits)
    write_json(args.out / "half_replication.json", half)
    print("half replication:", json.dumps(clean_json(half), sort_keys=True))

    jackknife = animal_jackknife(trials, fold_size=args.fold_size)
    write_json(args.out / "cluster_ci.json", jackknife)
    print(
        "jackknife: estimate={:.4f} se={:.4f} CI=[{:.4f}, {:.4f}] animals={}".format(
            jackknife["estimate"],
            jackknife["se"],
            jackknife["ci_low"],
            jackknife["ci_high"],
            int(jackknife["n_animals"]),
        )
    )
    print(f"elapsed {time.perf_counter() - started:.1f}s; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
