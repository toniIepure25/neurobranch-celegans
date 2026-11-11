"""Analyze audited target-versus-sham trial outcomes at the animal level.

The input must contain one row per trial with ``animal_id``, ``arm`` (``target`` or
``sham``), and a binary ``outcome``.  This script is the conservative fallback
described in ``docs/INTERVENTION_POWER_PLAN.md``; it does not treat trials as
independent observations.

Example:
    python scripts/analyze_intervention_trials.py \
        --trials data/incoming/intervention/trials_with_outcomes.csv \
        --out reports/vienna_workbench/intervention_analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from neurobranch.interventions.trial_analysis import (  # noqa: E402
    analyze_trial_outcomes,
    read_trial_outcomes,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args(argv)

    try:
        rows = read_trial_outcomes(args.trials)
        report = analyze_trial_outcomes(
            rows,
            n_bootstrap=args.n_bootstrap,
            n_permutations=args.n_permutations,
            seed=args.seed,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
