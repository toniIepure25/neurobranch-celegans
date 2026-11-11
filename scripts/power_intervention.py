"""Simulate power for a cluster-randomized behavioural intervention.

This writes a design report.  It does not use or imply any intervention data.

Example:
    python scripts/power_intervention.py --reps 500 \
        --out reports/vienna_workbench/intervention_power.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from neurobranch.interventions.power import build_power_report  # noqa: E402


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animals", type=_ints, default=(6, 12))
    parser.add_argument("--iccs", type=_floats, default=(0.05, 0.1, 0.2, 0.4))
    parser.add_argument("--design-icc", type=float, default=0.2)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--p-sham", type=float, default=0.30)
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--target-power", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "vienna_workbench" / "intervention_power.json",
    )
    args = parser.parse_args(argv)

    report = build_power_report(
        animals_per_arm=args.animals,
        iccs=args.iccs,
        design_icc=args.design_icc,
        trials_per_animal=args.trials,
        p_sham=args.p_sham,
        reps=args.reps,
        alpha=args.alpha,
        target_power=args.target_power,
        seed=args.seed,
    )
    text = json.dumps(report, indent=2)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
