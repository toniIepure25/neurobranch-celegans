"""Validate a randomized intervention trial table before unblinding.

Example:
    python scripts/validate_intervention_trials.py \
        --trials data/incoming/intervention/trials.csv --min-per-arm 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from neurobranch.data.validation import validate_intervention_trials  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--min-per-arm", type=int, default=6)
    parser.add_argument("--check-paths", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    report = validate_intervention_trials(
        args.trials,
        min_per_arm=args.min_per_arm,
        check_paths=args.check_paths,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
