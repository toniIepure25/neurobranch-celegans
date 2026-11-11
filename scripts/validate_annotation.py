"""Validate blinded eight-behaviour annotations and compute agreement.

Examples:
    python scripts/validate_annotation.py --annotations data/incoming/annotation
    python scripts/validate_annotation.py --annotations data/incoming/annotation --allow-low-kappa
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from neurobranch.data.validation import (  # noqa: E402
    validate_annotation_csv,
    validate_annotation_directory,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--kappa-threshold", type=float, default=0.6)
    parser.add_argument(
        "--allow-low-kappa",
        action="store_true",
        help="report agreement but do not fail when macro kappa is below threshold",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    if args.annotations.is_dir():
        report = validate_annotation_directory(
            args.annotations,
            kappa_threshold=args.kappa_threshold,
            require_kappa=not args.allow_low_kappa,
        )
    else:
        report = validate_annotation_csv(
            args.annotations,
            kappa_threshold=args.kappa_threshold,
            require_kappa=not args.allow_low_kappa,
        )
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
