"""Validate Phase 9 JSON reports against the bundled schemas.

Usage:
    python scripts/validate_reports.py            # validate reports/phase9
    python scripts/validate_reports.py --strict   # also fail on missing files
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schemas"
TARGETS = {
    "reliability_summary.json": "reliability_summary.schema.json",
    "claims.json": "claims.schema.json",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=ROOT / "reports" / "phase9")
    parser.add_argument("--strict", action="store_true", help="fail when an expected report is missing")
    args = parser.parse_args(argv)

    failures: list[str] = []
    for name, schema_name in TARGETS.items():
        report = args.reports / name
        if not report.exists():
            message = f"missing report: {report}"
            if args.strict:
                failures.append(message)
            else:
                print(f"skip: {message}")
            continue
        schema = json.loads((SCHEMA_DIR / schema_name).read_text(encoding="utf-8"))
        payload = json.loads(report.read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
        if errors:
            failures.append(f"{name}: " + "; ".join(error.message for error in errors))
        else:
            print(f"ok: {name}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
