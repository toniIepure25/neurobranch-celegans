"""Validate NeuroBranch reports against the bundled schemas and table contracts.

Usage:
    python scripts/validate_reports.py                 # validate reports/phase10
    python scripts/validate_reports.py --reports DIR   # any report directory
    python scripts/validate_reports.py --strict        # fail on missing files
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schemas"
JSON_TARGETS = {
    "reliability_summary.json": ("reliability_summary.schema.json", "reliability_summary_v1.schema.json"),
    "claims.json": ("claims.schema.json", "claims_v1.schema.json"),
}
CSV_TARGETS = {
    "reliability_by_target.csv": {
        "required": ["target", "repeat_reliability", "single_animal_reliability"],
        "bounded": ["repeat_reliability", "single_animal_reliability", "pooled_reliability", "animal_mean_reliability"],
        "positive": ["n_animals"],
    },
    "ceiling_curve.csv": {
        "required": ["n_eval", "n_targets", "mean_reliability", "ceiling"],
        "bounded": ["mean_reliability", "ceiling", "ci_low", "ci_high"],
        "positive": ["n_eval", "n_targets"],
    },
}


def _schema_path(name: str, choices: tuple[str, ...], schema_version: int) -> Path:
    if schema_version == 1 and len(choices) > 1:
        return SCHEMA_DIR / choices[1]
    return SCHEMA_DIR / choices[0]


def _validate_json(report: Path, choices: tuple[str, ...], failures: list[str]) -> None:
    payload = json.loads(report.read_text(encoding="utf-8"))
    version = int(payload.get("schema_version", 1))
    schema = json.loads(_schema_path(report.name, choices, version).read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    if errors:
        failures.append(f"{report.name}: " + "; ".join(error.message for error in errors))
    else:
        print(f"ok: {report.name} (schema v{version})")


def _validate_csv(report: Path, contract: dict[str, list[str]], failures: list[str]) -> None:
    table = pd.read_csv(report)
    missing = set(contract["required"]) - set(table.columns)
    if missing:
        failures.append(f"{report.name}: missing columns {sorted(missing)}")
        return
    for column in contract["bounded"]:
        if column not in table.columns:
            continue
        values = pd.to_numeric(table[column], errors="coerce")
        finite = values.dropna()
        if not finite.between(0.0, 1.0).all():
            failures.append(f"{report.name}: {column} outside [0, 1]")
    for column in contract["positive"]:
        if column not in table.columns:
            continue
        values = pd.to_numeric(table[column], errors="coerce")
        if (values.dropna() < 0).any():
            failures.append(f"{report.name}: {column} has negative values")
    if table.empty:
        failures.append(f"{report.name}: table is empty")
    else:
        print(f"ok: {report.name} ({len(table)} rows)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=ROOT / "reports" / "phase10")
    parser.add_argument("--strict", action="store_true", help="fail when an expected report is missing")
    args = parser.parse_args(argv)

    failures: list[str] = []
    for name, choices in JSON_TARGETS.items():
        report = args.reports / name
        if not report.exists():
            message = f"missing report: {report}"
            if args.strict:
                failures.append(message)
            else:
                print(f"skip: {message}")
            continue
        _validate_json(report, choices, failures)

    for name, contract in CSV_TARGETS.items():
        report = args.reports / name
        if not report.exists():
            message = f"missing report: {report}"
            if args.strict and args.reports.name == "phase10":
                failures.append(message)
            elif not args.strict:
                print(f"skip: {message}")
            continue
        _validate_csv(report, contract, failures)

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
