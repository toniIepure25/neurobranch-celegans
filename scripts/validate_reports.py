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
from importlib.resources import files
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


def _schema_text(name: str, choices: tuple[str, ...], schema_version: int) -> str:
    """Prefer the packaged copy so an installed wheel validates itself."""
    filename = choices[1] if schema_version == 1 and len(choices) > 1 else choices[0]
    try:
        return files("neurobranch.schemas").joinpath(filename).read_text(encoding="utf-8")
    except (ModuleNotFoundError, FileNotFoundError):
        return _schema_path(name, choices, schema_version).read_text(encoding="utf-8")


def _validate_json(report: Path, choices: tuple[str, ...], failures: list[str]) -> None:
    payload = json.loads(report.read_text(encoding="utf-8"))
    version = int(payload.get("schema_version", 1))
    schema = json.loads(_schema_text(report.name, choices, version))
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


def _validate_phase12(reports: Path, failures: list[str]) -> None:
    summary = reports / "summary.json"
    if not summary.exists():
        failures.append(f"missing report: {summary}")
        return
    payload = json.loads(summary.read_text(encoding="utf-8"))
    schema = json.loads(
        files("neurobranch.schemas").joinpath("phase12_summary.schema.json").read_text(encoding="utf-8")
    )
    errors = sorted(Draft202012Validator(schema).iter_errors(payload), key=lambda error: list(error.path))
    if errors:
        failures.append("summary.json: " + "; ".join(error.message for error in errors))
    else:
        print("ok: summary.json (phase12 schema)")

    targets = reports / "hierarchical_by_target.csv"
    if targets.exists():
        table = pd.read_csv(targets)
        for column in ("target", "status", "reliability_n22", "ceiling_n22", "grid_edge_mass"):
            if column not in table.columns:
                failures.append(f"hierarchical_by_target.csv: missing {column}")
        if "reliability_n22" in table.columns:
            values = pd.to_numeric(table["reliability_n22"], errors="coerce").dropna()
            if not values.between(0, 1).all():
                failures.append("hierarchical_by_target.csv: reliability outside [0, 1]")
        print(f"ok: hierarchical_by_target.csv ({len(table)} rows)")
    else:
        failures.append(f"missing report: {targets}")

    benchmark = reports / "benchmark_cluster.csv"
    if benchmark.exists():
        table = pd.read_csv(benchmark)
        for column in ("delta", "delta_ci_low", "delta_ci_high", "p_holm", "mde_80", "status"):
            if column not in table.columns:
                failures.append(f"benchmark_cluster.csv: missing {column}")
        print(f"ok: benchmark_cluster.csv ({len(table)} rows)")
    else:
        failures.append(f"missing report: {benchmark}")


def _validate_phase12r(reports: Path, failures: list[str]) -> None:
    summary = reports / "summary.json"
    if not summary.exists():
        failures.append(f"missing report: {summary}")
        return
    payload = json.loads(summary.read_text(encoding="utf-8"))
    schema = json.loads(
        files("neurobranch.schemas")
        .joinpath("phase12r_summary.schema.json")
        .read_text(encoding="utf-8")
    )
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload), key=lambda error: list(error.path)
    )
    if errors:
        failures.append("summary.json: " + "; ".join(error.message for error in errors))
    else:
        print("ok: summary.json (phase12r schema)")

    targets = reports / "hierarchical_by_target.csv"
    if not targets.exists():
        failures.append(f"missing report: {targets}")
        return
    table = pd.read_csv(targets)
    for column in ("target", "status", "reason", "reliability_n22", "ceiling_n22", "grid_edge_mass"):
        if column not in table.columns:
            failures.append(f"hierarchical_by_target.csv: missing {column}")
    if "reliability_n22" in table.columns:
        values = pd.to_numeric(table["reliability_n22"], errors="coerce").dropna()
        if not values.between(0, 1).all():
            failures.append("hierarchical_by_target.csv: reliability outside [0, 1]")
    if "status" in table.columns and "reason" in table.columns:
        bad = table[(table["status"] != "ok") & table["reason"].isna()]
        if not bad.empty:
            failures.append("hierarchical_by_target.csv: ineligible rows lack a reason")
    print(f"ok: hierarchical_by_target.csv ({len(table)} rows)")

    for name in ("sbc.json", "heldout_ppc.json", "negative_control.json", "prior_sensitivity.json"):
        path = reports / name
        if not path.exists():
            failures.append(f"missing report: {path}")
            continue
        json.loads(path.read_text(encoding="utf-8"))
        print(f"ok: {name}")

    sensitivity = reports / "min_events_sensitivity.csv"
    if not sensitivity.exists():
        failures.append(f"missing report: {sensitivity}")
    else:
        table = pd.read_csv(sensitivity)
        for column in ("min_events", "n_ok", "pooled_reliability_median"):
            if column not in table.columns:
                failures.append(f"min_events_sensitivity.csv: missing {column}")
        print(f"ok: min_events_sensitivity.csv ({len(table)} rows)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=ROOT / "reports" / "phase10")
    parser.add_argument("--strict", action="store_true", help="fail when an expected report is missing")
    args = parser.parse_args(argv)

    failures: list[str] = []
    if args.reports.name == "simulator":
        report = args.reports / "gates.json"
        if not report.exists():
            failures.append(f"missing report: {report}")
        else:
            _validate_json(
                report,
                ("simulator_gates.schema.json", "simulator_gates.schema.json"),
                failures,
            )
        if failures:
            for failure in failures:
                print(f"FAIL: {failure}", file=sys.stderr)
            return 1
        return 0
    if args.reports.name == "phase12" or args.reports.name.startswith("phase12r"):
        if args.reports.name == "phase12":
            _validate_phase12(args.reports, failures)
        else:
            _validate_phase12r(args.reports, failures)
        if failures:
            for failure in failures:
                print(f"FAIL: {failure}", file=sys.stderr)
            return 1
        return 0
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
