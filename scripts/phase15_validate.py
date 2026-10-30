"""Validate the Phase 15 summary, gate logic and artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS = ROOT / "reports" / "phase15"
SCHEMA = ROOT / "schemas" / "phase15_summary.schema.json"


def _issue(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _sha256(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in {".json", ".csv", ".md", ".txt", ".sha256"}:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _close(left: Any, right: Any, tolerance: float = 1e-8) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def _expected_gate_statuses(summary: dict[str, Any], reports: Path) -> dict[str, str]:
    estimands = summary["estimands"]
    expected: dict[str, str] = {}
    expected["G1_estimand_completeness"] = (
        "pass"
        if all(key in estimands for key in ("R_obs", "R_22", "R_id"))
        and all(estimands[key]["n_targets"] > 0 for key in ("R_obs", "R_22", "R_id"))
        else "fail"
    )
    expected["G2_jackknife_integrity"] = (
        "pass"
        if all(
            estimands[key]["jackknife"].get("status") == "ok"
            and estimands[key]["jackknife"].get("valid_fraction", 0.0) >= 0.95
            for key in ("R_obs", "R_22", "R_id")
        )
        else "fail"
    )
    calibration = summary.get("calibration", {}).get("estimands", {})
    expected["G3_calibration"] = (
        "pass"
        if all(calibration.get(key, {}).get("status") == "calibrated" for key in ("obs", "fold", "id"))
        else "fail"
    )
    null_ok = True
    for key in ("R_obs", "R_22", "R_id"):
        nulls = estimands[key].get("null", {})
        for mode in ("permutation", "signflip"):
            entry = nulls.get(mode, {})
            if not (
                entry.get("median", float("inf")) <= 0.05
                and entry.get("q95", float("inf")) <= 0.15
            ):
                null_ok = False
    expected["G4_null_control"] = "pass" if null_ok else "fail"
    extrapolation = summary["extrapolation"]
    expected["G5_extrapolation"] = (
        "pass"
        if bool(extrapolation.get("gate_passed", 0.0))
        and float(extrapolation.get("median_abs_error", float("inf"))) <= 0.10
        and abs(float(extrapolation.get("bias", float("inf")))) <= 0.05
        else "fail"
    )
    expected["G6_ranking_stability"] = "pass" if summary["ranking"].get("status") == "ok" else "fail"
    variants = summary.get("variance_sensitivity", {})
    try:
        v1 = variants["v1_primary"]["estimands"]
        v0 = variants["v0_historical"]["estimands"]
        deltas = [abs(v1[key]["estimate"] - v0[key]["estimate"]) for key in ("R_obs", "R_22", "R_id")]
        expected["G7_variance_robustness"] = "pass" if max(deltas) <= 0.05 else "fail"
    except (KeyError, TypeError, ValueError):
        expected["G7_variance_robustness"] = "fail"
    expected["G8_reproducibility"] = "pass"
    del reports
    return expected


def validate(reports: Path) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    summary_path = reports / "summary.json"
    if not summary_path.is_file():
        return {
            "status": "fail",
            "n_issues": 1,
            "issues": [_issue("missing_summary", str(summary_path))],
            "decision": None,
            "n_gates_failed": 0,
            "primary_estimate": None,
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    try:
        import jsonschema
    except ImportError:
        issues.append(_issue("missing_jsonschema", "jsonschema is not installed"))
    else:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        validator = jsonschema.Draft202012Validator(schema)
        for error in sorted(validator.iter_errors(summary), key=lambda item: list(item.path)):
            issues.append(_issue("schema", f"{'/'.join(map(str, error.path))}: {error.message}"))

    table_path = reports / "estimands.csv"
    if not table_path.is_file():
        issues.append(_issue("missing_estimands", str(table_path)))
        table = pd.DataFrame()
    else:
        table = pd.read_csv(table_path)
        required = {"target", "n_animals", "extrapolated_22", "in_identified_subset"}
        missing = required - set(table.columns)
        if missing:
            issues.append(_issue("estimands_columns", f"missing {sorted(missing)}"))
        else:
            n_all = int(len(table))
            n_id = int(table["in_identified_subset"].sum())
            n_extrapolated = int(table["extrapolated_22"].sum())
            estimands = summary["estimands"]
            if int(estimands["R_obs"]["n_targets"]) != n_all:
                issues.append(_issue("estimand_count", "R_obs target count does not match table"))
            if int(estimands["R_22"]["n_targets"]) != n_all:
                issues.append(_issue("estimand_count", "R_22 target count does not match table"))
            if int(estimands["R_id"]["n_targets"]) != n_id:
                issues.append(_issue("estimand_count", "R_id target count does not match table"))
            if int(estimands["R_22"]["n_extrapolated"]) != n_extrapolated:
                issues.append(_issue("extrapolation_count", "R_22 extrapolated count does not match table"))
            if int(estimands["R_obs"]["n_extrapolated"]) != 0:
                issues.append(_issue("extrapolation_count", "R_obs must not mark extrapolation"))
            if int(summary["decision"]["r22_extrapolated_targets"]) != n_extrapolated:
                issues.append(_issue("decision_extrapolation", "decision extrapolated count is inconsistent"))

    for key in ("R_obs", "R_22", "R_id"):
        block = summary["estimands"].get(key, {})
        jack = block.get("jackknife", {})
        if not _close(block.get("estimate"), jack.get("estimate")):
            issues.append(_issue("jackknife_estimate", f"{key} jackknife estimate differs from point estimate"))
        if not (0.0 <= jack.get("ci_low", float("nan")) <= jack.get("ci_high", float("nan")) <= 1.0):
            issues.append(_issue("jackknife_interval", f"{key} has an invalid interval"))
        if jack.get("status") != "ok":
            issues.append(_issue("jackknife_status", f"{key} jackknife status is {jack.get('status')}"))
        if not (0.0 <= block.get("estimate", float("nan")) <= 1.0):
            issues.append(_issue("estimate_range", f"{key} estimate is outside [0, 1]"))
        if not _close(block.get("ceiling"), float(block.get("estimate", 0.0)) ** 0.5, tolerance=1e-8):
            issues.append(_issue("ceiling", f"{key} ceiling is not sqrt(estimate)"))

    expected = _expected_gate_statuses(summary, reports)
    for name, status in expected.items():
        recorded = summary.get("gates", {}).get(name, {}).get("status")
        if name == "G8_reproducibility":
            continue
        if recorded != status:
            issues.append(_issue("gate_mismatch", f"{name}: recorded {recorded}, expected {status}"))
    decision = summary.get("decision", {})
    if (
        expected["G3_calibration"] != "pass" or expected["G4_null_control"] != "pass"
    ) and decision.get("calibrated_interval") is True:
        issues.append(_issue("decision_calibration", "calibrated_interval is true while calibration or nulls failed"))
    if expected["G5_extrapolation"] != "pass" and decision.get("extrapolation_status") == "supported":
        issues.append(_issue("decision_extrapolation", "R_22 is called supported while extrapolation failed"))
    if expected["G6_ranking_stability"] != "pass" and decision.get("ranking_claim") is True:
        issues.append(_issue("decision_ranking", "ranking is claimed while stability failed"))
    if expected["G3_calibration"] != "pass" and str(decision.get("label", "")).startswith("calibrated"):
        issues.append(_issue("decision_label", "calibration label is used while G3 failed"))

    manifest_path = reports / "manifest.json"
    manifest_g8 = "fail"
    if not manifest_path.is_file():
        issues.append(_issue("missing_manifest", str(manifest_path)))
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, expected_hash in manifest.get("artifacts", {}).items():
            path = reports / name
            if not path.is_file():
                issues.append(_issue("missing_artifact", name))
            elif _sha256(path) != expected_hash:
                issues.append(_issue("artifact_hash_mismatch", name))
        checksum_path = reports / "manifest.sha256"
        if not checksum_path.is_file():
            issues.append(_issue("missing_manifest_checksum", str(checksum_path)))
        else:
            recorded = checksum_path.read_text(encoding="utf-8").split()[0]
            if recorded != _sha256(manifest_path):
                issues.append(_issue("manifest_checksum_mismatch", "manifest.sha256 is stale"))
        if manifest.get("git_dirty") is True:
            issues.append(_issue("dirty_worktree", "decision artifacts were produced from a dirty worktree"))
        else:
            manifest_g8 = "pass"
    if manifest_g8 == "pass":
        summary.setdefault("gates", {}).setdefault("G8_reproducibility", {})["status"] = "pass"

    n_failed = sum(1 for gate in expected.values() if gate == "fail")
    return {
        "status": "pass" if not issues else "fail",
        "n_issues": len(issues),
        "issues": issues,
        "decision": decision,
        "n_gates_failed": n_failed,
        "primary_estimate": summary.get("estimands", {}).get("R_obs", {}).get("estimate"),
        "manifest_g8": manifest_g8,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    reports = args.reports.resolve()
    result = validate(reports)
    out = args.out or (reports / "validation.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: result[key] for key in ("status", "n_issues", "n_gates_failed")}, indent=2))
    for issue in result["issues"][:20]:
        print(f"{issue['code']}: {issue['message']}")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
