"""Validate the Phase 13 summary, gate logic and artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas" / "phase13_summary.schema.json"
DEFAULT_REPORTS = ROOT / "reports" / "phase13"
TEXT_SUFFIXES = {".json", ".csv", ".md", ".txt", ".sha256"}


def _sha256(path: Path) -> str:
    """Hash with LF normalization so the check is portable across checkouts."""
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _issue(code: str, message: str, **context: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **context}


def validate(reports: Path = DEFAULT_REPORTS) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    summary_path = reports / "summary.json"
    if not summary_path.is_file():
        return {"status": "fail", "issues": [_issue("missing_summary", str(summary_path))]}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    try:
        import jsonschema
    except ImportError:
        issues.append(_issue("missing_jsonschema", "jsonschema is not installed"))
    else:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        validator = jsonschema.Draft202012Validator(schema)
        for error in sorted(validator.iter_errors(summary), key=lambda item: list(item.path)):
            issues.append(
                _issue(
                    "schema",
                    error.message,
                    path="/".join(str(part) for part in error.path),
                )
            )

    gates = summary.get("gates", [])
    if not isinstance(gates, list) or len(gates) < 11:
        issues.append(_issue("gate_count", "expected at least 11 frozen gates"))
    failures = sum(1 for gate in gates if gate.get("result") == "fail")
    if failures != summary.get("n_gates_failed"):
        issues.append(
            _issue(
                "gate_count_mismatch",
                "n_gates_failed does not match gate array",
                observed=summary.get("n_gates_failed"),
                counted=failures,
            )
        )
    expected_decision = "adopted" if failures == 0 else "not_adopted_gate_failures"
    if summary.get("decision") != expected_decision:
        issues.append(
            _issue(
                "decision_mismatch",
                "decision does not follow the frozen all-gates rule",
                observed=summary.get("decision"),
                expected=expected_decision,
            )
        )
    if summary.get("quick") is True and summary.get("decision") == "adopted":
        issues.append(_issue("quick_adoption", "quick mode cannot adopt a model"))

    real = summary.get("real", {})
    for model in ("m0_reference", "m1_projection_scaled", "m2_event_variance"):
        block = real.get(model) if isinstance(real, dict) else None
        if not isinstance(block, dict):
            issues.append(_issue("missing_model", "missing real-data fit", model=model))
            continue
        for field in ("n_targets", "n_ok", "pooled_reliability_median"):
            if field not in block:
                issues.append(_issue("missing_field", field, model=model))
        if block.get("n_ok", 0) > block.get("n_targets", 0):
            issues.append(_issue("impossible_counts", "n_ok exceeds n_targets", model=model))

    primary = real.get("m2_event_variance", {}) if isinstance(real, dict) else {}
    estimate = primary.get("pooled_reliability_median")
    if isinstance(estimate, (int, float)) and not 0.0 <= float(estimate) <= 1.0:
        issues.append(_issue("out_of_range", "primary reliability outside [0, 1]"))

    neg = summary.get("negative_control", {})
    if neg.get("n_eligible", 0) <= 0:
        issues.append(_issue("empty_negative", "negative control has no eligible targets"))
    sbc = summary.get("sbc_m2", {})
    if sbc.get("n_failed_fits", 0) > 0.05 * max(sbc.get("n_fits", 1), 1):
        issues.append(_issue("sbc_integrity", "too many failed SBC fits"))
    if not isinstance(sbc.get("event_variance_coverage"), (int, float)):
        issues.append(_issue("missing_event_coverage", "event-variance coverage absent"))

    manifest_path = reports / "manifest.json"
    if not manifest_path.is_file():
        issues.append(_issue("missing_manifest", str(manifest_path)))
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, expected in manifest.get("artifacts", {}).items():
            path = reports / name
            if not path.is_file():
                issues.append(_issue("missing_artifact", name))
            elif _sha256(path) != expected:
                issues.append(_issue("artifact_hash_mismatch", name))
        checksum_path = reports / "manifest.sha256"
        if not checksum_path.is_file():
            issues.append(_issue("missing_manifest_checksum", str(checksum_path)))
        else:
            expected_checksum = _sha256(manifest_path)
            recorded = checksum_path.read_text(encoding="utf-8").split()[0]
            if recorded != expected_checksum:
                issues.append(_issue("manifest_checksum_mismatch", "manifest.sha256 is stale"))
        if manifest.get("git_dirty") is True and not summary.get("quick"):
            issues.append(
                _issue(
                    "dirty_worktree",
                    "full decision artifacts were produced from a dirty worktree",
                )
            )

    return {
        "status": "pass" if not issues else "fail",
        "n_issues": len(issues),
        "issues": issues,
        "decision": summary.get("decision"),
        "n_gates_failed": failures,
        "primary_estimate": estimate,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = validate(args.reports)
    out = args.out or args.reports / "validation.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
