"""Validate the Phase 14 summary, gate logic, PPC and artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas" / "phase14_summary.schema.json"
DEFAULT_REPORTS = ROOT / "reports" / "phase14"
TEXT_SUFFIXES = {".json", ".csv", ".md", ".txt", ".sha256"}
EXPECTED_GATES = {
    "G1a_sbc_coverage",
    "G1b_sbc_rank",
    "G1c_grid_edge",
    "G1d_fit_integrity",
    "G1e_one_event_stratum",
    "G2_heldout_ppc",
    "G3_negative_control",
    "G3b_signflip_control",
    "G4_prior_sensitivity",
    "G5_event_variance_recovery",
    "G6_reference_agreement",
    "G7_spike_calibration",
}


def _sha256(path: Path) -> str:
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

    gates = summary.get("gates", {})
    if not isinstance(gates, dict):
        issues.append(_issue("gate_type", "gates must be an object"))
        gates = {}
    missing_gates = EXPECTED_GATES - set(gates)
    if missing_gates:
        issues.append(_issue("gate_count", "missing frozen gates", missing=sorted(missing_gates)))
    failures = [name for name, gate in gates.items() if not bool(gate.get("passed", False))]
    if len(failures) != summary.get("n_gates_failed"):
        issues.append(
            _issue(
                "gate_count_mismatch",
                "n_gates_failed does not match gate object",
                observed=summary.get("n_gates_failed"),
                counted=len(failures),
            )
        )
    if set(failures) != set(summary.get("failed_gates", [])):
        issues.append(_issue("failed_gate_mismatch", "failed_gates does not match gate object"))
    expected_decision = "adopted" if not failures else "not_adopted_gate_failures"
    if summary.get("decision") != expected_decision:
        issues.append(
            _issue(
                "decision_mismatch",
                "decision does not follow the frozen all-gates rule",
                observed=summary.get("decision"),
                expected=expected_decision,
            )
        )

    real = summary.get("real_fit", {})
    estimate = real.get("pooled_reliability_median_n22")
    if not isinstance(estimate, (int, float)) or not 0.0 <= float(estimate) <= 1.0:
        issues.append(_issue("out_of_range", "real R(22) outside [0, 1]"))
    if real.get("n_ok", 0) > real.get("n_targets", 0):
        issues.append(_issue("impossible_counts", "n_ok exceeds n_targets"))

    ppc_path = reports / "heldout_ppc.json"
    if not ppc_path.is_file():
        issues.append(_issue("missing_ppc", "held-out PPC artifact is required"))
    else:
        ppc = json.loads(ppc_path.read_text(encoding="utf-8"))
        if ppc.get("status") != "ok" or int(ppc.get("n", 0)) < 100:
            issues.append(
                _issue(
                    "ppc_not_run",
                    "held-out PPC must be run with at least 100 target-level records",
                    status=ppc.get("status"),
                    n=ppc.get("n"),
                )
            )
        if "G2_heldout_ppc" in gates:
            coverage = ppc.get("coverage")
            mae = ppc.get("median_abs_error")
            gate = gates["G2_heldout_ppc"]
            recomputed = (
                isinstance(coverage, (int, float))
                and isinstance(mae, (int, float))
                and float(coverage) >= 0.90
                and float(mae) <= 0.10
            )
            if bool(gate.get("passed")) != recomputed:
                issues.append(_issue("ppc_gate_mismatch", "PPC gate does not match PPC artifact"))

    sbc_path = reports / "sbc.json"
    if not sbc_path.is_file():
        issues.append(_issue("missing_sbc", "SBC artifact is required"))
    else:
        sbc = json.loads(sbc_path.read_text(encoding="utf-8"))
        if int(sbc.get("n_fits", 0)) < 3000:
            issues.append(_issue("sbc_underpowered", "SBC must contain at least 3000 fits"))
        if int(sbc.get("n_failed_fits", 0)) > 0.05 * max(int(sbc.get("n_fits", 1)), 1):
            issues.append(_issue("sbc_integrity", "too many failed SBC fits"))
        if "spike_calibration" not in sbc:
            issues.append(_issue("missing_spike_calibration", "spike calibration absent"))
        if "event_variance_coverage" not in sbc:
            issues.append(_issue("missing_event_coverage", "event-variance coverage absent"))

    reference_path = reports / "reference_audit.json"
    if not reference_path.is_file():
        issues.append(_issue("missing_reference_audit", "reference audit artifact is required"))
    else:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if not bool(reference.get("decisions", {}).get("A1_reproduced", False)):
            issues.append(_issue("reference_a1_failed", "V0 did not reproduce the Phase 11 reference"))

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
        if manifest.get("git_dirty") is True:
            issues.append(_issue("dirty_worktree", "decision artifacts were produced from a dirty worktree"))

    return {
        "status": "pass" if not issues else "fail",
        "n_issues": len(issues),
        "issues": issues,
        "decision": summary.get("decision"),
        "n_gates_failed": len(failures),
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
