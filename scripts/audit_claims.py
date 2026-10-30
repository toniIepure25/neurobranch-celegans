"""Audit the claims ledger against the artifacts it cites.

A claim is a derived object: its estimate and interval must be recomputable from
the summary or table named in ``evidence``.  This script is deliberately
explicit about the mapping so a silent handbook edit cannot drift away from the
numbers.

Exit codes: 0 clean, 1 audit failure, 2 missing input.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from neurobranch.benchmark.reporting import ROOT

STATUSES = {
    "supported",
    "underpowered",
    "bounded_null",
    "not_supported",
    "rejected",
    "exploratory",
}
EXPECTED: dict[str, dict[str, Any]] = {
    "phase9-single-animal-ceiling-below-half": {
        "source": "reliability_summary.json",
        "estimate": "mean_animal_reliability",
        "ci": ("animal_ci_low", "animal_ci_high"),
    },
    "phase9-repeat-ceiling-below-half": {
        "source": "reliability_summary.json",
        "estimate": "mean_repeat_reliability",
        "ci": ("repeat_ci_low", "repeat_ci_high"),
    },
    "phase10-fold-ceiling": {
        "source": "reliability_summary.json",
        "estimate": "pooled_ceiling_at_fold",
        "ci": None,
    },
}


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _numbers_match(left: float | None, right: float | None, tolerance: float) -> bool:
    if left is None or right is None:
        return left is right
    return abs(float(left) - float(right)) <= tolerance


def audit(reports: Path, *, tolerance: float = 1e-9) -> list[str]:
    claims_path = reports / "claims.json"
    summary_path = reports / "reliability_summary.json"
    if not claims_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"missing claims or summary under {reports}")
    payload = json.loads(claims_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    claims = payload.get("claims", [])
    failures: list[str] = []

    ids = [claim.get("id") for claim in claims]
    if len(ids) != len(set(ids)):
        failures.append("duplicate claim ids")
    if payload.get("schema_version") not in (1, 2):
        failures.append(f"unsupported claims schema: {payload.get('schema_version')}")

    for claim in claims:
        cid = str(claim.get("id"))
        status = claim.get("status")
        if status not in STATUSES:
            failures.append(f"{cid}: unknown status {status!r}")
        evidence = claim.get("evidence") or []
        if not evidence:
            failures.append(f"{cid}: no evidence")
        for item in evidence:
            if not _resolve(str(item)).exists():
                failures.append(f"{cid}: missing evidence {item}")
        low, high = claim.get("ci_low"), claim.get("ci_high")
        if low is not None and high is not None and float(low) > float(high):
            failures.append(f"{cid}: ci_low > ci_high")
        p_value, p_holm = claim.get("p_value"), claim.get("p_holm")
        if p_value is not None and p_holm is not None and float(p_holm) + tolerance < float(p_value):
            failures.append(f"{cid}: p_holm < p_value")
        if claim.get("test") == "upper_bound" and status == "supported":
            margin = claim.get("margin")
            if margin is None or high is None or float(high) >= float(margin):
                failures.append(f"{cid}: upper-bound claim does not have ci_high < margin")
        if claim.get("test") == "descriptive" and claim.get("statement"):
            match = re.search(r"about\s+([0-9]*\.?[0-9]+)", str(claim["statement"]))
            if match and claim.get("estimate") is not None:
                stated = float(match.group(1))
                if abs(stated - float(claim["estimate"])) > 0.005 + tolerance:
                    failures.append(
                        f"{cid}: statement says {stated}, estimate is {claim['estimate']}"
                    )
        expected = EXPECTED.get(str(cid))
        if expected is None:
            continue
        source = reports / str(expected["source"])
        if not source.exists():
            failures.append(f"{cid}: missing mapped source {source.name}")
            continue
        source_payload: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
        target = source_payload.get(str(expected["estimate"]))
        if not _numbers_match(claim.get("estimate"), target, tolerance):
            failures.append(f"{cid}: estimate {claim.get('estimate')} != {target}")
        ci_names = expected["ci"]
        if ci_names is not None:
            for claim_key, source_key in zip(("ci_low", "ci_high"), ci_names, strict=True):
                if not _numbers_match(claim.get(claim_key), source_payload.get(source_key), tolerance):
                    failures.append(
                        f"{cid}: {claim_key} {claim.get(claim_key)} != {source_payload.get(source_key)}"
                    )
        if cid == "phase10-fold-ceiling" and claim.get("n_eval_animals") != int(
            summary.get("fold_size", -1)
        ):
            failures.append(f"{cid}: n_eval_animals does not match summary fold_size")
    failures.extend(audit_phase11(tolerance=tolerance))
    failures.extend(audit_phase12(tolerance=tolerance))
    return failures


def audit_phase11(*, tolerance: float = 1e-9) -> list[str]:
    """Cross-check the Phase 11 card, summary and sensitivity table."""
    phase11 = ROOT / "reports" / "phase11"
    required = ["summary.json", "benchmark_card.json", "coverage_sensitivity.csv"]
    if not all((phase11 / name).exists() for name in required):
        return []
    summary = json.loads((phase11 / "summary.json").read_text(encoding="utf-8"))
    card = json.loads((phase11 / "benchmark_card.json").read_text(encoding="utf-8"))
    with (phase11 / "coverage_sensitivity.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = next((item for item in rows if float(item["min_responder_coverage"]) == 0.8), None)
    failures: list[str] = []
    if row is None:
        return ["phase11: coverage 0.8 row missing"]
    reliability = float(row["pooled_reliability_at_fold"])
    expected_ceiling = math.sqrt(reliability)
    if not _numbers_match(card.get("ceiling", {}).get("value"), expected_ceiling, tolerance):
        failures.append("phase11: card ceiling is not sqrt(coverage-aware reliability)")
    if card.get("claim_statuses", {}).get("phase10-fold-ceiling") != "not_supported":
        failures.append("phase11: fold-ceiling claim must be not_supported")
    headline = summary.get("headline", {})
    if not _numbers_match(
        headline.get("coverage_0_8_pooled_reliability"), reliability, tolerance
    ):
        failures.append("phase11: summary headline does not match sensitivity table")
    if headline.get("extrapolation_gate_passed"):
        failures.append("phase11: extrapolation gate reported as passed")
    if headline.get("shrinkage_gate_passed"):
        failures.append("phase11: shrinkage gate reported as passed")
    return failures



def audit_phase12(*, tolerance: float = 1e-9) -> list[str]:
    """Cross-check the Phase 12 gates, estimates and benchmark status."""
    phase12 = ROOT / "reports" / "phase12"
    required = ["summary.json", "hierarchical_by_target.csv", "benchmark_cluster.csv"]
    if not all((phase12 / name).exists() for name in required):
        return []
    summary = json.loads((phase12 / "summary.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    gates = summary.get("gates", {})
    gate_values = [gates.get(name) for name in ("g1_sbc", "g2_heldout_ppc", "g3_negative_control", "g4_prior_sensitivity")]
    if not all(isinstance(value, bool) for value in gate_values):
        failures.append("phase12: gate flags must be booleans")
    expected_status = "adopted" if all(gate_values) else "not_adopted_gate_failures"
    if summary.get("status") != expected_status:
        failures.append("phase12: status does not follow the gate flags")
    hierarchical = summary.get("hierarchical", {})
    for key in ("pooled_reliability_median", "pooled_ceiling_median"):
        value = hierarchical.get(key)
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            failures.append(f"phase12: {key} is not a probability")
    ceiling = hierarchical.get("pooled_ceiling_median")
    reliability = hierarchical.get("pooled_reliability_median")
    if (
        isinstance(ceiling, (int, float))
        and isinstance(reliability, (int, float))
        and not _numbers_match(float(ceiling), math.sqrt(float(reliability)), 1e-9)
    ):
        failures.append("phase12: hierarchical ceiling is not sqrt(reliability)")
    moment = summary.get("moment_phase11", {})
    phase11_summary_path = ROOT / "reports" / "phase11" / "summary.json"
    if phase11_summary_path.exists():
        phase11 = json.loads(phase11_summary_path.read_text(encoding="utf-8"))
        expected = phase11.get("headline", {}).get("coverage_0_8_pooled_reliability")
        if expected is not None and not _numbers_match(moment.get("pooled_reliability"), expected, tolerance):
            failures.append("phase12: Phase 11 moment reference drifted")
    if gates.get("g1_sbc") is False and summary.get("status") == "adopted":
        failures.append("phase12: adopted despite failed SBC")
    benchmark_path = phase12 / "benchmark_cluster.json"
    if benchmark_path.exists():
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        if benchmark.get("n_supported", 0) > 0:
            # Supported entries must have a positive paired lower bound and MDE <= SESOI.
            with (phase12 / "benchmark_cluster.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                if (
                    row.get("status") == "supported"
                    and (
                        float(row["delta_ci_low"]) <= 0
                        or float(row["mde_80"]) > float(benchmark.get("sesoi", 0.05))
                    )
                ):
                    failures.append("phase12: unsupported benchmark status")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=ROOT / "reports" / "phase10")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args(argv)
    try:
        failures = audit(args.reports, tolerance=args.tolerance)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 2
    if failures:
        print(f"claims audit failed ({len(failures)}):", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"claims audit passed: {args.reports / 'claims.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
