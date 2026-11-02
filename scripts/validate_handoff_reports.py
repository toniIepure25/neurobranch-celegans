"""Validate the supervisor handoff pack against frozen joint-dynamics results.

This validator is deliberately independent of the PDF renderer. It checks
document contracts, exact numbers, provenance notes, figures, page counts and
the SHA256 manifest.

Usage:
    python scripts/validate_handoff_reports.py
    python scripts/validate_handoff_reports.py --check-hashes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "simulator" / "supervisor_handoff"
JOINT = ROOT / "reports" / "simulator" / "joint_dynamics"
DOCS = ROOT / "docs"

WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION \u2014 NOT CAUSALLY VALIDATED."
TITLE = (
    "NeuroBranch: Evaluating Generative Neural Population Dynamics and "
    "Model-Dependent Perturbations in C. elegans"
)
DOC_NAMES = [
    "HANDOFF_EXECUTIVE_SUMMARY.md",
    "HANDOFF_RESEARCH_MEMO.md",
    "HANDOFF_PRESENTATION.md",
    "HANDOFF_SCIENTIFIC_CLAIMS.md",
    "HANDOFF_METHODS_AND_LIMITATIONS.md",
    "HANDOFF_RESEARCH_PROPOSALS.md",
    "HANDOFF_QUESTIONS_FOR_DENIS.md",
    "HANDOFF_DEMO_GUIDE.md",
    "HANDOFF_REPRODUCIBILITY.md",
]
REQUIRED_FILES = [
    "summary.json",
    "phase_timeline.csv",
    "claim_ledger.csv",
    "provenance_audit.json",
    "manifest.json",
    "NeuroBranch_Handoff_Memo.pdf",
    "NeuroBranch_Handoff_Presentation.html",
    "NeuroBranch_Handoff_Presentation.pdf",
]
FIGURE_NAMES = [
    "fig01_evidence_timeline.png",
    "fig02_central_result.png",
    "fig03_calibration.png",
    "fig04_example_trajectory.png",
    "fig05_proposed_scope.png",
]
STATUSES = {"supported", "exploratory", "mixed", "technical", "negative", "withdrawn", "not_evaluable"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def page_count(path: Path) -> int:
    return len(re.findall(rb"/Type\s*/Page\b", path.read_bytes()))


def main(check_hashes: bool = False) -> int:
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    for name in DOC_NAMES:
        check((DOCS / name).exists(), f"missing document: {name}")
    for name in REQUIRED_FILES:
        check((OUT / name).exists(), f"missing artifact: {name}")
    for name in FIGURE_NAMES:
        check((OUT / "figures" / name).exists(), f"missing figure: {name}")
    for path in [
        ROOT / "app" / "handoff_demo.py",
        JOINT / "example_observed.csv",
        JOINT / "example_bands.csv",
        JOINT / "perturbation_status.json",
    ]:
        check(path.exists(), f"missing demo dependency: {path.relative_to(ROOT)}")
    if failures:
        print("FAILED")
        print("\n".join(failures))
        return 1

    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    check(summary.get("title") == TITLE, "summary title does not match the required title")
    check(summary.get("warning") == WARNING, "summary warning is not exact")
    check(summary.get("status", {}).get("warning") == WARNING, "status warning is not exact")
    check(summary.get("status", {}).get("biological_validation") == "NOT_EVALUABLE", "biological validation is not NOT_EVALUABLE")
    check(summary.get("status", {}).get("probabilistic") == "PROBABILISTIC CALIBRATION NOT ESTABLISHED", "probabilistic label changed")
    check(summary.get("data", {}).get("animal_independence", "").startswith("not documented"), "animal independence is overstated")
    check("not confirmatory" in summary.get("central_claim", {}).get("status", ""), "central claim is not marked exploratory")

    memo = (DOCS / "HANDOFF_RESEARCH_MEMO.md").read_text(encoding="utf-8")
    normalized = memo.replace("*C. elegans*", "C. elegans")
    check(normalized.count(f"# {TITLE}") == 1, "memo does not carry the exact required title once")
    check(memo.count("## ") == 7, "memo section count changed")

    for name in DOC_NAMES:
        text = (DOCS / name).read_text(encoding="utf-8")
        check(WARNING in text, f"exact warning missing from {name}")
        check("NOT_EVALUABLE" in text, f"biological validation label missing from {name}")

    for name in ["HANDOFF_EXECUTIVE_SUMMARY.md", "HANDOFF_RESEARCH_MEMO.md", "HANDOFF_SCIENTIFIC_CLAIMS.md"]:
        text = (DOCS / name).read_text(encoding="utf-8").lower()
        check("not a systematic review" in text or "not systematic" in text, f"systematic-review disclaimer missing from {name}")

    forbidden_positive = [
        "is a validated digital twin",
        "achieves causal control",
        "demonstrates causally valid",
        "animal-independent replication is established",
        "confirmatory significance was achieved",
        "reliably predicts stimulation",
    ]
    corpus = "\n".join((DOCS / name).read_text(encoding="utf-8") for name in DOC_NAMES)
    corpus += "\n" + (ROOT / "app" / "handoff_demo.py").read_text(encoding="utf-8")
    for phrase in forbidden_positive:
        check(phrase not in corpus.lower(), f"forbidden positive claim present: {phrase}")
    check("PROBABILISTIC CALIBRATION NOT ESTABLISHED" in corpus, "stop label missing from docs")
    check("five sessions are not five" in corpus.lower() or "5 sessions, not 5 animals" in corpus, "session/animal warning missing")

    timeline = pd.read_csv(OUT / "phase_timeline.csv")
    check(len(timeline) == 12, f"expected 12 timeline rows, found {len(timeline)}")
    check(timeline["order"].tolist() == list(range(1, 13)), "timeline order is not 1..12")
    check(set(timeline["status"]).issubset(STATUSES), "timeline has an unknown status")
    check(timeline["source"].str.len().min() > 0, "timeline source is empty")

    claims = pd.read_csv(OUT / "claim_ledger.csv")
    check(len(claims) == 18, f"expected 18 claims, found {len(claims)}")
    check(claims["id"].tolist() == [f"C{i}" for i in range(1, 19)], "claim IDs are not C1..C18")
    check(claims["id"].is_unique, "claim IDs are not unique")
    check(set(claims["status"]).issubset(STATUSES), "claim ledger has an unknown status")
    check((claims["caveat"].str.len() > 0).all(), "a claim is missing its caveat")

    provenance = json.loads((OUT / "provenance_audit.json").read_text(encoding="utf-8"))
    ids = {item["id"] for item in provenance.get("discrepancies", [])}
    for needed in ["five_sessions_animals", "test_reuse", "phase15_ceiling", "kato_stim_availability", "j2_j5_equivalence"]:
        check(needed in ids, f"provenance audit missing {needed}")
    check(provenance.get("unresolved"), "provenance unresolved list is empty")

    scores = pd.read_csv(JOINT / "score_summary.csv").set_index("model")
    summary_scores = {row["model"]: row for row in summary["scores"]}
    for model in [
        "J0_persistence",
        "J1_independent_diag",
        "J2_population_diag",
        "J3_population_full",
        "J4_independent_full",
        "J5_population_lead1_boot",
    ]:
        check(model in summary_scores, f"summary missing model {model}")
        if model in summary_scores:
            check(
                np.isclose(summary_scores[model]["energy_h20"], scores.loc[model, "energy_h20"], atol=1e-6),
                f"summary energy changed for {model}",
            )
            check(
                np.isclose(summary_scores[model]["trajectory_h20"], scores.loc[model, "trajectory_energy_h20"], atol=1e-6),
                f"summary trajectory changed for {model}",
            )

    paired = pd.read_csv(JOINT / "paired_scores.csv")
    primary = summary["central_claim"]["primary"]
    row = paired[
        (paired["model"] == "J5_population_lead1_boot")
        & (paired["reference"] == "J1_independent_diag")
        & (paired["metric"] == "energy_h20")
    ].iloc[0]
    check(np.isclose(primary["improvement"], row["mean_improvement"], atol=1e-6), "primary improvement changed")
    check(np.isclose(primary["ci_low"], row["ci_low"], atol=1e-6), "primary CI low changed")
    check(np.isclose(primary["ci_high"], row["ci_high"], atol=1e-6), "primary CI high changed")
    check(primary["modal_status"] == "exploratory" if "modal_status" in primary else True, "status drift")

    gates = json.loads((JOINT / "gates.json").read_text(encoding="utf-8"))
    check(gates["G4_marginal_calibration"]["status"] == "FAIL", "G4 is not FAIL")
    check(gates["G5_joint_trajectory"]["status"] == "FAIL", "G5 is not FAIL")
    check(gates["G8_biological_intervention"]["status"] == "NOT_EVALUABLE", "G8 is not NOT_EVALUABLE")
    check(summary["gates"]["stop_label"] == "PROBABILISTIC CALIBRATION NOT ESTABLISHED", "summary stop label changed")

    calibration = pd.read_csv(JOINT / "calibration_h20.csv").set_index("model")
    j5 = calibration.loc["J5_population_lead1_boot"]
    check(abs(float(j5["coverage_full_50"]) - 0.587781) < 1e-5, "J5 full 50% coverage changed")
    check(abs(float(j5["coverage_full_95"]) - 0.946998) < 1e-5, "J5 full 95% coverage changed")
    check(abs(float(scores.loc["J5_population_lead1_boot", "neural_cov_error"]) - 0.604035) < 1e-5, "J5 covariance error changed")

    sessions = summary["sessions"]
    check(len(sessions) == 5, "expected five session contrasts")
    check(sum(1 for row in sessions if row["improvement"] > 0) == 4, "session sign count changed")
    check(sum(1 for row in sessions if row["ci_low"] > 0) == 3, "session CI count changed")
    blocks = summary["block_sensitivity"]
    check([row["block"] for row in blocks] == [3, 5, 10], "block sensitivity set changed")
    check(blocks[2]["ci_low"] < 0, "block-10 interval should include zero")

    check(page_count(OUT / "NeuroBranch_Handoff_Memo.pdf") == 2, "memo is not two pages")
    check(page_count(OUT / "NeuroBranch_Handoff_Presentation.pdf") == 11, "presentation is not eleven pages")

    for name in FIGURE_NAMES:
        path = OUT / "figures" / name
        check(path.stat().st_size > 20_000, f"figure is suspiciously small: {name}")
        check(path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"figure is not a PNG: {name}")

    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    check(manifest.get("root") == "reports/simulator/supervisor_handoff", "manifest root changed")
    check(manifest.get("hash_algorithm") == "sha256", "manifest hash algorithm changed")
    check("trajectory_examples.npz" not in manifest.get("files", {}), "compact manifest includes the optional NPZ")
    check(all(not name.endswith((".mat", ".pt", ".pth", ".ckpt")) for name in manifest.get("files", {})), "manifest includes raw data or weights")

    if check_hashes:
        for name, record in manifest.get("files", {}).items():
            path = OUT / name
            check(path.exists(), f"manifest file missing: {name}")
            if path.exists():
                check(path.stat().st_size == record["bytes"], f"manifest size mismatch: {name}")
                check(sha256(path) == record["sha256"], f"manifest hash mismatch: {name}")
        for name, record in manifest.get("documents", {}).items():
            path = DOCS / name
            check(path.exists(), f"manifest document missing: {name}")
            if path.exists() and sha256(path) != record["sha256"]:
                failures.append(f"manifest document hash mismatch: {name}")
        for name, record in manifest.get("referenced_sources", {}).items():
            path = ROOT / name
            check(path.exists(), f"referenced source missing: {name}")
            if path.exists() and sha256(path) != record["sha256"]:
                failures.append(f"referenced source hash mismatch: {name}")
        actual = {p.relative_to(OUT).as_posix() for p in OUT.rglob("*") if p.is_file() and p.name != "manifest.json"}
        check(actual == set(manifest.get("files", {})), "manifest file set does not match the report tree")

    if failures:
        print("FAILED")
        print("\n".join(failures))
        return 1
    print(
        f"validated {len(DOC_NAMES)} handoff docs, {len(REQUIRED_FILES)} artifacts, "
        f"{len(FIGURE_NAMES)} figures, 12 phases and 18 claims"
    )
    if check_hashes:
        print("all manifest and document hashes verified")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-hashes", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(check_hashes=args.check_hashes))
