"""Conformance status of the external references this benchmark compares against.

The rule is that a reference is only marked as reproduced when its own code ran and
produced the numbers being compared.  Reading a paper, loading somebody's published
embeddings, or re-implementing an algorithm is *not* reproduction, and each case is
labelled separately.

Statuses used, matching the Stage-One deliverable vocabulary:

``verified``
    The item was checked against its own source in this environment.
``simulated``
    The item is our own synthetic construction, not an external method.
``unidentifiable``
    The item cannot be estimated from the information the protocol permits.
``falsified``
    The item was tested and failed.
``not_reproduced``
    The item's own implementation did not run here, whatever its documentation says.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neurobranch.config import PROJECT_ROOT

REFERENCE_ROOT = PROJECT_ROOT / "references" / "external"
NC_MCM_ROOT = REFERENCE_ROOT / "nc-mcm"


@dataclass(frozen=True)
class ConformanceCheck:
    name: str
    status: str
    detail: str
    evidence: str

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "evidence": self.evidence,
        }


def probe_environment() -> dict[str, Any]:
    """What is actually available here, without executing any third-party code."""
    julia = shutil.which("julia")
    npz = sorted(NC_MCM_ROOT.glob("**/*.npz")) if NC_MCM_ROOT.exists() else []
    julia_sources = sorted(NC_MCM_ROOT.glob("**/*.jl")) if NC_MCM_ROOT.exists() else []
    return {
        "julia_executable": julia,
        "nc_mcm_root_present": NC_MCM_ROOT.exists(),
        "nc_mcm_julia_sources": len(julia_sources),
        "nc_mcm_precomputed_models": len(npz),
        "bundlenet_root_present": (REFERENCE_ROOT / "bundlenet").exists() or (REFERENCE_ROOT / "bundlenet-net").exists(),
        "reference_root_present": REFERENCE_ROOT.exists(),
    }


def nc_mcm_check(environment: dict[str, Any] | None = None) -> ConformanceCheck:
    env = environment or probe_environment()
    if env["julia_executable"]:
        detail = (
            "Julia is available, but the vendored NC-MCM code has not been executed in this "
            "Stage-One run; reproduction is therefore unverified."
        )
        status = "not_reproduced"
    else:
        detail = (
            "The official NC-MCM implementation is Julia and no Julia executable is present, "
            f"so its {env['nc_mcm_julia_sources']} source files cannot run here. "
            f"{env['nc_mcm_precomputed_models']} precomputed model archives exist and are "
            "loadable, but a precomputed embedding is an output, not the method: it cannot "
            "establish that a re-implementation is conformant."
        )
        status = "not_reproduced"
    return ConformanceCheck(
        name="nc-mcm",
        status=status,
        detail=detail,
        evidence=f"{NC_MCM_ROOT.as_posix()} (vendored, GPL-3.0, read-only)",
    )


def bundlenet_check(environment: dict[str, Any] | None = None) -> ConformanceCheck:
    env = environment or probe_environment()
    detail = (
        "BunDLe-Net v5 is a preprint with no executed reference here; the identifier was "
        "verified at Crossref, but no conformance run is claimed."
        if not env["bundlenet_root_present"]
        else "BunDLe-Net is vendored, but no conformance run was performed in this branch."
    )
    return ConformanceCheck(
        name="bundlenet-v5",
        status="not_reproduced",
        detail=detail,
        evidence="DOI 10.1101/2023.08.08.551978 (preprint, v5)",
    )


def oracle_check(agreeing_fixtures: int) -> ConformanceCheck:
    return ConformanceCheck(
        name="enumerating-reference",
        status="simulated" if agreeing_fixtures else "falsified",
        detail=(
            "The oracle is an exact enumeration over synthetic kernels. It is ground truth for "
            f"the benchmark, not evidence about any biological system; it agreed with the "
            f"independent sampling estimator on {agreeing_fixtures} fixtures."
        ),
        evidence="neurobranch.abstraction.reference.exact_commutation",
    )


def witness_check(plateau: bool) -> ConformanceCheck:
    return ConformanceCheck(
        name="identifiability-witness",
        status="unidentifiable" if plateau else "falsified",
        detail=(
            "Under the witness operator the error does not shrink with the observation window, "
            "because the operator's effect depends on a context that never enters the "
            "observation. The correct report is not_identified, not a small error."
            if plateau
            else "The witness error shrank with the window, so the fixture does not witness "
            "non-identifiability as constructed and must be fixed before use."
        ),
        evidence="ring_witness/L{1,2,4}, operator asym",
    )


def conformance_checks(
    *,
    agreeing_fixtures: int = 0,
    witness_plateau: bool = False,
    environment: dict[str, Any] | None = None,
) -> list[ConformanceCheck]:
    env = environment or probe_environment()
    return [
        nc_mcm_check(env),
        bundlenet_check(env),
        oracle_check(agreeing_fixtures),
        witness_check(witness_plateau),
    ]


def conformance_markdown(checks: list[ConformanceCheck], environment: dict[str, Any]) -> str:
    lines = [
        "# Conformance report",
        "",
        "Generated by `neurobranch abstraction stage1`. No third-party implementation was",
        "executed for this report; every claim below is about what ran here.",
        "",
        "| reference | status | evidence |",
        "|---|---|---|",
    ]
    for check in checks:
        lines.append(f"| {check.name} | `{check.status}` | {check.evidence} |")
    lines += ["", "## Detail", ""]
    for check in checks:
        lines += [f"### {check.name} — `{check.status}`", "", check.detail, ""]
    lines += [
        "## Environment probe",
        "",
        "| fact | value |",
        "|---|---|",
    ]
    for key, value in environment.items():
        lines.append(f"| {key} | `{value}` |")
    lines += [
        "",
        "## What is deliberately not claimed",
        "",
        "- That NC-MCM or BunDLe-Net were reproduced. Neither implementation ran.",
        "- That any result here says anything about *C. elegans*.",
        "- That a synthetic negative control falsifies a published method.",
        "",
    ]
    return "\n".join(lines)


def write_conformance(path: Path, checks: list[ConformanceCheck], environment: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(conformance_markdown(checks, environment), encoding="utf-8")
