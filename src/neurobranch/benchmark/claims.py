"""Machine-readable claims ledger with explicit status and evidence.

Every quantitative statement the project makes should appear here with:

* the exact estimate and interval,
* the test that produced the status,
* the smallest effect of interest (``margin``),
* the design resolution (``mde``),
* the files a reader can open to verify it.

Statuses are fixed by :mod:`neurobranch.benchmark.power`:

``supported`` / ``not_supported`` / ``bounded_null`` / ``underpowered``.

The ledger is deliberately conservative.  A pair-level gain of +0.044 with a
design MDE of +0.059 against a +0.020 smallest effect of interest is
``underpowered``, not ``supported``: the design could not resolve the effect
the project declared meaningful.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neurobranch.benchmark.power import (
    classify_bound,
    classify_superiority,
    mde_from_ci,
    tost,
)


@dataclass(frozen=True)
class Claim:
    """One auditable claim."""

    id: str
    statement: str
    status: str
    test: str
    estimate: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    margin: float | None = None
    mde: float | None = None
    equivalence_p: float | None = None
    evidence: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "status": self.status,
            "test": self.test,
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "margin": self.margin,
            "mde": self.mde,
            "equivalence_p": self.equivalence_p,
            "evidence": list(self.evidence),
            "notes": self.notes,
        }


def _superiority_claim(
    claim_id: str,
    statement: str,
    estimate: float,
    ci_low: float,
    ci_high: float,
    margin: float,
    evidence: tuple[str, ...],
    notes: str = "",
) -> Claim:
    mde = mde_from_ci(ci_low, ci_high)
    standard_error = (ci_high - ci_low) / (2.0 * 1.959963984540054)
    equivalence = tost(estimate, standard_error, margin)
    status = classify_superiority(estimate, ci_low, ci_high, margin=margin, mde=mde)
    return Claim(
        id=claim_id,
        statement=statement,
        status=status,
        test="superiority",
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        margin=margin,
        mde=mde,
        equivalence_p=equivalence.p_max,
        evidence=evidence,
        notes=notes,
    )


def build_claims(reliability_summary: dict[str, Any]) -> list[Claim]:
    """Build the Phase 9 ledger from the reliability summary and Phase 8 table."""
    animal_mean = float(reliability_summary["mean_animal_reliability"])
    animal_low = float(reliability_summary["animal_ci_low"])
    animal_high = float(reliability_summary["animal_ci_high"])
    repeat_mean = float(reliability_summary["mean_repeat_reliability"])
    repeat_low = float(reliability_summary["repeat_ci_low"])
    repeat_high = float(reliability_summary["repeat_ci_high"])

    claims = [
        Claim(
            id="phase9-single-animal-ceiling-below-half",
            statement="Across-animal single-profile reliability is below 0.5.",
            status=classify_bound(animal_mean, animal_low, animal_high, bound=0.5, direction="below"),
            test="upper_bound",
            estimate=animal_mean,
            ci_low=animal_low,
            ci_high=animal_high,
            margin=0.5,
            evidence=(
                "reports/phase9/reliability_summary.json",
                "reports/phase9/reliability_by_target.csv",
            ),
            notes="Bound claim; no equivalence test is implied.",
        ),
        Claim(
            id="phase9-repeat-ceiling-below-half",
            statement="Within-animal repeat-stimulation reliability is below 0.5.",
            status=classify_bound(repeat_mean, repeat_low, repeat_high, bound=0.5, direction="below"),
            test="upper_bound",
            estimate=repeat_mean,
            ci_low=repeat_low,
            ci_high=repeat_high,
            margin=0.5,
            evidence=(
                "reports/phase9/reliability_summary.json",
                "reports/phase9/reliability_by_target.csv",
            ),
            notes="Repeat facet uses at least two stimulations per animal.",
        ),
        _superiority_claim(
            claim_id="phase8-pair-advantage",
            statement="APT(k=0) improves pair-level Spearman over the static CRO baseline.",
            estimate=0.0442,
            ci_low=0.0080,
            ci_high=0.0901,
            margin=0.02,
            evidence=(
                "docs/PHASE8_RESULTS.md",
                "reports/phase8/",
            ),
            notes=(
                "Preregistered H1 pair gate passed, but the design MDE exceeds the "
                "0.02 smallest effect of interest."
            ),
        ),
        Claim(
            id="phase8-trajectory-advantage",
            statement="APT(k=0) improves trajectory correlation over a per-responder template.",
            status="not_supported",
            test="superiority",
            estimate=-0.0000066,
            evidence=("docs/PHASE8_RESULTS.md",),
            notes="Effect is numerically zero; no interval reported, so no MDE audit is possible.",
        ),
        Claim(
            id="phase8-sequential-rollout",
            statement="Three-event sequential rollout beats persistence on trajectory correlation.",
            status="not_supported",
            test="coverage_gate",
            estimate=0.0230,
            margin=0.95,
            evidence=("docs/PHASE8_RESULTS.md",),
            notes="Empirical coverage 0.9037 is below the 0.95 nominal level.",
        ),
        Claim(
            id="phase8-eig-selection",
            statement="EIG-based active selection beats random selection at k=8.",
            status=classify_superiority(-0.0038, -0.0070, 0.0026, margin=0.01, mde=mde_from_ci(-0.0070, 0.0026)),
            test="superiority",
            estimate=-0.0038,
            ci_low=-0.0070,
            ci_high=0.0026,
            margin=0.01,
            mde=mde_from_ci(-0.0070, 0.0026),
            evidence=("docs/PHASE8_RESULTS.md",),
            notes="Interval includes zero; point estimate is negative.",
        ),
        Claim(
            id="phase7-cro-share-of-ceiling",
            statement="CRO reaches 64% of the split-half reliability ceiling.",
            status="not_supported",
            test="superseded",
            evidence=("docs/PHASE7_RESULTS.md", "reports/phase9/reliability_summary.json"),
            notes=(
                "Superseded: the split-half ceiling of 0.152 is biased downward. The "
                "variance-corrected single-animal ceiling is 0.473, so 0.098 is about "
                "21% of the ceiling, not 64%."
            ),
        ),
    ]
    return claims


def write_claims(path: Path, claims: list[Claim]) -> None:
    payload = {
        "schema_version": 1,
        "statuses": ["supported", "not_supported", "bounded_null", "underpowered"],
        "claims": [claim.as_dict() for claim in claims],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["Claim", "build_claims", "write_claims"]
