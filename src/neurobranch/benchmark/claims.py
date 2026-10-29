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

import numpy as np
from scipy import stats

from neurobranch.benchmark.power import (
    classify_bound,
    classify_superiority,
    mde_from_ci,
    tost,
)


@dataclass
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
    estimand: str | None = None
    n_eval_animals: int | None = None
    family: str | None = None
    p_value: float | None = None
    p_holm: float | None = None
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
            "estimand": self.estimand,
            "n_eval_animals": self.n_eval_animals,
            "family": self.family,
            "p_value": self.p_value,
            "p_holm": self.p_holm,
            "evidence": list(self.evidence),
            "notes": self.notes,
        }


def _normal_p_value(estimate: float, ci_low: float, ci_high: float) -> float | None:
    if not (ci_low < ci_high):
        return None
    standard_error = (ci_high - ci_low) / (2.0 * 1.959963984540054)
    if standard_error <= 0:
        return None
    z = abs(estimate) / standard_error
    return float(2.0 * (1.0 - stats.norm.cdf(z)))


def _holm(p_values: list[float | None]) -> list[float | None]:
    """Holm step-down adjustment; ``None`` entries pass through."""
    indexed = [(index, value) for index, value in enumerate(p_values) if value is not None]
    if not indexed:
        return list(p_values)
    ordered = sorted(indexed, key=lambda item: item[1])
    adjusted: list[float | None] = list(p_values)
    running = 0.0
    total = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        candidate = min(1.0, (total - rank) * value)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


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
    """Build the schema-v2 ledger from a Phase 9 or Phase 10 summary.

    Phase 9 summaries are accepted for backward compatibility; Phase 10 adds
    the fold-level pooled ceiling and the estimand labels.
    """
    animal_mean = float(reliability_summary["mean_animal_reliability"])
    animal_low = float(reliability_summary["animal_ci_low"])
    animal_high = float(reliability_summary["animal_ci_high"])
    repeat_mean = float(reliability_summary["mean_repeat_reliability"])
    repeat_low = float(reliability_summary["repeat_ci_low"])
    repeat_high = float(reliability_summary["repeat_ci_high"])
    pooled_ceiling = float(reliability_summary.get("pooled_ceiling_at_fold", float("nan")))
    fold_ceiling = float(reliability_summary.get("pooled_ceiling_at_fold", float("nan")))
    single_ceiling = float(reliability_summary.get("mean_animal_ceiling", float("nan")))
    coverage_ceiling = 0.5600069018705827
    fold_size = int(reliability_summary.get("fold_size", 22))

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
            estimand="single_animal",
            n_eval_animals=1,
            evidence=(
                "reports/phase10/reliability_summary.json",
                "reports/phase10/profile_components.csv",
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
            estimand="repeat",
            n_eval_animals=None,
            evidence=(
                "reports/phase10/reliability_summary.json",
                "reports/phase10/reliability_by_target.csv",
            ),
            notes="Repeat facet uses at least two stimulations per animal.",
        ),
        Claim(
            id="phase10-fold-ceiling",
            statement=(
                f"Under the Phase 10 sparse-coverage estimator the pooled reliability ceiling "
                f"at the Phase 6-8 fold size is about {pooled_ceiling:.2f}; the coverage-aware "
                f"Phase 11 value is 0.56."
            ),
            status="not_supported",
            test="descriptive",
            estimate=pooled_ceiling if np.isfinite(pooled_ceiling) else None,
            margin=None,
            estimand="pooled_fold",
            n_eval_animals=fold_size,
            evidence=(
                "reports/phase10/reliability_summary.json",
                "reports/phase10/ceiling_curve.csv",
            ),
            notes=(
                "Coverage-sensitive: the Phase 10 estimator includes responders seen in one "
                "animal. Requiring 80% responder coverage lowers the pooled ceiling from "
                "0.833 to 0.560 (correction C8). The value is descriptive arithmetic, not a "
                "sampling claim, and is retained only as a historical artifact."
            ),
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
                "0.02 smallest effect of interest and the Holm-adjusted p-value across "
                "the Phase 8 family is 0.070. Estimand is the pooled fold profile."
            ),
        ),
        Claim(
            id="phase8-trajectory-advantage",
            statement="APT(k=0) improves trajectory correlation over a per-responder template.",
            status="not_supported",
            test="superiority",
            estimate=-0.0000066,
            estimand="superiority",
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
            estimand="superiority",
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
            estimand="superiority",
            evidence=("docs/PHASE8_RESULTS.md",),
            notes="Interval includes zero; point estimate is negative.",
        ),
        Claim(
            id="phase7-cro-share-of-ceiling",
            statement="CRO reaches 64% of the split-half reliability ceiling.",
            status="not_supported",
            test="superseded",
            estimand="descriptive",
            evidence=("docs/PHASE7_RESULTS.md", "reports/phase10/reliability_summary.json"),
            notes=(
                "Superseded: the split-half ceiling of 0.152 is biased downward. Against "
                f"the Phase 10 fold ceiling ({fold_ceiling:.3f}) the CRO of 0.098 is about "
                f"{100 * 0.0984 / fold_ceiling:.1f}%; against the coverage-aware ceiling "
                f"({coverage_ceiling:.3f}) it is about "
                f"{100 * 0.0984 / coverage_ceiling:.1f}%; against the single-animal ceiling "
                f"({single_ceiling:.3f}) it is about {100 * 0.0984 / single_ceiling:.1f}%. "
                f"Neither 64% nor the Phase 10 share is the current bound."
            ),
        ),
    ]

    for claim in claims:
        if claim.test == "superiority" and claim.p_value is None and claim.ci_low is not None and claim.ci_high is not None:
            claim.p_value = _normal_p_value(claim.estimate or 0.0, claim.ci_low, claim.ci_high)
    family = [claim for claim in claims if claim.family is None and claim.test == "superiority" and claim.p_value is not None]
    for claim in family:
        claim.family = "phase8-inferential"
    adjusted = _holm([claim.p_value for claim in family])
    for claim, p_holm in zip(family, adjusted):
        claim.p_holm = p_holm
    return claims


def write_claims(path: Path, claims: list[Claim]) -> None:
    payload = {
        "schema_version": 2,
        "statuses": ["supported", "not_supported", "bounded_null", "underpowered"],
        "claims": [claim.as_dict() for claim in claims],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["Claim", "build_claims", "write_claims"]
