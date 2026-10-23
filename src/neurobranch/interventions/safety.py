"""Evidence gates for empirical intervention claims.

Observational data cannot license claims about novel interventions.  This
module therefore refuses to expose an "empirically validated" mode unless a
manifest supplies actual intervention provenance, matched controls, and enough
independent animals.  A configuration flag is never sufficient.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neurobranch.config import sha256_file


@dataclass(frozen=True)
class EmpiricalEvidence:
    dataset_path: str
    dataset_sha256: str
    intervention: str
    target_neurons: tuple[str, ...]
    n_intervention_animals: int
    n_control_animals: int
    provenance_url: str
    performed_in_living_animal: bool
    negative_control: bool
    preregistered_analysis: bool
    notes: str = ""

    def validate(self) -> list[str]:
        errors = []
        p = Path(self.dataset_path)
        if not p.exists():
            errors.append(f"Intervention dataset does not exist: {p}")
        elif sha256_file(p) != self.dataset_sha256:
            errors.append("Intervention dataset SHA256 does not match the manifest")
        if not self.intervention:
            errors.append("Intervention description is empty")
        if not self.target_neurons:
            errors.append("No target neurons recorded")
        if self.n_intervention_animals < 3:
            errors.append("Fewer than 3 independent intervention animals")
        if self.n_control_animals < 3:
            errors.append("Fewer than 3 independent matched control animals")
        if not self.provenance_url.startswith(("http://", "https://", "doi:")):
            errors.append("Missing verifiable provenance URL/DOI")
        if not self.performed_in_living_animal:
            errors.append("Evidence does not document an in vivo intervention")
        if not self.negative_control:
            errors.append("No negative/sham control documented")
        if not self.preregistered_analysis:
            errors.append("Analysis protocol was not preregistered or frozen before evaluation")
        return errors


@dataclass(frozen=True)
class CausalStatus:
    level: str
    validated: bool
    label: str
    reasons: tuple[str, ...]


OBSERVATIONAL_STATUS = CausalStatus(
    level="observational_model",
    validated=False,
    label="MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.",
    reasons=(
        "The dynamics model was fitted only to observational recordings.",
        "No randomized or targeted perturbation of these neurons was performed.",
        "Latent state, model class, and noise assumptions are not identified from data.",
    ),
)


def load_empirical_evidence(path: str | Path) -> EmpiricalEvidence:
    with open(path, encoding="utf-8") as fh:
        payload: dict[str, Any] = json.load(fh)
    payload["target_neurons"] = tuple(payload.get("target_neurons", ()))
    known = set(EmpiricalEvidence.__dataclass_fields__)
    return EmpiricalEvidence(**{k: v for k, v in payload.items() if k in known})


def require_empirical_validation(path: str | Path) -> CausalStatus:
    evidence = load_empirical_evidence(path)
    errors = evidence.validate()
    if errors:
        return CausalStatus(
            level="empirical_rejected",
            validated=False,
            label="EMPIRICAL INTERVENTION MODE REJECTED - EVIDENCE GATE FAILED",
            reasons=tuple(errors),
        )
    return CausalStatus(
        level="empirical_intervention",
        validated=True,
        label="EMPIRICAL INTERVENTION EVIDENCE PASSED PROVENANCE GATE",
        reasons=("Validation is limited to the recorded intervention, animals, and assay.",),
    )
