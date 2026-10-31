"""Aggregate the frozen simulator gate statuses into a schema-validated report."""

from __future__ import annotations

import json
from typing import Any

from neurobranch.config import PROJECT_ROOT


def main() -> int:
    reports = PROJECT_ROOT / "reports" / "simulator"
    summary = json.loads((reports / "summary.json").read_text(encoding="utf-8"))
    branching = json.loads((reports / "branching_summary.json").read_text(encoding="utf-8"))["gate_3"]
    perturb = json.loads((reports / "perturbation_summary.json").read_text(encoding="utf-8"))["gate_5"]
    selection = json.loads((reports / "selection.json").read_text(encoding="utf-8"))

    g1 = summary["gate_1"]
    g2 = summary["gate_2"]
    gates: dict[str, Any] = {
        "G1": {
            "status": g1["status"],
            "criterion": g1["criterion"],
            "evidence_paths": ["reports/simulator/summary.json", "reports/simulator/selected_model_test.csv"],
            "notes": "Free-running rollouts are finite and start from the observed initial state; no future teacher forcing.",
        },
        "G2": {
            "status": g2["status"],
            "criterion": g2["criterion"],
            "evidence_paths": ["reports/simulator/summary.json", "reports/simulator/rollout_metrics.csv"],
            "notes": (
                "Selected model is chosen on validation one-step R2. It fails the free-running "
                "dynamical-advantage gate in all five worms; the best descriptive model (VAR) beats "
                "the constant-state baseline but was not selected by that criterion."
            ),
        },
        "G3": {
            "status": branching["status"],
            "criterion": branching["criterion"],
            "evidence_paths": [
                "reports/simulator/branching_summary.json",
                "reports/simulator/branching_events.csv",
            ],
            "notes": (
                "Simulated outcome distributions are not closer to observed reversal-conditioned "
                "distributions than persistence or the outcome-marginal null. Several block-null "
                "tests show an observed entropy DEFICIT, not excess. No bifurcation is claimed."
            ),
        },
        "G4": {
            "status": "pass",
            "criterion": (
                "A named predictor neuron can be selected, perturbed under an explicit protocol, "
                "and its model-dependent trajectory/outcome effects reported with controls."
            ),
            "evidence_paths": [
                "tests/simulator/test_core.py",
                "tests/simulator/test_perturbation.py",
                "reports/simulator/perturbation_specificity.csv",
            ],
            "notes": "Functional interface only; passing does not validate a biological intervention.",
        },
        "G5": {
            "status": perturb["status"],
            "criterion": perturb["criterion"],
            "evidence_paths": [
                "reports/simulator/perturbation_summary.json",
                "reports/simulator/perturbation_specificity.csv",
                "reports/simulator/perturbation_controls.csv",
            ],
            "notes": (
                "No candidate exceeds the random-neuron 95th percentile. Top TV effects are not "
                "robust across worms or model seeds, and >98% of perturbed states are outside "
                "training support. Initial-state probes move trajectories without changing outcomes."
            ),
        },
        "G6": {
            "status": "not_evaluable",
            "criterion": (
                "A matched empirical stimulation-response dataset and a held-out causal effect "
                "must be available."
            ),
            "evidence_paths": ["docs/QUESTIONS_FOR_DENIS.md", "docs/SIMULATOR_LIMITATIONS.md"],
            "notes": (
                "Kato 2015 is observational. No optogenetic/histamine response model is registered. "
                "Category C empirical_response is therefore refused by the code."
            ),
        },
    }
    payload = {
        "schema_version": 1,
        "generated_by": "scripts/simulator_gates.py",
        "decision": "functional_simulator_no_dynamical_or_branching_advantage",
        "warning": (
            "MODEL-DEPENDENT VIRTUAL PERTURBATION - NOT CAUSALLY VALIDATED. "
            "No result establishes a bifurcation, a causal intervention, whole-brain dynamics, or "
            "a biological validation."
        ),
        "gates": gates,
        "source_reports": {
            "selection": str((reports / "selection.json").relative_to(PROJECT_ROOT)),
            "selected_model": str(selection.get("selected_kind")),
            "summary": "reports/simulator/summary.json",
            "branching": "reports/simulator/branching_summary.json",
            "perturbation": "reports/simulator/perturbation_summary.json",
        },
    }
    (reports / "gates.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote reports/simulator/gates.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
