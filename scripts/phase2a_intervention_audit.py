"""Run the Phase 2 intervention-specificity audit on persisted MLP/VAR/GRU models.

This is intentionally not a hyperparameter search.  It asks whether the Phase 1
RIVL result survives model architecture, model seed, amplitude, and a matched
non-candidate control set.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from neurobranch.config import ExperimentConfig
from neurobranch.dynamics.checkpointing import load_model
from neurobranch.dynamics.training import load_worm_data
from neurobranch.interventions.audit import (
    candidate_specificity,
    rank_stability,
    selection_adjusted_max_test,
)
from neurobranch.interventions.engine import run_counterfactual
from neurobranch.interventions.search import random_control, run_sparse_search
from neurobranch.interventions.spec import InterventionSpec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "phase2a_scientific_audit" / "intervention"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "mlp_seed17": "artifacts/runs/blocked_suite/mlp__worm0__seed17/model.pt",
    "mlp_seed23": "artifacts/runs/blocked_suite/mlp__worm0__seed23/model.pt",
    "mlp_seed31": "artifacts/runs/blocked_suite/mlp__worm0__seed31/model.pt",
    "mlp_residual_seed17": "artifacts/runs/blocked_suite/mlp_residual__worm0__seed17/model.pt",
    "var_seed17": "artifacts/runs/blocked_suite/var__worm0__seed17/model.pt",
    "gru_seed17": "artifacts/runs/blocked_suite/gru__worm0__seed17/model.pt",
}


def main() -> None:
    cfg = ExperimentConfig.load(ROOT / "configs" / "experiments" / "blocked_suite.yaml")
    data = load_worm_data(cfg)[0]
    scans: dict[str, pd.DataFrame] = {}
    controls: dict[str, pd.DataFrame] = {}
    for name, rel in MODELS.items():
        path = ROOT / rel
        if not path.exists():
            print(f"skip missing {name}: {path}")
            continue
        model, _ = load_model(path)
        scan, _ = run_sparse_search(
            model,
            data,
            seed=17,
            amplitudes=(-2.0, 2.0),
            n_initial=30,
            sim_seeds=(0, 1, 2),
            horizon=12,
            max_candidates=20,
        )
        ctrl = random_control(
            model,
            data,
            n_random=20,
            amplitude=2.0,
            seed=17,
            n_initial=30,
            sim_seeds=(0, 1, 2),
            horizon=12,
        )
        scan.to_csv(OUT / f"scan_{name}.csv", index=False)
        ctrl.to_csv(OUT / f"controls_{name}.csv", index=False)
        scans[name] = scan
        controls[name] = ctrl
        print(name, "scan", len(scan), "controls", len(ctrl), flush=True)

    specificity = {
        name: candidate_specificity(scan, controls[name]).to_dict(orient="records")
        for name, scan in scans.items()
    }
    (OUT / "specificity_by_model.json").write_text(json.dumps(specificity, indent=2), encoding="utf-8")

    max_tests = {name: selection_adjusted_max_test(scan, controls[name]) for name, scan in scans.items()}
    (OUT / "selection_adjusted_max_tests.json").write_text(
        json.dumps(max_tests, indent=2), encoding="utf-8"
    )

    stability = rank_stability(scans, amplitude=2.0, top_k=5)
    (OUT / "rank_stability.json").write_text(json.dumps(stability, indent=2), encoding="utf-8")

    # Amplitude sensitivity on a small, fixed set: the top Phase 1 cell, its
    # bilateral partner, the strongest random-control cell, and two controls.
    model, _ = load_model(ROOT / MODELS["mlp_seed17"])
    cells = ["RIVL", "RIVR", "OLQVL", "OLQVR", "RIFR", "ASKR"]
    amps = (-3.0, -2.0, -1.0, 0.5, 1.0, 2.0, 3.0)
    amp_rows = []
    for cell in cells:
        if cell not in data.predictor_names:
            continue
        for amp in amps:
            spec = InterventionSpec(
                name=f"{cell}_{amp:+.1f}",
                targets=(cell,),
                mode="additive",
                amplitude=amp,
                onset=0,
                duration=12,
                horizon=12,
                n_initial=30,
                seeds=(0, 1, 2),
                initial_behavior="REV",
                target_behavior="TURN",
            )
            r = run_counterfactual(model, data, spec, seed=17)
            amp_rows.append(
                {
                    "cell": cell,
                    "amplitude": amp,
                    "delta": r.delta["TURN"],
                    "ci_low": r.delta_ci[0],
                    "ci_high": r.delta_ci[1],
                    "ood_fraction": r.ood_fraction,
                    "mean_displacement": r.mean_trajectory_displacement,
                }
            )
            print("amp", cell, amp, round(r.delta["TURN"], 3), flush=True)
    amp_df = pd.DataFrame(amp_rows)
    amp_df.to_csv(OUT / "amplitude_sensitivity.csv", index=False)

    summary = {
        "candidate_cells": sorted(scans[next(iter(scans))]["candidate"].unique().tolist()) if scans else [],
        "models_run": list(scans),
        "n_candidate_cells": int(len(scans[next(iter(scans))])) if scans else 0,
        "n_controls_per_model": int(len(controls[next(iter(controls))])) if controls else 0,
        "match_criteria": [
            "same standardized additive amplitude",
            "same initial-state distribution and seed",
            "same simulation seeds",
            "same horizon and target behaviour",
            "controls are non-candidate named predictor cells",
        ],
        "notes": [
            "Matching on observed effect would be circular and is not done.",
            "Candidate list is literature-motivated, not selected from this worm's effects.",
            "All effects are model-dependent virtual perturbations, not biological interventions.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
