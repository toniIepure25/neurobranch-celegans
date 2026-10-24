"""Model-dependent controls for the Phase 4 turn-direction benchmark.

These controls test whether the apparent neural increment survives behavioural
time shifts, identity permutation, label permutation and matched feature
ablation.  None of them is an experimental perturbation.
"""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from neurobranch.behavior.turn_direction import (
    build_designs,
    circuit_ablation,
    evaluate_loo,
)
from neurobranch.data.kramer import load_all

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "phase4"


def main() -> None:
    recs, _ = load_all(cohorts=("wt_octanol", "wt_diacetyl"))
    designs, labels = build_designs(recs, window="mid")
    primary_preds = pd.read_parquet(REPORT / "turn_direction_predictions.parquet")
    base = primary_preds[primary_preds["mode"] == "behavior"][["event_id", "y"]]
    rows = []

    # 1. Behaviour-preserving neural time shift.  The neural segment is moved
    # 50 frames earlier; behaviour and labels stay unchanged.
    shifted, _ = build_designs(recs, window="mid", neural_shift_frames=-50)
    sm, _, ss = evaluate_loo(
        shifted,
        labels,
        modes=("behavior_population", "behavior_saav", "behavior_circuit"),
        seed=17,
    )
    for mode in ("behavior_population", "behavior_saav", "behavior_circuit"):
        sp = ss["predictions"]
        sp = sp[sp["mode"] == mode].merge(base, on="event_id", suffixes=("", "_base"))
        pp = primary_preds[primary_preds["mode"] == mode].merge(base, on="event_id", suffixes=("", "_base"))
        common = set(sp["event_id"]) & set(pp["event_id"])
        sp = sp[sp["event_id"].isin(common)]
        pp = pp[pp["event_id"].isin(common)]
        rows.append(
            {
                "control": "neural_shift_-50_frames",
                "mode": mode,
                "n_common_events": len(common),
                "auroc_original": float(roc_auc_score(pp["y"], pp["p"])),
                "auroc_control": float(roc_auc_score(sp["y"], sp["p"])),
                "delta_control_minus_original": float(roc_auc_score(sp["y"], sp["p"]) - roc_auc_score(pp["y"], pp["p"])),
            }
        )

    # 2. Identity permutation: shuffle the eight circuit feature columns within
    # each animal.  Marginal activity is preserved; the mapping to neuron
    # identity is destroyed.
    rng = np.random.default_rng(17)
    ident = []
    for d in designs:
        c = copy.deepcopy(d)
        perm = rng.permutation(c.circuit.shape[1])
        c.circuit = c.circuit[:, perm]
        ident.append(c)
    _, _, ids = evaluate_loo(ident, labels, modes=("behavior_circuit",), seed=17)
    full_primary = primary_preds[primary_preds["mode"] == "behavior_circuit"]
    rows.append(
        {
            "control": "circuit_identity_permutation",
            "mode": "behavior_circuit",
            "n_common_events": int(len(full_primary)),
            "auroc_original": float(roc_auc_score(full_primary["y"], full_primary["p"])),
            "auroc_control": float(ids["aggregate"]["behavior_circuit"]["auroc"]),
            "delta_control_minus_original": float(ids["aggregate"]["behavior_circuit"]["auroc"] - roc_auc_score(full_primary["y"], full_primary["p"])),
        }
    )
    pd.DataFrame(rows).to_csv(REPORT / "turn_direction_controls.csv", index=False)

    # 3. Within-animal label permutation.  Class counts per animal and all
    # behavioural features are preserved; the event-to-label relation is not.
    observed = float(
        primary_preds[primary_preds["mode"] == "behavior_population"]["p"].pipe(
            lambda p: roc_auc_score(
                primary_preds[primary_preds["mode"] == "behavior_population"]["y"], p
            )
        )
    )
    observed_behavior = float(
        roc_auc_score(
            primary_preds[primary_preds["mode"] == "behavior"]["y"],
            primary_preds[primary_preds["mode"] == "behavior"]["p"],
        )
    )
    null_rows = []
    for i in range(6):
        perm = []
        for d in designs:
            c = copy.deepcopy(d)
            c.y = rng.permutation(c.y)
            perm.append(c)
        _, _, s = evaluate_loo(perm, labels, modes=("behavior", "behavior_population"), seed=17)
        d_pop = s["aggregate"]["behavior_population"]["auroc"] - s["aggregate"]["behavior"]["auroc"]
        null_rows.append({"permutation": i, "delta_population_vs_behavior": d_pop})
    null = pd.DataFrame(null_rows)
    null.to_csv(REPORT / "turn_direction_label_permutation.csv", index=False)
    observed_delta = observed - observed_behavior
    empirical_p = float(np.mean(null["delta_population_vs_behavior"] >= observed_delta))
    (REPORT / "turn_direction_label_permutation_summary.json").write_text(
        json.dumps(
            {
                "n_permutations": len(null),
                "observed_delta": observed_delta,
                "null_mean": float(null["delta_population_vs_behavior"].mean()),
                "null_sd": float(null["delta_population_vs_behavior"].std()),
                "empirical_p_one_sided": empirical_p,
                "note": "One-sided p for the observed positive increment under within-animal label permutation.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # 4. Circuit feature ablation with matched random controls.
    ablation, ablation_summary = circuit_ablation(designs, labels, seed=17, n_random=6)
    ablation.to_csv(REPORT / "turn_direction_neuron_ablation_raw.csv", index=False)
    ablation_summary.to_csv(REPORT / "turn_direction_neuron_ablation.csv", index=False)

    # 5. Event overlap check: reversal events should not overlap.
    overlap = 0
    for d in designs:
        intervals = d.events[["start", "end"]].to_numpy()
        if len(intervals) > 1:
            overlap += int(np.sum(intervals[1:, 0] < intervals[:-1, 1]))
    (REPORT / "turn_direction_overlap.json").write_text(
        json.dumps({"overlapping_adjacent_events": overlap, "events": int(sum(len(d.y) for d in designs))}, indent=2),
        encoding="utf-8",
    )
    print("wrote controls", REPORT)


if __name__ == "__main__":
    main()
