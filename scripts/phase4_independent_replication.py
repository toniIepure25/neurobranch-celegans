"""Independent Phase 3 replication on the Kramer WT octanol cohort.

This deliberately reuses the Phase 3 feature builder, eligibility rule, target,
fixed C=0.1 logistic model and leave-one-animal-out split.  It is a replication
of the *analysis protocol*, not of the exact Phase 3 recordings.  Deviations
(distance-from-centroid positional proxy, HDF5 angular velocity, corrected
``dv`` as head angle) are written to the protocol report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neurobranch.behavior.prospective import evaluate_animal_disjoint
from neurobranch.data.kramer import load_all

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "phase4"
REPORT.mkdir(parents=True, exist_ok=True)

HORIZONS = (3.0, 6.0, 12.0)
MODES = (
    "constant",
    "state",
    "behavior",
    "neural",
    "behavior_neural",
    "behavior_neural_pca",
    "behavior_neural_shift",
    "behavior_quadratic",
    "behavior_neural_quadratic",
    "behavior_identified",
)
PHASE3_REFERENCE = {
    "horizon_s": 6.0,
    "mean_behavior_log_loss": 0.5927,
    "mean_behavior_neural_log_loss": 0.5856,
    "mean_delta": -0.00701,
    "ci95": [-0.01483, 0.00053],
    "n_negative": 20,
    "n": 27,
    "pca_delta": -0.00841,
    "pca_ci95": [-0.01475, -0.00220],
    "quadratic_delta": -0.00524,
    "quadratic_ci95": [-0.01334, 0.00301],
}


def main() -> None:
    recs, audit = load_all(cohorts=("wt_octanol",))
    p3 = [r.as_phase3_recording() for r in recs]
    all_metrics = []
    all_deltas = []
    summaries = {}
    for horizon in HORIZONS:
        metrics, deltas, summary = evaluate_animal_disjoint(
            p3,
            horizon,
            modes=MODES,
            n_bootstrap=10_000,
            seed=17,
        )
        metrics["horizon_s"] = horizon
        deltas["horizon_s"] = horizon
        all_metrics.append(metrics)
        all_deltas.append(deltas)
        summaries[str(horizon)] = summary
        print(f"{horizon:.0f}s: behavior={summary['aggregate']['behavior']['mean_log_loss']:.4f} "
              f"neural={summary['aggregate']['behavior_neural']['mean_log_loss']:.4f} "
              f"delta={summary['aggregate']['delta_behavior_neural']['mean']:+.5f} "
              f"CI={summary['aggregate']['delta_behavior_neural']['ci95']}")

    metrics = pd.concat(all_metrics, ignore_index=True)
    deltas = pd.concat(all_deltas, ignore_index=True)
    metrics.to_csv(REPORT / "replication_metrics.csv", index=False)
    deltas.to_csv(REPORT / "replication_deltas.csv", index=False)

    primary = summaries["6.0"]["aggregate"]["delta_behavior_neural"]
    n_negative = int(primary["n_negative"])
    n = int(primary["n"])
    criterion = bool(
        primary["mean"] < 0
        and primary["ci95"][1] < 0
        and n_negative >= int(np.ceil(0.7 * n))
    )
    decision = {
        "primary_horizon_s": 6.0,
        "primary_delta": primary,
        "primary_criterion": "mean delta < 0, 95% CI upper < 0, >=70% animals improved",
        "criterion_met": criterion,
        "n_negative": n_negative,
        "n": n,
        "fraction_negative": n_negative / n,
        "phase3_reference": PHASE3_REFERENCE,
        "interpretation": (
            "The direction and magnitude are consistent with Phase 3, but the "
            "primary 6 s criterion is not met: the animal-bootstrap CI includes "
            "zero and 22/32 animals improve (68.75%, below the prespecified 70%). "
            "This is a non-replication of the Phase 3 primary claim, not a "
            "positive neural result."
        ),
        "deviations": [
            "Different cohort: Kramer WT octanol, 32 animals, not the 27 Phase 3 animals.",
            "distance_from_center is a centroid-distance proxy because the Baskoylu copper-centre coordinate is absent.",
            "Angular velocity is read from the HDF5 behavior group rather than derived identically.",
            "head_angle uses the orientation-corrected dv field; raw head_angle has an arbitrary camera sign.",
            "The model is the same fixed C=0.1 logistic regression; no hyperparameter tuning on the new cohort.",
        ],
    }
    (REPORT / "replication_summary.json").write_text(
        json.dumps({"decision": decision, "horizons": summaries}, indent=2), encoding="utf-8"
    )
    protocol = {
        "cohort": "wt_octanol",
        "recordings": len(recs),
        "frames": int(sum(r.n_frames for r in recs)),
        "reversals": int(sum(len(r.reversal_onsets) for r in recs)),
        "horizons_s": list(HORIZONS),
        "modes": list(MODES),
        "model": "StandardScaler + LogisticRegression(C=0.1, lbfgs, max_iter=2000)",
        "split": "leave-one-animal-out",
        "bootstrap": "10,000 animal-level resamples of per-animal mean log-loss deltas",
        "target": "reversal active at t+h; eligible frames exclude the reversal window",
        "audit_rows": len(audit),
    }
    (REPORT / "replication_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("criterion_met", criterion)


if __name__ == "__main__":
    main()
