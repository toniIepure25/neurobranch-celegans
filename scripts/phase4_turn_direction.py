"""Primary Phase 4 turn-direction benchmark and secondary sensitivity analyses.

Primary question: does neural activity before the reversal offset add
out-of-animal prospective information about the post-offset dorsal/ventral turn
beyond a strong behavioural/context baseline?

The primary representation is the four-frame middle window, ending at least two
frames before the offset.  The label is the mean ``dv`` over the first two
post-offset frames.  All evaluation is leave-one-animal-out; preprocessing and
identified-label selection are fit on training animals only.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from neurobranch.behavior.turn_direction import (
    MODES,
    PRIMARY_WINDOW,
    build_designs,
    evaluate_external,
    evaluate_loo,
    evaluate_within_animal,
)
from neurobranch.data.kramer import load_all

ROOT = Path(__file__).resolve().parents[1]
warnings.filterwarnings("ignore", category=UserWarning)
REPORT = ROOT / "reports" / "phase4"
REPORT.mkdir(parents=True, exist_ok=True)

SECONDARY_MODES = ("behavior", "behavior_population", "behavior_saav", "behavior_circuit")
ALL_EXTERNAL_MODES = ("behavior", "behavior_population", "behavior_saav", "behavior_circuit")


def _clean_summary(summary: dict) -> dict:
    return {k: v for k, v in summary.items() if k != "predictions"}


def calibration_table(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, group in preds.groupby("mode"):
        if group["y"].nunique() < 2:
            continue
        try:
            frac_pos, mean_pred = calibration_curve(
                group["y"], group["p"], n_bins=5, strategy="quantile"
            )
        except ValueError:
            continue
        for i, (f, m) in enumerate(zip(frac_pos, mean_pred)):
            rows.append({"mode": mode, "bin": i, "mean_predicted": float(m), "fraction_positive": float(f)})
    return pd.DataFrame(rows)


def decision(metrics: pd.DataFrame, summary: dict) -> dict:
    base = metrics.set_index("mode")["auroc"].to_dict()
    delta = summary["deltas"].get("delta_behavior_population_vs_behavior", {})
    out = {
        "primary_contrast": "behavior_population vs behavior",
        "auroc": base,
        "delta_population": delta,
        "classification": "no_incremental_evidence",
        "reason": "The 95% animal-bootstrap CI for the AUROC increment includes zero.",
    }
    ci = delta.get("ci95", [np.nan, np.nan])
    if np.isfinite(ci[1]) and ci[1] < 0:
        out["classification"] = "negative_increment"
        out["reason"] = "The 95% CI is entirely below zero; neural features reduced held-out AUROC."
    elif np.isfinite(ci[0]) and ci[0] > 0:
        out["classification"] = "positive_increment"
        out["reason"] = "The 95% CI is entirely above zero."
    out["secondary"] = {
        mode: summary["deltas"].get(f"delta_{mode}_vs_behavior")
        for mode in ("behavior_saav", "behavior_circuit", "behavior_identified")
    }
    return out


def main() -> None:
    recs, audit = load_all(cohorts=("wt_octanol", "wt_diacetyl"))
    primary_designs, labels = build_designs(recs, window=PRIMARY_WINDOW)
    print("primary events", sum(len(d.y) for d in primary_designs), "animals", len(primary_designs))

    metrics, per_animal, summary = evaluate_loo(primary_designs, labels, seed=17)
    preds = summary.pop("predictions")
    metrics.to_csv(REPORT / "turn_direction_metrics.csv", index=False)
    per_animal.to_csv(REPORT / "turn_direction_per_animal.csv", index=False)
    preds.to_parquet(REPORT / "turn_direction_predictions.parquet", index=False)
    calibration_table(preds).to_csv(REPORT / "turn_direction_calibration.csv", index=False)
    d = decision(metrics, summary)
    (REPORT / "turn_direction_summary.json").write_text(
        json.dumps(
            {
                "design": {
                    "cohorts": ["wt_octanol", "wt_diacetyl"],
                    "animals": len(primary_designs),
                    "events": int(len(preds[preds["mode"] == "behavior"])),
                    "dorsal": int(preds[preds["mode"] == "behavior"]["y"].sum()),
                    "ventral": int((1 - preds[preds["mode"] == "behavior"]["y"]).sum()),
                    "window": PRIMARY_WINDOW,
                    "segment_frames": 4,
                    "guard_frames_before_offset": 2,
                    "split": "leave-one-animal-out",
                    "model": "StandardScaler + LogisticRegression(C=0.1)",
                    "bootstrap": "2,000 animal-cluster resamples",
                    "labels": len(labels),
                },
                "decision": d,
                "summary": _clean_summary(summary),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))
    print(json.dumps(d, indent=2))

    # Secondary window sensitivity.
    secondary = []
    for window in ("early", "late"):
        designs, wlabels = build_designs(recs, window=window)
        m, _, s = evaluate_loo(designs, wlabels, modes=SECONDARY_MODES, seed=17)
        m["window"] = window
        secondary.append(m)
        (REPORT / f"turn_direction_summary_{window}.json").write_text(
            json.dumps({"window": window, "summary": _clean_summary(s)}, indent=2), encoding="utf-8"
        )
    pd.concat(secondary, ignore_index=True).to_csv(
        REPORT / "turn_direction_window_sensitivity.csv", index=False
    )

    # Cohort-specific animal-disjoint analyses.
    cohort_tables = []
    for cohort in ("wt_octanol", "wt_diacetyl"):
        designs = [d for d in primary_designs if d.cohort == cohort]
        m, _, s = evaluate_loo(designs, labels, modes=SECONDARY_MODES, seed=17)
        m["cohort"] = cohort
        cohort_tables.append(m)
        (REPORT / f"turn_direction_summary_{cohort}.json").write_text(
            json.dumps({"cohort": cohort, "summary": _clean_summary(s)}, indent=2), encoding="utf-8"
        )
    pd.concat(cohort_tables, ignore_index=True).to_csv(
        REPORT / "turn_direction_cohorts.csv", index=False
    )

    # SAAV-available subset: the label is present in only part of the archive.
    saav_idx = labels.index("SAAV") if "SAAV" in labels else None
    if saav_idx is not None:
        saav_designs = [d for d in primary_designs if d.label_present[:, saav_idx].any()]
        m, _, s = evaluate_loo(
            saav_designs,
            labels,
            modes=("behavior", "behavior_saav", "neural_saav"),
            seed=17,
        )
        m["subset"] = "SAAV_available"
        m.to_csv(REPORT / "turn_direction_saav_subset.csv", index=False)
        (REPORT / "turn_direction_summary_saav_subset.json").write_text(
            json.dumps(
                {
                    "subset": "SAAV_available",
                    "animals": len(saav_designs),
                    "events": int(sum(len(d.y) for d in saav_designs)),
                    "summary": _clean_summary(s),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # Within-animal five-fold event split for comparison.  It is not the primary
    # animal-disjoint result; it shows how much event-level dependence changes.
    wm, wp = evaluate_within_animal(
        primary_designs,
        labels,
        modes=("behavior", "behavior_population", "behavior_circuit"),
        seed=17,
    )
    wm.to_csv(REPORT / "turn_direction_within_animal_cv.csv", index=False)
    wp.to_parquet(REPORT / "turn_direction_within_animal_predictions.parquet", index=False)

    # External cohort shifts.  No tdc1 animal is used for fitting or selection.
    tdc_designs, _ = build_designs([r for r in load_all(cohorts=("tdc1_octanol",))[0]], window=PRIMARY_WINDOW)
    oct_designs = [d for d in primary_designs if d.cohort == "wt_octanol"]
    dia_designs = [d for d in primary_designs if d.cohort == "wt_diacetyl"]
    external = []
    for train_name, train_designs, test_name, test_designs in [
        ("wt_octanol", oct_designs, "wt_diacetyl", dia_designs),
        ("wt_all", primary_designs, "tdc1_octanol", tdc_designs),
        ("wt_octanol", oct_designs, "tdc1_octanol", tdc_designs),
    ]:
        m, p = evaluate_external(
            train_designs, test_designs, labels, modes=ALL_EXTERNAL_MODES, seed=17
        )
        if m.empty:
            continue
        m["train"] = train_name
        m["test"] = test_name
        external.append(m)
        p.to_parquet(REPORT / f"turn_direction_external_{test_name}_from_{train_name}.parquet", index=False)
    pd.concat(external, ignore_index=True).to_csv(
        REPORT / "turn_direction_external.csv", index=False
    )

    protocol = {
        "dataset_doi": "doi:10.5061/dryad.8sf7m0cz2",
        "recordings_total": len(recs),
        "audit_rows": len(audit),
        "primary_animals": len(primary_designs),
        "primary_events": int(sum(len(d.y) for d in primary_designs)),
        "primary_dorsal": int(sum(int(d.y.sum()) for d in primary_designs)),
        "primary_ventral": int(sum(int((1 - d.y).sum()) for d in primary_designs)),
        "min_reversal_frames": 12,
        "segment_frames": 4,
        "post_frames": 2,
        "guard_frames_before_offset": 2,
        "primary_window": PRIMARY_WINDOW,
        "modes": list(MODES),
        "no_post_segment_frames_in_features": True,
        "segment_selection": "event-aligned offline benchmark; segment location is defined relative to the completed reversal, as in the paper, but every feature frame precedes the two-frame guard.",
        "notes": [
            "The primary model is linear logistic regression with fixed C=0.1, not the paper's RNN.",
            "Reversal duration and time-to-offset are excluded from the behaviour features because they are future information.",
            "Circuit labels are prespecified; missing labels are zero after causal z-scoring.",
            "Identified-label vocabulary is fixed before scoring, but fold-level coverage is training-only.",
            "Ablations and permutation controls are in phase4_controls.py.",
        ],
    }
    (REPORT / "turn_direction_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
