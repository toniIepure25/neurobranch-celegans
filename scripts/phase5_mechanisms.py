"""Phase 5D: SAAV and compact-circuit mechanism analyses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neurobranch.behavior.mechanisms import (
    holm_adjust,
    incremental_channel_tests,
    stability_by_channel,
)
from neurobranch.behavior.saav_alignment import CIRCUIT_LABELS, build_aligned_dataset
from neurobranch.data.kramer import load_all

OUT = Path("reports/phase5/mechanisms")
TRUE_CHANNELS = ("saav", *[f"neuron:{name}" for name in CIRCUIT_LABELS if name != "SAAV"])
SELECTED = ("saav", "neuron:SMDV", "neuron:SMDD", "neuron:SMBV", "neuron:SMBD")


def _plot_incremental(frame: pd.DataFrame) -> None:
    data = frame[frame["metric"] == "log_loss"].sort_values("delta")
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ypos = np.arange(len(data))
    colors = ["#1f6f8b" if row["ci_high"] < 0 else "#9aa7b2" for _, row in data.iterrows()]
    ax.errorbar(
        data["delta"],
        ypos,
        xerr=[data["delta"] - data["ci_low"], data["ci_high"] - data["delta"]],
        fmt="o",
        color="#1f6f8b",
        ecolor="#7f8c8d",
        capsize=3,
    )
    ax.scatter(data["delta"], ypos, c=colors, zorder=3, s=30)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(data["channel"].str.replace("neuron:", "", regex=False), fontsize=8)
    ax.set_xlabel("Δ held-out event log loss (+ channel − behavior); negative is better")
    ax.set_title("Incremental information over head curvature\n(animal-disjoint cross-fitting)")
    fig.tight_layout()
    fig.savefig(OUT / "mechanism_incremental_logloss.png", dpi=180)
    plt.close(fig)


def _plot_stability(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    data = frame.sort_values("mean_dorsal_minus_ventral")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(data))
    ax.errorbar(
        data["mean_dorsal_minus_ventral"],
        y,
        xerr=[
            data["mean_dorsal_minus_ventral"] - data["ci_low"],
            data["ci_high"] - data["mean_dorsal_minus_ventral"],
        ],
        fmt="o",
        color="#8e44ad",
        ecolor="#95a5a6",
        capsize=3,
    )
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(data["channel"].str.replace("neuron:", "", regex=False), fontsize=8)
    ax.set_xlabel("Dorsal − ventral aligned activity (animal-cluster 95% CI)")
    ax.set_title("Cross-animal stability of direction-related activity")
    fig.tight_layout()
    fig.savefig(OUT / "mechanism_stability.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    recordings, _ = load_all()
    data, audit = build_aligned_dataset(recordings)
    audit.to_csv(OUT / "mechanism_alignment_audit.csv", index=False)
    stability, animal_effects = stability_by_channel(data, TRUE_CHANNELS)
    stability.to_csv(OUT / "mechanism_stability.csv", index=False)
    animal_effects.to_csv(OUT / "mechanism_animal_effects.csv", index=False)
    _plot_stability(stability)

    true_tests, true_predictions = incremental_channel_tests(data, TRUE_CHANNELS)
    control_tests, control_predictions = incremental_channel_tests(
        data, SELECTED, roll_frames=24
    )
    true_tests["analysis"] = "observed_alignment"
    control_tests["analysis"] = "temporally_matched_control_roll24"
    combined = pd.concat([true_tests, control_tests], ignore_index=True)
    # Holm correction within the observed family, separately per metric.
    for metric, group in combined[combined["analysis"] == "observed_alignment"].groupby("metric"):
        adjusted = holm_adjust(dict(zip(group["channel"], group["bootstrap_p"])))
        mask = (combined["analysis"] == "observed_alignment") & (combined["metric"] == metric)
        combined.loc[mask, "holm_p"] = combined.loc[mask, "channel"].map(adjusted)
    combined.to_csv(OUT / "mechanism_channel_tests.csv", index=False)
    true_predictions.to_parquet(OUT / "mechanism_predictions.parquet", index=False)
    control_predictions.to_parquet(OUT / "mechanism_control_predictions.parquet", index=False)
    _plot_incremental(combined[combined["analysis"] == "observed_alignment"])

    # Matched-drop table is the observed family with a clear analysis label.
    matched = combined[combined["analysis"] == "observed_alignment"].copy()
    matched["ablation"] = "drop_channel_from_behavior_plus_channel"
    matched.to_csv(OUT / "mechanism_matched_ablations.csv", index=False)

    logloss = combined[
        (combined["metric"] == "log_loss") & (combined["analysis"] == "observed_alignment")
    ].sort_values("delta")
    best = logloss.iloc[0].to_dict() if not logloss.empty else {}
    summary: dict[str, Any] = {
        "status": "OBSERVATIONAL_MECHANISM_ANALYSIS",
        "n_events": data.n_events,
        "n_animals": int(data.meta["animal_id"].nunique()),
        "n_dorsal": int(data.y.sum()),
        "n_ventral": int(len(data.y) - data.y.sum()),
        "channels": list(TRUE_CHANNELS),
        "metrics": {
            "incremental_logloss_rows": int(len(logloss)),
            "best_channel_by_logloss": best.get("channel"),
            "best_delta_logloss": best.get("delta"),
            "best_ci": [best.get("ci_low"), best.get("ci_high")],
            "best_holm_p": best.get("holm_p"),
            "any_holm_significant_improvement": bool(
                (logloss["holm_p"] < 0.05).any() if "holm_p" in logloss else False
            ),
            "temporal_control_rows": int(len(control_tests)),
        },
        "caveat": "Incremental decoding is not causal necessity; absence of incremental decoding is not proof of no function.",
    }
    (OUT / "mechanism_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
