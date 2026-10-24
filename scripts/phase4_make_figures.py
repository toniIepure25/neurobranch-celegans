"""Phase 4 figures.  Every figure is generated from a saved report table."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "phase4"
FIG = REPORT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
COLORS = {"behavior": "#4C78A8", "neural": "#E4572E", "population": "#F58518", "saav": "#54A24B", "circuit": "#B279A2"}


def save(fig, name: str) -> None:
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_primary() -> None:
    m = pd.read_csv(REPORT / "turn_direction_metrics.csv").sort_values("auroc")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = ["#E4572E" if "neural" in x else "#4C78A8" for x in m["mode"]]
    ax.barh(m["mode"], m["auroc"], color=colors)
    ax.axvline(0.5, color="#333", ls="--", lw=1, label="chance")
    ax.set_xlim(0.35, 0.68)
    ax.set_xlabel("pooled out-of-animal AUROC")
    ax.set_title("Turn direction: held-out animal AUROC")
    ax.legend(frameon=False, loc="lower right")
    for i, v in enumerate(m["auroc"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    save(fig, "phase4_turn_direction_auroc.png")


def _delta_rows() -> pd.DataFrame:
    rows = []

    def add(label, group, source):
        if group:
            rows.append(
                {
                    "contrast": label,
                    "mean": group["mean"],
                    "lo": group["ci95"][0],
                    "hi": group["ci95"][1],
                    "source": source,
                }
            )

    s = json.loads((REPORT / "turn_direction_summary.json").read_text(encoding="utf-8"))
    add("all WT: +population", s["summary"]["deltas"].get("delta_behavior_population_vs_behavior"), "primary")
    add("all WT: +SAAV", s["summary"]["deltas"].get("delta_behavior_saav_vs_behavior"), "primary")
    add("all WT: +circuit", s["summary"]["deltas"].get("delta_behavior_circuit_vs_behavior"), "primary")
    for cohort, label in [("wt_octanol", "octanol: +population"), ("wt_diacetyl", "diacetyl: +population")]:
        p = REPORT / f"turn_direction_summary_{cohort}.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))["summary"]["deltas"]
            add(label, d.get("delta_behavior_population_vs_behavior"), "cohort")
    for window in ("early", "late"):
        p = REPORT / f"turn_direction_summary_{window}.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))["summary"]["deltas"]
            add(f"{window} window: +population", d.get("delta_behavior_population_vs_behavior"), "window")
    return pd.DataFrame(rows)


def fig_forest() -> None:
    d = _delta_rows()
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(d))
    ax.errorbar(
        d["mean"],
        y,
        xerr=[d["mean"] - d["lo"], d["hi"] - d["mean"]],
        fmt="o",
        color="#222",
        ecolor="#777",
        capsize=3,
    )
    ax.axvline(0, color="#C1121F", ls="--", lw=1)
    ax.set_yticks(y, d["contrast"])
    ax.set_xlabel("Δ AUROC vs behaviour baseline (95% animal-cluster bootstrap CI)")
    ax.set_title("Neural increment is not established in the primary cohort")
    save(fig, "phase4_turn_direction_delta_forest.png")


def fig_calibration() -> None:
    p = REPORT / "turn_direction_calibration.csv"
    if not p.exists():
        return
    cal = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for mode, g in cal.groupby("mode"):
        ax.plot(g["mean_predicted"], g["fraction_positive"], marker="o", label=mode)
    ax.plot([0, 1], [0, 1], ls="--", color="#333", label="ideal")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraction dorsal")
    ax.set_title("Calibration (held-out animals)")
    ax.legend(fontsize=7, frameon=False)
    save(fig, "phase4_turn_direction_calibration.png")


def fig_dynamics() -> None:
    p = REPORT / "turn_dynamics_timeresolved.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for mode, g in d.groupby("mode"):
        ax.plot(g["relative_frame"], g["auroc"], marker="o", ms=3, label=mode)
    ax.axvline(0, color="#333", ls="--", lw=1)
    ax.axvspan(-12, 0, color="#4C78A8", alpha=0.06)
    ax.set_xlabel("frame relative to reversal offset (negative = prospective)")
    ax.set_ylabel("held-out AUROC")
    ax.set_title("Behaviour dominates immediately before the offset; neural adds little")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "phase4_turn_dynamics_timeresolved.png")


def fig_replication() -> None:
    p = REPORT / "replication_summary.json"
    if not p.exists():
        return
    s = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for h in ("3.0", "6.0", "12.0"):
        a = s["horizons"][h]["aggregate"]["delta_behavior_neural"]
        rows.append({"h": float(h), "mean": a["mean"], "lo": a["ci95"][0], "hi": a["ci95"][1], "nneg": a["n_negative"], "n": a["n"]})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.errorbar(d["h"], d["mean"], yerr=[d["mean"] - d["lo"], d["hi"] - d["mean"]], fmt="o", capsize=4, color="#4C78A8")
    ax.axhline(0, color="#C1121F", ls="--", lw=1)
    ax.set_xticks(d["h"], [f"{int(h)} s" for h in d["h"]])
    ax.set_ylabel("Δ log loss (behaviour+neural − behaviour)")
    ax.set_title("Independent Phase 3 replication on WT octanol")
    for _, r in d.iterrows():
        ax.annotate(f"{r['nneg']:.0f}/{r['n']:.0f}", (r["h"], r["mean"]), textcoords="offset points", xytext=(8, 8), fontsize=8)
    save(fig, "phase4_replication_delta.png")


def fig_pca() -> None:
    p = REPORT / "turn_pca_trajectories.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for direction, g in d.groupby("direction"):
        ax.plot(g["pc1_mean"], g["pc2_mean"], marker="o", ms=3, label=direction)
        ax.annotate(str(int(g["k"].iloc[0])), (g["pc1_mean"].iloc[0], g["pc2_mean"].iloc[0]), fontsize=7)
        ax.annotate(str(int(g["k"].iloc[-1])), (g["pc1_mean"].iloc[-1], g["pc2_mean"].iloc[-1]), fontsize=7)
    ax.set_xlabel("PC1 of population state")
    ax.set_ylabel("PC2 of population state")
    ax.set_title("Population-state trajectory around the reversal offset")
    ax.legend(frameon=False)
    save(fig, "phase4_turn_pca_trajectories.png")


def fig_peaks() -> None:
    p = REPORT / "turn_neuron_peak_times.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    labels = list(d["label"].unique())
    y = np.arange(len(labels))
    for direction, offset in [("ventral", -0.15), ("dorsal", 0.15)]:
        g = d[d["direction"] == direction].set_index("label").reindex(labels)
        ax.scatter(g["mean"], y + offset, label=direction, s=35)
    ax.axvline(0, color="#333", ls="--", lw=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("mean relative frame of peak activity (negative = before offset)")
    ax.set_title("Descriptive peak timing of prespecified circuit neurons")
    ax.legend(frameon=False)
    save(fig, "phase4_turn_neuron_peak_times.png")


def fig_external() -> None:
    p = REPORT / "turn_direction_external.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d["setting"] = d["train"] + " → " + d["test"]
    fig, ax = plt.subplots(figsize=(8, 4.0))
    settings = list(d["setting"].unique())
    modes = ["behavior", "behavior_population", "behavior_saav", "behavior_circuit"]
    width = 0.18
    for j, mode in enumerate(modes):
        vals = [d[(d["setting"] == s) & (d["mode"] == mode)]["auroc"].mean() for s in settings]
        ax.bar(np.arange(len(settings)) + (j - 1.5) * width, vals, width=width, label=mode)
    ax.axhline(0.5, color="#333", ls="--", lw=1)
    ax.set_xticks(np.arange(len(settings)), settings)
    ax.set_ylabel("AUROC")
    ax.set_title("External cohort shift")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    save(fig, "phase4_turn_direction_external.png")


def fig_inventory() -> None:
    p = REPORT / "behavior_event_inventory.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for cohort, g in d.groupby("cohort"):
        ax.scatter(g["n_long_12_valid_post"], g["fraction_dorsal_12"], label=cohort, s=28)
    ax.set_xlabel("long (≥12 frame) reversals per animal")
    ax.set_ylabel("fraction dorsal")
    ax.set_title("Severe class imbalance and animal heterogeneity")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "phase4_event_inventory.png")


def fig_ablation() -> None:
    p = REPORT / "turn_direction_neuron_ablation.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(d["feature"], d["delta_vs_full"], color="#B279A2")
    ax.axhline(0, color="#333", lw=1)
    ax.set_ylabel("Δ AUROC after dropping feature")
    ax.set_title("Model-dependent circuit-feature ablation")
    plt.xticks(rotation=35, ha="right")
    save(fig, "phase4_turn_neuron_ablation.png")


def fig_nonlinear() -> None:
    p = REPORT / "turn_direction_nonlinear.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.bar(d["mode"], d["auroc"], color=["#4C78A8", "#F58518"])
    ax.axhline(0.5, color="#333", ls="--", lw=1)
    ax.set_ylim(0.4, 0.7)
    ax.set_ylabel("AUROC")
    ax.set_title("Nonlinear gradient-boosting sensitivity")
    plt.xticks(rotation=20, ha="right")
    save(fig, "phase4_turn_direction_nonlinear.png")


def main() -> None:
    fig_primary()
    fig_forest()
    fig_calibration()
    fig_dynamics()
    fig_replication()
    fig_pca()
    fig_peaks()
    fig_external()
    fig_inventory()
    fig_ablation()
    fig_nonlinear()
    print("wrote figures", FIG)


if __name__ == "__main__":
    main()
