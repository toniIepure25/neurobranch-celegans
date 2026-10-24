"""Generate Phase 5 summary figures from saved reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("reports/phase5")
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MODES = ["behavior", "behavior_sensory", "behavior_saav", "behavior_circuit", "behavior_population"]
COLORS = {
    "behavior": "#555555",
    "behavior_sensory": "#2E86AB",
    "behavior_saav": "#E4572E",
    "behavior_circuit": "#8E44AD",
    "behavior_population": "#1B998B",
    "prior": "#B0B0B0",
}


def protocol() -> None:
    path = ROOT / "protocol_comparison" / "protocol_metrics.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    regimes = [r for r in ["original", "grouped", "chronological", "external"] if r in set(frame.regime)]
    modes = [m for m in ["behavior", "saav", "both"] if m in set(frame["mode"])]
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    width = 0.8 / max(1, len(modes))
    x = np.arange(len(regimes))
    for k, mode in enumerate(modes):
        vals = [float(frame[(frame.regime == r) & (frame["mode"] == mode)]["auroc"].iloc[0]) for r in regimes]
        ax.bar(x + (k - (len(modes) - 1) / 2) * width, vals, width, label=mode.replace("behavior", "HC"))
    ax.axhline(0.5, color="black", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_ylim(0.35, 0.7)
    ax.set_ylabel("Held-out event AUROC")
    ax.set_title("Offline SAAV protocol comparison (independent reconstruction)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "protocol_auroc.png", dpi=180)
    plt.close(fig)


def online() -> None:
    path = ROOT / "online_decoding" / "online_metrics.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    linear = frame[frame.model == "linear"]
    if linear.empty:
        return
    for metric, ylabel in (("auroc", "Held-out event AUROC"), ("log_loss", "Held-out event log loss")):
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        for mode in MODES:
            sub = linear[linear["mode"] == mode].sort_values("landmark_s")
            if sub.empty:
                continue
            ax.plot(sub["landmark_s"], sub[metric], marker="o", label=mode.replace("behavior_", "+"), color=COLORS[mode])
        prior = linear[linear["mode"] == "behavior"][["landmark_s"]].drop_duplicates()
        prior_values = []
        for lm in prior["landmark_s"]:
            p = frame[(frame.landmark_s == lm) & (frame.model == "prior") & (frame["mode"] == "behavior")]["log_loss"]
            prior_values.append(float(p.iloc[0]) if len(p) else np.nan)
        if metric == "log_loss":
            ax.plot(prior["landmark_s"], prior_values, linestyle="--", marker="x", label="prior", color=COLORS["prior"])
        ax.set_xlabel("Time since causal reversal onset (s)")
        ax.set_ylabel(ylabel)
        ax.set_title("Online-prospective turn-direction prediction\n(animal-disjoint; fixed elapsed landmarks)")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(OUT / f"online_{metric}.png", dpi=180)
        plt.close(fig)

    delta_path = ROOT / "online_decoding" / "online_deltas.csv"
    if not delta_path.exists():
        return
    d = pd.read_csv(delta_path)
    d = d[(d.model == "linear") & (d.metric == "log_loss")]
    if d.empty:
        return
    comparisons = ["behavior_saav_vs_behavior", "behavior_population_vs_behavior"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for k, comp in enumerate(comparisons):
        sub = d[d.comparison == comp].sort_values("landmark_s")
        if sub.empty:
            continue
        ax.errorbar(
            sub["landmark_s"] + (k - 0.5) * 0.08,
            sub["delta"],
            yerr=[sub["delta"] - sub["ci_low"], sub["ci_high"] - sub["delta"]],
            fmt="o",
            capsize=3,
            label=comp.replace("_vs_behavior", "").replace("behavior_", "+"),
        )
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Time since causal reversal onset (s)")
    ax.set_ylabel("Δ log loss vs behaviour (negative is better)")
    ax.set_title("Neural increment over behaviour in the online benchmark")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "online_neural_delta.png", dpi=180)
    plt.close(fig)


def main() -> None:
    protocol()
    online()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
