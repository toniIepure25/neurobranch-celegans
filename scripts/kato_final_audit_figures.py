"""Generate supervisor-facing audit figures from frozen audit tables."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "reports" / "simulator" / "kato_final_audit"
OUT = AUDIT / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.dpi": 160,
        "savefig.dpi": 220,
    }
)


def forest(ax, labels, means, lows, highs, xlabel, title, colors=None):
    y = np.arange(len(labels))[::-1]
    if colors is None:
        colors = ["#1f6f8b"] * len(labels)
    for yi, mean, low, high, color in zip(y, means, lows, highs, colors, strict=True):
        ax.plot([low, high], [yi, yi], color=color, lw=2.2, solid_capstyle="round")
        ax.plot(mean, yi, "o", color=color, ms=5)
    ax.axvline(0, color="#555555", lw=0.8, ls="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold")


# Figure 1: uncertainty across blocks, sessions and horizons.
blocks = pd.read_csv(AUDIT / "uncertainty_bootstrap.csv")
sessions = pd.read_csv(AUDIT / "session_estimates.csv")
horizons = pd.read_csv(AUDIT / "horizon_uncertainty.csv")
fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)
forest(
    axes[0],
    [f"block {b}" for b in blocks["block"]],
    blocks["mean_advantage"],
    blocks["ci_low"],
    blocks["ci_high"],
    "B5 - B3 energy (positive: B3 better)",
    "A. Block-length sensitivity",
)
forest(
    axes[1],
    [f"session {s} ({n})" for s, n in zip(sessions["session"], sessions["n_landmarks"], strict=True)],
    sessions["mean_advantage"],
    sessions["within_session_block5_low"],
    sessions["within_session_block5_high"],
    "B5 - B3 energy",
    "B. Repeated sessions (landmarks)",
    ["#b23a48" if v < 0 else "#1f6f8b" for v in sessions["mean_advantage"]],
)
forest(
    axes[2],
    [f"H{h}" for h in horizons["horizon"]],
    horizons["mean_advantage"],
    horizons["ci_low"],
    horizons["ci_high"],
    "B5 - B3 energy",
    "C. Horizon dependence",
    ["#b23a48" if v < 0 else "#1f6f8b" for v in horizons["mean_advantage"]],
)
axes[0].set_xlim(-0.02, 0.46)
fig.suptitle(
    "Kato WT_NoStim audit: exploratory energy advantage is block-robust but heterogeneous",
    x=0.01,
    ha="left",
    fontsize=11,
    fontweight="bold",
)
fig.savefig(OUT / "kato_final_audit_uncertainty.png", bbox_inches="tight")
plt.close(fig)

# Figure 2: representation specificity and ordered decomposition.
rep = pd.read_csv(AUDIT / "clean_rerun/representation_sensitivity.csv")
decomp = pd.read_csv(AUDIT / "decomposition_order_sensitivity.csv")
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), constrained_layout=True)
labels = ["PCA-10\nenergy", "neuron\nenergy", "PCA-10\nMSE", "neuron\nMSE"]
forest(
    axes[0],
    labels,
    rep["mean_improvement"],
    rep["ci_low"],
    rep["ci_high"],
    "B5 - B3 (positive: B3 better)",
    "A. Representation and metric",
    ["#1f6f8b" if v > 0 else "#b23a48" for v in rep["mean_improvement"]],
)
axes[0].set_xlim(-0.25, 0.55)

x = np.arange(len(decomp))
width = 0.36
axes[1].bar(
    x - width / 2,
    decomp["common_increment"],
    width,
    color="#1f6f8b",
    label="common-factor increment (B4m-B5)",
)
axes[1].bar(
    x + width / 2,
    decomp["residual_increment"],
    width,
    color="#d08c34",
    label="residual increment (B5-B3)",
)
axes[1].axhline(0, color="#555555", lw=0.8)
axes[1].set_xticks(x, decomp["baseline"], rotation=12)
axes[1].set_ylabel("energy increment")
axes[1].set_title("B. Ordered decomposition depends on baseline", loc="left", fontweight="bold")
axes[1].legend(frameon=False, fontsize=7.5, loc="upper left")
for xi, share in zip(x, decomp["common_share"], strict=True):
    axes[1].text(
        xi,
        max(decomp["common_increment"]) * 1.08,
        f"{share:.0%} common",
        ha="center",
        fontsize=7.5,
        color="#333333",
    )
fig.suptitle(
    "The 51/49 split is ordered, representation-specific, and not a unique variance partition",
    x=0.01,
    ha="left",
    fontsize=11,
    fontweight="bold",
)
fig.savefig(OUT / "kato_final_audit_decomposition.png", bbox_inches="tight")
plt.close(fig)

# Figure 3: primary paired result and evidence tiers.
tiers = json.loads((AUDIT / "claims_ledger.json").read_text(encoding="utf-8"))
counts = tiers["counts"]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
axes[0].bar(
    ["B3 population", "B5-10 own+factors"],
    [2.7001497504411485, 2.9645877431095045],
    color=["#1f6f8b", "#d08c34"],
    width=0.55,
)
axes[0].set_ylabel("H20 PCA-10 energy (lower is better)")
axes[0].set_ylim(2.55, 3.28)
axes[0].set_title("A. Primary paired result", loc="left", fontweight="bold")
axes[0].text(
    0.5,
    3.05,
    "B5 - B3 = +0.264\n95% block-5 CI [+0.075, +0.403]",
    ha="center",
    fontsize=8.5,
    bbox={"boxstyle": "round,pad=0.3", "fc": "#f2f2f2", "ec": "#cccccc"},
)
order = ["SUPPORTED_EXPLORATORY", "LIMITED", "UNSUPPORTED", "INVALIDATED", "NOT_EVALUABLE"]
vals = [counts.get(k, 0) for k in order]
colors = ["#1f6f8b", "#7fa7b8", "#9a9a9a", "#b23a48", "#6b4c9a"]
axes[1].barh(order[::-1], vals[::-1], color=colors[::-1], height=0.55)
axes[1].set_xlabel("number of claims")
axes[1].set_title("B. Final claims ledger (no CONFIRMED)", loc="left", fontweight="bold")
for i, value in enumerate(vals[::-1]):
    axes[1].text(value + 0.1, i, str(value), va="center", fontsize=8)
fig.suptitle(
    "Kato final audit: exploratory predictive evidence, bounded claims",
    x=0.01,
    ha="left",
    fontsize=11,
    fontweight="bold",
)
fig.savefig(OUT / "kato_final_audit_summary.png", bbox_inches="tight")
plt.close(fig)

print("wrote", [p.name for p in sorted(OUT.glob("*.png"))])
