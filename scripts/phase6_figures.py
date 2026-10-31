"""Generate Phase 6 figures from frozen machine-readable reports."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports/simulator/phase6")
    parser.add_argument("--figures", type=Path, default=PROJECT_ROOT / "figures/simulator/phase6")
    args = parser.parse_args(argv)
    reports = Path(args.reports)
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)

    # 1. Validation grid for VAR.
    grid = pd.read_csv(reports / "var_validation_grid.csv")
    grid["lag"] = grid["config"].str.extract(r"_p(\d+)").astype(int)
    grid["ridge"] = grid["config"].str.extract(r"_r([0-9.eE+-]+)_").astype(float)
    pivot = grid.pivot_table(index="lag", columns="ridge", values="skill", aggfunc="median")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=0.3)
    ax.set_xticks(range(len(pivot.columns)), [f"{v:g}" for v in pivot.columns], rotation=0)
    ax.set_yticks(range(len(pivot.index)), [f"p={v}" for v in pivot.index])
    ax.set_xlabel("ridge")
    ax.set_title("VAR validation skill (median across worms), H=20")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="skill vs constant state")
    fig.tight_layout()
    fig.savefig(figures / "validation_var_heatmap.png", dpi=180)
    plt.close(fig)

    # 2. Audit: legacy vs corrected autocorrelation.
    audit = pd.read_csv(reports / "audit/reproduction.csv")
    by = audit.groupby("kind")[["legacy_autocorr_error", "corrected_autocorr_error"]].mean()
    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(len(by))
    ax.bar(x - 0.2, by["legacy_autocorr_error"], 0.4, label="Phase 5 formula", color="#9aa5b1")
    ax.bar(x + 0.2, by["corrected_autocorr_error"], 0.4, label="corrected", color="#2b6cb0")
    ax.set_xticks(x, by.index)
    ax.set_ylabel("mean absolute lag-correlation error")
    ax.set_title("The old autocorrelation metric understated mismatch")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "audit_autocorr.png", dpi=180)
    plt.close(fig)

    # 3. Test skill by family and horizon.
    test = pd.read_csv(reports / "test_summary.csv")
    agg = test.groupby(["family", "horizon"])["skill_vs_persistence"].mean().reset_index()
    families = ["persistence", "phase5_mlp", "phase5_var", "var", "stochastic_var", "switching_var"]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    width = 0.38
    for k, horizon in enumerate((20, 100)):
        sub = agg[agg["horizon"] == horizon].set_index("family").reindex(families)
        ax.bar(np.arange(len(families)) + (k - 0.5) * width, sub["skill_vs_persistence"], width, label=f"H={horizon}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(families)), families, rotation=20, ha="right")
    ax.set_ylabel("mean skill vs constant state")
    ax.set_title("Exploratory test skill (block 2 reused)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "test_skill.png", dpi=180)
    plt.close(fig)

    # 4. Branching TV.
    branching = pd.read_csv(reports / "branching_summary.csv")
    cols = ["mean_tv_sim", "mean_tv_persistence", "mean_tv_markov", "mean_tv_marginal"]
    b = branching.groupby("family")[cols].mean()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(b))
    for k, c in enumerate(cols):
        ax.bar(x + (k - 1.5) * 0.2, b[c], 0.2, label=c.replace("mean_tv_", ""))
    ax.set_xticks(x, b.index)
    ax.set_ylabel("mean total variation")
    ax.set_title("Conditional REV outcomes: lower is better")
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(figures / "branching_tv.png", dpi=180)
    plt.close(fig)

    # 5. Perturbation support vs effect.
    pert = pd.read_csv(reports / "perturbation_summary.csv")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for protocol, sub in pert.groupby("protocol"):
        ax.scatter(sub["mean_in_support"], sub["mean_tv"], s=18, alpha=0.7, label=protocol)
    ax.axvline(0.90, color="#c53030", linestyle="--", linewidth=1, label="G5 threshold")
    ax.set_xlabel("mean in-support fraction")
    ax.set_ylabel("mean TV vs baseline")
    ax.set_title("Support-constrained virtual perturbations")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "perturbation_support.png", dpi=180)
    plt.close(fig)

    # 6. Long-run stability.
    long_run = pd.read_csv(reports / "long_run_stability.csv")
    if len(long_run):
        fig, ax = plt.subplots(figsize=(6, 3))
        for family, sub in long_run.groupby("family"):
            ax.scatter(np.arange(len(sub)), sub["final_norm_ratio"], label=family, s=25)
        ax.axhline(1.0, color="black", linewidth=0.8)
        ax.set_ylabel("final / initial state norm")
        ax.set_title("Long-run (H=200) diagnostic")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "long_run_stability.png", dpi=180)
        plt.close(fig)

    (figures / "README.md").write_text(
        "# Phase 6 figures\n\nGenerated by `scripts/phase6_figures.py` from frozen reports. "
        "Test figures are exploratory because block 2 was reused.\n",
        encoding="utf-8",
    )
    print(f"wrote figures to {figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
