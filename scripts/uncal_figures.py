"""Figures for the uncertainty-calibration report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("reports/simulator/uncertainty_calibration")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "figure.dpi": 160})


def _save(fig, name: str) -> None:
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def energy_figure() -> None:
    s = pd.read_csv(OUT / "score_summary.csv").sort_values("energy_h20")
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    colors = ["#2f6f9f" if m.startswith("P2") else ("#b54a3a" if m == "P6_adaptive_block" else "#8a9aa8") for m in s.model]
    ax.barh(s.model, s.energy_h20, color=colors)
    for y, v in enumerate(s.energy_h20):
        ax.text(v + 0.03, y, f"{v:.3f}", va="center", fontsize=7)
    ax.set_xlabel("mean energy score at H20 (lower is better)")
    ax.set_title("Proper multivariate score, fixed 10-D training PCA")
    ax.invert_yaxis()
    _save(fig, "uncal_energy_comparison.png")


def coverage_figure() -> None:
    c = pd.read_csv(OUT / "coverage_by_horizon.csv")
    models = ["P2_independent_ar_lead1", "P0_persistence", "P2_fixed_train_scale", "P6_adaptive_block", "P6_adaptive_lead1"]
    labels = {
        "P2_independent_ar_lead1": "indep. AR, lead-1",
        "P0_persistence": "persistence",
        "P2_fixed_train_scale": "indep. AR, fixed scale",
        "P6_adaptive_block": "population block (historical)",
        "P6_adaptive_lead1": "population lead-1",
    }
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    for model in models:
        g = c[c.model == model].groupby("horizon")[["coverage_50", "coverage_80", "coverage_95"]].mean()
        for level, style in ((50, ":"), (80, "--"), (95, "-")):
            ax.plot(g.index, g[f"coverage_{level}"], style, marker="o", ms=3, lw=1.1, label=f"{labels[model]} {level}%")
    for level in (50, 80, 95):
        ax.axhline(level / 100, color="k", lw=0.6, alpha=0.35)
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 20, 50, 100], labels=["1", "5", "10", "20", "50", "100"])
    ax.set_xlabel("forecast horizon (frames)")
    ax.set_ylabel("empirical central coverage")
    ax.set_ylim(0.4, 1.01)
    ax.legend(fontsize=5.2, ncol=2, loc="lower left")
    ax.set_title("Marginal interval coverage by horizon")
    _save(fig, "uncal_coverage_by_horizon.png")


def variance_figure() -> None:
    v = pd.read_csv(OUT / "variance_summary.csv")
    models = ["P2_independent_ar_lead1", "P0_persistence", "P6_adaptive_block", "P6_adaptive_lead1"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
    for model in models:
        g = v[v.model == model].sort_values("horizon")
        axes[0].plot(g.horizon, g.within_var_mean, marker="o", ms=3, label=model)
        axes[1].plot(g.horizon, g.spread_error_ratio, marker="o", ms=3, label=model)
    axes[0].set_yscale("log")
    axes[0].set_xscale("log")
    axes[0].set_xticks([1, 5, 10, 20, 50, 100], labels=["1", "5", "10", "20", "50", "100"])
    axes[0].set_xlabel("horizon")
    axes[0].set_ylabel("mean ensemble variance")
    axes[0].set_title("Predictive variance growth")
    axes[1].axhline(1.0, color="k", lw=0.7, ls="--")
    axes[1].set_xscale("log")
    axes[1].set_xticks([1, 5, 10, 20, 50, 100], labels=["1", "5", "10", "20", "50", "100"])
    axes[1].set_xlabel("horizon")
    axes[1].set_ylabel("spread / squared error")
    axes[1].set_title("Spread-error ratio")
    axes[1].legend(fontsize=6, loc="upper left")
    _save(fig, "uncal_variance_growth.png")


def lead_figure() -> None:
    p = pd.read_csv(OUT / "lead_profile.csv")
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for model, label in (("adaptive_var_block50", "adaptive block-50"), ("adaptive_var", "adaptive block-10"), ("independent_ar", "independent AR")):
        g = p[p.model == model].sort_values("lead")
        ax.plot(g.lead, g.trace_cov, marker="o", ms=3, label=label)
    ax.set_xlabel("prequential lead admitted to innovation pool")
    ax.set_ylabel("median residual covariance trace")
    ax.set_title("Pooling multi-step residuals inflates innovation variance")
    ax.legend(fontsize=7)
    _save(fig, "uncal_lead_pooling.png")


def trajectory_figure() -> None:
    bands = pd.read_csv(OUT / "example_bands.csv")
    obs = pd.read_csv(OUT / "example_observed.csv")
    selected = "population_mean"
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    for ax, model, title in (
        (axes[0], "P2_independent_ar_lead1", "Independent AR, lead-1 diagonal"),
        (axes[1], "P6_adaptive_block", "Historical population block"),
    ):
        g = bands[(bands.model == model) & (bands.series == selected)].sort_values("step")
        ax.fill_between(g.step, g.q05, g.q95, alpha=0.25, label="5-95%")
        ax.plot(g.step, g.q50, lw=1.2, label="median")
        o = obs[obs.series == selected].sort_values("step")
        ax.plot(o.step, o.value, "k", lw=1.2, label="observed")
        ax.set_title(title)
        ax.set_xlabel("step after landmark")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("population mean activity")
    axes[0].legend(fontsize=6)
    _save(fig, "uncal_example_trajectory.png")


def calibration_figure() -> None:
    c = pd.read_csv(OUT / "calibration_curves.csv")
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for model in ["P2_independent_ar_lead1", "P0_persistence", "P6_adaptive_block", "P6_adaptive_lead1"]:
        g = c[c.model == model].sort_values("bin")
        ax.plot(np.arange(10) / 10 + 0.05, g.density, marker="o", ms=3, label=model)
    ax.axhline(0.1, color="k", ls="--", lw=0.7, label="uniform")
    ax.set_xlabel("PIT bin")
    ax.set_ylabel("density")
    ax.set_title("PIT calibration, H20")
    ax.legend(fontsize=6)
    _save(fig, "uncal_pit.png")


def main() -> None:
    energy_figure()
    coverage_figure()
    variance_figure()
    lead_figure()
    trajectory_figure()
    calibration_figure()
    rows = []
    for path in sorted(FIG.glob("*.png")):
        rows.append({"figure": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (OUT / "figure_manifest.json").write_text(json.dumps(rows, indent=2))
    print("figures", len(rows))


if __name__ == "__main__":
    main()
