"""Figures for the probabilistic adaptive-dynamics report."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "reports/simulator/probabilistic_adaptive"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "figure.dpi": 160})


def _load(name):
    return pd.read_csv(OUT / name)


def energy_figure():
    summary = _load("prob_summary.csv")
    e = summary[summary.metric == "energy_h20"].sort_values("mean")
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    colors = ["#2f6f9f" if m.startswith("P6") else "#8a9aa8" for m in e.model]
    ax.barh(e.model, e["mean"], color=colors)
    for y, (_m, v) in enumerate(zip(e.model, e["mean"], strict=True)):
        ax.text(v + 0.02, y, f"{v:.3f}", va="center", fontsize=7)
    ax.set_xlabel("mean energy score at H20 (lower is better)")
    ax.set_title("Multivariate energy score, 10-D training PCA projection")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "prob_energy_comparison.png")
    plt.close(fig)


def calibration_figure():
    summary = _load("prob_summary.csv")
    models = ["P0_persistence", "P1_local_mean", "P2_independent_ar", "P6_adaptive_block", "P7_adaptive_diag"]
    labels = {"P0_persistence": "persistence", "P1_local_mean": "local mean", "P2_independent_ar": "indep. AR", "P6_adaptive_block": "adaptive block", "P7_adaptive_diag": "adaptive diag"}
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for model in models:
        vals = []
        for cov in ("coverage_50", "coverage_80", "coverage_95"):
            row = summary[(summary.model == model) & (summary.metric == cov)]
            vals.append(float(row["mean"].iloc[0]))
        ax.plot([50, 80, 95], vals, marker="o", label=labels[model], lw=1.3)
    ax.plot([50, 80, 95], [0.5, 0.8, 0.95], "k--", lw=1, label="nominal")
    ax.set_xlabel("nominal central interval (%)")
    ax.set_ylabel("empirical coverage")
    ax.set_ylim(0.3, 1.02)
    ax.legend(fontsize=6, ncol=2)
    ax.set_title("Interval calibration across 175 landmarks")
    fig.tight_layout()
    fig.savefig(FIG / "prob_calibration.png")
    plt.close(fig)


def variance_figure():
    v = _load("prob_variance_summary.csv").sort_values("total_implied_var")
    x = np.arange(len(v))
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(x - 0.2, v.within_ensemble_var, width=0.4, label="within-ensemble", color="#3b7ea1")
    ax.bar(x + 0.2, v.total_implied_var, width=0.4, label="total implied", color="#d08a3e")
    ax.axhline(float(v.observed_var.iloc[0]), color="k", ls="--", lw=1, label="observed variance")
    ax.set_xticks(x, v.model, rotation=35, ha="right")
    ax.set_ylabel("mean variance (population mean)")
    ax.legend(fontsize=7)
    ax.set_title("Variance decomposition: overdispersion of adaptive stochastic models")
    fig.tight_layout()
    fig.savefig(FIG / "prob_variance_decomposition.png")
    plt.close(fig)


def horizon_figure():
    h = _load("prob_horizon.csv")
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for model, group in h.groupby("model"):
        if model in {"P0_persistence", "P1_local_mean", "P2_independent_ar", "P5_adaptive_boot", "P6_adaptive_block", "P7_adaptive_diag"}:
            ax.plot(group.horizon, group.mean_energy, marker="o", label=model.replace("_", " "), lw=1.2)
    ax.set_xlabel("forecast horizon (frames)")
    ax.set_ylabel("mean energy score")
    ax.legend(fontsize=6, ncol=2)
    ax.set_title("Energy score growth with horizon")
    fig.tight_layout()
    fig.savefig(FIG / "prob_horizon.png")
    plt.close(fig)


def example_figure():
    tr = _load("prob_example_trajectories.csv")
    bands = _load("prob_example_bands.csv")
    p6 = bands[bands.model == "P6_adaptive_block"]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for _sample, group in tr.groupby("sample"):
        ax.plot(group.step, group.population_mean, color="#8a9aa8", lw=0.6, alpha=0.55)
    ax.fill_between(p6.step, p6.q05, p6.q95, color="#3b7ea1", alpha=0.18, label="90% predictive band")
    ax.plot(p6.step, p6.ensemble_population_mean, color="#1f4e66", lw=1.7, label="ensemble mean")
    ax.plot(p6.step, p6.deterministic_population_mean, color="#d08a3e", lw=1.4, ls="--", label="deterministic conditional mean")
    ax.plot(p6.step, p6.truth_population_mean, color="black", lw=1.4, label="observed continuation")
    ax.set_xlabel("steps after landmark")
    ax.set_ylabel("population-mean activity")
    ax.set_title("One history, many plausible futures (session 0, landmark 2800)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "prob_example_rollouts.png")
    plt.close(fig)


def residual_figure():
    acf = _load("residual_autocorrelation.csv")
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for model in ("adaptive", "independent_ar", "local_mean", "persistence"):
        g = acf[acf.model == model].groupby("lag").median_abs_acf.mean()
        ax.plot(g.index, g.values, marker="o", label=model, lw=1.2)
    ax.set_xlabel("residual lag (frames)")
    ax.set_ylabel("median |autocorrelation|")
    ax.set_title("Training residuals retain temporal dependence")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "prob_residual_autocorrelation.png")
    plt.close(fig)


def main() -> int:
    energy_figure()
    calibration_figure()
    variance_figure()
    horizon_figure()
    example_figure()
    residual_figure()
    meta = {"figures": sorted(p.name for p in FIG.glob("*.png")), "warning": "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED."}
    (FIG / "figure_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
