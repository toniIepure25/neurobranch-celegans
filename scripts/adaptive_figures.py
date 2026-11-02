"""Figures for the adaptive-dynamics stability experiment."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from neurobranch.config import PROJECT_ROOT

REPORT = PROJECT_ROOT / "reports/simulator/adaptive_dynamics"
DOCS = PROJECT_ROOT / "docs/assets/adaptive_dynamics"
REPORT.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(REPORT / name, dpi=160)
    fig.savefig(DOCS / name, dpi=160)
    plt.close(fig)


def main() -> int:
    horizon = pd.read_csv(REPORT / "model_horizon_curves.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for model, group in horizon.groupby("model"):
        ax.plot(group.horizon, group.mean_mse, marker="o", ms=3, label=model)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("forecast horizon (frames)")
    ax.set_ylabel("mean free-running MSE (log scale)")
    ax.set_title("Frozen versus causal adaptive VAR")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    save(fig, "forecast_horizon.png")

    session = pd.read_csv(REPORT / "model_per_session_h20.csv")
    pivot = session.pivot(index="worm_id", columns="model", values="mean_mse")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("H20 mean free-running MSE")
    ax.set_xlabel("Kato WT_NoStim session")
    ax.set_title("Paired H20 error by session")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "session_h20.png")

    windows = pd.read_csv(REPORT / "nonstationarity_windows.csv")
    null = pd.read_csv(REPORT / "nonstationarity_null.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for worm, group in windows.groupby("worm_id"):
        ax.plot(group.window_center, group.reference_angle_mean_deg, lw=1.1, label=f"session {worm}")
    q95 = null.groupby("worm_id").reference_angle_mean_deg.quantile(0.95)
    ax.plot([], [], "k--", label="surrogate 95th percentile (per session)")
    for _worm, value in q95.items():
        ax.axhline(value, color="0.4", ls="--", lw=0.6, alpha=0.5)
    ax.set_xlabel("window centre (frames)")
    ax.set_ylabel("mean principal angle to train subspace (deg)")
    ax.set_title("Sliding-window latent orientation versus stationary surrogate")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    save(fig, "subspace_drift.png")

    calcium = pd.read_csv(REPORT / "calcium_summary.csv")
    h20 = calcium[calcium.horizon == 20].pivot(index="case", columns="model", values="mean_mse")
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    h20.plot(kind="bar", logy=True, ax=ax)
    ax.set_ylabel("H20 mean MSE (log scale)")
    ax.set_title("Calcium rollout: original feedback versus corrected recursion")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "calcium_h20.png")

    strata = pd.read_csv(REPORT / "adaptation_strata.csv")
    selected = strata[(strata.model == "rolling_w500") & strata.indicator.isin(["occupancy_shift", "amplitude_shift", "covariance_shift"])]
    fig, ax = plt.subplots(figsize=(7.1, 4.0))
    for indicator, group in selected.groupby("indicator"):
        ax.plot(group.quartile, group.mean_relative_improvement, marker="o", label=indicator)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("indicator quartile (available at prediction time)")
    ax.set_ylabel("mean relative H20 improvement vs frozen")
    ax.set_title("Where adaptation helps: rolling VAR, window 500")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    save(fig, "adaptation_strata.png")
    print("figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
