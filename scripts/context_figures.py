"""Static figures for the context-dynamics reports."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from neurobranch.config import PROJECT_ROOT, ExperimentConfig
from neurobranch.dynamics.training import load_worm_data
from neurobranch.preprocessing import split_starts

REPORTS = PROJECT_ROOT / "reports/simulator/context_dynamics"
FIGURES = REPORTS / "figures"


def _trajectory_figure() -> None:
    datasets = load_worm_data(ExperimentConfig.load(PROJECT_ROOT / "configs/experiments/forward_stress.yaml"))
    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for data, color in zip(datasets, colors):
        path = REPORTS / "trajectories" / f"latent_lds_worm{data.worm_id}.npz"
        if not path.exists():
            continue
        payload = np.load(path)
        starts = split_starts(data, "train", history=1, horizon=1)
        pca = PCA(n_components=3, random_state=17).fit(data.X[starts])
        obs = pca.transform(payload["observed"])
        sim = pca.transform(payload["simulated"])
        fig.add_trace(go.Scatter3d(x=obs[:, 0], y=obs[:, 1], z=obs[:, 2], mode="lines",
                                   line=dict(color=color, width=6), name=f"worm{data.worm_id} observed"))
        fig.add_trace(go.Scatter3d(x=sim[:, 0], y=sim[:, 1], z=sim[:, 2], mode="lines",
                                   line=dict(color=color, width=3, dash="dash"), name=f"worm{data.worm_id} latent rollout"))
    fig.update_layout(
        title="Observed vs autonomous latent rollout (3-D PCA; model output, not animal response)",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        height=760,
    )
    fig.write_html(FIGURES / "latent_rollout_3d.html", include_plotlyjs="cdn")


def _bars() -> None:
    latent = pd.read_csv(REPORTS / "free_running_latent.csv")
    latent = latent[latent["model"].isin(["latent_lds", "stable_var"])]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (model, group) in enumerate(latent.groupby("model")):
        ax.bar(np.arange(len(group)) + i * 0.35, group["skill_vs_persistence"], width=0.34, label=model)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(group)) + 0.17, [f"w{int(w)} h{int(h)}" for w, h in zip(group["worm_id"], group["horizon"])])
    ax.set_ylabel("skill vs persistence")
    ax.set_title("Autonomous rollout: negative skill (lower is worse)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "autonomous_skill.png", dpi=150)
    plt.close(fig)

    oxygen = pd.read_csv(REPORTS / "oxygen_model_aggregate.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(oxygen["model"], oxygen["mean_skill"], color="#4c78a8")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("mean H20 skill vs persistence")
    ax.set_title("Known oxygen schedule does not produce positive free-running skill")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIGURES / "oxygen_models.png", dpi=150)
    plt.close(fig)

    branching = pd.read_csv(REPORTS / "branching_aggregate.csv")
    cols = ["mean_tv_sim", "mean_tv_persistence", "mean_tv_marginal", "mean_tv_markov"]
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = [branching[c].dropna().mean() for c in cols]
    ax.bar(["latent\nstochastic", "persistence", "marginal", "Markov"], vals,
           color=["#e4572e", "#7d8491", "#54a24b", "#4c78a8"])
    ax.set_ylabel("mean TV to observed local outcome")
    ax.set_title("Branching diagnostic: lower is closer to observed local outcomes")
    fig.tight_layout()
    fig.savefig(FIGURES / "branching_tv.png", dpi=150)
    plt.close(fig)

    support = pd.read_csv(REPORTS / "phase6_support_summary.csv")
    pivot = support.groupby(["family", "q"])["fraction_starts_fully_inside_at_q"].mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 4))
    for family in pivot.index:
        ax.plot([90, 95, 99], pivot.loc[family].values, marker="o", label=family)
    ax.set_xlabel("support quantile")
    ax.set_ylabel("fraction of starts with maximum distance inside")
    ax.set_title("Phase 6 support audit (frozen official threshold remains q99)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "support_audit.png", dpi=150)
    plt.close(fig)

    coef = pd.read_csv(REPORTS / "operator_coefficient_trajectories.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    for worm, group in coef.groupby("worm_id"):
        ax.plot(group["frame"], group["c1"], linewidth=1, label=f"w{worm} c1")
    ax.set_ylabel("coefficient c1 (causal policy)")
    ax.set_xlabel("test frame")
    ax.set_title("Compositional coefficients are low-entropy and weakly varying")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURES / "operator_coefficients.png", dpi=150)
    plt.close(fig)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    _trajectory_figure()
    _bars()
    (FIGURES / "figure_manifest.json").write_text(
        json.dumps({"figures": sorted(p.name for p in FIGURES.iterdir() if p.is_file())}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
