"""Static figures for the neural-dynamics simulator evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, ExperimentConfig
from neurobranch.simulator.branching import reversal_states
from neurobranch.simulator.latent import PCAEncoder
from neurobranch.simulator.train import checkpoint_root, load_datasets, load_simulator_for

OUT = PROJECT_ROOT / "figures" / "simulator"
COLORS = {"FWD": "#4C78A8", "REV": "#F58518", "TURN": "#54A24B", "NOSTATE": "#B279A2"}


def _setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def trajectory_figure(cfg, reports: Path) -> None:
    selected = json.loads((reports / "selection.json").read_text(encoding="utf-8"))["selected_kind"]
    root = checkpoint_root(cfg)
    datasets = load_datasets(cfg)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, data in zip(axes, datasets):
        pca = PCAEncoder.fit(data, n_components=3)
        sim = load_simulator_for(data, selected, 17, root=root)
        frames = [int(t) for t in reversal_states(data, horizon=30) if t + 30 < data.n_frames]
        frame = frames[0] if frames else int(sim.test_starts(horizon=30)[0])
        ensemble = sim.rollout_ensemble(
            data.X[frame], 30, 30, base_seed=1000 + frame, noise_std=0.05
        )
        Zobs = pca.transform(data.X[frame : frame + 31])
        Zsim = [pca.transform(r.states) for r in ensemble.rollouts]
        ax.plot(Zobs[:, 0], Zobs[:, 1], color="black", lw=2.4, label="observed", zorder=5)
        for z in Zsim:
            ax.plot(z[:, 0], z[:, 1], color="#4C78A8", alpha=0.18, lw=0.9)
        ax.scatter(Zobs[0, 0], Zobs[0, 1], marker="*", s=90, color="#F58518", zorder=6)
        ax.set_title(f"worm {data.worm_id} | frame {frame} | {selected}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_aspect("equal", adjustable="datalim")
    axes[-1].axis("off")
    handles = [
        plt.Line2D([0], [0], color="black", lw=2.4, label="observed held-out trajectory"),
        plt.Line2D([0], [0], color="#4C78A8", alpha=0.5, label=f"{selected} free-running (30 seeds)"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#F58518", markersize=10, label="initial state"),
    ]
    axes[-1].legend(handles=handles, frameon=False, loc="center", fontsize=8)
    fig.suptitle("Observed and simulated low-dimensional trajectories (PCA fit on training frames only)", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "trajectory_pca.png", bbox_inches="tight")
    plt.close(fig)


def model_figure(reports: Path) -> None:
    df = pd.read_csv(reports / "rollout_metrics.csv")
    order = ["persistence", "var", "mlp", "mlp_residual", "gru", "latent_mlp", "latent_knn"]
    means = df.groupby("kind")[["r2", "delta_mse_vs_persistence", "ood_fraction"]].mean().reindex(order)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].bar(means.index, means["r2"], color="#4C78A8")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("Free-running R$^2$ (test)")
    axes[0].tick_params(axis="x", rotation=40)
    axes[1].bar(means.index, means["delta_mse_vs_persistence"], color="#F58518")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("MSE minus constant-state baseline")
    axes[1].tick_params(axis="x", rotation=40)
    axes[2].bar(means.index, means["ood_fraction"], color="#54A24B")
    axes[2].set_title("Fraction outside training support")
    axes[2].tick_params(axis="x", rotation=40)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("One-step fit does not guarantee multi-step dynamical validity", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", bbox_inches="tight")
    plt.close(fig)


def branching_figure(reports: Path) -> None:
    ev = pd.read_csv(reports / "branching_events.csv")
    summary = json.loads((reports / "branching_summary.json").read_text(encoding="utf-8"))["gate_3"]["worms"]
    worms = sorted(ev.worm_id.unique())
    obs = [ev[ev.worm_id == w].observed_entropy.mean() for w in worms]
    sim = [ev[ev.worm_id == w].simulated_entropy.mean() for w in worms]
    pers = [next(s["mean_entropy_persistence"] for s in summary if s["worm_id"] == w) for w in worms]
    marg = [next(s["mean_entropy_marginal"] for s in summary if s["worm_id"] == w) for w in worms]
    tv_sim = [ev[ev.worm_id == w].tv_sim_observed.mean() for w in worms]
    tv_pers = [ev[ev.worm_id == w].tv_persistence_observed.mean() for w in worms]
    tv_marg = [ev[ev.worm_id == w].tv_marginal_observed.mean() for w in worms]
    x = np.arange(len(worms))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    width = 0.2
    for i, (vals, label, color) in enumerate(
        [(obs, "observed", "black"), (sim, "simulated", "#4C78A8"), (pers, "persistence", "#F58518"), (marg, "marginal", "#54A24B")]
    ):
        axes[0].bar(x + (i - 1.5) * width, vals, width, label=label, color=color)
    axes[0].set_xticks(x, [f"worm {w}" for w in worms])
    axes[0].set_ylabel("mean local outcome entropy (bits)")
    axes[0].set_title("Local outcome diversity")
    axes[0].legend(frameon=False, fontsize=7)
    for i, (vals, label, color) in enumerate([(tv_sim, "simulated", "#4C78A8"), (tv_pers, "persistence", "#F58518"), (tv_marg, "marginal", "#54A24B")]):
        axes[1].bar(x + (i - 1) * width, vals, width, label=label, color=color)
    axes[1].set_xticks(x, [f"worm {w}" for w in worms])
    axes[1].set_ylabel("TV distance to observed")
    axes[1].set_title("Conditional outcome-distribution error")
    axes[1].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Simulated reversals are not better than null baselines (Gate 3 fail)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "branching.png", bbox_inches="tight")
    plt.close(fig)


def perturbation_figure(reports: Path) -> None:
    spec = pd.read_csv(reports / "perturbation_specificity.csv")
    dose = pd.read_csv(reports / "perturbation_dose.csv")
    controls = pd.read_csv(reports / "perturbation_controls.csv")
    summary = json.loads((reports / "perturbation_summary.json").read_text(encoding="utf-8"))["gate_5"]
    top = spec.sort_values("abs_mean_delta_tv", ascending=False).head(8).iloc[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    labels = [f"{r.target} {r.direction}" for r in top.itertuples()]
    axes[0].barh(labels, top.mean_delta_turn, color="#4C78A8")
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_xlabel("mean delta P(TURN)")
    axes[0].set_title("Named-neuron effects (selected model)")
    axes[1].hist(controls.delta_tv.abs(), bins=20, color="#B279A2", alpha=0.75, label="random neurons")
    axes[1].axvline(summary["random_control_q95_abs_delta_tv"], color="black", ls="--", label="95th percentile")
    axes[1].set_xlabel("|delta TV|")
    axes[1].set_ylabel("count")
    axes[1].set_title("Specificity control")
    axes[1].legend(frameon=False, fontsize=7)
    for target in ["AIBL", "AIBR", "AVEL", "RIVL"]:
        sub = dose[dose.target == target].groupby("amplitude").delta_turn.mean()
        axes[2].plot(sub.index, sub.values, marker="o", label=target)
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_xlabel("signed amplitude (SD)")
    axes[2].set_ylabel("mean delta P(TURN)")
    axes[2].set_title("Dose response")
    axes[2].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Virtual perturbations are model-dependent and not specific vs random controls", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "perturbation.png", bbox_inches="tight")
    plt.close(fig)


def latent_figure(reports: Path) -> None:
    df = pd.read_csv(reports / "latent_metrics.csv")
    agg = df.groupby(["kind", "encoder"])[["latent_energy_distance", "path_length_ratio", "curvature_difference"]].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    labels = [f"{r.kind}\n{r.encoder}" for r in agg.itertuples()]
    axes[0].bar(labels, agg.latent_energy_distance, color="#4C78A8")
    axes[0].set_title("Latent energy distance")
    axes[1].bar(labels, agg.path_length_ratio, color="#F58518")
    axes[1].axhline(1.0, color="black", ls="--", lw=0.8)
    axes[1].set_title("Simulated / observed path length")
    axes[2].bar(labels, agg.curvature_difference, color="#54A24B")
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_title("Curvature difference (rad)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Latent-space comparison: PCA has an inverse; BunDLe is encoder-only", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "latent.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _setup()
    cfg = ExperimentConfig.load(PROJECT_ROOT / "configs/experiments/blocked_suite.yaml")
    reports = PROJECT_ROOT / "reports/simulator"
    trajectory_figure(cfg, reports)
    model_figure(reports)
    branching_figure(reports)
    perturbation_figure(reports)
    latent_figure(reports)
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
