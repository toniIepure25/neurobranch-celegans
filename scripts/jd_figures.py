"""Scientific figures for the joint-dynamics experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from neurobranch.joint_dynamics.evaluation import autocorrelation_profile, pooled_neural_covariance

OUT = Path("reports/simulator/joint_dynamics")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "figure.dpi": 170})
LABELS = {
    "J0_persistence": "persistence",
    "J1_independent_diag": "independent AR",
    "J2_population_diag": "population VAR (diag)",
    "J3_population_full": "population VAR (full)",
    "J4_independent_full": "independent AR (full)",
    "J5_population_lead1_boot": "population lead-1",
}
COLORS = {
    "J0_persistence": "#333333",
    "J1_independent_diag": "#2f6f9f",
    "J2_population_diag": "#b5522f",
    "J3_population_full": "#7b5aa6",
    "J4_independent_full": "#4b8f73",
    "J5_population_lead1_boot": "#d4a017",
}


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


def score_figure() -> None:
    s = pd.read_csv(OUT / "score_summary.csv").set_index("model")
    models = [m for m in LABELS if m in s.index]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1))
    axes[0].barh([LABELS[m] for m in models], [s.loc[m, "energy_h20"] for m in models], color=[COLORS[m] for m in models])
    axes[0].set_xlabel("H20 energy score (lower better)")
    axes[0].set_title("Multivariate state score")
    axes[0].invert_yaxis()
    axes[1].barh([LABELS[m] for m in models], [s.loc[m, "trajectory_energy_h20"] for m in models], color=[COLORS[m] for m in models])
    axes[1].set_xlabel("H20 trajectory energy (lower better)")
    axes[1].set_title("Joint trajectory score")
    axes[1].invert_yaxis()
    _save(fig, "jd_scores.png")


def calibration_figure() -> None:
    c = pd.read_csv(OUT / "coverage_by_horizon.csv")
    s = pd.read_csv(OUT / "score_summary.csv").set_index("model")
    models = ["J0_persistence", "J1_independent_diag", "J2_population_diag", "J5_population_lead1_boot"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for model in models:
        g = c[c.model == model].sort_values("horizon")
        for level, style in ((50, ":"), (80, "--"), (95, "-")):
            axes[0].plot(g.horizon, g[f"coverage_{level}"], style, marker="o", ms=2.5, lw=1, color=COLORS[model], label=f"{LABELS[model]} {level}%")
    for level in (50, 80, 95):
        axes[0].axhline(level / 100, color="k", lw=0.5, alpha=0.3)
    axes[0].set_xscale("log")
    axes[0].set_xticks([1, 5, 10, 20, 50, 100], labels=["1", "5", "10", "20", "50", "100"])
    axes[0].set_xlabel("horizon (frames)")
    axes[0].set_ylabel("empirical central coverage")
    axes[0].set_title("Calibration by horizon (10-D PCA)")
    axes[0].legend(fontsize=5.0, ncol=2, loc="lower left")
    x = np.arange(len(models))
    for i, level in enumerate((50, 80, 95)):
        axes[1].bar(x + (i - 1) * 0.25, [s.loc[m, f"coverage_h20_full_{level}"] for m in models], width=0.23, label=f"{level}%")
        axes[1].axhline(level / 100, color="k", lw=0.5, ls=":", alpha=0.5)
    axes[1].set_xticks(x, [LABELS[m] for m in models], rotation=18, ha="right")
    axes[1].set_ylim(0.4, 1.0)
    axes[1].set_ylabel("coverage, full neuron space")
    axes[1].set_title("H20 marginal calibration")
    axes[1].legend(fontsize=6)
    _save(fig, "jd_calibration.png")


def covariance_figure() -> None:
    z = np.load(OUT / "trajectory_examples.npz")
    keys = [k for k in z.files if not k.startswith("truth_")]
    truth = np.concatenate([z[k] for k in z.files if k.startswith("truth_")], axis=0)
    truth_cov = pooled_neural_covariance(truth)
    models = ["J1_independent_diag", "J5_population_lead1_boot"]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
    mats = [truth_cov] + [pooled_neural_covariance(np.concatenate([z[k] for k in keys if k.endswith(m)], axis=0)) for m in models]
    titles = ["observed", LABELS[models[0]], LABELS[models[1]]]
    for ax, mat, title in zip(axes, mats, titles):
        corr = mat / np.sqrt(np.outer(np.diag(mat), np.diag(mat)))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("PC")
        ax.set_ylabel("PC")
    fig.colorbar(im, ax=axes, shrink=0.8, label="correlation")
    _save(fig, "jd_covariance.png")


def autocorr_figure() -> None:
    z = np.load(OUT / "trajectory_examples.npz")
    keys = [k for k in z.files if not k.startswith("truth_")]
    truth = np.concatenate([z[k] for k in z.files if k.startswith("truth_")], axis=0)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(range(1, 11), autocorrelation_profile(truth), "k", marker="o", ms=3, label="observed")
    for model in ("J1_independent_diag", "J5_population_lead1_boot"):
        ens = np.concatenate([z[k] for k in keys if k.endswith(model)], axis=0)
        ax.plot(range(1, 11), autocorrelation_profile(ens), marker="o", ms=3, color=COLORS[model], label=LABELS[model])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("lag (frames)")
    ax.set_ylabel("mean autocorrelation")
    ax.set_title("Temporal structure of simulated trajectories")
    ax.legend(fontsize=7)
    _save(fig, "jd_autocorrelation.png")


def pit_figure() -> None:
    c = pd.read_csv(OUT / "calibration_curves.csv")
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for model in ("J1_independent_diag", "J2_population_diag", "J5_population_lead1_boot"):
        g = c[c.model == model].sort_values("bin")
        ax.plot(np.arange(10) / 10 + 0.05, g.density, marker="o", ms=3, color=COLORS[model], label=LABELS[model])
    ax.axhline(0.1, color="k", ls="--", lw=0.7, label="uniform")
    ax.set_xlabel("PIT bin")
    ax.set_ylabel("density")
    ax.set_title("PIT, H20, fixed 10-D PCA")
    ax.legend(fontsize=6)
    _save(fig, "jd_pit.png")


def factorial_figure() -> None:
    f = pd.read_csv(OUT / "factorial_decomposition.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x = np.arange(len(f))
    ax.bar(x - 0.18, f.mean_main_effect_positive_is_better, width=0.34, label="population mean effect")
    ax.bar(x + 0.18, f.covariance_main_effect_positive_is_better, width=0.34, label="full-covariance effect")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, ["H20 state energy", "H20 trajectory energy"], rotation=8)
    ax.set_ylabel("improvement (positive better)")
    ax.set_title("Factorial decomposition: mean vs covariance")
    ax.legend(fontsize=6)
    _save(fig, "jd_factorial.png")


def parameter_figure() -> None:
    real = pd.read_csv(OUT / "parameter_summary.csv").iloc[0]
    base = pd.read_csv(OUT / "score_summary.csv").set_index("model").loc["J2_population_diag"]
    synth = pd.read_csv(OUT / "parameter_synthetic_audit.csv").set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1))
    x = np.arange(3)
    axes[0].bar(x - 0.18, [base.coverage_h20_full_50, base.coverage_h20_full_80, base.coverage_h20_full_95], width=0.34, label="population diag")
    axes[0].bar(x + 0.18, [real.coverage_h20_50, real.coverage_h20_80, real.coverage_h20_95], width=0.34, label="+ parameter ensemble")
    axes[0].axhline(0, color="k", lw=0.5)
    for _i, level in enumerate((50, 80, 95)):
        axes[0].axhline(level / 100, color="k", lw=0.5, ls=":", alpha=0.4)
    axes[0].set_xticks(x, ["50%", "80%", "95%"])
    axes[0].set_ylabel("coverage (PCA)")
    axes[0].set_title("Kato test: parameter uncertainty")
    axes[0].legend(fontsize=6)
    idx = ["innovation_only", "parameter_ensemble", "oracle"]
    axes[1].bar(np.arange(3) - 0.18, synth.loc[idx, "coverage_95"], width=0.34, label="95%")
    axes[1].bar(np.arange(3) + 0.18, synth.loc[idx, "coverage_50"], width=0.34, label="50%")
    axes[1].set_xticks(np.arange(3), ["innovation", "parameters", "oracle"], rotation=10)
    axes[1].set_ylim(0.35, 1.0)
    axes[1].set_title("Synthetic audit")
    axes[1].legend(fontsize=6)
    _save(fig, "jd_parameter.png")


def trajectory_figure() -> None:
    z = np.load(OUT / "trajectory_examples.npz")
    key = "0_2690_J5_population_lead1_boot"
    obs = z["truth_0_2690"]
    ens = z[key]
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    steps = np.arange(1, 21)
    for m in range(min(40, ens.shape[0])):
        ax.plot(steps, ens[m, :, 0], color="#d4a017", alpha=0.10, lw=0.6)
    ax.fill_between(steps, np.quantile(ens[:, :, 0], 0.05, axis=0), np.quantile(ens[:, :, 0], 0.95, axis=0), color="#d4a017", alpha=0.20, label="5-95% ensemble")
    ax.plot(steps, np.median(ens[:, :, 0], axis=0), color="#8a5a00", lw=1.4, label="ensemble median")
    ax.plot(steps, obs[:, 0], "k", lw=1.4, label="observed")
    ax.set_xlabel("step after landmark (frames)")
    ax.set_ylabel("PC1 activity (training-scaled)")
    ax.set_title("Held-out trajectory, session 0, landmark 2690")
    ax.legend(fontsize=6)
    _save(fig, "jd_example_trajectory.png")


def interactive() -> None:
    z = np.load(OUT / "trajectory_examples.npz")
    key = "0_2690"
    obs = z[f"truth_{key}"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(1, 21), y=obs[:, 0], name="observed PC1", line={"color": "black", "width": 3}))
    for model in ("J0_persistence", "J1_independent_diag", "J2_population_diag", "J5_population_lead1_boot"):
        ens = z[f"{key}_{model}"]
        med = np.median(ens[:, :, 0], axis=0)
        fig.add_trace(go.Scatter(x=np.arange(1, 21), y=med, name=LABELS[model], visible=model == "J5_population_lead1_boot"))
    buttons = []
    for model in ("J0_persistence", "J1_independent_diag", "J2_population_diag", "J5_population_lead1_boot"):
        visible = [True] + [m == model for m in ("J0_persistence", "J1_independent_diag", "J2_population_diag", "J5_population_lead1_boot")]
        buttons.append({"label": LABELS[model], "method": "update", "args": [{"visible": visible}]})
    fig.update_layout(
        title="Joint neural dynamics: held-out trajectory, session 0, landmark 2690",
        xaxis_title="step after landmark (frames)",
        yaxis_title="PC1 activity (training-scaled)",
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 1.02, "y": 1.0}],
        template="plotly_white",
    )
    fig.write_html(FIG / "jd_trajectories.html", include_plotlyjs="cdn")
    meta = {
        "session": 0,
        "landmark": 2690,
        "selection_rule": "first prespecified test landmark in session 0; not selected by score",
        "component": "PC1 of the fixed training PCA",
        "members": int(z[f"{key}_J5_population_lead1_boot"].shape[0]),
    }
    (OUT / "example_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    score_figure()
    calibration_figure()
    covariance_figure()
    autocorr_figure()
    pit_figure()
    factorial_figure()
    parameter_figure()
    trajectory_figure()
    interactive()
    rows = []
    for path in sorted(FIG.glob("*")):
        rows.append({"figure": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (OUT / "figure_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("figures", len(rows))


if __name__ == "__main__":
    main()
