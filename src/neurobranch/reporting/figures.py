"""Static and interactive figures built only from saved artifacts."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

from neurobranch.config import PROJECT_ROOT  # noqa: E402
from neurobranch.data.kato import MACRO_ORDER, STATE_ORDER, STATE_TO_MACRO  # noqa: E402

FIG = PROJECT_ROOT / "figures"
RUNS = PROJECT_ROOT / "artifacts" / "runs"
COLORS = {"FWD": "#2166AC", "REV": "#B2182B", "TURN": "#E69F00", "NOSTATE": "#777777"}
FINE_COLORS = {
    "dt": "#F0E442",
    "fwd": "#0072B2",
    "nostate": "#999999",
    "rev1": "#D55E00",
    "rev2": "#CC79A7",
    "revsus": "#009E73",
    "slow": "#56B4E9",
    "vt": "#E69F00",
}
plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "font.size": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
    }
)


def _save(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def _rollout_frame(df: pd.DataFrame, h: int = 3) -> pd.Series:
    def parse(s):
        try:
            return ast.literal_eval(s).get(str(h))
        except Exception:
            return np.nan

    return df.get("rollout_mse", pd.Series(index=df.index)).map(parse)


def plot_dataset() -> Path:
    p = PROJECT_ROOT / "reports" / "recording_summary.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    d = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1))
    x = np.arange(len(d))
    bottom = np.zeros(len(d))
    for state in MACRO_ORDER:
        vals = np.array([ast.literal_eval(s).get(state, 0) for s in d.macro_counts])
        frac = vals / np.array(d.n_frames)
        axes[0].bar(x, frac, bottom=bottom, label=state, color=COLORS[state], width=0.72)
        bottom += frac
    axes[0].set_xticks(x, [f"worm {i}" for i in x])
    axes[0].set_ylabel("fraction of recording")
    axes[0].set_title("Macro-behaviour composition (all frames)")
    axes[0].legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    axes[1].bar(x - 0.18, d.n_identified, width=0.36, color="#333333", label="named cells")
    axes[1].bar(x + 0.18, d.n_neurons_total, width=0.36, color="#BBBBBB", label="segmented traces")
    axes[1].set_xticks(x, [f"worm {i}" for i in x])
    axes[1].set_ylabel("count")
    axes[1].set_title("Coverage is partial (302 cells total)")
    axes[1].legend(frameon=False)
    return _save(fig, "fig01_dataset_coverage.png")


def plot_model_comparison() -> Path:
    p = RUNS / "blocked_suite" / "suite_metrics.csv"
    d = pd.read_csv(p)
    agg = d.groupby("model").agg(
        r2=("one_step_r2", "mean"),
        r2sd=("one_step_r2", "std"),
        acc=("behavior_accuracy", "mean"),
        tv=("transition_total_variation", "mean"),
    )
    order = agg.sort_values("r2", ascending=False).index.tolist()
    agg = agg.loc[order]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), sharey=True)
    y = np.arange(len(agg))
    axes[0].barh(y, agg.r2, xerr=agg.r2sd, color="#4C72B0", error_kw={"lw": 0.8})
    axes[0].axvline(0, color="black", lw=0.7)
    axes[0].set_yticks(y, agg.index)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("one-step $R^2$ (mean ± SD over 5 worms × 3 seeds)")
    axes[0].set_title("Neural prediction")
    base = d[d.model == "persistence"].behavior_persistence_accuracy.mean()
    axes[1].barh(y, agg.acc, color="#DD8452")
    axes[1].axvline(base, color="black", ls="--", lw=1, label=f"label persistence {base:.2f}")
    axes[1].set_xlabel("behaviour accuracy")
    axes[1].set_title("Decoding")
    axes[1].legend(frameon=False, loc="lower right")
    axes[2].barh(y, agg.tv, color="#55A868")
    pbase = d[d.model == "persistence"].transition_persistence_tv.mean()
    axes[2].axvline(pbase, color="black", ls="--", lw=1, label=f"persistence TV {pbase:.2f}")
    axes[2].set_xlabel("TV to observed transition distribution")
    axes[2].set_title("Transition distribution")
    axes[2].legend(frameon=False, loc="lower right")
    fig.suptitle("No dynamics model clearly beats persistence (blocked interpolation benchmark)", y=1.02)
    return _save(fig, "fig02_model_comparison.png")


def plot_forward_stress() -> Path:
    p = RUNS / "forward_stress" / "suite_metrics.csv"
    d = pd.read_csv(p)
    pivot = d.pivot_table(index="model", columns="worm_id", values="one_step_r2")
    base = pivot.loc["persistence"]
    rel = pivot.subtract(base, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 2.8))
    im = axes[0].imshow(rel.values, cmap="RdBu_r", vmin=-2, vmax=1, aspect="auto")
    axes[0].set_xticks(range(rel.shape[1]), [f"worm {c}" for c in rel.columns])
    axes[0].set_yticks(range(rel.shape[0]), rel.index)
    axes[0].set_title("$\\Delta R^2$ vs persistence (forward holdout)")
    fig.colorbar(im, ax=axes[0], fraction=0.03, pad=0.02)
    for i in range(rel.shape[0]):
        for j in range(rel.shape[1]):
            v = rel.values[i, j]
            if np.isfinite(v):
                axes[0].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5)
    acc = d.pivot_table(index="model", columns="worm_id", values="behavior_accuracy")
    im2 = axes[1].imshow(acc.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axes[1].set_xticks(range(acc.shape[1]), [f"worm {c}" for c in acc.columns])
    axes[1].set_yticks(range(acc.shape[0]), acc.index)
    axes[1].set_title("Behaviour accuracy (forward holdout)")
    fig.colorbar(im2, ax=axes[1], fraction=0.03, pad=0.02)
    fig.suptitle("Strict chronological holdout: non-stationarity breaks nonlinear forecasting", y=1.03)
    return _save(fig, "fig03_forward_stress.png")


def plot_bundlenet_latent() -> Path:
    p = PROJECT_ROOT / "references" / "external" / "nc-mcm" / "precomputed_results" / "bundlenet_consistent_embedding_worm_0.npz"
    d = np.load(p, allow_pickle=True)
    Z = np.asarray(d["x"])
    b = np.asarray(d["b"]).ravel().astype(int).tolist()
    fine = [STATE_ORDER[i] for i in b]
    macro = [STATE_TO_MACRO[s] for s in fine]
    idx = np.arange(0, len(Z), 2)
    fig = plt.figure(figsize=(9.2, 4.3))
    for k, (elev, azim, title) in enumerate([(22, -60, "view A"), (12, 30, "view B")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        for s in MACRO_ORDER:
            m = np.array([x == s for x in macro])[idx]
            if m.sum():
                ax.scatter(Z[idx][m, 0], Z[idx][m, 1], Z[idx][m, 2], s=2.0, alpha=0.65, color=COLORS[s], label=s, linewidths=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_xlabel("BunDLe 1")
        ax.set_ylabel("BunDLe 2")
        ax.set_zlabel("BunDLe 3")
        ax.grid(False)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[s], markersize=6, label=s) for s in MACRO_ORDER]
    fig.legend(handles=handles, ncol=4, loc="lower center", frameon=False)
    fig.suptitle("Official BunDLe-Net 3-D artifact, worm 0 (upstream-fitted; embedding-level leakage)", y=0.98)
    fig.subplots_adjust(bottom=0.12)
    out = _save(fig, "fig04_bundlenet_official_3d.png")
    try:
        import plotly.graph_objects as go

        f = go.Figure()
        for s in MACRO_ORDER:
            m = np.array([x == s for x in macro])
            f.add_trace(
                go.Scatter3d(
                    x=Z[m, 0], y=Z[m, 1], z=Z[m, 2], mode="markers",
                    marker=dict(size=2, color=COLORS[s], opacity=0.7), name=s,
                )
            )
        f.update_layout(
            title="Official BunDLe-Net artifact (worm 0) — upstream-fitted, not held-out",
            scene=dict(xaxis_title="BunDLe 1", yaxis_title="BunDLe 2", zaxis_title="BunDLe 3"),
            legend=dict(orientation="h"),
        )
        f.write_html(FIG / "bundlenet_latent_3d.html", include_plotlyjs="cdn")
    except Exception:
        pass
    return out


def plot_branching() -> Path:
    p = PROJECT_ROOT / "artifacts" / "branching_worm0.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    top = d["summary"]["top_frames"]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0))
    ent = [r["entropy"] for r in top]
    axes[0].barh(range(len(top)), ent, color="#6A4C93")
    axes[0].set_yticks(range(len(top)), [str(r["frame"]) for r in top])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("local outcome entropy (bits)")
    axes[0].set_ylabel("held-out reversal frame")
    axes[0].set_title("Highest-entropy local neighbourhoods")
    cats = list(MACRO_ORDER)
    bottom = np.zeros(len(top))
    for c in cats:
        vals = np.array([r["probabilities"].get(c, 0.0) for r in top])
        axes[1].barh(range(len(top)), vals, left=bottom, color=COLORS[c], label=c)
        bottom += vals
    axes[1].set_yticks(range(len(top)), [str(r["frame"]) for r in top])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("neighbour outcome fraction")
    axes[1].set_title("Outcome mixtures at those frames")
    axes[1].legend(ncol=4, frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    perm = d.get("permutation", {})
    axes[2].bar(["observed", "shuffled null"], [perm.get("mean_entropy", np.nan), perm.get("null_mean", np.nan)], color=["#6A4C93", "#BBBBBB"])
    axes[2].set_ylabel("mean entropy (bits)")
    axes[2].set_title(f"No excess diversity (p={perm.get('p_value', float('nan')):.2f})")
    axes[2].text(0.5, 0.9, "projection-sensitive:\nSpearman 0.71–0.76 across dims", ha="center", va="top", transform=axes[2].transAxes, fontsize=7)
    fig.suptitle("Branching in a 3-D projection is not evidence of a bifurcation", y=1.02)
    return _save(fig, "fig05_branching_worm0.png")


def plot_interventions() -> Path:
    cand = pd.read_csv(PROJECT_ROOT / "artifacts" / "intervention_search_worm0.csv")
    rnd = pd.read_csv(PROJECT_ROOT / "artifacts" / "intervention_random_controls_worm0.csv")
    top = cand.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    y = np.arange(len(top))
    ax.errorbar(top.delta_prob, y, xerr=[top.delta_prob - top.ci_low, top.ci_high - top.delta_prob], fmt="o", color="#C1121F", capsize=2, label="named candidate")
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(y, [f"{r.candidate} {r.amplitude:+.0f} SD" for r in top.itertuples()])
    ax.set_xlabel("virtual change in P(TURN)")
    ax.set_title("Named-neuron effects are within the random-neuron control range")
    rnd_y = -1.2
    ax.scatter(rnd.delta_prob, np.full(len(rnd), rnd_y), marker="s", color="#777777", s=16, label="random neuron (+2 SD)")
    ax.axhline(-0.6, color="#CCCCCC", lw=0.6)
    ax.set_ylim(-1.8, len(top) - 0.3)
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.99, 0.02, "MODEL-DEPENDENT — NOT CAUSALLY VALIDATED", ha="right", va="bottom", transform=ax.transAxes, fontsize=7, color="#C1121F")
    return _save(fig, "fig06_intervention_forest.png")


def plot_ncmcm() -> Path:
    rows = json.loads((PROJECT_ROOT / "artifacts" / "reproduction_ncmcm.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    for s in rows:
        axes[0].plot(np.arange(1, 21), s["markov_p_curve"], marker="o", ms=2.5, label=f"worm {s['worm_id']}")
    axes[0].axhline(0.05, color="black", ls="--", lw=0.8)
    axes[0].axvline(3, color="#999999", lw=0.7, ls=":")
    axes[0].set_xlabel("number of cognitive states K")
    axes[0].set_ylabel("Markovianity p-value")
    axes[0].set_title("Distributed p-value curves")
    axes[0].legend(frameon=False, ncol=2, fontsize=6.5)
    x = np.arange(len(rows))
    axes[1].plot(x, [s["markov_p_precomputed"] for s in rows], "o", label="precomputed")
    axes[1].plot(x, [s["markov_p_reimplemented"] for s in rows], "x", ms=8, label="Python reimplementation")
    axes[1].set_xticks(x, [f"worm {s['worm_id']}" for s in rows])
    axes[1].set_ylabel("p at K=3")
    axes[1].set_title("Surrogate test port agrees closely")
    axes[1].legend(frameon=False)
    return _save(fig, "fig07_ncmcm_verification.png")


def plot_observed_rollout() -> Path:
    from sklearn.decomposition import PCA

    from neurobranch.config import ExperimentConfig
    from neurobranch.dynamics.checkpointing import load_model
    from neurobranch.dynamics.training import load_worm_data
    from neurobranch.preprocessing import split_starts

    cfg = ExperimentConfig.load(PROJECT_ROOT / "configs" / "experiments" / "blocked_suite.yaml")
    data = load_worm_data(cfg)[0]
    model, _ = load_model(RUNS / "blocked_suite" / "mlp__worm0__seed17" / "model.pt")
    pca = PCA(3, random_state=0).fit(data.X[split_starts(data, "train", history=1, horizon=1)])
    Z = pca.transform(data.X)
    test = np.flatnonzero(data.masks["test"])
    start = int(test[len(test) // 2])
    seq = model.rollout(data.X[start], 20, seed=17, noise_std=0.04)
    Zs = pca.transform(seq)
    seg = test[(test > start - 120) & (test < start + 120)]
    fig = plt.figure(figsize=(5.4, 4.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(Z[seg, 0], Z[seg, 1], Z[seg, 2], color="#222222", lw=1, alpha=0.5)
    ax.scatter(Z[seg, 0], Z[seg, 1], Z[seg, 2], c=[COLORS[MACRO_ORDER[int(i)]] for i in data.y_macro[seg]], s=5, alpha=0.7)
    ax.plot(Zs[:, 0], Zs[:, 1], Zs[:, 2], color="#6A4C93", lw=2.5, label="model rollout")
    ax.scatter(Zs[1:, 0], Zs[1:, 1], Zs[1:, 2], color="#6A4C93", marker="^", s=12)
    ax.scatter([Z[start, 0]], [Z[start, 1]], [Z[start, 2]], color="black", s=40, label="initial state")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Observed (points) vs open-loop model rollout (purple)")
    ax.legend(frameon=False, loc="upper left")
    ax.text2D(0.02, 0.02, "Prediction, not data", transform=ax.transAxes, fontsize=7, color="#6A4C93")
    return _save(fig, "fig08_observed_vs_rollout.png")


def make_all() -> list[Path]:
    out = []
    for fn in [
        plot_dataset,
        plot_model_comparison,
        plot_forward_stress,
        plot_bundlenet_latent,
        plot_branching,
        plot_interventions,
        plot_ncmcm,
        plot_observed_rollout,
    ]:
        try:
            out.append(fn())
        except Exception as exc:  # keep figure generation auditable, not silent
            print(f"[figure skipped] {fn.__name__}: {exc}")
    return out
