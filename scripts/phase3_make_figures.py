"""Generate Phase 3 figures from committed result tables."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("reports/phase3/figures")
OUT.mkdir(parents=True, exist_ok=True)


def prospective():
    fig, ax = plt.subplots(figsize=(7, 4))
    modes = ["behavior", "behavior_neural", "behavior_neural_pca", "behavior_quadratic", "behavior_neural_quadratic"]
    labels = ["behaviour", "+ neural", "+ PCA", "quad. behaviour", "quad. + neural"]
    colors = ["#333333", "#C1121F", "#2E86AB", "#7D8491", "#E4572E"]
    for mode, lab, col in zip(modes, labels, colors):
        xs, ys = [], []
        for h in [3, 6, 12]:
            p = Path(f"reports/phase3/prospective_metrics_{h}s.csv")
            if p.exists():
                d = pd.read_csv(p)
                row = d[d["mode"] == mode]
                if len(row):
                    xs.append(h)
                    ys.append(row["log_loss"].mean())
        ax.plot(xs, ys, marker="o", label=lab, color=col)
    ax.set_xlabel("forecast horizon (s)")
    ax.set_ylabel("mean animal log loss (lower is better)")
    ax.set_xticks([3, 6, 12])
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Prospective reversal prediction, animal-disjoint")
    fig.tight_layout()
    fig.savefig(OUT / "prospective_logloss.png", dpi=160)
    plt.close(fig)

    s = json.loads(Path("reports/phase3/prospective_summary.json").read_text(encoding="utf-8"))
    keys = [
        ("delta_behavior_neural", "population"),
        ("delta_behavior_neural_pca", "PCA(10)"),
        ("delta_neural_within_quadratic", "vs quad. behaviour"),
        ("delta_behavior_neural_shift", "time-shifted null"),
        ("delta_behavior_identified", "identified 52"),
    ]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, (key, _lab) in enumerate(keys):
        d = s["6s"]["aggregate"].get(key)
        if not d:
            continue
        mean = d["mean"]
        lo, hi = d["ci95"]
        ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]], fmt="o", color="#C1121F", capsize=4)
        ax.text(hi + 0.001, i, f"{mean:+.4f}", va="center", fontsize=8)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([k[1] for k in keys], fontsize=8)
    ax.set_xlabel("Δ log loss (negative = neural helps), H=6 s")
    ax.set_title("Primary comparison and controls")
    fig.tight_layout()
    fig.savefig(OUT / "neural_delta_ci.png", dpi=160)
    plt.close(fig)


def randi():
    rel = pd.read_csv("reports/phase3/randi_reliability_by_target.csv")
    s = json.loads(Path("reports/phase3/randi_phase3_summary.json").read_text(encoding="utf-8"))
    prof = s["temporal_profile"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    names = ["peak", "mean", "integral", "per-neuron\nshape", "population\ntime-course", "amplitude\nvector"]
    vals = [
        rel.loc[rel.target == "response", "median_pairwise_r"].iloc[0],
        rel.loc[rel.target == "response_mean", "median_pairwise_r"].iloc[0],
        rel.loc[rel.target == "response_integral", "median_pairwise_r"].iloc[0],
        prof["per_neuron_temporal_r"]["median"],
        prof["population_timecourse_r"]["median"],
        prof["amplitude_vector_r"]["median"],
    ]
    ax.bar(names, vals, color=["#C1121F", "#E4572E", "#F3A712", "#2E86AB", "#6A4C93", "#7D8491"])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("median repeated-response r")
    ax.set_title("Randi reliability by target and shape statistic")
    fig.tight_layout()
    fig.savefig(OUT / "randi_reliability.png", dpi=160)
    plt.close(fig)

    avg = pd.read_csv("reports/phase3/randi_repeated_average.csv")
    d = avg[avg.target == "response"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(d["model"], d["mae"], color="#7D8491")
    ax.set_ylabel("held-out event MAE")
    ax.set_title("Repeated-event averaging vs nested means (signed peak)")
    fig.tight_layout()
    fig.savefig(OUT / "randi_repeated_average.png", dpi=160)
    plt.close(fig)


def coverage():
    c = pd.read_csv("reports/phase3/dataset_coverage.csv")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.scatter(c["event_rate_per_min"], c["reversal_fraction"], c="#2E86AB")
    ax.set_xlabel("reversals per minute")
    ax.set_ylabel("fraction of frames reversing")
    ax.set_title("Baskoylu 2026: 27 animals, structural QC")
    fig.tight_layout()
    fig.savefig(OUT / "dataset_coverage.png", dpi=160)
    plt.close(fig)


def cross_animal():
    d = pd.read_csv("reports/phase3/cross_animal_dynamics.csv")
    v = d[d.task == "next_velocity"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(v["model"], v["mae"], color=["#333333", "#2E86AB", "#C1121F"])
    ax.set_ylabel("next-frame velocity MAE")
    ax.set_title("Cross-animal one-step dynamics probe")
    fig.tight_layout()
    fig.savefig(OUT / "cross_animal_dynamics.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    prospective()
    randi()
    coverage()
    cross_animal()
    print("wrote", OUT)
