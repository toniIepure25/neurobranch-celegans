"""Static summary figures for the generative-failure diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "reports/simulator/failure_diagnostics"
OUT = DIAG / "figures"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    agg = pd.read_csv(DIAG / "forecastability/forecast_aggregate.csv")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for model, group in agg.groupby("model"):
        if model == "persistence":
            continue
        ax.plot(group.horizon, group.mean_skill, marker="o", label=model)
    ax.axhline(0, color="black", linestyle="--", linewidth=1, label="persistence")
    ax.set_xlabel("forecast horizon (frames)")
    ax.set_ylabel("mean skill vs persistence")
    ax.set_title("No model beats persistence on average")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "forecast_skill.png", dpi=150)
    plt.close(fig)

    ident = pd.read_csv(DIAG / "identifiability/identifiability_summary.csv")
    ident_json = json.loads((DIAG / "identifiability/identifiability_summary.json").read_text(encoding="utf-8"))
    bootstrap = {
        int(item["worm_id"]): item["subspace"]["bootstrap_mean_angle_deg"]
        for item in ident_json["results"]
    }
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(ident.worm_id - 0.18, ident.subspace_mean_angle_deg, width=0.36, label="training halves")
    ax.bar(
        ident.worm_id + 0.18,
        [bootstrap[int(w)] for w in ident.worm_id],
        width=0.36,
        label="bootstrap",
    )
    ax.set_xlabel("worm")
    ax.set_ylabel("mean principal angle (deg)")
    ax.set_title("Latent subspace rotates across the recording")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "latent_subspace_stability.png", dpi=150)
    plt.close(fig)

    window = pd.read_csv(DIAG / "oxygen/oxygen_window_aggregate.csv")
    pairs = pd.read_csv(DIAG / "perturbation/pairwise_consistency.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].barh(window.model, window.mean_skill, color="#8E6C8A")
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("mean skill vs no response")
    axes[0].set_title("Oxygen event windows")
    axes[1].barh(pairs.model, pairs.mean_cosine, color="#4C78A8")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlim(-1, 1)
    axes[1].set_xlabel("mean response cosine")
    axes[1].set_title("Cross-model virtual perturbation")
    fig.tight_layout()
    fig.savefig(OUT / "oxygen_and_perturbation.png", dpi=150)
    plt.close(fig)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
