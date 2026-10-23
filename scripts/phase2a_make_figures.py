"""Build the Phase 2 summary figure from committed report tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BR = ROOT / "reports" / "phase2a_scientific_audit" / "branching"
IV = ROOT / "reports" / "phase2a_scientific_audit" / "intervention"
BE = ROOT / "reports" / "phase2a_scientific_audit" / "behavior"
RA = ROOT / "reports" / "phase2b_optogenetic_atlas"
OUT = ROOT / "figures" / "fig09_phase2_summary.png"

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "font.size": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
    }
)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2))

    # A. branch entropy: observed vs nulls
    b = pd.read_csv(BR / "branching_all_worms_long.csv")
    x = np.arange(5)
    w = 0.26
    for i, (method, label, color) in enumerate(
        [("iid", "iid null", "#B2182B"), ("block", "block-10 null", "#E69F00")]
    ):
        sub = b[(b.null_method == method) & ((b.null_block_size == 10) | (b.null_method == "iid"))]
        sub = sub.drop_duplicates("worm")
        axes[0, 0].bar(x + (i + 1) * w, sub.null_mean, w, label=label, color=color, alpha=0.75)
    obs = b.drop_duplicates("worm")
    axes[0, 0].bar(x, obs.mean_entropy, w, label="observed", color="#2166AC")
    axes[0, 0].set_xticks(x, [f"worm {i}" for i in range(5)])
    axes[0, 0].set_ylabel("mean local outcome entropy (bits)")
    axes[0, 0].set_title("A. Branching: observed diversity is below null")
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper left")
    axes[0, 0].set_ylim(0, 1.08)

    # B. RIVL vs matched controls, MLP seed 17
    scan = pd.read_csv(IV / "scan_mlp_seed17.csv")
    ctrl = pd.read_csv(IV / "controls_mlp_seed17.csv")
    rivl = scan[(scan.candidate == "RIVL") & (scan.amplitude == 2.0)].iloc[0]
    axes[0, 1].axhline(0, color="#444", lw=0.8)
    axes[0, 1].scatter(
        np.zeros(len(ctrl)), ctrl.delta_prob, color="#777777", s=16, alpha=0.8, label="matched controls"
    )
    axes[0, 1].scatter([1], [rivl.delta_prob], color="#B2182B", s=50, zorder=3, label="RIVL +2")
    axes[0, 1].errorbar(
        [1], [rivl.delta_prob], yerr=[[rivl.delta_prob - rivl.ci_low], [rivl.ci_high - rivl.delta_prob]],
        fmt="none", ecolor="#B2182B", capsize=3,
    )
    axes[0, 1].set_xticks([0, 1], ["controls", "RIVL"])
    axes[0, 1].set_xlim(-0.5, 1.5)
    axes[0, 1].set_ylabel("ΔP(TURN)")
    axes[0, 1].set_title("B. RIVL is not specific (MLP seed 17)")
    axes[0, 1].legend(frameon=False, loc="upper left")

    # C. external optogenetic MAE
    agg = pd.read_csv(RA / "benchmark_aggregate_response.csv")
    tasks = ["heldout_events", "heldout_targets", "heldout_animals"]
    labels = ["new events", "new targets", "new animals"]
    glob = [agg[(agg.task == t) & (agg.model == "global_mean")].mae_mean.iloc[0] for t in tasks]
    conn = [agg[(agg.task == t) & (agg.model == "connectome_ridge")].mae_mean.iloc[0] for t in tasks]
    xx = np.arange(3)
    axes[1, 0].bar(xx - 0.18, glob, 0.36, label="constant mean", color="#777777")
    axes[1, 0].bar(xx + 0.18, conn, 0.36, label="connectome ridge", color="#2166AC")
    axes[1, 0].set_xticks(xx, labels)
    axes[1, 0].set_ylabel("held-out MAE")
    axes[1, 0].set_ylim(5.4, 6.2)
    axes[1, 0].set_title("C. Randi 2023: no positive transfer")
    axes[1, 0].legend(frameon=False)

    # D. behaviour incremental effect
    blocked = pd.read_csv(BE / "behavior_summary_blocked.csv").set_index("model")
    forward = pd.read_csv(BE / "behavior_summary_forward.csv").set_index("model")
    models = ["logistic_state", "logistic_state_neural"]
    names = ["state only", "state + neural"]
    xx = np.arange(2)
    axes[1, 1].bar(xx - 0.18, [blocked.loc[m, "balanced_accuracy_mean"] for m in models], 0.36, label="blocked", color="#2166AC")
    axes[1, 1].bar(xx + 0.18, [forward.loc[m, "balanced_accuracy_mean"] for m in models], 0.36, label="forward", color="#E69F00")
    axes[1, 1].set_xticks(xx, names)
    axes[1, 1].set_ylim(0.6, 1.0)
    axes[1, 1].set_ylabel("balanced accuracy")
    axes[1, 1].set_title("D. Neural activity does not improve behaviour")
    axes[1, 1].legend(frameon=False)

    fig.suptitle("Phase 2: corrected branching, intervention specificity, external optogenetics, behaviour", y=1.02)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
