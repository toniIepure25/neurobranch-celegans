"""Render the Phase 10 metric-aligned ceiling figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CURVE = ROOT / "reports" / "phase10" / "ceiling_curve.csv"
BY_TARGET = ROOT / "reports" / "phase10" / "ceiling_curve_by_target.csv"
OUT = ROOT / "reports" / "phase10" / "ceiling_curve.png"


def main() -> int:
    if not CURVE.exists() or not BY_TARGET.exists():
        print("missing Phase 10 curve artefacts; run scripts/phase10_reliability.py first")
        return 2
    curve = pd.read_csv(CURVE)
    by_target = pd.read_csv(BY_TARGET)
    single = by_target.loc[by_target["n_eval"] == 1, "reliability"].dropna().to_numpy()
    fold = by_target.loc[by_target["n_eval"] == 22, "reliability"].dropna().to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axis = axes[0]
    axis.fill_between(curve["n_eval"], curve["ci_low"], curve["ci_high"], color="#2b6cb0", alpha=0.2, label="95% target-level CI")
    axis.plot(curve["n_eval"], curve["mean_reliability"], color="#2b6cb0", marker="o", label="Mean reliability")
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="adequate = 0.5")
    axis.axvline(22, color="#c05621", linestyle=":", linewidth=1.2, label="Phase 6-8 fold size (22)")
    axis.set_xscale("log")
    axis.set_xticks([1, 2, 5, 10, 22, 55, 110])
    axis.set_xticklabels(["1", "2", "5", "10", "22", "55", "110"])
    axis.set_xlabel("Animals pooled per profile")
    axis.set_ylabel("Reliability")
    axis.set_ylim(0, 1)
    axis.set_title("Reliability rises with evaluation animals")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1]
    bins = np.linspace(0, 1, 26)
    axis.hist(single, bins=bins, color="#c05621", alpha=0.75, label=f"single animal (mean {single.mean():.3f})")
    axis.hist(fold, bins=bins, color="#2b6cb0", alpha=0.55, label=f"22 animals (mean {fold.mean():.3f})")
    axis.axvline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Reliability")
    axis.set_ylabel("Targets")
    axis.set_title("The ceiling depends on the estimand")
    axis.legend(frameon=False, fontsize=8)

    fig.suptitle("Phase 10 metric-aligned ceilings, Randi optogenetic atlas", fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
