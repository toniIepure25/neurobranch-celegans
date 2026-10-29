"""Render the Phase 9 reliability distribution figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "reports" / "phase9" / "reliability_by_target.csv"
OUT = ROOT / "reports" / "phase9" / "reliability_distribution.png"


def main() -> int:
    if not TABLE.exists():
        print(f"missing {TABLE}; run scripts/phase9_reliability.py first")
        return 2
    table = pd.read_csv(TABLE)
    repeat = table["repeat_reliability"].dropna().to_numpy()
    animal = table["single_animal_reliability"].dropna().to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, values, label, colour in (
        (axes[0], repeat, "Repeat stimulation (within animal)", "#2b6cb0"),
        (axes[1], animal, "Single animal (held out)", "#c05621"),
    ):
        axis.hist(values, bins=np.linspace(0, 1, 26), color=colour, alpha=0.85, edgecolor="white")
        axis.axvline(0.5, color="black", linestyle="--", linewidth=1, label="adequate = 0.5")
        axis.axvline(float(values.mean()), color="black", linestyle=":", linewidth=1, label=f"mean = {values.mean():.3f}")
        axis.set_xlim(0, 1)
        axis.set_xlabel("Reliability")
        axis.set_title(f"{label}\nn = {values.size} targets")
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Targets")
    fig.suptitle("Phase 9 reliability ceilings, Randi optogenetic atlas", fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
