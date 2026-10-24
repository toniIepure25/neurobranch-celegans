"""Matched random-neuron feature ablation, run separately from the controls."""

from __future__ import annotations

from pathlib import Path

from neurobranch.behavior.turn_direction import build_designs, circuit_ablation
from neurobranch.data.kramer import load_all

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "phase4"


def main() -> None:
    recs, _ = load_all(cohorts=("wt_octanol", "wt_diacetyl"))
    designs, labels = build_designs(recs, window="mid")
    raw, summary = circuit_ablation(designs, labels, seed=17, n_random=24)
    raw.to_csv(REPORT / "turn_direction_neuron_ablation_raw.csv", index=False)
    summary.to_csv(REPORT / "turn_direction_neuron_ablation.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
