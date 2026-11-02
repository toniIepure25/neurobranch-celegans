"""Figures for the Kato common-state control audit."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from neurobranch.kato_interactions.metrics import session_block_ci

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "simulator" / "kato_common_state"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
LABELS = {
    "B0_persistence": "B0 persistence",
    "B1_matched_own": "B1m own AR",
    "B2_own_trend": "B2 own + drift",
    "B4_matched_own_500": "B4m own AR (500)",
    "B5_common_k10": "B5 common factors (10)",
    "B3_population": "B3 population VAR",
}


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=160, metadata={"Software": "neurobranch"})
    plt.close(fig)


def _energy_ci(frame: pd.DataFrame, model: str) -> tuple[float, float, float]:
    sub = frame[frame["model"] == model]
    ci = session_block_ci(sub, "pca10_energy_h20", block=5)
    return ci["mean"], ci["lo"], ci["hi"]


def main() -> None:
    frame = pd.read_csv(OUT / "model_per_landmark.csv")
    order = [
        "B0_persistence",
        "B1_matched_own",
        "B2_own_trend",
        "B4_matched_own_500",
        "B5_common_k10",
        "B3_population",
    ]
    means, lows, highs = [], [], []
    for model in order:
        mean, lo, hi = _energy_ci(frame, model)
        means.append(mean)
        lows.append(mean - lo)
        highs.append(hi - mean)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.barh([LABELS[m] for m in order], means, xerr=[lows, highs], color="#4c72b0", alpha=0.85)
    for y, value in enumerate(means):
        ax.text(value + 0.03, y, f"{value:.2f}", va="center", fontsize=9)
    ax.set_xlabel("H20 multivariate energy (lower is better)")
    ax.set_title("Matched control ladder, Kato WT_NoStim")
    ax.invert_yaxis()
    _save(fig, "model_ladder.png")

    decomp = pd.read_csv(OUT / "decomposition.csv")
    labels = {
        "own_history_B1m_vs_B0": "own history\n(B1m vs B0)",
        "local_drift_B2_vs_B1m": "drift\n(B2 vs B1m)",
        "window_B4m_vs_B1m": "window\n(B4m vs B1m)",
        "common_factors_B5_vs_B4m": "common factors\n(B5 vs B4m)",
        "residual_cross_B3_vs_B5": "residual cross-neuron\n(B3 vs B5)",
        "population_B3_vs_B4m": "population\n(B3 vs B4m)",
    }
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    values = decomp["mean_improvement"].to_numpy()
    err = [
        values - decomp["ci_low"].to_numpy(),
        decomp["ci_high"].to_numpy() - values,
    ]
    ax.barh([labels.get(v, v) for v in decomp["contrast"]], values, xerr=err, color="#55a868", alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("mean paired H20 energy improvement")
    ax.set_title("Decomposition of the population advantage")
    ax.invert_yaxis()
    _save(fig, "decomposition.png")

    session = pd.read_csv(OUT / "session_effects.csv")
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.bar([f"session {int(s)}" for s in session["session"]], session["mean_energy_improvement"], color="#c44e52")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("B3 - B5 H20 energy")
    ax.set_title("Residual cross-neuron advantage by session")
    _save(fig, "session_effects.png")

    surrogate_path = OUT / "surrogate_summary.csv"
    if surrogate_path.exists():
        surrogate = pd.read_csv(surrogate_path)
        labels = {
            "circular_full": "circular full",
            "block_full": "block full",
            "common_preserved_residual_shift": "common preserved,\nresidual shifted",
            "residual_preserved_factor_shift": "residual preserved,\nfactor shifted",
        }
        fig, ax = plt.subplots(figsize=(7.4, 3.4))
        ax.bar(
            [labels.get(m, m) for m in surrogate["method"]],
            surrogate["delta_vs_B3"],
            color="#8172b3",
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("surrogate - observed B3 energy")
        ax.set_title("Causal-window surrogate controls")
        _save(fig, "surrogate_controls.png")

    synthetic_path = OUT / "synthetic_summary.csv"
    if synthetic_path.exists():
        synthetic = pd.read_csv(synthetic_path)
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        ax.bar(synthetic["scenario"], synthetic["b3_minus_b5_3_mse_gain"], color="#937860")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("B3 - B5(3) H20 MSE gain")
        ax.set_title("Synthetic specificity check (positive = B3 better)")
        ax.tick_params(axis="x", rotation=20)
        _save(fig, "synthetic_controls.png")


if __name__ == "__main__":
    main()
