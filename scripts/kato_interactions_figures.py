"""Figures for the Kato-only interaction and sensitivity audit."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "simulator" / "kato_interactions"
FIG = OUT / "figures"
WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION \u2014 NOT CAUSALLY VALIDATED."

TEAL = "#117a65"
AMBER = "#b9770e"
RED = "#922b21"
GRAY = "#5d6d7e"
INK = "#17202a"


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def cross_neuron() -> None:
    frame = pd.read_csv(OUT / "cross_neuron_summary.csv")
    order = [
        "B0_persistence",
        "B1_independent_common",
        "B2_local_mean",
        "B4_diagonal_population",
        "B4_refit_diagonal",
        "B5_circular",
        "B5_block",
        "B3_population",
    ]
    frame = frame.set_index("model").reindex([m for m in order if m in set(frame.model)]).reset_index()
    colors = [TEAL if m == "B3_population" else GRAY for m in frame.model]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.barh(frame.model, frame.energy_h20, color=colors, alpha=0.9)
    for y, value in enumerate(frame.energy_h20):
        ax.text(value + 0.03, y, f"{value:.3f}", va="center", fontsize=8, color=INK)
    ax.set_xlabel("H20 energy score (lower is better)")
    ax.set_title("Matched-innovation cross-neuron comparison")
    ax.grid(axis="x", color="#e5e8e8")
    ax.set_axisbelow(True)
    fig.text(
        0.01,
        0.01,
        "B3 and B1 use the same J5 innovations. Scores are in the fixed 10-D PCA.",
        fontsize=7,
        color=GRAY,
    )
    save(fig, "fig_cross_neuron_energy.png")


def cross_controls() -> None:
    frame = pd.read_csv(OUT / "cross_neuron_paired.csv")
    frame = frame[(frame.metric == "energy_h20") & (frame.block == 5)]
    wanted = [
        ("B3_population", "B1_independent_common"),
        ("B3_population", "B4_refit_diagonal"),
        ("B3_population", "B5_circular"),
    ]
    rows = []
    for model, reference in wanted:
        part = frame[(frame.model == model) & (frame.reference == reference)]
        if len(part):
            rows.append(part.iloc[0])
    if not rows:
        return
    rows = pd.DataFrame(rows)
    labels = [f"{r.reference}\nminus {r.model}" for _, r in rows.iterrows()]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    y = np.arange(len(rows))
    ax.errorbar(
        rows.mean_improvement,
        y,
        xerr=[rows.mean_improvement - rows.ci_low, rows.ci_high - rows.mean_improvement],
        fmt="o",
        color=TEAL,
        ecolor=TEAL,
        capsize=4,
    )
    ax.axvline(0, color=RED, linewidth=1, linestyle="--")
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("Paired H20 energy improvement (95% session-block interval)")
    ax.set_title("Cross-neuron ablation and matched surrogate controls")
    ax.grid(axis="x", color="#e5e8e8")
    save(fig, "fig_cross_neuron_controls.png")


def surrogate_controls() -> None:
    frame = pd.read_csv(OUT / "surrogate_draws.csv")
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    data = [frame.loc[frame.method == method, "energy_h20"].to_numpy() for method in ("circular", "block")]
    ax.boxplot(data, tick_labels=["circular shift", "block permutation"], patch_artist=True,
               boxprops=dict(facecolor="#d5f5e3", color=TEAL),
               medianprops=dict(color=INK))
    ax.set_ylabel("H20 energy score")
    ax.set_title("Matched temporal-surrogate controls")
    ax.grid(axis="y", color="#e5e8e8")
    save(fig, "fig_surrogate_controls.png")


def operator_spectra() -> None:
    modes = pd.read_csv(OUT / "operator_modes.csv")
    modes = modes[modes.fit_id == "frozen_train"]
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color=GRAY, linewidth=0.8, linestyle="--")
    scatter = ax.scatter(modes.real, modes.imag, c=modes.magnitude, cmap="viridis", s=16)
    ax.set_aspect("equal")
    ax.set_xlabel("Real eigenvalue")
    ax.set_ylabel("Imaginary eigenvalue")
    ax.set_title("Frozen training operator spectra")
    fig.colorbar(scatter, ax=ax, label="|eigenvalue|")
    ax.grid(color="#e5e8e8")
    save(fig, "fig_operator_spectra.png")


def operator_stability() -> None:
    frame = pd.read_csv(OUT / "operator_stability.csv")
    frame = frame[frame.kind == "segment"]
    if frame.empty:
        return
    pivot = frame.pivot_table(index="session", columns="metric", values="cv")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(pivot.index))
    width = 0.25
    for offset, metric, color in [
        (-width, "spectral_radius", TEAL),
        (0, "leading_decay_frames", AMBER),
        (width, "nonnormality", RED),
    ]:
        if metric in pivot:
            ax.bar(x + offset, pivot[metric], width=width, label=metric.replace("_", " "), color=color, alpha=0.85)
    ax.set_xticks(x, [f"session {int(i)}" for i in pivot.index], rotation=0)
    ax.set_ylabel("Coefficient of variation across training segments")
    ax.set_title("Operator stability across chronological segments")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(axis="y", color="#e5e8e8")
    save(fig, "fig_operator_stability.png")


def state_transitions() -> None:
    frame = pd.read_csv(OUT / "state_transition_probabilities.csv")
    session = int(frame.session.min())
    landmark = int(frame[frame.session == session].landmark.min())
    part = frame[(frame.session == session) & (frame.landmark == landmark)]
    if part.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    panels = [
        ("observed", part[part.model == part.model.iloc[0]].pivot(index="from_state", columns="to_state", values="observed_probability")),
        ("B3 population", part[part.model == "B3_population"].pivot(index="from_state", columns="to_state", values="probability")),
        ("B1 independent", part[part.model == "B1_independent_common"].pivot(index="from_state", columns="to_state", values="probability")),
    ]
    for ax, (title, matrix) in zip(axes, panels):
        im = ax.imshow(matrix.to_numpy(), cmap="magma", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("to state")
        ax.set_ylabel("from state")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="probability")
    fig.suptitle(f"Training-only state transitions, session {session}, landmark {landmark}", y=1.02, fontsize=10)
    save(fig, "fig_state_transitions.png")


def sensitivity_agreement() -> None:
    frame = pd.read_csv(OUT / "sensitivity_identifiability.csv")
    if frame.empty:
        return
    targets = sorted(frame.target.unique())
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    data = [frame.loc[(frame.target == t) & (frame.role == "target"), "mean_response_cosine"].to_numpy() for t in targets]
    data += [frame.loc[(frame.target == t) & (frame.role == "control"), "mean_response_cosine"].to_numpy() for t in targets]
    positions = np.arange(len(targets))
    ax.boxplot(data[: len(targets)], positions=positions - 0.15, widths=0.25, patch_artist=True,
               boxprops=dict(facecolor="#fdebd0", color=AMBER), medianprops=dict(color=INK))
    ax.boxplot(data[len(targets):], positions=positions + 0.15, widths=0.25, patch_artist=True,
               boxprops=dict(facecolor="#eaeded", color=GRAY), medianprops=dict(color=INK))
    ax.set_xticks(positions, targets)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color=RED, linewidth=0.8, linestyle="--")
    ax.set_ylabel("Pairwise response cosine")
    ax.set_title("Sensitivity agreement across permitted fits (target vs matched control)")
    ax.grid(axis="y", color="#e5e8e8")
    fig.text(0.01, 0.01, WARNING, fontsize=7, color=RED)
    save(fig, "fig_sensitivity_agreement.png")


def sensitivity_ood() -> None:
    frame = pd.read_csv(OUT / "sensitivity_summary.csv")
    if frame.empty:
        return
    frame = frame[frame.role == "target"]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.bar(frame.target, frame.ood_max, color=AMBER, alpha=0.9, label="max")
    ax.bar(frame.target, frame.ood_median, color=TEAL, alpha=0.9, label="median")
    ax.set_ylabel("Fraction outside training 1-99% range")
    ax.set_title("Empirical support of bounded virtual perturbations")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color="#e5e8e8")
    fig.text(0.01, 0.01, WARNING, fontsize=7, color=RED)
    save(fig, "fig_sensitivity_ood.png")


def j2_j5() -> None:
    frame = pd.read_csv(OUT / "exact_reproduction_per_landmark.csv")
    j2 = frame[frame.model == "J2_population_diag"].set_index(["worm_id", "landmark"]).energy_h20
    j5 = frame[frame.model == "J5_population_lead1_boot"].set_index(["worm_id", "landmark"]).energy_h20
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(j5, j2, s=12, alpha=0.7, color=TEAL)
    lo = float(min(j2.min(), j5.min()))
    hi = float(max(j2.max(), j5.max()))
    ax.plot([lo, hi], [lo, hi], color=GRAY, linestyle="--", linewidth=0.8)
    ax.set_xlabel("J5 H20 energy")
    ax.set_ylabel("J2 H20 energy")
    ax.set_title("J2/J5 mean equivalence, not per-landmark identity")
    ax.grid(color="#e5e8e8")
    save(fig, "fig_j2_j5_equivalence.png")


def main() -> None:
    cross_neuron()
    cross_controls()
    surrogate_controls()
    operator_spectra()
    operator_stability()
    state_transitions()
    sensitivity_agreement()
    sensitivity_ood()
    j2_j5()
    print(f"figures written to {FIG}")


if __name__ == "__main__":
    main()
