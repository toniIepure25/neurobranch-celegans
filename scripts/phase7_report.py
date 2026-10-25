"""Generate the Phase 7 results markdown from canonical artifacts."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, resolve_path

WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED."


def _fmt(value: float, digits: int = 4) -> str:
    return "n/a" if value is None or value != value else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="reports/phase7")
    parser.add_argument("--output", default="docs/PHASE7_RESULTS.md")
    args = parser.parse_args()
    root = resolve_path(args.root, PROJECT_ROOT)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(root / "metrics.csv")
    deltas = pd.read_csv(root / "deltas.csv")
    primary = metrics[metrics["budget"].round(2).eq(float(summary["primary_budget"]))]
    by_strategy = (
        primary.groupby("strategy")[
            ["pair_spearman", "profile_normalized", "coverage_calibrated", "interval_score"]
        ]
        .mean()
        .reset_index()
        .sort_values("pair_spearman", ascending=False)
    )
    h1_rows = summary.get("primary_hybrid_vs_random", [])
    h1 = h1_rows[0] if h1_rows else {}
    edge = pd.DataFrame(summary.get("edge_metrics", []))
    full_atlas = float(summary.get("full_atlas_mean_pair_spearman", float("nan")))
    medians = summary.get("targets_to_90pct_full_median", {})
    medians_text = (
        ", ".join(
            f"{name}: {'not reached' if value is None or value == float('inf') else f'{value:.0f}'}"
            for name, value in sorted(medians.items())
        )
        if medians
        else "n/a"
    )
    curves = (
        metrics.groupby(["budget", "strategy"])["pair_spearman"]
        .mean()
        .reset_index()
        .sort_values(["budget", "strategy"])
    )
    lines = [
        "# Phase 7 results — Active Discrepancy Atlas",
        "",
        f"**Warning:** {WARNING}",
        "",
        "## Decision",
        "",
    ]
    if summary.get("h1_supported"):
        lines.append(
            "**H1 SUPPORTED.** The preregistered hybrid policy beat random acquisition at the "
            "40% target budget."
        )
    else:
        lines.append(
            "**H1 NOT SUPPORTED.** The preregistered hybrid policy did not meet the +0.02 "
            "absolute gate with a 95% CI above zero at the 40% target budget. No secondary "
            "result is promoted to a primary claim."
        )
    lines += [
        "",
        f"Hybrid minus random at 40%: **{_fmt(h1.get('delta'))}** "
        f"(95% block-bootstrap CI {_fmt(h1.get('ci_low'))} to {_fmt(h1.get('ci_high'))}; "
        f"{int(h1.get('wins', 0))}/{int(h1.get('n_blocks', 0))} paired blocks positive).",
        "",
        "The active policies were directionally better than random at 40% in the point "
        "estimate, but the preregistered effect size and CI gate were not met. This is a "
        "negative primary result, not a licence to promote the best secondary strategy.",
        "",
        "## Primary budget (40% of all targets)",
        "",
        "| Strategy | Pair Spearman | Profile / ceiling | 90% calibrated coverage | Interval score |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in by_strategy.itertuples(index=False):
        lines.append(
            f"| {row.strategy} | {_fmt(row.pair_spearman)} | {_fmt(row.profile_normalized)} | "
            f"{_fmt(row.coverage_calibrated)} | {_fmt(row.interval_score, 2)} |"
        )
    lines += [
        "",
        f"Full-atlas reference (60% budget, 5 seeds): **{_fmt(full_atlas)}** pair Spearman.",
        "",
        f"Median targets to reach 90% of the full-atlas score: **{medians_text}**. "
        "`not reached` means the strategy did not reach 90% of the full-atlas score by the "
        "50% budget.",
        "",
        "The `Profile / ceiling` column is unstable in this run: several values exceed 1, "
        "which means the split-half ceiling (10 draws, `min_events=2`) is underestimated for "
        "these test targets. The raw profile Spearman is the interpretable quantity; the "
        "normalised value is not used as a success criterion or a claim that the model beats "
        "the measurement ceiling.",
        "",
        "## Budget curves",
        "",
        "| Budget | Strategy | Pair Spearman | Targets |",
        "|---:|---|---:|---:|",
    ]
    for row in curves.itertuples(index=False):
        budget_rows = metrics[
            np.isclose(metrics["budget"], float(row.budget)) & (metrics["strategy"] == row.strategy)
        ]
        n_targets = int(budget_rows["n_targets"].iloc[0]) if not budget_rows.empty else 0
        lines.append(
            f"| {float(row.budget):.2f} | {row.strategy} | {_fmt(row.pair_spearman)} | {n_targets} |"
        )
    lines += [
        "",
        "At the 20% budget all strategies share the same stratified random seed set by "
        "construction; differences begin only after the first acquisition step. The 60% "
        "full-atlas reference is a separate five-seed model, not an acquisition strategy.",
    ]
    lines += [
        "",
        "## Paired deltas versus random",
        "",
        "| Budget | Strategy | Delta | 95% CI | Wins / blocks |",
        "|---:|---|---:|---|---:|",
    ]
    for row in deltas[
        (deltas["strategy"] != "random")
        & (deltas["strategy"] != "full_atlas")
        & (deltas["budget"] < 0.60)
    ].sort_values(["budget", "strategy"]).itertuples(index=False):
        lines.append(
            f"| {float(row.budget):.2f} | {row.strategy} | {_fmt(row.delta)} | "
            f"{_fmt(row.ci_low)} to {_fmt(row.ci_high)} | {int(row.wins)}/{int(row.n_blocks)} |"
        )
    if not edge.empty:
        lines += [
            "",
            "## Anatomical association and anatomy-unexplained readout",
            "",
            "| Strategy | Direct-edge AUROC | Anatomy-unexplained AUROC | Pairs |",
            "|---|---:|---:|---:|",
        ]
        for row in edge.sort_values("auroc_direct_edge", ascending=False).itertuples(index=False):
            lines.append(
                f"| {row.strategy} | {_fmt(row.auroc_direct_edge)} | "
                f"{_fmt(row.auroc_anatomy_unexplained)} | {int(row.n_pairs)} |"
            )
        lines += [
            "",
            f"H2 discrepancy minus uncertainty on anatomy-unexplained AUROC: "
            f"**{_fmt(summary.get('h2_discrepancy_minus_uncertainty_auroc'))}**; "
            f"supported: **{bool(summary.get('h2_supported'))}**.",
            f"H4 hybrid minus random on direct-edge AUROC: "
            f"**{_fmt(summary.get('h4_hybrid_minus_random_edge_auroc'))}**; "
            f"supported: **{bool(summary.get('h4_supported'))}**.",
            "",
            "The anatomy-unexplained label means no direct chemical/electrical edge and an "
            "observed response in the top decile. It is an association candidate, not evidence "
            "of an extrasynaptic mechanism.",
        ]
    state_path = root / "state_ablation" / "summary.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        lines += [
            "",
            "## H3 state-conditioned ablation",
            "",
            f"Mean state minus base pair Spearman: **{_fmt(state.get('mean_delta_state_minus_base'))}** "
            f"over {int(state.get('n_folds', 0))} folds; supported: **{bool(state.get('h3_supported'))}**.",
            "",
            "The state ablation is separate from the primary H1 benchmark.",
        ]
    else:
        lines += [
            "",
            "## H3 state-conditioned ablation",
            "",
            "Not run in the canonical Phase 7 report (compute-constrained deviation D4). The "
            "state cache and leakage-safe projector are implemented and unit-tested; no H3 "
            "claim is made.",
        ]
    lines += [
        "",
        "## What this does and does not show",
        "",
        "- It is a pseudo-prospective computational benchmark on held-out targets from the same atlas.",
        "- It is not a wet-lab closed loop.",
        "- It does not establish causal effects or a causal graph.",
        "- It does not establish cross-dataset or cross-species transfer.",
        "- The final target list is a set of hypotheses for a future experiment.",
        "",
        "## Artifacts",
        "",
        "- `reports/phase7/metrics.csv`, `budget_curves.csv`, `deltas.csv`, `targets_to_threshold.csv`",
        "- `reports/phase7/selection_log.jsonl`, `audit.json`, `predictions.parquet`",
        "- `reports/phase7/connectivity/edge_metrics.csv`, `discrepancy_pairs.csv`",
        "- `reports/phase7/design/target_candidates.csv`",
        "",
        f"**{WARNING}**",
        "",
    ]
    out = resolve_path(args.output, PROJECT_ROOT)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
