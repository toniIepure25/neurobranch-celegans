"""Evaluate Phase 6 gates from frozen machine-readable artifacts.

A failed gate stays failed.  No gate is promoted because the test block was
reused or because a post-hoc subgroup looked better.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT


def _bootstrap_skill_ci(rows: pd.DataFrame, *, n_boot: int = 4000, seed: int = 0):
    """Worm-cluster bootstrap: resample whole worms, not individual starts."""
    valid = rows[rows["finite"] == True]  # noqa: E712
    if valid.empty:
        return float("nan"), float("nan")
    worms = sorted(valid["worm_id"].unique())
    groups = {w: valid[valid["worm_id"] == w] for w in worms}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sample = rng.choice(worms, size=len(worms), replace=True)
        chosen = pd.concat([groups[w] for w in sample], ignore_index=True)
        denom = chosen["persistence_mse"].mean()
        if denom > 0:
            vals.append(1.0 - chosen["mse"].mean() / denom)
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports/simulator/phase6")
    args = parser.parse_args(argv)
    reports = Path(args.reports)
    test = pd.read_csv(reports / "test_summary.csv")
    per_start = pd.read_csv(reports / "test_per_start.csv")
    branching = json.loads((reports / "branching_overview.json").read_text(encoding="utf-8"))
    perturbation = json.loads((reports / "perturbation_overview.json").read_text(encoding="utf-8"))
    long_run = pd.read_csv(reports / "long_run_stability.csv")
    gates: list[dict[str, Any]] = []

    selected = test[(test["family"] == "var") & (test["horizon"] == 20) & (test["seed"] == 17)]
    finite = float(selected["finite_fraction"].min()) if len(selected) else float("nan")
    gates.append(
        {
            "gate": "G1",
            "name": "functionality",
            "status": "pass" if finite >= 0.95 and len(selected) == 5 else "fail",
            "evidence": {"min_finite_fraction": finite, "n_worms": int(len(selected))},
            "tier": "exploratory",
        }
    )

    phase5 = test[(test["family"] == "phase5_var") & (test["horizon"] == 20) & (test["seed"] == 17)]
    base = test[(test["family"] == "persistence") & (test["horizon"] == 20) & (test["seed"] == 17)]
    merged = selected.merge(phase5[["worm_id", "mse"]], on="worm_id", suffixes=("", "_phase5"))
    merged = merged.merge(base[["worm_id", "mse"]], on="worm_id", suffixes=("", "_persistence"))
    beats_p5 = merged["mse"] < merged["mse_phase5"]
    beats_pers = merged["mse"] < merged["mse_persistence"]
    skill_rows = per_start[(per_start["family"] == "var") & (per_start["horizon"] == 20)]
    ci_low, ci_high = _bootstrap_skill_ci(skill_rows)
    mean_skill = float(selected["skill_vs_persistence"].mean())
    g2 = bool(
        mean_skill > 0.02
        and ci_low > 0
        and int(beats_p5.sum()) >= 4
        and int(beats_pers.sum()) >= 4
    )
    gates.append(
        {
            "gate": "G2",
            "name": "autonomous skill and baseline superiority",
            "status": "pass" if g2 else "fail",
            "evidence": {
                "mean_skill_vs_persistence": mean_skill,
                "skill_ci_low": ci_low,
                "skill_ci_high": ci_high,
                "worms_beating_persistence": int(beats_pers.sum()),
                "worms_beating_phase5_var": int(beats_p5.sum()),
                "criterion": "skill>0.02, CI>0, >=4/5 vs persistence and frozen Phase 5 VAR",
            },
            "tier": "exploratory",
        }
    )

    dist = selected.merge(base[["worm_id", "energy_distance", "autocorr_error", "covariance_error"]], on="worm_id", suffixes=("", "_base"))
    g3_worms = int(
        (
            (dist["energy_distance"] < dist["energy_distance_base"])
            & (dist["autocorr_error"] < dist["autocorr_error_base"])
            & (dist["covariance_error"] < dist["covariance_error_base"])
        ).sum()
    )
    gates.append(
        {
            "gate": "G3",
            "name": "dynamical and distributional fidelity",
            "status": "pass" if g3_worms >= 4 else "fail",
            "evidence": {
                "worms_with_all_three_better_than_persistence": g3_worms,
                "mean_energy_selected": float(selected["energy_distance"].mean()),
                "mean_energy_persistence": float(base["energy_distance"].mean()),
                "mean_autocorr_selected": float(selected["autocorr_error"].mean()),
                "mean_autocorr_persistence": float(base["autocorr_error"].mean()),
            },
            "tier": "exploratory",
        }
    )

    agg = {r["family"]: r for r in branching["aggregate"]}
    sim_tv = float(agg["stochastic_var"]["mean_tv_sim"])
    mark_tv = float(agg["stochastic_var"]["mean_tv_markov"])
    marg_tv = float(agg["stochastic_var"]["mean_tv_marginal"])
    gates.append(
        {
            "gate": "G4",
            "name": "stochastic branching fidelity",
            "status": "pass" if sim_tv < min(mark_tv, marg_tv) else "fail",
            "evidence": {
                "mean_tv_sim": sim_tv,
                "mean_tv_markov": mark_tv,
                "mean_tv_marginal": marg_tv,
                "mean_entropy_observed": float(agg["stochastic_var"]["mean_entropy_observed"]),
                "mean_entropy_simulated": float(agg["stochastic_var"]["mean_entropy_simulated"]),
            },
            "tier": "exploratory",
        }
    )

    in_support = float(perturbation["in_support_fraction_mean"])
    gates.append(
        {
            "gate": "G5",
            "name": "support-constrained perturbation",
            "status": "pass" if in_support >= 0.90 else "fail",
            "evidence": {
                "mean_in_support_fraction": in_support,
                "conditions_in_support": perturbation["n_conditions_in_support"],
                "conditions": perturbation["n_conditions"],
                "criterion": ">=90% perturbed states inside validation-calibrated support",
            },
            "tier": "exploratory",
        }
    )

    events = pd.read_csv(reports / "perturbation_events.csv")
    per_worm = events.groupby(["target", "worm_id"], as_index=False).agg(
        tv=("tv_effect", "mean"), control=("tv_control", "mean")
    )
    per_worm["excess"] = per_worm["tv"] - per_worm["control"]
    target_table = (
        per_worm.groupby("target", as_index=False)
        .agg(
            mean_tv=("tv", "mean"),
            mean_control=("control", "mean"),
            mean_excess=("excess", "mean"),
            worms_positive=("excess", lambda s: int((s > 0).sum())),
        )
        .sort_values("mean_excess", ascending=False)
    )
    robust = target_table[
        (target_table["worms_positive"] >= 4)
        & (target_table["mean_excess"] > 0)
        & (target_table["mean_tv"] >= 0.02)
    ]
    n_families = int(events["family"].nunique())
    if n_families < 2:
        g6_status = "not_evaluable"
    elif len(robust) >= 1:
        g6_status = "pass"
    else:
        g6_status = "fail"
    gates.append(
        {
            "gate": "G6",
            "name": "robust named-neuron virtual sensitivity",
            "status": g6_status,
            "evidence": {
                "n_model_families_tested": n_families,
                "n_robust_targets": int(len(robust)),
                "criterion": "one target/direction has consistent sign in >=4/5 worms and >=2/3 model families, conditional on G5",
                "reason": "only one family was evaluated; the protocol requires >=2/3 families"
                if n_families < 2
                else "",
                "exploratory_targets": target_table.to_dict(orient="records"),
                "control_note": "controls used a deterministic random neuron; protocol asked for innovation-SD/autocorrelation matching, so specificity is descriptive only",
            },
            "tier": "exploratory",
        }
    )
    gates.append(
        {
            "gate": "G7",
            "name": "biological validation",
            "status": "not_evaluable",
            "evidence": {
                "reason": "Kato is unstimulated and contains no matched intervention or stimulation-response data",
            },
            "tier": "not_evaluable",
        }
    )

    payload = {
        "schema_version": 1,
        "phase": 6,
        "evidence_tier": "exploratory",
        "test_block_reused": True,
        "gates": gates,
        "long_run": long_run.to_dict(orient="records") if len(long_run) else [],
        "warning": "A failed gate is not relabelled. No causal or digital-twin claim is supported.",
    }
    (reports / "gates.json").write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps({"gates": [{k: g[k] for k in ("gate", "status")} for g in gates]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
