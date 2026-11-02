"""Summarise joint-dynamics scores, paired contrasts, calibration and gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neurobranch.adaptive.evaluation import block_bootstrap_ci

OUT = Path("reports/simulator/joint_dynamics")
PRIOR = Path("reports/simulator/uncertainty_calibration")
HORIZONS = (1, 5, 10, 20, 50, 100)
LEVELS = (50, 80, 95)
PRIMARY = "J5_population_lead1_boot"
REFERENCE = "J1_independent_diag"
PERSISTENCE = "J0_persistence"


def paired(scores: pd.DataFrame, model: str, reference: str, metric: str, block: int = 5) -> dict:
    a = scores[scores.model == model][["worm_id", "landmark", metric]]
    b = scores[scores.model == reference][["worm_id", "landmark", metric]]
    m = a.merge(b, on=["worm_id", "landmark"], suffixes=("_model", "_reference"))
    diff = (m[f"{metric}_reference"] - m[f"{metric}_model"]).to_numpy()
    mean, lo, hi = block_bootstrap_ci(diff, block=block, n_boot=1000, seed=211)
    return {
        "model": model,
        "reference": reference,
        "metric": metric,
        "n_pairs": int(len(diff)),
        "mean_improvement": mean,
        "ci_low": lo,
        "ci_high": hi,
        "fraction_better": float((diff > 0).mean()),
    }


def main() -> None:
    scores = pd.read_csv(OUT / "scores_by_landmark.csv")
    neurons = pd.read_csv(OUT / "neuron_h20.csv")
    coverage = pd.read_csv(OUT / "coverage_by_landmark.csv")
    subset = pd.read_csv(OUT / "neural_subset_scores.csv")
    summary = (
        scores.groupby("model")
        .agg(
            n=("landmark", "size"),
            energy_h1=("energy_h1", "mean"),
            energy_h5=("energy_h5", "mean"),
            energy_h10=("energy_h10", "mean"),
            energy_h20=("energy_h20", "mean"),
            energy_h50=("energy_h50", "mean"),
            energy_h100=("energy_h100", "mean"),
            crps_h20=("crps_h20", "mean"),
            crps_h20_full=("crps_h20_full", "mean"),
            coverage_h20_full_50=("coverage_h20_full_50", "mean"),
            coverage_h20_full_80=("coverage_h20_full_80", "mean"),
            coverage_h20_full_95=("coverage_h20_full_95", "mean"),
            trajectory_energy_h20=("trajectory_energy_h20", "mean"),
            trajectory_energy_h100=("trajectory_energy_h100", "mean"),
            variogram_h20=("variogram_h20", "mean"),
            coverage_50=("coverage_h20_50", "mean"),
            coverage_80=("coverage_h20_80", "mean"),
            coverage_95=("coverage_h20_95", "mean"),
            width_95=("width_h20_95", "mean"),
            neural_cov_error=("neural_cov_error", "mean"),
            path_cov_error=("path_cov_error", "mean"),
            autocorr_error=("autocorr_error", "mean"),
            mean_rmse=("mean_rmse", "mean"),
            ensemble_std=("ensemble_std", "mean"),
            ood_fraction=("ood_fraction", "mean"),
            occupancy_tv=("occupancy_tv", "mean"),
            recurrence_model=("recurrence_model", "mean"),
            recurrence_observed=("recurrence_observed", "mean"),
            energy_h20_full=("energy_h20_full", "mean"),
            energy_h20_no_top5=("energy_h20_no_top5", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUT / "score_summary.csv", index=False)
    # Pairwise contrasts, overall and per session.
    rows = []
    for model in sorted(scores.model.unique()):
        for ref in (REFERENCE, PERSISTENCE):
            if model == ref:
                continue
            for metric in ("energy_h20", "trajectory_energy_h20"):
                rows.append(paired(scores, model, ref, metric))
    paired_df = pd.DataFrame(rows)
    paired_df.to_csv(OUT / "paired_scores.csv", index=False)
    secondary = []
    for metric in ("crps_h20_full", "variogram_h20", "energy_h20_full", "energy_h20_no_top5"):
        for model, ref in ((PRIMARY, REFERENCE), ("J2_population_diag", REFERENCE)):
            secondary.append(paired(scores, model, ref, metric))
    pd.DataFrame(secondary).to_csv(OUT / "secondary_paired.csv", index=False)
    session_rows = []
    for (model, sid), group in scores.groupby(["model", "worm_id"]):
        for ref in (REFERENCE, PERSISTENCE):
            if model == ref:
                continue
            sub = scores[(scores.model == ref) & (scores.worm_id == sid)]
            m = group.merge(sub, on=["worm_id", "landmark"], suffixes=("_model", "_reference"))
            for metric in ("energy_h20", "trajectory_energy_h20"):
                diff = (m[f"{metric}_reference"] - m[f"{metric}_model"]).to_numpy()
                session_rows.append(
                    {
                        "model": model,
                        "reference": ref,
                        "worm_id": int(sid),
                        "metric": metric,
                        "mean_improvement": float(np.mean(diff)),
                        "fraction_better": float(np.mean(diff > 0)),
                        "n": int(len(diff)),
                    }
                )
    session = pd.DataFrame(session_rows)
    session.to_csv(OUT / "session_paired_scores.csv", index=False)
    ci_rows = []
    for (model, ref, sid), _group in session.groupby(["model", "reference", "worm_id"]):
        metric = "energy_h20"
        a = scores[(scores.model == model) & (scores.worm_id == sid)][["landmark", metric]]
        b = scores[(scores.model == ref) & (scores.worm_id == sid)][["landmark", metric]]
        m = a.merge(b, on="landmark", suffixes=("_model", "_reference"))
        diff = (m[f"{metric}_reference"] - m[f"{metric}_model"]).to_numpy()
        mean, lo, hi = block_bootstrap_ci(diff, block=5, n_boot=800, seed=311 + int(sid))
        ci_rows.append(
            {"model": model, "reference": ref, "worm_id": int(sid), "mean_improvement": mean, "ci_low": lo, "ci_high": hi}
        )
    pd.DataFrame(ci_rows).to_csv(OUT / "session_paired_ci.csv", index=False)
    block_rows = []
    for block in (3, 5, 10):
        for model, ref in ((PRIMARY, REFERENCE), (PRIMARY, PERSISTENCE), ("J2_population_diag", REFERENCE)):
            block_rows.append({**paired(scores, model, ref, "energy_h20", block=block), "block": block})
    pd.DataFrame(block_rows).to_csv(OUT / "bootstrap_block_sensitivity.csv", index=False)
    session_summary = (
        scores.groupby(["model", "worm_id"])
        .agg(
            energy_h20=("energy_h20", "mean"),
            crps_h20=("crps_h20", "mean"),
            trajectory_energy_h20=("trajectory_energy_h20", "mean"),
            coverage_50=("coverage_h20_50", "mean"),
            coverage_80=("coverage_h20_80", "mean"),
            coverage_95=("coverage_h20_95", "mean"),
            neural_cov_error=("neural_cov_error", "mean"),
            autocorr_error=("autocorr_error", "mean"),
        )
        .reset_index()
    )
    session_summary.to_csv(OUT / "session_scores.csv", index=False)
    # Calibration curves and calibration table.
    cal_rows = []
    pit_rows = []
    for model, group in neurons.groupby("model"):
        density, _ = np.histogram(group.pit_h20.to_numpy(), bins=10, range=(0, 1))
        density = density / max(1, density.sum())
        for i, value in enumerate(density):
            pit_rows.append({"model": model, "bin": i, "density": float(value)})
        cal_rows.append(
            {
                "model": model,
                "coverage_50": group.coverage_50.mean(),
                "coverage_80": group.coverage_80.mean(),
                "coverage_95": group.coverage_95.mean(),
                "coverage_full_50": scores[scores.model == model].coverage_h20_full_50.mean(),
                "coverage_full_80": scores[scores.model == model].coverage_h20_full_80.mean(),
                "coverage_full_95": scores[scores.model == model].coverage_h20_full_95.mean(),
                "width_95": group.width_95.mean(),
                "crps_h20": group.crps_h20.mean(),
                "crps_h20_full": scores[scores.model == model].crps_h20_full.mean(),
                "energy_h20": scores[scores.model == model].energy_h20.mean(),
                "trajectory_energy_h20": scores[scores.model == model].trajectory_energy_h20.mean(),
                "pit_calibration_error": float(np.abs(density - 0.1).mean()),
            }
        )
    pd.DataFrame(pit_rows).to_csv(OUT / "calibration_curves.csv", index=False)
    cal = pd.DataFrame(cal_rows)
    ref_width = cal.loc[cal.model == PERSISTENCE, "width_95"]
    cal["width_ratio_vs_persistence"] = cal.width_95 / float(ref_width.iloc[0]) if len(ref_width) else np.nan
    cal.to_csv(OUT / "calibration_h20.csv", index=False)
    d4_path = OUT / "d4_summary.csv"
    d1d2 = cal[cal.model.isin([PRIMARY, "J3_population_full"])].copy()
    d1d2["calibration_method"] = d1d2.model.map({PRIMARY: "D1 historical corrected", "J3_population_full": "D2 regularized full covariance"})
    if d4_path.exists():
        d4 = pd.read_csv(d4_path)
        row = {
            "model": "D4_population_student_t",
            "coverage_50": float(d4.coverage_h20_50.iloc[0]),
            "coverage_80": float(d4.coverage_h20_80.iloc[0]),
            "coverage_95": float(d4.coverage_h20_95.iloc[0]),
            "coverage_full_50": np.nan,
            "coverage_full_80": np.nan,
            "coverage_full_95": np.nan,
            "width_95": float(d4.width_h20_95.iloc[0]),
            "crps_h20": float(d4.crps_h20.iloc[0]),
            "crps_h20_full": np.nan,
            "energy_h20": float(d4.energy_h20.iloc[0]),
            "trajectory_energy_h20": float(d4.trajectory_energy_h20.iloc[0]),
            "pit_calibration_error": np.nan,
            "calibration_method": "D4 validation-fitted Student-t",
        }
        d1d2 = pd.concat([d1d2, pd.DataFrame([row])], ignore_index=True)
    not_fitted = pd.DataFrame(
        [{"model": "D3_temporal_innovation", "calibration_method": "D3 not fitted: Stage C did not justify temporal innovations"}]
    )
    pd.concat([d1d2, not_fitted], ignore_index=True).to_csv(OUT / "calibration_models.csv", index=False)
    fac = summary.set_index("model")
    unc_rows = [
        {
            "component": "process innovation (diagonal)",
            "model": "J2_population_diag",
            "energy_h20": float(fac.loc["J2_population_diag", "energy_h20"]),
            "coverage_50": float(fac.loc["J2_population_diag", "coverage_h20_full_50"]),
            "coverage_80": float(fac.loc["J2_population_diag", "coverage_h20_full_80"]),
            "coverage_95": float(fac.loc["J2_population_diag", "coverage_h20_full_95"]),
            "status": "estimated",
        },
        {
            "component": "parameter uncertainty",
            "model": "H_parameter_ensemble",
            "energy_h20": float(pd.read_csv(OUT / "parameter_summary.csv").energy_h20.iloc[0]),
            "coverage_50": float(pd.read_csv(OUT / "parameter_summary.csv").coverage_h20_50.iloc[0]),
            "coverage_80": float(pd.read_csv(OUT / "parameter_summary.csv").coverage_h20_80.iloc[0]),
            "coverage_95": float(pd.read_csv(OUT / "parameter_summary.csv").coverage_h20_95.iloc[0]),
            "status": "estimated; worsens dispersion",
        },
        {
            "component": "initial-state uncertainty",
            "model": "none",
            "energy_h20": np.nan,
            "coverage_50": np.nan,
            "coverage_80": np.nan,
            "coverage_95": np.nan,
            "status": "set to observed initial state",
        },
        {
            "component": "observation uncertainty",
            "model": "none",
            "energy_h20": np.nan,
            "coverage_50": np.nan,
            "coverage_80": np.nan,
            "coverage_95": np.nan,
            "status": "NOT_IDENTIFIABLE",
        },
    ]
    pd.DataFrame(unc_rows).to_csv(OUT / "uncertainty_decomposition.csv", index=False)
    cov_h = (
        coverage.groupby(["model", "horizon", "level"])
        .covered.mean()
        .reset_index()
        .pivot_table(index=["model", "horizon"], columns="level", values="covered")
        .reset_index()
        .rename(columns={50: "coverage_50", 80: "coverage_80", 95: "coverage_95"})
    )
    cov_h.to_csv(OUT / "coverage_by_horizon.csv", index=False)
    cov_summary = []
    for model, group in cov_h.groupby("model"):
        dev = np.maximum.reduce(
            [
                np.abs(group.coverage_50 - 0.5),
                np.abs(group.coverage_80 - 0.8),
                np.abs(group.coverage_95 - 0.95),
            ]
        )
        cov_summary.append(
            {
                "model": model,
                "max_horizon_deviation": float(np.max(dev)),
                "mean_horizon_deviation": float(np.mean(dev)),
                "h20_deviation": float(np.max(dev[group.horizon == 20])),
            }
        )
    pd.DataFrame(cov_summary).to_csv(OUT / "coverage_summary.csv", index=False)
    # Energy by horizon table.
    e_rows = []
    for h in HORIZONS:
        for model, group in scores.groupby("model"):
            e_rows.append({"model": model, "horizon": h, "energy": group[f"energy_h{h}"].mean()})
    pd.DataFrame(e_rows).to_csv(OUT / "energy_by_horizon.csv", index=False)
    # Factorial mean/covariance decomposition at H20.
    fac = summary.set_index("model")
    cells = ["J1_independent_diag", "J4_independent_full", "J2_population_diag", "J3_population_full"]
    fact_rows = []
    if all(m in fac.index for m in cells):
        e = fac.loc[cells, "energy_h20"].to_dict()
        te = fac.loc[cells, "trajectory_energy_h20"].to_dict()
        for metric, vals in (("energy_h20", e), ("trajectory_energy_h20", te)):
            mean_effect = 0.5 * (vals["J1_independent_diag"] + vals["J4_independent_full"]) - 0.5 * (
                vals["J2_population_diag"] + vals["J3_population_full"]
            )
            cov_effect = 0.5 * (vals["J1_independent_diag"] + vals["J2_population_diag"]) - 0.5 * (
                vals["J4_independent_full"] + vals["J3_population_full"]
            )
            interaction = (
                vals["J1_independent_diag"]
                - vals["J4_independent_full"]
                - vals["J2_population_diag"]
                + vals["J3_population_full"]
            )
            fact_rows.append(
                {
                    "metric": metric,
                    "I_diag": vals["J1_independent_diag"],
                    "I_full": vals["J4_independent_full"],
                    "P_diag": vals["J2_population_diag"],
                    "P_full": vals["J3_population_full"],
                    "mean_main_effect_positive_is_better": mean_effect,
                    "covariance_main_effect_positive_is_better": cov_effect,
                    "interaction": interaction,
                }
            )
    factorial = pd.DataFrame(fact_rows)
    factorial.to_csv(OUT / "factorial_decomposition.csv", index=False)
    # Sensitivity table.
    sens = []
    for model in (PRIMARY, REFERENCE, PERSISTENCE, "J3_population_full"):
        if model not in fac.index:
            continue
        row = summary[summary.model == model].iloc[0]
        subset_row = subset[subset.model == model]
        sens.append(
            {
                "model": model,
                "pca10_energy_h20": row.energy_h20,
                "full_neuron_energy_h20": row.energy_h20_full,
                "no_top5_energy_h20": row.energy_h20_no_top5,
                "top20_neuron_energy_h20": subset_row.energy_h20_neural20.mean() if len(subset_row) else np.nan,
                "session_energy_min": row.energy_h20,
            }
        )
    pd.DataFrame(sens).to_csv(OUT / "sensitivity_representation.csv", index=False)
    # Reproduction against the frozen previous report.
    repro = []
    if (PRIOR / "score_summary.csv").exists():
        old = pd.read_csv(PRIOR / "score_summary.csv").set_index("model")
        pairs = [("J0_persistence", "P0_persistence"), (PRIMARY, "P6_adaptive_lead1")]
        for new, prev in pairs:
            if new not in fac.index or prev not in old.index:
                continue
            for metric, old_key in (
                ("energy_h20", "energy_h20"),
                ("crps_h20_full", "crps_h20"),
                ("coverage_h20_full_50", "coverage_50"),
                ("coverage_h20_full_80", "coverage_80"),
                ("coverage_h20_full_95", "coverage_95"),
            ):
                a, b = float(fac.loc[new, metric]), float(old.loc[prev, old_key])
                repro.append(
                    {
                        "new_model": new,
                        "historical_model": prev,
                        "metric": metric,
                        "historical": b,
                        "rerun": a,
                        "relative_difference": abs(a - b) / max(abs(b), 1e-12),
                    }
                )
    repro_df = pd.DataFrame(repro)
    hist_path = OUT / "historical_block_scores.csv"
    if hist_path.exists() and (PRIOR / "score_summary.csv").exists():
        hist = pd.read_csv(hist_path)
        old = pd.read_csv(PRIOR / "score_summary.csv").set_index("model")
        if "P6_adaptive_block" in old.index:
            pairs = {
                "energy_h20": ("energy_h20", "energy_h20"),
                "crps_h20_full": ("crps_h20_full", "crps_h20"),
                "coverage_full_50": ("coverage_full_50", "coverage_50"),
                "coverage_full_80": ("coverage_full_80", "coverage_80"),
                "coverage_full_95": ("coverage_full_95", "coverage_95"),
            }
            for new_key, (hist_key, old_key) in pairs.items():
                a, b = float(hist[hist_key].mean()), float(old.loc["P6_adaptive_block", old_key])
                repro_df = pd.concat(
                    [
                        repro_df,
                        pd.DataFrame(
                            [
                                {
                                    "new_model": "P6_historical_block",
                                    "historical_model": "P6_adaptive_block",
                                    "metric": new_key,
                                    "historical": b,
                                    "rerun": a,
                                    "relative_difference": abs(a - b) / max(abs(b), 1e-12),
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
    repro_df.to_csv(OUT / "reproduction_check.csv", index=False)
    # Gates.
    thresholds = json.loads((OUT / "gate_thresholds.json").read_text(encoding="utf-8"))
    g1 = bool(len(repro_df) and repro_df.relative_difference.max() <= thresholds["G1_reproduction"]["max_relative_difference"])
    p_vs_ind = paired_df[(paired_df.model == PRIMARY) & (paired_df.reference == REFERENCE) & (paired_df.metric == "energy_h20")]
    g2 = bool(
        len(p_vs_ind)
        and p_vs_ind.iloc[0].mean_improvement >= thresholds["G2_population_information"]["minimum_mean_improvement"]
        and p_vs_ind.iloc[0].ci_low > 0
    )
    structure = pd.read_csv(OUT / "innovation_structure.csv")
    pop = structure[structure.model == "population_var"]
    g3 = bool(
        np.median(np.abs(pop.acf1)) <= thresholds["G3_innovation_validity"]["population_median_abs_acf1_max"]
        and np.median(np.abs(pop.acf5)) <= thresholds["G3_innovation_validity"]["population_median_abs_acf5_max"]
    )
    cal_primary = cal[cal.model == PRIMARY]
    cov_primary = pd.DataFrame(cov_summary).set_index("model").loc[PRIMARY]
    g4 = bool(
        len(cal_primary)
        and max(
            abs(cal_primary.iloc[0].coverage_full_50 - 0.5),
            abs(cal_primary.iloc[0].coverage_full_80 - 0.8),
            abs(cal_primary.iloc[0].coverage_full_95 - 0.95),
        )
        <= thresholds["G4_marginal_calibration"]["max_abs_coverage_error_each_level"]
        and cov_primary.max_horizon_deviation <= thresholds["G4_marginal_calibration"]["max_horizon_deviation"]
    )
    p_row = fac.loc[PRIMARY]
    p0_row = fac.loc[PERSISTENCE]
    g5 = bool(
        p_row.trajectory_energy_h20 / p0_row.trajectory_energy_h20
        <= thresholds["G5_joint_trajectory"]["trajectory_energy_max_relative_to_persistence"]
        and p_row.neural_cov_error <= thresholds["G5_joint_trajectory"]["neural_covariance_relative_error_max"]
        and p_row.autocorr_error <= thresholds["G5_joint_trajectory"]["autocorrelation_error_max"]
    )
    s_prim = session[(session.model == PRIMARY) & (session.reference == REFERENCE) & (session.metric == "energy_h20")]
    g6 = bool((s_prim.mean_improvement > 0).sum() >= thresholds["G6_robustness"]["sessions_with_same_direction_min"])
    gates = {
        "G1_reproduction": {"status": "PASS" if g1 else "FAIL", "max_relative_difference": float(repro_df.relative_difference.max()) if len(repro_df) else None},
        "G2_population_information": {"status": "PASS" if g2 else "FAIL", "evidence": p_vs_ind.to_dict(orient="records")},
        "G3_innovation_validity": {"status": "PASS" if g3 else "FAIL", "median_abs_acf1": float(np.median(np.abs(pop.acf1))), "median_abs_acf5": float(np.median(np.abs(pop.acf5)))},
        "G4_marginal_calibration": {"status": "PASS" if g4 else "FAIL", "coverage_h20": cal_primary.to_dict(orient="records"), "max_horizon_deviation": float(cov_primary.max_horizon_deviation)},
        "G5_joint_trajectory": {"status": "PASS" if g5 else "FAIL", "trajectory_ratio_vs_persistence": float(p_row.trajectory_energy_h20 / p0_row.trajectory_energy_h20), "neural_cov_error": float(p_row.neural_cov_error), "autocorr_error": float(p_row.autocorr_error)},
        "G6_robustness": {"status": "PASS" if g6 else "FAIL", "sessions_positive": int((s_prim.mean_improvement > 0).sum())},
        "G7_generative_credibility": {
            "status": "PASS" if (g2 and g4 and g5 and g6) else "FAIL",
            "requires": ["G2", "G4", "G5", "G6"],
        },
        "G8_biological_intervention": {"status": "NOT_EVALUABLE", "reason": "no compatible experimental intervention data"},
        "stop_label": "PROBABILISTIC CALIBRATION NOT ESTABLISHED" if not (g2 and g4 and g5 and g6) else "CALIBRATED GENERATIVE SIMULATOR SUPPORTED WITHIN KATO TEST SPLIT",
    }
    (OUT / "gates.json").write_text(json.dumps(gates, indent=2, default=float), encoding="utf-8")
    print(summary.to_string())
    print(factorial.to_string())
    print(json.dumps(gates, indent=2, default=float))


if __name__ == "__main__":
    main()
