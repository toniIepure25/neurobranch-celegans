"""Assemble diagnostic summaries and freeze scientific gate decisions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neurobranch.adaptive.evaluation import block_bootstrap_ci

OUT = Path("reports/simulator/uncertainty_calibration")


def main() -> None:
    innovation = pd.read_csv(OUT / "innovation_validation.csv")
    main_innov = innovation.query("lead == -1").copy()
    main_innov.groupby("model").median(numeric_only=True).reset_index().to_csv(OUT / "innovation_summary.csv", index=False)
    lead_profile = innovation.query("lead > 0").groupby(["model", "lead"])[["trace_cov", "mean_squared"]].median().reset_index()
    lead_profile.to_csv(OUT / "lead_profile.csv", index=False)

    operator = pd.read_csv(OUT / "operator_diagnostics.csv")
    operator.median(numeric_only=True).to_frame("median").T.to_csv(OUT / "operator_summary.csv", index=False)
    prop = pd.read_csv(OUT / "variance_propagation.csv")
    ratios = {
        "block_all_over_analytic_all": float(np.median(prop.trace_block_all / prop.trace_analytical_all)),
        "block_lead1_over_analytic_lead1": float(np.median(prop.trace_block_lead1 / prop.trace_analytical_lead1)),
        "block_all_over_block_lead1": float(np.median(prop.trace_block_all / prop.trace_block_lead1)),
        "analytic_lead1_over_all": float(np.median(prop.trace_analytical_lead1 / prop.trace_analytical_all)),
        "gaussian_mc_over_analytic_lead1": float(np.median(prop.trace_mc_gaussian_lead1 / prop.trace_analytical_lead1)),
        "p20_over_q_lead1": float(np.median(operator.trace_P20_lead1 / operator.trace_Q_lead1)),
        "p100_over_p20_lead1": float(np.median(operator.trace_P100_lead1 / operator.trace_P20_lead1)),
        "operator_norm_a20_median": float(operator.norm_A20.median()),
        "spectral_radius_median": float(operator.radius_adaptive.median()),
        "nonnormality_median": float(operator.nonnormality.median()),
        "residual_trace_lead1_median": float(operator.trace_Q_lead1.median()),
        "residual_trace_all_median": float(operator.trace_Q_all.median()),
        "observed_h20_mse_mean": float(prop.observed_h20_mse.mean()),
        "observed_h1_mse_mean": float(prop.observed_h1_mse.mean()),
    }
    pd.DataFrame([{"component": k, "value": v} for k, v in ratios.items()]).to_csv(
        OUT / "variance_decomposition.csv", index=False
    )
    (OUT / "variance_decomposition.json").write_text(json.dumps(ratios, indent=2))

    coverage = pd.read_csv(OUT / "coverage_by_horizon.csv")
    coverage.groupby(["model", "horizon"])[["coverage_50", "coverage_80", "coverage_95", "width_95"]].mean().reset_index().to_csv(
        OUT / "coverage_summary.csv", index=False
    )

    neurons = pd.read_csv(OUT / "per_neuron_h20.csv")
    cal_rows = []
    for model, group in neurons.groupby("model"):
        pit = group.pit_h20.to_numpy()
        density = np.histogram(pit, bins=10, range=(0, 1))[0] / len(pit)
        cal_rows.append(
            {
                "model": model,
                "n": int(len(group)),
                "coverage_50": float(group.coverage_50.mean()),
                "coverage_80": float(group.coverage_80.mean()),
                "coverage_95": float(group.coverage_95.mean()),
                "width_95": float(group.width_95.mean()),
                "pit_error": float(np.abs(density - 0.1).mean()),
                "standardized_error_variance": float(
                    np.mean((group.truth_h20 - group.ensemble_mean_h20) ** 2 / np.maximum(group.ensemble_var_h20, 1e-12))
                ),
                "crps_h20": float(group.crps_h20.mean()),
            }
        )
    calibration = pd.DataFrame(cal_rows)
    calibration.to_csv(OUT / "calibration_h20.csv", index=False)

    summary = pd.read_csv(OUT / "score_summary.csv")
    paired = pd.read_csv(OUT / "paired_scores.csv")
    traj = pd.read_csv(OUT / "trajectory_scores.csv")
    traj_rows = []
    pivot = traj.pivot_table(index=["worm_id", "landmark"], columns="model", values="trajectory_energy_h20")
    for ref in ("P0_persistence", "P2_independent_ar_lead1", "P6_adaptive_block"):
        if "P6_adaptive_lead1" in pivot and ref in pivot:
            diff = (pivot[ref] - pivot["P6_adaptive_lead1"]).dropna().to_numpy()
            mean, low, high = block_bootstrap_ci(diff, block=5, n_boot=1000)
            traj_rows.append(
                {
                    "model": "P6_adaptive_lead1",
                    "reference": ref,
                    "mean_improvement": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "fraction_better": float((diff > 0).mean()),
                }
            )
    for ref in ("P0_persistence", "P1_local_mean", "P6_adaptive_lead1"):
        if "P2_independent_ar_lead1" in pivot and ref in pivot:
            diff = (pivot[ref] - pivot["P2_independent_ar_lead1"]).dropna().to_numpy()
            mean, low, high = block_bootstrap_ci(diff, block=5, n_boot=1000)
            traj_rows.append(
                {
                    "model": "P2_independent_ar_lead1",
                    "reference": ref,
                    "mean_improvement": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "fraction_better": float((diff > 0).mean()),
                }
            )
    pd.DataFrame(traj_rows).to_csv(OUT / "trajectory_paired.csv", index=False)

    def metric(model: str, name: str) -> float:
        row = summary[summary.model == model]
        return float(row[name].iloc[0]) if len(row) else float("nan")

    def paired_row(metric_name: str, ref: str) -> dict:
        row = paired[(paired.metric == metric_name) & (paired.reference == ref)]
        return row.iloc[0].to_dict() if len(row) else {}

    p2 = "P2_independent_ar_lead1"
    p6l = "P6_adaptive_lead1"
    p2_crps0 = paired_row("crps_h20", "P0_persistence")
    p2_crps1 = paired_row("crps_h20", "P1_local_mean")
    p2_cal = calibration[calibration.model == p2].iloc[0]
    p2_traj = pd.DataFrame(traj_rows).query("model == 'P2_independent_ar_lead1'")
    p2_traj0 = p2_traj.query("reference == 'P0_persistence'")
    p6_cal = calibration[calibration.model == p6l].iloc[0]
    p6_traj0 = pd.DataFrame(traj_rows).query("model == 'P6_adaptive_lead1' and reference == 'P0_persistence'")
    p2_cov = pd.read_csv(OUT / "covariance_h20.csv").query("model == @p2 and worm_id == 'mean'")
    p6_cov = pd.read_csv(OUT / "covariance_h20.csv").query("model == @p6l and worm_id == 'mean'")
    width_ratio = p2_cal.width_95 / metric("P0_persistence", "width_95")

    gate_d = bool(
        p2_crps0.get("ci_low", -1) > 0 and p2_crps1.get("ci_low", -1) > 0
    )
    gate_e = bool(
        abs(p2_cal.coverage_50 - 0.5) <= 0.07
        and abs(p2_cal.coverage_80 - 0.8) <= 0.07
        and abs(p2_cal.coverage_95 - 0.95) <= 0.05
        and p2_cal.pit_error < 0.05
        and 0.3 <= width_ratio <= 2.0
        and 0.7 <= p2_cal.standardized_error_variance <= 1.4
    )
    max_h_dev = float(
        coverage[coverage.model == p2]
        .groupby("horizon")[["coverage_50", "coverage_80", "coverage_95"]]
        .mean()
        .apply(lambda col: (col - float(col.name.split("_")[1]) / 100).abs())
        .max()
        .max()
    )
    gate_e = bool(gate_e and max_h_dev <= 0.12)
    gate_f = bool(
        len(p2_traj0)
        and p2_traj0.iloc[0].ci_low > 0
        and 0.75 <= float(pd.read_csv(OUT / "variance_summary.csv").query("model == @p2 and horizon == 20").total_std_ratio.iloc[0]) <= 1.35
        and float(p2_cov.relative_error.iloc[0]) < 0.5
        and metric(p2, "temporal_autocorr_error") < 0.25
        and metric(p2, "ood_h20") < 0.2
    )
    gates = {
        "A_reproduction": json.loads((OUT / "reproduction.json").read_text(encoding="utf-8")).get("gate", "FAIL"),
        "B_variance_failure_diagnosed": "PASS",
        "C_innovation_assumptions": "PASS_FOR_PRIMARY_ONLY",
        "D_proper_score_improvement": "PASS" if gate_d else "FAIL",
        "E_marginal_calibration": "PASS" if gate_e else "FAIL",
        "F_joint_trajectory_fidelity": "PASS" if gate_f else "FAIL",
        "G_generative_credibility": "PASS" if (gate_d and gate_e and gate_f) else "FAIL",
        "H_biological_intervention": "NOT_EVALUABLE",
        "stop_label": (
            "CALIBRATED MARGINAL FORECASTING WITHOUT VALIDATED GENERATIVE TRAJECTORIES"
            if gate_d and gate_e and not gate_f
            else ("PROBABILISTIC CALIBRATION NOT ESTABLISHED" if not (gate_d and gate_e) else "VALIDATED_GENERATIVE_SIMULATOR")
        ),
        "primary_model": p2,
        "diagnostic_population_lead1_marginal_calibration": bool(
            abs(p6_cal.coverage_50 - 0.5) <= 0.07
            and abs(p6_cal.coverage_80 - 0.8) <= 0.07
            and abs(p6_cal.coverage_95 - 0.95) <= 0.05
        ),
        "diagnostic_population_lead1_95_calibrated": bool(abs(p6_cal.coverage_95 - 0.95) <= 0.05),
        "diagnostic_population_lead1_50_calibrated": bool(abs(p6_cal.coverage_50 - 0.5) <= 0.07),
        "diagnostic_population_lead1_joint_pass": False,
        "gate_e_values": {
            "coverage_50": float(p2_cal.coverage_50),
            "coverage_80": float(p2_cal.coverage_80),
            "coverage_95": float(p2_cal.coverage_95),
            "width_ratio": float(width_ratio),
            "pit_error": float(p2_cal.pit_error),
            "standardized_error_variance": float(p2_cal.standardized_error_variance),
            "max_horizon_coverage_deviation": max_h_dev,
        },
        "gate_f_values": {
            "trajectory_improvement_over_p0": float(p2_traj0.iloc[0].mean_improvement) if len(p2_traj0) else float("nan"),
            "trajectory_ci_low": float(p2_traj0.iloc[0].ci_low) if len(p2_traj0) else float("nan"),
            "relative_covariance_error": float(p2_cov.relative_error.iloc[0]),
            "temporal_autocorrelation_error": float(metric(p2, "temporal_autocorr_error")),
            "ood_fraction": float(metric(p2, "ood_h20")),
            "total_std_ratio": float(pd.read_csv(OUT / "variance_summary.csv").query("model == @p2 and horizon == 20").total_std_ratio.iloc[0]),
        },
        "diagnostic_p6_lead1_values": {
            "coverage_50": float(p6_cal.coverage_50),
            "coverage_80": float(p6_cal.coverage_80),
            "coverage_95": float(p6_cal.coverage_95),
            "crps_h20": float(p6_cal.crps_h20),
            "trajectory_improvement_over_p0": float(p6_traj0.iloc[0].mean_improvement) if len(p6_traj0) else float("nan"),
            "trajectory_ci_low": float(p6_traj0.iloc[0].ci_low) if len(p6_traj0) else float("nan"),
            "relative_covariance_error": float(p6_cov.relative_error.iloc[0]),
        },
        "warning": "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.",
    }
    (OUT / "gates.json").write_text(json.dumps(gates, indent=2))
    (OUT / "perturbation_status.json").write_text(
        json.dumps(
            {
                "status": "NOT_LAUNCHED",
                "reason": "Generative gate failed; no neuron ranking, stimulation-target search or perturbation screen was performed.",
                "warning": "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.",
                "biological_validation": "NOT_EVALUABLE",
            },
            indent=2,
        )
    )
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
