"""Resume common-state post-processing from the saved per-landmark frame.

The full run writes ``model_per_landmark.csv`` before the expensive surrogate
stage.  This script rebuilds the remaining summaries, gates and synthetic audit
without refitting the Kato models.  It is useful after a late-stage write failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import kato_common_state_main as M  # noqa: E402

from neurobranch.kato_common_state.synthetic import evaluate_system, simulate_system  # noqa: E402
from neurobranch.kato_interactions.metrics import paired_table  # noqa: E402

OUT = ROOT / "reports" / "simulator" / "kato_common_state"


def main() -> None:
    frame = pd.read_csv(OUT / "model_per_landmark.csv")
    primary_models = [
        "B0_persistence",
        "B1_matched_own",
        "B2_own_trend",
        "B4_matched_own_500",
        "B5_common_k10",
        "B3_population",
        "B6_coefficient_ablation",
    ]
    summary = (
        frame[frame["model"].isin(primary_models)]
        .groupby("model")
        .agg(
            energy_h20=("pca10_energy_h20", "mean"),
            trajectory_energy_h20=("pca10_trajectory_energy_h20", "mean"),
            mse_mean_h20=("pca10_mse_mean_h20", "mean"),
            coverage_50=("pca10_coverage_50", "mean"),
            coverage_80=("pca10_coverage_80", "mean"),
            coverage_95=("pca10_coverage_95", "mean"),
            neural_cov_error=("pca10_neural_cov_error", "mean"),
            autocorr_error=("pca10_autocorr_error", "mean"),
            neuron_energy_h20=("neuron_energy_h20", "mean"),
            neuron_mse_h20=("neuron_mse_mean_h20", "mean"),
            factor_energy_h20=("factor_energy_h20", "mean"),
            residual_energy_h20=("residual_energy_h20", "mean"),
        )
        .reset_index()
    )
    summary["delta_energy_vs_B0"] = summary["energy_h20"] - float(
        summary.loc[summary["model"] == "B0_persistence", "energy_h20"].iloc[0]
    )
    summary.to_csv(OUT / "model_summary.csv", index=False)

    paired = paired_table(
        frame,
        reference="B3_population",
        models=[m for m in primary_models if m != "B3_population"],
        metric="pca10_energy_h20",
        blocks=(3, 5, 10),
    )
    paired["metric"] = "pca10_energy_h20"
    paired["sign_convention"] = "reference minus model; positive means the model has lower energy"
    residual_paired = paired_table(
        frame,
        reference="B5_common_k10",
        models=["B3_population"],
        metric="pca10_energy_h20",
        blocks=(3, 5, 10),
    )
    residual_paired["metric"] = "pca10_energy_h20"
    residual_paired["sign_convention"] = "B5 minus B3; positive means B3 has lower energy"
    residual_paired.to_csv(OUT / "residual_cross_neuron.csv", index=False)
    extra = []
    for metric in ("pca10_mse_mean_h20", "pca10_trajectory_energy_h20", "neuron_energy_h20", "neuron_mse_mean_h20"):
        part = paired_table(
            frame,
            reference="B5_common_k10",
            models=["B3_population"],
            metric=metric,
            blocks=(3, 5, 10),
        )
        if not part.empty:
            extra.append(part)
    if extra:
        paired = pd.concat([paired, *extra], ignore_index=True)
    paired.to_csv(OUT / "paired_contrasts.csv", index=False)

    decomposition_rows = []
    contrasts = [
        ("own_history_B1m_vs_B0", "B0_persistence", "B1_matched_own"),
        ("local_drift_B2_vs_B1m", "B1_matched_own", "B2_own_trend"),
        ("window_B4m_vs_B1m", "B1_matched_own", "B4_matched_own_500"),
        ("common_factors_B5_vs_B4m", "B4_matched_own_500", "B5_common_k10"),
        ("residual_cross_B3_vs_B5", "B5_common_k10", "B3_population"),
        ("population_B3_vs_B4m", "B4_matched_own_500", "B3_population"),
    ]
    for name, worse, better in contrasts:
        one = paired_table(frame, reference=worse, models=[better], metric="pca10_energy_h20", blocks=(5,))
        if not one.empty:
            row = one.iloc[0].to_dict()
            row["contrast"] = name
            row["positive_means"] = better
            decomposition_rows.append(row)
    decomp = pd.DataFrame(decomposition_rows)
    denom = float(decomp.loc[decomp["contrast"] == "population_B3_vs_B4m", "mean_improvement"].iloc[0])
    for _name, key in (
        ("common_factor_fraction_of_population_gain", "common_factors_B5_vs_B4m"),
        ("residual_fraction_of_population_gain", "residual_cross_B3_vs_B5"),
    ):
        value = float(decomp.loc[decomp["contrast"] == key, "mean_improvement"].iloc[0])
        decomp.loc[decomp["contrast"] == key, "fraction_of_population_gain"] = (
            value / denom if abs(denom) > 1e-12 else float("nan")
        )
    decomp.to_csv(OUT / "decomposition.csv", index=False)

    session_effect = (
        frame[frame["model"].isin(["B3_population", "B5_common_k10"])]
        .pivot_table(index=["session", "landmark"], columns="model", values="pca10_energy_h20")
        .dropna()
    )
    session_effect["improvement_B3_over_B5"] = session_effect["B5_common_k10"] - session_effect["B3_population"]
    session_effect = (
        session_effect.groupby("session")["improvement_B3_over_B5"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_energy_improvement", "count": "n_landmarks"})
    )
    session_effect.to_csv(OUT / "session_effects.csv", index=False)

    rep_rows = []
    for metric, label in (
        ("pca10_energy_h20", "pca10_energy"),
        ("pca10_mse_mean_h20", "pca10_mse"),
        ("neuron_energy_h20", "neuron_energy"),
        ("neuron_mse_mean_h20", "neuron_mse"),
    ):
        one = paired_table(
            frame, reference="B5_common_k10", models=["B3_population"], metric=metric, blocks=(5,)
        )
        if not one.empty:
            row = one.iloc[0].to_dict()
            row["representation"] = label
            rep_rows.append(row)
    pd.DataFrame(rep_rows).to_csv(OUT / "representation_sensitivity.csv", index=False)

    horizon_rows = []
    for model in ("B0_persistence", "B1_matched_own", "B2_own_trend", "B4_matched_own_500", "B5_common_k10", "B3_population"):
        sub = frame[frame["model"] == model]
        for step in (1, 5, 10, 20):
            horizon_rows.append(
                {
                    "model": model,
                    "horizon": step,
                    "mean_energy": float(sub[f"pca10_energy_h{step}"].mean()),
                    "mean_mse": float(sub[f"pca10_mse_h{step}"].mean()),
                }
            )
    pd.DataFrame(horizon_rows).to_csv(OUT / "horizon_profile.csv", index=False)

    if (OUT / "surrogate_draws.csv").exists():
        surrogate = pd.read_csv(OUT / "surrogate_draws.csv")
        by_landmark = (
            surrogate.groupby(["session", "worm_id", "landmark", "method"])
            .agg(energy_h20=("energy_h20", "mean"), mse_mean_h20=("mse_mean_h20", "mean"))
            .reset_index()
        )
        by_landmark["model"] = "surrogate_" + by_landmark["method"]
        real = frame[frame["model"] == "B3_population"][
            ["session", "worm_id", "landmark", "pca10_energy_h20", "pca10_mse_mean_h20"]
        ].rename(columns={"pca10_energy_h20": "energy_h20", "pca10_mse_mean_h20": "mse_mean_h20"})
        real["method"] = "observed_B3"
        real["model"] = "B3_population"
        columns = ["session", "worm_id", "landmark", "energy_h20", "mse_mean_h20", "method", "model"]
        combined = pd.concat([real[columns], by_landmark[columns]], ignore_index=True)
        surrogate_summary = (
            by_landmark.groupby("method")
            .agg(energy_h20=("energy_h20", "mean"), mse_mean_h20=("mse_mean_h20", "mean"), n=("energy_h20", "size"))
            .reset_index()
        )
        surrogate_summary["delta_vs_B3"] = surrogate_summary["energy_h20"] - float(real["energy_h20"].mean())
        surrogate_summary.to_csv(OUT / "surrogate_summary.csv", index=False)
        sp_rows = []
        for method in M.SURROGATE_METHODS:
            one = paired_table(
                combined,
                reference=f"surrogate_{method}",
                models=["B3_population"],
                metric="energy_h20",
                blocks=(3, 5, 10),
            )
            if not one.empty:
                sp_rows.append(one)
        sp = pd.concat(sp_rows, ignore_index=True) if sp_rows else pd.DataFrame()
        sp["sign_convention"] = "surrogate minus B3; positive means B3 has lower energy"
        sp.to_csv(OUT / "surrogate_paired.csv", index=False)

    synthetic_path = OUT / "synthetic_audit.csv"
    if not synthetic_path.exists():
        synthetic_rows = []
        for scenario in ("independent", "common", "coupled", "coupled_common", "time_varying_common"):
            for seed in (11, 23, 37):
                synthetic_rows.append(evaluate_system(simulate_system(scenario, seed)))
        synthetic = pd.DataFrame(synthetic_rows)
        synthetic.to_csv(synthetic_path, index=False)
        synthetic_summary = (
            synthetic.groupby("scenario")
            .agg(
                b3_minus_b5_3_mse_gain=("b3_minus_b5_3_mse_gain", "mean"),
                b3_minus_b5_3_sd=("b3_minus_b5_3_mse_gain", "std"),
                b3_minus_b5_10_mse_gain=("b3_minus_b5_10_mse_gain", "mean"),
                b5_10_minus_b1_mse_gain=("b5_10_minus_b1_mse_gain", "mean"),
                b3_offdiag_norm=("b3_offdiag_norm", "mean"),
                true_offdiag_norm=("true_offdiag_norm", "mean"),
                n=("b3_minus_b5_3_mse_gain", "size"),
            )
            .reset_index()
        )
        synthetic_summary.to_csv(OUT / "synthetic_summary.csv", index=False)
        common = synthetic[synthetic["scenario"] == "common"]["b3_minus_b5_3_mse_gain"]
        direct = synthetic[synthetic["scenario"].isin(["coupled", "coupled_common"])]["b3_minus_b5_3_mse_gain"]
        status = {
            "status": "PASS"
            if float(np.mean(direct)) > 0 and float(np.mean(common)) <= 0
            else ("PARTIAL" if float(np.mean(direct)) > 0 else "FAIL"),
            "direct_mean_b3_minus_b5_3_mse_gain": float(np.mean(direct)),
            "common_mean_b3_minus_b5_3_mse_gain": float(np.mean(common)),
            "direct_seeds_positive": int((direct > 0).sum()),
            "n_direct": int(len(direct)),
        }
        (OUT / "synthetic_verification.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    reproduction = json.loads((OUT / "reproduction_summary.json").read_text(encoding="utf-8"))
    gates: dict[str, object] = {}
    gates["G1_historical_reproduction"] = {
        "status": reproduction["status"],
        "max_abs_difference": reproduction["max_abs_reproduction_difference"],
        "j2_j5_equivalent_mean": reproduction["j2_j5_equivalent_mean"],
    }
    counts = frame.groupby("model").size().to_dict()
    required = ["B3_population", "B4_matched_own_500", "B5_common_k10", "B1_matched_own"]
    gates["G2_fair_matched_baselines"] = {
        "status": "PASS" if all(counts.get(m, 0) == 175 for m in required) else "FAIL",
        "counts": {m: int(counts.get(m, 0)) for m in required},
        "common_innovation": "J5 sampled once per landmark and reused",
        "windows": {"B3": 500, "B4m": 500, "B5-primary": 500},
    }
    factor_rows = pd.read_csv(OUT / "factor_variance.csv")
    gates["G3_common_factor_explanation"] = {
        "status": "PASS" if len(factor_rows) == 20 else "FAIL",
        "factor_rows": int(len(factor_rows)),
        "ks": [1, 3, 5, 10],
    }
    g4row = residual_paired[residual_paired["block"] == 5].iloc[0].to_dict()
    g4 = (
        "PASS"
        if g4row["mean_improvement"] > 0 and g4row["ci_low"] > 0 and g4row["fraction_better"] > 0.5
        else ("PARTIAL" if g4row["mean_improvement"] > 0 else "FAIL")
    )
    gates["G4_residual_cross_neuron_information"] = {"status": g4, **g4row}
    syn_status = json.loads((OUT / "synthetic_verification.json").read_text(encoding="utf-8"))["status"]
    gates["G5_synthetic_specificity"] = {"status": syn_status}
    session_positive = int((session_effect["mean_energy_improvement"] > 0).sum())
    rep = pd.read_csv(OUT / "representation_sensitivity.csv")
    pca_dir = float(rep.loc[rep["representation"] == "pca10_energy", "mean_improvement"].iloc[0])
    neuron_dir = float(rep.loc[rep["representation"] == "neuron_energy", "mean_improvement"].iloc[0])
    if session_positive >= 4 and pca_dir > 0 and neuron_dir > 0:
        g6 = "PASS"
    elif session_positive >= 3 and (pca_dir > 0 or neuron_dir > 0):
        g6 = "PARTIAL"
    else:
        g6 = "FAIL"
    gates["G6_robustness"] = {
        "status": g6,
        "positive_sessions": session_positive,
        "n_sessions": int(len(session_effect)),
        "pca10_energy_improvement": pca_dir,
        "neuron_energy_improvement": neuron_dir,
    }
    gates["G7_structural_identifiability"] = {
        "status": "FAIL",
        "reason": "observational prediction cannot identify direct coupling or common-input structure",
    }
    gates["G8_biological_intervention_validity"] = {
        "status": "NOT_EVALUABLE",
        "reason": "no matched sham-controlled perturbation data",
    }
    (OUT / "gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
    analysis = {
        "dataset": "Kato 2015 WT_NoStim only",
        "sessions": 5,
        "frames": 15662,
        "session_neurons": {str(k): int(v) for k, v in frame.groupby("session")["n_neurons"].first().items()},
        "landmarks": int((frame["model"] == "B3_population").sum()),
        "members": 100,
        "horizon": 20,
        "primary_k": 10,
        "ks": [1, 3, 5, 10],
        "surrogate_draws": 6,
        "note": "post-processed from the saved per-landmark frame after a late-stage surrogate write failure",
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print("post-processing complete", gates["G4_residual_cross_neuron_information"]["status"])


if __name__ == "__main__":
    main()
