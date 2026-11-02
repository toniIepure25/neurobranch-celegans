"""Kato common-state predictive controls.

This script does not add an architecture.  It fits a small fixed set of causal
linear models on the existing Kato WT_NoStim recordings, reuses the historical
J5 innovations, and asks whether full population history retains H20 predictive
information beyond own-neuron history and train-only common factors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import jd_evaluate as JD  # noqa: E402

from neurobranch.adaptive.evaluation import evaluation_landmarks  # noqa: E402
from neurobranch.diagnostics.common import load_nostim  # noqa: E402
from neurobranch.kato_common_state.evaluation import score_ensemble  # noqa: E402
from neurobranch.kato_common_state.factors import FactorModel, variance_table  # noqa: E402
from neurobranch.kato_common_state.models import (  # noqa: E402
    coefficient_ablation,
    fit_window_common,
    fit_window_own,
    fit_window_trend,
    rollout_common_factor,
    rollout_matched,
    rollout_persistence,
    rollout_trend,
)
from neurobranch.kato_common_state.surrogates import fit_common_state_surrogate  # noqa: E402
from neurobranch.kato_common_state.synthetic import evaluate_system, simulate_system  # noqa: E402
from neurobranch.kato_interactions.metrics import paired_table, session_block_ci  # noqa: E402
from neurobranch.kato_interactions.models import FrozenLinearModel, fit_window  # noqa: E402
from neurobranch.kato_interactions.surrogates import fit_shuffled_window  # noqa: E402
from neurobranch.probabilistic.baselines import IndependentARModel  # noqa: E402
from neurobranch.probabilistic.metrics import energy_score  # noqa: E402
from neurobranch.uncertainty_calibration.trajectory import trajectory_energy_score  # noqa: E402

OUT = ROOT / "reports" / "simulator" / "kato_common_state"
PRIOR = ROOT / "reports" / "simulator" / "kato_interactions"
MEMBERS = 100
HORIZON = 100
PRIMARY_HORIZON = 20
STRIDE = 10
KS = (1, 3, 5, 10)
PRIMARY_K = 10
SURROGATE_DRAWS = 6
SURROGATE_METHODS = (
    "circular_full",
    "block_full",
    "common_preserved_residual_shift",
    "residual_preserved_factor_shift",
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=False), encoding="utf-8")


def rollout_linear(model: FrozenLinearModel, x0: np.ndarray, eps: np.ndarray) -> np.ndarray:
    x = np.repeat(np.asarray(x0, dtype=np.float64)[None, :], eps.shape[0], axis=0)
    out = np.empty((eps.shape[0], eps.shape[1], x.shape[1]), dtype=np.float64)
    for step in range(eps.shape[1]):
        x = x @ model.A.T + model.b + eps[:, step]
        out[:, step] = x
    return out


def exact_pca_metrics(ensemble: np.ndarray, truth: np.ndarray, pca) -> dict[str, float]:
    """Historical scoring path, preserving the original float32 inputs."""
    ens = np.asarray(ensemble)
    raw_truth = np.asarray(truth)
    proj = pca.transform(ens.reshape(-1, ens.shape[-1])).reshape(ens.shape[0], ens.shape[1], -1)
    truth_proj = pca.transform(raw_truth)
    return {
        "pca10_energy_h20": float(energy_score(proj[:, 19], truth_proj[19])),
        "pca10_trajectory_energy_h20": float(trajectory_energy_score(proj[:, :20], truth_proj[:20])),
        "pca10_mse_mean_h20": float(np.mean((proj[:, 19].mean(axis=0) - truth_proj[19]) ** 2)),
    }


def _model_config() -> pd.DataFrame:
    rows = [
        ("B0_persistence", "none", np.nan, 0, "none", "none", "common_J5", "persistence"),
        ("B1_historical", "historical independent AR", 200, 2, "weak (1 added to sum of squares)", "unpenalized", "J1 exact", "JD J1"),
        ("B1_matched_own", "matched independent AR", 200, 2, "ridge 1.0", "unpenalized", "common_J5", "own lag only"),
        ("B2_own_trend", "own AR + local linear drift", 200, 3, "ridge 1.0", "unpenalized", "common_J5", "own lag + target-frame time"),
        ("B4_matched_own_500", "matched independent AR", 500, 2, "ridge 1.0", "unpenalized", "common_J5", "own lag only, matched window"),
        ("B5_common_k1", "own AR + 1 common factor", 500, 3, "ridge 1.0", "unpenalized", "common_J5", "factor VAR + own lag"),
        ("B5_common_k3", "own AR + 3 common factors", 500, 5, "ridge 1.0", "unpenalized", "common_J5", "factor VAR + own lag"),
        ("B5_common_k5", "own AR + 5 common factors", 500, 7, "ridge 1.0", "unpenalized", "common_J5", "factor VAR + own lag"),
        ("B5_common_k10", "own AR + 10 common factors", 500, 12, "ridge 1.0", "unpenalized", "common_J5", "factor VAR + own lag, primary control"),
        ("B3_population", "full population VAR", 500, "n*(n+1)", "ridge 1.0", "unpenalized", "common_J5", "full lag-one population operator"),
        ("B6_coefficient_ablation", "B3 with off-diagonals zeroed", 500, "n diagonal + n intercept", "inherited", "unpenalized", "common_J5", "coefficient ablation, not refit"),
        ("B4r_historical", "historical independent AR", 500, 2, "weak (1 added to sum of squares)", "unpenalized", "J5 exact", "reproduction only"),
        ("J2_population_diag", "historical population VAR", 500, "n*(n+1)", "ridge 1.0", "unpenalized", "J2 exact", "reproduction only"),
        ("J5_population_lead1_boot", "historical population VAR", 500, "n*(n+1)", "ridge 1.0", "unpenalized", "J5 exact", "reproduction only"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "model",
            "family",
            "window",
            "n_parameters_per_neuron_or_total",
            "penalty",
            "intercept",
            "innovation",
            "role",
        ],
    )


def _load_prior() -> tuple[pd.DataFrame, pd.DataFrame]:
    exact = pd.read_csv(PRIOR / "exact_reproduction_per_landmark.csv")
    cross = pd.read_csv(PRIOR / "cross_neuron_per_landmark.csv")
    return exact, cross


def _prior_rows(exact: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    keep = ["session", "worm_id", "landmark", "model", "energy_h20", "trajectory_energy_h20", "mse_mean_h20"]
    a = exact[exact["model"].isin(["J1_independent_diag", "J2_population_diag", "J5_population_lead1_boot"])][keep].copy()
    b = cross[cross["model"].isin(["B3_population", "B4_refit_diagonal"])][keep].copy()
    b["model"] = b["model"].replace({"B4_refit_diagonal": "B4r_historical"})
    return pd.concat([a, b], ignore_index=True)


def _stage_a_compare(repro: pd.DataFrame, prior: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mapping = {
        "B1_historical": "J1_independent_diag",
        "J2_population_diag": "J2_population_diag",
        "J5_population_lead1_boot": "J5_population_lead1_boot",
        "B3_population": "B3_population",
        "B4r_historical": "B4r_historical",
    }
    rows = []
    for new_name, old_name in mapping.items():
        left = repro[repro["model"] == new_name][
            [
                "session",
                "landmark",
                "pca10_energy_h20",
                "pca10_trajectory_energy_h20",
                "pca10_mse_mean_h20",
            ]
        ].rename(
            columns={
                "pca10_energy_h20": "repro_energy",
                "pca10_trajectory_energy_h20": "repro_trajectory",
                "pca10_mse_mean_h20": "repro_mse",
            }
        )
        right = prior[prior["model"] == old_name][
            ["session", "landmark", "energy_h20", "trajectory_energy_h20", "mse_mean_h20"]
        ].rename(
            columns={
                "energy_h20": "prior_energy",
                "trajectory_energy_h20": "prior_trajectory",
                "mse_mean_h20": "prior_mse",
            }
        )
        merged = left.merge(right, on=["session", "landmark"], how="inner")
        for metric in ("energy", "trajectory", "mse"):
            diff = merged[f"repro_{metric}"] - merged[f"prior_{metric}"]
            rows.append(
                {
                    "model": new_name,
                    "prior_model": old_name,
                    "metric": metric,
                    "n": int(len(diff)),
                    "max_abs_diff": float(np.max(np.abs(diff))) if len(diff) else float("nan"),
                    "mean_abs_diff": float(np.mean(np.abs(diff))) if len(diff) else float("nan"),
                }
            )
    table = pd.DataFrame(rows)
    max_abs = float(table.loc[table["metric"] == "energy", "max_abs_diff"].max())
    max_mse = float(table.loc[table["metric"] == "mse", "max_abs_diff"].max())
    max_trajectory = float(table.loc[table["metric"] == "trajectory", "max_abs_diff"].max())
    j2 = repro[repro["model"] == "J2_population_diag"].set_index(["session", "landmark"])
    j5 = repro[repro["model"] == "J5_population_lead1_boot"].set_index(["session", "landmark"])
    joined = j2[["pca10_energy_h20", "pca10_trajectory_energy_h20"]].join(
        j5[["pca10_energy_h20", "pca10_trajectory_energy_h20"]], lsuffix="_j2", rsuffix="_j5"
    ).dropna()
    signed = joined["pca10_energy_h20_j2"] - joined["pca10_energy_h20_j5"]
    diff_frame = pd.DataFrame(
        {
            "session": [idx[0] for idx in joined.index],
            "landmark": [idx[1] for idx in joined.index],
            "improvement": signed.to_numpy(),
        }
    )
    ci = session_block_ci(diff_frame, "improvement", session="session", block=5)
    summary = {
        "status": "PASS" if np.isfinite(max_abs) and max_abs < 1e-6 else "FAIL",
        "max_abs_reproduction_difference": max_abs,
        "max_abs_mse_difference": max_mse,
        "max_abs_trajectory_difference": max_trajectory,
        "tolerance": 1e-6,
        "tolerance_applies_to": "H20 energy",
        "n_models": int(len(mapping)),
        "j2_j5_mean_signed_energy_difference": float(signed.mean()),
        "j2_j5_mean_abs_energy_difference": float(signed.abs().mean()),
        "j2_j5_energy_ci_low": ci["lo"],
        "j2_j5_energy_ci_high": ci["hi"],
        "j2_j5_energy_correlation": float(joined["pca10_energy_h20_j2"].corr(joined["pca10_energy_h20_j5"])),
        "j2_j5_mean_signed_trajectory_difference": float(
            (joined["pca10_trajectory_energy_h20_j2"] - joined["pca10_trajectory_energy_h20_j5"]).mean()
        ),
        "j2_j5_equivalent_mean": bool(abs(float(signed.mean())) < 0.01),
    }
    return table, summary


def _session_landmarks(d) -> np.ndarray:
    return evaluation_landmarks(d, horizon_max=HORIZON, stride=STRIDE)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surrogate-draws", type=int, default=SURROGATE_DRAWS)
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--skip-surrogates", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    cfg, data, recordings = load_nostim()
    prior_exact, prior_cross = _load_prior()
    prior = _prior_rows(prior_exact, prior_cross)

    rows: list[dict] = []
    repro_rows: list[dict] = []
    surrogate_rows: list[dict] = []
    surrogate_diag_rows: list[dict] = []
    factor_rows: list[dict] = []
    factor_cache: dict[int, FactorModel] = {}
    all_factors: dict[int, np.ndarray] = {}
    pca_cache: dict[int, object] = {}
    b3_cache: dict[tuple[int, int], FrozenLinearModel] = {}
    b3_energy: dict[tuple[int, int], float] = {}
    session_sizes: dict[int, int] = {}

    for d in data:
        sid = int(d.worm_id)
        pca, series, meta = JD.prepare(d)
        factor_model = FactorModel(
            pca=pca,
            n_features=int(d.X.shape[1]),
            n_components=int(pca.n_components_),
            fit_record={"fit_split": "train", "source": "JD.prepare PCA"},
        )
        factor_cache[sid] = factor_model
        all_factors[sid] = factor_model.scores(d.X, pca.n_components_)
        pca_cache[sid] = pca
        session_sizes[sid] = int(d.X.shape[1])
        for row in variance_table(factor_model, KS):
            factor_rows.append({"session": sid, **row})
        cache: dict = {}
        temporal_cache: dict = {}
        landmarks = _session_landmarks(d)
        for t in landmarks:
            t = int(t)
            x0 = np.asarray(d.X[t], dtype=np.float64)
            truth = np.asarray(d.X[t + 1 : t + HORIZON + 1], dtype=np.float64)
            truth_raw = d.X[t + 1 : t + HORIZON + 1]
            common_eps = None
            exact_specs = (
                JD.Spec("J1_historical", "independent_ar", "independent_ar", "diag"),
                JD.Spec("J2_population_diag", "var", "population", "diag"),
                JD.Spec("J5_population_lead1_boot", "var", "population", "block", 10),
            )
            exact_models: dict[str, object] = {}
            for spec in exact_specs:
                fitted = JD.fit_mean(spec, t, cache, d.X)
                innovation = JD.make_innovation(spec, series[spec.series], t, temporal_cache)
                eps = innovation.sample(MEMBERS, HORIZON, np.random.default_rng(7000 + sid * 10000 + t))
                ens = JD.rollout(spec.mean, fitted, x0, HORIZON, eps)
                exact_models[spec.name] = (fitted, eps, ens)
                if spec.name == "J5_population_lead1_boot":
                    common_eps = eps
                metrics = exact_pca_metrics(ens, truth_raw, pca)
                model_name = "B1_historical" if spec.name == "J1_historical" else spec.name
                repro_rows.append(
                    {
                        "session": sid,
                        "worm_id": sid,
                        "landmark": t,
                        "model": model_name,
                        **metrics,
                    }
                )
                rows.append(
                    {
                        "session": sid,
                        "worm_id": sid,
                        "landmark": t,
                        "n_neurons": session_sizes[sid],
                        "model": model_name,
                        **metrics,
                    }
                )
            assert common_eps is not None
            b3 = fit_window(d.X, t, window=500, ridge=1.0, spectral_cap=0.995)
            if b3 is None:
                continue
            b3_linear = FrozenLinearModel(A=b3.A, b=b3.b, fit_record=dict(b3.fit_record))
            b3_cache[(sid, t)] = b3_linear
            b4r = IndependentARModel(window=500).fit_at(d.X, t)
            b3_ens = rollout_linear(b3_linear, x0, common_eps)
            b3_hist = rollout_linear(b3_linear, d.X[t], common_eps)
            if b4r is not None:
                b4r_hist = np.empty((MEMBERS, HORIZON, x0.size), dtype=np.float64)
                z_hist = np.repeat(d.X[t][None, :], MEMBERS, axis=0)
                for step in range(HORIZON):
                    z_hist = b4r.a * z_hist + b4r.b + common_eps[:, step]
                    b4r_hist[:, step] = z_hist
            else:
                b4r_hist = None
            repro_rows.append(
                {
                    "session": sid,
                    "worm_id": sid,
                    "landmark": t,
                    "model": "B3_population",
                    **exact_pca_metrics(b3_hist, truth_raw, pca),
                }
            )
            if b4r_hist is not None:
                repro_rows.append(
                    {
                        "session": sid,
                        "worm_id": sid,
                        "landmark": t,
                        "model": "B4r_historical",
                        **exact_pca_metrics(b4r_hist, truth_raw, pca),
                    }
                )
            metrics = score_ensemble(b3_ens, truth, pca, factor_k=PRIMARY_K)
            rows.append(
                {
                    "session": sid,
                    "worm_id": sid,
                    "landmark": t,
                    "n_neurons": session_sizes[sid],
                    "model": "B3_population",
                    **metrics,
                }
            )
            b3_energy[(sid, t)] = float(metrics["pca10_energy_h20"])

            # Fixed matched control ladder, all with the same J5 innovations.
            b1 = fit_window_own(d.X, t, window=200, ridge=1.0)
            b2 = fit_window_trend(d.X, t, window=200, ridge=1.0)
            b4 = fit_window_own(d.X, t, window=500, ridge=1.0)
            b6 = coefficient_ablation(b3_linear, d.X, t, window=500)
            ensembles: dict[str, np.ndarray] = {
                "B0_persistence": rollout_persistence(x0, common_eps),
                "B6_coefficient_ablation": rollout_linear(b6, x0, common_eps),
            }
            if b1 is not None:
                ensembles["B1_matched_own"] = rollout_matched(b1, x0, common_eps)
            if b2 is not None:
                ensembles["B2_own_trend"] = rollout_trend(b2, x0, t, common_eps)
            if b4 is not None:
                ensembles["B4_matched_own_500"] = rollout_matched(b4, x0, common_eps)
            for k in KS:
                b5 = fit_window_common(
                    d.X,
                    all_factors[sid],
                    t,
                    k=k,
                    window=500,
                    ridge=1.0,
                )
                if b5 is not None:
                    ensembles[f"B5_common_k{k}"] = rollout_common_factor(
                        b5, x0, all_factors[sid][t, :k], common_eps
                    )
            for model_name, ens in ensembles.items():
                metrics = score_ensemble(ens, truth, pca, factor_k=PRIMARY_K)
                rows.append(
                    {
                        "session": sid,
                        "worm_id": sid,
                        "landmark": t,
                        "n_neurons": session_sizes[sid],
                        "model": model_name,
                        **metrics,
                    }
                )

            if not args.skip_surrogates:
                for method in SURROGATE_METHODS:
                    for draw in range(int(args.surrogate_draws)):
                        seed = 91_000 + sid * 1000 + t * 11 + draw
                        if method in {"circular_full", "block_full"}:
                            model = fit_shuffled_window(
                                d.X,
                                t,
                                window=500,
                                method="circular" if method == "circular_full" else "block",
                                block=25,
                                seed=seed,
                            )
                            diag = {"method": method, "seed": seed}
                        else:
                            model, diag = fit_common_state_surrogate(
                                d.X,
                                t,
                                factors=factor_model,
                                k=PRIMARY_K,
                                method=method,
                                window=500,
                                seed=seed,
                            )
                        if model is None:
                            continue
                        ens = rollout_linear(model, x0, common_eps)
                        metrics = score_ensemble(ens, truth, pca, factor_k=PRIMARY_K)
                        surrogate_rows.append(
                            {
                                "session": sid,
                                "worm_id": sid,
                                "landmark": t,
                                "method": method,
                                "draw": draw,
                                "seed": seed,
                                "energy_h20": metrics["pca10_energy_h20"],
                                "mse_mean_h20": metrics["pca10_mse_mean_h20"],
                                "offdiagonal_energy": float(model.offdiagonal_energy),
                            }
                        )
                        surrogate_diag_rows.append(
                            {
                                "session": sid,
                                "landmark": t,
                                "method": method,
                                "draw": draw,
                                **diag,
                            }
                        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no Kato model rows were produced")
    frame.to_csv(OUT / "model_per_landmark.csv", index=False)
    _model_config().to_csv(OUT / "model_config.csv", index=False)
    pd.DataFrame(factor_rows).to_csv(OUT / "factor_variance.csv", index=False)
    stage_a, reproduction_summary = _stage_a_compare(pd.DataFrame(repro_rows), prior)
    stage_a.to_csv(OUT / "stage_a_comparison.csv", index=False)
    write_json(OUT / "reproduction_summary.json", reproduction_summary)

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

    pair_models = [m for m in primary_models if m != "B3_population"]
    paired = paired_table(
        frame,
        reference="B3_population",
        models=pair_models,
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
        one = paired_table(
            frame,
            reference=worse,
            models=[better],
            metric="pca10_energy_h20",
            blocks=(5,),
        )
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
            frame,
            reference="B5_common_k10",
            models=["B3_population"],
            metric=metric,
            blocks=(5,),
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

    if not args.skip_surrogates and surrogate_rows:
        surrogate = pd.DataFrame(surrogate_rows)
        surrogate.to_csv(OUT / "surrogate_draws.csv", index=False)
        pd.DataFrame(surrogate_diag_rows).to_csv(OUT / "surrogate_diagnostics.csv", index=False)
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
        surrogate_summary["delta_vs_B3"] = surrogate_summary["energy_h20"] - float(
            real["energy_h20"].mean()
        )
        surrogate_summary.to_csv(OUT / "surrogate_summary.csv", index=False)
        sp_rows = []
        for method in SURROGATE_METHODS:
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

    if not args.skip_synthetic:
        synthetic_rows = []
        for scenario in ("independent", "common", "coupled", "coupled_common", "time_varying_common"):
            for seed in (11, 23, 37):
                system = simulate_system(scenario, seed)
                synthetic_rows.append(evaluate_system(system))
        synthetic = pd.DataFrame(synthetic_rows)
        synthetic.to_csv(OUT / "synthetic_audit.csv", index=False)
        synthetic_summary = (
            synthetic.groupby("scenario")
            .agg(
                b3_minus_b5_3_mse_gain=("b3_minus_b5_3_mse_gain", "mean"),
                b3_minus_b5_3_sd=("b3_minus_b5_3_mse_gain", "std"),
                b3_minus_b5_10_mse_gain=("b3_minus_b5_10_mse_gain", "mean"),
                b5_10_minus_b1_mse_gain=("b5_10_minus_b1_mse_gain", "mean"),
                b3_offdiag_norm=("b3_offdiag_norm", "mean"),
                true_offdiag_norm=("true_offdiag_norm", "mean"),
                n=("b3_minus_b5_10_mse_gain", "size"),
            )
            .reset_index()
        )
        synthetic_summary.to_csv(OUT / "synthetic_summary.csv", index=False)
        common = synthetic[synthetic["scenario"] == "common"]["b3_minus_b5_3_mse_gain"]
        direct = synthetic[synthetic["scenario"].isin(["coupled", "coupled_common"])]["b3_minus_b5_3_mse_gain"]
        synthetic_status = {
            "status": "PASS"
            if float(np.mean(direct)) > 0 and float(np.mean(common)) <= 0
            else ("PARTIAL" if float(np.mean(direct)) > 0 else "FAIL"),
            "direct_mean_b3_minus_b5_3_mse_gain": float(np.mean(direct)),
            "common_mean_b3_minus_b5_3_mse_gain": float(np.mean(common)),
            "direct_seeds_positive": int((direct > 0).sum()),
            "n_direct": int(len(direct)),
        }
        write_json(OUT / "synthetic_verification.json", synthetic_status)

    # Gates are evaluated only after all artifacts above exist.
    gates: dict[str, object] = {}
    gates["G1_historical_reproduction"] = {
        "status": reproduction_summary["status"],
        "max_abs_difference": reproduction_summary["max_abs_reproduction_difference"],
        "j2_j5_equivalent_mean": reproduction_summary["j2_j5_equivalent_mean"],
    }
    counts = frame.groupby("model").size().to_dict()
    required = ["B3_population", "B4_matched_own_500", "B5_common_k10", "B1_matched_own"]
    g2 = all(counts.get(m, 0) >= 175 for m in required) and all(
        int(frame.loc[frame["model"] == m, "n_neurons"].nunique()) >= 1 for m in required
    )
    gates["G2_fair_matched_baselines"] = {
        "status": "PASS" if g2 else "FAIL",
        "counts": {m: int(counts.get(m, 0)) for m in required},
        "common_innovation": "J5 sampled once per landmark and reused",
        "windows": {"B3": 500, "B4m": 500, "B5-primary": 500},
    }
    gates["G3_common_factor_explanation"] = {
        "status": "PASS" if len(factor_rows) == 5 * len(KS) and "B5_common_k10" in counts else "FAIL",
        "factor_rows": int(len(factor_rows)),
        "ks": list(KS),
    }
    primary = residual_paired[residual_paired["block"] == 5]
    if primary.empty:
        g4 = "FAIL"
        g4row = {}
    else:
        g4row = primary.iloc[0].to_dict()
        if g4row["mean_improvement"] > 0 and g4row["ci_low"] > 0 and g4row["fraction_better"] > 0.5:
            g4 = "PASS"
        elif g4row["mean_improvement"] > 0:
            g4 = "PARTIAL"
        else:
            g4 = "FAIL"
    gates["G4_residual_cross_neuron_information"] = {"status": g4, **g4row}
    syn_status = "NOT_RUN"
    if (OUT / "synthetic_verification.json").exists():
        syn_status = json.loads((OUT / "synthetic_verification.json").read_text())["status"]
    gates["G5_synthetic_specificity"] = {"status": syn_status}
    session_positive = int((session_effect["mean_energy_improvement"] > 0).sum())
    neuron_row = pd.read_csv(OUT / "representation_sensitivity.csv")
    pca_dir = float(
        neuron_row.loc[neuron_row["representation"] == "pca10_energy", "mean_improvement"].iloc[0]
    )
    neuron_dir = float(
        neuron_row.loc[neuron_row["representation"] == "neuron_energy", "mean_improvement"].iloc[0]
    )
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
    write_json(OUT / "gates.json", gates)
    analysis = {
        "dataset": "Kato 2015 WT_NoStim only",
        "sessions": 5,
        "frames": int(sum(int(d.n_frames) for d in data)),
        "session_neurons": {str(k): int(v) for k, v in sorted(session_sizes.items())},
        "landmarks": int(frame[frame["model"] == "B3_population"].shape[0]),
        "members": MEMBERS,
        "horizon": PRIMARY_HORIZON,
        "primary_k": PRIMARY_K,
        "ks": list(KS),
        "surrogate_draws": int(args.surrogate_draws),
        "elapsed_seconds": float(perf_counter() - started),
    }
    write_json(OUT / "analysis_summary.json", analysis)
    print(f"kato common-state complete {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    _main()
