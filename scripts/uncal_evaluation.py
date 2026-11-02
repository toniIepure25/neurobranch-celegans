"""Stage C-F: calibrated probabilistic baselines, scores and trajectory checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from neurobranch.adaptive.evaluation import block_bootstrap_ci, evaluation_landmarks
from neurobranch.adaptive.var import RollingVAR
from neurobranch.diagnostics.common import load_nostim
from neurobranch.probabilistic.baselines import IndependentARModel, LocalMeanModel
from neurobranch.probabilistic.metrics import (
    coverage_summary,
    crps_ensemble,
    energy_score,
    interval_coverage,
    pit_histogram,
    variogram_score,
)
from neurobranch.probabilistic.residual import (
    adaptive_prequential_residuals,
    independent_ar_residuals,
    local_mean_residuals,
    persistence_residuals,
)
from neurobranch.probabilistic.stochastic import Innovation, build_innovation, ensemble_rollout
from neurobranch.uncertainty_calibration.models import StateVarianceModel, diagonal_scale
from neurobranch.uncertainty_calibration.trajectory import trajectory_energy_score

OUT = Path("reports/simulator/uncertainty_calibration")
OUT.mkdir(parents=True, exist_ok=True)
HORIZONS = (1, 5, 10, 20, 50, 100)
PRIMARY_HORIZON = 20
PRIMARY = "P2_independent_ar_lead1"
TRAJECTORY_MODELS = {PRIMARY, "P0_persistence", "P1_local_mean", "P3_state_ar", "P6_adaptive_block", "P6_adaptive_lead1"}


@dataclass(frozen=True)
class Spec:
    name: str
    mean: str
    series: str
    kind: str
    block: int = 5
    lead: int | None = None


SPECS = (
    Spec("P0_persistence", "persistence", "persistence", "block", 20),
    Spec("P1_local_mean", "local_mean", "local_mean", "block", 10),
    Spec(PRIMARY, "independent_ar", "independent_ar", "gauss_diag", lead=1),
    Spec("P2_fixed_train_scale", "independent_ar", "independent_ar", "gauss_diag", lead=1),
    Spec("P3_state_ar", "independent_ar", "independent_ar", "gauss_diag", lead=1),
    Spec("P6_adaptive_block", "var", "adaptive_hist", "block", 10),
    Spec("P6_adaptive_lead1", "var", "adaptive_lead1", "block", 10, lead=1),
)


def _pool(series, t: int, lead: int | None, max_history: int = 1000) -> np.ndarray:
    mask = series.targets <= int(t)
    if lead is not None:
        mask = mask & (series.lead == int(lead))
    idx = np.flatnonzero(mask)[-int(max_history) :]
    return series.values[idx]


def _projection(train: np.ndarray, n_components: int = 10) -> PCA:
    return PCA(n_components=min(n_components, train.shape[1]), random_state=17).fit(train)


def _temporal_cov_error(projected: np.ndarray, truth: np.ndarray) -> float:
    centered = projected - projected.mean(axis=1, keepdims=True)
    member_cov = np.einsum("mtd,mte->de", centered, centered) / max(1, projected.shape[1] - 1)
    truth_c = truth - truth.mean(axis=0, keepdims=True)
    truth_cov = truth_c.T @ truth_c / max(1, len(truth_c) - 1)
    denom = np.linalg.norm(member_cov) + np.linalg.norm(truth_cov)
    return float(np.linalg.norm(member_cov - truth_cov) / denom) if denom > 0 else float("nan")


def _autocorr_error(projected: np.ndarray, truth: np.ndarray) -> float:
    pc, tc = [], []
    for j in range(projected.shape[2]):
        a = projected[:, :, j].mean(axis=0)
        b = truth[:, j]
        if np.std(a) > 1e-10:
            pc.append(np.corrcoef(a[:-1], a[1:])[0, 1])
        if np.std(b) > 1e-10:
            tc.append(np.corrcoef(b[:-1], b[1:])[0, 1])
    return float(abs(np.nanmean(pc) - np.nanmean(tc))) if pc and tc else float("nan")


def _fit_mean(spec: Spec, t: int, X: np.ndarray, cache: dict):
    if spec.mean == "persistence":
        return "persistence"
    if spec.mean == "local_mean":
        key = ("local", t)
        if key not in cache:
            cache[key] = LocalMeanModel(window=100).fit_at(X, t)
        return cache[key]
    if spec.mean == "independent_ar":
        key = ("ar", t)
        if key not in cache:
            cache[key] = IndependentARModel(window=200).fit_at(X, t)
        return cache[key]
    if spec.mean == "var":
        key = ("var", t)
        if key not in cache:
            cache[key] = RollingVAR(window=500).fit_at(X, t)
        return cache[key]
    raise ValueError(spec.mean)


def _variogram_marginal(projected: np.ndarray, obs: np.ndarray) -> float:
    return variogram_score(projected, obs)


def main(n_members: int = 100, stride: int = 10) -> None:
    cfg, data, _ = load_nostim()
    score_rows, neuron_rows, traj_rows, update_rows = [], [], [], []
    coverage_rows = []
    var_acc: dict[str, dict[int, dict[str, list[np.ndarray]]]] = {}
    error_cov: dict[str, list[np.ndarray]] = {}
    model_cov: dict[str, list[np.ndarray]] = {}
    fixed_scales = {}
    state_models = {}
    for d in data:
        sid = int(d.worm_id)
        train = d.X[d.masks["train"]]
        train_end = int(d.split_bounds["train"][1] - 1)
        pca = _projection(train, 10)
        landmarks = evaluation_landmarks(d, horizon_max=100, stride=stride)
        series = {
            "persistence": persistence_residuals(d),
            "local_mean": local_mean_residuals(d, window=100),
            "independent_ar": independent_ar_residuals(d, window=200, block=10),
            "adaptive_hist": adaptive_prequential_residuals(d, window=500, block=50),
            "adaptive_lead1": adaptive_prequential_residuals(d, window=500, block=10),
        }
        lead1_train = _pool(series["independent_ar"], train_end, 1, max_history=5000)
        fixed_scales[sid] = diagonal_scale(lead1_train)
        state_models[sid] = StateVarianceModel.fit(d.X, series["independent_ar"], train_end)
        lo = np.quantile(train, 0.01, axis=0)
        hi = np.quantile(train, 0.99, axis=0)
        cache: dict = {}
        for t in landmarks:
            t = int(t)
            x0 = d.X[t]
            truth = d.X[t + 1 : t + 101]
            truth_proj = pca.transform(truth)
            for spec in SPECS:
                started = perf_counter()
                fitted = _fit_mean(spec, t, d.X, cache)
                if fitted is None:
                    continue
                pool = _pool(series[spec.series], t, spec.lead, max_history=1000)
                if len(pool) < 25:
                    continue
                if spec.name in {"P2_fixed_train_scale"}:
                    innovation = Innovation("gauss_diag", scale=fixed_scales[sid], metadata={"kind": "fixed_train"})
                elif spec.name == "P3_state_ar":
                    scale = np.clip(state_models[sid].predict(d.X, t), 0.01, 3.0)
                    innovation = Innovation("gauss_diag", scale=scale, metadata={"kind": "state"})
                elif spec.kind == "block":
                    innovation = build_innovation(pool, "block", block=spec.block)
                else:
                    innovation = build_innovation(pool, "gauss_diag")
                ens = ensemble_rollout(spec.mean, fitted, x0, 100, n_members, innovation, np.random.default_rng(7000 + sid * 10000 + t))
                if not np.isfinite(ens).all():
                    continue
                ens_proj = pca.transform(ens.reshape(-1, ens.shape[-1])).reshape(n_members, 100, -1)
                energy = {h: energy_score(ens_proj[:, h - 1, :], truth_proj[h - 1]) for h in HORIZONS}
                for h in HORIZONS:
                    cov_h = coverage_summary(ens[:, h - 1, :], truth[h - 1])
                    coverage_rows.append(
                        {
                            "worm_id": sid,
                            "landmark": t,
                            "model": spec.name,
                            "horizon": h,
                            "coverage_50": cov_h["coverage_50"],
                            "coverage_80": cov_h["coverage_80"],
                            "coverage_95": cov_h["coverage_95"],
                            "width_95": cov_h["width_95"],
                        }
                    )
                h20_ens = ens[:, PRIMARY_HORIZON - 1, :]
                h20_truth = truth[PRIMARY_HORIZON - 1]
                crps = crps_ensemble(h20_ens, h20_truth)
                cov = coverage_summary(h20_ens, h20_truth)
                covered = {lev: interval_coverage(h20_ens, h20_truth, lev) for lev in (0.5, 0.8, 0.95)}
                pit = (h20_ens <= h20_truth[None, :]).mean(axis=0)
                ood = float(np.mean((h20_truth < lo) | (h20_truth > hi)))
                score_rows.append(
                    {
                        "worm_id": sid,
                        "landmark": t,
                        "model": spec.name,
                        "energy_h1": energy[1],
                        "energy_h5": energy[5],
                        "energy_h10": energy[10],
                        "energy_h20": energy[20],
                        "energy_h50": energy[50],
                        "energy_h100": energy[100],
                        "crps_h20": float(crps.mean()),
                        "coverage_50": cov["coverage_50"],
                        "coverage_80": cov["coverage_80"],
                        "coverage_95": cov["coverage_95"],
                        "width_50": cov["width_50"],
                        "width_80": cov["width_80"],
                        "width_95": cov["width_95"],
                        "variogram_h20": _variogram_marginal(ens_proj[:, 19, :], truth_proj[19]),
                        "spread_error_h20": float(h20_ens.var(axis=0).mean() / max(np.mean((h20_ens.mean(0) - h20_truth) ** 2), 1e-12)),
                        "temporal_covariance_error": _temporal_cov_error(ens_proj[:, :20, :], truth_proj[:20]),
                        "temporal_autocorr_error": _autocorr_error(ens_proj[:, :20, :], truth_proj[:20]),
                        "ood_h20": ood,
                        "update_seconds": perf_counter() - started,
                        "n_members": n_members,
                    }
                )
                for j, neuron in enumerate(d.predictor_names):
                    neuron_rows.append(
                        {
                            "worm_id": sid,
                            "landmark": t,
                            "model": spec.name,
                            "neuron": neuron,
                            "neuron_index": j,
                            "crps_h20": float(crps[j]),
                            "coverage_50": float(covered[0.5][0][j]),
                            "coverage_80": float(covered[0.8][0][j]),
                            "coverage_95": float(covered[0.95][0][j]),
                            "width_95": float(covered[0.95][1][j]),
                            "pit_h20": float(pit[j]),
                            "ensemble_mean_h20": float(h20_ens.mean(0)[j]),
                            "truth_h20": float(h20_truth[j]),
                            "ensemble_var_h20": float(h20_ens.var(0, ddof=1)[j]),
                        }
                    )
                for h in HORIZONS:
                    acc = var_acc.setdefault(spec.name, {}).setdefault(
                        h, {"within": [], "errors": [], "truth": [], "std_ratio": [], "spread_ratio": []}
                    )
                    within_vec = ens[:, h - 1, :].var(axis=0, ddof=1)
                    err_vec = (ens[:, h - 1, :].mean(axis=0) - truth[h - 1]) ** 2
                    truth_vec = truth[h - 1]
                    acc["within"].append(float(within_vec.mean()))
                    acc["errors"].append(float(err_vec.mean()))
                    acc["truth"].append(float(truth_vec.var(ddof=1)))
                    acc["spread_ratio"].append(float(within_vec.mean() / max(err_vec.mean(), 1e-12)))
                    acc["std_ratio"].append(
                        float(np.median(np.sqrt(within_vec) / np.maximum(truth_vec.std(ddof=1), 1e-12)))
                    )
                error_cov.setdefault(spec.name, {}).setdefault(sid, []).append((h20_ens.mean(0) - h20_truth)[:, None] @ (h20_ens.mean(0) - h20_truth)[None, :])
                model_cov.setdefault(spec.name, {}).setdefault(sid, []).append(np.cov(h20_ens, rowvar=False, ddof=1))
                if spec.name in TRAJECTORY_MODELS:
                    traj_rows.append(
                        {
                            "worm_id": sid,
                            "landmark": t,
                            "model": spec.name,
                            "trajectory_energy_h20": trajectory_energy_score(ens_proj[:, :20, :], truth_proj[:20]),
                            "trajectory_variogram_h20": _variogram_marginal(ens_proj[:, 19, :], truth_proj[19]),
                        }
                    )
                update_rows.append(
                    {
                        "worm_id": sid,
                        "landmark": t,
                        "model": spec.name,
                        "n_pool": int(len(pool)),
                        "innovation": spec.kind,
                        "lead": spec.lead,
                        "update_seconds": perf_counter() - started,
                    }
                )
    pd.DataFrame(score_rows).to_csv(OUT / "scores.csv", index=False)
    pd.DataFrame(neuron_rows).to_csv(OUT / "per_neuron_h20.csv", index=False)
    pd.DataFrame(traj_rows).to_csv(OUT / "trajectory_scores.csv", index=False)
    pd.DataFrame(update_rows).to_csv(OUT / "updates.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(OUT / "coverage_by_horizon.csv", index=False)

    var_rows = []
    for model, by_h in var_acc.items():
        for h, acc in by_h.items():
            var_rows.append(
                {
                    "model": model,
                    "horizon": h,
                    "within_var_mean": float(np.mean(acc["within"])),
                    "error_var_mean": float(np.mean(acc["errors"])),
                    "observed_var_mean": float(np.mean(acc["truth"])),
                    "spread_error_ratio": float(np.mean(acc["spread_ratio"])),
                    "total_std_ratio": float(np.mean(acc["std_ratio"])),
                }
            )
    pd.DataFrame(var_rows).to_csv(OUT / "variance_summary.csv", index=False)

    cov_rows = []
    for model in error_cov:
        per_session = []
        for sid in error_cov[model]:
            err = np.mean(error_cov[model][sid], axis=0)
            mod = np.mean(model_cov[model][sid], axis=0)
            denom = np.linalg.norm(mod) + np.linalg.norm(err)
            per_session.append(float(np.linalg.norm(mod - err) / denom) if denom > 0 else float("nan"))
            cov_rows.append(
                {
                    "model": model,
                    "worm_id": sid,
                    "error_cov_trace": float(np.trace(err)),
                    "model_cov_trace": float(np.trace(mod)),
                    "relative_error": per_session[-1],
                }
            )
        cov_rows.append(
            {
                "model": model,
                "worm_id": "mean",
                "error_cov_trace": float(np.nanmean([r["error_cov_trace"] for r in cov_rows if r["model"] == model and r["worm_id"] != "mean"])),
                "model_cov_trace": float(np.nanmean([r["model_cov_trace"] for r in cov_rows if r["model"] == model and r["worm_id"] != "mean"])),
                "relative_error": float(np.nanmean(per_session)),
            }
        )
    pd.DataFrame(cov_rows).to_csv(OUT / "covariance_h20.csv", index=False)

    scores = pd.DataFrame(score_rows)
    if not scores.empty:
        summary = (
            scores.groupby("model")
            .agg(
                n=("energy_h20", "size"),
                energy_h20=("energy_h20", "mean"),
                energy_h20_median=("energy_h20", "median"),
                energy_h5=("energy_h5", "mean"),
                energy_h50=("energy_h50", "mean"),
                energy_h100=("energy_h100", "mean"),
                crps_h20=("crps_h20", "mean"),
                coverage_50=("coverage_50", "mean"),
                coverage_80=("coverage_80", "mean"),
                coverage_95=("coverage_95", "mean"),
                width_95=("width_95", "mean"),
                variogram_h20=("variogram_h20", "mean"),
                spread_error_h20=("spread_error_h20", "median"),
                temporal_covariance_error=("temporal_covariance_error", "mean"),
                temporal_autocorr_error=("temporal_autocorr_error", "mean"),
                ood_h20=("ood_h20", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(OUT / "score_summary.csv", index=False)
        # Paired block-bootstrap improvements.
        paired = []
        for metric in ("crps_h20", "energy_h20", "variogram_h20", "temporal_covariance_error", "temporal_autocorr_error"):
            pivot = scores.pivot_table(index=["worm_id", "landmark"], columns="model", values=metric)
            if PRIMARY not in pivot:
                continue
            for ref in ("P0_persistence", "P1_local_mean", "P6_adaptive_block"):
                if ref not in pivot:
                    continue
                diff = (pivot[ref] - pivot[PRIMARY]).dropna().to_numpy()
                if len(diff) < 5:
                    continue
                mean, low, high = block_bootstrap_ci(diff, block=5, n_boot=1000)
                paired.append(
                    {
                        "metric": metric,
                        "primary": PRIMARY,
                        "reference": ref,
                        "n_pairs": len(diff),
                        "mean_improvement": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "fraction_better": float((diff > 0).mean()),
                    }
                )
        pd.DataFrame(paired).to_csv(OUT / "paired_scores.csv", index=False)
        # Calibration curves and per-session summaries.
        pit_rows = []
        for model, group in pd.DataFrame(neuron_rows).groupby("model"):
            density, err = pit_histogram(group.pit_h20.to_numpy(), bins=10)
            for i, value in enumerate(density):
                pit_rows.append({"model": model, "bin": i, "density": float(value), "calibration_error": err})
        pd.DataFrame(pit_rows).to_csv(OUT / "calibration_curves.csv", index=False)
        per_session = (
            pd.DataFrame(neuron_rows)
            .groupby(["model", "worm_id"])
            .agg(
                coverage_50=("coverage_50", "mean"),
                coverage_80=("coverage_80", "mean"),
                coverage_95=("coverage_95", "mean"),
                crps_h20=("crps_h20", "mean"),
                width_95=("width_95", "mean"),
            )
            .reset_index()
        )
        per_session.to_csv(OUT / "per_session_calibration.csv", index=False)
        by_h = []
        for h in HORIZONS:
            for model, group in scores.groupby("model"):
                by_h.append({"model": model, "horizon": h, "energy": group[f"energy_h{h}"].mean()})
        pd.DataFrame(by_h).to_csv(OUT / "energy_by_horizon.csv", index=False)
    print("evaluation complete", len(score_rows), len(neuron_rows), len(traj_rows))


if __name__ == "__main__":
    main()
