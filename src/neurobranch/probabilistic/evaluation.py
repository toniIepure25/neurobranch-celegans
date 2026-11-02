"""Probabilistic evaluation of adaptive and baseline mean models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from neurobranch.adaptive.evaluation import block_bootstrap_ci, evaluation_landmarks
from neurobranch.adaptive.var import FrozenVAR, RollingVAR
from neurobranch.preprocessing import split_starts
from neurobranch.probabilistic.baselines import (
    BaselineConfig,
    IndependentARModel,
    LocalMeanModel,
    PersistenceForecast,
)
from neurobranch.probabilistic.metrics import (
    covariance_relative_error,
    coverage_summary,
    crps_ensemble,
    energy_score,
    interval_coverage,
    pit_histogram,
    variogram_score,
)
from neurobranch.probabilistic.residual import (
    ResidualSeries,
    adaptive_prequential_residuals,
    frozen_residuals,
    independent_ar_residuals,
    local_mean_residuals,
    persistence_residuals,
)
from neurobranch.probabilistic.stochastic import build_innovation, ensemble_rollout

OUT = Path(__file__).resolve().parents[3] / "reports/simulator/probabilistic_adaptive"
HORIZONS = (5, 20, 50, 100)
PRIMARY_HORIZON = 20


@dataclass(frozen=True)
class ProbModel:
    name: str
    mean: str
    innovation: str
    residual_key: str
    block: int = 5


def model_specs(primary: str = "P5_adaptive_boot") -> list[ProbModel]:
    specs = [
        ProbModel("P0_persistence", "persistence", "block", "persistence", block=20),
        ProbModel("P1_local_mean", "local_mean", "block", "local_mean", block=10),
        ProbModel("P2_independent_ar", "independent_ar", "gauss_diag", "independent_ar"),
        ProbModel("P3_frozen_var", "var", "block", "frozen", block=10),
        ProbModel("P4_adaptive_gauss", "var", "gauss", "adaptive"),
        ProbModel("P5_adaptive_boot", "var", "bootstrap", "adaptive"),
        ProbModel("P6_adaptive_block", "var", "block", "adaptive", block=10),
        ProbModel("P7_adaptive_diag", "var", "gauss_diag", "adaptive"),
    ]
    if primary not in {s.name for s in specs}:
        raise ValueError(f"unknown primary {primary}")
    return specs


def _projection(train: np.ndarray, n_components: int = 10) -> PCA:
    return PCA(n_components=min(int(n_components), train.shape[1]), random_state=17).fit(train)


def _project(ensemble: np.ndarray, pca: PCA) -> np.ndarray:
    flat = ensemble.reshape(-1, ensemble.shape[-1])
    return pca.transform(flat).reshape(ensemble.shape[0], ensemble.shape[1], -1)


def _temporal_cov_error(projected: np.ndarray, truth_projected: np.ndarray) -> float:
    centered = projected - projected.mean(axis=1, keepdims=True)
    member_cov = np.einsum("mtd,mte->de", centered, centered) / max(1, projected.shape[1] - 1)
    truth_c = truth_projected - truth_projected.mean(axis=0, keepdims=True)
    truth_cov = truth_c.T @ truth_c / max(1, len(truth_c) - 1)
    return covariance_relative_error(member_cov, truth_cov)


def _autocorr_error(projected: np.ndarray, truth_projected: np.ndarray) -> float:
    pc, tc = [], []
    for j in range(projected.shape[2]):
        a = projected[:, :, j].mean(axis=0)
        b = truth_projected[:, j]
        if np.std(a) > 1e-10:
            pc.append(np.corrcoef(a[:-1], a[1:])[0, 1])
        if np.std(b) > 1e-10:
            tc.append(np.corrcoef(b[:-1], b[1:])[0, 1])
    if not pc or not tc:
        return float("nan")
    return float(abs(np.nanmean(pc) - np.nanmean(tc)))


def _build_series(data, config: BaselineConfig) -> dict[str, ResidualSeries]:
    return {
        "persistence": persistence_residuals(data),
        "local_mean": local_mean_residuals(data, window=config.local_window),
        "independent_ar": independent_ar_residuals(data, window=config.ar_window),
        "frozen": frozen_residuals(data),
        "adaptive": adaptive_prequential_residuals(data, window=500),
    }


def _fit_mean(spec: ProbModel, t: int, X: np.ndarray, config: BaselineConfig, frozen: FrozenVAR, cache: dict):
    if spec.mean == "persistence":
        return PersistenceForecast()
    if spec.mean == "local_mean":
        key = ("local", t)
        if key not in cache:
            cache[key] = LocalMeanModel(window=config.local_window).fit_at(X, t)
        return cache[key]
    if spec.mean == "independent_ar":
        key = ("ar", t)
        if key not in cache:
            cache[key] = IndependentARModel(window=config.ar_window).fit_at(X, t)
        return cache[key]
    if spec.mean == "var":
        if spec.name == "P3_frozen_var":
            return frozen
        key = ("rolling", t)
        if key not in cache:
            cache[key] = RollingVAR(window=500).fit_at(X, t)
        return cache[key]
    raise ValueError(spec.mean)


def evaluate_probabilistic_session(
    data,
    config: BaselineConfig,
    *,
    n_members: int = 100,
    horizon_max: int = 100,
    stride: int = 10,
    primary: str = "P5_adaptive_boot",
    seed: int = 17,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed + int(data.worm_id))
    landmarks = evaluation_landmarks(data, horizon_max=horizon_max, stride=stride)
    frozen = FrozenVAR.fit(
        data.X, split_starts(data, "train", history=1, horizon=1), ridge=1.0, spectral_cap=0.995
    )
    series = _build_series(data, config)
    pca = _projection(data.X[data.masks["train"]], 10)
    train = data.X[data.masks["train"]]
    train_lo = np.quantile(train, 0.01, axis=0)
    train_hi = np.quantile(train, 0.99, axis=0)
    specs = model_specs(primary)
    rows, neuron_rows, update_rows = [], [], []
    ens_means: dict[str, list[np.ndarray]] = {s.name: [] for s in specs}
    within_vars: dict[str, list[np.ndarray]] = {s.name: [] for s in specs}
    errors: dict[str, list[np.ndarray]] = {s.name: [] for s in specs}

    for t in landmarks:
        t = int(t)
        x0 = data.X[t].copy()
        truth_all = data.X[t + 1 : t + horizon_max + 1]
        truth_proj = pca.transform(truth_all)
        cache: dict = {}
        for spec in specs:
            started = perf_counter()
            fitted = _fit_mean(spec, t, data.X, config, frozen, cache)
            if fitted is None:
                continue
            pool = series[spec.residual_key].subset(t, max_history=1000)
            if len(pool) < 30:
                continue
            innovation = build_innovation(
                pool,
                spec.innovation,
                block=spec.block,
                covariance_method="ledoit" if spec.innovation != "gauss_diag" else "diagonal",
            )
            ens = ensemble_rollout(
                spec.mean, fitted, x0, horizon_max, n_members, innovation, rng
            )
            elapsed = perf_counter() - started
            if not np.isfinite(ens).all():
                continue
            ens_proj = _project(ens, pca)
            mean_h20 = ens[:, PRIMARY_HORIZON - 1, :].mean(axis=0)
            within_h20 = ens[:, PRIMARY_HORIZON - 1, :].var(axis=0, ddof=1)
            error_h20 = (mean_h20 - truth_all[PRIMARY_HORIZON - 1]) ** 2
            ens_means[spec.name].append(mean_h20)
            within_vars[spec.name].append(within_h20)
            errors[spec.name].append(error_h20)
            h20_obs = truth_all[PRIMARY_HORIZON - 1]
            h20_ens = ens[:, PRIMARY_HORIZON - 1, :]
            crps = crps_ensemble(h20_ens, h20_obs)
            cov = coverage_summary(h20_ens, h20_obs)
            covered = {
                level: interval_coverage(h20_ens, h20_obs, level)
                for level in (0.5, 0.8, 0.95)
            }
            pit = (h20_ens <= h20_obs[None, :]).mean(axis=0)
            interval_ood = float(
                np.mean((h20_ens < train_lo[None, :]) | (h20_ens > train_hi[None, :]))
            )
            projected_h20 = ens_proj[:, PRIMARY_HORIZON - 1, :]
            projected_obs = pca.transform(h20_obs[None, :])[0]
            var_score = variogram_score(projected_h20, projected_obs)
            energy_by_h = {}
            for h in HORIZONS:
                projected = ens_proj[:, h - 1, :]
                obs_proj = truth_proj[h - 1]
                energy_by_h[h] = energy_score(projected, obs_proj)
            temporal_cov = _temporal_cov_error(ens_proj[:, :20, :], truth_proj[:20])
            temporal_ac = _autocorr_error(ens_proj[:, :20, :], truth_proj[:20])
            rows.append(
                {
                    "worm_id": int(data.worm_id),
                    "landmark": t,
                    "model": spec.name,
                    "n_members": int(n_members),
                    "energy_h20": energy_by_h[20],
                    "energy_h5": energy_by_h[5],
                    "energy_h50": energy_by_h[50],
                    "energy_h100": energy_by_h[100],
                    "crps_h20_mean": float(crps.mean()),
                    "crps_h20_median": float(np.median(crps)),
                    "coverage_50": cov["coverage_50"],
                    "coverage_80": cov["coverage_80"],
                    "coverage_95": cov["coverage_95"],
                    "width_50": cov["width_50"],
                    "width_80": cov["width_80"],
                    "width_95": cov["width_95"],
                    "variogram_h20": var_score,
                    "spread_error_ratio_h20": float(
                        np.mean(within_h20) / max(np.mean(error_h20), 1e-12)
                    ),
                    "temporal_covariance_error": temporal_cov,
                    "temporal_autocorr_error": temporal_ac,
                    "ood_h20": interval_ood,
                    "finite_fraction": 1.0,
                    "update_seconds": float(elapsed),
                }
            )
            for j, neuron in enumerate(data.predictor_names):
                neuron_rows.append(
                    {
                        "worm_id": int(data.worm_id),
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
                        "ensemble_mean_h20": float(mean_h20[j]),
                        "truth_h20": float(h20_obs[j]),
                        "ensemble_var_h20": float(within_h20[j]),
                    }
                )
            update_rows.append(
                {
                    "worm_id": int(data.worm_id),
                    "landmark": t,
                    "model": spec.name,
                    "update_seconds": float(elapsed),
                    "innovation_kind": spec.innovation,
                    "n_residuals": int(len(pool)),
                    "innovation_shrinkage": float(
                        (innovation.metadata or {}).get("shrinkage", float("nan"))
                    ),
                }
            )
    scores = pd.DataFrame(rows)
    neurons = pd.DataFrame(neuron_rows)
    updates = pd.DataFrame(update_rows)

    variance_rows = []
    for spec in specs:
        if not ens_means[spec.name]:
            continue
        means = np.asarray(ens_means[spec.name])
        wv = np.asarray(within_vars[spec.name])
        err = np.asarray(errors[spec.name])
        observed_var = data.X[data.masks["test"]].var(axis=0, ddof=1)
        across = means.var(axis=0, ddof=1)
        within = wv.mean(axis=0)
        total = across + within
        variance_rows.append(
            {
                "worm_id": int(data.worm_id),
                "model": spec.name,
                "observed_var_mean": float(np.mean(observed_var)),
                "across_landmark_mean_var_mean": float(np.mean(across)),
                "within_ensemble_var_mean": float(np.mean(within)),
                "total_implied_var_mean": float(np.mean(total)),
                "error_var_mean": float(np.mean(err)),
                "spread_error_ratio": float(np.mean(within) / max(np.mean(err), 1e-12)),
                "ensemble_std_ratio": float(
                    np.median(np.sqrt(within) / np.maximum(np.sqrt(observed_var), 1e-12))
                ),
                "total_std_ratio": float(
                    np.median(np.sqrt(total) / np.maximum(np.sqrt(observed_var), 1e-12))
                ),
            }
        )
    return scores, neurons, updates, pd.DataFrame(variance_rows)


def summarize_scores(scores: pd.DataFrame, *, primary: str, reference: str = "P0_persistence") -> pd.DataFrame:
    rows = []
    for model, group in scores.groupby("model"):
        for metric in (
            "energy_h20",
            "crps_h20_mean",
            "coverage_50",
            "coverage_80",
            "coverage_95",
            "width_95",
            "variogram_h20",
            "spread_error_ratio_h20",
            "temporal_covariance_error",
            "temporal_autocorr_error",
            "ood_h20",
        ):
            values = group[metric].dropna().to_numpy()
            if len(values):
                rows.append(
                    {
                        "model": model,
                        "metric": metric,
                        "n": int(len(values)),
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                    }
                )
    frame = pd.DataFrame(rows)
    frame["primary"] = frame.model == primary
    frame["reference"] = reference
    return frame


def paired_scores(
    scores: pd.DataFrame, *, primary: str, references: tuple[str, ...] = ("P0_persistence", "P3_frozen_var")
) -> pd.DataFrame:
    rows = []
    for metric in ("energy_h20", "crps_h20_mean", "variogram_h20"):
        pivot = scores.pivot_table(index=["worm_id", "landmark"], columns="model", values=metric)
        if primary not in pivot:
            continue
        for ref in references:
            if ref not in pivot:
                continue
            diff = (pivot[ref] - pivot[primary]).dropna().to_numpy()
            if len(diff) < 5:
                continue
            mean, lo, hi = block_bootstrap_ci(diff, block=5, n_boot=1000)
            rows.append(
                {
                    "metric": metric,
                    "model": primary,
                    "reference": ref,
                    "n_pairs": int(len(diff)),
                    "mean_improvement": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "fraction_better": float((diff > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def aggregate_pit(neurons: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in neurons.groupby("model"):
        density, err = pit_histogram(group.pit_h20.to_numpy(), bins=10)
        for i, value in enumerate(density):
            rows.append({"model": model, "bin": i, "density": float(value), "calibration_error": err})
    return pd.DataFrame(rows)


def load_primary(path: Path = OUT / "innovation_choice.json") -> str:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")).get("primary_model", "P5_adaptive_boot")
    return "P5_adaptive_boot"
