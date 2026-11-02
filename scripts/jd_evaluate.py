"""Joint-dynamics factorial evaluation.

Core question: does a population mean/covariance add information beyond a strong
independent-neuron forecast?  All models use identical held-out landmarks, the
same fixed training PCA, and causal one-step innovation pools.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from neurobranch.adaptive.evaluation import evaluation_landmarks
from neurobranch.adaptive.var import RollingVAR
from neurobranch.diagnostics.common import load_nostim
from neurobranch.joint_dynamics.evaluation import (
    autocorrelation_error,
    neural_covariance_error,
    occupancy_and_recurrence,
    path_covariance_error,
    total_variation,
)
from neurobranch.joint_dynamics.models import LowRankTemporalInnovation
from neurobranch.joint_dynamics.residuals import lead1_innovations
from neurobranch.probabilistic.baselines import IndependentARModel
from neurobranch.probabilistic.metrics import (
    coverage_summary,
    crps_ensemble,
    energy_score,
    interval_coverage,
    pit_values,
    variogram_score,
)
from neurobranch.probabilistic.residual import (
    adaptive_prequential_residuals,
    independent_ar_residuals,
    persistence_residuals,
)
from neurobranch.probabilistic.stochastic import Innovation, build_innovation
from neurobranch.uncertainty_calibration.trajectory import trajectory_energy_score

OUT = Path("reports/simulator/joint_dynamics")
HORIZONS = (1, 5, 10, 20, 50, 100)
PRIMARY_HORIZON = 20
MEMBERS = 100
STRIDE = 10


@dataclass(frozen=True)
class Spec:
    name: str
    mean: str
    series: str
    innovation: str
    block: int = 1
    lead1: bool = True


SPECS = (
    Spec("J0_persistence", "persistence", "persistence", "block", 20, False),
    Spec("J1_independent_diag", "independent_ar", "independent_ar", "diag"),
    Spec("J2_population_diag", "var", "population", "diag"),
    Spec("J3_population_full", "var", "population", "full"),
    Spec("J4_independent_full", "independent_ar", "independent_ar", "full"),
    Spec("J5_population_lead1_boot", "var", "population", "block", 10),
)


def prepare(data) -> tuple[PCA, dict, dict]:
    train = data.X[data.masks["train"]]
    pca = PCA(n_components=min(10, train.shape[1]), random_state=17).fit(train)
    persistence = persistence_residuals(data)
    indep = independent_ar_residuals(data, window=200, block=10)
    population = adaptive_prequential_residuals(data, window=500, block=10)
    train_end = int(data.split_bounds["train"][1] - 1)
    return pca, {"persistence": persistence, "independent_ar": indep, "population": population}, {
        "train_end": train_end,
        "train_var": train.var(axis=0),
    }


def make_innovation(spec: Spec, series, t: int, temporal_cache: dict) -> Innovation | np.ndarray:
    if spec.innovation == "block":
        mask = np.isfinite(series.values).all(axis=1)
        if spec.lead1:
            mask &= series.lead == 1
        mask &= series.targets <= int(t)
        idx = np.flatnonzero(mask)[-1000:]
        pool = series.values[idx]
        return build_innovation(pool, "block", block=spec.block)
    sample = lead1_innovations(series, t=t)
    if spec.innovation == "diag":
        return build_innovation(sample.values, "gauss_diag")
    if spec.innovation == "full":
        return build_innovation(sample.values, "gauss", covariance_method="ledoit")
    if spec.innovation == "temporal":
        key = int(t)
        if key not in temporal_cache:
            temporal_cache[key] = LowRankTemporalInnovation.fit(sample.values, rank=8)
        return temporal_cache[key]
    raise ValueError(spec.innovation)


def rollout(mean: str, fitted, x0: np.ndarray, horizon: int, eps: np.ndarray) -> np.ndarray:
    x = np.repeat(np.asarray(x0, dtype=np.float64)[None, :], eps.shape[0], axis=0)
    out = np.empty((eps.shape[0], int(horizon), x.shape[1]), dtype=np.float64)
    if mean == "var":
        for step in range(int(horizon)):
            x = x @ fitted.A.T + fitted.b + eps[:, step]
            out[:, step] = x
    elif mean == "independent_ar":
        for step in range(int(horizon)):
            x = fitted.a * x + fitted.b + eps[:, step]
            out[:, step] = x
    elif mean == "persistence":
        for step in range(int(horizon)):
            x = x + eps[:, step]
            out[:, step] = x
    else:
        raise ValueError(mean)
    return out


def fit_mean(spec: Spec, t: int, cache: dict, X: np.ndarray):
    if spec.mean == "persistence":
        return None
    key = (spec.mean, int(t))
    if key not in cache:
        if spec.mean == "var":
            cache[key] = RollingVAR(window=500).fit_at(X, t)
        else:
            cache[key] = IndependentARModel(window=200).fit_at(X, t)
    return cache[key]


def main(n_members: int = MEMBERS, stride: int = STRIDE) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, data, recordings = load_nostim()
    score_rows, neuron_rows, coverage_rows = [], [], []
    occupancy_rows, config_rows = [], []
    ensemble_store: dict[tuple[int, int, str], dict[str, np.ndarray]] = {}
    started_all = perf_counter()
    for d in data:
        sid = int(d.worm_id)
        pca, series, meta = prepare(d)
        train = d.X[d.masks["train"]]
        projected_train = pca.transform(train)
        clusters = KMeans(n_clusters=6, random_state=23, n_init=10).fit(projected_train)
        train_lo = np.quantile(projected_train, 0.01, axis=0)
        train_hi = np.quantile(projected_train, 0.99, axis=0)
        landmarks = evaluation_landmarks(d, horizon_max=100, stride=stride)
        cache: dict = {}
        temporal_cache: dict = {}
        top_neurons = np.argsort(meta["train_var"])[-20:]
        drop_high_variance = np.argsort(meta["train_var"])[-5:]
        keep_no_top5 = np.setdiff1d(np.arange(d.X.shape[1]), drop_high_variance)
        for t in landmarks:
            t = int(t)
            x0 = d.X[t]
            truth = d.X[t + 1 : t + 101]
            truth_proj = pca.transform(truth)
            obs_occ, obs_rec = occupancy_and_recurrence(truth_proj[None, :, :], clusters.cluster_centers_)
            for spec in SPECS:
                started = perf_counter()
                fitted = fit_mean(spec, t, cache, d.X)
                if spec.mean != "persistence" and fitted is None:
                    continue
                innovation = make_innovation(spec, series[spec.series], t, temporal_cache)
                if isinstance(innovation, Innovation):
                    eps = innovation.sample(n_members, 100, np.random.default_rng(7000 + sid * 10000 + t))
                else:
                    eps = innovation.sample(n_members, 100, np.random.default_rng(7000 + sid * 10000 + t))
                ens = rollout(spec.mean, fitted, x0, 100, eps)
                if not np.isfinite(ens).all():
                    continue
                ens_proj = pca.transform(ens.reshape(-1, ens.shape[-1])).reshape(ens.shape[0], 100, -1)
                row = {
                    "model": spec.name,
                    "worm_id": sid,
                    "landmark": t,
                    "elapsed_s": perf_counter() - started,
                }
                for h in HORIZONS:
                    obs = truth_proj[h - 1]
                    row[f"energy_h{h}"] = energy_score(ens_proj[:, h - 1], obs)
                    row[f"crps_h{h}"] = float(crps_ensemble(ens_proj[:, h - 1], obs).mean())
                    cov = coverage_summary(ens_proj[:, h - 1], obs)
                    for level in (50, 80, 95):
                        row[f"coverage_h{h}_{level}"] = cov[f"coverage_{level}"]
                        row[f"width_h{h}_{level}"] = cov[f"width_{level}"]
                    row[f"variogram_h{h}"] = variogram_score(ens_proj[:, h - 1], obs)
                h = PRIMARY_HORIZON
                obs = truth_proj[h - 1]
                ens_h = ens_proj[:, h - 1]
                row["mean_rmse"] = float(np.sqrt(np.mean((ens_h.mean(0) - obs) ** 2)))
                row["ensemble_std"] = float(ens_h.std(0).mean())
                row["error_variance"] = float(np.mean((ens_h.mean(0) - obs) ** 2))
                row["ood_fraction"] = float(((obs < train_lo) | (obs > train_hi)).mean())
                row["energy_h20_full"] = energy_score(ens[:, h - 1], truth[h - 1])
                row["crps_h20_full"] = float(crps_ensemble(ens[:, h - 1], truth[h - 1]).mean())
                row["coverage_h20_full_50"] = float(interval_coverage(ens[:, h - 1], truth[h - 1], 0.5)[0].mean())
                row["coverage_h20_full_80"] = float(interval_coverage(ens[:, h - 1], truth[h - 1], 0.8)[0].mean())
                row["coverage_h20_full_95"] = float(interval_coverage(ens[:, h - 1], truth[h - 1], 0.95)[0].mean())
                row["energy_h20_no_top5"] = energy_score(
                    ens[:, h - 1, keep_no_top5], truth[h - 1, keep_no_top5]
                )
                row["trajectory_energy_h20"] = trajectory_energy_score(ens_proj[:, :20], truth_proj[:20])
                row["trajectory_energy_h100"] = trajectory_energy_score(ens_proj, truth_proj)
                row["neural_cov_error"] = neural_covariance_error(ens_proj[:, :20], truth_proj[:20])
                row["path_cov_error"] = path_covariance_error(ens_proj[:, :20], truth_proj[:20])
                row["autocorr_error"] = autocorrelation_error(ens_proj[:, :20], truth_proj[:20])
                row["finite_h100"] = bool(np.isfinite(ens_proj).all())
                row["max_abs_h100"] = float(np.max(np.abs(ens_proj[:, -1])))
                model_occ, model_rec = occupancy_and_recurrence(ens_proj, clusters.cluster_centers_)
                row["occupancy_tv"] = total_variation(model_occ, obs_occ)
                row["recurrence_model"] = model_rec
                row["recurrence_observed"] = obs_rec
                score_rows.append(row)
                # Alternative prespecified neural-space subset: top-20 training-variance neurons.
                subset_ens = ens[:, :20, top_neurons]
                subset_truth = truth[:20, top_neurons]
                occupancy_rows.append(
                    {
                        "model": spec.name,
                        "worm_id": sid,
                        "landmark": t,
                        "energy_h20_neural20": energy_score(subset_ens[:, -1], subset_truth[-1]),
                    }
                )
                ensemble_store[(sid, t, spec.name)] = {
                    "ensemble_proj_h20": ens_proj[:, :20].astype(np.float32),
                    "truth_proj_h20": truth_proj[:20].astype(np.float32),
                }
                for j in range(ens_proj.shape[-1]):
                    covered, width = interval_coverage(ens_h, obs, 0.95)
                    neuron_rows.append(
                        {
                            "model": spec.name,
                            "worm_id": sid,
                            "landmark": t,
                            "neuron": j,
                            "crps_h20": crps_ensemble(ens_h, obs)[j],
                            "coverage_50": interval_coverage(ens_h, obs, 0.5)[0][j],
                            "coverage_80": interval_coverage(ens_h, obs, 0.8)[0][j],
                            "coverage_95": covered[j],
                            "width_95": width[j],
                            "pit_h20": pit_values(ens_h, obs)[j],
                        }
                    )
                for h2 in HORIZONS:
                    for level in (50, 80, 95):
                        coverage_rows.append(
                            {
                                "model": spec.name,
                                "worm_id": sid,
                                "landmark": t,
                                "horizon": h2,
                                "level": level,
                                "covered": row[f"coverage_h{h2}_{level}"],
                            }
                        )
        print(f"session {sid} complete", perf_counter() - started_all, flush=True)
    scores = pd.DataFrame(score_rows)
    scores.to_csv(OUT / "scores_by_landmark.csv", index=False)
    pd.DataFrame(neuron_rows).to_csv(OUT / "neuron_h20.csv", index=False)
    pd.DataFrame(occupancy_rows).to_csv(OUT / "neural_subset_scores.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(OUT / "coverage_by_landmark.csv", index=False)
    # Compact visualization subset only; scores use the full 100-member ensemble.
    np.savez_compressed(
        OUT / "trajectory_examples.npz",
        **{
            f"{sid}_{t}_{name}": value["ensemble_proj_h20"][:10].astype(np.float32)
            for (sid, t, name), value in ensemble_store.items()
        },
        **{
            f"truth_{sid}_{t}": value["truth_proj_h20"].astype(np.float32)
            for (sid, t, name), value in ensemble_store.items()
            if name == "J0_persistence"
        },
    )
    summary = (
        scores.groupby("model")
        .agg(
            energy_h20=("energy_h20", "mean"),
            crps_h20=("crps_h20", "mean"),
            crps_h20_full=("crps_h20_full", "mean"),
            coverage_h20_full_50=("coverage_h20_full_50", "mean"),
            coverage_h20_full_80=("coverage_h20_full_80", "mean"),
            coverage_h20_full_95=("coverage_h20_full_95", "mean"),
            energy_h20_full=("energy_h20_full", "mean"),
            energy_h20_no_top5=("energy_h20_no_top5", "mean"),
            trajectory_energy_h20=("trajectory_energy_h20", "mean"),
            trajectory_energy_h100=("trajectory_energy_h100", "mean"),
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
        )
        .reset_index()
    )
    summary.to_csv(OUT / "score_summary.csv", index=False)
    config_rows = [
        {"name": s.name, "mean": s.mean, "series": s.series, "innovation": s.innovation, "block": s.block, "lead1": s.lead1}
        for s in SPECS
    ]
    (OUT / "configurations.json").write_text(
        json.dumps(
            {
                "horizons": list(HORIZONS),
                "primary_horizon": PRIMARY_HORIZON,
                "members": n_members,
                "stride": stride,
                "n_landmarks": int(len(scores[["worm_id", "landmark"]].drop_duplicates())),
                "sessions": [int(r.worm_id) for r in recordings],
                "fps": [float(r.fps) for r in recordings],
                "models": config_rows,
                "projection": "fixed 10-component PCA fitted on training frames only",
                "subset": "top-20 training-variance neurons",
                "seeds": "7000 + worm_id*10000 + landmark",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("joint evaluation complete", len(score_rows), perf_counter() - started_all, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", type=int, default=MEMBERS)
    parser.add_argument("--stride", type=int, default=STRIDE)
    args = parser.parse_args()
    main(args.members, args.stride)
