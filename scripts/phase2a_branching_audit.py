"""Reproduce the Phase 2 branching audit on synthetic fixtures and all Kato worms."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA

from neurobranch.branching import (
    candidate_frames,
    corrected_branching_test,
    entropy_across_latent_dims,
)
from neurobranch.branching.stats import compare_null_specs
from neurobranch.branching.synthetic import make_scenario, outcomes_for
from neurobranch.config import ExperimentConfig
from neurobranch.dynamics.training import load_worm_data
from neurobranch.preprocessing import split_starts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "phase2a_scientific_audit" / "branching"
OUT.mkdir(parents=True, exist_ok=True)


def synthetic_audit() -> pd.DataFrame:
    rows = []
    for kind in ["deterministic", "stochastic", "rare", "hidden", "projection"]:
        sc = make_scenario(kind, n_candidates=60, seed=0)
        outcomes = outcomes_for(sc, horizon=20)
        for dim_name, Z in [("3d", sc.Z), ("8d", sc.Z_high)]:
            specs = compare_null_specs(
                Z,
                sc.candidate_frames,
                outcomes,
                k=15,
                n_permutations=300,
                block_sizes=(10, 30, 60),
                seed=0,
            )
            for s in specs:
                rows.append(
                    {
                        "scenario": kind,
                        "ground_truth": sc.ground_truth,
                        "projection": dim_name,
                        "mean_entropy": s["mean_entropy"],
                        "null_mean": s["null_mean"],
                        "null_method": s["null_method"],
                        "null_block_size": s["null_block_size"],
                        "p_excess": s["p_excess"],
                        "p_deficit": s["p_deficit"],
                        "p_two_sided": s["p_two_sided"],
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    synth = synthetic_audit()
    synth.to_csv(OUT / "synthetic_branching_audit.csv", index=False)
    print("synthetic done", flush=True)

    cfg = ExperimentConfig.load(ROOT / "configs" / "experiments" / "blocked_suite.yaml")
    data_all = load_worm_data(cfg)
    results = {}
    for data in data_all:
        train_starts = split_starts(data, "train", history=1, horizon=1)
        pca = PCA(n_components=3, random_state=0, svd_solver="randomized").fit(data.X[train_starts])
        Z = pca.transform(data.X)
        frames = candidate_frames(data, horizon=20)
        corrected = corrected_branching_test(
            Z, data, frames, k=40, horizon=20, n_permutations=200, seed=0
        )
        dims = entropy_across_latent_dims(data, dims=(3, 8, 16, 32), k=40, horizon=20)
        results[f"worm{data.worm_id}"] = {
            "n_candidate_frames": int(len(frames)),
            "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
            "corrected": corrected,
            "latent_dimension_sensitivity": {
                "spearman_vs_3d": dims["spearman_vs_base"],
                "n_common_frames": len(set.intersection(*(set(v) for v in dims["scores"].values()))),
            },
        }
        print("worm", data.worm_id, json.dumps(corrected["outcome_counts"]), flush=True)
    (OUT / "branching_all_worms.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    flat = []
    for worm, r in results.items():
        for s in r["corrected"]["tests"]:
            flat.append(
                {
                    "worm": worm,
                    "n_candidates": r["n_candidate_frames"],
                    "mean_entropy": s["mean_entropy"],
                    "null_mean": s["null_mean"],
                    "excess": s["excess_entropy_bits"],
                    "null_method": s["null_method"],
                    "null_block_size": s["null_block_size"],
                    "p_excess": s["p_excess"],
                    "p_deficit": s["p_deficit"],
                    "p_two_sided": s["p_two_sided"],
                    "ci_low": s["mean_entropy_ci_low"],
                    "ci_high": s["mean_entropy_ci_high"],
                }
            )
    pd.DataFrame(flat).to_csv(OUT / "branching_all_worms_long.csv", index=False)
    print("real done", flush=True)


if __name__ == "__main__":
    main()

