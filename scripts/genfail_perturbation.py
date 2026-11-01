"""GENFAIL Stage F: cross-model virtual-perturbation consistency."""

from __future__ import annotations

import json

import pandas as pd

from neurobranch.config import PROJECT_ROOT, git_state
from neurobranch.diagnostics.common import load_nostim
from neurobranch.diagnostics.perturbation import WARNING, compare, fit_models, seed_stability

OUT = PROJECT_ROOT / "reports/simulator/failure_diagnostics/perturbation"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, datasets, _ = load_nostim()
    rows = []
    seed_rows = []
    status = []
    for data in datasets:
        print(f"perturbation worm {data.worm_id}", flush=True)
        models = fit_models(data, latent_dim=10, compositional=(data.worm_id == 0))
        status.append(
            {
                "worm_id": int(data.worm_id),
                "models": sorted(models),
                "compositional_fit": "compositional" in models,
            }
        )
        rows.extend(compare(data, models, n_states=5, horizon=20))
        if data.worm_id == 0:
            seed_rows.extend(seed_stability(data, latent_dim=10, horizon=20))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cross_model_perturbations.csv", index=False)
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(OUT / "seed_stability.csv", index=False)

    pairs = df[df.model.str.contains("__vs__", regex=False)].copy()
    pair_summary = (
        pairs.groupby("model", as_index=False)
        .agg(
            mean_cosine=("cosine_between_models", "mean"),
            median_cosine=("cosine_between_models", "median"),
            mean_relative_disagreement=("relative_response_disagreement", "mean"),
            max_relative_disagreement=("relative_response_disagreement", "max"),
            n=("cosine_between_models", "size"),
        )
        .sort_values("mean_cosine")
    )
    pair_summary.to_csv(OUT / "pairwise_consistency.csv", index=False)

    individual = df[~df.model.str.contains("__vs__", regex=False)].copy()
    ranking = []
    for (worm, frame), group in individual.groupby(["worm_id", "frame"]):
        pivot = group.pivot_table(index="target", columns="model", values="response_target")
        if pivot.shape[1] < 2:
            continue
        corr = pivot.corr(method="spearman")
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                ranking.append(
                    {
                        "worm_id": int(worm),
                        "frame": int(frame),
                        "model_a": corr.columns[i],
                        "model_b": corr.columns[j],
                        "spearman_target_response_ranking": float(corr.iloc[i, j]),
                    }
                )
    rank_df = pd.DataFrame(ranking)
    rank_df.to_csv(OUT / "target_ranking_agreement.csv", index=False)
    summary = {
        "stage": "F_perturbation_identifiability",
        "evidence_tier": "exploratory",
        "warning": WARNING,
        "n_worms": len(datasets),
        "model_status": status,
        "n_perturbation_rows": len(df),
        "pairwise_consistency": pair_summary.to_dict(orient="records"),
        "target_ranking_agreement_median": float(rank_df.spearman_target_response_ranking.median())
        if len(rank_df)
        else None,
        "seed_stability_median_cosine": float(seeds.cosine.median()) if len(seeds) else None,
        "biological_intervention_validation": "NOT_EVALUABLE",
        "reason_not_evaluable": "no matched experimental perturbation; model perturbations are virtual and observationally non-identified",
        "git": git_state(),
    }
    (OUT / "perturbation_summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(pair_summary.to_string(index=False))
    print(rank_df.head().to_string(index=False))
    print(summary["target_ranking_agreement_median"], summary["seed_stability_median_cosine"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
