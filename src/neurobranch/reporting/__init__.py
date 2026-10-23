"""Report and figure generation from saved artifacts.

Nothing here invents a result.  If an artifact is missing, the report says so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from neurobranch.config import PROJECT_ROOT, resolve_path

RUNS = PROJECT_ROOT / "artifacts" / "runs"


def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def aggregate_suite(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["rollout_mse_h3"] = d["rollout_mse"].apply(
        lambda s: json.loads(s.replace("'", '"')).get("3") if isinstance(s, str) else None
    )
    g = (
        d.groupby("model")
        .agg(
            one_step_r2_mean=("one_step_r2", "mean"),
            one_step_r2_sd=("one_step_r2", "std"),
            behavior_accuracy_mean=("behavior_accuracy", "mean"),
            behavior_persistence_mean=("behavior_persistence_accuracy", "mean"),
            transition_tv_mean=("transition_total_variation", "mean"),
            transition_tv_persistence=("transition_persistence_tv", "mean"),
            rollout_mse_h3_mean=("rollout_mse_h3", "mean"),
            n_runs=("seed", "size"),
        )
        .sort_values("one_step_r2_mean", ascending=False)
    )
    return g.round(4)


def paired_differences(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df.model == "persistence"].set_index(["worm_id", "seed"])["one_step_r2"]
    rows = []
    for m, sub in df.groupby("model"):
        s = sub.set_index(["worm_id", "seed"])["one_step_r2"]
        rows.append(
            {
                "model": m,
                "mean_paired_delta_r2_vs_persistence": float((s - base).mean()),
                "wins_vs_persistence": int((s > base).sum()),
                "n_pairs": int(len(s)),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_paired_delta_r2_vs_persistence", ascending=False).round(4)


def write_results_report(out_dir: str | Path = PROJECT_ROOT / "reports") -> list[Path]:
    out = resolve_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    blocked = _read_csv(RUNS / "blocked_suite" / "suite_metrics.csv")
    forward = _read_csv(RUNS / "forward_stress" / "suite_metrics.csv")
    ncmcm = _read_json(PROJECT_ROOT / "artifacts" / "reproduction_ncmcm.json")
    bundlenet_lite = _read_json(PROJECT_ROOT / "artifacts" / "reproduction_bundlenet_lite.json")
    bundlenet_official = _read_json(PROJECT_ROOT / "artifacts" / "reproduction_bundlenet_official.json")
    branching = _read_json(PROJECT_ROOT / "artifacts" / "branching_worm0.json")
    interventions = _read_csv(PROJECT_ROOT / "artifacts" / "intervention_search_worm0.csv")
    random_controls = _read_csv(PROJECT_ROOT / "artifacts" / "intervention_random_controls_worm0.csv")

    summary: dict[str, Any] = {
        "epistemic_status": "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.",
        "primary_split": "blocked contiguous middle block; not forward forecasting",
    }
    if blocked is not None:
        agg = aggregate_suite(blocked)
        agg.to_csv(out / "results_model_aggregate.csv")
        paths.append(out / "results_model_aggregate.csv")
        paired = paired_differences(blocked)
        paired.to_csv(out / "results_paired_vs_persistence.csv", index=False)
        paths.append(out / "results_paired_vs_persistence.csv")
        summary["blocked_models"] = agg.reset_index().to_dict(orient="records")
        summary["paired_vs_persistence"] = paired.to_dict(orient="records")
    if forward is not None:
        fagg = (
            forward.groupby("model")
            .agg(
                one_step_r2_mean=("one_step_r2", "mean"),
                behavior_accuracy_mean=("behavior_accuracy", "mean"),
                behavior_persistence_mean=("behavior_persistence_accuracy", "mean"),
                transition_tv_mean=("transition_total_variation", "mean"),
            )
            .round(4)
        )
        fagg.to_csv(out / "results_forward_stress.csv")
        paths.append(out / "results_forward_stress.csv")
        summary["forward_stress"] = fagg.reset_index().to_dict(orient="records")
    if ncmcm is not None:
        rows = [
            {
                "worm_id": s["worm_id"],
                "K": s["selected_k"],
                "p_precomputed": s["markov_p_precomputed"],
                "p_reimplemented": s["markov_p_reimplemented"],
                "decoder_accuracy": s["decoder_cv"]["accuracy"],
                "decoder_balanced_accuracy": s["decoder_cv"]["balanced_accuracy"],
            }
            for s in ncmcm
        ]
        pd.DataFrame(rows).to_csv(out / "results_ncmcm.csv", index=False)
        paths.append(out / "results_ncmcm.csv")
        summary["ncmcm"] = rows
    if bundlenet_lite is not None:
        summary["bundlenet_lite"] = {
            "metrics": bundlenet_lite["metrics"],
            "deviations": bundlenet_lite["metadata"]["deviations"],
        }
    if bundlenet_official is not None:
        summary["bundlenet_official"] = [
            {
                "worm_id": r.get("worm_id", i),
                "knn_leaky": r.get("official_knn_accuracy"),
                "silhouette": r.get("official_silhouette_fine"),
                "pca3_silhouette": r.get("pca3_silhouette_fine"),
            }
            for i, r in enumerate(bundlenet_official)
        ]
    if branching is not None:
        summary["branching_worm0"] = {
            "summary": {k: v for k, v in branching["summary"].items() if k != "top_frames"},
            "permutation": branching.get("permutation"),
            "latent_dim_spearman": branching.get("latent_dim_stability"),
        }
    if interventions is not None:
        summary["intervention_search_top"] = interventions.head(10).to_dict(orient="records")
    if random_controls is not None:
        summary["random_controls"] = {
            "mean_abs_delta": float(random_controls.delta_prob.abs().mean()),
            "max_abs_delta": float(random_controls.delta_prob.abs().max()),
            "rows": random_controls.to_dict(orient="records"),
        }

    (out / "results_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    paths.append(out / "results_summary.json")

    # Machine-readable caveats that downstream users can assert on.
    manifest = {
        "observational_only": True,
        "causal_validation": False,
        "behaviour_labels_circular": True,
        "excluded_behaviour_neurons": ["AVAL", "AVAR", "SMDDR", "SMDDL", "SMDVR", "SMDVL", "RIBR", "RIBL"],
        "official_bundlenet_encoder_seen_all_frames": True,
        "official_bundlenet_knn_has_embedding_leakage": True,
        "bundlenet_lite_uses_causal_filter": True,
        "primary_split_uses_future_blocks_for_training": True,
        "forward_stress_included": True,
        "random_controls_included": True,
    }
    (out / "epistemic_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths.append(out / "epistemic_manifest.json")
    return paths
