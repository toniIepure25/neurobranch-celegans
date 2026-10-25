"""Anatomical association benchmark for predicted response profiles.

This is deliberately *not* called causal discovery.  It asks whether a model's
held-out response profile ranks directly connected target-responder pairs above
unconnected pairs.  A positive result is consistent with, but not proof of, a
causal effect of the connectome.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from neurobranch.causal.features import FeatureBundle
from neurobranch.config import PROJECT_ROOT, resolve_path


def _unique_pairs(bundle: FeatureBundle) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "target": [bundle.target_names[i] for i in bundle.target_idx],
            "neuron": [bundle.responder_names[i] for i in bundle.responder_idx],
            "direct_chem": bundle.direct_chem,
            "direct_elec": bundle.direct_elec,
            "twohop_chem": bundle.twohop_chem,
            "twohop_elec": bundle.twohop_elec,
            "known": bundle.pair_known,
        }
    )
    frame = frame.groupby(["target", "neuron"], as_index=False).agg(
        direct_chem=("direct_chem", "max"),
        direct_elec=("direct_elec", "max"),
        twohop_chem=("twohop_chem", "max"),
        twohop_elec=("twohop_elec", "max"),
        known=("known", "max"),
    )
    frame["direct"] = ((frame["direct_chem"] > 0) | (frame["direct_elec"] > 0)).astype(int)
    frame["twohop_only"] = (
        (frame["direct"] == 0) & ((frame["twohop_chem"] > 0) | (frame["twohop_elec"] > 0))
    ).astype(int)
    return frame


def _per_target_auc(
    pairs: pd.DataFrame,
    *,
    score_col: str,
    label_col: str = "direct",
    min_pos: int = 2,
    min_neg: int = 2,
) -> pd.DataFrame:
    rows = []
    for target, group in pairs.groupby("target"):
        y = group[label_col].to_numpy()
        if y.sum() < min_pos or (1 - y).sum() < min_neg:
            continue
        score = group[score_col].to_numpy(dtype=float)
        if not np.isfinite(score).all() or np.unique(score).size < 2:
            continue
        rows.append(
            {
                "target": target,
                "n_pairs": len(group),
                "n_positive": int(y.sum()),
                "auroc": float(roc_auc_score(y, score)),
                "auprc": float(average_precision_score(y, score)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_connectivity(
    bundle: FeatureBundle,
    predictions: pd.DataFrame,
    *,
    output_dir: str | Path,
    root: str | Path | None = None,
) -> dict[str, object]:
    base = Path(root) if root is not None else PROJECT_ROOT
    out = resolve_path(output_dir, base)
    out.mkdir(parents=True, exist_ok=True)
    labels = _unique_pairs(bundle)
    # Aggregate model predictions to one score per pair.
    pair_scores = predictions.groupby(["model", "target", "neuron"], as_index=False).agg(
        y_pred=("y_pred", "mean"),
        y_true=("y_true", "mean"),
        n_events=("y_true", "size"),
    )
    pair_scores["abs_pred"] = pair_scores["y_pred"].abs()
    pair_scores["abs_observed"] = pair_scores["y_true"].abs()
    merged = pair_scores.merge(labels, on=["target", "neuron"], how="inner")
    merged = merged[merged["known"]]
    merged.to_csv(out / "edge_pair_scores.csv.gz", index=False, compression="gzip")

    rows = []
    for model, group in merged.groupby("model"):
        for score_col, score_name in (("abs_pred", "predicted_magnitude"), ("y_pred", "signed_prediction")):
            detail = _per_target_auc(group, score_col=score_col)
            if detail.empty:
                continue
            rows.append(
                {
                    "model": model,
                    "score": score_name,
                    "n_targets": len(detail),
                    "mean_auroc": float(detail["auroc"].mean()),
                    "median_auroc": float(detail["auroc"].median()),
                    "mean_auprc": float(detail["auprc"].mean()),
                    "fraction_auroc_above_chance": float((detail["auroc"] > 0.5).mean()),
                }
            )
        detail = _per_target_auc(group, score_col="abs_observed")
        if not detail.empty:
            rows.append(
                {
                    "model": model,
                    "score": "observed_magnitude",
                    "n_targets": len(detail),
                    "mean_auroc": float(detail["auroc"].mean()),
                    "median_auroc": float(detail["auroc"].median()),
                    "mean_auprc": float(detail["auprc"].mean()),
                    "fraction_auroc_above_chance": float((detail["auroc"] > 0.5).mean()),
                }
            )
    metrics = pd.DataFrame(rows).sort_values(["score", "mean_auroc"], ascending=[True, False])
    metrics.to_csv(out / "edge_metrics.csv", index=False)

    # Label-shuffled null within target: the ranking should collapse to chance.
    null_rows = []
    rng = np.random.default_rng(17)
    cro = merged[merged["model"] == "cro"]
    for _ in range(100):
        shuffled = cro.copy()
        shuffled["direct"] = shuffled.groupby("target")["direct"].transform(
            lambda s: rng.permutation(s.to_numpy())
        )
        detail = _per_target_auc(shuffled, score_col="abs_pred")
        if not detail.empty:
            null_rows.append(float(detail["auroc"].mean()))
    null = np.asarray(null_rows, dtype=float)
    unique_pairs = merged.drop_duplicates(["target", "neuron"])
    summary = {
        "n_pairs": int(len(unique_pairs)),
        "n_targets": int(merged["target"].nunique()),
        "n_positive_pairs": int(unique_pairs["direct"].sum()),
        "metrics": metrics.to_dict(orient="records"),
        "cro_label_shuffled_null": {
            "mean": float(np.nanmean(null)) if null.size else float("nan"),
            "q2_5": float(np.nanpercentile(null, 2.5)) if null.size else float("nan"),
            "q97_5": float(np.nanpercentile(null, 97.5)) if null.size else float("nan"),
            "n_permutations": int(null.size),
        },
    }
    (out / "edge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def direct_twohop_gradient(bundle: FeatureBundle, predictions: pd.DataFrame) -> pd.DataFrame:
    """Mean predicted magnitude by direct / two-hop-only / unconnected pairs."""
    labels = _unique_pairs(bundle)
    scores = predictions.groupby(["model", "target", "neuron"], as_index=False)["y_pred"].mean()
    scores["abs_pred"] = scores["y_pred"].abs()
    merged = scores.merge(labels, on=["target", "neuron"], how="inner")
    merged["relation"] = np.where(
        merged["direct"] == 1,
        "direct",
        np.where(merged["twohop_only"] == 1, "two_hop_only", "unconnected"),
    )
    out = merged.groupby(["model", "relation"])["abs_pred"].agg(["mean", "median", "count"]).reset_index()
    return out
