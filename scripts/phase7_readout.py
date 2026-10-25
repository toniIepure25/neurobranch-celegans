"""Anatomical-association and anatomy-unexplained readout for Phase 7."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from neurobranch.causal.features import load_connectome_adjacency
from neurobranch.config import PROJECT_ROOT, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="reports/phase7")
    parser.add_argument(
        "--connectome",
        default="references/external/aconnectome_witvliet_2020_8_utf8.csv",
    )
    args = parser.parse_args()
    root = resolve_path(args.root, PROJECT_ROOT)
    pred_path = root / "predictions.parquet"
    if not pred_path.exists():
        raise SystemExit("primary predictions missing; run the active benchmark first")
    frame = pd.read_parquet(pred_path)
    if "strategy" not in frame.columns:
        frame["strategy"] = frame["model"]
    pairs = (
        frame.groupby(["strategy", "target", "neuron"])
        .agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"), n_events=("y_true", "size"))
        .reset_index()
    )
    pairs = pairs[pairs["n_events"] >= 2]
    targets = sorted(pairs["target"].unique())
    responders = sorted(pairs["neuron"].unique())
    chem, elec, nodes, _ = load_connectome_adjacency(
        args.connectome, targets=targets, responders=responders
    )
    node_map = {n: i for i, n in enumerate(nodes)}
    rows = []
    for strategy, group in pairs.groupby("strategy"):
        direct = np.array(
            [
                (
                    (node_map.get(t, -1) >= 0)
                    and (node_map.get(r, -1) >= 0)
                    and (
                        chem[node_map[t], node_map[r]] > 0
                        or elec[node_map[t], node_map[r]] > 0
                    )
                )
                for t, r in zip(group["target"], group["neuron"])
            ],
            dtype=bool,
        )
        magnitude = np.abs(group["y_pred"].to_numpy(dtype=float))
        observed = np.abs(group["y_true"].to_numpy(dtype=float))
        edge_auc = (
            float(roc_auc_score(direct, magnitude))
            if len(np.unique(direct)) > 1
            else float("nan")
        )
        threshold = float(np.quantile(observed, 0.90))
        unexplained = (~direct) & (observed >= threshold)
        if len(np.unique(unexplained)) > 1:
            unexplained_auc = float(roc_auc_score(unexplained, magnitude))
        else:
            unexplained_auc = float("nan")
        rows.append(
            {
                "strategy": strategy,
                "score": "predicted_magnitude",
                "auroc_direct_edge": edge_auc,
                "auroc_anatomy_unexplained": unexplained_auc,
                "n_pairs": int(len(group)),
                "n_direct": int(direct.sum()),
                "n_unexplained": int(unexplained.sum()),
            }
        )
    edge_metrics = pd.DataFrame(rows).sort_values("auroc_direct_edge", ascending=False)
    out = root / "connectivity"
    out.mkdir(parents=True, exist_ok=True)
    edge_metrics.to_csv(out / "edge_metrics.csv", index=False)

    hybrid = pairs[pairs["strategy"] == "hybrid"].copy()
    if not hybrid.empty:
        direct = np.array(
            [
                (
                    (node_map.get(t, -1) >= 0)
                    and (node_map.get(r, -1) >= 0)
                    and (
                        chem[node_map[t], node_map[r]] > 0
                        or elec[node_map[t], node_map[r]] > 0
                    )
                )
                for t, r in zip(hybrid["target"], hybrid["neuron"])
            ],
            dtype=bool,
        )
        hybrid["direct_edge"] = direct
        hybrid["observed_magnitude"] = hybrid["y_true"].abs()
        hybrid["predicted_magnitude"] = hybrid["y_pred"].abs()
        candidates = hybrid[~hybrid["direct_edge"]].sort_values(
            "observed_magnitude", ascending=False
        )
        candidates["rank"] = np.arange(1, len(candidates) + 1)
        candidates = candidates[
            ["rank", "target", "neuron", "y_true", "y_pred", "n_events"]
        ].head(100)
        candidates.to_csv(root / "discrepancy_pairs.csv", index=False)
    summary = {
        "edge_metrics": edge_metrics.to_dict(orient="records"),
        "warning": "Anatomical association only; not causal discovery.",
        "discrepancy_definition": (
            "No direct chemical/electrical edge and observed |response| in the top decile "
            "of evaluated test pairs. Labelled anatomy-unexplained coupling candidate, "
            "not extrasynaptic mechanism."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
