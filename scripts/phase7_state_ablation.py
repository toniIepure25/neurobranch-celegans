"""Optional H3 ablation: base CRO versus pre-stimulus state-conditioned CRO."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from neurobranch.active.simulate import (
    ActiveConfig,
    _evaluate,
    _fit_ensemble,
)
from neurobranch.active.state_features import (
    attach_state_features,
    fit_state_projector,
    load_state_cache,
)
from neurobranch.benchmark.randi_response import load_trials
from neurobranch.benchmark.reliability import split_half_target_profiles
from neurobranch.benchmark.splits import build_folds
from neurobranch.causal.features import build_feature_bundle
from neurobranch.config import PROJECT_ROOT, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase7/active.yaml")
    parser.add_argument("--output", default="reports/phase7/state_ablation")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--model-seeds", default="17,23,31")
    args = parser.parse_args()
    import yaml

    raw = yaml.safe_load(resolve_path(args.config, PROJECT_ROOT).read_text(encoding="utf-8"))
    cfg = ActiveConfig(
        **{
            k: v
            for k, v in raw.items()
            if k in ActiveConfig.__dataclass_fields__ and k not in {"budgets", "strategies"}
        }
    )
    seeds = tuple(int(x) for x in args.model_seeds.split(","))
    trials = load_trials(cfg.trials_path)
    bundle = build_feature_bundle(
        trials, connectome_path=cfg.connectome_path, response_column=cfg.response_column
    )
    cache = load_state_cache(resolve_path(cfg.state_path or "reports/phase7/state/event_state.npz", PROJECT_ROOT))
    folds = build_folds(trials, cfg.outer_split, n_folds=cfg.n_folds, seed=17)[: args.folds]
    rows = []
    for fold in folds:
        train_targets = set(
            bundle.target_names[i] for i in np.unique(bundle.target_idx[fold.train_idx])
        )
        train_events = set(bundle.event_ids[fold.train_idx].astype(str))
        projector = fit_state_projector(cache, train_event_ids=train_events, n_components=16)
        state_bundle = attach_state_features(bundle, projector, cache)
        _, reliability = split_half_target_profiles(
            trials.iloc[fold.test_idx], response_column=cfg.response_column, n_splits=10, min_events=2
        )
        for label, data in (("base", bundle), ("state", state_bundle)):
            ensemble = _fit_ensemble(
                data,
                acquired_targets=train_targets,
                fold_train_idx=fold.train_idx,
                fold_val_idx=fold.val_idx,
                config=replace(cfg, use_state=(label == "state")),
                model_seeds=seeds,
            )
            row, _ = _evaluate(
                data,
                ensemble,
                test_idx=fold.test_idx,
                val_idx=fold.val_idx,
                strategy=label,
                fold=fold.fold,
                budget=cfg.reference_budget,
                n_targets=len(train_targets),
                reliability_detail=reliability,
                seed=0,
            )
            rows.append(row)
    frame = pd.DataFrame(rows)
    out = Path(resolve_path(args.output, PROJECT_ROOT))
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "metrics.csv", index=False)
    pivot = frame.pivot_table(index="fold", columns="strategy", values="pair_spearman")
    delta = (pivot["state"] - pivot["base"]).dropna()
    summary = {
        "n_folds": int(len(delta)),
        "mean_delta_state_minus_base": float(delta.mean()) if len(delta) else float("nan"),
        "wins": int((delta > 0).sum()),
        "h3_supported": bool(len(delta) and delta.mean() >= 0.01),
        "metrics": frame.to_dict(orient="records"),
        "warning": "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
