"""Stage B: is adaptive population VAR better than local-mean tracking?"""

from __future__ import annotations

import json

import pandas as pd

from neurobranch.config import PROJECT_ROOT, git_state
from neurobranch.diagnostics.common import load_nostim
from neurobranch.probabilistic.baselines import (
    evaluate_deterministic_session,
    paired_deterministic,
    select_baseline_config,
)

OUT = PROJECT_ROOT / "reports/simulator/probabilistic_adaptive"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    _cfg, data, _recs = load_nostim()
    config, validation = select_baseline_config(data)
    validation.to_csv(OUT / "baseline_validation.csv", index=False)
    landmark, neuron, updates = [], [], []
    for d in data:
        lm, neu, upd = evaluate_deterministic_session(d, config)
        landmark.append(lm)
        neuron.append(neu)
        updates.append(upd)
    lm = pd.concat(landmark, ignore_index=True)
    neu = pd.concat(neuron, ignore_index=True)
    upd = pd.concat(updates, ignore_index=True)
    lm.to_csv(OUT / "baseline_per_landmark.csv", index=False)
    neu.to_csv(OUT / "baseline_per_neuron_h20.csv", index=False)
    upd.to_csv(OUT / "baseline_updates.csv", index=False)

    h20 = lm[lm.horizon == 20]
    session = (
        h20.groupby(["worm_id", "model"], as_index=False)
        .agg(mean_mse=("mse", "mean"), mean_skill=("skill_vs_persistence", "mean"))
        .sort_values(["worm_id", "mean_mse"])
    )
    session.to_csv(OUT / "baseline_per_session_h20.csv", index=False)
    horizon = (
        lm.groupby(["model", "horizon"], as_index=False)
        .agg(mean_mse=("mse", "mean"), mean_skill=("skill_vs_persistence", "mean"))
    )
    horizon.to_csv(OUT / "baseline_horizon.csv", index=False)

    overall = paired_deterministic(lm, reference="B1_frozen_var")
    overall.to_csv(OUT / "baseline_paired_vs_frozen.csv", index=False)
    for ref, name in (
        ("B0_persistence", "vs_persistence"),
        ("B2_local_mean", "vs_local_mean"),
        ("B3_ewma_mean", "vs_ewma_mean"),
        ("B4_independent_ar", "vs_independent_ar"),
    ):
        paired_deterministic(lm, reference=ref).to_csv(OUT / f"baseline_paired_{name}.csv", index=False)

    by_session = []
    for worm, group in lm.groupby("worm_id"):
        for ref, name in (
            ("B1_frozen_var", "vs_frozen"),
            ("B2_local_mean", "vs_local_mean"),
            ("B4_independent_ar", "vs_independent_ar"),
        ):
            p = paired_deterministic(group, reference=ref)
            p["worm_id"] = int(worm)
            p["comparison"] = name
            by_session.append(p)
    pd.concat(by_session, ignore_index=True).to_csv(OUT / "baseline_paired_by_session.csv", index=False)

    payload = {
        "stage": "B_local_mean_and_independent_ar_ablation",
        "git": git_state(),
        "selected_config": {
            "local_mean_window": config.local_window,
            "ewma_halflife": config.ewma_halflife,
            "independent_ar_window": config.ar_window,
        },
        "selection_data": "last 20% of the training split only; no test frames",
        "candidate_sets": {
            "local_windows": [100, 200, 500, 1000],
            "ewma_halflives": [100, 300, 1000],
            "ar_windows": [200, 500, 1000],
        },
        "baselines": {
            "B0": "persistence (last observation)",
            "B1": "frozen train-only VAR",
            "B2": "rolling local mean (selected window)",
            "B3": "exponentially weighted local mean",
            "B4": "independent per-neuron adaptive AR(1)",
            "B5": "prespecified full Rolling VAR w500",
        },
        "primary_question": "Does B5 improve over B2/B4 on matched H20 landmarks?",
        "evidence_tier": "exploratory repeated-session reanalysis",
    }
    (OUT / "baseline_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(validation.to_string(index=False))
    print(session.to_string(index=False))
    print(overall.to_string(index=False))
    print(pd.read_csv(OUT / "baseline_paired_vs_local_mean.csv").to_string(index=False))
    print(pd.read_csv(OUT / "baseline_paired_vs_independent_ar.csv").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
