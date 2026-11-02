"""Stage E extension: adaptation around measured oxygen transitions in WT_Stim."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from neurobranch.adaptive.evaluation import block_bootstrap_ci, evaluate_session
from neurobranch.config import PROJECT_ROOT, git_state
from neurobranch.data.external import build_kato_stim_context_worm_data, load_kato_stim

OUT = PROJECT_ROOT / "reports/simulator/adaptive_dynamics"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ("rolling_w500", "ewma_h1000")
WINDOW = 20


def main() -> int:
    path = Path(os.environ.get("TEMP", ".")) / "neurobranch_external" / "WT_Stim.mat"
    if not path.exists():
        print("WT_Stim not present; oxygen stratification skipped")
        return 0
    recordings = load_kato_stim(path, verify=True)
    rows, updates = [], []
    for rec in recordings:
        data = build_kato_stim_context_worm_data(rec)
        oxygen = np.asarray(data.external_input, dtype=np.float64)
        changes = np.flatnonzero(np.diff(oxygen) != 0) + 1
        since = np.full(len(oxygen), np.inf)
        direction = np.zeros(len(oxygen), dtype=np.int8)
        last = -10**9
        change_set = set(int(i) for i in changes)
        for t in range(len(oxygen)):
            if t in change_set:
                last = t
                direction[t] = 1 if oxygen[t] > oxygen[t - 1] else -1
            since[t] = t - last
        lm, _neu, upd = evaluate_session(data, horizon_max=100, stride=10)
        updates.append(upd)
        h20 = lm[lm.horizon == 20].pivot_table(
            index="landmark", columns="model", values="mse"
        )
        for t, row in h20.iterrows():
            if "frozen_var" not in row:
                continue
            entry = {
                "session": str(rec.session_id),
                "worm_id": int(data.worm_id),
                "landmark": int(t),
                "oxygen_percent": float(oxygen[int(t)]),
                "frames_since_transition": float(since[int(t)]),
                "near_transition": bool(since[int(t)] <= WINDOW),
                "frozen_mse": float(row["frozen_var"]),
                "persistence_mse": float(
                    np.mean((data.X[int(t)] - data.X[int(t) + 20]) ** 2)
                ),
            }
            for model in MODELS:
                if model in row:
                    entry[f"{model}_mse"] = float(row[model])
                    entry[f"{model}_improvement"] = float(row["frozen_var"] - row[model])
            rows.append(entry)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "oxygen_adaptive_per_landmark.csv", index=False)

    summary = []
    for model in MODELS:
        col = f"{model}_improvement"
        if col not in frame:
            continue
        for subset, group in frame.groupby("near_transition"):
            diff = group[col].dropna().to_numpy()
            mean, lo, hi = block_bootstrap_ci(diff, block=5)
            summary.append(
                {
                    "model": model,
                    "subset": "within_20_frames_of_transition" if subset else "outside_transition_window",
                    "n": int(len(diff)),
                    "mean_improvement": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
        for level, group in frame.groupby("oxygen_percent"):
            diff = group[col].dropna().to_numpy()
            mean, lo, hi = block_bootstrap_ci(diff, block=5)
            summary.append(
                {
                    "model": model,
                    "subset": f"oxygen_{level:g}_percent",
                    "n": int(len(diff)),
                    "mean_improvement": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT / "oxygen_adaptive_summary.csv", index=False)
    pd.concat(updates, ignore_index=True).to_csv(OUT / "oxygen_adaptive_updates.csv", index=False)
    payload = {
        "stage": "E_oxygen_transition_stratification",
        "n_sessions": int(len(recordings)),
        "n_landmarks": int(len(frame)),
        "window_frames": WINDOW,
        "caveat": (
            "Seven repeated sessions, not independent animals. The 30 s oxygen schedule is fixed "
            "across sessions, so transition proximity remains confounded with elapsed time. Oxygen "
            "is an environmental sensory input, not targeted neuronal stimulation."
        ),
        "summary": summary_df.to_dict(orient="records"),
        "git": git_state(),
    }
    (OUT / "oxygen_adaptive_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
