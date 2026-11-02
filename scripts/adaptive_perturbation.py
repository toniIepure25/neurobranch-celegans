"""Stage F: descriptive impulse sensitivity under frozen and adaptive VARs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from neurobranch.adaptive.evaluation import evaluation_landmarks
from neurobranch.adaptive.var import FrozenVAR, RollingVAR
from neurobranch.config import PROJECT_ROOT, git_state
from neurobranch.diagnostics.common import load_nostim
from neurobranch.preprocessing import split_starts

OUT = PROJECT_ROOT / "reports/simulator/adaptive_dynamics"
WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED."


def trajectory_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.sum(a * b) / denom) if denom > 0 else float("nan")


def main() -> int:
    _cfg, data, _recs = load_nostim()
    rows = []
    for d in data:
        frozen = FrozenVAR.fit(
            d.X,
            split_starts(d, "train", history=1, horizon=1),
            ridge=1.0,
            spectral_cap=0.995,
        )
        landmarks = evaluation_landmarks(d, horizon_max=100, stride=10)
        selected = landmarks[[0, len(landmarks) // 2, -1]]
        # Prespecified alphabetical subset; no ranking or selection by effect size.
        neuron_names = sorted(d.predictor_names)[:5]
        indices = [d.predictor_names.index(n) for n in neuron_names]
        for t in selected:
            adaptive = RollingVAR(window=500).fit_at(d.X, int(t))
            if adaptive is None:
                continue
            for name, idx in zip(neuron_names, indices):
                x0 = d.X[int(t)].copy()
                xp = x0.copy()
                xp[idx] += 1.0  # one train-SD bounded impulse
                f0 = frozen.forecast(x0, 20)
                f1 = frozen.forecast(xp, 20)
                a0 = adaptive.forecast(x0, 20)
                a1 = adaptive.forecast(xp, 20)
                delta_f = (f1 - f0).ravel()
                delta_a = (a1 - a0).ravel()
                rows.append(
                    {
                        "worm_id": int(d.worm_id),
                        "landmark": int(t),
                        "neuron": name,
                        "neuron_index": int(idx),
                        "cosine_frozen_adaptive": trajectory_cosine(delta_f, delta_a),
                        "norm_frozen": float(np.linalg.norm(delta_f)),
                        "norm_adaptive": float(np.linalg.norm(delta_a)),
                        "max_abs_delta_frozen": float(np.max(np.abs(delta_f))),
                        "max_abs_delta_adaptive": float(np.max(np.abs(delta_a))),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "perturbation_sensitivity.csv", index=False)
    summary = (
        df.groupby("neuron", as_index=False)
        .agg(
            n=("cosine_frozen_adaptive", "size"),
            median_cosine=("cosine_frozen_adaptive", "median"),
            min_cosine=("cosine_frozen_adaptive", "min"),
            max_cosine=("cosine_frozen_adaptive", "max"),
        )
        .sort_values("neuron")
    )
    summary.to_csv(OUT / "perturbation_summary.csv", index=False)
    payload = {
        "stage": "F_descriptive_sensitivity",
        "status": "descriptive only; no biological intervention validation",
        "warning": WARNING,
        "n_trajectories": int(len(df)),
        "median_cosine": float(df.cosine_frozen_adaptive.median()) if len(df) else float("nan"),
        "range_cosine": [float(df.cosine_frozen_adaptive.min()), float(df.cosine_frozen_adaptive.max())] if len(df) else [],
        "git": git_state(),
    }
    (OUT / "perturbation_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(WARNING)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
