"""Stage A: time-resolved nonstationarity audit."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from neurobranch.adaptive.nonstationarity import (
    stationary_null,
    summarize_windows,
    synthetic_observed,
    window_stats,
)
from neurobranch.config import PROJECT_ROOT, git_state
from neurobranch.diagnostics.common import load_nostim

OUT = PROJECT_ROOT / "reports/simulator/adaptive_dynamics"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    _cfg, data, _recs = load_nostim()
    windows, nulls, summaries = [], [], []
    for d in data:
        stats = window_stats(
            d.X, reference_mask=d.masks["train"], window=256, step=64, k=10
        )
        stats["worm_id"] = int(d.worm_id)
        stats["fps"] = float(d.recording.fps)
        stats["window_seconds"] = 256.0 / float(d.recording.fps)
        windows.append(stats)
        null = stationary_null(
            d.X,
            d.masks["train"],
            window=256,
            step=64,
            k=10,
            n_surrogates=8,
        )
        null["worm_id"] = int(d.worm_id)
        nulls.append(null)
        summary = summarize_windows(stats, null)
        summary["worm_id"] = int(d.worm_id)
        summary["fps"] = float(d.recording.fps)
        summaries.append(summary)

    window_df = pd.concat(windows, ignore_index=True)
    null_df = pd.concat(nulls, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    window_df.to_csv(OUT / "nonstationarity_windows.csv", index=False)
    null_df.to_csv(OUT / "nonstationarity_null.csv", index=False)
    summary_df.to_csv(OUT / "nonstationarity_summary.csv", index=False)

    synthetic_rows = []
    for regime in (False, True):
        observed = synthetic_observed(regime=regime)
        reference = np.zeros(len(observed), dtype=bool)
        reference[: int(0.3 * len(observed))] = True
        stats = window_stats(observed, reference_mask=reference, window=256, step=64, k=10)
        null = stationary_null(
            observed, reference, window=256, step=64, k=10, n_surrogates=4
        )
        summary = summarize_windows(stats, null)
        summary["synthetic_condition"] = "regime_switch" if regime else "stationary"
        synthetic_rows.append(summary)
    synthetic_df = pd.DataFrame(synthetic_rows)
    synthetic_df.to_csv(OUT / "nonstationarity_synthetic.csv", index=False)

    payload = {
        "stage": "A_nonstationarity",
        "evidence_tier": "exploratory repeated-session; previously inspected test tails",
        "window_frames": 256,
        "step_frames": 64,
        "principal_components": 10,
        "summary": summary_df.to_dict(orient="records"),
        "synthetic": synthetic_df.to_dict(orient="records"),
        "git": git_state(),
    }
    (OUT / "nonstationarity_metadata.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(summary_df.to_string(index=False))
    print(synthetic_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
