"""Independent statistical audit of the frozen B3-vs-B5 landmark table.

No model is refitted.  The script re-quantifies the frozen paired differences at
the session (animal) level, by horizon, and under a cluster bootstrap that
resamples recordings as well as within-recording blocks.  Its purpose is to make
the unit-of-inference problem explicit.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
FRAME = ROOT / "reports" / "simulator" / "kato_common_state" / "model_per_landmark.csv"
OUT = ROOT / "reports" / "master_review"
MODELS = ("B3_population", "B5_common_k10")
PRIMARY = "pca10_energy_h20"


def load_pairs() -> pd.DataFrame:
    frame = pd.read_csv(FRAME)
    keep = frame[frame["model"].isin(MODELS)][
        ["session", "landmark", "model", PRIMARY, "pca10_energy_h1", "pca10_energy_h5", "pca10_energy_h10"]
    ]
    wide = keep.pivot_table(index=["session", "landmark"], columns="model", values=PRIMARY)
    wide = wide.dropna().reset_index()
    wide["advantage"] = wide["B5_common_k10"] - wide["B3_population"]
    for h in (1, 5, 10):
        col = f"pca10_energy_h{h}"
        sub = frame[frame["model"].isin(MODELS)][["session", "landmark", "model", col]].pivot_table(
            index=["session", "landmark"], columns="model", values=col
        )
        sub = sub.dropna().reset_index()
        sub[f"advantage_h{h}"] = sub["B5_common_k10"] - sub["B3_population"]
        wide = wide.merge(sub[["session", "landmark", f"advantage_h{h}"]], on=["session", "landmark"], how="left")
    return wide.sort_values(["session", "landmark"]).reset_index(drop=True)


def block_sample(values: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    block = max(1, min(int(block), n))
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    return np.concatenate([values[s : s + block] for s in starts])[:n]


def main() -> None:
    pairs = load_pairs()
    sessions = sorted(pairs["session"].unique())
    groups = {int(s): pairs[pairs["session"] == s]["advantage"].to_numpy() for s in sessions}
    session_means = {int(s): float(g.mean()) for s, g in groups.items()}
    values = pairs["advantage"].to_numpy()
    equal_session = float(np.mean(list(session_means.values())))
    weighted = float(values.mean())
    sd_sessions = float(np.std(list(session_means.values()), ddof=1))
    se_sessions = sd_sessions / math.sqrt(len(sessions))
    tcrit = float(stats.t.ppf(0.975, len(sessions) - 1))
    t_lo = equal_session - tcrit * se_sessions
    t_hi = equal_session + tcrit * se_sessions
    n_positive = int(sum(m > 0 for m in session_means.values()))
    sign_one_sided = float(stats.binomtest(n_positive, len(sessions), 0.5, alternative="greater").pvalue)
    sign_two_sided = float(stats.binomtest(n_positive, len(sessions), 0.5).pvalue)

    # Cluster bootstrap: resample recordings, then moving blocks within each selected recording.
    rng = np.random.default_rng(20261103)
    cluster = {}
    for block in (5, 20, 100):
        means = []
        for _ in range(5000):
            chosen = rng.choice(sessions, size=len(sessions), replace=True)
            sampled = [block_sample(groups[int(s)], block, rng).mean() for s in chosen]
            means.append(float(np.mean(sampled)))
        arr = np.asarray(means)
        cluster[str(block)] = {
            "block": block,
            "mean": float(arr.mean()),
            "ci_low": float(np.quantile(arr, 0.025)),
            "ci_high": float(np.quantile(arr, 0.975)),
            "fraction_le_zero": float((arr <= 0).mean()),
            "n_boot": 5000,
            "resamples_recordings": True,
        }

    # Horizon-by-session descriptive table.
    horizon_rows = []
    for h in (1, 5, 10, 20):
        col = "advantage" if h == 20 else f"advantage_h{h}"
        sub = pairs.dropna(subset=[col])
        for s in sessions:
            g = sub[sub["session"] == s][col].to_numpy()
            horizon_rows.append(
                {
                    "horizon": h,
                    "session": int(s),
                    "n": int(len(g)),
                    "mean_advantage": float(g.mean()),
                    "fraction_better": float((g > 0).mean()),
                }
            )
        hg = sub[col].to_numpy()
        horizon_rows.append(
            {
                "horizon": h,
                "session": "all",
                "n": int(len(hg)),
                "mean_advantage": float(hg.mean()),
                "fraction_better": float((hg > 0).mean()),
            }
        )

    # Effective units: overlapping 100-frame forecasts cannot exceed one per 100 frames.
    units = []
    for s in sessions:
        g = pairs[pairs["session"] == s]
        test_frames = int(g["landmark"].max() - g["landmark"].min() + 100)
        units.append(
            {
                "session": int(s),
                "n_landmarks": int(len(g)),
                "test_frames_span": test_frames,
                "nonoverlapping_h100_windows_approx": int(math.ceil(test_frames / 100)),
                "landmark_to_window_ratio": float(len(g) / math.ceil(test_frames / 100)),
            }
        )

    payload = {
        "schema": "neurobranch.master_review.statistics.v1",
        "source": str(FRAME.relative_to(ROOT)),
        "contrast": "B5_common_k10 - B3_population on pca10_energy_h20; positive = B3 lower energy",
        "n_landmarks": int(len(pairs)),
        "n_sessions": int(len(sessions)),
        "weighted_landmark_mean": weighted,
        "equal_session_mean": equal_session,
        "session_means": {str(k): v for k, v in session_means.items()},
        "session_sd": sd_sessions,
        "session_t_interval_95": [float(t_lo), float(t_hi)],
        "session_t_interval_note": "t interval across five recordings; treats recordings as the independent units and ignores within-session landmark precision.",
        "sessions_positive": n_positive,
        "sign_test_one_sided_p": sign_one_sided,
        "sign_test_two_sided_p": sign_two_sided,
        "cluster_bootstrap": cluster,
        "horizon": horizon_rows,
        "effective_units": units,
        "overall_nonoverlapping_proxy": int(sum(u["nonoverlapping_h100_windows_approx"] for u in units)),
        "landmark_inflation_factor": float(len(pairs) / sum(u["nonoverlapping_h100_windows_approx"] for u in units)),
        "multiplicity": {
            "primary_horizon": 20,
            "secondary_horizons_tested": [1, 5, 10],
            "horizon_results_are_descriptive": True,
            "bonferroni_95_percent_threshold_if_4_tests": 0.0125,
            "note": "The committed analysis reports four horizons; H20 was primary, but H1 is negative and the horizon profile is part of the result. No family-wise confirmatory claim is made.",
        },
        "interpretation": {
            "unit_of_inference": "The 175 landmarks are overlapping windows from five recordings. The within-session moving-block interval is conditional on these five recordings. The cluster bootstrap is the appropriate uncertainty check for recording-level generalization, but five units give very low power.",
            "strongest_defensible_statement": "In these five recordings, B3 had lower H20 PCA-10 energy than B5-10 in four of five sessions (equal-session mean +0.26); the recording-level t interval includes zero and the sign test is not significant. The result is exploratory and horizon-dependent.",
            "h1_crossover": "At H1 the contrast reverses: B3 is worse than B5-10 by 0.064 energy units. Any claim of a general predictive advantage must be restricted to multi-step horizons and cannot be described as uniform.",
        },
    }
    (OUT / "statistical_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "session": s,
                "n_landmarks": len(groups[s]),
                "mean_advantage": session_means[s],
                "sd": float(np.std(groups[s], ddof=1)),
                "fraction_better": float((groups[s] > 0).mean()),
            }
            for s in sessions
        ]
    ).to_csv(OUT / "session_level_estimates.csv", index=False)
    pd.DataFrame(horizon_rows).to_csv(OUT / "horizon_session_estimates.csv", index=False)
    pd.DataFrame(units).to_csv(OUT / "effective_units.csv", index=False)
    pd.DataFrame(
        [{"block": int(k), **v} for k, v in cluster.items()]
    ).to_csv(OUT / "cluster_bootstrap.csv", index=False)

    print(json.dumps({k: payload[k] for k in (
        "weighted_landmark_mean",
        "equal_session_mean",
        "session_means",
        "session_t_interval_95",
        "sessions_positive",
        "sign_test_one_sided_p",
        "cluster_bootstrap",
        "overall_nonoverlapping_proxy",
        "landmark_inflation_factor",
    )}, indent=2))


if __name__ == "__main__":
    main()
