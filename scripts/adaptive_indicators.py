"""Stage E: when and where does adaptation help?"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from neurobranch.adaptive.evaluation import evaluation_landmarks
from neurobranch.adaptive.nonstationarity import principal_angles_deg
from neurobranch.adaptive.var import FrozenVAR
from neurobranch.config import PROJECT_ROOT, git_state
from neurobranch.diagnostics.common import load_nostim
from neurobranch.preprocessing import split_starts

OUT = PROJECT_ROOT / "reports/simulator/adaptive_dynamics"
METRICS = ["covariance_shift", "subspace_angle", "amplitude_shift", "occupancy_shift", "recent_error"]


def landmark_indicators(data, models: dict[str, np.ndarray]) -> pd.DataFrame:
    train = data.X[data.masks["train"]]
    pca_ref = PCA(n_components=10, random_state=17).fit(train)
    q_ref = pca_ref.components_.T
    cov_ref = np.cov(train, rowvar=False)
    std_ref = train.std(axis=0)
    med_ref = np.median(train, axis=0)
    frozen = models["frozen_var"]
    rows = []
    for t in evaluation_landmarks(data, horizon_max=100, stride=10):
        recent = data.X[max(0, t - 199) : t + 1]
        q = PCA(n_components=10, random_state=17).fit(recent).components_.T
        angle = float(np.mean(principal_angles_deg(q, q_ref)))
        cov = np.cov(recent, rowvar=False)
        cov_shift = float(np.linalg.norm(cov - cov_ref) / max(1e-12, np.linalg.norm(cov_ref)))
        std = recent.std(axis=0)
        amp = float(np.mean(np.abs(np.log((std + 1e-8) / (std_ref + 1e-8)))))
        occ = float(np.mean(np.abs(np.median(recent, axis=0) - med_ref) / (std_ref + 1e-8)))
        lo = max(1, t - 99)
        starts = np.arange(lo, t)
        errs = []
        for i in starts:
            pred = frozen.A @ data.X[i] + frozen.b
            errs.append(float(np.mean((pred - data.X[i + 1]) ** 2)))
        recent_error = float(np.mean(errs)) if errs else float("nan")
        rows.append(
            {
                "worm_id": int(data.worm_id),
                "landmark": int(t),
                "covariance_shift": cov_shift,
                "subspace_angle": angle,
                "amplitude_shift": amp,
                "occupancy_shift": occ,
                "recent_error": recent_error,
            }
        )
    return pd.DataFrame(rows)


def cluster_block_bootstrap_coef(
    frame: pd.DataFrame, y: np.ndarray, X: np.ndarray, *, block: int = 5, n_boot: int = 500, seed: int = 17
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = np.linalg.lstsq(X, y, rcond=None)[0]
    draws = []
    worms = frame.worm_id.to_numpy()
    for _ in range(n_boot):
        idx = []
        for w in np.unique(worms):
            w_idx = np.flatnonzero(worms == w)
            n_blocks = int(np.ceil(len(w_idx) / block))
            starts = rng.integers(0, max(1, len(w_idx) - block + 1), size=n_blocks)
            pick = np.concatenate([w_idx[s : s + block] for s in starts])[: len(w_idx)]
            idx.extend(pick.tolist())
        idx_arr = np.asarray(idx)
        draws.append(np.linalg.lstsq(X[idx_arr], y[idx_arr], rcond=None)[0])
    draws_arr = np.asarray(draws)
    return base, np.quantile(draws_arr, 0.025, axis=0), np.quantile(draws_arr, 0.975, axis=0)


def main() -> int:
    _cfg, data, _recs = load_nostim()
    all_ind = []
    for d in data:
        lm = pd.read_csv(OUT / "model_per_landmark.csv")
        h20 = lm[(lm.worm_id == d.worm_id) & (lm.horizon == 20)]
        # Refit frozen in the same way as the model script for residual indicators.
        frozen = FrozenVAR.fit(
            d.X,
            split_starts(d, "train", history=1, horizon=1),
            ridge=1.0,
            spectral_cap=0.995,
        )
        all_ind.append(landmark_indicators(d, {"frozen_var": frozen}))
    ind = pd.concat(all_ind, ignore_index=True)
    ind.to_csv(OUT / "adaptation_indicators.csv", index=False)

    lm = pd.read_csv(OUT / "model_per_landmark.csv")
    h20 = lm[lm.horizon == 20].pivot_table(
        index=["worm_id", "landmark"], columns="model", values="mse"
    ).reset_index()
    merged = ind.merge(h20, on=["worm_id", "landmark"], how="inner")
    reg_rows, strata_rows = [], []
    for model in ("rolling_w500", "ewma_h100", "rolling_w200"):
        frame = merged.dropna(subset=["frozen_var", model]).copy()
        frame["improvement"] = frame["frozen_var"] - frame[model]
        frame["relative_improvement"] = 1 - frame[model] / frame["frozen_var"]
        for metric in METRICS:
            frame[f"z_{metric}"] = (frame[metric] - frame[metric].mean()) / max(
                1e-12, frame[metric].std()
            )
        cols = [f"z_{m}" for m in METRICS]
        X = frame[cols].to_numpy(dtype=np.float64)
        X = np.column_stack([np.ones(len(X)), X])
        for w in sorted(frame.worm_id.unique())[1:]:
            X = np.column_stack([X, (frame.worm_id == w).to_numpy(dtype=float)])
        y = frame.relative_improvement.to_numpy(dtype=np.float64)
        base, lo, hi = cluster_block_bootstrap_coef(frame, y, X)
        names = ["intercept", *METRICS, *[f"worm_{w}" for w in sorted(frame.worm_id.unique())[1:]]]
        for name, coef, lower, upper in zip(names, base, lo, hi):
            reg_rows.append(
                {
                    "model": model,
                    "term": name,
                    "coefficient": float(coef),
                    "ci_low": float(lower),
                    "ci_high": float(upper),
                    "n": int(len(frame)),
                }
            )
        for metric in METRICS:
            frame["quartile"] = pd.qcut(frame[metric], 4, labels=False, duplicates="drop")
            for q, group in frame.groupby("quartile"):
                strata_rows.append(
                    {
                        "model": model,
                        "indicator": metric,
                        "quartile": int(q),
                        "n": int(len(group)),
                        "mean_relative_improvement": float(group.relative_improvement.mean()),
                        "mean_indicator": float(group[metric].mean()),
                    }
                )
    reg = pd.DataFrame(reg_rows)
    strata = pd.DataFrame(strata_rows)
    reg.to_csv(OUT / "adaptation_regression.csv", index=False)
    strata.to_csv(OUT / "adaptation_strata.csv", index=False)
    correlations = []
    for model in ("rolling_w500", "ewma_h100"):
        frame = merged.dropna(subset=["frozen_var", model]).copy()
        frame["relative_improvement"] = 1 - frame[model] / frame["frozen_var"]
        for metric in METRICS:
            correlations.append(
                {
                    "model": model,
                    "indicator": metric,
                    "pearson": float(frame[metric].corr(frame.relative_improvement)),
                    "spearman": float(frame[metric].corr(frame.relative_improvement, method="spearman")),
                }
            )
    pd.DataFrame(correlations).to_csv(OUT / "adaptation_correlations.csv", index=False)
    payload = {
        "stage": "E_adaptation_heterogeneity",
        "n_landmarks": int(len(merged)),
        "indicators": METRICS,
        "available_at_prediction_time": True,
        "regression": reg.to_dict(orient="records"),
        "correlations": correlations,
        "git": git_state(),
    }
    (OUT / "adaptation_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(reg[reg.term.isin(METRICS)].to_string(index=False))
    print(pd.DataFrame(correlations).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
