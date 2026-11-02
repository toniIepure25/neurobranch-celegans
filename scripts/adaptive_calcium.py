"""Stage B: diagnose and correct calcium-LDS autonomous divergence."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from neurobranch.adaptive.calcium import CorrectedCalciumForecast, diagnose_fit
from neurobranch.adaptive.evaluation import block_bootstrap_ci, evaluation_landmarks
from neurobranch.config import PROJECT_ROOT, git_state
from neurobranch.diagnostics.common import load_nostim
from neurobranch.diagnostics.observation import CausalCalciumLDS

OUT = PROJECT_ROOT / "reports/simulator/adaptive_dynamics"
OUT.mkdir(parents=True, exist_ok=True)
HORIZONS = (1, 5, 10, 20, 50, 100)


def synthetic_calcium(
    *, n_frames: int = 2400, latent_dim: int = 4, n_neurons: int = 24, seed: int = 23
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = np.diag(rng.uniform(0.75, 0.92, size=latent_dim))
    D = rng.normal(size=(latent_dim, n_neurons)) / np.sqrt(latent_dim)
    z = rng.normal(size=latent_dim)
    c = np.zeros(n_neurons)
    out = np.empty((n_frames, n_neurons), dtype=np.float64)
    alpha = 0.3
    for t in range(n_frames):
        z = A @ z + rng.normal(scale=0.1, size=latent_dim)
        drive = D.T @ z
        c = (1 - alpha) * c + alpha * drive
        out[t] = c + rng.normal(scale=0.05, size=n_neurons)
    return out


def evaluate_calcium_case(
    X: np.ndarray,
    *,
    label: str,
    train_mask: np.ndarray,
    landmarks: np.ndarray,
    history: int = 32,
) -> tuple[list[dict], dict]:
    train_starts = np.arange(2, int(np.flatnonzero(train_mask)[-1]) - 1)
    model = CausalCalciumLDS.fit(X, train_mask, train_starts)
    corrected = CorrectedCalciumForecast(model)
    rows = []
    for t in landmarks:
        hist = X[int(t) - history + 1 : int(t) + 1]
        x0 = X[int(t)].copy()
        truth = X[int(t) + 1 : int(t) + 101]
        with np.errstate(over="ignore", invalid="ignore"):
            old = model.rollout(x0, 100, initial_history=hist)
            new = corrected.forecast(hist, 100)
        for h in HORIZONS:
            y = truth[h - 1]
            for name, pred in (("old_calcium", old), ("corrected_calcium", new), ("persistence", np.repeat(x0[None], 100, 0))):
                p = pred[h - 1]
                mse = float(np.mean((p - y) ** 2)) if np.isfinite(p).all() else float("inf")
                rows.append(
                    {
                        "case": label,
                        "landmark": int(t),
                        "model": name,
                        "horizon": int(h),
                        "mse": mse,
                        "finite": bool(np.isfinite(p).all()),
                        "max_abs": float(np.max(np.abs(p))) if np.isfinite(p).all() else float("inf"),
                    }
                )
    diag = diagnose_fit(model, type("D", (), {"X": X})(), history=history)
    diag["case"] = label
    diag["n_landmarks"] = int(len(landmarks))
    return rows, diag


def main() -> int:
    _cfg, data, _recs = load_nostim()
    all_rows, diagnostics = [], []
    for d in data:
        landmarks = evaluation_landmarks(d, horizon_max=100, stride=10)
        rows, diag = evaluate_calcium_case(
            d.X, label=f"kato_nostim_{int(d.worm_id)}", train_mask=d.masks["train"], landmarks=landmarks
        )
        for row in rows:
            row["worm_id"] = int(d.worm_id)
        all_rows.extend(rows)
        diag["worm_id"] = int(d.worm_id)
        diagnostics.append(diag)

    synthetic = synthetic_calcium()
    syn_mask = np.zeros(len(synthetic), dtype=bool)
    syn_mask[: int(0.7 * len(synthetic))] = True
    syn_landmarks = np.arange(int(0.7 * len(synthetic)), len(synthetic) - 100, 10)
    rows, diag = evaluate_calcium_case(
        synthetic, label="synthetic_stable_calcium", train_mask=syn_mask, landmarks=syn_landmarks
    )
    all_rows.extend(rows)
    diagnostics.append(diag)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "calcium_forecasts.csv", index=False)
    agg = (
        df.groupby(["case", "model", "horizon"], as_index=False)
        .agg(
            mean_mse=("mse", "mean"),
            median_mse=("mse", "median"),
            finite_fraction=("finite", "mean"),
            max_abs=("max_abs", "max"),
        )
    )
    agg.to_csv(OUT / "calcium_summary.csv", index=False)
    h20 = df[df.horizon == 20]
    paired = []
    for case, group in h20.groupby("case"):
        pivot = group.pivot_table(index="landmark", columns="model", values="mse")
        for model in ("old_calcium", "corrected_calcium"):
            if model not in pivot or "persistence" not in pivot:
                continue
            diff = (pivot["persistence"] - pivot[model]).replace([np.inf, -np.inf], np.nan).dropna()
            mean, lo, hi = block_bootstrap_ci(diff.to_numpy(), block=5)
            paired.append(
                {
                    "case": case,
                    "model": model,
                    "n": len(diff),
                    "mean_improvement_vs_persistence": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    pd.DataFrame(paired).to_csv(OUT / "calcium_paired.csv", index=False)
    payload = {
        "stage": "B_calcium_stability",
        "version": "causal_calcium_lds_corrected_v1",
        "diagnostics": diagnostics,
        "aggregate": agg.to_dict(orient="records"),
        "paired": paired,
        "git": git_state(),
    }
    (OUT / "calcium_diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(agg.to_string(index=False))
    print(pd.DataFrame(diagnostics)[["case", "alpha", "spectral_radius", "old_finite_fraction", "new_finite_fraction", "old_max_abs", "new_max_abs"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
