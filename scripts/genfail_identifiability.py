"""GENFAIL Stage D: dimensionality and latent/operator identifiability."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from neurobranch.config import PROJECT_ROOT, git_state
from neurobranch.diagnostics.common import load_nostim
from neurobranch.diagnostics.identifiability import evaluate, fit_ar1
from neurobranch.preprocessing import split_starts

OUT = PROJECT_ROOT / "reports/simulator/failure_diagnostics/identifiability"
CONTEXT = PROJECT_ROOT / "reports/simulator/context_dynamics"


def _seed_operator_stability(data, *, latent_dim: int = 15, seed: int = 17) -> list[dict]:
    starts = split_starts(data, "train", history=1, horizon=1)
    rng = np.random.default_rng(seed)
    rows = []
    base = fit_ar1(data, latent_dim=latent_dim, starts=starts)
    base_eig = np.sort(np.abs(np.linalg.eigvals(base.A)))[::-1]
    for b in range(3):
        pick = np.sort(rng.choice(starts, size=int(0.8 * len(starts)), replace=False))
        model = fit_ar1(data, latent_dim=latent_dim, starts=pick)
        eig = np.sort(np.abs(np.linalg.eigvals(model.A)))[::-1]
        rows.append(
            {
                "worm_id": int(data.worm_id),
                "seed": b,
                "n_starts": int(len(pick)),
                "eigenvalue_magnitude_mae": float(np.mean(np.abs(eig - base_eig))),
                "spectral_radius_difference": float(
                    abs(np.max(eig) - np.max(base_eig))
                ),
                "sorted_magnitudes": json.dumps([float(x) for x in eig]),
            }
        )
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, datasets, _ = load_nostim()
    results = []
    spectrum_rows = []
    seed_rows = []
    for data in datasets:
        print(f"identifiability worm {data.worm_id}", flush=True)
        result = evaluate(data, latent_dim=15)
        results.append(result)
        pca = PCA().fit(data.X[data.masks["train"]])
        for i, (ev, ratio) in enumerate(
            zip(pca.explained_variance_, pca.explained_variance_ratio_), start=1
        ):
            spectrum_rows.append(
                {
                    "worm_id": int(data.worm_id),
                    "component": i,
                    "explained_variance": float(ev),
                    "explained_variance_ratio": float(ratio),
                    "cumulative_ratio": float(np.cumsum(pca.explained_variance_ratio_)[i - 1]),
                }
            )
        seed_rows.extend(_seed_operator_stability(data))
    (OUT / "identifiability_summary.json").write_text(
        json.dumps(
            {
                "stage": "D_identifiability",
                "evidence_tier": "exploratory",
                "rotation_ambiguity": "subspace and eigenvalue diagnostics only; latent coordinates are not anatomical",
                "n_sessions": len(datasets),
                "results": results,
                "git": git_state(),
            },
            indent=2,
            default=float,
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(spectrum_rows).to_csv(OUT / "pca_spectrum.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(OUT / "operator_seed_stability.csv", index=False)

    rows = []
    for r in results:
        rows.append(
            {
                "worm_id": r["worm_id"],
                **r["pca"],
                **{f"subspace_{k}": v for k, v in r["subspace"]["half_subspace"].items()},
                **{f"operator_{k}": v for k, v in r["operator"].items() if not isinstance(v, list)},
                **r["reconstruction"],
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "identifiability_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
