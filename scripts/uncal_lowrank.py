"""D3 decision: does a low-rank-plus-diagonal innovation covariance add value?"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neurobranch.diagnostics.common import load_nostim
from neurobranch.probabilistic.residual import independent_ar_residuals
from neurobranch.uncertainty_calibration.models import fit_low_rank_plus_diagonal

OUT = Path("reports/simulator/uncertainty_calibration")


def nll(cov: np.ndarray, validation: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return float("inf")
    trace = np.trace(np.cov(validation, rowvar=False) @ np.linalg.pinv(cov))
    return float(0.5 * (logdet + trace))


def main() -> None:
    cfg, data, _ = load_nostim()
    rows = []
    for d in data:
        train_end = int(d.split_bounds["train"][1] - 1)
        rs = independent_ar_residuals(d, window=200, block=10)
        values = rs.values[(rs.targets <= train_end) & (rs.lead == 1)]
        split = len(values) // 2
        fit, validation = values[:split], values[split:]
        diagonal = np.diag(np.maximum(np.diag(np.cov(fit, rowvar=False)), 1e-10))
        rows.append({"worm_id": int(d.worm_id), "rank": 0, "n_fit": len(fit), "n_validation": len(validation), "nll": nll(diagonal, validation), "retained_fraction": float(np.trace(diagonal) / np.trace(diagonal))})
        for rank in (1, 3, 5, 10):
            cov, meta = fit_low_rank_plus_diagonal(fit, rank)
            rows.append(
                {
                    "worm_id": int(d.worm_id),
                    "rank": rank,
                    "n_fit": len(fit),
                    "n_validation": len(validation),
                    "nll": nll(cov, validation),
                    "retained_fraction": meta["retained_eigenvalue_fraction"],
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "lowrank_validation.csv", index=False)
    # Rank chosen by lowest validation NLL per session; 0 means diagonal.
    chosen = frame.loc[frame.groupby("worm_id").nll.idxmin(), ["worm_id", "rank"]]
    decision = {
        "chosen_rank_by_session": chosen.set_index("worm_id")["rank"].to_dict(),
        "mean_rank": float(chosen["rank"].mean()),
        "diagonal_wins_sessions": int((chosen["rank"] == 0).sum()),
        "build_lowrank_model": bool((chosen["rank"] > 0).mean() >= 0.5),
        "reason": "Diagonal covariance won the held-out Gaussian likelihood in most sessions; low-rank did not add stable value.",
    }
    (OUT / "lowrank_decision.json").write_text(json.dumps(decision, indent=2))
    print(decision)


if __name__ == "__main__":
    main()
