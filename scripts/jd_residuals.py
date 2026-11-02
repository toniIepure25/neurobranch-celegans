"""Stage C: audit true one-step innovation structure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neurobranch.diagnostics.common import load_nostim
from neurobranch.joint_dynamics.models import LowRankTemporalInnovation
from neurobranch.joint_dynamics.residuals import (
    acf_profile,
    conditional_residual_r2,
    lead1_innovations,
    residual_structure,
)
from neurobranch.probabilistic.residual import (
    adaptive_prequential_residuals,
    independent_ar_residuals,
)

OUT = Path("reports/simulator/joint_dynamics")
MAX_HISTORY = 1000


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _, data, _ = load_nostim()
    rows, acf_rows, factor_rows = [], [], []
    for d in data:
        sid = int(d.worm_id)
        series = {
            "independent_ar": independent_ar_residuals(d, window=200, block=10),
            "population_var": adaptive_prequential_residuals(d, window=500, block=10),
        }
        for name, values in series.items():
            info = residual_structure(values, max_history=MAX_HISTORY)
            sample = lead1_innovations(values, max_history=MAX_HISTORY)
            info["conditional_mean_r2"] = conditional_residual_r2(sample.values, sample.states)
            model = LowRankTemporalInnovation.fit(sample.values, rank=8)
            info.update(
                {
                    "model": name,
                    "worm_id": sid,
                    "factor_phi_median": model.metadata["phi_median"],
                    "factor_phi_abs_max": model.metadata["phi_abs_max"],
                    "factor_captured_variance": model.metadata["captured_variance"],
                    "factor_rank": model.metadata["rank"],
                }
            )
            rows.append(info)
            profile = acf_profile(sample.values, max_lag=20)
            profile["model"] = name
            profile["worm_id"] = sid
            acf_rows.append(profile)
            for k, (phi, cap) in enumerate(zip(model.phi, model.metadata["captured_variance"] * np.ones_like(model.phi))):
                factor_rows.append(
                    {
                        "model": name,
                        "worm_id": sid,
                        "factor": k,
                        "phi": float(phi),
                        "eta_scale": float(model.eta_scale[k]),
                        "captured_variance": float(cap),
                    }
                )
    structure = pd.DataFrame(rows)
    structure.to_csv(OUT / "innovation_structure.csv", index=False)
    pd.concat(acf_rows, ignore_index=True).to_csv(OUT / "innovation_acf.csv", index=False)
    pd.DataFrame(factor_rows).to_csv(OUT / "innovation_factors.csv", index=False)
    median = structure.groupby("model")[["acf1", "acf5", "temporal_cross_cov_lag1"]].median()
    justified = bool((median.loc["population_var", "acf1"] >= 0.15) and (median.loc["population_var", "acf5"] >= 0.08))
    decision = {
        "rule": "temporal innovation model justified only if median one-step population ACF1 >= 0.15 and ACF5 >= 0.08",
        "median_by_model": median.to_dict(orient="index"),
        "temporal_model_justified": justified,
        "chosen_rank": 8,
        "max_history": MAX_HISTORY,
        "note": "Only lead==1 prequential residuals enter this analysis; multi-step errors are excluded.",
    }
    # Reconstruction check on a held-out half of the residual sample.
    recon = []
    for d in data:
        sid = int(d.worm_id)
        sample = lead1_innovations(adaptive_prequential_residuals(d, window=500, block=10), max_history=MAX_HISTORY)
        n = len(sample.values)
        cut = int(n * 0.7)
        model = LowRankTemporalInnovation.fit(sample.values[:cut], rank=8)
        recon.append(
            {
                "worm_id": sid,
                "n": int(n),
                "heldout_factor_variance": float(
                    np.var(sample.values[cut:] @ model.factors) / max(np.var(sample.values[cut:]), 1e-12)
                ),
            }
        )
    pd.DataFrame(recon).to_csv(OUT / "innovation_reconstruction.csv", index=False)
    (OUT / "innovation_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
