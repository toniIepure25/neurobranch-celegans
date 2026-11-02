"""Stage C: innovation residual diagnostics and innovation-model choice."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, git_state
from neurobranch.diagnostics.common import load_nostim
from neurobranch.probabilistic.baselines import BaselineConfig
from neurobranch.probabilistic.residual import (
    adaptive_prequential_residuals,
    covariance_sensitivity,
    frozen_residuals,
    independent_ar_residuals,
    local_mean_residuals,
    persistence_residuals,
    regularized_covariance,
    residual_diagnostics,
)

OUT = PROJECT_ROOT / "reports/simulator/probabilistic_adaptive"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    _cfg, data, _recs = load_nostim()
    config = BaselineConfig(local_window=100, ewma_halflife=100.0, ar_window=200)
    diag_rows, cov_rows, acf_rows, sensitivity_rows = [], [], [], []
    for d in data:
        train_end = int(np.flatnonzero(d.masks["train"])[-1])
        series = {
            "adaptive": adaptive_prequential_residuals(d, window=500),
            "frozen": frozen_residuals(d),
            "independent_ar": independent_ar_residuals(d, window=config.ar_window),
            "local_mean": local_mean_residuals(d, window=config.local_window),
            "persistence": persistence_residuals(d),
        }
        for name, item in series.items():
            for split, mask in (
                ("train", item.targets <= train_end),
                ("test", item.targets > train_end),
            ):
                subset_values = item.values[mask]
                if len(subset_values) < 10:
                    continue
                subset = type(item)(
                    item.model,
                    subset_values,
                    item.targets[mask],
                    item.fit_times[mask],
                    item.lead[mask],
                    item.states[mask],
                )
                diag = residual_diagnostics(subset)
                diag_rows.append({"worm_id": int(d.worm_id), "model": name, "split": split, **diag})
                if split == "train":
                    cov, meta = regularized_covariance(subset_values)
                    eig = np.linalg.eigvalsh(cov)
                    cov_rows.append(
                        {
                            "worm_id": int(d.worm_id),
                            "model": name,
                            **meta,
                            "trace": float(np.trace(cov)),
                            "condition_number": float(
                                eig.max() / max(eig[eig > 1e-12].min(), 1e-12)
                            ),
                            "offdiag_fraction": float(
                                np.linalg.norm(cov - np.diag(np.diag(cov)))
                                / max(np.linalg.norm(cov), 1e-12)
                            ),
                        }
                    )
                    centered = subset_values - subset_values.mean(axis=0)
                    for lag in (1, 2, 5, 10):
                        vals = []
                        for j in range(centered.shape[1]):
                            if np.std(centered[:, j]) > 1e-10:
                                vals.append(
                                    float(
                                        np.corrcoef(centered[:-lag, j], centered[lag:, j])[0, 1]
                                    )
                                )
                        acf_rows.append(
                            {
                                "worm_id": int(d.worm_id),
                                "model": name,
                                "lag": lag,
                                "median_abs_acf": float(np.nanmedian(np.abs(vals)))
                                if vals
                                else float("nan"),
                            }
                        )
                    sens = covariance_sensitivity(subset_values)
                    sens["worm_id"] = int(d.worm_id)
                    sens["model"] = name
                    sensitivity_rows.append(sens)
    diagnostics = pd.DataFrame(diag_rows)
    diagnostics.to_csv(OUT / "residual_diagnostics.csv", index=False)
    covariance = pd.DataFrame(cov_rows)
    covariance.to_csv(OUT / "residual_covariance.csv", index=False)
    acf = pd.DataFrame(acf_rows)
    acf.to_csv(OUT / "residual_autocorrelation.csv", index=False)
    sensitivity = pd.concat(sensitivity_rows, ignore_index=True)
    sensitivity.to_csv(OUT / "residual_covariance_sensitivity.csv", index=False)

    train = diagnostics[(diagnostics.model == "adaptive") & (diagnostics.split == "train")]
    train_acf = acf[(acf.model == "adaptive") & (acf.lag == 1)]
    median_acf = float(train_acf.median_abs_acf.median())
    primary = "P6_adaptive_block" if median_acf > 0.1 else "P5_adaptive_boot"
    choice = {
        "decision_rule": "block bootstrap if median absolute lag-1 residual autocorrelation > 0.1, otherwise iid bootstrap",
        "train_median_abs_lag1_acf": median_acf,
        "primary_model": primary,
        "n_train_residuals": int(train.n.sum()),
        "evidence": "training-split residual diagnostics only; no test residuals used for this choice",
        "git": git_state(),
    }
    (OUT / "innovation_choice.json").write_text(json.dumps(choice, indent=2), encoding="utf-8")

    summary = (
        diagnostics.groupby(["model", "split"], as_index=False)
        .agg(
            mean_acf_lag1=("acf_abs_lag1", "mean"),
            mean_abs_cross_corr=("mean_abs_cross_corr", "mean"),
            mean_kurtosis=("mean_kurtosis", "mean"),
            mean_abs_skew=("mean_abs_skew", "mean"),
            mean_effective_rank=("effective_rank", "mean"),
            mean_cov_drift=("covariance_drift_relative", "mean"),
            mean_predictability_r2=("predictability_r2", "mean"),
            mean_state_corr=("state_norm_correlation", "mean"),
        )
    )
    summary.to_csv(OUT / "residual_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\ninnovation choice:", choice)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
