"""Audit utilities for virtual-perturbation candidate specificity.

Controls are matched on the intervention design (same amplitude, initial-state
distribution, simulation seeds, horizon, and target behaviour), not on the
observed effect.  Matching on the outcome would make the comparison circular.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def candidate_specificity(
    scan: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    target: str = "TURN",
) -> pd.DataFrame:
    """Attach empirical control p-values and percentiles to each candidate row."""
    if target not in scan.columns and "delta_prob" not in scan.columns:
        raise ValueError("scan must contain delta_prob or a target-specific delta column")
    d = scan.copy()
    c = controls.copy()
    cvals = np.abs(c["delta_prob"].to_numpy(float))
    rows = []
    for _, r in d.iterrows():
        val = float(r["delta_prob"])
        p = float((1 + np.sum(cvals >= abs(val))) / (1 + len(cvals)))
        rows.append(
            {
                "candidate": r["candidate"],
                "amplitude": float(r["amplitude"]),
                "delta": val,
                "abs_delta": abs(val),
                "ci_low": float(r["ci_low"]),
                "ci_high": float(r["ci_high"]),
                "ood_fraction": float(r["ood_fraction"]) if pd.notna(r.get("ood_fraction")) else np.nan,
                "control_n": int(len(cvals)),
                "control_mean_abs_delta": float(cvals.mean()),
                "control_max_abs_delta": float(cvals.max()),
                "empirical_p_two_sided": p,
                "percentile_vs_controls": float(np.mean(cvals < abs(val))),
                "specific": bool(p < 0.05),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_delta", ascending=False).reset_index(drop=True)


def selection_adjusted_max_test(
    scan: pd.DataFrame,
    controls: pd.DataFrame,
) -> dict[str, Any]:
    """A conservative max-statistic test for the best candidate.

    The candidate set was chosen from prior literature, but the *winner* is
    selected from 26 candidate/amplitude cells.  The null distribution here is
    the observed set of non-candidate control effects; the p-value is the
    fraction of controls whose absolute effect equals or exceeds the best
    candidate.  With 19 controls the resolution is 1/20.
    """
    best = float(scan["abs_delta"].max())
    cvals = np.abs(controls["delta_prob"].to_numpy(float))
    return {
        "best_candidate_abs_delta": best,
        "control_max_abs_delta": float(cvals.max()),
        "control_mean_abs_delta": float(cvals.mean()),
        "p_max_control_ge_best": float((1 + np.sum(cvals >= best)) / (1 + len(cvals))),
        "n_controls": int(len(cvals)),
        "n_candidate_cells": int(len(scan)),
        "resolution": float(1 / (len(cvals) + 1)),
        "interpretation": "not specific if this p is not small; controls were selected without using effects",
    }


def rank_stability(
    scans: dict[str, pd.DataFrame],
    *,
    amplitude: float = 2.0,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compare candidate rankings across models/seeds by Spearman and top-k overlap."""
    frames = {}
    for name, d in scans.items():
        sub = d[np.isclose(d["amplitude"], amplitude)].set_index("candidate")["delta_prob"]
        frames[name] = sub
    mat = pd.DataFrame(frames)
    common = mat.dropna(how="any")
    if len(common) < 3 or len(mat.columns) < 2:
        return {"n_common": int(len(common)), "models": list(mat.columns)}
    rho = spearmanr(common.to_numpy()).statistic
    if np.ndim(rho) == 0:
        rho = np.array([[1.0, float(rho)], [float(rho), 1.0]])
    tops = {m: set(common[m].abs().nlargest(top_k).index) for m in common}
    pairs = []
    cols = list(common.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            pairs.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "spearman": float(rho[i, j]),
                    "top_k_overlap": len(tops[a] & tops[b]),
                    "top_k": top_k,
                }
            )
    return {
        "n_common": int(len(common)),
        "models": cols,
        "candidates": common.index.tolist(),
        "spearman_matrix": rho.tolist(),
        "pairwise": pairs,
        "top_sets": {k: sorted(v) for k, v in tops.items()},
    }


def amplitude_sensitivity(
    results: dict[tuple[str, float], float],
) -> pd.DataFrame:
    """Summarise effect vs amplitude for one or more cells."""
    rows = []
    for (cell, amp), delta in results.items():
        rows.append({"cell": cell, "amplitude": float(amp), "delta": float(delta)})
    d = pd.DataFrame(rows).sort_values(["cell", "amplitude"])
    if len(d):
        d["abs_delta"] = d["delta"].abs()
    return d
