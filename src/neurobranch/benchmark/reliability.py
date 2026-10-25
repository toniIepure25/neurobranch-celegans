"""Measurement-error ceiling for target response profiles.

A pair-level Spearman of 0.09 can be near the ceiling if the underlying profile
is only modestly reliable.  This module estimates that ceiling directly rather
than assuming the observed response is error-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from neurobranch.benchmark.metrics import spearman


@dataclass(frozen=True)
class ReliabilitySummary:
    n_targets: int
    n_splits: int
    mean_half_reliability: float
    median_half_reliability: float
    mean_full_reliability: float
    median_full_reliability: float
    mean_attenuation_ceiling: float

    def as_dict(self) -> dict[str, float]:
        return {
            "n_targets": float(self.n_targets),
            "n_splits": float(self.n_splits),
            "mean_half_reliability": self.mean_half_reliability,
            "median_half_reliability": self.median_half_reliability,
            "mean_full_reliability": self.mean_full_reliability,
            "median_full_reliability": self.median_full_reliability,
            "mean_attenuation_ceiling": self.mean_attenuation_ceiling,
        }


def split_half_target_profiles(
    trials: pd.DataFrame,
    *,
    response_column: str = "response_mean",
    n_splits: int = 20,
    min_events: int = 4,
    min_responders: int = 3,
    seed: int = 17,
) -> tuple[ReliabilitySummary, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    work = trials[["target", "neuron", "event_id", response_column]].dropna().copy()
    work = work.rename(columns={response_column: "y"})
    pair_groups = {key: g for key, g in work.groupby(["target", "neuron"])}
    eligible = {key: g for key, g in pair_groups.items() if g["event_id"].nunique() >= min_events}
    per_split: list[pd.DataFrame] = []
    for split in range(n_splits):
        rows = []
        for (target, neuron), group in eligible.items():
            events = group["event_id"].drop_duplicates().to_numpy()
            rng.shuffle(events)
            half = len(events) // 2
            a_events = set(events[:half].tolist())
            b_events = set(events[half:].tolist())
            a = group[group["event_id"].isin(a_events)]["y"].mean()
            b = group[group["event_id"].isin(b_events)]["y"].mean()
            rows.append({"target": target, "neuron": neuron, "a": a, "b": b})
        profiles = pd.DataFrame(rows)
        if profiles.empty:
            continue
        for target, group in profiles.groupby("target"):
            if len(group) < min_responders:
                continue
            rho = spearman(group["a"], group["b"])
            if np.isfinite(rho):
                # A negative split-half correlation means the profile is not
                # reliably recoverable; Spearman-Brown is undefined/unstable
                # there, so the attenuation ceiling is set to zero rather than
                # letting r -> -1 explode.
                full = 2.0 * rho / (1.0 + rho) if rho > 0 else 0.0
                per_split.append(
                    pd.DataFrame(
                        {
                            "split": [split],
                            "target": [target],
                            "n_responders": [len(group)],
                            "half_reliability": [rho],
                            "full_reliability": [full],
                            "attenuation_ceiling": [np.sqrt(max(full, 0.0))],
                        }
                    )
                )
    detail = pd.concat(per_split, ignore_index=True) if per_split else pd.DataFrame()
    if detail.empty:
        return ReliabilitySummary(0, 0, np.nan, np.nan, np.nan, np.nan, np.nan), detail
    summary = ReliabilitySummary(
        n_targets=int(detail["target"].nunique()),
        n_splits=int(detail["split"].nunique()),
        mean_half_reliability=float(detail["half_reliability"].mean()),
        median_half_reliability=float(detail["half_reliability"].median()),
        mean_full_reliability=float(detail["full_reliability"].mean()),
        median_full_reliability=float(detail["full_reliability"].median()),
        mean_attenuation_ceiling=float(detail["attenuation_ceiling"].mean()),
    )
    return summary, detail


def model_profile_correlation(predictions: pd.DataFrame, *, model: str = "cro") -> pd.DataFrame:
    """Per-target Spearman between predicted and observed pair means."""
    frame = predictions[predictions["model"] == model]
    pairs = (
        frame.groupby(["target", "neuron"])
        .agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"), n_events=("y_true", "size"))
        .reset_index()
    )
    pairs = pairs[pairs["n_events"] >= 2]
    rows = []
    for target, group in pairs.groupby("target"):
        if len(group) < 3 or group["y_true"].nunique() < 2 or group["y_pred"].nunique() < 2:
            continue
        rows.append(
            {
                "target": target,
                "n_responders": len(group),
                "profile_spearman": spearman(group["y_true"], group["y_pred"]),
                "profile_pearson": float(np.corrcoef(group["y_true"], group["y_pred"])[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def attenuation_corrected_profile(
    predictions: pd.DataFrame,
    reliability_detail: pd.DataFrame,
    *,
    model: str = "cro",
) -> pd.DataFrame:
    profile = model_profile_correlation(predictions, model=model)
    if profile.empty:
        return profile
    ceiling = reliability_detail.groupby("target")["attenuation_ceiling"].mean().rename("attenuation_ceiling")
    out = profile.merge(ceiling, on="target", how="left")
    valid = out["attenuation_ceiling"] > 1e-6
    out["normalized_profile_spearman"] = np.nan
    out.loc[valid, "normalized_profile_spearman"] = (
        out.loc[valid, "profile_spearman"] / out.loc[valid, "attenuation_ceiling"]
    )
    return out
