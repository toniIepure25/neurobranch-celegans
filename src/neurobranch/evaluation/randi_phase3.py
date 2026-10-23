"""Phase 3 reanalysis of the Randi optogenetic atlas.

This module asks whether *better targets, repeated-event averaging, stimulus
context or temporal shape* improve prediction of an observed intervention
response.  It does not claim that an observed response generalises to a new
intervention, and it keeps the Phase 2 warning:

    MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from neurobranch.data.randi import (
    ResponseConfig,
    build_trial_table,
    discover_datasets,
    load_recording,
)
from neurobranch.evaluation.randi_benchmark import (
    _cluster_bootstrap,
    _metrics,
    response_reliability,
)

TARGETS = ("response", "response_mean", "response_integral")


def _clean(table: pd.DataFrame, target: str) -> pd.DataFrame:
    d = table.dropna(subset=[target, "target", "neuron", "pre_activity"]).copy()
    d["y"] = d[target].astype(float)
    d["event_id"] = d["event_id"].astype(str)
    d["target"] = d["target"].astype(str)
    d["neuron"] = d["neuron"].astype(str)
    d["dataset_id"] = d["dataset_id"].astype(int)
    return d


def _prediction_summary(
    task: str,
    model: str,
    y: np.ndarray,
    yhat: np.ndarray,
    groups: np.ndarray,
    *,
    n_bootstrap: int = 500,
    seed: int = 17,
) -> dict[str, Any]:
    m = _metrics(y, yhat)
    ci = _cluster_bootstrap(y, yhat, groups, n_bootstrap=n_bootstrap, seed=seed)
    return {"task": task, "model": model, **m, **ci}


class _ContextRidge:
    """One-hot target/neuron plus prespecified pre-stimulus context."""

    def __init__(self, alpha: float = 10.0, use_context: bool = True):
        self.alpha = alpha
        self.use_context = use_context
        self.encoder: OneHotEncoder | None = None
        self.scaler: StandardScaler | None = None
        self.median: np.ndarray | None = None
        self.model: Ridge | None = None

    def _numeric(self, d: pd.DataFrame) -> np.ndarray:
        cols = ["pre_activity"]
        if self.use_context:
            # Deliberately exclude absolute event_order/session index: it can act
            # as an animal identifier and inflate held-out-event performance.
            cols += ["pre_sd", "prev_stim_interval", "prev_same_target"]
        x = d[cols].to_numpy(float)
        if self.median is not None:
            bad = ~np.isfinite(x)
            x[bad] = np.take(self.median, np.where(bad)[1])
        return x

    def fit(self, d: pd.DataFrame, y: np.ndarray) -> _ContextRidge:
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        cat = self.encoder.fit_transform(d[["target", "neuron"]])
        num = self._numeric(d)
        self.median = np.nanmedian(num, axis=0)
        bad = ~np.isfinite(num)
        num[bad] = np.take(self.median, np.where(bad)[1])
        self.scaler = StandardScaler().fit(num)
        X = sparse.hstack([cat, sparse.csr_matrix(self.scaler.transform(num))]).tocsr()
        self.model = Ridge(alpha=self.alpha, solver="lsqr")
        self.model.fit(X, y)
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        assert self.encoder is not None and self.scaler is not None and self.model is not None
        cat = self.encoder.transform(d[["target", "neuron"]])
        num = self.scaler.transform(self._numeric(d))
        X = sparse.hstack([cat, sparse.csr_matrix(num)]).tocsr()
        return np.asarray(self.model.predict(X))


def _mean_map(d: pd.DataFrame, keys: list[str], y: np.ndarray) -> dict[tuple, float]:
    g = d.assign(_y=y).groupby(keys)["_y"].mean()
    return {tuple(k) if isinstance(k, tuple) else (k,): float(v) for k, v in g.items()}


def _repeated_average_predictions(
    train: pd.DataFrame, test: pd.DataFrame, target: str
) -> dict[str, np.ndarray]:
    y = train["y"].to_numpy(float)
    global_mean = float(np.mean(y))
    target_map = _mean_map(train, ["target"], y)
    animal_target = _mean_map(train, ["dataset_id", "target"], y)
    pair = _mean_map(train, ["dataset_id", "target", "neuron"], y)
    animal = _mean_map(train, ["dataset_id"], y)

    def fallback(row, mapping, fallback_value):
        return mapping.get(row, fallback_value)

    pred = {"global": np.full(len(test), global_mean), "target": [], "animal": [], "animal_target": [], "pair": []}
    for _, row in test.iterrows():
        t = (row["target"],)
        at = (int(row["dataset_id"]), row["target"])
        p = (int(row["dataset_id"]), row["target"], row["neuron"])
        a = (int(row["dataset_id"]),)
        pred["target"].append(target_map.get(t, global_mean))
        pred["animal"].append(animal.get(a, global_mean))
        base = animal_target.get(at, target_map.get(t, global_mean))
        pred["animal_target"].append(base)
        pred["pair"].append(pair.get(p, base))
    for k in ["target", "animal", "animal_target", "pair"]:
        pred[k] = np.asarray(pred[k], dtype=float)
    return pred


def evaluate_repeated_average(
    table: pd.DataFrame, target: str, *, seed: int = 17, n_bootstrap: int = 500
) -> pd.DataFrame:
    """Event-held-out test of repeated-event averaging versus nested means."""
    d = _clean(table, target)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(splitter.split(d, groups=d["event_id"]))
    train, test = d.iloc[tr], d.iloc[te]
    preds = _repeated_average_predictions(train, test, target)
    rows = []
    for model, yhat in preds.items():
        rows.append(
            _prediction_summary(
                "heldout_events_repeated_average",
                model,
                test["y"].to_numpy(),
                yhat,
                test["event_id"].to_numpy(),
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        )
    return pd.DataFrame(rows)


def evaluate_context(
    table: pd.DataFrame, target: str, *, seed: int = 17, n_bootstrap: int = 500
) -> pd.DataFrame:
    """Event-held-out test of pre-stimulus context, with identical splits."""
    d = _clean(table, target)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(splitter.split(d, groups=d["event_id"]))
    train, test = d.iloc[tr], d.iloc[te]
    rows = []
    for use_context, label in [(False, "onehot_pre"), (True, "onehot_pre_context")]:
        model = _ContextRidge(use_context=use_context).fit(train, train["y"].to_numpy())
        rows.append(
            _prediction_summary(
                "heldout_events_context",
                label,
                test["y"].to_numpy(),
                model.predict(test),
                test["event_id"].to_numpy(),
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        )
    return pd.DataFrame(rows)


def temporal_profile_reliability(data_dir: str | Path, *, min_common: int = 3) -> dict[str, Any]:
    """Pairwise repeatability of the full post-stimulus temporal shape.

    For each animal/target with repeated stimulations, compare per-neuron
    12-frame profiles for neurons recorded in both events.  Also compares the
    population-average time course and the amplitude vector.
    """
    cfg = ResponseConfig()
    profile_corrs: list[float] = []
    population_corrs: list[float] = []
    amplitude_corrs: list[float] = []
    sign_agreement: list[float] = []
    groups = 0
    pairs = 0
    for ds in discover_datasets(data_dir):
        rec = load_recording(data_dir, ds)
        by_target: dict[str, list[dict[str, Any]]] = {}
        named = [
            i
            for i, n in enumerate(rec.labels)
            if n is not None and not (i < len(rec.ambiguous) and rec.ambiguous[i])
        ]
        for i0, target_idx, target in zip(rec.stim_frames, rec.stim_trace_indices, rec.stimulated_labels):
            if target is None or target_idx < 0:
                continue
            if i0 < cfg.pre_frames or i0 + cfg.post_start + cfg.post_frames >= rec.n_frames:
                continue
            pre = rec.traces[i0 - cfg.pre_frames : i0]
            post = rec.traces[i0 + cfg.post_start : i0 + cfg.post_start + cfg.post_frames]
            if pre.size == 0 or post.size == 0:
                continue
            baseline = np.median(pre, axis=0)
            profile = post - baseline
            if not np.isfinite(profile).all():
                continue
            by_target.setdefault(target, []).append(
                {
                    "names": {rec.labels[j]: j for j in named if j != target_idx},
                    "profile": profile,
                }
            )
        for _target, events in by_target.items():
            if len(events) < 2:
                continue
            groups += 1
            for a in range(len(events)):
                for b in range(a + 1, len(events)):
                    common = sorted(set(events[a]["names"]) & set(events[b]["names"]))
                    if len(common) < min_common:
                        continue
                    ia = [events[a]["names"][n] for n in common]
                    ib = [events[b]["names"][n] for n in common]
                    pa = events[a]["profile"][:, ia]
                    pb = events[b]["profile"][:, ib]
                    # Per-neuron temporal shape.
                    for k in range(len(common)):
                        if np.std(pa[:, k]) > 0 and np.std(pb[:, k]) > 0:
                            profile_corrs.append(float(np.corrcoef(pa[:, k], pb[:, k])[0, 1]))
                    # Population mean time course.
                    ma, mb = pa.mean(axis=1), pb.mean(axis=1)
                    if np.std(ma) > 0 and np.std(mb) > 0:
                        population_corrs.append(float(np.corrcoef(ma, mb)[0, 1]))
                    # Amplitude vector across neurons.
                    va, vb = pa.mean(axis=0), pb.mean(axis=0)
                    if np.std(va) > 0 and np.std(vb) > 0:
                        amplitude_corrs.append(float(np.corrcoef(va, vb)[0, 1]))
                    sign_agreement.append(float(np.mean(np.sign(va) == np.sign(vb))))
                    pairs += 1
    def q(a):
        arr = np.asarray(a, dtype=float)
        if len(arr) == 0:
            return {"n": 0, "median": float("nan"), "q25": float("nan"), "q75": float("nan")}
        return {
            "n": int(len(arr)),
            "median": float(np.median(arr)),
            "q25": float(np.quantile(arr, 0.25)),
            "q75": float(np.quantile(arr, 0.75)),
            "fraction_positive": float(np.mean(arr > 0)),
        }

    return {
        "n_target_groups": int(groups),
        "n_event_pairs": int(pairs),
        "per_neuron_temporal_r": q(profile_corrs),
        "population_timecourse_r": q(population_corrs),
        "amplitude_vector_r": q(amplitude_corrs),
        "sign_agreement": q(sign_agreement),
        "window": {"pre_frames": cfg.pre_frames, "post_frames": cfg.post_frames, "post_start": cfg.post_start},
    }


def write_randi_phase3(
    *,
    data_dir: str | Path = "data/raw/randi2023/exported_data",
    out_dir: str | Path = "reports/phase3",
    n_bootstrap: int = 500,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = build_trial_table(data_dir)
    table.to_csv(out / "randi_trial_table_phase3.csv.gz", index=False, compression="gzip")
    summary: dict[str, Any] = {
        "n_rows": int(len(table)),
        "n_events": int(table["event_id"].nunique()),
        "n_animals": int(table["dataset_id"].nunique()),
        "n_targets": int(table["target"].nunique()),
        "n_neurons": int(table["neuron"].nunique()),
        "targets": {},
    }
    rel_rows, avg_rows, ctx_rows = [], [], []
    for target in TARGETS:
        rel = response_reliability(table, response_column=target)
        summary["targets"][target] = {"reliability": rel}
        rel_rows.append({"target": target, **rel})
        avg = evaluate_repeated_average(table, target, n_bootstrap=n_bootstrap)
        avg["target"] = target
        avg_rows.append(avg)
        ctx = evaluate_context(table, target, n_bootstrap=n_bootstrap)
        ctx["target"] = target
        ctx_rows.append(ctx)
        summary["targets"][target]["repeated_average"] = avg.to_dict(orient="records")
        summary["targets"][target]["context"] = ctx.to_dict(orient="records")
    pd.DataFrame(rel_rows).to_csv(out / "randi_reliability_by_target.csv", index=False)
    pd.concat(avg_rows, ignore_index=True).to_csv(out / "randi_repeated_average.csv", index=False)
    pd.concat(ctx_rows, ignore_index=True).to_csv(out / "randi_context_models.csv", index=False)
    profile = temporal_profile_reliability(data_dir)
    summary["temporal_profile"] = profile
    (out / "randi_phase3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
