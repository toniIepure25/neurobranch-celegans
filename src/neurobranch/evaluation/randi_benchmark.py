"""Benchmark experimental optogenetic responses from the Randi et al. atlas.

The benchmark predicts a downstream trace's event-triggered response from
information available before/at stimulation.  It is deliberately *within the
intervention protocol*: a model fitted on optogenetic responses is not evidence
that spontaneous activity identifies interventions.

Three honestly different generalization questions are evaluated:

* held-out stimulation events (same animals and targets);
* held-out stimulated neurons (new targets, same animals);
* held-out animals (new recordings).

All splits are group-disjoint, scalers and response means are fitted on training
rows only, and uncertainty is bootstrapped over stimulation events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class BenchmarkConfig:
    response_column: str = "response"
    alpha: float = 10.0
    n_bootstrap: int = 200
    test_fraction: float = 0.2
    seed: int = 0
    min_events_per_target: int = 2


def _clean(table: pd.DataFrame, response_column: str) -> pd.DataFrame:
    required = {"event_id", "dataset_id", "target", "neuron", response_column, "pre_activity"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"trial table missing columns: {sorted(missing)}")
    d = table.dropna(subset=[response_column, "target", "neuron", "pre_activity"]).copy()
    d["response"] = d[response_column].astype(float)
    d["event_id"] = d["event_id"].astype(str)
    d["target"] = d["target"].astype(str)
    d["neuron"] = d["neuron"].astype(str)
    d["dataset_id"] = d["dataset_id"].astype(int)
    d["pre_activity"] = d["pre_activity"].astype(float)
    return d


def _shrinkage_map(
    d: pd.DataFrame, key: str, target: np.ndarray, *, prior: float, k: float = 20.0
) -> tuple[dict[str, float], float]:
    global_mean = float(np.mean(target))
    stats = d.assign(_y=target).groupby(key)["_y"].agg(["sum", "count"])
    vals = (stats["sum"] + k * global_mean) / (stats["count"] + k)
    return {str(i): float(v) for i, v in vals.items()}, global_mean


class _MeanBaseline:
    def __init__(self, kind: str):
        self.kind = kind
        self.global_mean = 0.0
        self.by_neuron: dict[str, float] = {}
        self.by_target: dict[str, float] = {}

    def fit(self, d: pd.DataFrame, y: np.ndarray) -> _MeanBaseline:
        self.global_mean = float(np.mean(y))
        if self.kind in {"neuron_mean", "additive"}:
            self.by_neuron, _ = _shrinkage_map(d, "neuron", y, prior=self.global_mean)
        if self.kind in {"target_mean", "additive"}:
            self.by_target, _ = _shrinkage_map(d, "target", y, prior=self.global_mean)
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        if self.kind == "global_mean":
            return np.full(len(d), self.global_mean)
        if self.kind == "neuron_mean":
            return np.array([self.by_neuron.get(n, self.global_mean) for n in d["neuron"]])
        if self.kind == "target_mean":
            return np.array([self.by_target.get(n, self.global_mean) for n in d["target"]])
        # Additive shrunk effects, centred on the global mean.
        vals = []
        for n, t in zip(d["neuron"], d["target"]):
            vals.append(
                self.global_mean
                + (self.by_neuron.get(n, self.global_mean) - self.global_mean)
                + (self.by_target.get(t, self.global_mean) - self.global_mean)
            )
        return np.asarray(vals)


class _OneHotRidge:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.encoder: OneHotEncoder | None = None
        self.scaler: StandardScaler | None = None
        self.model: Ridge | None = None

    def fit(self, d: pd.DataFrame, y: np.ndarray) -> _OneHotRidge:
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        cat = self.encoder.fit_transform(d[["target", "neuron"]])
        num = d[["pre_activity"]].to_numpy(float)
        self.scaler = StandardScaler().fit(num)
        X = sparse.hstack([cat, sparse.csr_matrix(self.scaler.transform(num))]).tocsr()
        self.model = Ridge(alpha=self.alpha, solver="lsqr")
        self.model.fit(X, y)
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        assert self.encoder is not None and self.scaler is not None and self.model is not None
        cat = self.encoder.transform(d[["target", "neuron"]])
        num = self.scaler.transform(d[["pre_activity"]].to_numpy(float))
        X = sparse.hstack([cat, sparse.csr_matrix(num)]).tocsr()
        return np.asarray(self.model.predict(X))


class _ConnectomeRidge:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.scaler: StandardScaler | None = None
        self.model: Ridge | None = None

    @staticmethod
    def features(d: pd.DataFrame, connectome: pd.DataFrame) -> np.ndarray:
        feat = d[["target", "neuron"]].merge(
            connectome, on=["target", "neuron"], how="left"
        )
        chem = np.log1p(feat["chemical_synapses"].fillna(0).to_numpy(float))
        elec = np.log1p(feat["electrical_synapses"].fillna(0).to_numpy(float))
        connected = ((chem > 0) | (elec > 0)).astype(float)
        pre = d["pre_activity"].to_numpy(float)
        return np.column_stack([chem, elec, connected, pre])

    def fit(self, d: pd.DataFrame, y: np.ndarray, connectome: pd.DataFrame) -> _ConnectomeRidge:
        X = self.features(d, connectome)
        self.scaler = StandardScaler().fit(X)
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(self.scaler.transform(X), y)
        return self

    def predict(self, d: pd.DataFrame, connectome: pd.DataFrame) -> np.ndarray:
        assert self.scaler is not None and self.model is not None
        return np.asarray(self.model.predict(self.scaler.transform(self.features(d, connectome))))


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_name: str,
    cfg: BenchmarkConfig,
    connectome: pd.DataFrame | None,
) -> np.ndarray:
    y = train["response"].to_numpy(float)
    if model_name in {"zero", "global_mean", "neuron_mean", "target_mean", "additive"}:
        if model_name == "zero":
            return np.zeros(len(test))
        return _MeanBaseline(model_name).fit(train, y).predict(test)
    if model_name == "onehot_ridge":
        return _OneHotRidge(cfg.alpha).fit(train, y).predict(test)
    if model_name == "connectome_ridge":
        if connectome is None:
            raise ValueError("connectome table required for connectome_ridge")
        return _ConnectomeRidge(cfg.alpha).fit(train, y, connectome).predict(test, connectome)
    raise ValueError(f"unknown model {model_name!r}")


def _group_stats(y: np.ndarray, yhat: np.ndarray) -> dict[str, np.ndarray]:
    err = yhat - y
    return {
        "n": np.ones(len(y)),
        "abs": np.abs(err),
        "sq": err**2,
        "y": y,
        "y2": y**2,
        "yh": yhat,
        "yh2": yhat**2,
        "yyh": y * yhat,
        "sign": (np.sign(yhat) == np.sign(y)).astype(float),
    }


def _cluster_bootstrap(
    y: np.ndarray,
    yhat: np.ndarray,
    groups: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    uniq, inv = np.unique(groups, return_inverse=True)
    stats = _group_stats(y, yhat)
    names = list(stats)
    gsum = {k: np.bincount(inv, weights=stats[k], minlength=len(uniq)) for k in names}
    rng = np.random.default_rng(seed)
    out = {"mae": [], "rmse": [], "corr": []}
    for _ in range(n_bootstrap):
        draw = rng.integers(0, len(uniq), len(uniq))
        s = {k: float(np.sum(gsum[k][draw])) for k in names}
        n = s["n"]
        out["mae"].append(s["abs"] / n)
        out["rmse"].append(float(np.sqrt(s["sq"] / n)))
        if n > 2:
            num = s["yyh"] - s["y"] * s["yh"] / n
            var_y = max(0.0, s["y2"] - s["y"] ** 2 / n)
            var_yh = max(0.0, s["yh2"] - s["yh"] ** 2 / n)
            den = np.sqrt(var_y * var_yh)
            out["corr"].append(float(num / den) if den > 0 else float("nan"))
    result = {}
    for k, v in out.items():
        finite = [x for x in v if np.isfinite(x)]
        result[f"{k}_ci_low"] = float(np.quantile(finite, 0.025)) if finite else float("nan")
        result[f"{k}_ci_high"] = float(np.quantile(finite, 0.975)) if finite else float("nan")
    return result


def _metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    err = yhat - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum(err**2) / ss) if ss > 0 else float("nan")
    corr = float(np.corrcoef(y, yhat)[0, 1]) if np.std(y) > 0 and np.std(yhat) > 0 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson": corr,
        "sign_accuracy": float(np.mean(np.sign(yhat) == np.sign(y))),
        "n": int(len(y)),
    }


def _prediction_interval(
    train_residual: np.ndarray, yhat: np.ndarray, alpha: float = 0.10
) -> tuple[np.ndarray, np.ndarray, float]:
    lo = float(np.quantile(train_residual, alpha / 2))
    hi = float(np.quantile(train_residual, 1 - alpha / 2))
    return yhat + lo, yhat + hi, hi - lo


def _summary_row(
    task: str,
    model: str,
    y: np.ndarray,
    yhat: np.ndarray,
    groups: np.ndarray,
    *,
    cfg: BenchmarkConfig,
    n_train: int,
    n_test_animals: int,
    fold: int | None = None,
    interval_width: float | None = None,
    coverage: float | None = None,
) -> dict[str, Any]:
    m = _metrics(y, yhat)
    ci = _cluster_bootstrap(y, yhat, groups, n_bootstrap=cfg.n_bootstrap, seed=cfg.seed)
    m.update(
        {
            "task": task,
            "model": model,
            "fold": fold,
            "n_train": n_train,
            "n_test_events": int(len(np.unique(groups))),
            "n_test_animals": n_test_animals,
            "pi90_width": interval_width,
            "pi90_coverage": coverage,
        }
    )
    m.update(ci)
    return m


def evaluate_randi_benchmark(
    table: pd.DataFrame,
    *,
    connectome: pd.DataFrame | None = None,
    cfg: BenchmarkConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run event-, target-, and animal-held-out benchmarks.

    Returns ``(summary, oof_predictions)``.  The out-of-fold table is kept in
    memory for audit but is not required to reproduce the summary.
    """
    cfg = cfg or BenchmarkConfig()
    d = _clean(table, cfg.response_column)
    models = ["zero", "global_mean", "neuron_mean", "target_mean", "additive", "onehot_ridge"]
    if connectome is not None:
        models.append("connectome_ridge")
    rows: list[dict[str, Any]] = []
    oof_frames: list[pd.DataFrame] = []

    # 1. Held-out events: no event contributes to both train and test.
    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.test_fraction, random_state=cfg.seed)
    train_idx, test_idx = next(splitter.split(d, groups=d["event_id"]))
    train, test = d.iloc[train_idx], d.iloc[test_idx]
    for model in models:
        yhat = _fit_predict(train, test, model, cfg, connectome)
        rows.append(
            _summary_row(
                "heldout_events",
                model,
                test["response"].to_numpy(),
                yhat,
                test["event_id"].to_numpy(),
                cfg=cfg,
                n_train=len(train),
                n_test_animals=int(test["dataset_id"].nunique()),
            )
        )
        oof_frames.append(test[["event_id", "dataset_id", "target", "neuron"]].assign(y=test["response"], yhat=yhat, model=model, task="heldout_events"))

    # 2. Held-out stimulated neurons; collect out-of-fold predictions.
    gkf = GroupKFold(n_splits=5)
    for fold, (tr, te) in enumerate(gkf.split(d, groups=d["target"])):
        train, test = d.iloc[tr], d.iloc[te]
        for model in models:
            yhat = _fit_predict(train, test, model, cfg, connectome)
            rows.append(
                _summary_row(
                    "heldout_targets",
                    model,
                    test["response"].to_numpy(),
                    yhat,
                    test["event_id"].to_numpy(),
                    cfg=cfg,
                    n_train=len(train),
                    n_test_animals=int(test["dataset_id"].nunique()),
                    fold=fold,
                )
            )
            oof_frames.append(test[["event_id", "dataset_id", "target", "neuron"]].assign(y=test["response"], yhat=yhat, model=model, task="heldout_targets", fold=fold))

    # 3. Held-out animals (datasets).
    gkf_a = GroupKFold(n_splits=5)
    for fold, (tr, te) in enumerate(gkf_a.split(d, groups=d["dataset_id"])):
        train, test = d.iloc[tr], d.iloc[te]
        for model in models:
            yhat = _fit_predict(train, test, model, cfg, connectome)
            rows.append(
                _summary_row(
                    "heldout_animals",
                    model,
                    test["response"].to_numpy(),
                    yhat,
                    test["event_id"].to_numpy(),
                    cfg=cfg,
                    n_train=len(train),
                    n_test_animals=int(test["dataset_id"].nunique()),
                    fold=fold,
                )
            )
            oof_frames.append(test[["event_id", "dataset_id", "target", "neuron"]].assign(y=test["response"], yhat=yhat, model=model, task="heldout_animals", fold=fold))

    return pd.DataFrame(rows), pd.concat(oof_frames, ignore_index=True)


def response_reliability(
    table: pd.DataFrame,
    *,
    response_column: str = "response",
    min_events: int = 2,
    min_common_neurons: int = 3,
) -> dict[str, Any]:
    """Split-half-style repeatability of responses to the same target.

    Correlations are computed between every pair of stimulation events for the
    same animal and target, using only neurons recorded in both events.  A low
    value bounds how well *any* model can predict single-trial responses from
    pre-stimulus information.
    """
    d = table.dropna(subset=[response_column, "target", "neuron"]).copy()
    d["response"] = d[response_column].astype(float)
    corrs: list[float] = []
    n_groups = 0
    n_pairs = 0
    for (_, _), g in d.groupby(["dataset_id", "target"]):
        events = g["event_id"].unique()
        if len(events) < min_events:
            continue
        n_groups += 1
        piv = g.pivot_table(index="event_id", columns="neuron", values="response", aggfunc="mean")
        for i in range(len(piv.index)):
            for j in range(i + 1, len(piv.index)):
                a = piv.iloc[i]
                b = piv.iloc[j]
                common = a.notna() & b.notna()
                if int(common.sum()) < min_common_neurons:
                    continue
                av, bv = a[common].to_numpy(float), b[common].to_numpy(float)
                if np.std(av) == 0 or np.std(bv) == 0:
                    continue
                corrs.append(float(np.corrcoef(av, bv)[0, 1]))
                n_pairs += 1
    arr = np.asarray(corrs)
    return {
        "n_groups": int(n_groups),
        "n_pairs": int(n_pairs),
        "median_pairwise_r": float(np.median(arr)) if n_pairs else float("nan"),
        "mean_pairwise_r": float(np.mean(arr)) if n_pairs else float("nan"),
        "fraction_positive": float(np.mean(arr > 0)) if n_pairs else float("nan"),
        "q25": float(np.quantile(arr, 0.25)) if n_pairs else float("nan"),
        "q75": float(np.quantile(arr, 0.75)) if n_pairs else float("nan"),
        "interpretation": "pairwise response-profile correlation across repeated stimulations of the same target in the same animal",
    }


def aggregate_fold_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD across folds for grouped tasks; pooled value for events."""
    metric_cols = ["mae", "rmse", "r2", "pearson", "sign_accuracy", "pi90_coverage", "pi90_width"]
    out = []
    for (task, model), g in summary.groupby(["task", "model"]):
        row: dict[str, Any] = {"task": task, "model": model, "n_folds": int(g["fold"].notna().sum())}
        for c in metric_cols:
            if c in g:
                row[f"{c}_mean"] = float(g[c].mean())
                row[f"{c}_sd"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
        out.append(row)
    return pd.DataFrame(out).sort_values(["task", "model"]).reset_index(drop=True)
