"""Deterministic adaptive baselines for the probabilistic stage."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from neurobranch.adaptive.evaluation import evaluation_landmarks
from neurobranch.adaptive.var import FrozenVAR, RollingVAR, spectral_radius, stabilize_matrix
from neurobranch.preprocessing import split_starts


@dataclass
class ConstantForecast:
    value: np.ndarray
    spectral_radius: float = 0.0
    fit_record: dict | None = None

    def forecast(self, x0: np.ndarray, horizon: int) -> np.ndarray:
        return np.repeat(self.value[None, :], int(horizon), axis=0)


@dataclass
class LocalMeanModel:
    window: int = 500
    ewma_halflife: float | None = None
    min_history: int = 50

    def fit_at(self, X: np.ndarray, t: int) -> ConstantForecast | None:
        t = int(t)
        if t < self.min_history:
            return None
        if self.ewma_halflife is None:
            lo = max(0, t - self.window + 1)
            values = X[lo : t + 1]
            mu = values.mean(axis=0)
        else:
            starts = np.arange(0, t + 1)
            weights = 0.5 ** ((t - starts) / float(self.ewma_halflife))
            mu = np.average(X[: t + 1], axis=0, weights=weights)
        return ConstantForecast(value=mu, fit_record={"landmark": t})


@dataclass
class IndependentARForecast:
    a: np.ndarray
    b: np.ndarray
    spectral_radius: float
    fit_record: dict

    def forecast(self, x0: np.ndarray, horizon: int) -> np.ndarray:
        x = np.asarray(x0, dtype=np.float64).copy()
        out = np.empty((int(horizon), x.size), dtype=np.float64)
        for step in range(int(horizon)):
            x = self.a * x + self.b
            out[step] = x
        return out


@dataclass
class IndependentARModel:
    window: int = 500
    ridge: float = 1.0
    spectral_cap: float = 0.995
    min_transitions: int = 50

    def fit_at(self, X: np.ndarray, t: int) -> IndependentARForecast | None:
        t = int(t)
        lo = max(1, t - self.window + 1)
        starts = np.arange(lo, t, dtype=np.int64)
        if len(starts) < self.min_transitions:
            return None
        design = X[starts]
        target = X[starts + 1]
        good = np.isfinite(design).all(axis=1) & np.isfinite(target).all(axis=1)
        design, target = design[good], target[good]
        if len(design) < self.min_transitions:
            return None
        mx = design.mean(axis=0)
        my = target.mean(axis=0)
        xc = design - mx
        yc = target - my
        denom = (xc**2).sum(axis=0) + float(self.ridge)
        a = (xc * yc).sum(axis=0) / denom
        b = my - a * mx
        raw = float(np.max(np.abs(a)))
        if raw > self.spectral_cap:
            a = a * (self.spectral_cap / raw)
            b = my - a * mx
        return IndependentARForecast(
            a=a,
            b=b,
            spectral_radius=float(np.max(np.abs(a))),
            fit_record={
                "n_transitions": int(len(design)),
                "raw_radius": raw,
                "landmark": t,
                "clipped": bool(raw > self.spectral_cap),
            },
        )


@dataclass
class PersistenceForecast:
    def forecast(self, x0: np.ndarray, horizon: int) -> np.ndarray:
        return np.repeat(np.asarray(x0, dtype=np.float64)[None, :], int(horizon), axis=0)


@dataclass
class BaselineConfig:
    local_window: int
    ewma_halflife: float
    ar_window: int


def _validation_starts(data, history: int, horizon: int = 20, stride: int = 10) -> np.ndarray:
    train = np.flatnonzero(data.masks["train"])
    if len(train) == 0:
        return np.array([], dtype=np.int64)
    start = int(train[int(0.8 * len(train))])
    stop = int(train[-1] - horizon)
    if stop <= start:
        return np.array([], dtype=np.int64)
    grid = np.arange(start, stop + 1, stride, dtype=np.int64)
    return grid[grid >= history]


def select_baseline_config(
    datasets,
    *,
    local_windows: tuple[int, ...] = (100, 200, 500, 1000),
    halflives: tuple[float, ...] = (100.0, 300.0, 1000.0),
    ar_windows: tuple[int, ...] = (200, 500, 1000),
    horizon: int = 20,
    stride: int = 10,
) -> tuple[BaselineConfig, pd.DataFrame]:
    """Select each baseline on the last 20% of the training split only."""
    rows = []
    for kind, candidates in (
        ("local_mean", local_windows),
        ("ewma_mean", halflives),
        ("independent_ar", ar_windows),
    ):
        for value in candidates:
            scores = []
            for data in datasets:
                starts = _validation_starts(data, history=int(value), horizon=horizon, stride=stride)
                if len(starts) < 5:
                    continue
                model = (
                    LocalMeanModel(window=int(value))
                    if kind == "local_mean"
                    else LocalMeanModel(ewma_halflife=float(value))
                    if kind == "ewma_mean"
                    else IndependentARModel(window=int(value))
                )
                for t in starts:
                    fitted = model.fit_at(data.X, int(t))
                    if fitted is None:
                        continue
                    pred = fitted.forecast(data.X[int(t)], horizon)
                    truth = data.X[int(t) + horizon]
                    scores.append(float(np.mean((pred[-1] - truth) ** 2)))
            if scores:
                rows.append(
                    {
                        "kind": kind,
                        "setting": float(value),
                        "n_validation_starts": len(scores),
                        "mean_validation_h20_mse": float(np.mean(scores)),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no validation starts for baseline selection")
    chosen = {}
    for kind in ("local_mean", "ewma_mean", "independent_ar"):
        part = frame[frame.kind == kind]
        if part.empty:
            raise RuntimeError(f"no validation result for {kind}")
        best = part.sort_values("mean_validation_h20_mse").iloc[0]
        chosen[kind] = float(best.setting)
    config = BaselineConfig(
        local_window=int(chosen["local_mean"]),
        ewma_halflife=float(chosen["ewma_mean"]),
        ar_window=int(chosen["independent_ar"]),
    )
    return config, frame


def evaluate_deterministic_session(
    data,
    config: BaselineConfig,
    *,
    horizon_max: int = 100,
    horizons: tuple[int, ...] = (1, 5, 10, 20, 50, 100),
    stride: int = 10,
    ridge: float = 1.0,
    spectral_cap: float = 0.995,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    landmarks = evaluation_landmarks(data, horizon_max=horizon_max, stride=stride)
    frozen = FrozenVAR.fit(
        data.X,
        split_starts(data, "train", history=1, horizon=1),
        ridge=ridge,
        spectral_cap=spectral_cap,
    )
    models = {
        "B0_persistence": PersistenceForecast(),
        "B1_frozen_var": frozen,
        "B2_local_mean": LocalMeanModel(window=config.local_window),
        "B3_ewma_mean": LocalMeanModel(ewma_halflife=config.ewma_halflife),
        "B4_independent_ar": IndependentARModel(window=config.ar_window),
        "B5_rolling_var_w500": RollingVAR(window=500),
    }
    rows, neuron_rows, updates = [], [], []
    for t in landmarks:
        x0 = data.X[int(t)].copy()
        truth_all = data.X[int(t) + 1 : int(t) + horizon_max + 1]
        for name, model in models.items():
            started = perf_counter()
            fitted = model if name in {"B0_persistence", "B1_frozen_var"} else model.fit_at(data.X, int(t))
            elapsed = perf_counter() - started
            if fitted is None:
                continue
            pred = fitted.forecast(x0, horizon_max)
            updates.append(
                {
                    "worm_id": int(data.worm_id),
                    "landmark": int(t),
                    "model": name,
                    "update_seconds": float(elapsed),
                    "spectral_radius": float(spectral_radius(fitted.A)) if hasattr(fitted, "A") else 0.0,
                }
            )
            for h in horizons:
                y = truth_all[h - 1]
                p = pred[h - 1]
                mse = float(np.mean((p - y) ** 2))
                base = float(np.mean((x0 - y) ** 2))
                rows.append(
                    {
                        "worm_id": int(data.worm_id),
                        "landmark": int(t),
                        "model": name,
                        "horizon": int(h),
                        "mse": mse,
                        "persistence_mse": base,
                        "skill_vs_persistence": 1.0 - mse / base if base > 0 else float("nan"),
                    }
                )
            p = pred[19]
            y = truth_all[19]
            for j, neuron in enumerate(data.predictor_names):
                pj = float((p[j] - y[j]) ** 2)
                bj = float((x0[j] - y[j]) ** 2)
                neuron_rows.append(
                    {
                        "worm_id": int(data.worm_id),
                        "landmark": int(t),
                        "model": name,
                        "neuron": neuron,
                        "neuron_index": int(j),
                        "mse_h20": pj,
                        "persistence_mse_h20": bj,
                        "skill_vs_persistence_h20": 1.0 - pj / bj if bj > 0 else float("nan"),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(neuron_rows), pd.DataFrame(updates)


def paired_deterministic(
    per_landmark: pd.DataFrame,
    *,
    reference: str = "B1_frozen_var",
    block: int = 5,
    n_boot: int = 1000,
) -> pd.DataFrame:
    from neurobranch.adaptive.evaluation import block_bootstrap_ci

    h20 = per_landmark[per_landmark.horizon == 20]
    pivot = h20.pivot_table(index=["worm_id", "landmark"], columns="model", values="mse")
    rows = []
    for model in pivot.columns:
        if model == reference:
            continue
        diff = (pivot[reference] - pivot[model]).dropna().to_numpy()
        mean, lo, hi = block_bootstrap_ci(diff, block=block, n_boot=n_boot)
        rows.append(
            {
                "model": model,
                "reference": reference,
                "n_pairs": int(len(diff)),
                "mean_improvement": mean,
                "ci_low": lo,
                "ci_high": hi,
                "fraction_better": float((diff > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "BaselineConfig",
    "ConstantForecast",
    "IndependentARForecast",
    "IndependentARModel",
    "LocalMeanModel",
    "PersistenceForecast",
    "evaluate_deterministic_session",
    "paired_deterministic",
    "select_baseline_config",
    "stabilize_matrix",
]
