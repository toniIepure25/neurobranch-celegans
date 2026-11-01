"""Oxygen-transition response diagnostics with explicit time-confound controls."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SENSORY = ("URXL", "URXR", "AQR", "PQR", "BAGL", "BAGR", "AWA", "AWB", "AWC")


def session_id(data) -> str:
    """Stable display identifier; external WormData lacks the source session_id."""
    return f"kato_stim_{int(data.worm_id)}"


def event_table(data) -> pd.DataFrame:
    oxygen = np.asarray(data.external_input, dtype=np.float64)
    diff = np.diff(oxygen)
    frames = np.flatnonzero(np.abs(diff) > 1e-9) + 1
    rows = []
    for frame in frames:
        frm = float(oxygen[frame - 1])
        to = float(oxygen[frame])
        rows.append(
            {
                "session_id": session_id(data),
                "worm_id": int(data.worm_id),
                "frame": int(frame),
                "time_seconds": float(data.time_seconds[frame]),
                "from_percent": frm,
                "to_percent": to,
                "direction": "down" if to < frm else "up",
            }
        )
    return pd.DataFrame(rows)


def response_vectors(data, events: pd.DataFrame, *, seconds: float = 5.0) -> pd.DataFrame:
    fps = float(data.recording.fps)
    pre_n = max(3, int(round(seconds * fps)))
    post_n = pre_n
    names = list(data.predictor_names)
    rows = []
    for _, event in events.iterrows():
        e = int(event["frame"])
        if e - pre_n < 0 or e + post_n > data.n_frames:
            continue
        pre = data.X[e - pre_n : e]
        post = data.X[e : e + post_n]
        resp = post.mean(axis=0) - pre.mean(axis=0)
        for j, name in enumerate(names):
            rows.append(
                {
                    "session_id": session_id(data),
                    "worm_id": int(data.worm_id),
                    "frame": e,
                    "time_seconds": float(event["time_seconds"]),
                    "direction": event["direction"],
                    "neuron": name,
                    "response_standardized": float(resp[j]),
                    "pre_mean": float(pre[:, j].mean()),
                    "post_mean": float(post[:, j].mean()),
                }
            )
    return pd.DataFrame(rows)


def time_design(t: np.ndarray, fps: float) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    x = (t - t.min()) / max(1e-9, t.max() - t.min())
    cols = [np.ones_like(x), x, x**2]
    for harmonic in (1, 2, 3):
        omega = 2 * np.pi * harmonic / 30.0
        cols.extend([np.sin(omega * t), np.cos(omega * t)])
    return np.column_stack(cols)


def event_design(
    t: np.ndarray,
    events: pd.DataFrame,
    fps: float,
    *,
    lags_s: float = 5.0,
    n_lags: int | None = None,
) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    if n_lags is None:
        n_lags = max(1, int(round(lags_s * fps)))
    n_lags = int(n_lags)
    E = np.zeros((len(t), n_lags), dtype=np.float64)
    frames = np.asarray(events.frame, dtype=np.int64) - 1
    for e, direction in zip(frames, events.direction):
        sign = 1.0 if direction == "up" else -1.0
        for lag in range(n_lags):
            idx = e + lag
            if 0 <= idx < len(t):
                E[idx, lag] = sign
    return E


def _ridge(design: np.ndarray, target: np.ndarray, ridge: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = design.mean(axis=0)
    sd = design.std(axis=0)
    sd[sd < 1e-8] = 1.0
    xs = (design - mu) / sd
    y = target - target.mean(axis=0)
    coef = np.linalg.solve(xs.T @ xs + ridge * np.eye(xs.shape[1]), xs.T @ y)
    intercept = target.mean(axis=0) - (mu / sd) @ coef
    return coef / sd[:, None], intercept, mu


def _predict(design: np.ndarray, coef: np.ndarray, intercept: np.ndarray, mu: np.ndarray) -> np.ndarray:
    # Design columns are standardized with train mu/sd; coef is in raw units.
    return design @ coef + intercept


def _fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    ridge: float = 1.0,
) -> np.ndarray:
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd[sd < 1e-8] = 1.0
    xs = (train_x - mu) / sd
    yc = train_y - train_y.mean(axis=0)
    coef = np.linalg.solve(xs.T @ xs + ridge * np.eye(xs.shape[1]), xs.T @ yc)
    intercept = train_y.mean(axis=0) - (mu / sd) @ coef
    return ((test_x - mu) / sd) @ coef + intercept


def window_response_comparison(
    data,
    events: pd.DataFrame,
    *,
    seconds: float = 5.0,
    ridge: float = 1.0,
) -> pd.DataFrame:
    """Baseline-corrected event-window model comparison.

    Targets are post-event deviations from the causal pre-event mean.  This
    removes the large absolute fluorescence drift that dominates an absolute-X
    fit.  Time-only controls use event-relative elapsed time, not future data.
    """
    fps = float(data.recording.fps)
    pre_n = max(3, int(round(seconds * fps)))
    post_n = pre_n
    lags = np.arange(post_n)
    T = np.column_stack([np.ones(post_n), lags / max(1, post_n - 1), (lags / max(1, post_n - 1)) ** 2])
    onehot = np.eye(post_n)
    rows = []

    def build(event_frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if event_frame - pre_n < 0 or event_frame + post_n > data.n_frames:
            return None
        pre = data.X[event_frame - pre_n : event_frame].mean(axis=0)
        post = data.X[event_frame : event_frame + post_n]
        return post - pre, T, onehot

    for name, design in [("time_only", T), ("event_lag", onehot), ("event_plus_time", np.column_stack([onehot, T]))]:
        train_x, train_y = [], []
        for frame in events.frame:
            item = build(int(frame))
            if item is None:
                continue
            y, _, _ = item
            if data.masks["train"][int(frame)]:
                train_x.append(design)
                train_y.append(y)
        if not train_x:
            continue
        x_train = np.vstack(train_x)
        y_train = np.vstack(train_y)
        test_x, test_y = [], []
        for frame in events.frame:
            item = build(int(frame))
            if item is None or not data.masks["test"][int(frame)]:
                continue
            y, _, _ = item
            test_x.append(design)
            test_y.append(y)
        if not test_x:
            continue
        x_test = np.vstack(test_x)
        y_test = np.vstack(test_y)
        pred = _fit_predict(x_train, y_train, x_test, ridge)
        mse = float(np.mean((pred - y_test) ** 2))
        zero = float(np.mean(y_test**2))
        rows.append(
            {
                "session_id": session_id(data),
                "worm_id": int(data.worm_id),
                "model": name,
                "n_train_events": len(train_x),
                "n_test_events": len(test_x),
                "mse_baseline_corrected": mse,
                "zero_baseline_mse": zero,
                "skill_vs_no_response": 1.0 - mse / zero if zero > 0 else float("nan"),
            }
        )
    # Placebo events halfway between transitions; fit on training placebos and
    # evaluate on the real post-event windows.
    placebo = events.copy()
    placebo["frame"] = np.clip(
        placebo.frame + int(round(15 * fps)), 0, data.n_frames - 1
    )
    train_x, train_y = [], []
    for frame in placebo.frame:
        item = build(int(frame))
        if item is not None and data.masks["train"][int(frame)]:
            y, _, _ = item
            train_x.append(onehot)
            train_y.append(y)
    test_x, test_y = [], []
    for frame in events.frame:
        item = build(int(frame))
        if item is not None and data.masks["test"][int(frame)]:
            y, _, _ = item
            test_x.append(onehot)
            test_y.append(y)
    if train_x and test_x:
        pred = _fit_predict(np.vstack(train_x), np.vstack(train_y), np.vstack(test_x), ridge)
        y_test = np.vstack(test_y)
        mse = float(np.mean((pred - y_test) ** 2))
        zero = float(np.mean(y_test**2))
        rows.append(
            {
                "session_id": session_id(data),
                "worm_id": int(data.worm_id),
                "model": "placebo_event_lag",
                "n_train_events": len(train_x),
                "n_test_events": len(test_x),
                "mse_baseline_corrected": mse,
                "zero_baseline_mse": zero,
                "skill_vs_no_response": 1.0 - mse / zero if zero > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def time_confound(data, events: pd.DataFrame) -> dict[str, Any]:
    idx = np.flatnonzero(data.masks["train"])
    t = data.time_seconds[idx]
    D = time_design(t, data.recording.fps)
    e = event_design(t, events[events.frame < len(data.X)], data.recording.fps)
    e_presence = (np.abs(e) > 0).astype(np.float64)
    # R2 of event columns explained by the time basis.
    coef = np.linalg.lstsq(D, e, rcond=None)[0]
    pred = D @ coef
    ss_res = float(np.sum((e - pred) ** 2))
    ss_tot = float(np.sum((e - e.mean(axis=0)) ** 2))
    coef_p = np.linalg.lstsq(D, e_presence, rcond=None)[0]
    pred_p = D @ coef_p
    ss_res_p = float(np.sum((e_presence - pred_p) ** 2))
    ss_tot_p = float(np.sum((e_presence - e_presence.mean(axis=0)) ** 2))
    return {
        "session_id": session_id(data),
        "n_train_frames": int(len(idx)),
        "n_events_total": int(len(events)),
        "event_indicator_time_r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "event_presence_time_r2": 1.0 - ss_res_p / ss_tot_p if ss_tot_p > 0 else float("nan"),
        "event_presence_point_biserial_time": float(
            np.corrcoef(t, e_presence.sum(axis=1))[0, 1]
        )
        if e_presence.size
        else float("nan"),
        "combined_design_condition": float(
            np.linalg.cond(np.column_stack([D, e_presence])) if e_presence.size else np.linalg.cond(D)
        ),
    }


def model_comparison(
    data,
    events: pd.DataFrame,
    *,
    ridge: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = np.asarray(data.masks["train"], dtype=bool)
    test_mask = np.asarray(data.masks["test"], dtype=bool)
    t = data.time_seconds
    T = time_design(t, data.recording.fps)
    E = event_design(t, events, data.recording.fps)
    # Matched placebo events halfway between real transitions.
    placebo = events.copy()
    placebo["frame"] = np.clip(placebo.frame + int(round(15 * data.recording.fps)), 0, data.n_frames - 1)
    Ep = event_design(t, placebo, data.recording.fps)
    test_event_frames = []
    for e in events.frame:
        test_event_frames.extend(range(int(e), min(data.n_frames, int(e) + int(round(5 * data.recording.fps)))))
    test_event_frames = np.asarray(sorted(set(f for f in test_event_frames if test_mask[f])), dtype=np.int64)
    if len(test_event_frames) == 0:
        return pd.DataFrame(), pd.DataFrame()

    pre_n = max(3, int(round(5 * data.recording.fps)))
    baseline = np.stack([data.X[max(0, f - pre_n) : f].mean(axis=0) for f in test_event_frames])
    truth = data.X[test_event_frames]
    designs = {
        "time_only": T,
        "event_only": E,
        "event_plus_time": np.column_stack([E, T]),
        "placebo_event": Ep,
        "placebo_plus_time": np.column_stack([Ep, T]),
    }
    rows, detail = [], []
    for name, design in designs.items():
        pred = _fit_predict(design[train_mask], data.X[train_mask], design[test_event_frames], ridge)
        mse = float(np.mean((pred - truth) ** 2))
        base_mse = float(np.mean((baseline - truth) ** 2))
        rows.append(
            {
                "session_id": session_id(data),
                "worm_id": int(data.worm_id),
                "model": name,
                "n_test_frames": int(len(test_event_frames)),
                "mse": mse,
                "persistence_mse": base_mse,
                "skill_vs_persistence": 1.0 - mse / base_mse if base_mse > 0 else float("nan"),
            }
        )
        detail.append(
            {
                "session_id": session_id(data),
                "worm_id": int(data.worm_id),
                "model": name,
                "test_frame": test_event_frames,
                "mse_by_neuron": np.mean((pred - truth) ** 2, axis=0),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(detail)


def event_reliability(responses: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, pair_rows = [], []
    for session, group in responses.groupby("session_id"):
        pivot = group.pivot_table(
            index=["frame", "direction"], columns="neuron", values="response_standardized"
        )
        if len(pivot) < 2:
            continue
        values = pivot.to_numpy()
        corr = np.corrcoef(values)
        tri = corr[np.triu_indices_from(corr, k=1)]
        rows.append(
            {
                "session_id": session,
                "worm_id": int(group.worm_id.iloc[0]),
                "n_events": int(len(pivot)),
                "mean_event_pair_correlation": float(np.nanmean(tri)),
                "median_event_pair_correlation": float(np.nanmedian(tri)),
                "n_neurons": int(values.shape[1]),
            }
        )
        # Direction contrast.
        if set(pivot.index.get_level_values("direction")) == {"up", "down"}:
            up = values[pivot.index.get_level_values("direction") == "up"].mean(axis=0)
            down = values[pivot.index.get_level_values("direction") == "down"].mean(axis=0)
            pair_rows.append(
                {
                    "session_id": session,
                    "worm_id": int(group.worm_id.iloc[0]),
                    "up_minus_down_cosine": float(
                        np.dot(up, down) / max(1e-12, np.linalg.norm(up) * np.linalg.norm(down))
                    ),
                    "up_response_mean": float(np.mean(up)),
                    "down_response_mean": float(np.mean(down)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)
