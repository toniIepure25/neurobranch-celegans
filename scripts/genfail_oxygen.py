"""GENFAIL Stage E: oxygen transition responses and elapsed-time confound."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, git_state
from neurobranch.data.external import build_kato_stim_context_worm_data, load_kato_stim
from neurobranch.diagnostics.oxygen import (
    SENSORY,
    event_design,
    event_reliability,
    event_table,
    model_comparison,
    response_vectors,
    session_id,
    time_confound,
    time_design,
    window_response_comparison,
)

OUT = PROJECT_ROOT / "reports/simulator/failure_diagnostics/oxygen"


def _stim_path() -> Path:
    candidates = [
        Path(os.environ.get("TEMP", "")) / "neurobranch_external/WT_Stim.mat",
        PROJECT_ROOT / "data/external/WT_Stim.mat",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("WT_Stim.mat not found; set TEMP or place it in data/external")


def _latency_table(data, events: pd.DataFrame, responses: pd.DataFrame) -> pd.DataFrame:
    fps = float(data.recording.fps)
    pre = int(round(5 * fps))
    post = int(round(10 * fps))
    rows = []
    for _, event in events.iterrows():
        e = int(event["frame"])
        if e - pre < 0 or e + post > data.n_frames:
            continue
        window = data.X[e - pre : e + post]
        base = window[:pre].mean(axis=0)
        resp = window - base
        for j, name in enumerate(data.predictor_names):
            lag = int(np.argmax(np.abs(resp[pre:, j]))) if post > 0 else 0
            rows.append(
                {
                    "session_id": session_id(data),
                    "worm_id": int(data.worm_id),
                    "frame": e,
                    "direction": event["direction"],
                    "neuron": name,
                    "peak_lag_seconds": float(lag / fps),
                    "peak_amplitude_standardized": float(resp[pre + lag, j]),
                }
            )
    return pd.DataFrame(rows)


def _session_disjoint(data_list, events_by_session) -> pd.DataFrame:
    names = {
        name
        for name in data_list[0].predictor_names
        if any(ch.isalpha() for ch in name) and not name.startswith("trace_")
    }
    for d in data_list[1:]:
        names &= {
            name
            for name in d.predictor_names
            if any(ch.isalpha() for ch in name) and not name.startswith("trace_")
        }
    names = sorted(names)
    rows = []
    for held_idx, held in enumerate(data_list):
        if not names:
            break
        train_rows_x = {"time_only": [], "event_only": [], "event_plus_time": []}
        train_rows_y = {name: [] for name in names}
        for i, d in enumerate(data_list):
            if i == held_idx:
                continue
            events = events_by_session[session_id(d)]
            T = time_design(d.time_seconds, d.recording.fps)
            E = event_design(d.time_seconds, events, d.recording.fps, n_lags=15)
            mask = d.masks["train"]
            train_rows_x["time_only"].append(T[mask])
            train_rows_x["event_only"].append(E[mask])
            train_rows_x["event_plus_time"].append(np.column_stack([T, E])[mask])
            for name in names:
                train_rows_y[name].append(d.X[mask, d.predictor_names.index(name)])
        X_train = {key: np.vstack(value) for key, value in train_rows_x.items()}
        test_events = events_by_session[session_id(held)]
        T = time_design(held.time_seconds, held.recording.fps)
        E = event_design(held.time_seconds, test_events, held.recording.fps, n_lags=15)
        frames = []
        for e in test_events.frame:
            frames.extend(range(int(e), min(held.n_frames, int(e) + int(round(5 * held.recording.fps)))))
        frames = np.asarray(sorted(set(f for f in frames if held.masks["test"][f])), dtype=np.int64)
        if len(frames) == 0:
            continue
        truth = np.stack([held.X[frames, held.predictor_names.index(n)] for n in names], axis=1)
        pre_n = max(3, int(round(5 * held.recording.fps)))
        baseline = np.stack(
            [held.X[max(0, f - pre_n) : f].mean(axis=0)[
                [held.predictor_names.index(n) for n in names]
            ] for f in frames]
        )
        zero_mse = float(np.mean((baseline - truth) ** 2))
        for model, design_test in [
            ("time_only", T[frames]),
            ("event_only", E[frames]),
            ("event_plus_time", np.column_stack([T, E])[frames]),
        ]:
            pred = []
            for name in names:
                y_train = np.concatenate(train_rows_y[name])
                pred.append(_fit_predict(X_train[model], y_train, design_test))
            pred = np.column_stack(pred)
            mse = float(np.mean((pred - truth) ** 2))
            rows.append(
                {
                    "held_out_session": session_id(held),
                    "model": model,
                    "n_common_named_neurons": len(names),
                    "n_test_frames": len(frames),
                    "mse": mse,
                    "no_response_mse": zero_mse,
                    "skill_vs_no_response": 1.0 - mse / zero_mse if zero_mse > 0 else float("nan"),
                    "n_neurons": len(names),
                }
            )
    return pd.DataFrame(rows)


def _fit_predict(X: np.ndarray, y: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1.0
    xs = (X - mu) / sd
    yc = y - y.mean()
    coef = np.linalg.solve(xs.T @ xs + np.eye(xs.shape[1]), xs.T @ yc)
    intercept = y.mean() - (mu / sd) @ coef
    return ((Xt - mu) / sd) @ coef + intercept


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    recs = load_kato_stim(_stim_path())
    data_list = [build_kato_stim_context_worm_data(rec) for rec in recs]
    events_by_session = {}
    events_all = []
    responses_all = []
    latency_all = []
    confound_rows = []
    model_rows = []
    window_rows = []
    for data in data_list:
        events = event_table(data)
        events_by_session[session_id(data)] = events
        events_all.append(events)
        responses = response_vectors(data, events)
        responses_all.append(responses)
        latency_all.append(_latency_table(data, events, responses))
        confound_rows.append(time_confound(data, events))
        m, _ = model_comparison(data, events)
        model_rows.append(m)
        window_rows.append(window_response_comparison(data, events))
    events = pd.concat(events_all, ignore_index=True)
    responses = pd.concat(responses_all, ignore_index=True)
    latency = pd.concat(latency_all, ignore_index=True)
    events.to_csv(OUT / "oxygen_events_verified.csv", index=False)
    responses.to_csv(OUT / "oxygen_event_responses.csv", index=False)
    latency.to_csv(OUT / "oxygen_event_latency.csv", index=False)
    reliability, direction = event_reliability(responses)
    reliability.to_csv(OUT / "oxygen_event_reliability.csv", index=False)
    direction.to_csv(OUT / "oxygen_direction_contrast.csv", index=False)
    confound = pd.DataFrame(confound_rows)
    confound.to_csv(OUT / "oxygen_time_confound.csv", index=False)
    model_df = pd.concat(model_rows, ignore_index=True)
    model_df.to_csv(OUT / "oxygen_model_comparison.csv", index=False)
    window_df = pd.concat(window_rows, ignore_index=True)
    window_df.to_csv(OUT / "oxygen_window_models.csv", index=False)
    window_agg = (
        window_df.groupby("model", as_index=False)
        .agg(
            mean_skill=("skill_vs_no_response", "mean"),
            median_skill=("skill_vs_no_response", "median"),
            n_sessions=("session_id", "nunique"),
        )
        .sort_values("mean_skill", ascending=False)
    )
    window_agg.to_csv(OUT / "oxygen_window_aggregate.csv", index=False)
    disjoint = _session_disjoint(data_list, events_by_session)
    disjoint.to_csv(OUT / "oxygen_session_disjoint.csv", index=False)

    # Sensory availability: exact labels only.
    availability = []
    for data in data_list:
        for name in sorted(set(data.predictor_names) & set(SENSORY)):
            availability.append(
                {"session_id": session_id(data), "neuron": name, "available": True}
            )
    pd.DataFrame(availability).to_csv(OUT / "oxygen_sensory_availability.csv", index=False)

    agg = (
        model_df.groupby("model", as_index=False)
        .agg(
            mean_skill=("skill_vs_persistence", "mean"),
            median_skill=("skill_vs_persistence", "median"),
            n_sessions=("session_id", "nunique"),
        )
        .sort_values("mean_skill", ascending=False)
    )
    agg.to_csv(OUT / "oxygen_model_aggregate.csv", index=False)
    schedule_jitter = (
        events.assign(event_index=events.groupby("session_id").cumcount())
        .groupby("event_index")
        .time_seconds.std()
        .max()
    )
    payload = {
        "stage": "E_oxygen_response",
        "evidence_tier": "exploratory; repeated fixed schedule, sessions are not independent animals",
        "source_sha256": recs[0].source_sha256,
        "n_sessions": len(data_list),
        "n_transitions_per_session": [int((events.session_id == session_id(d)).sum()) for d in data_list],
        "oxygen_levels": sorted(events.from_percent.unique().tolist()),
        "directions": events.direction.value_counts().to_dict(),
        "sensory_labels_present": sorted(set(availability_df.neuron)) if (availability_df := pd.DataFrame(availability)).size else [],
        "time_confound_max_r2": float(confound.event_presence_time_r2.max()),
        "schedule_jitter_sd_seconds_max": float(schedule_jitter),
        "aggregate_model": agg.to_dict(orient="records"),
        "baseline_corrected_window_model": window_agg.to_dict(orient="records"),
        "session_disjoint_models": disjoint.groupby("model").skill_vs_no_response.mean().to_dict() if len(disjoint) else {},
        "warnings": [
            "oxygen is a controlled environmental input, not targeted neuronal stimulation",
            "the 30 s schedule is repeated in every session; event and elapsed time are strongly confounded",
            "seven sessions are not seven independent animals",
            "no claim about sensory mechanism or behavioural causality",
        ],
        "git": git_state(),
    }
    (OUT / "oxygen_summary.json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(agg.to_string(index=False))
    print(window_agg.to_string(index=False))
    print(confound[["session_id", "event_indicator_time_r2", "combined_design_condition"]].to_string(index=False))
    print(reliability.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
