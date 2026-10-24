"""Time-resolved and trajectory analyses around the reversal offset.

Negative relative frames are prospective with respect to the offset; frame 0
and later are retrospective positive controls and are labelled as such.  These
analyses describe neural dynamics and do not establish a causal role.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from neurobranch.behavior.prospective import causal_zscore
from neurobranch.behavior.turn_direction import CIRCUIT_NEURONS, population_features
from neurobranch.data.kramer import load_all

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "phase4"
REPORT.mkdir(parents=True, exist_ok=True)
KS = tuple(range(-12, 13))
POST_TURN_FRAMES = 2


def _frame_behavior(rec, start: int, end: int, k: int) -> np.ndarray | None:
    f = end + k
    if f < 0 or f >= rec.n_frames:
        return None
    base = float(np.mean(rec.dv[start : min(start + 2, end)]))
    return np.asarray(
        [
            float(rec.dv[f]),
            float(rec.dv[f] - base),
            float(rec.velocity[f]),
            abs(float(rec.velocity[f])),
            float(rec.angular_velocity[f]),
            float(rec.pumping[f]),
            float(np.sin(rec.worm_angle[f])),
            float(np.cos(rec.worm_angle[f])),
        ],
        dtype=float,
    )


def _fit_loo(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    p = np.full(len(y), np.nan)
    for animal in np.unique(groups):
        te = groups == animal
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            p[te] = np.mean(y[tr])
            continue
        scaler = StandardScaler().fit(X[tr])
        a = scaler.transform(X[tr])
        b = scaler.transform(X[te])
        model = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs", random_state=17)
        model.fit(a, y[tr])
        p[te] = model.predict_proba(b)[:, 1]
    return p


def main() -> None:
    recs, _ = load_all(cohorts=("wt_octanol", "wt_diacetyl"))
    z_by_animal = {r.animal_id: causal_zscore(r.traces_raw) for r in recs}
    events = []
    for rec in recs:
        ev = rec.event_table(min_length=12)
        ev = ev[ev["valid_post"]]
        for row in ev.itertuples(index=False):
            events.append((rec, int(row.start), int(row.end), int(row.event_index), 1 if row.direction == "dorsal" else 0))
    print("events", len(events), "animals", len(recs))

    rows = []
    for k in KS:
        Xb, Xn, y = [], [], []
        groups = []
        for rec, _start, end, _event_index, label in events:
            b = _frame_behavior(rec, _start, end, k)
            if b is None:
                continue
            Xb.append(b)
            Xn.append(population_features(z_by_animal[rec.animal_id], end + k, end + k + 1))
            y.append(label)
            groups.append(rec.animal_id)
        y = np.asarray(y, dtype=int)
        groups = np.asarray(groups)
        if len(np.unique(y)) < 2:
            continue
        Xb = np.vstack(Xb)
        Xn = np.vstack(Xn)
        for mode, X in (("behavior", Xb), ("neural_population", Xn), ("behavior_population", np.hstack([Xb, Xn]))):
            p = _fit_loo(X, y, groups)
            auroc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
            rows.append(
                {
                    "relative_frame": k,
                    "mode": mode,
                    "n": int(len(y)),
                    "n_animals": int(len(np.unique(groups))),
                    "auroc": auroc,
                    "prospective": bool(k < 0),
                    "guard_frames": int(-k) if k < 0 else 0,
                }
            )
    pd.DataFrame(rows).to_csv(REPORT / "turn_dynamics_timeresolved.csv", index=False)

    # PCA trajectories, averaged within animal first so that animals with many
    # events do not dominate the population trajectory.
    vectors = []
    meta = []
    for rec, _start, end, _event_index, label in events:
        z = z_by_animal[rec.animal_id]
        for k in KS:
            f = end + k
            if 0 <= f < rec.n_frames:
                # Animals have different ROI counts; use the same
                # permutation-invariant population state used by the decoder.
                vectors.append(population_features(z, f, f + 1))
                meta.append((rec.animal_id, label, k))
    V = np.asarray(vectors)
    M = pd.DataFrame(meta, columns=["animal_id", "label", "k"])
    pca = PCA(n_components=3, random_state=17).fit(V)
    scores = pca.transform(V)
    M[["pc1", "pc2", "pc3"]] = scores
    # average within animal/direction/k, then across animals
    per_animal = M.groupby(["animal_id", "label", "k"], as_index=False)[["pc1", "pc2", "pc3"]].mean()
    traj = per_animal.groupby(["label", "k"], as_index=False)[["pc1", "pc2", "pc3"]].agg(["mean", "sem"])
    traj.columns = ["_".join([c for c in col if c]) for col in traj.columns]
    traj = traj.reset_index(drop=True)
    traj["direction"] = traj["label"].map({0: "ventral", 1: "dorsal"})
    traj = traj.drop(columns=["label"])
    traj.to_csv(REPORT / "turn_pca_trajectories.csv", index=False)
    evr = {"explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_], "n_vectors": int(len(V))}
    (REPORT / "turn_pca_variance.json").write_text(json.dumps(evr, indent=2), encoding="utf-8")

    # Per-event peak time for the prespecified circuit neurons.
    peak_rows = []
    for rec, _start, end, _event_index, label in events:
        z = z_by_animal[rec.animal_id]
        for name in CIRCUIT_NEURONS:
            idx = rec.identified_indices.get(name)
            if idx is None:
                continue
            trace = z[max(0, end - 12) : min(rec.n_frames, end + 13), idx]
            if len(trace) == 0:
                continue
            offsets = np.arange(max(0, end - 12), min(rec.n_frames, end + 13)) - end
            peak_rows.append(
                {
                    "animal_id": rec.animal_id,
                    "cohort": rec.cohort,
                    "event_index": _event_index,
                    "direction": "dorsal" if label else "ventral",
                    "label": name,
                    "peak_frame": int(offsets[int(np.argmax(trace))]),
                    "peak_z": float(np.max(trace)),
                }
            )
    peaks = pd.DataFrame(peak_rows)
    peaks.to_csv(REPORT / "turn_neuron_peak_events.csv", index=False)
    if not peaks.empty:
        summary = (
            peaks.groupby(["label", "direction"])["peak_frame"]
            .agg(["count", "mean", "median", "std"])
            .reset_index()
        )
        summary.to_csv(REPORT / "turn_neuron_peak_times.csv", index=False)
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
