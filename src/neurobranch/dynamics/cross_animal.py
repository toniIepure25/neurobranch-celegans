"""Cross-animal next-state dynamics probe.

This is a deliberately small test of RQ5: can a dynamics model trained on other
animals predict the next behavioural state without transductive representations?
It is a one-step probe, not a full behavioural simulator.  A negative result here
does not prove that no dynamics model can generalise; it bounds what these
features and this model class support.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, log_loss, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from neurobranch.behavior.prospective import behavior_matrix, population_neural_matrix
from neurobranch.data.baskoylu import FreelyMovingRecording


def _design(rec: FreelyMovingRecording, start: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    b, _ = behavior_matrix(rec)
    p, _ = population_neural_matrix(rec)
    idx = np.arange(start, rec.n_frames - 1)
    return b[idx], p[idx], rec.velocity[idx + 1], rec.reversal_vec[idx + 1].astype(int)


def _ridge_predict(Xtr, ytr, Xte) -> np.ndarray:
    scaler = StandardScaler().fit(Xtr)
    model = Ridge(alpha=10.0).fit(scaler.transform(Xtr), ytr)
    return model.predict(scaler.transform(Xte))


def _logistic_predict(Xtr, ytr, Xte) -> np.ndarray:
    scaler = StandardScaler().fit(Xtr)
    if len(np.unique(ytr)) < 2:
        return np.full(len(Xte), float(np.mean(ytr)))
    model = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
    model.fit(scaler.transform(Xtr), ytr)
    return model.predict_proba(scaler.transform(Xte))[:, 1]


def evaluate_cross_animal(recs: list[FreelyMovingRecording]) -> tuple[pd.DataFrame, dict[str, Any]]:
    designs = [_design(r) for r in recs]
    rows = []
    for i, (b, p, v_next, s_next) in enumerate(designs):
        train = [d for j, d in enumerate(designs) if j != i]
        btr = np.vstack([d[0] for d in train])
        ptr = np.vstack([d[1] for d in train])
        vtr = np.concatenate([d[2] for d in train])
        str_ = np.concatenate([d[3] for d in train])
        models = {
            "persistence": (b[:, 0], v_next),
            "behavior_ridge": (_ridge_predict(btr, vtr, b), v_next),
            "behavior_neural_ridge": (_ridge_predict(np.hstack([btr, ptr]), vtr, np.hstack([b, p])), v_next),
        }
        for name, (pred, y) in models.items():
            rows.append(
                {
                    "animal_id": recs[i].animal_id,
                    "task": "next_velocity",
                    "model": name,
                    "mae": float(mean_absolute_error(y, pred)),
                    "r2": float(r2_score(y, pred)),
                    "n": int(len(y)),
                }
            )
        for name, (Xtr, Xte) in {
            "behavior_logistic": (btr, b),
            "behavior_neural_logistic": (np.hstack([btr, ptr]), np.hstack([b, p])),
        }.items():
            prob = _logistic_predict(Xtr, str_, Xte)
            pred = (prob >= 0.5).astype(int)
            rows.append(
                {
                    "animal_id": recs[i].animal_id,
                    "task": "next_reversal_state",
                    "model": name,
                    "balanced_accuracy": float(balanced_accuracy_score(s_next, pred)),
                    "log_loss": float(log_loss(s_next, np.clip(prob, 1e-6, 1 - 1e-6), labels=[0, 1])),
                    "prevalence": float(np.mean(s_next)),
                    "n": int(len(s_next)),
                }
            )
    m = pd.DataFrame(rows)
    summary: dict[str, Any] = {"n_animals": len(recs), "aggregate": {}}
    for (task, model), g in m.groupby(["task", "model"]):
        summary["aggregate"][f"{task}:{model}"] = {
            k: float(g[k].mean())
            for k in ["mae", "r2", "balanced_accuracy", "log_loss", "prevalence"]
            if k in g
        }
    for task in ["next_velocity", "next_reversal_state"]:
        piv = m[m.task == task].pivot(index="animal_id", columns="model", values="mae" if task == "next_velocity" else "log_loss")
        if "behavior_neural_ridge" in piv and "behavior_ridge" in piv:
            d = (piv["behavior_neural_ridge"] - piv["behavior_ridge"]).dropna()
            summary[f"{task}_delta_neural"] = {
                "mean": float(d.mean()),
                "n_negative": int((d < 0).sum()),
                "n": int(len(d)),
            }
        if "behavior_neural_logistic" in piv and "behavior_logistic" in piv:
            d = (piv["behavior_neural_logistic"] - piv["behavior_logistic"]).dropna()
            summary[f"{task}_delta_neural"] = {
                "mean": float(d.mean()),
                "n_negative": int((d < 0).sum()),
                "n": int(len(d)),
            }
    return m, summary


def write_cross_animal(recs: list[FreelyMovingRecording], out_dir: str | Path = "reports/phase3") -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m, s = evaluate_cross_animal(recs)
    m.to_csv(out / "cross_animal_dynamics.csv", index=False)
    (out / "cross_animal_dynamics.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    return s
