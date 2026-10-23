"""NC-MCM reference-artifact verification.

The upstream method is Julia (nilab).  We verify the distributed per-worm
artifacts: cluster assignments, Markovianity p-values, and a reimplementation
of the lag-2 surrogate test.  We also cross-validate the behaviour decoder.
We do not present this as a full re-run of the Julia pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import KFold

from neurobranch.config import PROJECT_ROOT
from neurobranch.data.kato import STATE_ORDER

RESULTS_DIR = PROJECT_ROOT / "references" / "external" / "nc-mcm" / "precomputed_results"


def load_ncmcm(worm_id: int = 1) -> dict[str, Any] | None:
    p = RESULTS_DIR / f"nc_mcm_model_worm_{worm_id}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return {
        "x": d["arr_0"],
        "clusters": d["arr_1"].astype(int),
        "behavior": d["arr_2"].astype(int),
        "state_names_artifact": [str(s) for s in d["arr_3"]],
        "markov_p": d["arr_4"].astype(float),
        "behavior_probabilities": d["arr_5"],
        "source": str(p.relative_to(PROJECT_ROOT)),
        "worm_id": worm_id,
    }


def markovian_test(z: np.ndarray, n_simulations: int = 200, seed: int = 0) -> tuple[float, np.ndarray]:
    """Python port of nilab/src/causal.jl:markovian (lag-2 surrogate test)."""
    z = np.asarray(z, dtype=int)
    states = np.unique(z)
    remap = {s: i for i, s in enumerate(states)}
    z = np.array([remap[s] for s in z], dtype=int)
    N = len(states)
    M = len(z)
    counts = np.zeros((N, N, N))
    np.add.at(counts, (z[2:], z[1:-1], z[:-2]), 1)
    P = counts / max(1, counts.sum())
    Pz0z1 = P.sum(axis=2)
    Pz1 = P.sum(axis=(0, 2))
    P1 = np.divide(Pz0z1, Pz1, out=np.zeros_like(Pz0z1), where=Pz1 > 0)

    def lag2_variance(triple: np.ndarray) -> float:
        p2 = triple / np.maximum(triple.sum(axis=0, keepdims=True), 1e-12)
        return float(np.var(p2, axis=2).sum())

    observed = lag2_variance(counts.astype(float))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_simulations):
        # Simulate a first-order Markov chain with the empirical P1.  Inverse-CDF
        # sampling is much faster than one rng.choice call per frame, and remains
        # statistically identical.
        sim = np.empty(M, dtype=int)
        sim[0] = rng.integers(N)
        u = rng.random(M - 1)
        for m in range(1, M):
            prob = P1[:, sim[m - 1]]
            total = prob.sum()
            if total > 0:
                sim[m] = min(int(np.searchsorted(np.cumsum(prob), u[m - 1], side="right")), N - 1)
            else:
                sim[m] = rng.integers(N)
        c = np.zeros((N, N, N))
        np.add.at(c, (sim[2:], sim[1:-1], sim[:-2]), 1)
        null.append(lag2_variance(c))
    p = 1.0 - float(np.mean(observed >= np.asarray(null)))
    # Transition matrix P(z_t | z_{t-1})
    trans = np.zeros((N, N))
    np.add.at(trans, (z[1:], z[:-1]), 1)
    trans = trans / np.maximum(trans.sum(axis=0, keepdims=True), 1e-12)
    return p, trans


def behavior_decoder_cv(
    x: np.ndarray,
    b: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Blocked CV logistic decoder, matching the upstream predmodel family."""
    labels = np.asarray(b, dtype=int) - 1  # 1..8 -> 0..7
    kf = KFold(n_splits=n_splits, shuffle=False)
    preds = np.empty_like(labels)
    for tr, te in kf.split(x):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(x[tr], labels[tr])
        preds[te] = clf.predict(x[te])
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "confusion": confusion_matrix(labels, preds).tolist(),
        "n": int(len(labels)),
        "state_order": list(STATE_ORDER),
    }


def summarize_worm(worm_id: int = 1, *, selected_k: int = 3, n_markov_sims: int = 150) -> dict[str, Any]:
    d = load_ncmcm(worm_id)
    if d is None:
        return {"available": False, "worm_id": worm_id}
    c = d["clusters"][:, selected_k - 1]
    p_pre = float(d["markov_p"][selected_k - 1])
    p_re, trans = markovian_test(c, n_simulations=n_markov_sims, seed=worm_id)
    # Dwell-time and behavior composition per cognitive state.
    states = np.unique(c)
    dwell = {}
    composition = {}
    for s in states:
        runs, n = [], 0
        for v in c:
            if v == s:
                n += 1
            elif n:
                runs.append(n)
                n = 0
        if n:
            runs.append(n)
        dwell[int(s)] = {"n_runs": len(runs), "mean_frames": float(np.mean(runs)) if runs else 0.0}
        comp = {}
        for code in range(1, 9):
            comp[STATE_ORDER[code - 1]] = float(np.mean(d["behavior"][c == s] == code))
        composition[int(s)] = comp
    dec = behavior_decoder_cv(d["x"], d["behavior"], n_splits=5, seed=worm_id)
    # Verify the artifact's state-name array against observed counts.
    counts = np.bincount(d["behavior"], minlength=9)[1:]
    return {
        "available": True,
        "worm_id": worm_id,
        "source": d["source"],
        "n_frames": int(len(c)),
        "selected_k": selected_k,
        "markov_p_precomputed": p_pre,
        "markov_p_reimplemented": float(p_re),
        "markov_p_abs_diff": float(abs(p_pre - p_re)),
        "transition_matrix": trans.tolist(),
        "dwell": dwell,
        "behavior_composition": composition,
        "decoder_cv": dec,
        "state_counts": {STATE_ORDER[i]: int(counts[i]) for i in range(8)},
        "artifact_state_names": d["state_names_artifact"],
        "state_name_note": (
            "The distributed arr_3 is not aligned with the integer codes; observed counts "
            "match STATE_ORDER (b=1 -> dt, ..., b=8 -> vt). We use STATE_ORDER."
        ),
        "markov_p_curve": d["markov_p"].tolist(),
        "p_curve_note": (
            "p is the fraction of first-order Markov surrogates whose lag-2 heterogeneity "
            "is at least the observed value; high p means the first-order Markov null is not rejected."
        ),
    }


def summarize_all(*, selected_k: int = 3) -> pd.DataFrame:
    rows = []
    for w in range(1, 6):
        s = summarize_worm(w, selected_k=selected_k, n_markov_sims=80)
        if not s.get("available"):
            continue
        rows.append(
            {
                "worm_id": w,
                "n_frames": s["n_frames"],
                "K": selected_k,
                "markov_p_precomputed": s["markov_p_precomputed"],
                "markov_p_reimplemented": s["markov_p_reimplemented"],
                "decoder_accuracy": s["decoder_cv"]["accuracy"],
                "decoder_balanced_accuracy": s["decoder_cv"]["balanced_accuracy"],
                "mean_dwell_frames": float(np.mean([v["mean_frames"] for v in s["dwell"].values()])),
            }
        )
    return pd.DataFrame(rows)
