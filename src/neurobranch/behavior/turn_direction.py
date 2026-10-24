"""Animal-disjoint, prospective decoding of post-reversal turn direction.

The task is deliberately *not* the published five-fold event split.  Each
held-out animal is a complete group: no frames, events or neuron labels from
that animal enter feature selection, scaling or model fitting.  Features end at
least two frames before the reversal offset and the direction label is read from
the first two frames after the offset.  The middle four-frame window is the
primary, prospective representation; early and late windows are secondary.

The archive's ``dv`` field is the orientation-corrected dorsal/ventral head
angle.  ``head_angle`` is the raw camera sign and is not used for cross-animal
D/V labels.  Positive ``dv`` is dorsal.  This mapping is tested against
marker-neuron activity in the audit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from neurobranch.behavior.prospective import causal_zscore
from neurobranch.data.kramer import KramerRecording

SEGMENT_FRAMES = 4
POST_FRAMES = 2
MIN_REVERSAL_FRAMES = 12
PRIMARY_WINDOW = "mid"
WINDOWS = ("early", "mid", "late")
# Prespecified from the paper's head-steering circuit.  Missing labels are zero
# after causal normalisation; no held-out animal is used to decide this list.
CIRCUIT_NEURONS = ("SAAV", "RMDV", "RMDD", "SMDV", "SMDD", "RIV", "SMBV", "SMBD")
MODES = (
    "behavior",
    "behavior_population",
    "behavior_saav",
    "behavior_circuit",
    "behavior_identified",
    "neural_population",
    "neural_saav",
    "neural_circuit",
)


@dataclass
class TurnDesign:
    animal_id: str
    cohort: str
    events: pd.DataFrame
    y: np.ndarray
    behavior: np.ndarray
    population: np.ndarray
    saav: np.ndarray
    circuit: np.ndarray
    identified: np.ndarray
    label_present: np.ndarray
    segment_starts: np.ndarray
    segment_ends: np.ndarray


def segment_bounds(start: int, end: int, window: str) -> tuple[int, int] | None:
    """Return a four-frame segment that ends before the reversal offset.

    A two-frame guard band separates the last feature frame from the outcome
    window.  The middle window is therefore genuinely prospective even for the
    shortest accepted reversal.
    """
    length = end - start
    if length < MIN_REVERSAL_FRAMES:
        return None
    if window == "early":
        seg_start = start + 1
    elif window == "mid":
        seg_start = start + max(1, (length - SEGMENT_FRAMES) // 2)
    elif window == "late":
        seg_start = end - SEGMENT_FRAMES - POST_FRAMES
    else:
        raise ValueError(f"unknown window {window!r}")
    seg_end = seg_start + SEGMENT_FRAMES
    if seg_start < start or seg_end > end - POST_FRAMES:
        return None
    return seg_start, seg_end


def _slope(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(x), dtype=float), x, 1)[0])


def behavior_features(rec: KramerRecording, start: int, end: int, seg_start: int, seg_end: int) -> tuple[np.ndarray, list[str]]:
    """Causal behavioural/context features for one preselected segment."""
    seg = slice(seg_start, seg_end)
    dv = np.asarray(rec.dv, dtype=float).ravel()[seg]
    velocity = np.asarray(rec.velocity, dtype=float).ravel()[seg]
    angular = np.asarray(rec.angular_velocity, dtype=float).ravel()[seg]
    pumping = np.asarray(rec.pumping, dtype=float).ravel()[seg]
    worm = np.asarray(rec.worm_angle, dtype=float).ravel()[seg]
    base_slice = slice(start, min(start + 2, end))
    baseline = float(np.mean(rec.dv[base_slice])) if end > start else 0.0
    # Odour context is taken only up to the end of the feature segment; the
    # post-segment odour trace is not observed at prediction time.
    odor = np.asarray(rec.odor_idx, dtype=float).ravel()[start:seg_end]
    vals = [
        float(np.mean(dv)),
        float(np.std(dv)),
        float(np.min(dv)),
        float(np.max(dv)),
        float(dv[0]),
        float(dv[-1]),
        _slope(dv),
        float(np.mean(dv) - baseline),
        float(dv[-1] - dv[0]),
        baseline,
        float(np.mean(velocity)),
        float(np.std(velocity)),
        float(np.min(velocity)),
        float(np.max(velocity)),
        float(velocity[-1]),
        float(np.mean(np.abs(velocity))),
        float(np.mean(angular)),
        float(np.std(angular)),
        float(np.mean(pumping)),
        float(np.std(pumping)),
        float(np.mean(np.sin(worm))),
        float(np.mean(np.cos(worm))),
        float(np.mean(odor > 0)),
    ]
    names = [
        "head_mean",
        "head_sd",
        "head_min",
        "head_max",
        "head_first",
        "head_last",
        "head_slope",
        "head_rel_mean",
        "head_delta",
        "head_baseline",
        "velocity_mean",
        "velocity_sd",
        "velocity_min",
        "velocity_max",
        "velocity_last",
        "speed_mean",
        "angular_mean",
        "angular_sd",
        "pumping_mean",
        "pumping_sd",
        "worm_sin_mean",
        "worm_cos_mean",
        "odor_fraction",
    ]
    return np.asarray(vals, dtype=float), names


def population_features(z: np.ndarray, seg_start: int, seg_end: int) -> np.ndarray:
    """Permutation-invariant statistics for each frame of the segment."""
    q = np.asarray(z[seg_start:seg_end], dtype=float)
    if q.shape[0] == 0:
        return np.zeros(0, dtype=float)
    cols = [
        np.mean(q, axis=1),
        np.std(q, axis=1),
        np.quantile(q, 0.10, axis=1),
        np.quantile(q, 0.50, axis=1),
        np.quantile(q, 0.90, axis=1),
        np.mean(q > 1.0, axis=1),
        np.mean(q < -1.0, axis=1),
        np.mean(np.abs(q), axis=1),
    ]
    return np.concatenate(cols).astype(float)


def trace_means(z: np.ndarray, rec: KramerRecording, labels: tuple[str, ...], seg_start: int, seg_end: int) -> np.ndarray:
    """Mean causally normalised activity for named traces; missing labels -> 0."""
    out = np.zeros(len(labels), dtype=float)
    idx = rec.identified_indices
    for j, name in enumerate(labels):
        i = idx.get(name)
        if i is not None:
            out[j] = float(np.mean(z[seg_start:seg_end, i]))
    return out


def build_designs(
    recs: list[KramerRecording],
    *,
    window: str = PRIMARY_WINDOW,
    min_length: int = MIN_REVERSAL_FRAMES,
    circuit_labels: tuple[str, ...] = CIRCUIT_NEURONS,
    neural_shift_frames: int = 0,
) -> tuple[list[TurnDesign], list[str]]:
    """Build event-level designs and the study-wide identified-label vocabulary.

    The vocabulary is fixed from metadata before outcomes are scored.  Fold-level
    feature selection keeps only labels covered in training animals.
    """
    all_labels: set[str] = set()
    for rec in recs:
        all_labels.update(rec.identified_indices)
    labels = tuple(sorted(all_labels))
    designs: list[TurnDesign] = []
    for rec in recs:
        table = rec.event_table(min_length=min_length)
        table = table[table["valid_post"]].reset_index(drop=True)
        if table.empty:
            continue
        z = causal_zscore(rec.traces_raw)
        rows: list[dict[str, Any]] = []
        y: list[int] = []
        behavior: list[np.ndarray] = []
        population: list[np.ndarray] = []
        saav: list[np.ndarray] = []
        circuit: list[np.ndarray] = []
        identified: list[np.ndarray] = []
        present: list[np.ndarray] = []
        starts: list[int] = []
        ends: list[int] = []
        for row in table.itertuples(index=False):
            bounds = segment_bounds(int(row.start), int(row.end), window)
            if bounds is None:
                continue
            seg_start, seg_end = bounds
            neural_start = seg_start + neural_shift_frames
            neural_end = seg_end + neural_shift_frames
            if neural_start < 0 or neural_end > rec.n_frames:
                continue
            b, _ = behavior_features(rec, int(row.start), int(row.end), seg_start, seg_end)
            pop = population_features(z, neural_start, neural_end)
            saav_feat = trace_means(z, rec, ("SAAV",), neural_start, neural_end)
            circ = trace_means(z, rec, circuit_labels, neural_start, neural_end)
            ident = trace_means(z, rec, labels, neural_start, neural_end)
            pvec = np.zeros(len(labels), dtype=bool)
            for name in rec.identified_indices:
                if name in labels:
                    pvec[labels.index(name)] = True
            rows.append(
                {
                    "event_id": f"{rec.animal_id}:{int(row.event_index)}",
                    "animal_id": rec.animal_id,
                    "cohort": rec.cohort,
                    "event_index": int(row.event_index),
                    "start": int(row.start),
                    "end": int(row.end),
                    "post_dv": float(row.post_dv),
                    "direction": str(row.direction),
                    "near_zero": bool(row.near_zero),
                    "odor_present": bool(row.odor_present),
                    "segment_start": seg_start,
                    "segment_end": seg_end,
                    "neural_segment_start": neural_start,
                    "neural_segment_end": neural_end,
                }
            )
            y.append(1 if row.direction == "dorsal" else 0)
            behavior.append(b)
            population.append(pop)
            saav.append(saav_feat)
            circuit.append(circ)
            identified.append(ident)
            present.append(pvec)
            starts.append(seg_start)
            ends.append(seg_end)
        if not rows:
            continue
        designs.append(
            TurnDesign(
                animal_id=rec.animal_id,
                cohort=rec.cohort,
                events=pd.DataFrame(rows),
                y=np.asarray(y, dtype=int),
                behavior=np.vstack(behavior),
                population=np.vstack(population),
                saav=np.vstack(saav),
                circuit=np.vstack(circuit),
                identified=np.vstack(identified),
                label_present=np.vstack(present),
                segment_starts=np.asarray(starts, dtype=int),
                segment_ends=np.asarray(ends, dtype=int),
            )
        )
    return designs, list(labels)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    out: dict[str, float] = {
        "n": int(len(y)),
        "prevalence": float(np.mean(y)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, (p >= 0.5).astype(int))),
        "accuracy": float(np.mean((p >= 0.5).astype(int) == y)),
        "auroc": float("nan"),
        "auprc": float("nan"),
        "calibration_slope": float("nan"),
        "calibration_intercept": float("nan"),
    }
    if len(np.unique(y)) > 1:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
        logit = np.log(p / (1.0 - p)).reshape(-1, 1)
        cal = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs").fit(logit, y)
        out["calibration_slope"] = float(cal.coef_[0, 0])
        out["calibration_intercept"] = float(cal.intercept_[0])
    return out


def _context(designs: list[TurnDesign]) -> np.ndarray:
    # Cohort is known at prediction time.  Within a single-cohort run this is
    # constant and is harmlessly removed by the scaler.  One row per event.
    vals = np.concatenate(
        [np.full(len(d.y), 1.0 if d.cohort == "wt_diacetyl" else 0.0) for d in designs]
    )
    return vals[:, None]


def _stack(designs: list[TurnDesign], key: str) -> np.ndarray:
    return np.vstack([getattr(d, key) for d in designs])


def _fold_identified_labels(train: list[TurnDesign], labels: list[str], min_fraction: float = 0.5) -> list[int]:
    counts = np.sum(np.vstack([d.label_present for d in train]), axis=0)
    need = max(2, int(np.ceil(min_fraction * len(train))))
    idx = np.flatnonzero(counts >= need)
    if len(idx) == 0:
        idx = np.flatnonzero(counts >= 2)
    return idx.tolist()


def _matrices(
    train: list[TurnDesign],
    test: TurnDesign,
    mode: str,
    labels: list[str],
    *,
    identified_idx: list[int] | None = None,
    drop_circuit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    ctx_tr = _context(train)
    ctx_te = _context([test])
    b_tr, b_te = _stack(train, "behavior"), test.behavior
    p_tr, p_te = _stack(train, "population"), test.population
    s_tr, s_te = _stack(train, "saav"), test.saav
    c_tr, c_te = _stack(train, "circuit"), test.circuit
    if drop_circuit is not None:
        c_tr = np.delete(c_tr, drop_circuit, axis=1)
        c_te = np.delete(c_te, drop_circuit, axis=1)
    if mode == "behavior":
        return np.hstack([b_tr, ctx_tr]), np.hstack([b_te, ctx_te])
    if mode == "behavior_population":
        return np.hstack([b_tr, p_tr, ctx_tr]), np.hstack([b_te, p_te, ctx_te])
    if mode == "behavior_saav":
        return np.hstack([b_tr, s_tr, ctx_tr]), np.hstack([b_te, s_te, ctx_te])
    if mode == "behavior_circuit":
        return np.hstack([b_tr, c_tr, ctx_tr]), np.hstack([b_te, c_te, ctx_te])
    if mode == "behavior_identified":
        if identified_idx is None:
            raise ValueError("identified_idx required")
        i_tr = np.vstack([d.identified[:, identified_idx] for d in train])
        i_te = test.identified[:, identified_idx]
        return np.hstack([b_tr, i_tr, ctx_tr]), np.hstack([b_te, i_te, ctx_te])
    if mode == "neural_population":
        return p_tr, p_te
    if mode == "neural_saav":
        return s_tr, s_te
    if mode == "neural_circuit":
        return c_tr, c_te
    if mode == "behavior_circuit_ablate":
        return np.hstack([b_tr, c_tr, ctx_tr]), np.hstack([b_te, c_te, ctx_te])
    raise ValueError(mode)


def _fit_predict(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, seed: int = 17, C: float = 0.1) -> np.ndarray:
    scaler = StandardScaler().fit(np.nan_to_num(xtr, nan=0.0, posinf=0.0, neginf=0.0))
    a = np.nan_to_num(scaler.transform(xtr), nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(scaler.transform(xte), nan=0.0, posinf=0.0, neginf=0.0)
    if len(np.unique(ytr)) < 2:
        return np.full(len(xte), float(np.mean(ytr)))
    model = LogisticRegression(C=C, max_iter=2000, tol=1e-4, solver="lbfgs", random_state=seed)
    model.fit(a, ytr)
    return model.predict_proba(b)[:, 1]


def evaluate_loo(
    designs: list[TurnDesign],
    labels: list[str],
    *,
    modes: tuple[str, ...] = MODES,
    seed: int = 17,
    C: float = 0.1,
    drop_circuit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Leave-one-animal-out predictions, metrics and paired bootstrap deltas."""
    rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    for i, test in enumerate(designs):
        train = [d for j, d in enumerate(designs) if j != i]
        if not train:
            continue
        identified_idx = _fold_identified_labels(train, labels)
        # Keep the class prevalence of the training animals; no test calibration.
        for mode in modes:
            if mode == "behavior_identified" and not identified_idx:
                continue
            xtr, xte = _matrices(
                train,
                test,
                mode,
                labels,
                identified_idx=identified_idx,
                drop_circuit=drop_circuit,
            )
            p = _fit_predict(xtr, np.concatenate([d.y for d in train]), xte, seed=seed, C=C)
            for k, row in enumerate(test.events.itertuples(index=False)):
                pred_rows.append(
                    {
                        "animal_id": test.animal_id,
                        "cohort": test.cohort,
                        "event_id": row.event_id,
                        "event_index": int(row.event_index),
                        "start": int(row.start),
                        "end": int(row.end),
                        "post_dv": float(row.post_dv),
                        "direction": row.direction,
                        "y": int(test.y[k]),
                        "mode": mode,
                        "p": float(p[k]),
                        "window": "primary",
                        "segment_start": int(row.segment_start),
                        "segment_end": int(row.segment_end),
                    }
                )
    preds = pd.DataFrame(pred_rows)
    if preds.empty:
        return pd.DataFrame(), preds, {"error": "no predictions"}
    for mode, group in preds.groupby("mode"):
        m = _metrics(group["y"].to_numpy(), group["p"].to_numpy())
        rows.append({"mode": mode, **m})
    metrics = pd.DataFrame(rows)
    # Per-animal metrics, including only animals with both classes where AUC is
    # defined.  Pooled event-level metrics remain the primary report.
    per_animal_rows = []
    for (animal, mode), group in preds.groupby(["animal_id", "mode"]):
        per_animal_rows.append({"animal_id": animal, "mode": mode, **_metrics(group["y"].to_numpy(), group["p"].to_numpy())})
    per_animal = pd.DataFrame(per_animal_rows)
    summary = _summarise(preds, seed=seed)
    summary["n_designs"] = len(designs)
    summary["n_labels"] = len(labels)
    summary["identified_labels_median"] = float(np.median([len(_fold_identified_labels([d for j, d in enumerate(designs) if j != i], labels)) for i in range(len(designs))])) if designs else 0.0
    return metrics, per_animal, summary | {"predictions": preds}


def _summarise(preds: pd.DataFrame, *, seed: int = 17, n_boot: int = 2000) -> dict[str, Any]:
    out: dict[str, Any] = {"aggregate": {}, "deltas": {}}
    for mode, group in preds.groupby("mode"):
        out["aggregate"][mode] = _metrics(group["y"].to_numpy(), group["p"].to_numpy())
    animals = np.asarray(sorted(preds["animal_id"].unique()))
    if len(animals) < 3:
        return out
    rng = np.random.default_rng(seed)
    modes = sorted(preds["mode"].unique())
    # Align every mode on the same event order, then bootstrap by animal using
    # integer positions.  This avoids thousands of DataFrame concatenations and
    # guarantees that paired deltas use identical events.
    base_group = preds[preds["mode"] == "behavior"].sort_values("event_id")
    event_order = base_group["event_id"].to_numpy()
    base_y = base_group["y"].to_numpy(dtype=int)
    animal_order = base_group["animal_id"].to_numpy()
    p_by_mode: dict[str, np.ndarray] = {}
    for mode in modes:
        g = preds[preds["mode"] == mode].set_index("event_id").reindex(event_order)
        p_by_mode[mode] = g["p"].to_numpy(dtype=float)
    positions = {a: np.flatnonzero(animal_order == a) for a in animals}
    draws: dict[str, list[float]] = {f"delta_{m}_vs_behavior": [] for m in modes if m != "behavior"}
    for _ in range(n_boot):
        chosen = rng.choice(animals, size=len(animals), replace=True)
        idx = np.concatenate([positions[a] for a in chosen])
        yb = base_y[idx]
        if len(np.unique(yb)) < 2:
            continue
        base_p = p_by_mode["behavior"][idx]
        for mode in modes:
            if mode == "behavior":
                continue
            pm = p_by_mode[mode][idx]
            good = np.isfinite(pm) & np.isfinite(base_p)
            if good.sum() < 3 or len(np.unique(yb[good])) < 2:
                continue
            base_auc = roc_auc_score(yb[good], base_p[good])
            draws[f"delta_{mode}_vs_behavior"].append(
                float(roc_auc_score(yb[good], pm[good]) - base_auc)
            )
    for key, vals in draws.items():
        if not vals:
            continue
        arr = np.asarray(vals)
        out["deltas"][key] = {
            "mean": float(arr.mean()),
            "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
            "n_boot": int(len(arr)),
            "p_delta_le_0": float(np.mean(arr <= 0)),
        }
    return out


def evaluate_within_animal(designs: list[TurnDesign], labels: list[str], *, modes: tuple[str, ...] = MODES, seed: int = 17) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Non-generalising five-fold event split, for comparison with the paper.

    This is intentionally *not* the primary evaluation: events from one animal
    can appear in both training and test folds.  It quantifies how much of the
    published-style performance depends on that dependence.
    """
    y = np.concatenate([d.y for d in designs])
    idx = np.arange(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred_rows = []
    offsets = np.cumsum([0] + [len(d.y) for d in designs])
    for fold, (tr, te) in enumerate(skf.split(idx, y)):
        # Build pooled matrices once; fold selection is by event index.
        X = {
            "behavior": np.vstack([np.hstack([d.behavior, np.zeros((len(d.y), 1))]) for d in designs]),
            "population": np.vstack([d.population for d in designs]),
            "circuit": np.vstack([d.circuit for d in designs]),
        }
        identified_idx = _fold_identified_labels(designs, labels)
        X["identified"] = np.vstack([d.identified[:, identified_idx] for d in designs]) if identified_idx else np.zeros((len(y), 0))
        for mode in modes:
            if mode == "behavior":
                a = X["behavior"]
            elif mode == "behavior_population":
                a = np.hstack([X["behavior"], X["population"]])
            elif mode == "behavior_circuit":
                a = np.hstack([X["behavior"], X["circuit"]])
            elif mode == "behavior_identified":
                if not identified_idx:
                    continue
                a = np.hstack([X["behavior"], X["identified"]])
            elif mode == "neural_population":
                a = X["population"]
            elif mode == "neural_circuit":
                a = X["circuit"]
            else:
                continue
            p = _fit_predict(a[tr], y[tr], a[te], seed=seed + fold, C=0.1)
            for k, j in enumerate(te):
                design_i = int(np.searchsorted(offsets, j, side="right") - 1)
                local = int(j - offsets[design_i])
                d = designs[design_i]
                pred_rows.append(
                    {
                        "fold": fold,
                        "mode": mode,
                        "animal_id": d.animal_id,
                        "event_id": d.events.iloc[local]["event_id"],
                        "y": int(y[j]),
                        "p": float(p[k]),
                    }
                )
    preds = pd.DataFrame(pred_rows)
    rows = [{"mode": mode, **_metrics(g["y"], g["p"])} for mode, g in preds.groupby("mode")]
    return pd.DataFrame(rows), preds


def evaluate_external(train: list[TurnDesign], test: list[TurnDesign], labels: list[str], *, modes: tuple[str, ...] = MODES, seed: int = 17) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on all training animals and score a disjoint cohort."""
    if not train or not test:
        return pd.DataFrame(), pd.DataFrame()
    identified_idx = _fold_identified_labels(train, labels)
    ytr = np.concatenate([d.y for d in train])
    rows = []
    pred_rows = []
    for mode in modes:
        if mode == "behavior_identified" and not identified_idx:
            continue
        ps = []
        for d in test:
            xtr, xte = _matrices(train, d, mode, labels, identified_idx=identified_idx)
            p = _fit_predict(xtr, ytr, xte, seed=seed, C=0.1)
            ps.append(p)
            for k, row in enumerate(d.events.itertuples(index=False)):
                pred_rows.append({"animal_id": d.animal_id, "cohort": d.cohort, "event_id": row.event_id, "y": int(d.y[k]), "mode": mode, "p": float(p[k])})
        rows.append({"mode": mode, **_metrics(np.concatenate([d.y for d in test]), np.concatenate(ps))})
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def circuit_ablation(
    designs: list[TurnDesign],
    labels: list[str],
    *,
    seed: int = 17,
    n_random: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Model-dependent feature ablations for the prespecified circuit features.

    This is *not* an experimental neuron ablation.  It removes one feature from
    a predictive model and asks whether that model's held-out AUC changes more
    than removing a randomly chosen matched feature.
    """
    _, _, full = evaluate_loo(designs, labels, modes=("behavior_circuit",), seed=seed)
    full_auc = full["aggregate"]["behavior_circuit"]["auroc"]
    rows = []
    for j, name in enumerate(CIRCUIT_NEURONS):
        _, _, s = evaluate_loo(designs, labels, modes=("behavior_circuit",), seed=seed, drop_circuit=j)
        rows.append({"feature": name, "kind": "prespecified", "auroc": s["aggregate"]["behavior_circuit"]["auroc"], "delta_vs_full": s["aggregate"]["behavior_circuit"]["auroc"] - full_auc})
    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        j = int(rng.integers(0, len(CIRCUIT_NEURONS)))
        _, _, s = evaluate_loo(designs, labels, modes=("behavior_circuit",), seed=seed, drop_circuit=j)
        rows.append({"feature": CIRCUIT_NEURONS[j], "kind": "random", "auroc": s["aggregate"]["behavior_circuit"]["auroc"], "delta_vs_full": s["aggregate"]["behavior_circuit"]["auroc"] - full_auc})
    out = pd.DataFrame(rows)
    rand = out.loc[out["kind"] == "random", "delta_vs_full"].to_numpy(dtype=float)
    summary = out[out["kind"] == "prespecified"].copy()
    summary["full_auroc"] = full_auc
    summary["random_mean"] = float(rand.mean()) if len(rand) else float("nan")
    summary["random_sd"] = float(rand.std(ddof=1)) if len(rand) > 1 else float("nan")
    summary["random_n"] = int(len(rand))
    # One-sided empirical p for "dropping this feature helps at least as much
    # as dropping a randomly chosen matched feature".  The +1 correction keeps
    # the p value finite with a small random-control sample.
    summary["empirical_p"] = [
        float((1 + np.sum(rand <= d)) / (len(rand) + 1)) if np.isfinite(d) and len(rand) else float("nan")
        for d in summary["delta_vs_full"]
    ]
    return out, summary


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
