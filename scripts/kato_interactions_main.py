"""Kato-only neural interaction, mode, state and sensitivity audit.

Stages:
A. exact reproduction of the frozen J0/J1/J2/J5 H20 results;
B. cross-neuron information versus own-history and matched temporal surrogates;
C. operator spectra, timescales and stability;
D. training-only neural-state transitions;
E/F. bounded mathematical sensitivity of named neurons;
G. observational identifiability, including synthetic known-coupling controls.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import jd_evaluate as JD  # noqa: E402

from neurobranch.adaptive.evaluation import evaluation_landmarks  # noqa: E402
from neurobranch.diagnostics.common import load_nostim  # noqa: E402
from neurobranch.joint_dynamics.models import block_bootstrap_starts  # noqa: E402
from neurobranch.kato_interactions.metrics import (  # noqa: E402
    ensemble_metrics,
    paired_table,
    session_block_ci,
)
from neurobranch.kato_interactions.models import (  # noqa: E402
    FrozenLinearModel,
    diagonal_copy,
    fit_var,
    fit_window,
)
from neurobranch.kato_interactions.sensitivity import (  # noqa: E402
    matched_control_indices,
    pairwise_disagreement,
    response_metrics,
    support_fraction,
)
from neurobranch.kato_interactions.spectra import (  # noqa: E402
    complex_pair_stability,
    leading_decay_frames,
    mode_table,
    operator_summary,
)
from neurobranch.kato_interactions.states import (  # noqa: E402
    assign_states,
    fit_states,
    occupancy,
    row_normalize,
    simulate_markov,
    state_metrics,
    transition_counts,
)
from neurobranch.kato_interactions.surrogates import fit_shuffled_window, shuffle_window  # noqa: E402
from neurobranch.kato_interactions.synthetic import audit as synthetic_audit  # noqa: E402
from neurobranch.preprocessing import split_starts  # noqa: E402
from neurobranch.probabilistic.baselines import IndependentARModel, LocalMeanModel  # noqa: E402
from neurobranch.probabilistic.metrics import (  # noqa: E402
    crps_ensemble,
    energy_score,
    interval_coverage,
)

OUT = ROOT / "reports" / "simulator" / "kato_interactions"
FROZEN = ROOT / "reports" / "simulator" / "joint_dynamics"
MEMBERS = 100
HORIZON = 100
PRIMARY_HORIZON = 20
STRIDE = 10
N_STATES = 6
SURROGATE_DRAWS = 6
TARGETS = ("AVBL", "AVBR", "AVEL", "AVER", "RIVL", "RIVR")
AMPLITUDES = (1.0,)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=False), encoding="utf-8")


def full_neuron_metrics(ens: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    h = PRIMARY_HORIZON - 1
    cov = {}
    for level in (50, 80, 95):
        cov[f"coverage_full_{level}"] = float(
            interval_coverage(ens[:, h], truth[h], level / 100.0)[0].mean()
        )
    return {
        "energy_h20_full": float(energy_score(ens[:, h], truth[h])),
        "crps_h20_full": float(crps_ensemble(ens[:, h], truth[h]).mean()),
        **cov,
    }


def score_exact(ens: np.ndarray, truth: np.ndarray, pca: PCA) -> dict[str, float]:
    ens_proj = pca.transform(ens.reshape(-1, ens.shape[-1])).reshape(ens.shape[0], ens.shape[1], -1)
    truth_proj = pca.transform(truth)
    row = ensemble_metrics(ens_proj, truth_proj)
    row.update(full_neuron_metrics(ens, truth))
    return row


def score_common(ens: np.ndarray, truth: np.ndarray, pca: PCA) -> dict[str, float]:
    ens_proj = pca.transform(ens.reshape(-1, ens.shape[-1])).reshape(ens.shape[0], ens.shape[1], -1)
    truth_proj = pca.transform(truth)
    return ensemble_metrics(ens_proj, truth_proj)


def rollout_linear(model: FrozenLinearModel, z0: np.ndarray, eps: np.ndarray) -> np.ndarray:
    x = np.repeat(np.asarray(z0, dtype=np.float64)[None, :], eps.shape[0], axis=0)
    out = np.empty((eps.shape[0], eps.shape[1], x.shape[1]), dtype=np.float64)
    for step in range(eps.shape[1]):
        x = x @ model.A.T + model.b + eps[:, step]
        out[:, step] = x
    return out


def rollout_independent(model, x0: np.ndarray, eps: np.ndarray) -> np.ndarray:
    x = np.repeat(np.asarray(x0, dtype=np.float64)[None, :], eps.shape[0], axis=0)
    out = np.empty((eps.shape[0], eps.shape[1], x.shape[1]), dtype=np.float64)
    for step in range(eps.shape[1]):
        x = model.a * x + model.b + eps[:, step]
        out[:, step] = x
    return out


def rollout_persistence(x0: np.ndarray, eps: np.ndarray) -> np.ndarray:
    return np.asarray(x0, dtype=np.float64)[None, None, :] + np.cumsum(eps, axis=1)


def rollout_constant(mean: np.ndarray, eps: np.ndarray) -> np.ndarray:
    return np.asarray(mean, dtype=np.float64)[None, None, :] + eps


def fit_common_models(
    d,
    t: int,
    b3: FrozenLinearModel,
    b1,
    b2,
    common_eps: np.ndarray,
) -> dict[str, np.ndarray]:
    x0 = np.asarray(d.X[int(t)], dtype=np.float64)
    b4r = IndependentARModel(window=500).fit_at(d.X, int(t))
    out: dict[str, np.ndarray] = {
        "B0_persistence": rollout_persistence(x0, common_eps),
        "B1_independent_common": rollout_independent(b1, x0, common_eps),
        "B2_local_mean": rollout_constant(b2.value, common_eps),
        "B3_population": rollout_linear(b3, x0, common_eps),
        "B4_diagonal_population": rollout_linear(diagonal_copy(b3, d.X, t), x0, common_eps),
    }
    if b4r is not None:
        out["B4_refit_diagonal"] = rollout_independent(b4r, x0, common_eps)
    return out


def state_rows_for_models(
    *,
    session: int,
    landmark: int,
    pca: PCA,
    centers: np.ndarray,
    train_labels: np.ndarray,
    observed_labels: np.ndarray,
    ensembles: dict[str, np.ndarray],
    train_transition: np.ndarray,
    train_occ: np.ndarray,
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict]]:
    rows, matrices = [], []
    n = len(centers)
    for name, ens in ensembles.items():
        proj = pca.transform(ens.reshape(-1, ens.shape[-1])).reshape(ens.shape[0], ens.shape[1], -1)
        labels = assign_states(proj, centers)
        metrics, detail = state_metrics(
            labels,
            observed_labels,
            train_labels=train_labels,
            n_states=n,
        )
        rows.append({"session": session, "landmark": landmark, "model": name, **metrics})
        for a in range(n):
            for b in range(n):
                matrices.append(
                    {
                        "session": session,
                        "landmark": landmark,
                        "model": name,
                        "from_state": a,
                        "to_state": b,
                        "probability": float(detail["model_transition"][a, b]),
                        "observed_probability": float(detail["observed_transition"][a, b]),
                    }
                )
    # Markov null uses only the training transition matrix.
    markov = simulate_markov(
        train_transition,
        n_members=next(iter(ensembles.values())).shape[0],
        horizon=HORIZON,
        initial=np.repeat(observed_labels[0], next(iter(ensembles.values())).shape[0]),
        rng=rng,
    )
    metrics, _ = state_metrics(markov, observed_labels, train_labels=train_labels, n_states=n)
    rows.append({"session": session, "landmark": landmark, "model": "markov_null", **metrics})
    marginal = np.repeat(observed_labels[0], markov.shape[0])
    marginal_paths = rng.choice(n, size=markov.shape, p=train_occ)
    marginal_paths[:, 0] = marginal
    metrics, _ = state_metrics(marginal_paths, observed_labels, train_labels=train_labels, n_states=n)
    rows.append({"session": session, "landmark": landmark, "model": "marginal_null", **metrics})
    return rows, matrices


def select_state_landmarks(
    pca: PCA,
    centers: np.ndarray,
    train_projected: np.ndarray,
    landmarks: np.ndarray,
    x_test: np.ndarray,
    *,
    n_select: int = 3,
) -> list[int]:
    labels = assign_states(train_projected, centers)
    occ = occupancy(labels, len(centers))
    order = np.argsort(occ)[::-1]
    selected: list[int] = []
    test_proj = pca.transform(x_test)
    for state in order:
        dist = ((test_proj - centers[int(state)][None, :]) ** 2).sum(axis=1)
        for idx in np.argsort(dist):
            value = int(landmarks[int(idx)])
            if value not in selected:
                selected.append(value)
                break
        if len(selected) >= n_select:
            break
    if len(selected) < n_select:
        for value in landmarks[:: max(1, len(landmarks) // n_select)]:
            if int(value) not in selected:
                selected.append(int(value))
            if len(selected) >= n_select:
                break
    return selected[:n_select]


def operator_fit(
    d,
    *,
    fit_id: str,
    kind: str,
    segment: int,
    A_override: FrozenLinearModel | None = None,
) -> tuple[dict, pd.DataFrame]:
    if A_override is None:
        starts = split_starts(d, "train", history=1, horizon=1)
        if kind == "bootstrap":
            starts = block_bootstrap_starts(starts, block=25, rng=np.random.default_rng(10_000 + segment))
        elif kind == "segment":
            parts = np.array_split(starts, 3)
            starts = parts[int(segment)]
        model = fit_var(d.X, starts, ridge=1.0, spectral_cap=0.995)
        if model is None:
            raise RuntimeError(f"operator fit failed: {fit_id}")
    else:
        model = A_override
    row = {
        "session": int(d.worm_id),
        "fit_id": fit_id,
        "kind": kind,
        "segment": int(segment),
        "n_transitions": int(model.fit_record.get("n_transitions", 0)),
        "spectral_radius": model.spectral_radius,
        "leading_decay_frames": leading_decay_frames(model.A),
        "offdiagonal_energy": model.offdiagonal_energy,
        "diagonal_energy": model.diagonal_energy,
        **operator_summary(model.A, fps=float(d.recording.fps)),
    }
    modes = mode_table(model.A, fps=float(d.recording.fps))
    modes.insert(0, "fit_id", fit_id)
    modes.insert(0, "session", int(d.worm_id))
    return row, modes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surrogate-draws", type=int, default=SURROGATE_DRAWS)
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    cfg, data, recordings = load_nostim()

    cross_rows: list[dict] = []
    exact_rows: list[dict] = []
    surrogate_draws: list[dict] = []
    surrogate_diag: list[dict] = []
    state_rows: list[dict] = []
    state_matrices: list[dict] = []
    operator_rows: list[dict] = []
    mode_rows: list[pd.DataFrame] = []
    local_rows: list[dict] = []
    presence_rows: list[dict] = []
    selected_rows: list[dict] = []
    sensitivity_rows: list[dict] = []
    sensitivity_ident: list[dict] = []
    b3_models: dict[tuple[int, int], FrozenLinearModel] = {}
    b3_scores: dict[tuple[int, int], dict] = {}
    session_bundle: dict[int, dict] = {}
    cache_path = OUT / "_stage_a_cache.pkl"
    cached = None
    if args.resume and cache_path.exists():
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
        cross_rows = cached["cross_rows"]
        exact_rows = cached["exact_rows"]
        surrogate_draws = cached["surrogate_draws"]
        surrogate_diag = cached["surrogate_diag"]
        state_rows = cached["state_rows"]
        state_matrices = cached["state_matrices"]
        operator_rows = cached["operator_rows"]
        mode_rows = cached["mode_rows"]
        local_rows = cached["local_rows"]
        presence_rows = cached["presence_rows"]
        selected_rows = cached["selected_rows"]
        b3_models = cached["b3_models"]
        b3_scores = cached["b3_scores"]
        session_bundle = cached["session_bundle"]
        print(f"resumed stage A cache with {len(exact_rows)} rows", flush=True)
    session_data = [] if cached is not None else data

    for d in session_data:
        sid = int(d.worm_id)
        pca, series, meta = JD.prepare(d)
        train = d.X[d.masks["train"]]
        train_proj = pca.transform(train)
        centers = fit_states(train_proj, n_states=N_STATES).cluster_centers_
        train_labels = assign_states(train_proj, centers)
        train_counts = transition_counts(train_labels, N_STATES)
        train_T = row_normalize(train_counts)
        train_occ = occupancy(train_labels, N_STATES)
        landmarks = evaluation_landmarks(d, horizon_max=HORIZON, stride=STRIDE)
        cache: dict = {}
        temporal_cache: dict = {}
        named = {name: (d.predictor_names.index(name) if name in d.predictor_names else None) for name in TARGETS}
        for name, idx in named.items():
            presence_rows.append(
                {
                    "session": sid,
                    "neuron": name,
                    "present": bool(idx is not None),
                    "index": int(idx) if idx is not None else -1,
                }
            )
        selected = select_state_landmarks(pca, centers, train_proj, landmarks, d.X[landmarks])
        for value in selected:
            selected_rows.append(
                {"session": sid, "landmark": int(value), "rule": "top-training-state-centroid, first test match"}
            )

        for t in landmarks:
            t = int(t)
            x0 = d.X[t]
            truth = d.X[t + 1 : t + HORIZON + 1]
            truth_proj = pca.transform(truth)
            exact_ensembles: dict[str, np.ndarray] = {}
            exact_eps: dict[str, np.ndarray] = {}
            for spec in (
                JD.Spec("J0_persistence", "persistence", "persistence", "block", 20, False),
                JD.Spec("J1_independent_diag", "independent_ar", "independent_ar", "diag"),
                JD.Spec("J2_population_diag", "var", "population", "diag"),
                JD.Spec("J5_population_lead1_boot", "var", "population", "block", 10),
            ):
                fitted = JD.fit_mean(spec, t, cache, d.X)
                innovation = JD.make_innovation(spec, series[spec.series], t, temporal_cache)
                eps = innovation.sample(MEMBERS, HORIZON, np.random.default_rng(7000 + sid * 10000 + t))
                ens = JD.rollout(spec.mean, fitted, x0, HORIZON, eps)
                exact_ensembles[spec.name] = ens
                exact_eps[spec.name] = eps
                row = {
                    "session": sid,
                    "worm_id": sid,
                    "landmark": t,
                    "model": spec.name,
                    **score_exact(ens, truth, pca),
                }
                exact_rows.append(row)

            b3 = cache.get(("var", t))
            if b3 is None:
                b3 = fit_window(d.X, t, window=500)
            if b3 is None:
                continue
            b3_linear = FrozenLinearModel(A=b3.A, b=b3.b, fit_record=dict(b3.fit_record))
            b3_models[(sid, t)] = b3_linear
            b3_score = score_common(exact_ensembles["J5_population_lead1_boot"], truth, pca)
            b3_scores[(sid, t)] = b3_score

            b1 = cache.get(("independent_ar", t))
            b2 = LocalMeanModel(window=500).fit_at(d.X, t)
            common_eps = exact_eps["J5_population_lead1_boot"]

            cross_ens = fit_common_models(d, t, b3, b1, b2, common_eps)
            cross_ens["B1_independent_exact"] = exact_ensembles["J1_independent_diag"]
            cross_ens["B5_circular"] = np.empty_like(common_eps)
            cross_ens["B5_block"] = np.empty_like(common_eps)
            for key, ens in cross_ens.items():
                if key in {"B5_circular", "B5_block"}:
                    continue
                metrics = score_common(ens, truth, pca)
                row = {
                    "session": sid,
                    "landmark": t,
                    "model": key,
                    **metrics,
                    "spectral_radius": float(b3.spectral_radius) if key.startswith("B3") else float("nan"),
                    "offdiagonal_energy": float(
                        np.linalg.norm(b3.A - np.diag(np.diag(b3.A)), "fro")
                    )
                    if key.startswith("B3")
                    else float("nan"),
                }
                cross_rows.append(row)

            # Matched temporal surrogates, fitted only from the causal window.
            for method in ("circular", "block"):
                method_scores = []
                for draw in range(int(args.surrogate_draws)):
                    seed = 90_000 + sid * 1000 + t * 7 + draw
                    sur = fit_shuffled_window(
                        d.X,
                        t,
                        window=500,
                        method=method,
                        block=25,
                        seed=seed,
                    )
                    if sur is None:
                        continue
                    ens = rollout_linear(sur, np.asarray(d.X[t], dtype=np.float64), common_eps)
                    metrics = score_common(ens, truth, pca)
                    method_scores.append(metrics)
                    surrogate_draws.append(
                        {
                            "session": sid,
                            "landmark": t,
                            "method": method,
                            "draw": draw,
                            "seed": seed,
                            **metrics,
                            "offdiagonal_energy": sur.offdiagonal_energy,
                            "spectral_radius": sur.spectral_radius,
                        }
                    )
                    window = d.X[max(1, t - 499) : t + 1].astype(np.float64)
                    shifted, _ = shuffle_window(window, method=method, block=25, seed=seed)
                    original_acf = np.mean(
                        [np.corrcoef(window[:-1, j], window[1:, j])[0, 1] for j in range(window.shape[1])]
                    )
                    shifted_acf = np.mean(
                        [np.corrcoef(shifted[:-1, j], shifted[1:, j])[0, 1] for j in range(window.shape[1])]
                    )
                    surrogate_diag.append(
                        {
                            "session": sid,
                            "landmark": t,
                            "method": method,
                            "draw": draw,
                            "diag_ar_corr": float(
                                np.corrcoef(np.diag(sur.A), np.diag(b3.A))[0, 1]
                            ),
                            "window_acf1_original": float(original_acf),
                            "window_acf1_surrogate": float(shifted_acf),
                            "window_acf1_abs_error": float(abs(original_acf - shifted_acf)),
                        }
                    )
                if method_scores:
                    key = f"B5_{method}"
                    aggregate = {
                        metric: float(np.median([row[metric] for row in method_scores]))
                        for metric in method_scores[0]
                    }
                    cross_ens[key] = np.zeros_like(common_eps)
                    cross_rows.append(
                        {
                            "session": sid,
                            "landmark": t,
                            "model": key,
                            **aggregate,
                            "spectral_radius": float("nan"),
                            "offdiagonal_energy": float("nan"),
                            "surrogate_draws": int(len(method_scores)),
                        }
                    )

            # State metrics for the primary mean models and two nulls.
            obs_labels = assign_states(truth_proj, centers)
            state_ensembles = {
                "B0_persistence": cross_ens["B0_persistence"],
                "B1_independent_common": cross_ens["B1_independent_common"],
                "B3_population": cross_ens["B3_population"],
                "B4_diagonal_population": cross_ens["B4_diagonal_population"],
            }
            rows, matrices = state_rows_for_models(
                session=sid,
                landmark=t,
                pca=pca,
                centers=centers,
                train_labels=train_labels,
                observed_labels=obs_labels,
                ensembles=state_ensembles,
                train_transition=train_T,
                train_occ=train_occ,
                rng=np.random.default_rng(70_000 + sid * 10_000 + t),
            )
            state_rows.extend(rows)
            state_matrices.extend(matrices)

            local_rows.append(
                {
                    "session": sid,
                    "landmark": t,
                    **operator_summary(b3.A, fps=float(d.recording.fps)),
                    "prediction_mse_h20": b3_score["mse_mean_h20"],
                    "prediction_energy_h20": b3_score["energy_h20"],
                    "offdiagonal_energy": float(np.linalg.norm(b3.A - np.diag(np.diag(b3.A)), "fro")),
                }
            )

        # Frozen and resampled operators for this session.
        for fit_id, kind, segment in (("frozen_train", "frozen", 0),):
            row, modes = operator_fit(d, fit_id=fit_id, kind=kind, segment=segment)
            operator_rows.append(row)
            mode_rows.append(modes)
        for segment in range(3):
            row, modes = operator_fit(d, fit_id=f"train_segment_{segment}", kind="segment", segment=segment)
            operator_rows.append(row)
            mode_rows.append(modes)
        for draw in range(5):
            row, modes = operator_fit(d, fit_id=f"bootstrap_{draw}", kind="bootstrap", segment=draw)
            operator_rows.append(row)
            mode_rows.append(modes)
        session_bundle[sid] = {
            "pca": pca,
            "centers": centers,
            "train_proj": train_proj,
            "train_labels": train_labels,
            "train_T": train_T,
            "train_occ": train_occ,
            "landmarks": landmarks,
            "selected": selected,
            "named": named,
            "train": train,
            "fps": float(d.recording.fps),
        }
        with cache_path.open("wb") as handle:
            pickle.dump(
                {
                    "cross_rows": cross_rows,
                    "exact_rows": exact_rows,
                    "surrogate_draws": surrogate_draws,
                    "surrogate_diag": surrogate_diag,
                    "state_rows": state_rows,
                    "state_matrices": state_matrices,
                    "operator_rows": operator_rows,
                    "mode_rows": mode_rows,
                    "local_rows": local_rows,
                    "presence_rows": presence_rows,
                    "selected_rows": selected_rows,
                    "b3_models": b3_models,
                    "b3_scores": b3_scores,
                    "session_bundle": session_bundle,
                },
                handle,
            )
        print(f"session {sid} complete {perf_counter() - started:.1f}s", flush=True)

    exact = pd.DataFrame(exact_rows)
    cross = pd.DataFrame(cross_rows)
    cross["worm_id"] = cross["session"]
    exact.to_csv(OUT / "exact_reproduction_per_landmark.csv", index=False)
    cross.to_csv(OUT / "cross_neuron_per_landmark.csv", index=False)
    pd.DataFrame(surrogate_draws).to_csv(OUT / "surrogate_draws.csv", index=False)
    pd.DataFrame(surrogate_diag).to_csv(OUT / "surrogate_diagnostics.csv", index=False)
    pd.DataFrame(state_rows).to_csv(OUT / "state_transition_scores.csv", index=False)
    pd.DataFrame(state_matrices).to_csv(OUT / "state_transition_probabilities.csv", index=False)
    pd.DataFrame(operator_rows).to_csv(OUT / "operator_fits.csv", index=False)
    pd.concat(mode_rows, ignore_index=True).to_csv(OUT / "operator_modes.csv", index=False)
    pd.DataFrame(local_rows).to_csv(OUT / "local_operator.csv", index=False)
    pd.DataFrame(presence_rows).to_csv(OUT / "named_neuron_presence.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(OUT / "selected_sensitivity_landmarks.csv", index=False)

    # Exact reproduction against the frozen per-landmark table.
    frozen = pd.read_csv(FROZEN / "scores_by_landmark.csv")
    metrics = [
        "energy_h20",
        "trajectory_energy_h20",
        "coverage_h20_full_50",
        "coverage_h20_full_80",
        "coverage_h20_full_95",
        "neural_cov_error",
        "autocorr_error",
    ]
    reproduce_rows = []
    for model in ("J0_persistence", "J1_independent_diag", "J2_population_diag", "J5_population_lead1_boot"):
        left = exact[exact.model == model]
        right = frozen[frozen.model == model]
        merged = left.merge(right, on=["worm_id", "landmark"], suffixes=("_new", "_frozen"))
        for metric in metrics:
            if f"{metric}_new" not in merged or f"{metric}_frozen" not in merged:
                continue
            diff = (merged[f"{metric}_new"] - merged[f"{metric}_frozen"]).abs()
            reproduce_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "n": int(diff.notna().sum()),
                    "max_abs_diff": float(diff.max()),
                    "mean_abs_diff": float(diff.mean()),
                }
            )
    reproduction = pd.DataFrame(reproduce_rows)
    reproduction.to_csv(OUT / "baseline_reproduction.csv", index=False)
    max_diff = float(reproduction["max_abs_diff"].max())
    j2 = exact[exact.model == "J2_population_diag"].set_index(["worm_id", "landmark"])
    j5 = exact[exact.model == "J5_population_lead1_boot"].set_index(["worm_id", "landmark"])
    j2j5_signed = j2["energy_h20"] - j5["energy_h20"]
    j2j5_traj_signed = j2["trajectory_energy_h20"] - j5["trajectory_energy_h20"]
    j2j5 = j2j5_signed.abs()
    j2j5_traj = j2j5_traj_signed.abs()
    j2j5_frame = pd.DataFrame(
        {
            "worm_id": j2.index.get_level_values("worm_id"),
            "landmark": j2.index.get_level_values("landmark"),
            "improvement": j2j5_signed.to_numpy(),
        }
    )
    j2j5_ci = session_block_ci(j2j5_frame, "improvement", block=5, n_boot=1000)
    g1 = {
        "status": "PASS" if max_diff < 1e-6 else "FAIL",
        "max_abs_reproduction_difference": max_diff,
        "tolerance": 1e-6,
        "j2_j5_max_abs_energy_difference": float(j2j5.max()),
        "j2_j5_mean_signed_energy_difference": float(j2j5_signed.mean()),
        "j2_j5_mean_abs_energy_difference": float(j2j5.mean()),
        "j2_j5_mean_signed_trajectory_difference": float(j2j5_traj_signed.mean()),
        "j2_j5_mean_abs_trajectory_difference": float(j2j5_traj.mean()),
        "j2_j5_energy_ci_low": j2j5_ci["lo"],
        "j2_j5_energy_ci_high": j2j5_ci["hi"],
        "j2_j5_energy_correlation": float(np.corrcoef(j2["energy_h20"], j5["energy_h20"])[0, 1]),
        "j2_j5_equivalent_mean": bool(abs(float(j2j5_signed.mean())) < 0.01),
        "frozen_rows": int(len(frozen)),
        "rerun_rows": int(len(exact)),
    }
    write_json(OUT / "reproduction_summary.json", g1)

    # Cross-neuron means and paired tests.
    summary = cross.groupby("model").mean(numeric_only=True).reset_index()
    summary.to_csv(OUT / "cross_neuron_summary.csv", index=False)
    paired_rows = []
    for metric in ("energy_h20", "trajectory_energy_h20", "mse_mean_h20"):
        paired_rows.append(
            paired_table(
                cross,
                reference="B1_independent_common",
                models=["B3_population", "B4_diagonal_population", "B5_circular", "B5_block", "B2_local_mean"],
                metric=metric,
                blocks=(3, 5, 10),
                n_boot=1000,
            )
        )
        paired_rows.append(
            paired_table(
                cross,
                reference="B4_diagonal_population",
                models=["B3_population", "B5_circular"],
                metric=metric,
                blocks=(5, 10),
                n_boot=1000,
            )
        )
        paired_rows.append(
            paired_table(
                cross,
                reference="B4_refit_diagonal",
                models=["B3_population"],
                metric=metric,
                blocks=(5, 10),
                n_boot=1000,
            )
        )
        paired_rows.append(
            paired_table(
                cross,
                reference="B5_circular",
                models=["B3_population"],
                metric=metric,
                blocks=(5, 10),
                n_boot=1000,
            )
        )
    paired = pd.concat(paired_rows, ignore_index=True)
    paired.to_csv(OUT / "cross_neuron_paired.csv", index=False)

    # Session means for the same paired contrasts.
    session_rows = []
    for session, group in cross.groupby("session"):
        pivot = group.pivot_table(index="landmark", columns="model", values="energy_h20")
        for model in ("B3_population", "B4_diagonal_population", "B5_circular", "B5_block", "B2_local_mean"):
            if model not in pivot or "B1_independent_common" not in pivot:
                continue
            diff = (pivot["B1_independent_common"] - pivot[model]).dropna()
            session_rows.append(
                {
                    "session": int(session),
                    "model": model,
                    "reference": "B1_independent_common",
                    "mean_energy_improvement": float(diff.mean()),
                    "n_landmarks": int(len(diff)),
                }
            )
    pd.DataFrame(session_rows).to_csv(OUT / "cross_neuron_session.csv", index=False)

    surrogate = pd.DataFrame(surrogate_draws)
    if not surrogate.empty:
        surrogate.groupby(["method", "draw"]).mean(numeric_only=True).reset_index().to_csv(
            OUT / "surrogate_by_draw.csv", index=False
        )
        surrogate.groupby("method").agg(
            energy_mean=("energy_h20", "mean"),
            energy_median=("energy_h20", "median"),
            energy_q05=("energy_h20", lambda x: float(np.quantile(x, 0.05))),
            energy_q95=("energy_h20", lambda x: float(np.quantile(x, 0.95))),
            trajectory_mean=("trajectory_energy_h20", "mean"),
            mse_mean=("mse_mean_h20", "mean"),
            n=("energy_h20", "size"),
        ).reset_index().to_csv(OUT / "surrogate_summary.csv", index=False)

    # Operator stability within each session.
    op = pd.DataFrame(operator_rows)
    stability_rows = []
    for session, group in op.groupby("session"):
        for kind in ("segment", "bootstrap"):
            part = group[group.kind == kind]
            if len(part) < 2:
                continue
            for metric in ("spectral_radius", "leading_decay_frames", "nonnormality", "transient_peak"):
                values = part[metric].to_numpy(dtype=np.float64)
                stability_rows.append(
                    {
                        "session": int(session),
                        "kind": kind,
                        "metric": metric,
                        "n": int(len(values)),
                        "mean": float(np.nanmean(values)),
                        "sd": float(np.nanstd(values, ddof=1)),
                        "cv": float(np.nanstd(values, ddof=1) / max(abs(np.nanmean(values)), 1e-12)),
                        "min": float(np.nanmin(values)),
                        "max": float(np.nanmax(values)),
                    }
                )
    # Pairwise operator agreement on the same session coordinates is computed
    # below by refitting the three chronological segments.
    matrix_rows = []
    for d in data:
        starts = split_starts(d, "train", history=1, horizon=1)
        parts = np.array_split(starts, 3)
        fits = []
        for segment, part in enumerate(parts):
            model = fit_var(d.X, part, ridge=1.0, spectral_cap=0.995)
            if model is not None:
                fits.append((segment, model))
        for i in range(len(fits)):
            for j in range(i + 1, len(fits)):
                a, b = fits[i][1].A, fits[j][1].A
                mask = ~np.eye(a.shape[0], dtype=bool)
                corr = (
                    float(np.corrcoef(a[mask], b[mask])[0, 1])
                    if np.std(a[mask]) > 0 and np.std(b[mask]) > 0
                    else float("nan")
                )
                matrix_rows.append(
                    {
                        "session": int(d.worm_id),
                        "fit_a": fits[i][0],
                        "fit_b": fits[j][0],
                        "offdiag_corr": corr,
                        "relative_frobenius": float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12)),
                    }
                )
    pd.DataFrame(stability_rows).to_csv(OUT / "operator_stability.csv", index=False)
    pd.DataFrame(matrix_rows).to_csv(OUT / "operator_pairwise.csv", index=False)

    # Oscillation audit against observed training spectra.
    osc_rows = []
    for d in data:
        sid = int(d.worm_id)
        train = d.X[d.masks["train"]]
        freqs = np.fft.rfftfreq(len(train), d=1.0 / float(d.recording.fps))
        power = np.abs(np.fft.rfft(train - train.mean(axis=0), axis=0)) ** 2
        dom = freqs[np.argmax(power[1:], axis=0) + 1]
        observed_period = float(1.0 / max(np.median(dom), 1e-9))
        frozen_rows = op[(op.session == sid) & (op.kind == "frozen")]
        if frozen_rows.empty:
            continue
        row = frozen_rows.iloc[0]
        modes = mode_table(
            fit_var(d.X, split_starts(d, "train", history=1, horizon=1), ridge=1.0, spectral_cap=0.995).A,
            fps=float(d.recording.fps),
        )
        pair = complex_pair_stability(modes)
        osc_rows.append(
            {
                "session": sid,
                "observed_median_dominant_period_s": observed_period,
                "complex_pair_count": pair["n_positive_imaginary"],
                "mode_period_s": float(pair["median_period_frames"] / float(d.recording.fps))
                if np.isfinite(pair["median_period_frames"])
                else float("nan"),
                "mode_decay_frames": pair["median_decay_frames"],
                "mode_max_magnitude": pair["max_magnitude"],
                "spectral_radius": float(row["spectral_radius"]),
                "nonnormality": float(row["nonnormality"]),
                "transient_peak": float(row["transient_peak"]),
            }
        )
    pd.DataFrame(osc_rows).to_csv(OUT / "oscillation_audit.csv", index=False)

    # State summaries.
    states = pd.DataFrame(state_rows)
    if not states.empty:
        state_summary = (
            states.drop(columns=["session", "landmark"], errors="ignore")
            .groupby("model")
            .mean(numeric_only=True)
            .reset_index()
        )
        state_summary.to_csv(OUT / "state_transition_summary.csv", index=False)

    # Sensitivity at fixed, prespecified landmarks and amplitudes.
    fit_cache: dict[int, dict[str, FrozenLinearModel]] = {}
    for d in data:
        sid = int(d.worm_id)
        starts = split_starts(d, "train", history=1, horizon=1)
        fits: dict[str, FrozenLinearModel] = {"frozen_train": fit_var(d.X, starts, ridge=1.0, spectral_cap=0.995)}
        parts = np.array_split(starts, 3)
        for segment, part in enumerate(parts):
            model = fit_var(d.X, part, ridge=1.0, spectral_cap=0.995)
            if model is not None:
                fits[f"segment_{segment}"] = model
        for draw in range(5):
            resampled = block_bootstrap_starts(starts, block=25, rng=np.random.default_rng(20_000 + sid * 100 + draw))
            model = fit_var(d.X, resampled, ridge=1.0, spectral_cap=0.995)
            if model is not None:
                fits[f"bootstrap_{draw}"] = model
        fit_cache[sid] = fits

    for d in data:
        sid = int(d.worm_id)
        bundle = session_bundle[sid]
        pca = bundle["pca"]
        selected = bundle["selected"]
        named = bundle["named"]
        train = bundle["train"]
        present_targets = [name for name in TARGETS if named[name] is not None]
        target_indices = {name: named[name] for name in present_targets}
        exclude = {int(v) for v in target_indices.values() if v is not None}
        controls = {
            name: matched_control_indices(bundle["train"].var(axis=0), target_indices[name], exclude=exclude)
            for name in present_targets
        }
        for t in selected:
            t = int(t)
            x0 = np.asarray(d.X[t], dtype=np.float64)
            local = b3_models.get((sid, t))
            if local is None:
                continue
            fits = {"rolling": local, **fit_cache[sid]}
            for name in present_targets:
                target_idx = int(target_indices[name])
                for role, idx in [("target", target_idx)] + [
                    ("control", int(control)) for control in controls[name]
                ]:
                    baselines: dict[str, np.ndarray] = {}
                    responses: dict[str, np.ndarray] = {}
                    for fit_id, model in fits.items():
                        metrics, neuron, latent = response_metrics(
                            model,
                            pca,
                            x0,
                            idx,
                            amplitude=1.0,
                            horizon=PRIMARY_HORIZON,
                        )
                        base = model.forecast_mean(x0, PRIMARY_HORIZON)
                        baselines[fit_id] = base[-1]
                        responses[fit_id] = latent[-1]
                        pert = base + neuron
                        support = support_fraction(pert, train_reference=train)
                        sensitivity_rows.append(
                            {
                                "session": sid,
                                "landmark": t,
                                "target": name,
                                "role": role,
                                "control_of": name if role == "control" else "",
                                "neuron_index": idx,
                                "neuron_name": d.predictor_names[idx],
                                "fit_id": fit_id,
                                **metrics,
                                "ood_fraction": support,
                            }
                        )
                    disagreement = pairwise_disagreement(baselines, responses)
                    sensitivity_ident.append(
                        {
                            "session": sid,
                            "landmark": t,
                            "target": name,
                            "role": role,
                            "n_controls": len(controls[name]),
                            **disagreement,
                            "prediction_similar_response_disagrees": bool(
                                disagreement["prediction_relative_spread"] < 0.05
                                and disagreement["mean_response_cosine"] < 0.5
                            ),
                        }
                    )
        print(f"sensitivity session {sid} complete {perf_counter() - started:.1f}s", flush=True)

    sens = pd.DataFrame(sensitivity_rows)
    sens.to_csv(OUT / "sensitivity_responses.csv", index=False)
    ident = pd.DataFrame(sensitivity_ident)
    ident.to_csv(OUT / "sensitivity_identifiability.csv", index=False)
    if not sens.empty:
        sens_summary = sens.groupby(["target", "role"]).agg(
            response_mean=("latent_response_h20", "mean"),
            response_median=("latent_response_h20", "median"),
            relative_mean=("latent_relative_h20", "mean"),
            ood_median=("ood_fraction", "median"),
            ood_max=("ood_fraction", "max"),
            finite_difference_max=("finite_difference_error", "max"),
            n=("latent_response_h20", "size"),
        ).reset_index()
        sens_summary.to_csv(OUT / "sensitivity_summary.csv", index=False)
    if not ident.empty:
        ident_summary = ident.groupby(["target", "role"]).agg(
            mean_response_cosine=("mean_response_cosine", "median"),
            min_response_cosine=("min_response_cosine", "min"),
            prediction_relative_spread=("prediction_relative_spread", "median"),
            sign_agreement=("sign_agreement", "median"),
            n=("mean_response_cosine", "size"),
        ).reset_index()
        ident_summary.to_csv(OUT / "sensitivity_identifiability_summary.csv", index=False)

    synthetic = pd.DataFrame() if args.skip_synthetic else synthetic_audit()
    if not synthetic.empty:
        synthetic.to_csv(OUT / "synthetic_audit.csv", index=False)
        synthetic.groupby("scenario").mean(numeric_only=True).drop(columns=["seed"], errors="ignore").reset_index().to_csv(
            OUT / "synthetic_summary.csv", index=False
        )

    # Gates use fixed thresholds declared in the protocol.
    paired5 = paired[(paired.block == 5)]
    def get_pair(model: str, metric: str, reference: str) -> dict:
        part = paired5[
            (paired5.model == model)
            & (paired5.metric == metric)
            & (paired5.reference == reference)
        ]
        return part.iloc[0].to_dict() if len(part) else {}
    b3_b1 = get_pair("B3_population", "energy_h20", "B1_independent_common")
    b3_b4 = get_pair("B3_population", "energy_h20", "B4_refit_diagonal")
    b3_sur = get_pair("B3_population", "energy_h20", "B5_circular")
    gate2 = bool(
        b3_b1.get("ci_low", -1) > 0
        and b3_b4.get("ci_low", -1) > 0
        and b3_sur.get("ci_low", -1) > 0
    )
    gate2_partial = bool(b3_b1.get("ci_low", -1) > 0)

    stability = pd.DataFrame(stability_rows)
    seg = stability[stability.kind == "segment"]
    def median_cv(metric: str) -> float:
        part = seg[seg.metric == metric]
        return float(part.cv.median()) if len(part) else float("nan")
    pairwise = pd.DataFrame(matrix_rows)
    gate3 = bool(
        median_cv("spectral_radius") < 0.25
        and median_cv("leading_decay_frames") < 0.35
        and median_cv("nonnormality") < 0.50
        and (pairwise.offdiag_corr.median() > 0.50 if len(pairwise) else False)
    )

    state_summary = states.groupby("model").mean(numeric_only=True).reset_index() if not states.empty else pd.DataFrame()
    def state_val(model: str, col: str) -> float:
        part = state_summary[state_summary.model == model]
        return float(part[col].iloc[0]) if len(part) else float("nan")
    gate4 = bool(
        state_val("B3_population", "transition_tv_observed")
        < min(state_val("markov_null", "transition_tv_observed"), state_val("marginal_null", "transition_tv_observed")) - 0.02
    )

    if not ident.empty:
        gate5 = bool(
            ident[ident.role == "target"].mean_response_cosine.median() > 0.80
            and ident[ident.role == "target"].sign_agreement.median() > 0.80
        )
        ood_median = float(sens[sens.role == "target"].ood_fraction.median()) if not sens.empty else float("nan")
        gate5_value = {
            "median_response_cosine": float(ident[ident.role == "target"].mean_response_cosine.median()),
            "median_sign_agreement": float(ident[ident.role == "target"].sign_agreement.median()),
            "n": int(len(ident[ident.role == "target"])),
        }
    else:
        gate5 = False
        ood_median = float("nan")
        gate5_value = {}
    gate6 = bool(np.isfinite(ood_median) and ood_median < 0.05)
    joint_gates = json.loads((FROZEN / "gates.json").read_text(encoding="utf-8"))
    g4_key = next((key for key in joint_gates if key.startswith("G4")), "")
    g5_key = next((key for key in joint_gates if key.startswith("G5")), "")
    gate7 = bool(
        g4_key
        and g5_key
        and str(joint_gates.get(g4_key, {}).get("status", "")).upper() == "PASS"
        and str(joint_gates.get(g5_key, {}).get("status", "")).upper() == "PASS"
    )

    gates = {
        "G1_reproduction": {"status": g1["status"], **g1},
        "G2_cross_neuron_information": {
            "status": "PASS" if gate2 else ("PARTIAL" if gate2_partial else "FAIL"),
            "b3_vs_b1_energy": b3_b1,
            "b3_vs_b4r_energy": b3_b4,
            "b3_vs_circular_surrogate_energy": b3_sur,
        },
        "G3_operator_reliability": {
            "status": "PASS" if gate3 else "FAIL",
            "median_segment_cv_spectral_radius": median_cv("spectral_radius"),
            "median_segment_cv_leading_decay": median_cv("leading_decay_frames"),
            "median_segment_cv_nonnormality": median_cv("nonnormality"),
            "median_pairwise_offdiag_corr": float(pairwise.offdiag_corr.median()) if len(pairwise) else float("nan"),
        },
        "G4_state_transition_fidelity": {
            "status": "PASS" if gate4 else "FAIL",
            "b3_transition_tv": state_val("B3_population", "transition_tv_observed"),
            "markov_transition_tv": state_val("markov_null", "transition_tv_observed"),
            "marginal_transition_tv": state_val("marginal_null", "transition_tv_observed"),
        },
        "G5_sensitivity_reliability": {"status": "PASS" if gate5 else "FAIL", **gate5_value},
        "G6_empirical_support": {
            "status": "PASS" if gate6 else "FAIL",
            "median_target_ood_fraction": ood_median,
            "threshold": 0.05,
        },
        "G7_generative_validity": {
            "status": "PASS" if gate7 else "FAIL",
            "joint_calibration_gate": joint_gates.get(g4_key, {}).get("status"),
            "joint_trajectory_gate": joint_gates.get(g5_key, {}).get("status"),
        },
        "G8_biological_validation": {"status": "NOT_EVALUABLE", "reason": "no matched Kato intervention recording"},
    }
    write_json(OUT / "gates.json", gates)
    write_json(
        OUT / "analysis_summary.json",
        {
            "dataset": "Kato 2015 WT_NoStim only",
            "sessions": len(data),
            "frames": int(sum(d.n_frames for d in data)),
            "landmarks": int(len(cross[["session", "landmark"]].drop_duplicates())),
            "members": MEMBERS,
            "horizon": PRIMARY_HORIZON,
            "surrogate_draws": int(args.surrogate_draws),
            "elapsed_seconds": float(perf_counter() - started),
            "targets": list(TARGETS),
            "amplitudes_sd": list(AMPLITUDES),
        },
    )
    print("kato interactions complete", perf_counter() - started, flush=True)


if __name__ == "__main__":
    main()
