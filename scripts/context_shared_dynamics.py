"""Stage E: session-disjoint shared latent dynamics on a verified common panel.

Only alphabetic, explicitly named neurons shared by all five WT_NoStim recordings
are used. Numeric trace IDs are worm-specific and are never treated as common
identities. The shared transition is fitted on four sessions; the fifth session's
observation map is fitted on a calibration prefix of its training block. This is
session-disjoint, not animal-independent (the five recordings are not five
independently identified animals).
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from neurobranch.config import PROJECT_ROOT, ExperimentConfig, git_state
from neurobranch.dynamics.context_models import SharedLatentModel, StableLatentLDS
from neurobranch.dynamics.models import BehaviorProbe
from neurobranch.dynamics.stability import stabilize_matrix
from neurobranch.dynamics.training import load_worm_data
from neurobranch.preprocessing import WormData, split_starts
from neurobranch.simulator.adapters import NeuralSimulator
from neurobranch.simulator.metrics import rollout_metrics

DIM = 5
RIDGE = 1e-2
CAP = 0.995


def _common_named(datasets: list[WormData]) -> tuple[str, ...]:
    sets = [set(n for n in d.predictor_names if n.isalpha()) for d in datasets]
    common = sorted(set.intersection(*sets))
    if len(common) < 10:
        raise RuntimeError(f"only {len(common)} common named neurons; refusing a shared-panel claim")
    return tuple(common)


def _panel(data: WormData, names: tuple[str, ...]) -> WormData:
    idx = [data.predictor_names.index(n) for n in names]
    out = copy.copy(data)
    out.X = data.X[:, idx].copy()
    out.X_raw = data.X_raw[:, idx].copy()
    out.mean = data.mean[idx].copy()
    out.std = data.std[idx].copy()
    out.predictor_names = tuple(names)
    return out


def _canonical_pca(pca: PCA) -> np.ndarray:
    D = np.asarray(pca.components_, dtype=np.float64).T.copy()
    for j in range(D.shape[1]):
        k = int(np.argmax(np.abs(D[:, j])))
        if D[k, j] < 0:
            D[:, j] *= -1
    return D


def _fit_shared(donors: list[WormData], *, dim: int = DIM) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    Zs, Ys = [], []
    for d in donors:
        pca = PCA(n_components=dim, random_state=17)
        pca.fit(d.X[d.masks["train"]])
        Z = pca.transform(d.X)
        starts = split_starts(d, "train", history=1, horizon=1)
        Zs.append(Z[starts])
        Ys.append(Z[starts + 1])
    Z = np.vstack(Zs)
    Y = np.vstack(Ys)
    phi = np.column_stack([Z, np.ones(len(Z))])
    coef = np.linalg.solve(phi.T @ phi + RIDGE * np.eye(phi.shape[1]), phi.T @ Y)
    A = coef[:dim].T.copy()
    b = coef[-1].copy()
    A, st = stabilize_matrix(A, CAP)
    residuals = Y - (Z @ A.T + b)
    Q = np.asarray(LedoitWolf().fit(residuals).covariance_, dtype=np.float64)
    Q = 0.5 * (Q + Q.T) + 1e-10 * np.eye(dim)
    meta = {
        "n_donor_sessions": len(donors),
        "n_pooled_transitions": int(len(Z)),
        "latent_dim": dim,
        "ridge": RIDGE,
        "stability": dict(st.__dict__),
        "basis_alignment_warning": "PCA bases are not identifiable across sessions; shared A is a model coordinate, not a biological common coordinate.",
    }
    return A, b, Q, meta


def _fit_alignment(Z: np.ndarray, starts: np.ndarray, A: np.ndarray, b: np.ndarray, *, ridge: float = 1.0) -> np.ndarray:
    p = A.shape[0]
    eye = np.eye(p)
    rows, targets = [], []
    for t in starts:
        M = np.kron(Z[int(t) + 1][None, :], eye) - np.kron(Z[int(t)][None, :], A)
        rows.append(M)
        targets.append(b)
    M = np.vstack(rows)
    target = np.concatenate(targets)
    reg = np.sqrt(ridge) * np.eye(p * p)
    sol = np.linalg.lstsq(
        np.vstack([M, reg]),
        np.r_[target, np.sqrt(ridge) * np.eye(p).reshape(-1, order="F")],
        rcond=None,
    )[0]
    return sol.reshape((p, p), order="F")


def _session_shared_model(
    data: WormData,
    *,
    A: np.ndarray,
    b: np.ndarray,
    Q: np.ndarray,
    calibration_mask: np.ndarray,
    adapted: bool,
) -> SharedLatentModel:
    pca = PCA(n_components=A.shape[0], random_state=17)
    pca.fit(data.X[calibration_mask])
    D = _canonical_pca(pca)
    mean = np.asarray(pca.mean_, dtype=np.float64)
    Z = (data.X - mean) @ D
    starts = split_starts(data, "train", history=1, horizon=1)
    starts = starts[calibration_mask[starts] & calibration_mask[starts + 1]]
    R = _fit_alignment(Z, starts, A, b) if adapted else np.eye(A.shape[0])
    probe = BehaviorProbe(seed=17)
    probe.fit(data.X[calibration_mask], data.y_macro[calibration_mask])
    recon = mean + Z @ D.T
    obs_cov = np.diag(np.var(data.X[calibration_mask] - recon[calibration_mask], axis=0) + 1e-8)
    initial = np.diag(np.var(Z[calibration_mask], axis=0) + 1e-8)
    return SharedLatentModel.from_shared(
        A=A,
        b=b,
        process_cov=Q,
        D=D,
        mean=mean,
        R=R,
        probe=probe,
        n_neurons=data.n_neurons,
        latent_dim=A.shape[0],
        obs_cov=obs_cov,
        initial_cov=initial,
        spectral_cap=CAP,
        name="shared_latent_adapted" if adapted else "shared_latent_zero_shot",
        metadata={
            "alignment": "calibration_linear_map" if adapted else "identity",
            "R_condition": float(np.linalg.cond(R)),
            "R_distance_from_identity": float(np.linalg.norm(R - np.eye(A.shape[0]))),
        },
    )


def _eval(data: WormData, model, split: str, horizon: int):
    sim = NeuralSimulator(model, data)
    starts = sim.test_starts(horizon=horizon) if split == "test" else sim.validation_starts(horizon=horizon)
    return rollout_metrics(sim, starts=starts, horizon=horizon)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/forward_stress.yaml")
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports/simulator/context_dynamics")
    args = parser.parse_args(argv)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig.load(args.config)
    raw = load_worm_data(cfg)
    names = _common_named(raw)
    datasets = [_panel(d, names) for d in raw]

    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    for held in range(len(datasets)):
        test_data = datasets[held]
        train_bounds = test_data.split_bounds["train"]
        train_mid = int(train_bounds[0] + 0.6 * (train_bounds[1] - train_bounds[0]))
        calibration = test_data.masks["train"].copy()
        calibration[train_mid:] = False
        starts_cal = split_starts(test_data, "train", history=1, horizon=1)
        starts_cal = starts_cal[calibration[starts_cal] & calibration[starts_cal + 1]]
        donors = [d for i, d in enumerate(datasets) if i != held]
        A, b, Q, fit_meta = _fit_shared(donors)
        models = {
            "shared_zero_shot": _session_shared_model(
                test_data, A=A, b=b, Q=Q, calibration_mask=calibration, adapted=False
            ),
            "shared_calibrated": _session_shared_model(
                test_data, A=A, b=b, Q=Q, calibration_mask=calibration, adapted=True
            ),
            "local_calibration": StableLatentLDS.fit(
                test_data.X,
                test_data.y_macro,
                starts_cal,
                train_mask=calibration,
                latent_dim=DIM,
                ridge=RIDGE,
                spectral_cap=CAP,
            ),
            "local_full_train_reference": StableLatentLDS.fit(
                test_data.X,
                test_data.y_macro,
                split_starts(test_data, "train", history=1, horizon=1),
                train_mask=test_data.masks["train"],
                latent_dim=DIM,
                ridge=RIDGE,
                spectral_cap=CAP,
            ),
        }
        metadata[f"held_out_{held}"] = fit_meta
        for model_name, model in models.items():
            for horizon in (20, 100):
                summary, _ = _eval(test_data, model, "test", horizon)
                rows.append(
                    {
                        "model": model_name,
                        "held_out_session": int(held),
                        "worm_id": int(test_data.worm_id),
                        "horizon": horizon,
                        "n_starts": summary["n_starts"],
                        "mse": summary["mse"],
                        "persistence_mse": summary["persistence_mse"],
                        "skill_vs_persistence": summary["skill_vs_persistence"],
                        "r2": summary["r2"],
                        "energy_distance": summary["energy_distance"],
                        "support_q95_fraction": summary["support_q95_fraction"],
                    }
                )
        print(f"held-out session {held} done", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(reports / "shared_dynamics_results.csv", index=False)
    agg = frame.groupby(["model", "horizon"], as_index=False).agg(
        sessions=("held_out_session", "nunique"),
        mean_skill=("skill_vs_persistence", "mean"),
        median_skill=("skill_vs_persistence", "median"),
        mean_mse=("mse", "mean"),
        mean_persistence=("persistence_mse", "mean"),
    )
    agg.to_csv(reports / "shared_dynamics_aggregate.csv", index=False)
    payload = {
        "schema_version": 1,
        "stage": "E_shared_dynamics",
        "evidence_tier": "exploratory_session_disjoint",
        "common_named_neurons": list(names),
        "n_common_named_neurons": len(names),
        "panel_rule": "alphabetic labels present in all five recordings; numeric IDs excluded because they are worm-specific",
        "latent_dim": DIM,
        "ridge": RIDGE,
        "spectral_cap": CAP,
        "test_design": "leave-one-session-out; observation map and aligned map fitted on a calibration prefix of the held-out training block; test block never used",
        "independence_warning": "session-disjoint, not animal-independent; five recordings are not five independently identified animals",
        "models": {
            "shared_zero_shot": "shared A, held-out PCA observation map, R=I",
            "shared_calibrated": "shared A, held-out observation map, linear R fitted on calibration transitions",
            "local_calibration": "per-session PCA+LDS fitted on exactly the same calibration prefix",
            "local_full_train_reference": "per-session PCA+LDS fitted on the full training block (not a fair calibration baseline)",
        },
        "metadata": metadata,
        "git": git_state(),
    }
    (reports / "shared_dynamics_summary.json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(agg.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
