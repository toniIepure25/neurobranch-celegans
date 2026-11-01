"""Cross-model virtual-perturbation consistency diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

from neurobranch.diagnostics.observation import CausalCalciumLDS, SecondOrderLatentLDS
from neurobranch.dynamics.context_models import CompositionalLatentLDS, StableLatentLDS
from neurobranch.dynamics.stable_var import StableVARModel

WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED."


def fit_models(data, *, latent_dim: int = 10, compositional: bool = False) -> dict[str, Any]:
    from neurobranch.preprocessing import split_starts

    train1 = split_starts(data, "train", history=1, horizon=1)
    train2 = split_starts(data, "train", history=2, horizon=1)
    models: dict[str, Any] = {
        "stable_var": StableVARModel.fit(data.X, data.y_macro, train1, history=1, ridge=1.0),
        "latent_lds": StableLatentLDS.fit(
            data.X,
            data.y_macro,
            train1,
            train_mask=data.masks["train"],
            latent_dim=latent_dim,
            ridge=1e-2,
        ),
        "second_order": SecondOrderLatentLDS.fit(
            data.X, data.masks["train"], train2, latent_dim=latent_dim, ridge=1e-2
        ),
        "calcium": CausalCalciumLDS.fit(
            data.X, data.masks["train"], train2, latent_dim=latent_dim, ridge=1e-2
        ),
    }
    if compositional:
        models["compositional"] = CompositionalLatentLDS.fit(
            data.X,
            data.y_macro,
            train1,
            train_mask=data.masks["train"],
            latent_dim=latent_dim,
            n_operators=2,
            operator_ridge=1e-2,
            coefficient_ridge=1e-2,
            alternations=3,
        )
    return models


def _rollout_baseline(model, hist: np.ndarray, horizon: int) -> np.ndarray:
    return model.rollout(np.zeros(hist.shape[-1]), horizon, initial_history=hist)[1:]


def _latent_response(model, data, frame: int, target: int, delta: float, horizon: int) -> np.ndarray:
    hist = data.X[frame - 1 : frame + 1]
    base = _rollout_baseline(model, hist, horizon)
    if getattr(model, "A1", None) is not None:
        z = model.encode(hist)
        z_cur, z_prev = z[-1], z[-2]
        z_pert = z_cur + np.linalg.pinv(model.observation_matrix) @ _delta_vec(data.n_neurons, target, delta)
        out = []
        for _ in range(horizon):
            z_next = z_pert @ model.A1.T + z_prev @ model.A2.T + model.b
            out.append(model.decode(z_next))
            z_prev, z_pert = z_pert, z_next
        return np.stack(out) - base
    if hasattr(model, "alpha"):
        delta_vec = _delta_vec(data.n_neurons, target, delta)
        z_base = model.encode(hist)[-1]
        z_pert = z_base + np.linalg.pinv(model.observation_matrix) @ delta_vec
        c_base = data.X[frame].copy()
        c_pert = c_base + delta_vec
        out = []
        for _ in range(horizon):
            z_base = z_base @ model.A.T + model.b
            z_pert = z_pert @ model.A.T + model.b
            c_base = (1 - model.alpha) * c_base + model.alpha * model.decode_drive(z_base)
            c_pert = (1 - model.alpha) * c_pert + model.alpha * model.decode_drive(z_pert)
            out.append(c_pert - c_base)
        return np.stack(out)
    # AR(1) latent or compositional latent.
    delta_vec = _delta_vec(data.n_neurons, target, delta)
    z = model.filter(hist)["latent"][-1]
    z_pert = z + np.linalg.pinv(model.observation_matrix) @ delta_vec
    out = []
    for _ in range(horizon):
        z_pert = model._transition(z_pert, np.zeros(model.input_dim))
        out.append(model.decode(z_pert))
    return np.stack(out) - base


def _delta_vec(n: int, target: int, delta: float) -> np.ndarray:
    out = np.zeros(n)
    out[target] = delta
    return out


def _var_response(model, data, frame: int, target: int, delta: float, horizon: int) -> np.ndarray:
    base = np.empty((horizon, data.n_neurons), dtype=np.float64)
    current = data.X[frame].copy()
    for step in range(horizon):
        current = np.asarray(model.predict_next(current[None, None, :])).reshape(data.n_neurons)
        base[step] = current
    current = data.X[frame].copy()
    current[target] += delta
    out = np.empty((horizon, data.n_neurons), dtype=np.float64)
    for step in range(horizon):
        current = np.asarray(model.predict_next(current[None, None, :])).reshape(data.n_neurons)
        out[step] = current
    return out - base


def compare(data, models: dict[str, Any], *, n_states: int = 5, horizon: int = 20) -> list[dict]:
    from neurobranch.preprocessing import split_starts

    test = split_starts(data, "test", history=2, horizon=horizon)
    if len(test) < n_states:
        return []
    frames = np.linspace(test[5], test[-1], n_states, dtype=int)
    named = [
        (j, name)
        for j, name in enumerate(data.predictor_names)
        if any(ch.isalpha() for ch in name) and not name.startswith("trace_")
    ]
    targets = named[:6]
    q_lo = np.quantile(data.X[data.masks["train"]], 0.005, axis=0)
    q_hi = np.quantile(data.X[data.masks["train"]], 0.995, axis=0)
    rows = []
    for frame in frames:
        support = float(
            np.mean((data.X[frame] >= q_lo) & (data.X[frame] <= q_hi))
        )
        for target, name in targets:
            responses = {}
            for model_name, model in models.items():
                if model_name == "stable_var":
                    resp = _var_response(model, data, frame, target, 0.5, horizon)
                else:
                    resp = _latent_response(model, data, frame, target, 0.5, horizon)
                responses[model_name] = resp[-1]
                rows.append(
                    {
                        "worm_id": int(data.worm_id),
                        "frame": int(frame),
                        "target": name,
                        "target_index": int(target),
                        "model": model_name,
                        "horizon": int(horizon),
                        "response_target": float(resp[-1, target]),
                        "response_norm": float(np.linalg.norm(resp[-1])),
                        "initial_state_support_fraction": support,
                        "warning": WARNING,
                    }
                )
            names = list(responses)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = responses[names[i]], responses[names[j]]
                    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                    rows.append(
                        {
                            "worm_id": int(data.worm_id),
                            "frame": int(frame),
                            "target": name,
                            "target_index": int(target),
                            "model": f"{names[i]}__vs__{names[j]}",
                            "horizon": int(horizon),
                            "response_target": np.nan,
                            "response_norm": np.nan,
                            "cosine_between_models": float(np.dot(a, b) / denom) if denom > 1e-12 else np.nan,
                            "relative_response_disagreement": float(
                                np.linalg.norm(a - b) / max(1e-12, 0.5 * (np.linalg.norm(a) + np.linalg.norm(b)))
                            ),
                            "initial_state_support_fraction": support,
                            "warning": WARNING,
                        }
                    )
    return rows


def seed_stability(data, *, latent_dim: int = 10, horizon: int = 20) -> list[dict]:
    from neurobranch.preprocessing import split_starts

    train1 = split_starts(data, "train", history=1, horizon=1)
    train2 = split_starts(data, "train", history=2, horizon=1)
    rng = np.random.default_rng(17)
    rows = []
    starts_a = np.sort(rng.choice(train1, size=int(0.8 * len(train1)), replace=False))
    starts_b = np.sort(rng.choice(train1, size=int(0.8 * len(train1)), replace=False))
    a = StableLatentLDS.fit(
        data.X, data.y_macro, starts_a, train_mask=data.masks["train"], latent_dim=latent_dim, ridge=1e-2, seed=17
    )
    b = StableLatentLDS.fit(
        data.X, data.y_macro, starts_b, train_mask=data.masks["train"], latent_dim=latent_dim, ridge=1e-2, seed=23
    )
    starts_c = np.sort(rng.choice(train2, size=int(0.8 * len(train2)), replace=False))
    starts_d = np.sort(rng.choice(train2, size=int(0.8 * len(train2)), replace=False))
    c = SecondOrderLatentLDS.fit(
        data.X, data.masks["train"], starts_c, latent_dim=latent_dim, ridge=1e-2, seed=17
    )
    d = SecondOrderLatentLDS.fit(
        data.X, data.masks["train"], starts_d, latent_dim=latent_dim, ridge=1e-2, seed=23
    )
    test = split_starts(data, "test", history=2, horizon=horizon)
    frames = np.linspace(test[5], test[-1], 3, dtype=int)
    target = next(
        (
            j
            for j, name in enumerate(data.predictor_names)
            if any(ch.isalpha() for ch in name) and not name.startswith("trace_")
        ),
        None,
    )
    if target is None:
        return []
    for frame in frames:
        ra = _latent_response(a, data, frame, target, 0.5, horizon)[-1]
        rb = _latent_response(b, data, frame, target, 0.5, horizon)[-1]
        rc = _latent_response(c, data, frame, target, 0.5, horizon)[-1]
        rd = _latent_response(d, data, frame, target, 0.5, horizon)[-1]
        for label, x, y in [("ar1_start_bootstrap_vs_bootstrap", ra, rb), ("ar2_start_bootstrap_vs_bootstrap", rc, rd)]:
            denom = np.linalg.norm(x) * np.linalg.norm(y)
            rows.append(
                {
                    "worm_id": int(data.worm_id),
                    "frame": int(frame),
                    "model": label,
                    "cosine": float(np.dot(x, y) / denom) if denom > 1e-12 else np.nan,
                    "relative_disagreement": float(
                        np.linalg.norm(x - y) / max(1e-12, 0.5 * (np.linalg.norm(x) + np.linalg.norm(y)))
                    ),
                }
            )
    return rows
