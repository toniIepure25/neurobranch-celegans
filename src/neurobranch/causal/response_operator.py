"""Robust hierarchical low-rank Causal Response Operator (CRO).

The model is deliberately small.  It predicts a latent target-responder effect
from (i) target/responder biases, (ii) a low-rank interaction, (iii) anatomical
features, and (iv) stimulation context.  Held-out targets receive no trained
embedding; their factors come from the anatomical MLP.  This is the difference
between a lookup table and an inductive operator.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from scipy import stats
from sklearn.preprocessing import StandardScaler
from torch import nn

from neurobranch.causal.features import FeatureBundle


@dataclass(frozen=True)
class CROConfig:
    rank: int = 16
    hidden: int = 64
    epochs: int = 45
    batch_size: int = 8192
    learning_rate: float = 2e-3
    weight_decay: float = 1e-3
    patience: int = 8
    dropout: float = 0.10
    df: float = 4.0
    grad_clip: float = 5.0
    seed: int = 17
    device: str = "cpu"
    use_pair_features: bool = True
    use_context: bool = True


class CROModule(nn.Module):
    def __init__(
        self,
        *,
        n_targets: int,
        n_responders: int,
        n_animals: int,
        d_target: int,
        d_responder: int,
        d_pair: int,
        d_context: int,
        rank: int,
        hidden: int,
        dropout: float,
        use_pair_features: bool,
        use_context: bool,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.use_pair_features = use_pair_features
        self.use_context = use_context
        self.target_emb = nn.Embedding(n_targets, rank)
        self.responder_emb = nn.Embedding(n_responders, rank)
        self.target_bias = nn.Embedding(n_targets, 1)
        self.responder_bias = nn.Embedding(n_responders, 1)
        self.animal_bias = nn.Embedding(n_animals, 1)
        self.target_mlp = nn.Sequential(
            nn.Linear(d_target, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, rank + 1),
        )
        self.responder_mlp = nn.Sequential(
            nn.Linear(d_responder, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, rank + 1),
        )
        self.pair_coef = nn.Linear(d_pair, 1, bias=False)
        self.context_coef = nn.Linear(d_context, 1, bias=False)
        self.global_mu = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_global = nn.Parameter(torch.tensor(0.0))
        self.target_log_sigma = nn.Embedding(n_targets, 1)
        self.responder_log_sigma = nn.Embedding(n_responders, 1)

        for emb in (self.target_emb, self.responder_emb):
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        for emb in (self.target_bias, self.responder_bias, self.animal_bias):
            nn.init.zeros_(emb.weight)
        for emb in (self.target_log_sigma, self.responder_log_sigma):
            nn.init.zeros_(emb.weight)
        for mlp in (self.target_mlp, self.responder_mlp):
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)
        nn.init.zeros_(self.pair_coef.weight)
        nn.init.zeros_(self.context_coef.weight)

    def forward(
        self,
        target_idx: torch.Tensor,
        responder_idx: torch.Tensor,
        animal_idx: torch.Tensor,
        target_features: torch.Tensor,
        responder_features: torch.Tensor,
        pair_features: torch.Tensor,
        context: torch.Tensor,
        target_seen: torch.Tensor,
        animal_seen: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seen_t = target_seen.to(dtype=torch.float32).unsqueeze(1)
        seen_a = animal_seen.to(dtype=torch.float32).unsqueeze(1)

        t_out = self.target_mlp(target_features)
        r_out = self.responder_mlp(responder_features)
        u = self.target_emb(target_idx) * seen_t + t_out[:, : self.rank]
        v = self.responder_emb(responder_idx) + r_out[:, : self.rank]
        b_t = self.target_bias(target_idx) * seen_t + t_out[:, self.rank :]
        c_r = self.responder_bias(responder_idx) + r_out[:, self.rank :]
        animal = self.animal_bias(animal_idx) * seen_a

        interaction = (u * v).sum(dim=1, keepdim=True)
        mu = self.global_mu + animal + b_t + c_r + interaction
        if self.use_pair_features:
            mu = mu + self.pair_coef(pair_features)
        if self.use_context:
            mu = mu + self.context_coef(context)

        log_sigma = (
            self.log_sigma_global
            + self.target_log_sigma(target_idx) * seen_t
            + self.responder_log_sigma(responder_idx)
        )
        sigma = nn.functional.softplus(log_sigma).squeeze(1) + 1e-3
        return mu.squeeze(1), sigma


def calibration_scale(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    level: float = 0.90,
    df: float = 4.0,
) -> float:
    """Validation-only multiplicative scale for predictive intervals.

    A value >1 widens intervals; <1 narrows them.  The scale is the empirical
    ``level`` quantile of absolute residuals divided by the nominal StudentT
    half-width.  It must be fitted on validation animals/targets, never test.
    """
    q = float(stats.t.ppf(0.5 + level / 2.0, df=df))
    ratio = np.abs(y_true - mu) / np.maximum(q * sigma, 1e-9)
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        return 1.0
    return float(max(np.quantile(ratio, level), 1e-6))


def _fit_scalers(
    bundle: FeatureBundle,
    train_idx: np.ndarray,
) -> dict[str, Any]:
    y = bundle.y[train_idx].astype(float)
    y_mean = float(np.mean(y))
    y_std = float(np.std(y)) or 1.0
    target_scaler = StandardScaler().fit(bundle.target_features[np.unique(bundle.target_idx[train_idx])])
    responder_scaler = StandardScaler().fit(
        bundle.responder_features[np.unique(bundle.responder_idx[train_idx])]
    )
    pair_scaler = StandardScaler().fit(bundle.pair_features[train_idx])
    context_raw = bundle.context_raw[train_idx].astype(float)
    medians = np.nanmedian(context_raw, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(context_raw), context_raw, medians)
    context_scaler = StandardScaler().fit(filled)
    return {
        "y_mean": y_mean,
        "y_std": y_std,
        "target_scaler": target_scaler,
        "responder_scaler": responder_scaler,
        "pair_scaler": pair_scaler,
        "context_median": medians,
        "context_scaler": context_scaler,
    }


def _transform(
    bundle: FeatureBundle,
    idx: np.ndarray,
    scalers: dict[str, Any],
) -> tuple[np.ndarray, ...]:
    target_features = scalers["target_scaler"].transform(bundle.target_features).astype(np.float32)
    responder_features = scalers["responder_scaler"].transform(bundle.responder_features).astype(np.float32)
    pair_features = scalers["pair_scaler"].transform(bundle.pair_features[idx]).astype(np.float32)
    context_raw = bundle.context_raw[idx].astype(float)
    filled = np.where(np.isfinite(context_raw), context_raw, scalers["context_median"])
    context = scalers["context_scaler"].transform(filled).astype(np.float32)
    y = ((bundle.y[idx] - scalers["y_mean"]) / scalers["y_std"]).astype(np.float32)
    return target_features, responder_features, pair_features, context, y


def train_cro(
    bundle: FeatureBundle,
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    held_out_targets: set[str],
    held_out_animals: set[str],
    config: CROConfig | None = None,
) -> dict[str, Any]:
    cfg = config or CROConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device(cfg.device)
    scalers = _fit_scalers(bundle, train_idx)
    target_features, responder_features, pair_features, context, _ = _transform(bundle, train_idx, scalers)
    val_tf, val_rf, val_pf, val_ctx, _ = _transform(bundle, val_idx, scalers)
    test_tf, test_rf, test_pf, test_ctx, _ = _transform(bundle, test_idx, scalers)
    target_seen_row = np.array(
        [bundle.target_names[i] not in held_out_targets for i in bundle.target_idx], dtype=bool
    )
    animal_seen_row = np.array(
        [bundle.animal_names[i] not in held_out_animals for i in bundle.animal_idx], dtype=bool
    )

    n_targets = len(bundle.target_names)
    n_responders = len(bundle.responder_names)
    n_animals = len(bundle.animal_names)
    model = CROModule(
        n_targets=n_targets,
        n_responders=n_responders,
        n_animals=n_animals,
        d_target=target_features.shape[1],
        d_responder=responder_features.shape[1],
        d_pair=pair_features.shape[1],
        d_context=context.shape[1],
        rank=cfg.rank,
        hidden=cfg.hidden,
        dropout=cfg.dropout,
        use_pair_features=cfg.use_pair_features,
        use_context=cfg.use_context,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    def tensors(
        idx: np.ndarray,
        tf_np: np.ndarray,
        rf_np: np.ndarray,
        pf_np: np.ndarray,
        ctx_np: np.ndarray,
    ) -> tuple[torch.Tensor, ...]:
        t = torch.as_tensor(bundle.target_idx[idx], dtype=torch.long, device=device)
        r = torch.as_tensor(bundle.responder_idx[idx], dtype=torch.long, device=device)
        a = torch.as_tensor(bundle.animal_idx[idx], dtype=torch.long, device=device)
        tf_all = torch.as_tensor(tf_np, dtype=torch.float32, device=device)
        rf_all = torch.as_tensor(rf_np, dtype=torch.float32, device=device)
        return (
            t,
            r,
            a,
            tf_all[t],
            rf_all[r],
            torch.as_tensor(pf_np, dtype=torch.float32, device=device),
            torch.as_tensor(ctx_np, dtype=torch.float32, device=device),
            torch.as_tensor(target_seen_row[idx], dtype=torch.bool, device=device),
            torch.as_tensor(animal_seen_row[idx], dtype=torch.bool, device=device),
        )

    def run(
        idx: np.ndarray,
        tf_np: np.ndarray,
        rf_np: np.ndarray,
        pf_np: np.ndarray,
        ctx_np: np.ndarray,
        *,
        training: bool,
    ) -> float:
        t, r, a, tf, rf, pf, ctx, tseen, aseen = tensors(idx, tf_np, rf_np, pf_np, ctx_np)
        yy = torch.as_tensor(
            (bundle.y[idx] - scalers["y_mean"]) / scalers["y_std"],
            dtype=torch.float32,
            device=device,
        )
        if training:
            model.train()
            order = torch.randperm(len(idx), device=device)
            total = 0.0
            n_batches = 0
            for start in range(0, len(idx), cfg.batch_size):
                sel = order[start : start + cfg.batch_size]
                mu, sigma = model(
                    t[sel], r[sel], a[sel], tf[sel], rf[sel], pf[sel], ctx[sel], tseen[sel], aseen[sel]
                )
                loss = -torch.distributions.StudentT(df=cfg.df, loc=mu, scale=sigma).log_prob(yy[sel]).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                total += float(loss.detach())
                n_batches += 1
            return total / max(1, n_batches)
        model.eval()
        with torch.no_grad():
            mu, sigma = model(t, r, a, tf, rf, pf, ctx, tseen, aseen)
            return float(-torch.distributions.StudentT(df=cfg.df, loc=mu, scale=sigma).log_prob(yy).mean())

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    bad = 0
    history = []
    for epoch in range(cfg.epochs):
        train_loss = run(
            train_idx, target_features, responder_features, pair_features, context, training=True
        )
        val_loss = run(val_idx, val_tf, val_rf, val_pf, val_ctx, training=False)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)

    def predict(
        idx: np.ndarray,
        tf_np: np.ndarray,
        rf_np: np.ndarray,
        pf_np: np.ndarray,
        ctx_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        t, r, a, tf, rf, pf, ctx, tseen, aseen = tensors(idx, tf_np, rf_np, pf_np, ctx_np)
        model.eval()
        with torch.no_grad():
            mu, sigma = model(t, r, a, tf, rf, pf, ctx, tseen, aseen)
        mu_np = mu.cpu().numpy() * scalers["y_std"] + scalers["y_mean"]
        sigma_np = sigma.cpu().numpy() * scalers["y_std"]
        return mu_np, sigma_np

    train_mu, train_sigma = predict(train_idx, target_features, responder_features, pair_features, context)
    val_mu, val_sigma = predict(val_idx, val_tf, val_rf, val_pf, val_ctx)
    test_mu, test_sigma = predict(test_idx, test_tf, test_rf, test_pf, test_ctx)
    return {
        "config": asdict(cfg),
        "model_state": best_state,
        "scalers": scalers,
        "history": history,
        "best_val_loss": best_val,
        "train_mu": train_mu,
        "train_sigma": train_sigma,
        "val_mu": val_mu,
        "val_sigma": val_sigma,
        "test_mu": test_mu,
        "test_sigma": test_sigma,
    }
