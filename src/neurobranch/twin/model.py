"""Animal-conditional response twin.

The architecture is deliberately small enough for CPU reruns.  It has four
inductive biases that matter for the Phase 8 question:

1. target/responder factors come from anatomy, not from a lookup table, so an
   unseen target still has a representation;
2. a temporal basis predicts the full post-stimulus trajectory instead of only
   its mean;
3. a permutation-invariant support profile (mean/std over observed support
   events) modulates every prediction through FiLM;
4. a StudentT likelihood keeps the loss robust to calcium outliers.

The support profile is computed outside the module and passed in as a vector.
That keeps the training loop simple and makes the k=0 case explicit: a zero
support vector is the prior, not a missing-data accident.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class TwinConfig:
    rank: int = 16
    hidden: int = 64
    temporal_basis: int = 8
    support_dim: int = 16
    dropout: float = 0.10
    df: float = 5.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    epochs: int = 30
    batch_size: int = 4096
    patience: int = 6
    grad_clip: float = 5.0
    seed: int = 17
    device: str = "cpu"
    use_support: bool = True
    use_history: bool = True
    use_trajectory: bool = True
    traj_loss_weight: float = 1.0


@dataclass
class TwinPrediction:
    """Predictions for one event/responder set, in original response units."""

    scalar_mu: np.ndarray
    scalar_sigma: np.ndarray
    trajectory_mu: np.ndarray
    trajectory_sigma: np.ndarray
    support_hash: str = ""


class ResponseTwin(nn.Module):
    def __init__(
        self,
        *,
        d_target: int,
        d_responder: int,
        d_pair: int,
        d_context: int,
        pre_frames: int,
        post_frames: int,
        config: TwinConfig,
        support_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.post_frames = post_frames
        self.support_dim = support_dim
        self.target_mlp = nn.Sequential(
            nn.Linear(d_target, config.hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, config.rank + 1),
        )
        self.responder_mlp = nn.Sequential(
            nn.Linear(d_responder, config.hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, config.rank + 1),
        )
        row_in = d_target + d_responder + d_pair + d_context + pre_frames + 3
        self.row_mlp = nn.Sequential(
            nn.Linear(row_in, config.hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, config.hidden),
            nn.SiLU(),
        )
        self.support_mlp = nn.Sequential(
            nn.Linear(support_dim, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, 2 * config.hidden),
        )
        self.scalar_head = nn.Linear(config.hidden, 1)
        self.traj_head = nn.Linear(config.hidden, config.temporal_basis)
        self.scalar_log_sigma = nn.Linear(config.hidden, 1)
        self.traj_log_sigma = nn.Linear(config.hidden, post_frames)
        self.basis = nn.Parameter(torch.zeros(config.temporal_basis, post_frames))
        self.global_scalar = nn.Parameter(torch.zeros(1))
        self.global_log_sigma = nn.Parameter(torch.zeros(1))

        # Start from a smooth, low-frequency basis.  The network can rotate it.
        with torch.no_grad():
            t = torch.linspace(0.0, 1.0, post_frames)
            for k in range(config.temporal_basis):
                self.basis[k] = torch.sin((k + 1) * np.pi * t) if k % 2 == 0 else torch.cos(k * np.pi * t)
            self.basis.mul_(0.2)
        for mlp in (self.target_mlp, self.responder_mlp):
            layer = mlp[-1]
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.support_mlp[-1].weight)
        nn.init.zeros_(self.support_mlp[-1].bias)
        nn.init.zeros_(self.scalar_head.weight)
        nn.init.zeros_(self.scalar_head.bias)
        nn.init.zeros_(self.traj_head.weight)
        nn.init.zeros_(self.traj_head.bias)
        nn.init.zeros_(self.scalar_log_sigma.weight)
        nn.init.zeros_(self.scalar_log_sigma.bias)
        nn.init.zeros_(self.traj_log_sigma.weight)
        nn.init.zeros_(self.traj_log_sigma.bias)

    def forward(
        self,
        target_features: torch.Tensor,
        responder_features: torch.Tensor,
        pair_features: torch.Tensor,
        context: torch.Tensor,
        pre: torch.Tensor,
        prior: torch.Tensor,
        support: torch.Tensor,
        traj_base: torch.Tensor,
        *,
        support_active: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t_out = self.target_mlp(target_features)
        r_out = self.responder_mlp(responder_features)
        u = t_out[:, : self.config.rank]
        v = r_out[:, : self.config.rank]
        interaction = (u * v).sum(dim=1, keepdim=True)
        row = torch.cat(
            [target_features, responder_features, pair_features, context, pre, prior],
            dim=1,
        )
        h = self.row_mlp(row) + interaction + t_out[:, self.config.rank :] + r_out[:, self.config.rank :]
        if self.config.use_support and support.shape[1] > 0:
            if support_active is not None:
                support = support * support_active.to(dtype=support.dtype).unsqueeze(1)
            film = self.support_mlp(support)
            gamma, beta = film.chunk(2, dim=1)
            h = h * (1.0 + torch.tanh(gamma)) + beta
        scalar_mu = self.scalar_head(h).squeeze(1) + self.global_scalar
        coeff = self.traj_head(h)
        # Residual trajectory around the train-only per-responder template.
        # With a zero head this reduces exactly to that template.
        traj_mu = coeff @ self.basis + traj_base
        scalar_sigma = nn.functional.softplus(self.scalar_log_sigma(h)).squeeze(1)
        scalar_sigma = scalar_sigma + nn.functional.softplus(self.global_log_sigma).squeeze() + 1e-3
        traj_sigma = nn.functional.softplus(self.traj_log_sigma(h)) + 1e-3
        return scalar_mu, scalar_sigma, traj_mu, traj_sigma


def student_t_nll(
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    *,
    df: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean StudentT negative log-likelihood with an optional mask."""
    dist = torch.distributions.StudentT(df=df, loc=mu, scale=sigma)
    nll = -dist.log_prob(y)
    if mask is not None:
        nll = nll * mask.to(dtype=nll.dtype)
        denom = mask.to(dtype=nll.dtype).sum().clamp_min(1.0)
        return nll.sum() / denom
    return nll.mean()
