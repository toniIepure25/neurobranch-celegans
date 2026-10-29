"""Training and inference for the Phase 8 response twin.

The unit of fitting is an :class:`AnimalTask`: a support set of observed events
and a target-disjoint test set for the same animal.  The model never sees the
test outcomes of the animal it is being evaluated on; it only sees the support
outcomes, exactly as an experimentalist would after a small calibration block.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from neurobranch.causal.features import build_feature_bundle
from neurobranch.config import PROJECT_ROOT, resolve_path
from neurobranch.twin.model import ResponseTwin, TwinConfig, TwinPrediction, student_t_nll
from neurobranch.twin.tasks import AnimalTask, AnimalTaskSplit
from neurobranch.twin.windows import EventWindowCache

DEFAULT_TRIALS = "reports/phase3/randi_trial_table_phase3.csv.gz"
DEFAULT_CONNECTOME = "references/external/aconnectome_witvliet_2020_8_utf8.csv"


@dataclass
class TwinData:
    cache: EventWindowCache
    target_features: np.ndarray
    responder_features: np.ndarray
    pair_features: np.ndarray
    pair_known: np.ndarray
    context: np.ndarray
    config: TwinConfig

    @property
    def n_targets(self) -> int:
        return int(len(self.cache.target_names))

    @property
    def n_responders(self) -> int:
        return int(len(self.cache.responder_names))

    def event_lookup(self) -> dict[str, int]:
        return {str(e): i for i, e in enumerate(self.cache.event_ids.astype(str))}

    def rows_for_events(self, events: list[str] | tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        lookup = self.event_lookup()
        event_rows: list[np.ndarray] = []
        responder_rows: list[np.ndarray] = []
        for e in events:
            i = lookup.get(str(e))
            if i is None:
                continue
            ridx = np.flatnonzero(self.cache.mask[i])
            if ridx.size == 0:
                continue
            event_rows.append(np.full(ridx.size, i, dtype=np.int64))
            responder_rows.append(ridx.astype(np.int64))
        if not event_rows:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        return np.concatenate(event_rows), np.concatenate(responder_rows)


@dataclass
class TwinScalers:
    target: StandardScaler
    responder: StandardScaler
    pair: StandardScaler
    context: StandardScaler
    pre: StandardScaler
    support: StandardScaler
    y_mean: float
    y_std: float
    traj_mean: np.ndarray
    traj_std: np.ndarray
    traj_template: np.ndarray
    responder_prior: np.ndarray
    target_prior: np.ndarray
    target_seen: np.ndarray

    def state(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "responder": self.responder,
            "pair": self.pair,
            "context": self.context,
            "pre": self.pre,
            "support": self.support,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
            "traj_mean": self.traj_mean,
            "traj_std": self.traj_std,
            "traj_template": self.traj_template,
            "responder_prior": self.responder_prior,
            "target_prior": self.target_prior,
            "target_seen": self.target_seen,
        }


@dataclass
class TwinFit:
    module: ResponseTwin
    data: TwinData
    scalers: TwinScalers
    config: TwinConfig
    split: AnimalTaskSplit
    history: list[dict[str, float]]

    def support_profile(self, events: list[str] | tuple[str, ...]) -> np.ndarray:
        if not events:
            return np.zeros(int(self.config.support_dim), dtype=np.float32)
        raw = support_profile_raw(self.data.cache, events)
        z = self.scalers.support.transform(raw.reshape(1, -1)).astype(np.float32)[0]
        return np.clip(z, -8.0, 8.0).astype(np.float32)

    def predict_events(
        self,
        events: list[str] | tuple[str, ...],
        *,
        support_events: list[str] | tuple[str, ...] = (),
        include_metadata: bool = True,
    ) -> tuple[TwinPrediction, pd.DataFrame]:
        support = self.support_profile(support_events)
        active = 1.0 if support_events else 0.0
        pred = self._predict(events, support, active)
        if not include_metadata:
            return pred, pd.DataFrame()
        frame = self._metadata(events)
        if len(frame) != len(pred.scalar_mu):
            raise RuntimeError("metadata and prediction row counts differ")
        frame["y_pred"] = pred.scalar_mu
        frame["sigma"] = pred.scalar_sigma
        frame["traj_pred_mean"] = pred.trajectory_mu.mean(axis=1)
        frame["traj_sigma_mean"] = pred.trajectory_sigma.mean(axis=1)
        return pred, frame

    def _metadata(self, events: list[str] | tuple[str, ...]) -> pd.DataFrame:
        ev, ridx = self.data.rows_for_events(events)
        return pd.DataFrame(
            {
                "event_id": self.data.cache.event_ids[ev].astype(str),
                "dataset_id": self.data.cache.dataset_ids[ev].astype(str),
                "target": [self.data.cache.target_names[int(self.data.cache.target_idx[i])] for i in ev],
                "neuron": [self.data.cache.responder_names[int(i)] for i in ridx],
                "y_true": self.data.cache.scalar[ev, ridx].astype(float),
                "mask": np.ones(len(ev), dtype=bool),
            }
        )

    def _predict(self, events: list[str] | tuple[str, ...], support: np.ndarray, active: float) -> TwinPrediction:
        ev, ridx = self.data.rows_for_events(events)
        if ev.size == 0:
            p = self.data.cache.post.shape[-1]
            return TwinPrediction(
                scalar_mu=np.empty(0),
                scalar_sigma=np.empty(0),
                trajectory_mu=np.empty((0, p)),
                trajectory_sigma=np.empty((0, p)),
            )
        tensors = self._tensors(ev, ridx, support, active)
        self.module.eval()
        with torch.no_grad():
            smu, ssig, tmu, tsig = self.module(*tensors[:-1], support_active=tensors[-1])
        smu = smu.cpu().numpy() * self.scalers.y_std + self.scalers.y_mean
        ssig = ssig.cpu().numpy() * self.scalers.y_std
        tmu = tmu.cpu().numpy() * self.scalers.traj_std + self.scalers.traj_mean
        tsig = tsig.cpu().numpy() * self.scalers.traj_std
        return TwinPrediction(
            scalar_mu=smu,
            scalar_sigma=ssig,
            trajectory_mu=tmu,
            trajectory_sigma=tsig,
        )

    def _tensors(
        self,
        ev: np.ndarray,
        ridx: np.ndarray,
        support: np.ndarray,
        active: float,
    ) -> tuple[torch.Tensor, ...]:
        cfg = self.config
        target_idx = self.data.cache.target_idx[ev]
        tf = self.scalers.target.transform(self.data.target_features[target_idx]).astype(np.float32)
        rf = self.scalers.responder.transform(self.data.responder_features[ridx]).astype(np.float32)
        pf = self.scalers.pair.transform(self.data.pair_features[target_idx, ridx]).astype(np.float32)
        ctx = self.scalers.context.transform(self.data.context[ev]).astype(np.float32)
        pre = self.scalers.pre.transform(self.data.cache.pre[ev, ridx, :]).astype(np.float32)
        support_batch = np.repeat(support.reshape(1, -1), len(ev), axis=0).astype(np.float32)
        prior = np.column_stack(
            [
                (self.scalers.target_prior[target_idx] - self.scalers.y_mean) / self.scalers.y_std,
                (self.scalers.responder_prior[ridx] - self.scalers.y_mean) / self.scalers.y_std,
                self.scalers.target_seen[target_idx],
            ]
        ).astype(np.float32)
        traj_base = (
            (self.scalers.traj_template[ridx] - self.scalers.traj_mean) / self.scalers.traj_std
        ).astype(np.float32)
        return (
            torch.as_tensor(tf, dtype=torch.float32, device=cfg.device),
            torch.as_tensor(rf, dtype=torch.float32, device=cfg.device),
            torch.as_tensor(pf, dtype=torch.float32, device=cfg.device),
            torch.as_tensor(ctx, dtype=torch.float32, device=cfg.device),
            torch.as_tensor(pre, dtype=torch.float32, device=cfg.device),
            torch.as_tensor(prior, dtype=torch.float32, device=cfg.device),
            torch.as_tensor(support_batch, dtype=torch.float32, device=cfg.device),
            torch.as_tensor(traj_base, dtype=torch.float32, device=cfg.device),
            torch.full((len(ev),), float(active), dtype=torch.float32, device=cfg.device),
        )


def support_profile_raw(cache: EventWindowCache, events: list[str] | tuple[str, ...]) -> np.ndarray:
    """Permutation-invariant summary of observed support events."""
    lookup = {str(e): i for i, e in enumerate(cache.event_ids.astype(str))}
    stats = []
    intervals = []
    for e in events:
        i = lookup.get(str(e))
        if i is None:
            continue
        ridx = np.flatnonzero(cache.mask[i])
        if ridx.size == 0:
            continue
        post = cache.post[i, ridx, :].astype(float)
        pre = cache.pre[i, ridx, :].astype(float)
        scalar = cache.scalar[i, ridx].astype(float)
        stats.append(
            [
                float(np.mean(post)),
                float(np.std(post)),
                float(np.max(np.abs(post))),
                float(np.mean(pre)),
                float(np.std(pre)),
                float(np.mean(scalar)),
                float(ridx.size / 100.0),
            ]
        )
        if cache.history.shape[1] > 0:
            intervals.append(float(cache.history[i, 0]))
    if not stats:
        return np.zeros(16, dtype=np.float32)
    arr = np.asarray(stats, dtype=float)
    out = np.concatenate([arr.mean(axis=0), arr.std(axis=0), [len(stats) / 20.0, float(np.mean(intervals) if intervals else 0.0)]])
    out = np.asarray(out, dtype=np.float32)
    if out.size < 16:
        out = np.pad(out, (0, 16 - out.size))
    return out[:16]


def _feature_arrays(cache: EventWindowCache, trials_path: str | Path, connectome_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trials = pd.read_csv(resolve_path(trials_path, PROJECT_ROOT))
    bundle = build_feature_bundle(trials, connectome_path=connectome_path)
    t_map = {n: i for i, n in enumerate(bundle.target_names)}
    r_map = {n: i for i, n in enumerate(bundle.responder_names)}
    d_t = bundle.target_features.shape[1]
    d_r = bundle.responder_features.shape[1]
    d_p = bundle.pair_features.shape[1]
    tf = np.zeros((len(cache.target_names), d_t), dtype=np.float32)
    rf = np.zeros((len(cache.responder_names), d_r), dtype=np.float32)
    pf = np.zeros((len(cache.target_names), len(cache.responder_names), d_p), dtype=np.float32)
    known = np.zeros((len(cache.target_names), len(cache.responder_names)), dtype=bool)
    for i, name in enumerate(cache.target_names):
        if name in t_map:
            tf[i] = bundle.target_features[t_map[name]]
    for i, name in enumerate(cache.responder_names):
        if name in r_map:
            rf[i] = bundle.responder_features[r_map[name]]
    # Pair features are target-responder properties, so one row per pair is
    # enough even when the atlas contains repeated events.
    pair_frame = pd.DataFrame(
        {
            "target": [bundle.target_names[i] for i in bundle.target_idx],
            "neuron": [bundle.responder_names[i] for i in bundle.responder_idx],
            "row": np.arange(len(bundle.y)),
        }
    ).drop_duplicates(["target", "neuron"])
    for row in pair_frame.itertuples(index=False):
        ti = cache.target_names.index(str(row.target)) if str(row.target) in cache.target_names else None
        ri = cache.responder_names.index(str(row.neuron)) if str(row.neuron) in cache.responder_names else None
        if ti is None or ri is None:
            continue
        pf[ti, ri] = bundle.pair_features[int(row.row)]
        known[ti, ri] = True
    return tf, rf, pf, known


def _fit_scalers(data: TwinData, tasks: list[AnimalTask]) -> TwinScalers:
    events: list[str] = []
    for task in tasks:
        events.extend(task.support_events)
        events.extend(task.test_events)
    ev, ridx = data.rows_for_events(events)
    if ev.size == 0:
        raise ValueError("no training rows")
    target_idx = data.cache.target_idx[ev]
    tf = data.target_features[target_idx].astype(float)
    rf = data.responder_features[ridx].astype(float)
    pf = data.pair_features[target_idx, ridx].astype(float)
    ctx = data.context[ev].astype(float)
    pre = data.cache.pre[ev, ridx, :].astype(float)
    post = data.cache.post[ev, ridx, :].astype(float)
    y = data.cache.scalar[ev, ridx].astype(float)
    support = np.stack([support_profile_raw(data.cache, t.support_events) for t in tasks], axis=0)
    n_resp = len(data.cache.responder_names)
    resp_sum = np.zeros(n_resp, dtype=float)
    resp_count = np.zeros(n_resp, dtype=float)
    np.add.at(resp_sum, ridx, y)
    np.add.at(resp_count, ridx, 1.0)
    resp_prior = np.where(resp_count > 0, resp_sum / np.maximum(resp_count, 1.0), float(np.mean(y)))
    n_target = len(data.cache.target_names)
    tgt_sum = np.zeros(n_target, dtype=float)
    tgt_count = np.zeros(n_target, dtype=float)
    np.add.at(tgt_sum, target_idx, y)
    np.add.at(tgt_count, target_idx, 1.0)
    tgt_prior = np.where(tgt_count > 0, tgt_sum / np.maximum(tgt_count, 1.0), float(np.mean(y)))
    target_seen = (tgt_count > 0).astype(np.float32)
    # Train-only per-responder mean trajectory; predictions are a residual on
    # top of this template, so a zero trajectory head is the template baseline.
    traj_template = np.zeros((n_resp, post.shape[1]), dtype=float)
    for ri in range(n_resp):
        sel = ridx == ri
        if sel.any():
            traj_template[ri] = post[sel].mean(axis=0)
    traj_template[traj_template.sum(axis=1) == 0] = post.mean(axis=0)
    return TwinScalers(
        target=StandardScaler().fit(tf),
        responder=StandardScaler().fit(rf),
        pair=StandardScaler().fit(pf),
        context=StandardScaler().fit(ctx),
        pre=StandardScaler().fit(pre),
        support=StandardScaler().fit(support),
        y_mean=float(np.mean(y)),
        y_std=float(np.std(y)) or 1.0,
        traj_mean=np.mean(post, axis=0),
        traj_std=np.std(post, axis=0) + 1e-6,
        traj_template=traj_template,
        responder_prior=resp_prior,
        target_prior=tgt_prior,
        target_seen=target_seen,
    )


def _support_for_task(fit: TwinFit, task: AnimalTask, selected: list[str] | None = None) -> np.ndarray:
    events = selected if selected is not None else list(task.support_events)
    return fit.support_profile(events)


def _tensor_rows(
    data: TwinData,
    scalers: TwinScalers,
    events: list[str],
    support_vectors: dict[str, np.ndarray],
    *,
    device: str,
    support_dim: int = 16,
) -> tuple[torch.Tensor, ...]:
    ev, ridx = data.rows_for_events(events)
    target_idx = data.cache.target_idx[ev]
    tf = scalers.target.transform(data.target_features[target_idx]).astype(np.float32)
    rf = scalers.responder.transform(data.responder_features[ridx]).astype(np.float32)
    pf = scalers.pair.transform(data.pair_features[target_idx, ridx]).astype(np.float32)
    ctx = scalers.context.transform(data.context[ev]).astype(np.float32)
    pre = scalers.pre.transform(data.cache.pre[ev, ridx, :]).astype(np.float32)
    animal = data.cache.dataset_ids[ev].astype(str)
    support = np.stack(
        [support_vectors.get(a, np.zeros(support_dim, dtype=np.float32)) for a in animal], axis=0
    )
    active = np.array([1.0 if a in support_vectors else 0.0 for a in animal], dtype=np.float32)
    prior = np.column_stack(
        [
            (scalers.target_prior[target_idx] - scalers.y_mean) / scalers.y_std,
            (scalers.responder_prior[ridx] - scalers.y_mean) / scalers.y_std,
            scalers.target_seen[target_idx],
        ]
    ).astype(np.float32)
    y = ((data.cache.scalar[ev, ridx] - scalers.y_mean) / scalers.y_std).astype(np.float32)
    post = ((data.cache.post[ev, ridx, :] - scalers.traj_mean) / scalers.traj_std).astype(np.float32)
    traj_base = ((scalers.traj_template[ridx] - scalers.traj_mean) / scalers.traj_std).astype(np.float32)
    return (
        torch.as_tensor(tf, device=device),
        torch.as_tensor(rf, device=device),
        torch.as_tensor(pf, device=device),
        torch.as_tensor(ctx, device=device),
        torch.as_tensor(pre, device=device),
        torch.as_tensor(prior, device=device),
        torch.as_tensor(support, device=device),
        torch.as_tensor(traj_base, device=device),
        torch.as_tensor(active, device=device),
        torch.as_tensor(y, device=device),
        torch.as_tensor(post, device=device),
        torch.as_tensor(ev, device=device),
        torch.as_tensor(ridx, device=device),
    )


def _make_twin_data(
    cache: EventWindowCache,
    *,
    trials_path: str | Path,
    connectome_path: str | Path,
    config: TwinConfig,
) -> TwinData:
    tf, rf, pf, known = _feature_arrays(cache, trials_path, connectome_path)
    return TwinData(
        cache=cache,
        target_features=tf,
        responder_features=rf,
        pair_features=pf,
        pair_known=known,
        context=cache.history.astype(np.float32),
        config=config,
    )


def fit_twin(
    cache: EventWindowCache,
    split: AnimalTaskSplit,
    *,
    config: TwinConfig | None = None,
    trials_path: str | Path = DEFAULT_TRIALS,
    connectome_path: str | Path = DEFAULT_CONNECTOME,
    data: TwinData | None = None,
) -> TwinFit:
    cfg = config or TwinConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    if not split.train_tasks:
        raise ValueError(f"fold {split.fold} has no training tasks")
    if not split.val_tasks:
        raise ValueError(f"fold {split.fold} has no validation tasks")
    data = data or _make_twin_data(
        cache, trials_path=trials_path, connectome_path=connectome_path, config=cfg
    )
    tf = data.target_features
    rf = data.responder_features
    pf = data.pair_features
    scalers = _fit_scalers(data, list(split.train_tasks.values()))
    device = cfg.device
    module = ResponseTwin(
        d_target=tf.shape[1],
        d_responder=rf.shape[1],
        d_pair=pf.shape[2],
        d_context=data.context.shape[1],
        pre_frames=cache.pre.shape[-1],
        post_frames=cache.post.shape[-1],
        config=cfg,
        support_dim=cfg.support_dim,
    ).to(device)
    fit = TwinFit(
        module=module,
        data=data,
        scalers=scalers,
        config=cfg,
        split=split,
        history=[],
    )
    train_events: list[str] = []
    val_events: list[str] = []
    train_support: dict[str, np.ndarray] = {}
    val_support: dict[str, np.ndarray] = {}
    for animal, task in split.train_tasks.items():
        train_events.extend(task.test_events)
        train_support[animal] = fit.support_profile(task.support_events)
    for animal, task in split.val_tasks.items():
        val_events.extend(task.test_events)
        val_support[animal] = fit.support_profile(task.support_events)
    if not train_events or not val_events:
        raise ValueError("empty train or validation events")
    train_tensors = _tensor_rows(
        data, scalers, train_events, train_support, device=device, support_dim=cfg.support_dim
    )
    val_tensors = _tensor_rows(
        data, scalers, val_events, val_support, device=device, support_dim=cfg.support_dim
    )
    optimizer = torch.optim.AdamW(module.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    best_state = copy.deepcopy(module.state_dict())
    best_val = float("inf")
    bad = 0
    history: list[dict[str, float]] = []

    def loss_for(tensors: tuple[torch.Tensor, ...], *, training: bool) -> torch.Tensor:
        tf_b, rf_b, pf_b, ctx_b, pre_b, prior_b, sup_b, base_b, act_b, y_b, post_b, _, _ = tensors
        if training:
            module.train()
        else:
            module.eval()
        smu, ssig, tmu, tsig = module(
            tf_b, rf_b, pf_b, ctx_b, pre_b, prior_b, sup_b, base_b, support_active=act_b
        )
        loss = student_t_nll(y_b, smu, ssig, df=cfg.df)
        if cfg.use_trajectory:
            loss = loss + cfg.traj_loss_weight * student_t_nll(post_b, tmu, tsig, df=cfg.df)
        return loss

    n_train = train_tensors[0].shape[0]
    for epoch in range(cfg.epochs):
        module.train()
        order = torch.randperm(n_train)
        total = 0.0
        n_batches = 0
        for start in range(0, n_train, cfg.batch_size):
            sel = order[start : start + cfg.batch_size]
            batch = tuple(t[sel] for t in train_tensors)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(batch, training=True)
            loss.backward()
            nn.utils.clip_grad_norm_(module.parameters(), cfg.grad_clip)
            optimizer.step()
            total += float(loss.detach())
            n_batches += 1
        with torch.no_grad():
            val_loss = float(loss_for(val_tensors, training=False).detach())
        history.append({"epoch": epoch, "train_loss": total / max(1, n_batches), "val_loss": val_loss})
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = copy.deepcopy(module.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    module.load_state_dict(best_state)
    fit.history = history
    return fit


def save_fit(fit: TwinFit, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": fit.module.state_dict(),
            "config": asdict(fit.config),
            "scalers": fit.scalers.state(),
            "fold": fit.split.fold,
            "split": fit.split.split,
        },
        p,
    )
    return p


def load_fit(
    cache: EventWindowCache,
    split: AnimalTaskSplit,
    path: str | Path,
    *,
    trials_path: str | Path = DEFAULT_TRIALS,
    connectome_path: str | Path = DEFAULT_CONNECTOME,
    data: TwinData | None = None,
) -> TwinFit:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TwinConfig(**payload["config"])
    data = data or _make_twin_data(
        cache, trials_path=trials_path, connectome_path=connectome_path, config=cfg
    )
    tf = data.target_features
    rf = data.responder_features
    pf = data.pair_features
    state = payload["scalers"]
    scalers = TwinScalers(
        target=state["target"],
        responder=state["responder"],
        pair=state["pair"],
        context=state["context"],
        pre=state["pre"],
        support=state["support"],
        y_mean=float(state["y_mean"]),
        y_std=float(state["y_std"]),
        traj_mean=np.asarray(state["traj_mean"], dtype=float),
        traj_std=np.asarray(state["traj_std"], dtype=float),
        traj_template=np.asarray(
            state.get("traj_template", np.zeros((rf.shape[0], cache.post.shape[-1]))), dtype=float
        ),
        responder_prior=np.asarray(state["responder_prior"], dtype=float),
        target_prior=np.asarray(state["target_prior"], dtype=float),
        target_seen=np.asarray(state["target_seen"], dtype=float),
    )
    module = ResponseTwin(
        d_target=tf.shape[1],
        d_responder=rf.shape[1],
        d_pair=pf.shape[2],
        d_context=data.context.shape[1],
        pre_frames=cache.pre.shape[-1],
        post_frames=cache.post.shape[-1],
        config=cfg,
        support_dim=cfg.support_dim,
    )
    module.load_state_dict(payload["state_dict"])
    module.eval()
    return TwinFit(module=module, data=data, scalers=scalers, config=cfg, split=split, history=[])


def fit_summary(fit: TwinFit) -> dict[str, Any]:
    return {
        "fold": fit.split.fold,
        "split": fit.split.split,
        "epochs_ran": len(fit.history),
        "best_val_loss": min((h["val_loss"] for h in fit.history), default=float("nan")),
        "n_train_tasks": len(fit.split.train_tasks),
        "n_val_tasks": len(fit.split.val_tasks),
        "n_test_tasks": len(fit.split.test_tasks),
        "config": asdict(fit.config),
    }


def fit_metadata_json(fit: TwinFit) -> str:
    return json.dumps(fit_summary(fit), indent=2)
