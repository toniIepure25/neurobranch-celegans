"""BunDLe-Net reference-artifact integration and a transparent PyTorch reimplementation.

The upstream repository is TensorFlow/Keras.  We do not modify it and we do not
claim bitwise reproduction.  We (i) evaluate the official precomputed 3-D
embeddings shipped with the NC-MCM repository, and (ii) provide a PyTorch
reimplementation of the architecture and loss described in the upstream code.
All deviations are recorded in the returned metadata.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy import signal
from sklearn.manifold import trustworthiness
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from torch import nn

from neurobranch.config import PROJECT_ROOT
from neurobranch.data.kato import STATE_ORDER, load_kato
from neurobranch.dynamics.torch_utils import set_seed
from neurobranch.preprocessing import make_worm_data


def upstream_lowpass(X: np.ndarray, fs: float) -> np.ndarray:
    """Upstream 4th-order zero-phase low-pass (normalized 0.05).

    Kept for reference only.  It is acausal and would let held-out frames
    influence earlier inputs, so the held-out reimplementation uses
    :func:`causal_lowpass` instead.
    """
    cutoff_hz = 0.05 * fs / 2.0
    sos = signal.butter(4, cutoff_hz, "lowpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, X, axis=0)


def causal_lowpass(X: np.ndarray, fs: float) -> np.ndarray:
    """One-pass 4th-order Butterworth low-pass with the upstream cutoff.

    No future samples are used.  Compared with the upstream ``filtfilt`` this
    introduces a frequency-dependent phase delay; that is a documented deviation,
    not a hidden advantage.
    """
    cutoff_hz = 0.05 * fs / 2.0
    sos = signal.butter(4, cutoff_hz, "lowpass", fs=fs, output="sos")
    zi = signal.sosfilt_zi(sos)[:, :, None] * X[0][None, None, :]
    y, _ = signal.sosfilt(sos, X, axis=0, zi=zi)
    return y


class _GaussianNoise(nn.Module):
    def __init__(self, sigma: float):
        super().__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.sigma if self.training else x


class BunDLeLite(nn.Module):
    """Architecture from upstream functions.py, reimplemented in PyTorch."""

    def __init__(self, n_neurons: int, window: int = 15, latent_dim: int = 3, n_states: int = 8):
        super().__init__()
        in_dim = n_neurons * window
        self.tau = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 30),
            nn.ReLU(),
            nn.Linear(30, 25),
            nn.ReLU(),
            nn.Linear(25, 10),
            nn.ReLU(),
            nn.Linear(10, latent_dim),
            nn.Identity(),
            _GaussianNoise(0.05),
        )
        self.T_Y = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.Identity())
        self.predictor = nn.Linear(latent_dim, n_states)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # X: (B, 2, window, neurons)
        upper = self.tau(X[:, 1])
        lower0 = self.tau(X[:, 0])
        lower1 = lower0 + self.T_Y(lower0)
        return upper, lower1, self.predictor(upper)

    def encode(self, windows: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(windows, dtype=np.float32))
            return self.tau(t).cpu().numpy()


@dataclass
class BunDLeResult:
    latent: np.ndarray
    labels: np.ndarray
    frame_end: np.ndarray
    metrics: dict[str, Any]
    history: dict[str, Any]
    metadata: dict[str, Any]


def _windows(X: np.ndarray, endpoints: np.ndarray, window: int) -> np.ndarray:
    return np.stack([X[t - window + 1 : t + 1] for t in endpoints]).astype(np.float32)


def train_bundlenet_lite(
    worm_id: int = 0,
    *,
    window: int = 15,
    latent_dim: int = 3,
    epochs: int = 80,
    seed: int = 17,
) -> BunDLeResult:
    from neurobranch.config import ExperimentConfig

    set_seed(seed, num_threads=4)
    ds = load_kato()
    rec = ds[worm_id]
    cfg = ExperimentConfig()
    cfg.split.mode = "blocked"
    data = make_worm_data(rec, cfg.split, exclude_behavior=True)
    Xf = causal_lowpass(data.X.astype(np.float64), rec.fps)
    Xf = (Xf - Xf[data.masks["train"]].mean(0)) / np.maximum(Xf[data.masks["train"]].std(0), 1e-8)
    T = Xf.shape[0]
    fine_names = list(dict.fromkeys(rec.states))
    yfine = np.array([fine_names.index(s) for s in rec.states], dtype=np.int64)
    # Valid pair endpoint t: window t-14..t and next window t-13..t+1 both stay in split.
    candidates = np.arange(window - 1, T - 1)
    train_end = [t for t in candidates if data.masks["train"][t - window + 1 : t + 2].all()]
    val_end = [t for t in candidates if data.masks["val"][t - window + 1 : t + 2].all()]
    test_end = [t for t in candidates if data.masks["test"][t - window + 1 : t + 2].all()]
    train_end = np.asarray(train_end)
    val_end = np.asarray(val_end)
    test_end = np.asarray(test_end)
    if len(train_end) < 100:
        raise ValueError("Insufficient BunDLe-Net training windows")
    net = BunDLeLite(Xf.shape[1], window, latent_dim, len(fine_names))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    best = (float("inf"), -1, copy.deepcopy(net.state_dict()))
    history = {"train_loss": [], "val_loss": []}

    def make_pair(endpoints):
        X0 = np.stack([Xf[t - window + 1 : t + 1] for t in endpoints])
        X1 = np.stack([Xf[t - window + 2 : t + 2] for t in endpoints])
        return torch.from_numpy(np.stack([X0, X1], axis=1).astype(np.float32)), torch.from_numpy(
            yfine[endpoints + 1]
        )

    for epoch in range(epochs):
        net.train()
        order = rng.permutation(len(train_end))
        losses = []
        for lo in range(0, len(order), 128):
            idx = order[lo : lo + 128]
            X, y = make_pair(train_end[idx])
            up, low, logits = net(X)
            loss = 0.9 * nn.functional.mse_loss(up, low) + 0.1 * nn.functional.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        net.eval()
        with torch.no_grad():
            Xv, yv = make_pair(val_end)
            up, low, logits = net(Xv)
            val = float(0.9 * nn.functional.mse_loss(up, low) + 0.1 * nn.functional.cross_entropy(logits, yv))
        history["train_loss"].append(float(np.mean(losses)))
        history["val_loss"].append(val)
        if val < best[0]:
            best = (val, epoch, copy.deepcopy(net.state_dict()))
    # Upstream trains for a fixed 2000 epochs without early stopping; retaining
    # the final weights is therefore the faithful choice.  The best-validation
    # epoch is reported for transparency only.
    net.eval()

    # Embed every frame using its own preceding window, padding the first frames.
    all_end = np.arange(window - 1, T)
    windows_all = _windows(Xf, all_end, window)
    Z = net.encode(windows_all)
    labels_fine = yfine[all_end]
    metrics = evaluate_embedding(
        Z, labels_fine, all_end, data.masks, k=15, seed=seed, high_dim=windows_all.reshape(len(windows_all), -1)
    )
    metrics.update(
        {
            "model": "bundlenet_lite_pytorch",
            "best_epoch": best[1],
            "best_val_loss": best[0],
            "window": window,
            "latent_dim": latent_dim,
            "n_train_windows": int(len(train_end)),
            "n_test_windows": int(len(test_end)),
        }
    )
    metadata = {
        "faithful_elements": [
            "flattened 15-frame window encoder 50-30-25-10-3",
            "residual latent transition T_Y",
            "linear 8-state predictor",
            "gamma=0.9 weighted Markov-consistency + behaviour cross-entropy",
            "Gaussian noise sigma=0.05 during training",
        ],
        "deviations": [
            "PyTorch rather than TensorFlow/Keras",
            "Keras Normalization is left unadapted in upstream code and is therefore identity",
            "blocked split rather than upstream 7-fold random KFold",
            (
                "causal one-pass 4th-order low-pass at the upstream 0.05 normalized cutoff; "
                "upstream uses zero-phase filtfilt, which is acausal and would leak held-out "
                "frames into earlier inputs"
            ),
            "no PCA initialisation / best-of-5 initialisation",
            f"{epochs} epochs vs upstream 2000",
        ],
    }
    result = BunDLeResult(
        latent=Z,
        labels=labels_fine,
        frame_end=all_end,
        metrics=metrics,
        history=history,
        metadata=metadata,
    )
    out = PROJECT_ROOT / "artifacts" / "reproduction"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"bundlenet_lite_worm{worm_id}.npz", latent=Z, labels=labels_fine, frame_end=all_end
    )
    return result


def evaluate_embedding(
    Z: np.ndarray,
    labels: np.ndarray,
    frame_end: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    k: int = 15,
    seed: int = 17,
    high_dim: np.ndarray | None = None,
) -> dict[str, Any]:
    frame_test = masks["test"][frame_end]
    frame_train = masks["train"][frame_end]
    out: dict[str, Any] = {"n_embedded": int(len(Z))}
    if frame_train.sum() > 20 and frame_test.sum() > 20:
        clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
        clf.fit(Z[frame_train], labels[frame_train])
        pred = clf.predict(Z[frame_test])
        out["knn_accuracy"] = float(accuracy_score(labels[frame_test], pred))
        out["knn_macro_f1"] = float(f1_score(labels[frame_test], pred, average="macro", zero_division=0))
        out["n_train"] = int(frame_train.sum())
        out["n_test"] = int(frame_test.sum())
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) > 2 and counts.min() >= 2 and len(Z) < 6000:
        try:
            out["silhouette_fine"] = float(silhouette_score(Z, labels))
        except Exception:
            out["silhouette_fine"] = None
        if high_dim is not None:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(Z), size=min(1200, len(Z)), replace=False)
            out["trustworthiness"] = float(
                trustworthiness(np.asarray(high_dim)[idx], Z[idx], n_neighbors=10)
            )
    return out


def load_official_embedding(worm_id: int = 0) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    p = PROJECT_ROOT / "references" / "external" / "nc-mcm" / "precomputed_results" / (
        f"bundlenet_consistent_embedding_worm_{worm_id}.npz"
    )
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    Z = d["x"].astype(np.float64)
    b = d["b"].ravel().astype(int)
    metadata = {
        "source": str(p.relative_to(PROJECT_ROOT)),
        "n": int(len(Z)),
        "note": "Official artifact distributed with the NC-MCM repository; not retrained here.",
    }
    return Z, b, metadata


def official_vs_reimplementation(worm_id: int = 0, *, seed: int = 17) -> dict[str, Any]:
    ds = load_kato()
    rec = ds[worm_id]
    official = load_official_embedding(worm_id)
    if official is None:
        return {"available": False, "reason": "Official precomputed embedding missing"}
    Zoff, boff, meta = official
    # Official b is the upstream fine-state code (0..7).  Compute descriptive
    # 3-D geometry and nearest-neighbour label consistency without refitting.
    from sklearn.metrics import silhouette_score

    out: dict[str, Any] = {
        "available": True,
        "worm_id": worm_id,
        "official": meta,
        "evaluation_note": (
            "The embedding was fitted upstream on all frames. The kNN classifier is fit on "
            "train labels and scored on test labels, but the test embeddings were seen during "
            "upstream training, so this is embedding-level leakage, not held-out generalization."
        ),
    }
    vals, counts = np.unique(boff, return_counts=True)
    out["official_label_counts"] = {STATE_ORDER[i]: int(c) for i, c in zip(vals, counts) if i < len(STATE_ORDER)}
    if len(vals) > 2 and counts.min() > 1:
        try:
            out["official_silhouette_fine"] = float(silhouette_score(Zoff, boff))
        except Exception:
            out["official_silhouette_fine"] = None
    # Step smoothness of the 3-D trajectory.
    steps = np.linalg.norm(np.diff(Zoff, axis=0), axis=1)
    out["official_median_step"] = float(np.median(steps))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Zoff), size=min(2000, len(Zoff)), replace=False)
    out["official_trajectory_extent"] = float(np.ptp(Zoff[idx, 0]))
    from sklearn.neighbors import KNeighborsClassifier

    from neurobranch.config import ExperimentConfig
    from neurobranch.preprocessing import make_worm_data

    data = make_worm_data(rec, ExperimentConfig().split)
    end = np.arange(16, rec.n_frames)
    if len(end) == len(Zoff):
        tr = data.masks["train"][end]
        te = data.masks["test"][end]
        clf = KNeighborsClassifier(n_neighbors=15, weights="distance").fit(Zoff[tr], boff[tr])
        pred = clf.predict(Zoff[te])
        out["official_knn_accuracy"] = float(accuracy_score(boff[te], pred))
        out["official_knn_macro_f1"] = float(f1_score(boff[te], pred, average="macro", zero_division=0))
        out["official_knn_n_train"] = int(tr.sum())
        out["official_knn_n_test"] = int(te.sum())
    # PCA-3D baseline on the same full trace for context.
    from sklearn.decomposition import PCA

    X = rec.traces[:, [i for i, n in enumerate(rec.neurons) if n not in {
        "AVAL", "AVAR", "SMDDR", "SMDDL", "SMDVR", "SMDVL", "RIBR", "RIBL"
    }]]
    Zp = PCA(n_components=3, random_state=0).fit_transform(X)
    out["pca3_silhouette_fine"] = (
        float(silhouette_score(Zp[16:], boff[: len(Zp) - 16]))
        if len(boff) == len(Zp) - 16
        else None
    )
    return out
