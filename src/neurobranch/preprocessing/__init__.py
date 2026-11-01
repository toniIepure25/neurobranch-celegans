"""Leakage-safe preprocessing and time-respecting split construction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from neurobranch.config import SplitConfig
from neurobranch.data.kato import (
    MACRO_ORDER,
    WormRecording,
    predictor_indices,
)


@dataclass
class WormData:
    recording: WormRecording
    X: np.ndarray  # standardized predictor traces, (T, N)
    X_raw: np.ndarray  # unstandardized predictor traces, (T, N)
    y_macro: np.ndarray  # integer macro behaviour
    y_fine: np.ndarray  # integer fine state
    predictor_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    masks: dict[str, np.ndarray]
    split_bounds: dict[str, tuple[int, int]]
    excluded_neurons: tuple[str, ...]
    external_input: np.ndarray | None = None
    input_name: str = ""
    time_seconds: np.ndarray | None = None

    def with_external_input(
        self,
        values: np.ndarray,
        *,
        name: str,
        time_seconds: np.ndarray | None = None,
    ) -> WormData:
        """Attach a known external schedule without changing any neural data."""
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(values) != self.n_frames:
            raise ValueError(
                f"external input length {len(values)} does not match {self.n_frames} frames"
            )
        if not np.isfinite(values).all():
            raise ValueError("external input contains non-finite values")
        self.external_input = values
        self.input_name = str(name)
        if time_seconds is not None:
            time_seconds = np.asarray(time_seconds, dtype=np.float64).reshape(-1)
            if len(time_seconds) != self.n_frames:
                raise ValueError("time_seconds length does not match frames")
            self.time_seconds = time_seconds
        return self

    @property
    def worm_id(self) -> int:
        return self.recording.worm_id

    @property
    def n_frames(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_neurons(self) -> int:
        return int(self.X.shape[1])

    def neuron_index(self, name: str) -> int:
        try:
            return self.predictor_names.index(name)
        except ValueError as exc:
            if name in self.excluded_neurons:
                raise KeyError(
                    f"{name!r} was excluded from predictors because it contributed to behaviour labels"
                ) from exc
            raise KeyError(f"{name!r} is not a predictor neuron in worm {self.worm_id}") from exc

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def _smooth(X: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return X
    return gaussian_filter1d(X, sigma=sigma, axis=0, mode="nearest")


def causal_moving_average(X: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average; uses only current and past frames."""
    if window <= 1:
        return X.copy()
    T = X.shape[0]
    csum = np.cumsum(np.vstack([np.zeros((1, X.shape[1])), X]), axis=0)
    idx = np.arange(1, T + 1)
    lo = np.maximum(0, idx - window)
    return (csum[idx] - csum[lo]) / (idx - lo)[:, None]


def causal_highpass_segment(X: np.ndarray, window: int) -> np.ndarray:
    """Remove slow drift within a split without crossing its boundaries."""
    return X - causal_moving_average(X, window)


def _mask_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of True as half-open intervals."""
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def make_worm_data(
    rec: WormRecording,
    split_cfg: SplitConfig | None = None,
    *,
    exclude_behavior: bool = True,
    smoothing_sigma_frames: float = 0.0,
    detrend_window_frames: int = 0,
    detrend_warmup_frames: int = 0,
) -> WormData:
    cfg = split_cfg or SplitConfig()
    idx = predictor_indices(rec, exclude_behavior=exclude_behavior)
    excluded = tuple(n for n, keep in zip(rec.neurons, np.isin(np.arange(rec.n_neurons), idx)) if not keep)
    names = tuple(rec.neurons[i] for i in idx)
    raw = _smooth(rec.traces[:, idx].astype(np.float64), smoothing_sigma_frames)
    T = raw.shape[0]
    train_mask = np.zeros(T, bool)
    val_mask = np.zeros(T, bool)
    test_mask = np.zeros(T, bool)
    if cfg.mode == "temporal":
        n_train = max(1, int(np.floor(cfg.train_fraction * T)))
        n_val = max(1, int(np.floor(cfg.val_fraction * T)))
        train_end = n_train
        val_start = min(T, train_end + cfg.purge_frames)
        val_end = min(T, val_start + n_val)
        test_start = min(T, val_end + cfg.purge_frames)
        train_mask[:train_end] = True
        val_mask[val_start:val_end] = True
        test_mask[test_start:] = True
        bounds = {"train": (0, train_end), "val": (val_start, val_end), "test": (test_start, T)}
    elif cfg.mode == "blocked":
        blocks = np.array_split(np.arange(T), max(3, cfg.n_blocks))
        if cfg.test_block >= len(blocks) or cfg.val_block >= len(blocks) or cfg.test_block == cfg.val_block:
            raise ValueError("blocked split requires distinct valid test_block and val_block indices")
        for bi, block in enumerate(blocks):
            if bi == cfg.test_block:
                test_mask[block] = True
            elif bi == cfg.val_block:
                val_mask[block] = True
            else:
                train_mask[block] = True
        bounds = {
            "train": (int(np.flatnonzero(train_mask)[0]), int(np.flatnonzero(train_mask)[-1] + 1)),
            "val": (int(blocks[cfg.val_block][0]), int(blocks[cfg.val_block][-1] + 1)),
            "test": (int(blocks[cfg.test_block][0]), int(blocks[cfg.test_block][-1] + 1)),
        }
    else:
        raise ValueError(f"Unsupported split mode {cfg.mode!r} in make_worm_data")
    if detrend_window_frames and detrend_window_frames > 1:
        processed = raw.copy()
        for mask in (train_mask, val_mask, test_mask):
            for lo, hi in _mask_segments(mask):
                processed[lo:hi] = causal_highpass_segment(raw[lo:hi], detrend_window_frames)
                warm = min(detrend_warmup_frames, max(0, (hi - lo) // 4))
                mask[lo : lo + warm] = False
        raw = processed
    mean = raw[train_mask].mean(axis=0)
    std = raw[train_mask].std(axis=0)
    std[std < 1e-8] = 1.0
    X = (raw - mean) / std
    macro = rec.macro_labels()
    macro_idx = np.array([MACRO_ORDER.index(s) for s in macro], dtype=np.int64)
    fine_names = list(dict.fromkeys(rec.states))
    fine_idx = np.array([fine_names.index(s) for s in rec.states], dtype=np.int64)
    return WormData(
        recording=rec,
        X=X.astype(np.float32),
        X_raw=raw.astype(np.float32),
        y_macro=macro_idx,
        y_fine=fine_idx,
        predictor_names=names,
        mean=mean,
        std=std,
        masks={"train": train_mask, "val": val_mask, "test": test_mask},
        split_bounds=bounds,
        excluded_neurons=excluded,
    )


def prepare_all(
    recordings: Iterable[WormRecording],
    split_cfg: SplitConfig | None = None,
    *,
    exclude_behavior: bool = True,
    smoothing_sigma_frames: float = 0.0,
    detrend_window_frames: int = 0,
    detrend_warmup_frames: int = 0,
) -> list[WormData]:
    return [
        make_worm_data(
            r,
            split_cfg,
            exclude_behavior=exclude_behavior,
            smoothing_sigma_frames=smoothing_sigma_frames,
            detrend_window_frames=detrend_window_frames,
            detrend_warmup_frames=detrend_warmup_frames,
        )
        for r in recordings
    ]


def valid_start_indices(
    T: int,
    *,
    history: int = 1,
    horizon: int = 1,
    allowed: np.ndarray | None = None,
) -> np.ndarray:
    """Start indices whose full past/future window lies inside ``allowed``.

    This is the key leakage guard: a sample is usable only if every frame of
    its history and forecast horizon belongs to the same split.
    """
    low = max(0, history - 1)
    high = T - horizon
    candidate = np.arange(low, high)
    if allowed is None:
        return candidate
    ok = np.ones_like(candidate, dtype=bool)
    for t in candidate:
        lo = t - history + 1
        if not allowed[lo : t + horizon + 1].all():
            ok[t - low] = False
    return candidate[ok]


def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    starts: np.ndarray,
    *,
    history: int,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack windows: (B, history, N), target (B, N), target labels (B,)."""
    B = len(starts)
    Xh = np.empty((B, history, X.shape[1]), dtype=np.float32)
    Yh = np.empty((B, X.shape[1]), dtype=np.float32)
    yh = np.empty(B, dtype=np.int64)
    for j, t in enumerate(starts):
        Xh[j] = X[t - history + 1 : t + 1]
        Yh[j] = X[t + horizon]
        yh[j] = y[t + horizon]
    return Xh, Yh, yh


def split_starts(data: WormData, split: str, *, history: int, horizon: int) -> np.ndarray:
    mask = data.masks[split]
    return valid_start_indices(data.n_frames, history=history, horizon=horizon, allowed=mask)


def common_identified_neurons(datasets: Iterable[WormData], min_worms: int | None = None) -> list[str]:
    """Names of identified cells present in at least ``min_worms`` recordings."""
    names: dict[str, int] = {}
    sets = []
    for d in datasets:
        s = {n for n in d.predictor_names if any(ch.isalpha() for ch in n)}
        sets.append(s)
        for n in s:
            names[n] = names.get(n, 0) + 1
    threshold = min_worms or len(sets)
    return sorted(n for n, c in names.items() if c >= threshold)
