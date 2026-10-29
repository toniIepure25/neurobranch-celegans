"""Pseudo-prospective sequence rollout for the Phase 8 twin.

This is not a counterfactual simulation.  At each event the model is allowed to
see outcomes that a real sequential experiment would already have observed:
the support set plus the previous ``horizon`` test events.  It predicts the next
post-stimulus trajectory before seeing that event's response.  The protocol is
therefore prospective in event order, but it remains model-dependent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from neurobranch.twin.evaluate import add_intervals, trajectory_pearson
from neurobranch.twin.tasks import AnimalTask
from neurobranch.twin.train import TwinFit


@dataclass
class SequenceMetrics:
    trajectory_twin: float
    trajectory_persistence: float
    trajectory_delta: float
    coverage: float
    n_events: int


def _ensemble_prediction(
    fits: list[TwinFit], events: list[str], support: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mus = []
    sigmas = []
    trajs = []
    for fit in fits:
        pred, _ = fit.predict_events(events, support_events=support)
        mus.append(pred.scalar_mu)
        sigmas.append(pred.scalar_sigma)
        trajs.append(pred.trajectory_mu)
    mu = np.mean(mus, axis=0)
    var = np.mean([s**2 + (m - mu) ** 2 for s, m in zip(sigmas, mus, strict=False)], axis=0)
    return mu, np.sqrt(var), np.mean(trajs, axis=0)


def sequence_rollout(
    fits: list[TwinFit],
    task: AnimalTask,
    *,
    horizon: int = 3,
    scale: float = 1.0,
) -> SequenceMetrics:
    fit = fits[0]
    cache = fit.data.cache
    lookup = fit.data.event_lookup()
    order = sorted(
        [e for e in task.test_events if e in lookup],
        key=lambda e: float(cache.event_index[lookup[e]]),
    )
    observed = list(task.support_events)
    # The fit already stores the train-only per-responder trajectory template.
    # Recomputing it per task was quadratic and dominated the summary wall time.
    template = np.asarray(fit.scalers.traj_template, dtype=float)
    true_rows: list[np.ndarray] = []
    twin_rows: list[np.ndarray] = []
    persist_rows: list[np.ndarray] = []
    coverage_rows: list[pd.DataFrame] = []
    previous_test: str | None = None
    for event in order:
        support = list(task.support_events) + observed[-horizon:] if observed else list(task.support_events)
        mu, sigma, traj = _ensemble_prediction(fits, [event], support)
        ev, ridx = fit.data.rows_for_events([event])
        if ev.size == 0:
            continue
        true = cache.post[ev, ridx, :]
        true_rows.append(true)
        twin_rows.append(traj)
        if previous_test is not None:
            pi = lookup[previous_test]
            prev = cache.post[pi, ridx, :]
            persist_rows.append(prev)
        else:
            persist_rows.append(template[ridx])
        frame = pd.DataFrame(
            {
                "y_true": cache.scalar[ev, ridx],
                "y_pred": mu,
                "sigma": sigma,
            }
        )
        coverage_rows.append(add_intervals(frame, df=fit.config.df, scale=scale))
        observed.append(event)
        previous_test = event
    if not true_rows:
        return SequenceMetrics(float("nan"), float("nan"), float("nan"), float("nan"), 0)
    twin = trajectory_pearson(np.concatenate(true_rows), np.concatenate(twin_rows))
    persist = trajectory_pearson(np.concatenate(true_rows), np.concatenate(persist_rows))
    cov_frame = pd.concat(coverage_rows, ignore_index=True)
    coverage = float(
        np.mean((cov_frame["y_true"] >= cov_frame["interval_low"]) & (cov_frame["y_true"] <= cov_frame["interval_high"]))
    )
    return SequenceMetrics(
        trajectory_twin=float(twin),
        trajectory_persistence=float(persist),
        trajectory_delta=float(twin - persist),
        coverage=coverage,
        n_events=len(true_rows),
    )


def sequence_metrics_to_dict(m: SequenceMetrics) -> dict[str, float]:
    return {
        "trajectory_twin": m.trajectory_twin,
        "trajectory_persistence": m.trajectory_persistence,
        "trajectory_delta": m.trajectory_delta,
        "coverage": m.coverage,
        "n_events": float(m.n_events),
    }
