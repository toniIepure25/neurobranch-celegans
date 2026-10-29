"""Training-only design audit for Phase 8 (no test outcomes)."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.config import PROJECT_ROOT, resolve_path
from neurobranch.twin.tasks import build_animal_tasks
from neurobranch.twin.windows import load_window_cache


def _eta2(values: np.ndarray, groups: np.ndarray) -> float:
    frame = pd.DataFrame({"y": values, "g": groups}).dropna()
    if frame.empty or frame["y"].var() == 0:
        return float("nan")
    grand = frame["y"].mean()
    between = frame.groupby("g")["y"].agg(["mean", "count"])
    ss_between = float((between["count"] * (between["mean"] - grand) ** 2).sum())
    ss_total = float(((frame["y"] - grand) ** 2).sum())
    return ss_between / ss_total if ss_total > 0 else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/phase8/design.json")
    parser.add_argument("--split-seed", type=int, default=41)
    args = parser.parse_args()
    cache = load_window_cache(resolve_path("reports/phase8/cache/windows.npz", PROJECT_ROOT))
    tasks = build_animal_tasks(cache, split="unseen_animals", n_folds=5, seed=args.split_seed)
    train_events = sorted({e for split in tasks for t in split.train_tasks.values() for e in t.test_events})
    train_events += sorted({e for split in tasks for t in split.train_tasks.values() for e in t.support_events})
    lookup = {str(e): i for i, e in enumerate(cache.event_ids.astype(str))}
    rows = [lookup[e] for e in train_events if e in lookup]
    if not rows:
        raise SystemExit("no training events")
    ridx = np.concatenate([np.flatnonzero(cache.mask[i]) for i in rows])
    ev = np.repeat(np.asarray(rows), [int(cache.mask[i].sum()) for i in rows])
    y = cache.scalar[ev, ridx].astype(float)
    targets = np.array([cache.target_names[int(cache.target_idx[i])] for i in ev])
    animals = cache.dataset_ids[ev].astype(str)
    responder_names = np.asarray(cache.responder_names, dtype=object)
    pair = np.array([f"{t}|{r}" for t, r in zip(targets, responder_names[ridx], strict=False)])
    # Per-event mean history probe; avoids pseudo-replication across responders.
    event_y = pd.DataFrame({"event": cache.event_ids[ev].astype(str), "y": y}).groupby("event")["y"].mean()
    history_rows = []
    for j, name in enumerate(cache.history_names):
        hist = pd.Series(cache.history[np.unique(ev), j], index=cache.event_ids[np.unique(ev)].astype(str))
        joined = pd.concat([event_y, hist.rename("h")], axis=1).dropna()
        corr = joined["y"].corr(joined["h"], method="spearman") if len(joined) > 3 else float("nan")
        history_rows.append({"feature": name, "spearman_with_event_mean": float(corr)})
    result: dict[str, Any] = {
        "split_seed": args.split_seed,
        "n_train_events": len(rows),
        "n_train_rows": int(len(y)),
        "n_animals": int(len(np.unique(animals))),
        "n_targets": int(len(np.unique(targets))),
        "n_pairs": int(len(np.unique(pair))),
        "eta2_animal": _eta2(y, animals),
        "eta2_target": _eta2(y, targets),
        "eta2_pair": _eta2(y, pair),
        "history_probe": history_rows,
        "power_note": (
            "Animal eta2 is computed on training rows only. A small value bounds "
            "the expected gain from animal-level support calibration; H2/H3 are "
            "reported with intervals even when the gate is not met."
        ),
        "warning": "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED.",
    }
    def _safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_safe(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, (np.floating, np.integer)):
            return _safe(value.item())
        return value

    result = _safe(result)
    out = resolve_path(args.output, PROJECT_ROOT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
