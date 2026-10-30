"""Small simulation audit of the animal-cluster paired inference.

This does not validate the biological claim.  It checks the statistical
procedure under a known synthetic model: type I error, interval coverage and
power, plus the real saved-prediction alignment audit.  The synthetic data use
the same event truth for baseline and model, so a failure here is about the
bootstrap procedure rather than about a known pairing defect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neurobranch.benchmark.inference import (
    audit_paired_inputs,
    paired_cluster_inference,
)
from neurobranch.benchmark.metrics import spearman

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "reports" / "phase8" / "predictions.parquet"
OUT = ROOT / "reports" / "phase12r" / "inference_validation.json"


def _synthetic(
    *,
    signal: float,
    seed: int,
    n_animals: int = 30,
    n_targets: int = 6,
    n_neurons: int = 8,
    n_events: int = 2,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for animal in range(n_animals):
        for target in range(n_targets):
            for neuron in range(n_neurons):
                truth = float(rng.normal())
                for event in range(n_events):
                    event_truth = truth + float(rng.normal(0.0, 0.2))
                    event_id = f"{animal}:{target}:{neuron}:{event}"
                    common = {
                        "dataset_id": f"a{animal}",
                        "target": f"T{target}",
                        "neuron": f"N{neuron}",
                        "event_id": event_id,
                        "fold": animal % 5,
                        "y_true": event_truth,
                    }
                    rows.append(
                        {
                            **common,
                            "model": "baseline",
                            "k": 0,
                            "strategy": "none",
                            "y_pred": float(rng.normal()),
                        }
                    )
                    rows.append(
                        {
                            **common,
                            "model": "model",
                            "k": 4,
                            "strategy": "random",
                            "y_pred": signal * truth + float(rng.normal()),
                        }
                    )
    return pd.DataFrame(rows)


def _population_delta(signal: float, *, seed: int = 90210, n: int = 200_000, n_events: int = 2) -> float:
    """Population delta after the same event averaging used by the fit."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=n)
    event_truth = truth[:, None] + rng.normal(0.0, 0.2, size=(n, n_events))
    observed_truth = event_truth.mean(axis=1)
    model = (signal * truth[:, None] + rng.normal(size=(n, n_events))).mean(axis=1)
    baseline = rng.normal(size=(n, n_events)).mean(axis=1)
    return float(
        spearman(pd.Series(observed_truth), pd.Series(model))
        - spearman(pd.Series(observed_truth), pd.Series(baseline))
    )


def _one_run(frame: pd.DataFrame, *, seed: int, n_boot: int = 200) -> dict[str, Any]:
    out = paired_cluster_inference(frame, baseline="baseline", n_boot=n_boot, seed=seed)
    row = out.iloc[0]
    return {
        "delta": float(row["delta"]),
        "ci_low": float(row["delta_ci_low"]),
        "ci_high": float(row["delta_ci_high"]),
        "p_value": float(row["p_value"]),
        "mde": float(row["mde_80"]),
        "reject": bool(float(row["p_value"]) < 0.05),
        "ci_excludes_zero": bool(float(row["delta_ci_low"]) > 0.0 or float(row["delta_ci_high"]) < 0.0),
    }


def validate(*, n_reps: int = 120, n_boot: int = 200) -> dict[str, Any]:
    null_rows: list[dict[str, Any]] = []
    for rep in range(n_reps):
        frame = _synthetic(signal=0.0, seed=10_000 + rep)
        null_rows.append(_one_run(frame, seed=20_000 + rep, n_boot=n_boot))
    null = pd.DataFrame(null_rows)
    effects = (0.0, 0.1, 0.2, 0.4)
    curve: list[dict[str, Any]] = []
    for effect in effects:
        truth = _population_delta(effect)
        rows = [
            _one_run(_synthetic(signal=effect, seed=30_000 + rep), seed=40_000 + rep, n_boot=n_boot)
            for rep in range(n_reps)
        ]
        frame = pd.DataFrame(rows)
        covered = (frame["ci_low"] <= truth) & (truth <= frame["ci_high"])
        curve.append(
            {
                "signal": effect,
                "population_delta": truth,
                "n_reps": int(n_reps),
                "mean_estimate": float(frame["delta"].mean()),
                "bias": float(frame["delta"].mean() - truth),
                "coverage": float(covered.mean()),
                "power_ci": float(frame["ci_excludes_zero"].mean()),
                "power_p": float(frame["reject"].mean()),
                "mean_mde": float(frame["mde"].mean()),
            }
        )
    return {
        "status": "ok",
        "method": "synthetic paired event data; animal-cluster bootstrap",
        "n_reps": int(n_reps),
        "n_boot": int(n_boot),
        "null_type_i_p": float(null["reject"].mean()),
        "null_type_i_ci": float(null["ci_excludes_zero"].mean()),
        "null_mean_delta": float(null["delta"].mean()),
        "curve": curve,
        "caveat": "Synthetic procedure check, not a biological validation; MDE is a normal-approximation diagnostic.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    payload: dict[str, Any] = {}
    if PREDICTIONS.is_file():
        payload["alignment_audit"] = audit_paired_inputs(pd.read_parquet(PREDICTIONS))
    else:
        payload["alignment_audit"] = {"status": "missing_predictions"}
    payload["simulation"] = validate(
        n_reps=30 if args.quick else 120,
        n_boot=100 if args.quick else 200,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
