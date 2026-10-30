"""Tests for animal-cluster paired inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from neurobranch.benchmark.inference import inference_summary, paired_cluster_inference


def _predictions(n_animals: int = 16, n_targets: int = 4, n_neurons: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    rows: list[dict[str, object]] = []
    for animal in range(n_animals):
        for target in range(n_targets):
            for neuron in range(n_neurons):
                truth = rng.normal()
                for event in range(2):
                    rows.append(
                        {
                            "dataset_id": f"a{animal}",
                            "target": f"T{target}",
                            "neuron": f"N{neuron}",
                            "event_id": f"{animal}:{target}:{neuron}:{event}",
                            "fold": animal % 4,
                            "model": "baseline",
                            "k": 0,
                            "strategy": "none",
                            "y_true": truth + rng.normal(0, 0.3),
                            "y_pred": 0.0,
                        }
                    )
                    rows.append(
                        {
                            "dataset_id": f"a{animal}",
                            "target": f"T{target}",
                            "neuron": f"N{neuron}",
                            "event_id": f"{animal}:{target}:{neuron}:{event}",
                            "fold": animal % 4,
                            "model": "good",
                            "k": 4,
                            "strategy": "random",
                            "y_true": truth + rng.normal(0, 0.3),
                            "y_pred": truth + rng.normal(0, 0.1),
                        }
                    )
    return pd.DataFrame(rows)


def test_good_model_has_positive_paired_delta() -> None:
    frame = _predictions()
    frame = frame[frame["model"] != "baseline"].assign(y_pred=lambda d: d["y_true"])
    baseline = frame.copy()
    baseline["model"] = "baseline"
    baseline["y_pred"] = np.random.default_rng(2).normal(0, 0.5, size=len(baseline))
    combined = pd.concat([frame, baseline], ignore_index=True)
    out = paired_cluster_inference(combined, baseline="baseline", n_boot=200, seed=3)
    row = out[out["model"] == "good"].iloc[0]
    assert row["delta"] > 0.8
    assert row["delta_ci_low"] > 0
    assert row["status"] in {"supported", "underpowered"}
    assert row["n_animal_clusters"] == 16


def test_holm_adjustment_is_monotone_and_bounded() -> None:
    frame = _predictions()
    out = paired_cluster_inference(frame, baseline="baseline", n_boot=120, seed=5)
    adjusted = out["p_holm"].dropna().to_numpy()
    raw = out.loc[out["p_holm"].notna(), "p_value"].to_numpy()
    assert np.all(adjusted >= raw - 1e-12)
    assert np.all((adjusted >= 0) & (adjusted <= 1))
    summary = inference_summary(out)
    assert summary["n_configurations"] == len(out)
    assert summary["unit"] == "animal_cluster"


def test_single_cluster_is_not_given_an_interval() -> None:
    frame = _predictions(n_animals=1)
    out = paired_cluster_inference(frame, baseline="baseline", n_boot=100)
    assert out["status"].eq("insufficient_clusters").all()
    assert out["delta_ci_low"].isna().all()


def test_deterministic_seed() -> None:
    frame = _predictions()
    first = paired_cluster_inference(frame, baseline="baseline", n_boot=100, seed=9)
    second = paired_cluster_inference(frame, baseline="baseline", n_boot=100, seed=9)
    pd.testing.assert_frame_equal(first, second)
