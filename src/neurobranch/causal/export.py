"""Export a portable, explicitly model-dependent response operator.

The exported matrix is not a connectome and not a causal effect.  It is the CRO's
predicted mean response for every target-responder pair that was evaluated, with
interval width and observation count.  Unmeasured pairs are NaN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neurobranch.causal.features import FeatureBundle


def export_operator(
    bundle: FeatureBundle,
    predictions: pd.DataFrame,
    *,
    operator_path: str | Path,
    uncertainty_path: str | Path,
) -> dict[str, int]:
    cro = predictions[predictions["model"] == "cro"].copy()
    if cro.empty:
        raise ValueError("no CRO predictions to export")
    cro["target_idx"] = cro["target"].map({n: i for i, n in enumerate(bundle.target_names)})
    cro["responder_idx"] = cro["neuron"].map({n: i for i, n in enumerate(bundle.responder_names)})
    pair = cro.groupby(["target_idx", "responder_idx"], as_index=False).agg(
        predicted=("y_pred", "mean"),
        observed=("y_true", "mean"),
        lower=("lower", "mean"),
        upper=("upper", "mean"),
        n_events=("y_true", "size"),
    )
    pair["width"] = pair["upper"] - pair["lower"]
    shape = (len(bundle.target_names), len(bundle.responder_names))
    effect = np.full(shape, np.nan, dtype=np.float32)
    observed = np.full(shape, np.nan, dtype=np.float32)
    width = np.full(shape, np.nan, dtype=np.float32)
    count = np.zeros(shape, dtype=np.int32)
    effect[pair["target_idx"], pair["responder_idx"]] = pair["predicted"]
    observed[pair["target_idx"], pair["responder_idx"]] = pair["observed"]
    width[pair["target_idx"], pair["responder_idx"]] = pair["width"]
    count[pair["target_idx"], pair["responder_idx"]] = pair["n_events"]
    np.savez_compressed(
        operator_path,
        target_names=np.asarray(bundle.target_names, dtype=object),
        responder_names=np.asarray(bundle.responder_names, dtype=object),
        predicted_effect=effect,
        observed_effect=observed,
        interval_width=width,
        n_events=count,
        warning=np.asarray(
            "MODEL-DEPENDENT VIRTUAL PERTURBATION - NOT CAUSALLY VALIDATED",
            dtype=object,
        ),
    )
    np.savez_compressed(
        uncertainty_path,
        target_names=np.asarray(bundle.target_names, dtype=object),
        responder_names=np.asarray(bundle.responder_names, dtype=object),
        interval_width=width,
        n_events=count,
    )
    return {
        "n_targets": len(bundle.target_names),
        "n_responders": len(bundle.responder_names),
        "n_measured_pairs": int(np.isfinite(effect).sum()),
    }
