"""Secondary nonlinear sensitivity analysis for turn direction.

The primary benchmark is a fixed logistic model.  This script repeats the same
animal-disjoint folds with a small gradient-boosted tree model to ask whether
the null increment is an artefact of an insufficient linear representation.
It is secondary and is not used to select the primary result.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from neurobranch.behavior.turn_direction import (
    _fold_identified_labels,
    _matrices,
    _metrics,
    _summarise,
    build_designs,
)
from neurobranch.data.kramer import load_all

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "phase4"


def main() -> None:
    recs, _ = load_all(cohorts=("wt_octanol", "wt_diacetyl"))
    designs, labels = build_designs(recs, window="mid")
    # Focused nonlinear sensitivity check against the primary contrast.
    modes = ("behavior", "behavior_population")
    pred_rows = []
    for i, test in enumerate(designs):
        train = [d for j, d in enumerate(designs) if j != i]
        identified_idx = _fold_identified_labels(train, labels)
        ytr = np.concatenate([d.y for d in train])
        for mode in modes:
            xtr, xte = _matrices(train, test, mode, labels, identified_idx=identified_idx)
            model = HistGradientBoostingClassifier(
                max_iter=120,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=17,
            )
            model.fit(xtr, ytr)
            p = model.predict_proba(xte)[:, 1]
            for k, row in enumerate(test.events.itertuples(index=False)):
                pred_rows.append(
                    {
                        "animal_id": test.animal_id,
                        "event_id": row.event_id,
                        "mode": mode,
                        "y": int(test.y[k]),
                        "p": float(p[k]),
                    }
                )
    preds = pd.DataFrame(pred_rows)
    preds.to_parquet(REPORT / "turn_direction_nonlinear_predictions.parquet", index=False)
    metrics = pd.DataFrame(
        [{"mode": mode, **_metrics(g["y"].to_numpy(), g["p"].to_numpy())} for mode, g in preds.groupby("mode")]
    )
    metrics.to_csv(REPORT / "turn_direction_nonlinear.csv", index=False)
    summary = _summarise(preds, seed=17, n_boot=2000)
    (REPORT / "turn_direction_nonlinear_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
