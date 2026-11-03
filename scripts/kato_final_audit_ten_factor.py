"""Audit what the ten-factor B5 control can and cannot represent.

The diagnostic is descriptive and uses only causal windows ending at each
forecast landmark.  It does not fit or select a new model on test outcomes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import jd_evaluate as JD  # noqa: E402

from neurobranch.adaptive.evaluation import evaluation_landmarks  # noqa: E402
from neurobranch.diagnostics.common import load_nostim  # noqa: E402
from neurobranch.kato_interactions.models import fit_window  # noqa: E402

OUT = ROOT / "reports" / "simulator" / "kato_final_audit"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _, data, _ = load_nostim()
    rows: list[dict] = []
    session_rows: list[dict] = []
    for d in data:
        pca, _, _ = JD.prepare(d)
        loading = np.asarray(pca.components_, dtype=np.float64)
        residual_projector = np.eye(d.X.shape[1]) - loading.T @ loading
        landmarks = evaluation_landmarks(d, horizon_max=100, stride=10)
        cross_ratios = []
        offdiag_ratios = []
        for t in landmarks:
            b3 = fit_window(d.X, t, window=500, ridge=1.0, spectral_cap=0.995)
            a3 = np.asarray(b3.A, dtype=np.float64)
            r3 = a3 @ residual_projector
            cross3 = loading @ r3
            within3 = loading @ a3 @ loading.T
            total3 = loading @ a3
            cross_ratios.append(float(np.linalg.norm(cross3) / max(np.linalg.norm(within3), 1e-15)))
            # B5's residual-source row for output j is proportional to Q[j].
            # Measure how much of B3's residual-source row lies outside that
            # one-dimensional own-row pattern.
            unexplained = []
            for j in range(d.X.shape[1]):
                q = residual_projector[j]
                r = r3[j]
                alpha = float(r @ q / max(q @ q, 1e-15))
                unexplained.append(float(np.linalg.norm(r - alpha * q) / max(np.linalg.norm(r), 1e-15)))
            offdiag_ratios.append(float(np.median(unexplained)))
            rows.append(
                {
                    "session": int(d.worm_id),
                    "landmark": int(t),
                    "b3_cross_over_within": cross_ratios[-1],
                    "b3_residual_unexplained_ratio_median": offdiag_ratios[-1],
                    "b3_full_norm": float(np.linalg.norm(total3)),
                    "b3_cross_norm": float(np.linalg.norm(cross3)),
                    "b3_within_norm": float(np.linalg.norm(within3)),
                }
            )
        session_rows.append(
            {
                "session": int(d.worm_id),
                "n_landmarks": int(len(landmarks)),
                "b3_cross_over_within_median": float(np.median(cross_ratios)),
                "b3_cross_over_within_min": float(np.min(cross_ratios)),
                "b3_cross_over_within_max": float(np.max(cross_ratios)),
                "b3_residual_unexplained_ratio_median": float(np.median(offdiag_ratios)),
                "pca_variance_10": float(pca.explained_variance_ratio_.sum()),
            }
        )
    frame = pd.DataFrame(rows)
    sessions = pd.DataFrame(session_rows)
    frame.to_csv(OUT / "ten_factor_operators.csv", index=False)
    sessions.to_csv(OUT / "ten_factor_sessions.csv", index=False)
    summary = {
        "schema": "neurobranch.kato_final_audit.ten_factor.v1",
        "evaluation_pca_is_common_factor_pca": True,
        "pca_fit": "training frames only; random_state=17; n_components=10",
        "b5_uses": "target own current value + all ten training-PCA factor scores; factor VAR rollout",
        "b3_uses": "full current population state; full lag-one VAR",
        "pca_variance_10_by_session": {
            str(r["session"]): r["pca_variance_10"] for r in session_rows
        },
        "b3_cross_over_within_median_by_session": {
            str(r["session"]): r["b3_cross_over_within_median"] for r in session_rows
        },
        "b3_cross_over_within": {
            "median": float(frame["b3_cross_over_within"].median()),
            "min": float(frame["b3_cross_over_within"].min()),
            "max": float(frame["b3_cross_over_within"].max()),
        },
        "b3_residual_unexplained_ratio": {
            "median": float(frame["b3_residual_unexplained_ratio_median"].median()),
            "min": float(frame["b3_residual_unexplained_ratio_median"].min()),
            "max": float(frame["b3_residual_unexplained_ratio_median"].max()),
        },
        "interpretation": (
            "B5 and the evaluation metric use the same training-only ten-PC transform. "
            "B5 therefore retains all ten evaluation dimensions, but its residual-source "
            "operator has the restricted form diag(own_scale) @ Q, i.e. one own-row direction "
            "per output neuron. B3's residual-source rows are almost entirely outside that "
            "one-dimensional pattern. This proves B3 has access to cross-neuron residual "
            "information that B5 cannot represent; it does not prove that this access causes "
            "the observed score advantage."
        ),
        "unresolved": (
            "Separating residual-coordinate information from B3's much larger parameter count "
            "would require a prespecified restricted model or new data; no such model is fitted "
            "or selected on the reused test tails here."
        ),
    }
    (OUT / "ten_factor_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
