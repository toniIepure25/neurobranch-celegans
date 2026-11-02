"""Finalize joint-dynamics metadata, model cards, stage ledger and manifest."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import scipy
import sklearn

from neurobranch.adaptive.manifest import sha256_file, write_manifest
from neurobranch.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "reports/simulator/joint_dynamics"
WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED."


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    cards = json.loads((OUT / "model_cards.json").read_text(encoding="utf-8"))
    cards["J3_population_full"]["role"] = "prespecified covariance stress test; rejected"
    cards["J4_independent_full"]["role"] = "full-covariance factorial control; rejected"
    (OUT / "model_cards.json").write_text(json.dumps(cards, indent=2), encoding="utf-8")

    configs = json.loads((OUT / "configurations.json").read_text(encoding="utf-8"))
    configs.update(
        {
            "dataset": "Kato et al. 2015 WT_NoStim",
            "dataset_file": "data/raw/kato2015/WT_NoStim.mat",
            "sessions": 5,
            "frames": [3137, 3134, 3059, 3311, 3021],
            "predictors": [103, 103, 102, 103, 103],
            "split": {"train": "0:2195", "validation": "2207:2677", "test": "2689:3137", "purge": 12},
            "primary_horizon": 20,
            "landmarks": "35 per session, stride 10, horizon max 100, overlapping",
            "innovation_rule": "only lead==1 causal prequential residuals",
            "d4": {"distribution": "multivariate Student-t", "df": 5, "scale": 0.85, "selected_on": "validation only"},
            "parameter_uncertainty": {"block": 25, "draws": 5, "members_per_draw": 20, "window": 500},
            "historical_invalid_model": "P6_historical_block, multi-step pool, diagnostic only",
        }
    )
    (OUT / "configurations.json").write_text(json.dumps(configs, indent=2), encoding="utf-8")

    stages = pd.DataFrame(
        [
            {"stage": "A reproduction", "status": "PASS", "detail": "max relative difference 5.5e-9"},
            {"stage": "B population information", "status": "PARTIAL", "detail": "mean helps; covariance hurts; marginal CRPS unresolved"},
            {"stage": "C innovation structure", "status": "PASS", "detail": "lead-1 pool near white; temporal model not justified"},
            {"stage": "D calibration", "status": "FAIL", "detail": "D1 best but 50% coverage fails; D2/D4 worse; D3 not fitted"},
            {"stage": "E probabilistic evaluation", "status": "FAIL", "detail": "energy improves; coverage and covariance fail"},
            {"stage": "F robustness", "status": "PARTIAL", "detail": "4/5 sessions; all-neuron CI includes zero; block-10 persistence CI crosses zero"},
            {"stage": "G generative trajectory", "status": "FAIL", "detail": "trajectory energy and autocorr pass; covariance fails"},
            {"stage": "H parameter uncertainty", "status": "REJECTED", "detail": "parameter ensemble worsens energy and coverage"},
        ]
    )
    stages.to_csv(OUT / "stage_status.csv", index=False)

    negatives = pd.DataFrame(
        [
            {"finding": "full covariance", "evidence": "H20 energy 3.011 vs 2.700; covariance error 0.762"},
            {"finding": "marginal CRPS improvement", "evidence": "+0.00484, 95% CI [-0.00830, 0.01566]"},
            {"finding": "all-neuron energy improvement", "evidence": "+0.203, 95% CI [-0.015, 0.412]"},
            {"finding": "50% calibration", "evidence": "full 0.588; PCA 0.546"},
            {"finding": "neural covariance gate", "evidence": "0.604 > 0.5"},
            {"finding": "D4 Student-t", "evidence": "energy 2.829, coverage 0.690/0.921/0.984"},
            {"finding": "parameter uncertainty", "evidence": "energy 2.807, trajectory 2.271, coverage 0.615/0.883/0.969"},
            {"finding": "session consistency", "evidence": "session 1 negative; session 2 ambiguous"},
        ]
    )
    negatives.to_csv(OUT / "negative_findings.csv", index=False)

    (OUT / "perturbation_status.json").write_text(
        json.dumps(
            {
                "status": "NOT_LAUNCHED",
                "warning": WARNING,
                "biological_validation": "NOT_EVALUABLE",
                "reason": "no compatible intervention data; calibration and covariance gates failed",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    figures = []
    for path in sorted((OUT / "figures").iterdir()):
        if path.is_file():
            figures.append({"figure": path.name, "sha256": sha256_file(path), "bytes": int(path.stat().st_size)})
    (OUT / "figure_manifest.json").write_text(json.dumps(figures, indent=2), encoding="utf-8")

    metadata = {
        "experiment": "joint neural dynamics and calibration",
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "head_at_finalize": _git("rev-parse", "HEAD"),
        "status": "exploratory repeated-session reanalysis",
        "not_a_systematic_review": True,
        "not_external_validation": True,
        "warning": WARNING,
        "biological_validation": "NOT_EVALUABLE",
        "created_utc": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "dataset": "Kato et al. 2015 WT_NoStim",
        "dataset_sha256": sha256_file(PROJECT_ROOT / "data/raw/kato2015/WT_NoStim.mat")
        if (PROJECT_ROOT / "data/raw/kato2015/WT_NoStim.mat").exists()
        else None,
        "n_landmarks": 175,
        "primary_horizon": 20,
        "members": 100,
        "stop_label": "PROBABILISTIC CALIBRATION NOT ESTABLISHED",
        "raw_recordings_committed": False,
        "model_weights_committed": False,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    payload = write_manifest(OUT)
    print(f"finalized {payload['n_files']} files; metadata HEAD {metadata['head_at_finalize']}")


if __name__ == "__main__":
    main()
