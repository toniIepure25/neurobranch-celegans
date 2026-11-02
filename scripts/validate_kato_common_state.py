"""Validate Kato common-state artifacts and their byte manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "simulator" / "kato_common_state"
REQUIRED = (
    "model_per_landmark.csv",
    "model_summary.csv",
    "model_config.csv",
    "paired_contrasts.csv",
    "residual_cross_neuron.csv",
    "decomposition.csv",
    "factor_variance.csv",
    "session_effects.csv",
    "representation_sensitivity.csv",
    "horizon_profile.csv",
    "stage_a_comparison.csv",
    "reproduction_summary.json",
    "gates.json",
    "analysis_summary.json",
    "manifest.json",
    "figures/model_ladder.png",
    "figures/decomposition.png",
    "figures/session_effects.png",
    "figures/surrogate_controls.png",
    "figures/synthetic_controls.png",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing = [name for name in REQUIRED if not (OUT / name).exists()]
    assert not missing, f"missing artifacts: {missing}"
    frame = pd.read_csv(OUT / "model_per_landmark.csv")
    for model in ("B3_population", "B5_common_k10", "B1_matched_own", "B4_matched_own_500"):
        count = int((frame["model"] == model).sum())
        assert count == 175, f"{model}: {count} rows"
    assert frame["pca10_energy_h20"].notna().all()
    dims = frame.groupby("session")["n_neurons"].first().to_dict()
    assert dims == {0: 103, 1: 127, 2: 123, 3: 118, 4: 121}, dims
    gates = json.loads((OUT / "gates.json").read_text(encoding="utf-8"))
    assert set(gates) == {
        "G1_historical_reproduction",
        "G2_fair_matched_baselines",
        "G3_common_factor_explanation",
        "G4_residual_cross_neuron_information",
        "G5_synthetic_specificity",
        "G6_robustness",
        "G7_structural_identifiability",
        "G8_biological_intervention_validity",
    }
    assert gates["G7_structural_identifiability"]["status"] == "FAIL"
    assert gates["G8_biological_intervention_validity"]["status"] == "NOT_EVALUABLE"
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    for name, record in manifest["files"].items():
        path = OUT / name
        assert path.exists(), f"manifest file missing: {name}"
        assert sha256(path) == record["sha256"], f"manifest hash mismatch: {name}"
    text = (ROOT / "docs" / "KATO_COMMON_STATE_PROTOCOL.md").read_text(encoding="utf-8")
    assert "103" in text and "127" in text and "session 0 only" in text
    text = (ROOT / "docs" / "KATO_COMMON_STATE_LIMITATIONS.md").read_text(encoding="utf-8")
    assert "MODEL-DEPENDENT VIRTUAL PERTURBATION" in text
    print(
        f"PASS: {len(manifest['files'])} hashed artifacts; "
        f"G4={gates['G4_residual_cross_neuron_information']['status']}; "
        f"G5={gates['G5_synthetic_specificity']['status']}"
    )


if __name__ == "__main__":
    main()
