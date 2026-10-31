"""Freeze and audit Phase 6, including the 95-to-99 support-threshold change.

The script is read-only with respect to the official Phase 6 artifacts.  It
writes separate external-validation audit files.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from neurobranch.config import PROJECT_ROOT, sha256_file

PHASE6 = PROJECT_ROOT / "reports/simulator/phase6"
CHECKPOINTS = PROJECT_ROOT / "artifacts/runs/blocked_suite/phase6"
REPORTS = PROJECT_ROOT / "reports/external_validation"


def _lf_sha256(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _verify_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = lf = other = missing = 0
    mismatches: list[dict[str, Any]] = []
    for rel, expected in manifest.get("artifacts", {}).items():
        path = Path(rel)
        if not path.exists():
            missing += 1
            mismatches.append({"path": rel, "status": "missing"})
            continue
        actual_raw = sha256_file(path)
        actual_lf = _lf_sha256(path)
        if actual_raw == expected["sha256"]:
            raw += 1
        elif actual_lf == expected["sha256"]:
            lf += 1
        else:
            other += 1
            mismatches.append(
                {"path": rel, "status": "hash_mismatch", "expected": expected["sha256"], "actual": actual_raw}
            )
    return {
        "manifest": str(manifest_path),
        "manifest_sha256_lf_normalized": _lf_sha256(manifest_path),
        "n_artifacts": len(manifest.get("artifacts", {})),
        "raw_byte_matches": raw,
        "lf_normalized_matches": lf,
        "other_mismatches": other,
        "missing": missing,
        "all_verify_under_lf_normalization": missing == 0 and other == 0,
        "mismatches": mismatches,
        "note": (
            "Windows working-tree CRLF causes raw-byte mismatches; the manifest hashes are "
            "LF-normalized. Both values are reported rather than silently rewriting the manifest."
        ),
    }


def _checkpoint_spectral_audit() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for family in ("var", "stochastic_var"):
        for path in sorted((CHECKPOINTS / family).glob("worm*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            A = np.asarray(payload["A"], dtype=float)
            n = int(payload["n_neurons"])
            history = int(payload.get("history_len", 2))
            companion = np.zeros((n * history, n * history))
            companion[:n, :] = A
            if history > 1:
                companion[n:, :-n] = np.eye(n * (history - 1))
            rho = float(np.max(np.abs(np.linalg.eigvals(companion))))
            cap = float(payload.get("spectral_cap", float("nan")))
            rows.append(
                {
                    "family": family,
                    "worm_id": int(path.stem.replace("worm", "")),
                    "n_neurons": n,
                    "history": history,
                    "nominal_cap": cap,
                    "realized_spectral_radius": rho,
                    "exceeds_nominal_cap": bool(np.isfinite(cap) and rho > cap + 1e-6),
                }
            )
    return {
        "rows": rows,
        "n_checkpoints": len(rows),
        "n_exceeding": int(sum(r["exceeds_nominal_cap"] for r in rows)),
        "note": (
            "One-shot scaling of the VAR coefficient matrix does not scale companion-matrix "
            "eigenvalues linearly because the companion retains an identity block. The nominal "
            "0.995 cap is therefore not exactly enforced. This is reported as an implementation "
            "limitation; no post-hoc model change was applied to the official Phase 6 result."
        ),
    }


def _support_audit(sensitivity: pd.DataFrame) -> dict[str, Any]:
    required = {
        "family",
        "worm_id",
        "quantile",
        "threshold",
        "rollout_in_support",
        "observed_test_in_support",
    }
    missing = required - set(sensitivity.columns)
    if missing:
        raise ValueError(f"support sensitivity is missing columns: {sorted(missing)}")
    thresholds = sensitivity.pivot_table(index=["family", "worm_id"], columns="quantile", values="threshold")
    ratios = (thresholds[0.95] / thresholds[0.99]).rename("q95_over_q99")
    observed = sensitivity.pivot_table(
        index=["family", "worm_id"], columns="quantile", values="observed_test_in_support"
    )
    rollout = sensitivity.pivot_table(
        index=["family", "worm_id"], columns="quantile", values="rollout_in_support"
    )
    per_worm = []
    for key in thresholds.index:
        family, worm = key
        per_worm.append(
            {
                "family": family,
                "worm_id": int(worm),
                "threshold_q90": float(thresholds.loc[key, 0.90]),
                "threshold_q95": float(thresholds.loc[key, 0.95]),
                "threshold_q99": float(thresholds.loc[key, 0.99]),
                "q95_over_q99": float(ratios.loc[key]),
                "observed_test_support_q90": float(observed.loc[key, 0.90]),
                "observed_test_support_q95": float(observed.loc[key, 0.95]),
                "observed_test_support_q99": float(observed.loc[key, 0.99]),
                "rollout_support_q90": float(rollout.loc[key, 0.90]),
                "rollout_support_q95": float(rollout.loc[key, 0.95]),
                "rollout_support_q99": float(rollout.loc[key, 0.99]),
            }
        )
    switching_q95 = [r for r in per_worm if r["family"] == "switching_var"]
    return {
        "per_family_worm": per_worm,
        "q95_over_q99_min": float(ratios.min()),
        "q95_over_q99_max": float(ratios.max()),
        "q95_rollout_min": float(rollout[0.95].min()),
        "q99_rollout_min": float(rollout[0.99].min()),
        "switching_var_q95_rollout": [r["rollout_support_q95"] for r in switching_q95],
        "n_family_worm_below_90pct_at_q95": int((rollout[0.95] < 0.90).sum()),
        "n_family_worm_below_90pct_at_q99": int((rollout[0.99] < 0.90).sum()),
        "observed_test_below_90pct_at_q95": int((observed[0.95] < 0.90).sum()),
        "observed_test_below_90pct_at_q99": int((observed[0.99] < 0.90).sum()),
    }


def main() -> int:
    manifest_path = PHASE6 / "simulator_manifest.json"
    gates = json.loads((PHASE6 / "gates.json").read_text(encoding="utf-8"))
    sensitivity = pd.read_csv(PHASE6 / "support_sensitivity.csv")
    support = _support_audit(sensitivity)
    freeze = _verify_manifest(manifest_path)
    spectral = _checkpoint_spectral_audit()
    g5 = next(gate for gate in gates["gates"] if gate["gate"] == "G5")

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_phase6": {
            "status": g5["status"],
            "support_quantile": 0.99,
            "support_rule": ">=90% perturbed states inside validation-calibrated support",
            "evidence_tier": gates.get("evidence_tier"),
            "test_block_reused": gates.get("test_block_reused"),
            "note": "The official 99th-percentile result is preserved unchanged.",
        },
        "prespecified": {
            "support_quantile": 0.95,
            "source": "docs/PHASE6_SIMULATOR_DEVIATIONS.md and the original protocol wording",
            "deviation_documented_before_test": True,
            "reason_recorded": "training-only leave-one-out envelope flagged 79% of selected-model states as OOD",
        },
        "support_sensitivity": support,
        "g5_threshold_sensitivity": {
            "recomputable_from_published_events": False,
            "reason": (
                "perturbation_events.csv records in_support at q=0.99 but not support distances or "
                "thresholds, so condition-level G5 cannot be exactly reconstructed at q=0.95."
            ),
            "evidence_that_q95_may_matter": (
                "Switching-VAR rollout in-support falls to 0.755 for worm 3 and 0.93 for worm 4 at "
                "q=0.95, below or near the 0.90 condition-level rule; this is not a condition-level "
                "G5 recomputation."
            ),
            "status": "not_evaluable_at_condition_level",
        },
        "spectral_cap_audit": spectral,
        "freeze_manifest": freeze,
        "negative_findings_preserved": {
            "G2": "fail",
            "G3": "fail",
            "G4": "fail",
            "G6": "not_evaluable",
            "G7": "not_evaluable",
            "test_block_reused": True,
        },
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "phase6_threshold_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (REPORTS / "phase6_freeze.json").write_text(
        json.dumps(
            {
                "generated_at_utc": payload["generated_at_utc"],
                "manifest": freeze,
                "spectral_cap_audit": spectral,
                "official_gates": gates["gates"],
                "evidence_tier": gates.get("evidence_tier"),
                "test_block_reused": gates.get("test_block_reused"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (REPORTS / "phase6_freeze_manifest.sha256").write_text(
        f"{_lf_sha256(REPORTS / 'phase6_freeze.json')}  phase6_freeze.json\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "raw_manifest_matches": freeze["raw_byte_matches"],
                "lf_manifest_matches": freeze["lf_normalized_matches"],
                "other": freeze["other_mismatches"],
                "q95_rollout_min": support["q95_rollout_min"],
                "q99_rollout_min": support["q99_rollout_min"],
                "spectral_exceeding": spectral["n_exceeding"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
