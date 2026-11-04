"""Read-only release checks over frozen Studio and Kato artifacts.

This does not fit a model. It verifies that the current release candidate still
exposes the corrected PCA denominators, the primary B5-minus-B3 contrast, the
uncertainty boundaries, and the numerical root-cause control.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "reports" / "scientific_studio" / "final_audit"
DEFAULT_OUT = ROOT / "reports" / "scientific_studio" / "release_candidate" / "scientific_checks.json"
EXPECTED_PRIMARY = 0.2644379926683563
EXPECTED_PCA0 = {
    "original_feature_variance_sum": 103.04696655273438,
    "pca10_explained_variance_sum": 73.14903259277344,
    "pc1_pc3_fraction_of_original": 0.5199463963508606,
    "pc1_pc3_fraction_of_pca10": 0.7324619889259338,
    "pc4_pc10_fraction_of_pca10": 0.2675381004810333,
    "outside_pca10_fraction_of_original": 0.29013872146606445,
}


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    numerical = json.loads((FINAL / "independent_model_check.json").read_text(encoding="utf-8"))
    pca = json.loads((FINAL / "pca_variance_audit.json").read_text(encoding="utf-8"))
    alignment = json.loads(
        (FINAL / "studio_metrics_and_alignment.json").read_text(encoding="utf-8")
    )
    statistical = json.loads(
        (ROOT / "reports" / "master_review" / "statistical_audit.json").read_text(
            encoding="utf-8"
        )
    )

    session0 = next(item for item in pca["sessions"] if item["session"] == 0)
    h1 = next(
        item
        for item in statistical["horizon"]
        if item["horizon"] == 1 and item["session"] == "all"
    )
    h20 = next(
        item
        for item in statistical["horizon"]
        if item["horizon"] == 20 and item["session"] == "all"
    )
    metrics = alignment["displayed_metrics"]
    conditional = metrics["conditional_ci"]
    recording = metrics["recording_ci"]

    checks = {
        "numerical_status_pass": numerical["status"] == "PASS",
        "root_cause_control_pass": numerical["root_cause_control"]["status"] == "PASS",
        "extended_cases_11": len(numerical["cases"]) == 11,
        "pca_session0_values_match": all(
            close(session0[key], value) for key, value in EXPECTED_PCA0.items()
        ),
        "primary_matches_frozen_within_1e_12": bool(
            alignment["primary_matches_within_1e-12"]
        ),
        "primary_value_matches": close(metrics["primary_difference"], EXPECTED_PRIMARY),
        "all_alignment_checks_pass": bool(alignment["all_alignment_checks_pass"]),
        "deterministic_mean_passes": bool(alignment["deterministic_mean_check"]["passes"]),
        "conditional_ci_excludes_zero": conditional["ci_low"] > 0,
        "recording_ci_includes_zero": recording["ci_low"] <= 0 <= recording["ci_high"],
        "sessions_positive_4_of_5": metrics["sessions_positive"] == 4
        and metrics["n_sessions"] == 5,
        "mse_ci_includes_zero": metrics["mse_ci"][0] <= 0 <= metrics["mse_ci"][1],
        "h1_reverses": h1["mean_advantage"] < 0 < h20["mean_advantage"],
        "capacity_not_matched": metrics["capacity_ratio_min"] > 1.5,
        "perturbation_warning_exact": alignment["perturbation"]["tested"]
        and alignment["perturbation"]["first_step_matches_A_column"],
    }
    failed = sorted(key for key, value in checks.items() if not value)
    payload = {
        "status": "PASS" if not failed else "FAIL",
        "failed": failed,
        "checks": checks,
        "session0": {
            "original_feature_variance": session0["original_feature_variance_sum"],
            "pca10_explained": session0["pca10_explained_variance_sum"],
            "pc1_pc3_of_total": session0["pc1_pc3_fraction_of_original"],
            "pc1_pc3_of_pca10": session0["pc1_pc3_fraction_of_pca10"],
            "pc4_pc10_of_pca10": session0["pc4_pc10_fraction_of_pca10"],
            "outside_pca10_of_total": session0["outside_pca10_fraction_of_original"],
        },
        "primary": {
            "difference_b5_minus_b3": metrics["primary_difference"],
            "relative_to_b3": metrics["relative_to_b3"],
            "conditional_ci": [conditional["ci_low"], conditional["ci_high"]],
            "recording_ci": [recording["ci_low"], recording["ci_high"]],
            "sessions_positive": metrics["sessions_positive"],
            "mse_difference": metrics["mse_difference"],
            "mse_ci": metrics["mse_ci"],
            "coverage_b3": metrics["coverage_b3"],
            "coverage_b5": metrics["coverage_b5"],
            "capacity_ratio": [
                metrics["capacity_ratio_min"],
                metrics["capacity_ratio_max"],
            ],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"status={payload['status']} failed={failed}")
    print(f"wrote {args.out.relative_to(ROOT)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
