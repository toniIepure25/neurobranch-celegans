"""Build machine-readable evidence for the final Scientific Studio audit.

This script reads committed artifacts and the raw archive. It does not write to
frozen evidence directories and does not refit the 175-landmark study.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from neurobranch.kato_common_state.models import (  # noqa: E402
    fit_window_common,
    rollout_common_factor,
)
from neurobranch.kato_interactions.models import fit_window  # noqa: E402
from neurobranch.studio import load_snapshot, load_trajectory_store  # noqa: E402
from neurobranch.studio.activity import _all_recordings  # noqa: E402
from neurobranch.studio.perturbations import perturb  # noqa: E402

OUT = ROOT / "reports" / "scientific_studio" / "final_audit"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _roll_b3(model, x0: np.ndarray, eps: np.ndarray) -> np.ndarray:
    x = np.repeat(np.asarray(x0, dtype=np.float64)[None, :], eps.shape[0], axis=0)
    out = np.empty((eps.shape[0], eps.shape[1], x.shape[1]), dtype=np.float64)
    for step in range(eps.shape[1]):
        x = x @ model.A.T + model.b + eps[:, step]
        out[:, step] = x
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot = load_snapshot()
    store = load_trajectory_store()
    primary = snapshot.primary
    frozen = snapshot.per_landmark

    # Recompute the paired aggregate from the frozen rows.
    pivot = frozen.pivot(index=["session", "landmark"], columns="model", values="pca10_energy_h20")
    diff = (pivot["B5_common_k10"] - pivot["B3_population"]).reset_index()
    recomputed = {
        "n": int(len(diff)),
        "b3_mean": float(pivot["B3_population"].mean()),
        "b5_mean": float(pivot["B5_common_k10"].mean()),
        "difference": float(diff.iloc[:, 2].mean()),
    }
    recomputed["relative_to_b3"] = recomputed["difference"] / recomputed["b3_mean"]

    alignment = []
    for sid in range(5):
        observed = store.observed(sid)
        for t in (store.session(sid)["landmarks"][0], store.session(sid)["showcase_landmarks"][0]):
            case = store.case(sid, t)
            truth_ok = np.allclose(case["truth"], observed[t + 1 : t + 21], atol=1e-6)
            score_rows = frozen[(frozen["session"] == sid) & (frozen["landmark"] == t)]
            score_ok = all(
                abs(case["scores"][model] - float(score_rows[score_rows["model"] == model]["pca10_energy_h20"].iloc[0]))
                < 1e-12
                for model in case["scores"]
            )
            alignment.append(
                {
                    "session": sid,
                    "landmark": int(t),
                    "truth_matches_observed": bool(truth_ok),
                    "scores_match_frozen_rows": bool(score_ok),
                    "origin_time": case["origin_time"],
                    "first_future_time": float(case["future_time"][0]),
                    "means_shape": {k: list(v.shape) for k, v in case["means"].items()},
                }
            )

    # Recompute one displayed deterministic mean from the frozen model code.
    data = _all_recordings()[0]
    train = data.X[data.masks["train"]]
    pca = PCA(n_components=10, random_state=17).fit(train)
    factors = pca.transform(data.X)
    t_check = 2690
    b3_check = fit_window(data.X, t_check, window=500, ridge=1.0, spectral_cap=0.995)
    b5_check = fit_window_common(data.X, factors, t_check, k=10, window=500, ridge=1.0)
    zeros = np.zeros((1, 20, data.X.shape[1]), dtype=np.float64)
    b3_roll = _roll_b3(b3_check, data.X[t_check], zeros)[0]
    b5_roll = rollout_common_factor(b5_check, data.X[t_check], factors[t_check, :10], zeros)[0]
    displayed = store.case(0, t_check)
    mean_check = {
        "session": 0,
        "landmark": t_check,
        "b3_max_abs_difference": float(
            np.max(np.abs(pca.transform(b3_roll) - displayed["means"]["B3_population"]))
        ),
        "b5_max_abs_difference": float(
            np.max(np.abs(pca.transform(b5_roll) - displayed["means"]["B5_common_k10"]))
        ),
        "b3_fit_last_transition": int(b3_check.fit_record["last_transition"]),
        "origin": t_check,
        "no_future_transition_used": int(b3_check.fit_record["last_transition"]) < t_check,
        "zero_innovation": True,
    }
    mean_check["passes"] = bool(
        mean_check["no_future_transition_used"]
        and mean_check["b3_max_abs_difference"] < 2e-5
        and mean_check["b5_max_abs_difference"] < 2e-5
    )


    names = [str(name) for name in data.predictor_names]
    neuron = "AVAL" if "AVAL" in names else next(
        (name for name in names if any(ch.isalpha() for ch in name)), names[0]
    )
    perturb_result = perturb(0, 2690, neuron, 1.0, 20)
    perturbation = {
        "train_mean_abs_max": float(np.max(np.abs(train.mean(axis=0)))),
        "train_std_min": float(train.std(axis=0).min()),
        "train_std_max": float(train.std(axis=0).max()),
        "tested": perturb_result is not None,
    }
    if perturb_result is not None:
        b3 = fit_window(data.X, 2690, window=500, ridge=1.0, spectral_cap=0.995)
        j = names.index(neuron)
        expected_neuron = float(b3.A[j, j])
        expected_population = float(np.linalg.norm(b3.A[:, j]))
        perturbation.update(
            {
                "neuron": neuron,
                "delta_z": perturb_result.delta_z,
                "first_step_neuron_response": float(perturb_result.neuron_response[0]),
                "first_step_population_response": float(perturb_result.population_response[0]),
                "expected_first_step_neuron_response": expected_neuron,
                "expected_first_step_population_response": expected_population,
                "first_step_matches_A_column": bool(
                    abs(perturb_result.neuron_response[0] - expected_neuron) < 1e-12
                    and abs(perturb_result.population_response[0] - expected_population) < 1e-12
                ),
                "h20_model_disagreement": float(perturb_result.disagreement_h20),
                "ood": bool(perturb_result.ood),
                "b3_radius": float(perturb_result.b3_radius),
                "b5_factor_radius": float(perturb_result.b5_radius),
            }
        )

    payload = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dataset": snapshot.dataset_identity,
        "primary": primary,
        "recomputed_primary": recomputed,
        "primary_matches_within_1e-12": bool(
            abs(primary["difference"] - recomputed["difference"]) < 1e-12
            and primary["n"] == recomputed["n"]
        ),
        "displayed_metrics": {
            "primary_difference": primary["difference"],
            "relative_to_b3": primary["relative_to_b3"],
            "b3_mean": primary["b3_mean"],
            "b5_mean": primary["b5_mean"],
            "conditional_ci": primary["conditional_ci"],
            "recording_ci": primary["recording_ci"],
            "session_t_ci": primary["session_t_ci"],
            "sessions_positive": primary["sessions_positive"],
            "n_sessions": primary["n_sessions"],
            "mse_difference": primary["mse_difference"],
            "mse_ci": primary["mse_ci"],
            "coverage_b3": snapshot.statistical["paired_effect"]["coverage_at_nominal_95"]["B3_population"]["95"],
            "coverage_b5": snapshot.statistical["paired_effect"]["coverage_at_nominal_95"]["B5_common_k10"]["95"],
            "capacity_ratio_min": snapshot.capacity["ratio_min"],
            "capacity_ratio_max": snapshot.capacity["ratio_max"],
            "b5_training_variance_min": snapshot.capacity["b5_training_variance_fraction_min"],
            "b5_training_variance_max": snapshot.capacity["b5_training_variance_fraction_max"],
            "ordered_share_b4m_common": snapshot.capacity["ordered_shares"]["baseline_B4m"]["common"],
            "ordered_share_b4m_residual": snapshot.capacity["ordered_shares"]["baseline_B4m"]["residual"],
            "verdict": snapshot.overall_status,
        },
        "trajectory_alignment": alignment,
        "all_alignment_checks_pass": all(
            a["truth_matches_observed"] and a["scores_match_frozen_rows"] for a in alignment
        ),
        "deterministic_mean_check": mean_check,
        "perturbation": perturbation,
        "artifacts": {
            "trajectory_npz": {
                "bytes": (ROOT / "reports/scientific_studio/kato_trajectories.npz").stat().st_size,
                "sha256": sha256(ROOT / "reports/scientific_studio/kato_trajectories.npz"),
            },
            "trajectory_json": {
                "bytes": (ROOT / "reports/scientific_studio/kato_trajectories.json").stat().st_size,
                "sha256": sha256(ROOT / "reports/scientific_studio/kato_trajectories.json"),
            },
            "manifest_sha256": sha256(ROOT / "reports/scientific_studio/manifest.json"),
        },
        "frozen_rows": {
            "rows": int(len(frozen)),
            "models": int(frozen["model"].nunique()),
            "landmarks": int(frozen[["session", "landmark"]].drop_duplicates().shape[0]),
            "all_energy_finite": bool(np.isfinite(frozen["pca10_energy_h20"]).all()),
        },
    }
    (OUT / "studio_metrics_and_alignment.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: payload[k] for k in ("git_head", "primary_matches_within_1e-12", "all_alignment_checks_pass")}, indent=2))


if __name__ == "__main__":
    main()
