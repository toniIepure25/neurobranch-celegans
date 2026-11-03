"""Build the compact, reproducible trajectory artifact used by the Scientific Studio.

The artifact contains only projections of existing model outputs onto the fixed,
training-fitted 10-D PCA.  It does not fit a new model family, select a model, or
score held-out data.  Frozen H20 scores are copied from the committed
``model_per_landmark.csv`` and a representative subset is independently checked
against the recomputed 100-member ensemble.

Run from the repository root::

    python scripts/studio_build_trajectories.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import jd_evaluate as JD  # noqa: E402

from neurobranch.diagnostics.common import load_nostim  # noqa: E402
from neurobranch.kato_common_state.models import (  # noqa: E402
    fit_window_common,
    fit_window_own,
    rollout_common_factor,
    rollout_matched,
)
from neurobranch.kato_interactions.models import FrozenLinearModel, fit_window  # noqa: E402
from neurobranch.probabilistic.metrics import energy_score  # noqa: E402

OUT = ROOT / "reports" / "scientific_studio"
DATA = ROOT / "data" / "raw" / "kato2015" / "WT_NoStim.mat"
COMMON = ROOT / "reports" / "simulator" / "kato_common_state"
HORIZON = 100
EVAL_HORIZON = 20
PRIMARY_K = 10
MODEL_ORDER = ("B0_persistence", "B1_matched_own", "B3_population", "B5_common_k10")
MODEL_LABELS = {
    "B0_persistence": "Persistence",
    "B1_matched_own": "Independent AR (matched)",
    "B3_population": "Population VAR",
    "B5_common_k10": "Ten-factor control",
}
SHOWCASE_FRACTIONS = (0.18, 0.50, 0.82)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10
        ).strip()
    except Exception:
        return None


def project(pca, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    flat = arr.reshape(-1, arr.shape[-1])
    return pca.transform(flat).reshape(*arr.shape[:-1], -1)


def zero_eps(n_neurons: int, members: int = 1) -> np.ndarray:
    return np.zeros((members, HORIZON, n_neurons), dtype=np.float64)


def build_session(
    d, frame: pd.DataFrame
) -> tuple[dict, dict[str, np.ndarray], list[dict]]:
    sid = int(d.worm_id)
    pca, series, _ = JD.prepare(d)
    scores = project(pca, d.X).astype(np.float32)
    landmarks = np.asarray(
        sorted(frame.loc[frame["session"] == sid, "landmark"].astype(int).unique()),
        dtype=np.int32,
    )
    showcase = sorted(
        {
            int(landmarks[min(len(landmarks) - 1, max(0, round(f * (len(landmarks) - 1))))])
            for f in SHOWCASE_FRACTIONS
        }
    )

    n = int(d.X.shape[1])
    truth = np.empty((len(landmarks), EVAL_HORIZON, 10), dtype=np.float32)
    means = np.empty(
        (len(landmarks), len(MODEL_ORDER), EVAL_HORIZON, 10), dtype=np.float32
    )
    ensembles = np.empty(
        (len(landmarks), 2, 8, EVAL_HORIZON, 10), dtype=np.float32
    )
    ensemble_100: dict[str, np.ndarray] = {}
    checked: list[dict] = []
    temporal_cache: dict = {}
    factor_scores = pca.transform(d.X)

    for i, t in enumerate(landmarks):
        t = int(t)
        x0 = np.asarray(d.X[t], dtype=np.float64)
        truth[i] = scores[t + 1 : t + EVAL_HORIZON + 1]
        if not np.isfinite(truth[i]).all():
            raise ValueError(f"non-finite target at session {sid}, landmark {t}")

        spec = JD.Spec("J5_population_lead1_boot", "var", "population", "block", 10)
        innovation = JD.make_innovation(spec, series[spec.series], t, temporal_cache)
        eps = innovation.sample(100, HORIZON, np.random.default_rng(7000 + sid * 10000 + t))

        b3_fit = fit_window(d.X, t, window=500, ridge=1.0, spectral_cap=0.995)
        if b3_fit is None:
            raise RuntimeError(f"B3 fit failed at session {sid}, landmark {t}")
        b3 = FrozenLinearModel(A=b3_fit.A, b=b3_fit.b, fit_record=dict(b3_fit.fit_record))
        b5 = fit_window_common(
            d.X,
            factor_scores,
            t,
            k=PRIMARY_K,
            window=500,
            ridge=1.0,
        )
        if b5 is None:
            raise RuntimeError(f"B5 fit failed at session {sid}, landmark {t}")
        b1 = fit_window_own(d.X, t, window=200, ridge=1.0)
        if b1 is None:
            raise RuntimeError(f"B1 fit failed at session {sid}, landmark {t}")

        b3_mean = JD.rollout("var", b3, x0, HORIZON, zero_eps(n)).astype(np.float64)
        b5_mean = rollout_common_factor(
            b5, x0, factor_scores[t, :PRIMARY_K], zero_eps(n)
        ).astype(np.float64)
        means[i, 0] = project(pca, np.repeat(x0[None, :], HORIZON, axis=0))[:EVAL_HORIZON]
        means[i, 1] = project(pca, rollout_matched(b1, x0, zero_eps(n))[0])[:EVAL_HORIZON]
        means[i, 2] = project(pca, b3_mean[0])[:EVAL_HORIZON]
        means[i, 3] = project(pca, b5_mean[0])[:EVAL_HORIZON]

        b3_ens = JD.rollout("var", b3, x0, HORIZON, eps).astype(np.float64)
        b5_ens = rollout_common_factor(b5, x0, factor_scores[t, :PRIMARY_K], eps)
        ensembles[i, 0] = project(pca, b3_ens[:8])[:, :EVAL_HORIZON]
        ensembles[i, 1] = project(pca, b5_ens[:8])[:, :EVAL_HORIZON]

        if t in showcase:
            ensemble_100[f"ens100_s{sid}_{t}_B3"] = project(pca, b3_ens)[
                :, :EVAL_HORIZON
            ].astype(np.float32)
            ensemble_100[f"ens100_s{sid}_{t}_B5"] = project(pca, b5_ens)[
                :, :EVAL_HORIZON
            ].astype(np.float32)
            for key, ens, model in (
                ("B3", b3_ens, "B3_population"),
                ("B5", b5_ens, "B5_common_k10"),
            ):
                proj = project(pca, ens)[:, :EVAL_HORIZON]
                recomputed = float(energy_score(proj[:, 19], truth[i][19]))
                frozen = float(
                    frame.loc[
                        (frame["session"] == sid)
                        & (frame["landmark"] == t)
                        & (frame["model"] == model),
                        "pca10_energy_h20",
                    ].iloc[0]
                )
                checked.append(
                    {
                        "session": sid,
                        "landmark": t,
                        "model": key,
                        "recomputed": recomputed,
                        "frozen": frozen,
                        "abs_difference": abs(recomputed - frozen),
                    }
                )

    score_rows = []
    for t in landmarks:
        row = {"landmark": int(t)}
        for model in MODEL_ORDER:
            match = frame.loc[
                (frame["session"] == sid)
                & (frame["landmark"] == int(t))
                & (frame["model"] == model),
                "pca10_energy_h20",
            ]
            if len(match):
                row[model] = float(match.iloc[0])
        score_rows.append(row)

    meta = {
        "session": sid,
        "dataset_name": d.recording.dataset_name,
        "n_frames": int(d.n_frames),
        "n_predictors": int(d.X.shape[1]),
        "n_named": int(sum(any(c.isalpha() for c in name) for name in d.predictor_names)),
        "fps": float(d.recording.fps),
        "time_seconds_start": float(d.recording.time_seconds[0]),
        "time_seconds_end": float(d.recording.time_seconds[-1]),
        "split_bounds": {k: [int(v[0]), int(v[1])] for k, v in d.split_bounds.items()},
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "explained_variance_3d": float(np.sum(pca.explained_variance_ratio_[:3])),
        "landmarks": [int(v) for v in landmarks],
        "showcase_landmarks": showcase,
        "scores": score_rows,
    }
    arrays = {
        f"observed_s{sid}": scores,
        f"truth_s{sid}": truth,
        f"means_s{sid}": means,
        f"ens_s{sid}": ensembles,
        f"tv_s{sid}": np.asarray(d.recording.time_seconds, dtype=np.float64),
        **ensemble_100,
    }
    return meta, arrays, checked


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(COMMON / "model_per_landmark.csv")
    cfg, data, recordings = load_nostim()
    del cfg
    if DATA.stat().st_size != 56_390_975:
        raise RuntimeError(f"unexpected archive size {DATA.stat().st_size}")
    expected_hash = "7531811c1718e26a0455f59cb2f8188cf25809278cfea85770dfe03315a0098a"
    if sha256(DATA) != expected_hash:
        raise RuntimeError("archive SHA256 does not match the audited OSF object")

    session_meta: list[dict] = []
    all_arrays: dict[str, np.ndarray] = {}
    checks: list[dict] = []
    for d in data:
        meta, arrays, checked = build_session(d, frame)
        session_meta.append(meta)
        all_arrays.update(arrays)
        checks.extend(checked)

    max_check = max((row["abs_difference"] for row in checks), default=float("nan"))
    metadata = {
        "schema": "neurobranch.scientific_studio.trajectories.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_git_head": git_head(),
        "purpose": (
            "Deterministic-mean and bounded sampled model trajectories projected into the "
            "fixed training-fitted 10-D PCA for visual inspection only."
        ),
        "not_for": [
            "new model selection",
            "new test-set optimization",
            "animal-level inference",
            "anatomical localization",
            "causal interpretation",
        ],
        "archive": {
            "path": "data/raw/kato2015/WT_NoStim.mat",
            "size_bytes": DATA.stat().st_size,
            "sha256": expected_hash,
            "sessions": [rec.dataset_name for rec in recordings],
        },
        "pca": {
            "fit_split": "train",
            "n_components": 10,
            "random_state": 17,
            "implementation": "sklearn.decomposition.PCA",
        },
        "rollout_horizon": HORIZON,
        "evaluation_horizon": EVAL_HORIZON,
        "model_order": list(MODEL_ORDER),
        "model_labels": MODEL_LABELS,
        "ensemble_storage": {
            "all_landmarks_members": 8,
            "showcase_members": 100,
            "scoring_ensemble_members": 100,
            "note": (
                "The 8-member subsets are for rendering only. The frozen table's 100-member "
                "ensembles produced every score shown in the studio."
            ),
        },
        "landmark_scores_source": "reports/simulator/kato_common_state/model_per_landmark.csv",
        "score_validation": {
            "n_showcase_checks": len(checks),
            "max_abs_difference": max_check,
            "tolerance": 2e-5,
            "tolerance_reason": (
                "float32 trajectory storage and PCA sign/rounding; the value is a "
                "storage-path check, not a new inferential tolerance."
            ),
            "checks": checks,
        },
        "sessions": session_meta,
    }
    np.savez_compressed(OUT / "kato_trajectories.npz", **all_arrays)
    metadata["trajectory_file"] = {
        "name": "kato_trajectories.npz",
        "bytes": (OUT / "kato_trajectories.npz").stat().st_size,
        "sha256": sha256(OUT / "kato_trajectories.npz"),
    }
    (OUT / "kato_trajectories.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "sessions": len(session_meta),
                "landmarks": sum(len(s["landmarks"]) for s in session_meta),
                "checks": len(checks),
                "max_abs_difference": max_check,
                "bytes": metadata["trajectory_file"]["bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
