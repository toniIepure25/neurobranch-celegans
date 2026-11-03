"""Authoritative metadata for the Scientific Studio.

The studio reads committed machine-readable artifacts.  It does not maintain a
second set of hand-copied numbers in the UI.  If an artifact is missing or
inconsistent, the loader raises :class:`ArtifactUnavailable` and the page shows
an explicit unavailable state.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from neurobranch.studio.paths import StudioPaths


class ArtifactUnavailable(RuntimeError):
    """Raised when a required frozen artifact is missing or inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ArtifactUnavailable(
            f"Required artifact is missing: {path.relative_to(StudioPaths().root)}. "
            "Run the documented scientific-studio build command."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactUnavailable(f"Artifact is not valid JSON: {path}") from exc


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ArtifactUnavailable(f"Required artifact is missing: {path}")
    return pd.read_csv(path)


@dataclass(frozen=True)
class StudioSnapshot:
    """One read-only view of the current audited evidence."""

    paths: StudioPaths
    trajectories: dict[str, Any]
    dataset_identity: dict[str, Any]
    statistical: dict[str, Any]
    uncertainty: dict[str, Any]
    quality_gate: dict[str, Any]
    claims_payload: dict[str, Any]
    corrections_payload: dict[str, Any]
    capacity: dict[str, Any]
    model_config: tuple[dict[str, Any], ...]
    per_landmark: pd.DataFrame
    session_effects: pd.DataFrame
    horizon_profile: pd.DataFrame
    cluster_bootstrap: pd.DataFrame
    history: pd.DataFrame
    manifest: dict[str, Any]

    @property
    def claims(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.claims_payload["claims"])

    @property
    def corrections(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.corrections_payload["items"])

    @property
    def sessions(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.trajectories["sessions"])

    @property
    def primary(self) -> dict[str, Any]:
        """The prespecified B3 vs B5-10 H20 contrast, computed from paired rows."""
        pivot = self.per_landmark.pivot_table(
            index=["session", "landmark"],
            columns="model",
            values="pca10_energy_h20",
            aggfunc="first",
        )
        if "B3_population" not in pivot or "B5_common_k10" not in pivot:
            raise ArtifactUnavailable("Primary model columns are absent from model_per_landmark.csv")
        paired = (pivot["B5_common_k10"] - pivot["B3_population"]).dropna()
        if len(paired) != 175:
            raise ArtifactUnavailable(
                f"Expected 175 paired H20 landmarks, found {len(paired)}"
            )
        b3 = float(pivot["B3_population"].mean())
        b5 = float(pivot["B5_common_k10"].mean())
        difference = float(paired.mean())
        audit = self.statistical
        return {
            "n": int(len(paired)),
            "b3_mean": b3,
            "b5_mean": b5,
            "difference": difference,
            "relative_to_b3": difference / b3,
            "equal_session_mean": float(audit["equal_session_mean"]),
            "conditional_ci": next(
                block for block in self.uncertainty["blocks"] if int(block["block"]) == 5
            ),
            "session_t_ci": list(audit["session_t_interval_95"]),
            "recording_ci": audit["cluster_bootstrap"]["5"],
            "sessions_positive": int(audit["sessions_positive"]),
            "n_sessions": int(audit["n_sessions"]),
            "fraction_positive": float(audit["paired_effect"]["energy"]["fraction_positive"]),
            "dz": float(
                audit["paired_effect"]["energy"]["standardized_mean_dz_descriptive"]
            ),
            "mse_difference": float(audit["paired_effect"]["mse"]["mean"]),
            "mse_ci": list(audit["calibration_confound"]["mse_interval_block5"]),
            "sign_test_one_sided_p": float(audit["sign_test_one_sided_p"]),
        }

    @property
    def session_rows(self) -> tuple[dict[str, Any], ...]:
        rows = []
        for row in self.session_effects.to_dict(orient="records"):
            session = int(row["session"])
            session_meta = self.sessions[session]
            rows.append(
                {
                    **row,
                    "name": session_meta["dataset_name"],
                    "acquisition": str(session_meta["dataset_name"]).split("_")[0],
                    "n_frames": int(session_meta["n_frames"]),
                    "n_predictors": int(session_meta["n_predictors"]),
                    "n_landmarks": int(row["n_landmarks"]),
                }
            )
        return tuple(rows)

    @property
    def claim_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(claim["status"] for claim in self.claims).items()))

    @property
    def gates(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.quality_gate["gates"])

    @property
    def overall_status(self) -> str:
        return str(self.quality_gate["overall"])

    @property
    def dataset_ok(self) -> bool:
        archive = self.dataset_identity["archive"]
        return bool(
            archive["checksum_matches_expected"]
            and archive["checksum_matches_osf"]
            and archive["size_matches_osf"]
        )

    def model_row(self, model: str) -> dict[str, Any]:
        for row in self.model_config:
            if row["model"] == model:
                return row
        raise ArtifactUnavailable(f"Model {model} is absent from model_config.csv")

    def session_by_id(self, session: int) -> dict[str, Any]:
        for row in self.sessions:
            if int(row["session"]) == int(session):
                return row
        raise ArtifactUnavailable(f"Session {session} is absent from the trajectory index")

    def score(self, session: int, landmark: int, model: str) -> float | None:
        row = self.per_landmark[
            (self.per_landmark["session"] == int(session))
            & (self.per_landmark["landmark"] == int(landmark))
            & (self.per_landmark["model"] == model)
        ]
        if row.empty:
            return None
        value = row["pca10_energy_h20"].iloc[0]
        return None if pd.isna(value) else float(value)


def _validate(snapshot: StudioSnapshot) -> None:
    if not snapshot.dataset_ok:
        raise ArtifactUnavailable(
            "The raw Kato archive does not match the audited OSF hash/size. "
            "The studio will not render scientific claims from an unverified archive."
        )
    if len(snapshot.sessions) != 5:
        raise ArtifactUnavailable("Expected five Kato recordings in the trajectory index")
    landmarks = [int(meta["n_frames"]) for meta in snapshot.sessions]
    if landmarks != [3137, 3134, 3059, 3311, 3021]:
        raise ArtifactUnavailable(
            f"Recording lengths do not match the audited dataset: {landmarks}"
        )
    counts = [len(meta["landmarks"]) for meta in snapshot.sessions]
    if counts != [35, 35, 34, 38, 33]:
        raise ArtifactUnavailable(f"Unexpected H20 landmark counts: {counts}")
    primary = snapshot.primary
    expected = float(snapshot.statistical["weighted_landmark_mean"])
    if abs(primary["difference"] - expected) > 1e-9:
        raise ArtifactUnavailable(
            "Primary contrast recomputation disagrees with statistical_audit.json"
        )


@lru_cache(maxsize=2)
def load_snapshot(root: Path | None = None) -> StudioSnapshot:
    """Load and validate the committed evidence package.

    The returned object is cached because the artifacts are immutable during a
    session.  ``root`` is accepted for tests and for read-only inspection of a
    copied workspace.
    """
    paths = StudioPaths() if root is None else StudioPaths.at(Path(root))
    trajectories = _read_json(paths.trajectory_json)
    dataset_identity = _read_json(paths.master / "dataset_identity_audit.json")
    statistical = _read_json(paths.statistics)
    uncertainty = _read_json(paths.final_audit / "uncertainty_summary.json")
    quality_gate = _read_json(paths.quality_gate)
    claims = _read_json(paths.claims)
    corrections = _read_json(paths.corrections)
    capacity = _read_json(paths.master / "capacity_audit.json")
    model_config = tuple(_read_csv(paths.common / "model_config.csv").to_dict(orient="records"))
    per_landmark = _read_csv(paths.per_landmark)
    session_effects = _read_csv(paths.common / "session_effects.csv")
    horizon_profile = _read_csv(paths.common / "horizon_profile.csv")
    cluster_bootstrap = _read_csv(paths.master / "cluster_bootstrap.csv")
    history = _read_csv(paths.master / "history_timeline.csv")
    manifest = _read_json(paths.master / "manifest.json")

    snapshot = StudioSnapshot(
        paths=paths,
        trajectories=trajectories,
        dataset_identity=dataset_identity,
        statistical=statistical,
        uncertainty=uncertainty,
        quality_gate=quality_gate,
        claims_payload=claims,
        corrections_payload=corrections,
        capacity=capacity,
        model_config=model_config,
        per_landmark=per_landmark,
        session_effects=session_effects,
        horizon_profile=horizon_profile,
        cluster_bootstrap=cluster_bootstrap,
        history=history,
        manifest=manifest,
    )
    _validate(snapshot)
    return snapshot
