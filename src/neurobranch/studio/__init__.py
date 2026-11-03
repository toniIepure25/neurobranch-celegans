"""Scientific Studio data and presentation layer.

This package separates the authoritative, machine-readable evidence from the
Streamlit rendering.  It does not fit models or recompute test scores.
"""

from __future__ import annotations

from neurobranch.studio.metadata import (
    ArtifactUnavailable,
    StudioSnapshot,
    load_snapshot,
)
from neurobranch.studio.trajectories import TrajectoryStore, load_trajectory_store

__all__ = [
    "ArtifactUnavailable",
    "StudioSnapshot",
    "TrajectoryStore",
    "load_snapshot",
    "load_trajectory_store",
]
