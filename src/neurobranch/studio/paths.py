"""Filesystem locations for the Scientific Studio.

All paths are relative to the repository root and use ``pathlib``.  No absolute
Windows path is embedded in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from neurobranch.config import PROJECT_ROOT


@dataclass(frozen=True)
class StudioPaths:
    root: Path = PROJECT_ROOT
    master: Path = PROJECT_ROOT / "reports" / "master_review"
    common: Path = PROJECT_ROOT / "reports" / "simulator" / "kato_common_state"
    final_audit: Path = PROJECT_ROOT / "reports" / "simulator" / "kato_final_audit"
    studio: Path = PROJECT_ROOT / "reports" / "scientific_studio"
    docs: Path = PROJECT_ROOT / "docs"
    data: Path = PROJECT_ROOT / "data" / "raw" / "kato2015" / "WT_NoStim.mat"

    @classmethod
    def at(cls, root: Path) -> StudioPaths:
        """Build a complete path set rooted at an arbitrary checkout."""
        root = Path(root).resolve()
        return cls(
            root=root,
            master=root / "reports" / "master_review",
            common=root / "reports" / "simulator" / "kato_common_state",
            final_audit=root / "reports" / "simulator" / "kato_final_audit",
            studio=root / "reports" / "scientific_studio",
            docs=root / "docs",
            data=root / "data" / "raw" / "kato2015" / "WT_NoStim.mat",
        )

    @property
    def trajectory_json(self) -> Path:
        return self.studio / "kato_trajectories.json"

    @property
    def trajectory_npz(self) -> Path:
        return self.studio / "kato_trajectories.npz"

    @property
    def course(self) -> Path:
        return self.docs / "MASTER_REVIEW_EDUCATIONAL_COURSE.md"

    @property
    def claims(self) -> Path:
        return self.master / "claims_ledger.json"

    @property
    def corrections(self) -> Path:
        return self.master / "corrections.json"

    @property
    def quality_gate(self) -> Path:
        return self.master / "quality_gate.json"

    @property
    def statistics(self) -> Path:
        return self.master / "statistical_audit.json"

    @property
    def per_landmark(self) -> Path:
        return self.common / "model_per_landmark.csv"
