"""UI contract tests that do not require a browser."""

from __future__ import annotations

import pytest

from neurobranch.adaptive.var import WARNING as PERTURBATION_WARNING
from neurobranch.studio import load_snapshot, load_trajectory_store
from neurobranch.studio.components import badge
from neurobranch.studio.course import load_chapters
from neurobranch.studio.errata import counts, errata_rows
from neurobranch.studio.figures import trajectory_2d, trajectory_3d
from neurobranch.studio.theme import css

pytestmark = pytest.mark.requires_data


def test_course_is_read_not_duplicated() -> None:
    chapters = load_chapters()
    assert len(chapters) == 16
    assert chapters[0].number == 1
    assert "unit of inference" in chapters[0].body


def test_all_corrections_are_surfaced() -> None:
    snapshot = load_snapshot()
    rows = errata_rows(snapshot)
    assert len(rows) == 17
    assert {row["id"] for row in rows} == {f"X{i:02d}" for i in range(1, 18)}
    assert all(row["status"] for row in rows)
    assert counts(rows)


def test_no_confirmed_tier_in_studio_copy() -> None:
    snapshot = load_snapshot()
    assert "CONFIRMED" not in snapshot.claims_payload["tier_definitions"]
    assert snapshot.claims_payload["no_confirmed_tier"] is True


def test_perturbation_banner_is_exact() -> None:
    assert PERTURBATION_WARNING == (
        "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED."
    )


def test_badge_is_escaped() -> None:
    assert "&lt;script&gt;" in badge("<script>alert(1)</script>")
    assert "<script>" not in badge("<script>alert(1)</script>")


def test_theme_has_both_modes() -> None:
    dark = css(False)
    light = css(True)
    assert "--nb-bg" in dark and "--nb-bg" in light
    assert dark != light
    assert "radial-gradient" in dark


def test_3d_and_fallback_use_same_origin() -> None:
    store = load_trajectory_store()
    case = store.case(0, store.session(0)["showcase_landmarks"][1])
    fig3 = trajectory_3d(case)
    fig2 = trajectory_2d(case)
    assert len(fig3.data) >= 5
    assert len(fig3.frames) == 21
    assert len(fig2.data) >= 4
    html = fig3.to_html(include_plotlyjs="inline", full_html=True)
    assert "Observed continuation" in html
    assert "PC1" in html
