"""NeuroBranch Scientific Studio.

Launch from the repository root:

    streamlit run app/neurobranch_studio.py

The application is a read-only presentation layer over committed evidence.  It
does not fit new models for the inferential claims and it never rewrites the
frozen result files.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

from neurobranch.studio.components import badge, inject_css, sidebar_brand  # noqa: E402
from neurobranch.studio.metadata import ArtifactUnavailable, load_snapshot  # noqa: E402
from neurobranch.studio.sections import (  # noqa: E402
    activity,
    dynamics,
    evidence,
    models,
    notebook,
    overview,
    perturbations,
    present,
)
from neurobranch.studio.trajectories import load_trajectory_store  # noqa: E402

SECTIONS = {
    "Overview": "overview",
    "3D Neural Dynamics": "dynamics",
    "Neural Activity": "activity",
    "Models & Forecasting": "models",
    "Uncertainty & Evidence": "evidence",
    "Virtual Perturbations": "perturbations",
    "Research Notebook": "notebook",
}
SECTION_LABEL_BY_ID = {value: key for key, value in SECTIONS.items()}
RENDERERS = {
    "overview": overview.render,
    "dynamics": dynamics.render,
    "activity": activity.render,
    "models": models.render,
    "evidence": evidence.render,
    "perturbations": perturbations.render,
    "notebook": notebook.render,
}

st.set_page_config(
    page_title="NeuroBranch Scientific Studio",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "studio_section" not in st.session_state:
    st.session_state["studio_section"] = "overview"
if "studio_session" not in st.session_state:
    st.session_state["studio_session"] = 0
if "studio_models" not in st.session_state:
    st.session_state["studio_models"] = ("B3_population", "B5_common_k10")
if "studio_present" not in st.session_state:
    st.session_state["studio_present"] = False

try:
    snapshot = load_snapshot()
    store = load_trajectory_store()
except ArtifactUnavailable as exc:
    light = bool(st.session_state.get("studio_light", False))
    inject_css(light)
    st.error(str(exc))
    st.code(
        "python -X utf8 scripts/studio_build_trajectories.py\n"
        "python -X utf8 -m pytest -q tests/test_studio_artifacts.py",
        language="powershell",
    )
    st.stop()

presentation = bool(st.session_state.get("studio_present", False))

with st.sidebar:
    sidebar_brand()
    light = st.toggle("Light mode", value=bool(st.session_state.get("studio_light", False)))
    st.session_state["studio_light"] = light
    st.markdown(
        badge("FIVE RECORDINGS", "observed")
        + badge("ONE PREPARATION", "neutral")
        + badge("EXPLORATORY", "prediction"),
        unsafe_allow_html=True,
    )
    st.markdown("---")
    if not presentation:
        section_id = st.session_state.get("studio_section", "overview")
        if st.session_state.get("_nav_sync") != section_id:
            st.session_state["studio_nav"] = SECTION_LABEL_BY_ID.get(section_id, "Overview")
            st.session_state["_nav_sync"] = section_id
        choice = st.radio(
            "Navigate",
            list(SECTIONS),
            key="studio_nav",
            label_visibility="collapsed",
        )
        st.session_state["studio_section"] = SECTIONS[choice]
        st.session_state["_nav_sync"] = SECTIONS[choice]
        st.markdown("---")
        if st.button("Presentation mode", use_container_width=True):
            st.session_state["studio_present"] = True
            st.session_state["studio_present_index"] = 0
            st.rerun()
    else:
        st.markdown("**Presentation mode**")
        st.caption("Use the slide controls in the main panel.")
        if st.button("Exit to Studio", use_container_width=True):
            st.session_state["studio_present"] = False
            st.rerun()

    st.markdown("---")
    with st.expander("Advanced / Legacy", expanded=False):
        st.caption(
            "Frozen dashboards and the focused supervisor demo remain available. "
            "They are not the source of truth for current wording."
        )
        st.code(
            "streamlit run app/streamlit_app.py\n"
            "streamlit run app/handoff_demo.py",
            language="powershell",
        )
        st.markdown(
            "The legacy app preserves 22 exploratory tabs and historical artifacts. "
            "Use the Studio for the audited H20 comparison."
        )
        if st.button("Open the focused handoff demo", use_container_width=True):
            st.info(
                "Run `streamlit run app/handoff_demo.py` in a second terminal. "
                "This Studio does not silently substitute a different dataset."
            )
    with st.expander("Source of truth", expanded=False):
        st.markdown(
            "1. `reports/simulator/kato_common_state/model_per_landmark.csv`\n"
            "2. `reports/master_review/statistical_audit.json`\n"
            "3. `reports/master_review/claims_ledger.json`\n"
            "4. `reports/scientific_studio/kato_trajectories.npz`"
        )
        st.caption(
            "The Studio recomputes display aggregates from these files. It does not "
            "invent a second result set."
        )
    st.markdown("---")
    st.caption(
        "Prepared for methodological review. Not for confirmatory or biological "
        "claims. No data were collected, and no test-tail performance was optimized."
    )

inject_css(light)
if presentation:
    st.markdown(
        "<style>[data-testid='stSidebar']{display:none;} .block-container{max-width:1280px;}</style>",
        unsafe_allow_html=True,
    )
    present.render(snapshot, store, light=light)
    st.stop()

section = st.session_state.get("studio_section", "overview")
renderer = RENDERERS.get(section, overview.render)
if section in {"dynamics", "activity", "models", "perturbations"}:
    renderer(snapshot, store, light=light)
else:
    renderer(snapshot, light=light)

st.markdown("---")
st.markdown(
    """
    <div style="display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;
                color:var(--nb-muted);font-size:0.72rem;line-height:1.5">
      <div>NeuroBranch Scientific Studio · read-only evidence layer · five Kato WT_NoStim recordings</div>
      <div>Not a digital twin · not a connectome · not a causal simulator</div>
    </div>
    """,
    unsafe_allow_html=True,
)
