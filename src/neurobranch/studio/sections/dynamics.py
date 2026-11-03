"""Interactive 3D neural-state projection."""

from __future__ import annotations

import numpy as np
import streamlit as st

from neurobranch.studio.components import (
    badge,
    caption,
    figure_caption,
    note,
    section_header,
    warning,
)
from neurobranch.studio.controls import (
    case_for,
    go_to,
    landmark_selector,
    model_selector,
    session_selector,
)
from neurobranch.studio.figures import forecast_time_series, trajectory_2d, trajectory_3d
from neurobranch.studio.formatting import fixed, seconds, signed
from neurobranch.studio.metadata import StudioSnapshot
from neurobranch.studio.trajectories import TrajectoryStore


def render(
    snapshot: StudioSnapshot, store: TrajectoryStore, *, light: bool = False
) -> None:
    section_header(
        "02 · 3D NEURAL DYNAMICS",
        "A state space, not an anatomy.",
        "PC1–PC3 are the first three axes of a 10-component PCA fitted on training "
        "frames only. Nearby points mean similar population activity under that "
        "projection; they do not mean physical proximity, shared synapses, or a "
        "circuit diagram.",
    )
    warning(
        "The axes are statistical coordinates of a training-only PCA. The plot is "
        "not a worm brain, not a connectome, and not a 3D anatomical reconstruction."
    )

    c1, c2, c3 = st.columns([1.2, 1.48, 1.0], gap="small")
    with c1:
        session = session_selector(snapshot)
    with c2:
        landmark = landmark_selector(store, session)
    with c3:
        models = model_selector()

    meta = store.session(session)
    case = case_for(store, session, landmark)
    evr = np.asarray(case["explained_variance_ratio"])

    with st.expander("View controls · camera, observed context, ensembles, animation", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            show_truth = st.toggle("Observed continuation", value=True)
        with c2:
            show_ensemble = st.toggle("Sampled model paths (8)", value=False)
        with c3:
            context_tail = st.slider("History length (frames)", 40, 300, 160, 20)
        with c4:
            frame_ms = st.slider("Playback step (ms)", 60, 400, 140, 20)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            show_context = st.toggle("Observed history", value=True)
        with c2:
            reset_key = st.button("Reset camera", use_container_width=True)
        if reset_key:
            st.session_state["studio_3d_reset"] = int(st.session_state.get("studio_3d_reset", 0)) + 1
        with c3:
            st.caption("Drag to orbit · scroll to zoom · right-drag to pan.")
        with c4:
            st.caption("Play/Pause controls are inside the figure.")

    left, right = st.columns([2.35, 1], gap="large")
    with left:
        tabs = st.tabs(["3D explorer", "2D fallback · PC1–PC2"])
        with tabs[0]:
            fig = trajectory_3d(
                case,
                visible_models=models,
                show_truth=show_truth,
                show_ensemble=show_ensemble,
                show_context=show_context,
                context_tail=context_tail,
                light=light,
                frame_duration=frame_ms,
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                    "modeBarButtonsToRemove": ["lasso3d", "select2d", "lasso2d"],
                },
                key=f"studio3d_{session}_{landmark}_{st.session_state.get('studio_3d_reset', 0)}",
            )
            figure_caption(
                f"Recording {session} ({meta['dataset_name']}) · origin t={landmark} "
                f"at {seconds(case['origin_time'])} · forecast starts at t+1. "
                "Observed continuation is shown only to evaluate the forecasts; it is "
                "never an input. Sampled paths are eight draws from the frozen "
                "innovation ensemble; the heavy line is the deterministic mean."
            )
            st.download_button(
                "Download this view (HTML)",
                fig.to_html(include_plotlyjs="inline", full_html=True),
                file_name=f"neurobranch_3d_s{session}_t{landmark}.html",
                mime="text/html",
            )
        with tabs[1]:
            fig2 = trajectory_2d(case, visible_models=models, light=light)
            st.plotly_chart(fig2, use_container_width=True, key=f"studio2d_{session}_{landmark}")
            figure_caption(
                "The 2D fallback uses exactly the same projections and origin. It is "
                "provided for environments where WebGL is unavailable; it is not a "
                "different analysis."
            )

    with right:
        st.markdown(
            f"""
            <div class="nb-panel">
              <h4>Forecast origin</h4>
              <div style="font-size:1.35rem;font-weight:700">t = {landmark:,}</div>
              <div style="color:inherit;opacity:0.7;font-size:0.8rem">
                {seconds(case["origin_time"])} from recording start · recording {session}
              </div>
              <div class="nb-divider"></div>
              <div style="font-size:0.82rem;line-height:1.7">
                <div><b>PC1</b> {case["origin"][0]:+.3f} <span style="opacity:.55">({evr[0]*100:.1f}% train variance)</span></div>
                <div><b>PC2</b> {case["origin"][1]:+.3f} <span style="opacity:.55">({evr[1]*100:.1f}%)</span></div>
                <div><b>PC3</b> {case["origin"][2]:+.3f} <span style="opacity:.55">({evr[2]*100:.1f}%)</span></div>
                <div style="opacity:.65;margin-top:.5rem">3-D cumulative {case["explained_3d"]*100:.1f}% of training variance</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            badge(f"H20 energy · B3 {fixed(case['scores']['B3_population'])}", "prediction")
            + "<span style='display:inline-block;width:.45rem'></span>"
            + badge(f"B5 {fixed(case['scores']['B5_common_k10'])}", "control"),
            unsafe_allow_html=True,
        )
        diff = case["scores"]["B5_common_k10"] - case["scores"]["B3_population"]
        st.markdown(
            f"<div class='nb-caption'>Δ (B5 − B3) at H20: <b>{signed(diff)}</b> · "
            "positive means B3 lower energy. One landmark is not the population result.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("**Linked views**")
        if st.button("Inspect recorded activity here", use_container_width=True):
            go_to("activity", session, landmark)
        if st.button("See this landmark in the evidence review", use_container_width=True):
            go_to("evidence", session, landmark)
        if st.button("Compare fitted models at this origin", use_container_width=True):
            go_to("models", session, landmark)
        st.markdown("---")
        st.markdown("**Coordinates shown**")
        st.markdown(
            "- **Measured history:** observed PCA scores before the origin.\n"
            "- **Observed continuation:** held-out observed scores used for evaluation.\n"
            "- **Model mean:** deterministic rollout from the origin, zero innovation.\n"
            "- **Sampled paths:** eight innovations from the frozen block-bootstrap ensemble."
        )
        note(
            "A visually tight model path is not evidence of mechanism. The 3D geometry "
            "is a low-dimensional summary of a 592-dimensional predictor space across "
            "recordings with different predictor counts."
        )

    st.markdown("---")
    st.markdown("#### The trajectory in PC1 over time")
    st.plotly_chart(
        forecast_time_series(case, visible_models=models, component=0, light=light),
        use_container_width=True,
        key=f"studio_ts_{session}_{landmark}",
    )
    figure_caption(
        "The time axis is the recorded frame-to-second axis, not a resampled or "
        "idealised clock. Forecasts and observations are aligned at t+1."
    )
    caption(
        "Source: reports/scientific_studio/kato_trajectories.npz (built by "
        "scripts/studio_build_trajectories.py with H=100 innovations, storing the "
        "first 20 steps). Scores agree with model_per_landmark.csv within 2e-5."
    )
