"""Recorded neural activity and its link to the projected state space."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from neurobranch.studio.activity import (
    load_activity,
    neuron_panel,
    rendering_downsample,
)
from neurobranch.studio.components import (
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
    session_selector,
)
from neurobranch.studio.figures import activity_heatmap, activity_traces, forecast_time_series
from neurobranch.studio.metadata import StudioSnapshot
from neurobranch.studio.trajectories import TrajectoryStore


def render(
    snapshot: StudioSnapshot, store: TrajectoryStore, *, light: bool = False
) -> None:
    section_header(
        "03 · NEURAL ACTIVITY",
        "The measured signals behind the state space.",
        "Whole-brain calcium ΔF/F traces from the same recording and time axis used by "
        "the models. The heatmap is display-downsampled; the model never sees the "
        "downsample. Named traces are cell-type labels, while numeric ROIs are "
        "unregistered segments.",
    )
    c1, c2 = st.columns([1.2, 1.5], gap="small")
    with c1:
        session = session_selector(snapshot)
    with c2:
        landmark = landmark_selector(store, session)
    activity = load_activity(session)
    names = activity["names"]
    time = activity["time"]

    with st.expander("Trace controls · data transform, cells, colour scale", expanded=True):
        c1, c2, c3 = st.columns([1, 1, 1.2])
        with c1:
            raw = st.toggle("Raw ΔF/F", value=False, help="Off: standardized z-score used by the models. On: corrected ΔF/F.")
        with c2:
            named_only = st.toggle("Named cells only", value=False)
        with c3:
            sort_by_variance = st.toggle("Order by variance", value=True)
        candidates = [n for n, is_named in zip(names, activity["named_mask"]) if (is_named or not named_only)]
        variances = np.nanvar(activity["raw"] if raw else activity["z"], axis=0)
        var_map = dict(zip(names, variances))
        candidates = sorted(candidates, key=lambda n: var_map[n], reverse=sort_by_variance)
        default = candidates[:4]
        selected = st.multiselect(
            "Highlight traces (up to four in the line chart)",
            options=candidates,
            default=[n for n in default if n in candidates],
            max_selections=8,
            help="All predictor traces remain in the heatmap; the line chart shows the first four selections.",
        )

    matrix = activity["raw"] if raw else activity["z"]
    rendered, rendered_time, stride = rendering_downsample(matrix, time, max_columns=1100)
    split_time = {
        key: [float(time[v[0]]), float(time[v[1] - 1])]
        for key, v in activity["split_bounds"].items()
        if v[0] < len(time)
    }
    # The heatmap helper expects frame positions; convert the display-only axis
    # positions back to frame indices for the boundary lines.
    split_frames = {
        key: [int(np.searchsorted(time, value[0])), int(np.searchsorted(time, value[1]))]
        for key, value in split_time.items()
    }
    fig = activity_heatmap(
        rendered,
        rendered_time,
        names,
        split_bounds=split_frames,
        light=light,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"activity_heat_{session}")
    figure_caption(
        f"{activity['n_predictors']} predictor traces · {activity['n_frames']:,} frames · "
        f"display stride {stride} (full-resolution input is used everywhere else). "
        "Dotted vertical lines: train/validation boundaries. The test tail is not a "
        "fresh confirmatory holdout; it was reused across earlier phases."
    )

    if selected:
        panel = neuron_panel(activity, selected[:4], raw=raw)
        st.plotly_chart(
            activity_traces(panel, time, selected[:4], light=light),
            use_container_width=True,
            key=f"activity_lines_{session}_{len(selected)}",
        )
        figure_caption(
            "Four selected traces on the recorded clock. The four series are not "
            "one-to-one cell identities across recordings; labels are per-recording."
        )
    else:
        st.info("Select at least one trace to draw the line chart.")

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.markdown("#### Linked state-space view")
        case = case_for(store, session, landmark)
        st.plotly_chart(
            forecast_time_series(
                case,
                visible_models=("B3_population", "B5_common_k10"),
                component=0,
                light=light,
            ),
            use_container_width=True,
            key=f"activity_pc1_{session}_{landmark}",
        )
        figure_caption(
            "PC1 through the selected origin: measured history, held-out observed "
            "continuation, and model means. This is a state-space comparison, not a "
            "per-neuron simulated calcium trace."
        )
        if st.button("Open this origin in 3D", use_container_width=True):
            go_to("dynamics", session, landmark)
    with right:
        st.markdown("#### What the signals are")
        st.markdown(
            f"""
            <div class="nb-panel">
              <div style="font-size:0.86rem;line-height:1.65">
                <div><b>Recording</b> {activity['dataset_name']}</div>
                <div><b>Sampling</b> {activity['fps']:.3f} Hz · median frame {1/activity['fps']:.3f}s</div>
                <div><b>Traces</b> {activity['n_traces']} · <b>predictors</b> {activity['n_predictors']}</div>
                <div><b>Frames</b> {activity['n_frames']:,} · <b>duration</b> {time[-1]:.1f}s</div>
                <div><b>Named</b> {int(np.sum(activity['named_mask']))} · <b>numeric</b> {int(np.sum(~activity['named_mask']))}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        note(
            "Calcium ΔF/F is a filtered, indirect measurement. A lag-one model at "
            "~0.34 s per frame is an effective imaging-timescale transition, not a "
            "synaptic weight, membrane time constant, or gap-junction conductance."
        )
        warning(
            "Behaviour labels (FWD, SLOW, DT, VT, REV1, REV2, REVSUS, NOSTATE) are "
            "derived using activity in command/interneuron circuits. They are not an "
            "independent behavioural measurement, and the label-defining cells are "
            "removed from predictors only where present."
        )
        with st.expander("Behaviour-label counts", expanded=False):
            states = pd.Series(activity["states"]).value_counts()
            st.dataframe(
                pd.DataFrame({"state": states.index, "frames": states.values}),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("---")
    st.markdown("#### Recording inventory")
    rows = []
    for meta in snapshot.sessions:
        rows.append(
            {
                "session": meta["session"],
                "dataset": meta["dataset_name"],
                "frames": meta["n_frames"],
                "predictors": meta["n_predictors"],
                "named": meta["n_named"],
                "fps": round(meta["fps"], 4),
                "landmarks": len(meta["landmarks"]),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    caption(
        "Source: data/raw/kato2015/WT_NoStim.mat via neurobranch.diagnostics.common.load_nostim; "
        "the same loader used by the frozen evaluation. Display downsampling is the only "
        "transform applied in this section."
    )
