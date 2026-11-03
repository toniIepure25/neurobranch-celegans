"""Shared, session-linked controls for the Studio."""

from __future__ import annotations

import streamlit as st

from neurobranch.studio.metadata import StudioSnapshot
from neurobranch.studio.trajectories import TrajectoryStore

DEFAULT_MODELS = ("B3_population", "B5_common_k10")


def session_selector(snapshot: StudioSnapshot, *, key: str = "studio_session") -> int:
    ids = [int(row["session"]) for row in snapshot.sessions]
    current = int(st.session_state.get("studio_session", ids[0]))
    if current not in ids:
        current = ids[0]
    choice = st.selectbox(
        "Recording",
        ids,
        index=ids.index(current),
        format_func=lambda sid: (
            f"Recording {sid} · {snapshot.session_by_id(sid)['dataset_name']} · "
            f"{snapshot.session_by_id(sid)['n_frames']:,} frames"
        ),
        key=f"{key}_select",
    )
    st.session_state["studio_session"] = int(choice)
    return int(choice)


def landmark_selector(store: TrajectoryStore, session: int, *, key: str = "studio_landmark") -> int:
    meta = store.session(session)
    options = [int(v) for v in meta["landmarks"]]
    showcase = [int(v) for v in meta["showcase_landmarks"]]
    current = int(st.session_state.get("studio_landmark", showcase[1] if len(showcase) > 1 else options[0]))
    if current not in options:
        current = showcase[1] if len(showcase) > 1 else options[0]
    choice = st.selectbox(
        "Forecast origin (landmark)",
        options,
        index=options.index(current),
        format_func=lambda t: (
            f"t = {t:,}  ·  {store.time_seconds(session)[t]:.1f}s"
            + ("  ·  showcase" if t in showcase else "")
        ),
        key=f"{key}_select_{session}",
        help=(
            "Landmarks are frozen H20 evaluation origins. The forecast starts at t+1; "
            "the observed continuation is shown for evaluation only."
        ),
    )
    st.session_state["studio_landmark"] = int(choice)
    return int(choice)


def model_selector(*, key: str = "studio_models") -> tuple[str, ...]:
    labels = {
        "B3_population": "B3 · population VAR",
        "B5_common_k10": "B5 · ten common factors",
        "B1_matched_own": "B1m · independent AR",
        "B0_persistence": "B0 · persistence",
    }
    current = tuple(st.session_state.get("studio_models", DEFAULT_MODELS))
    chosen = st.multiselect(
        "Displayed models",
        list(labels),
        default=[m for m in current if m in labels] or list(DEFAULT_MODELS),
        format_func=lambda m: labels[m],
        key=f"{key}_widget",
        help="All four trajectories are model outputs on the same frozen PCA coordinates.",
    )
    if not chosen:
        chosen = list(DEFAULT_MODELS)
    st.session_state["studio_models"] = tuple(chosen)
    return tuple(chosen)


def case_for(store: TrajectoryStore, session: int, landmark: int, horizon: int = 20):
    return store.case(session, landmark, horizon=horizon)


def go_to(section: str, session: int | None = None, landmark: int | None = None) -> None:
    st.session_state["studio_section"] = section
    if session is not None:
        st.session_state["studio_session"] = int(session)
    if landmark is not None:
        st.session_state["studio_landmark"] = int(landmark)
    st.rerun()
