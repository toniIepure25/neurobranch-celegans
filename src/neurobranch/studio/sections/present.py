"""Supervisor presentation mode: ten slides, no new inference."""

from __future__ import annotations

import streamlit as st

from neurobranch.studio.components import badge, note, warning
from neurobranch.studio.figures import evidence_forest, trajectory_3d
from neurobranch.studio.formatting import interval, percent, signed
from neurobranch.studio.metadata import StudioSnapshot
from neurobranch.studio.trajectories import TrajectoryStore

SLIDES = [
    "A bounded result is still a result",
    "What was measured",
    "A state space, not an anatomy",
    "Four models, one contrast",
    "The primary result",
    "The boundary is part of the result",
    "Calibration and capacity",
    "What we can defend",
    "What would change our mind",
    "The next honest experiment",
]


def _notes(text: str) -> None:
    with st.expander("Presenter notes", expanded=False):
        st.markdown(text)


def _nav(index: int) -> None:
    c1, c2, c3 = st.columns([1, 2.4, 1])
    with c1:
        if st.button("← Previous", disabled=index == 0, use_container_width=True):
            st.session_state["studio_present_index"] = max(0, index - 1)
            st.rerun()
    with c2:
        st.progress((index + 1) / len(SLIDES), text=f"{index + 1} / {len(SLIDES)} · {SLIDES[index]}")
    with c3:
        if st.button("Next →", disabled=index == len(SLIDES) - 1, use_container_width=True):
            st.session_state["studio_present_index"] = min(len(SLIDES) - 1, index + 1)
            st.rerun()


def render(snapshot: StudioSnapshot, store: TrajectoryStore, *, light: bool = False) -> None:
    index = int(st.session_state.get("studio_present_index", 0))
    index = max(0, min(len(SLIDES) - 1, index))
    primary = snapshot.primary
    top = st.columns([1, 1, 1, 1])
    with top[0]:
        if st.button("Exit presentation", use_container_width=True):
            st.session_state["studio_present"] = False
            st.rerun()
    with top[3]:
        st.markdown(
            badge("SUPERVISOR MODE", "observed") + badge("EXPLORATORY", "prediction"),
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="nb-slide"><div class="num">SLIDE {index + 1:02d} / {len(SLIDES):02d}</div>'
        f"<h2>{SLIDES[index]}</h2>",
        unsafe_allow_html=True,
    )
    if index == 0:
        st.markdown(
            """
            <div class="body">
              <p><b>NeuroBranch Scientific Studio</b> is a reanalysis of five Kato 2015
              <i>WT_NoStim</i> whole-brain calcium recordings. It asks whether a full
              population model retains predictive structure beyond a ten-factor
              common-state control.</p>
              <p>The answer is an exploratory, horizon-dependent result — not a
              validated digital twin and not a biological mechanism.</p>
              <p style="margin-top:2rem;color:#8ca4ad">Prepared for methodological review · Denis Kovačević</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        note(
            "Opening line: 'I want to show you a result that is useful because we can "
            "state exactly what it does not establish.' Do not start with the 3D plot; "
            "start with the unit of inference."
        )
    elif index == 1:
        identity = snapshot.dataset_identity
        total_frames = sum(int(s["n_frames"]) for s in snapshot.sessions.values())
        total_predictors = sum(int(s["n_predictors"]) for s in snapshot.sessions.values())
        n_sessions = len(snapshot.sessions)
        st.markdown(
            f"""
            <div class="body">
              <p><b>{n_sessions} recordings, {total_frames:,} frames,
              {total_predictors} predictors.</b> The archive is
              verified against the OSF object by size and SHA256. The documentation is
              not the source of truth: its historical size and field names were wrong.</p>
              <p>The {n_sessions} acquisition preparations are consistent with the publication's
              {n_sessions} animals, but the archive exposes <b>no animal key</b>. The mapping is
              inferred. They are one laboratory/preparation, not an independent cohort.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("SHA256 verified", "yes")
        c2.metric("Frames", f"{total_frames:,}")
        c3.metric("Distinct preparations", str(n_sessions))
        note(
            f"Archive: {identity['archive']['actual_size_bytes']:,} bytes; "
            f"SHA256 {identity['archive']['actual_sha256'][:12]}… . "
            "The OSF readme has no worm ID. Say 'five inferred animals', not "
            "'five animals' as a documented primary key."
        )
    elif index == 2:
        st.markdown(
            """
            <div class="body">
              <p>PC1–PC3 are the first three axes of a <b>training-only</b> 10-component
              PCA. They summarise population activity. They are <b>not</b> anatomical
              coordinates, a connectome, or a physical brain space.</p>
              <p>A tight predicted path means the model reproduces this projection;
              it does not mean the model has found the circuit.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        sid = int(st.session_state.get("studio_session", 0))
        landmark = int(
            st.session_state.get(
                "studio_landmark", store.session(sid)["showcase_landmarks"][1]
            )
        )
        case = store.case(sid, landmark)
        st.plotly_chart(
            trajectory_3d(case, show_truth=True, show_ensemble=False, light=light),
            use_container_width=True,
            key=f"present3d_{sid}_{landmark}",
        )
        if st.button("Open the interactive 3D explorer", use_container_width=False):
            st.session_state["studio_present"] = False
            st.session_state["studio_section"] = "dynamics"
            st.rerun()
        note(
            "Drag the view, then use the Play control. Point out the gap between the "
            "amber/violet model paths and the dotted observed continuation. Do not call "
            "the axes anatomy."
        )
    elif index == 3:
        st.markdown(
            """
            <div class="body">
              <p><b>B0</b> persistence · <b>B1m</b> independent AR(1), 200-frame window ·
              <b>B3</b> full population VAR, 500-frame ridge · <b>B5</b> ten-factor
              common-state control, 500-frame ridge.</p>
              <p>All four are lag-one models at the calcium-imaging timescale. The
              coefficients are effective transition parameters, not synaptic weights.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Primary contrast", "B5 − B3")
        c2.metric("Endpoint", "H20 PCA-10 energy")
        c3.metric("Unit", "recording / landmark")
        note(
            "The contrast sign: positive means B3 has lower energy. This is a proper "
            "score for a probabilistic forecast, not a biological effect size."
        )
    elif index == 4:
        st.markdown(
            f"""
            <div class="body">
              <p><b>{signed(primary["difference"])}</b> weighted paired mean over
              {primary["n"]} overlapping landmarks; <b>{percent(primary["relative_to_b3"])}</b>
              of the B3 mean. {primary["sessions_positive"]} of {primary["n_sessions"]}
              recording means are positive.</p>
              <p>Conditional block-5 interval
              <b>{interval(primary["conditional_ci"]["ci_low"], primary["conditional_ci"]["ci_high"])}</b>.
              Recording-resampling interval
              <b>{interval(primary["recording_ci"]["ci_low"], primary["recording_ci"]["ci_high"])}</b>.
              The second is the relevant check for extending beyond these recordings,
              and it includes zero.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(evidence_forest(snapshot, light=light), use_container_width=True, key="present_forest")
        note(
            "Do not say 'significant'. Say: the conditional interval excludes zero, the "
            "recording-level interval does not, and the sign test p is "
            f"{primary['sign_test_one_sided_p']:.4f}. With five units, this is weak."
        )
    elif index == 5:
        st.markdown(
            f"""
            <div class="body">
              <p><b>H1 reverses.</b> At one step, B3 is worse than B5 by
              <b>{-snapshot.uncertainty["horizon"][0]["mean_advantage"]:+.3f}</b> energy
              units. The advantage only appears at multi-step horizons.</p>
              <p><b>MSE is null.</b> Mean difference
              <b>{signed(primary["mse_difference"])}</b>, interval
              <b>{interval(*primary["mse_ci"])}</b>, crosses zero. A lower energy score can
              reflect predictive spread, not a better conditional mean.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        note(
            "This is the slide that prevents overstatement. The horizon dependence and "
            "the MSE null are not footnotes; they constrain the claim."
        )
    elif index == 6:
        common_pct = round(100 * snapshot.capacity["ordered_shares"]["baseline_B4m"]["common"])
        st.markdown(
            f"""
            <div class="body">
              <p><b>Calibration is not established.</b> B5's 95% marginal coverage is
              {percent(snapshot.statistical["paired_effect"]["coverage_at_nominal_95"]["B5_common_k10"]["95"])};
              B3's is {percent(snapshot.statistical["paired_effect"]["coverage_at_nominal_95"]["B3_population"]["95"])}.
              This is average marginal per-dimension coverage, not joint 10-D calibration.</p>
              <p><b>Capacity is not matched.</b> B3 has
              {snapshot.capacity["ratio_min"]:.2f}–{snapshot.capacity["ratio_max"]:.2f}×
              the nominal coefficients of B5-10 and arbitrary residual coordinates. The
              {common_pct}/{100 - common_pct} ordered decomposition is arithmetic,
              not a structural partition.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        warning(
            "The residual advantage is not identifiable as cross-neuron information "
            "from this design. Do not describe a VAR coefficient as a directed connection."
        )
    elif index == 7:
        st.markdown(
            """
            <div class="body">
              <p><b>We can defend:</b> within these five recordings, at H20, on this
              training-only PCA, B3 has lower mean energy than B5-10; the direction is
              not uniform across recordings or horizons.</p>
              <p><b>We cannot defend:</b> a general forecasting advantage, an animal-level
              effect, joint calibration, a mechanism, a causal perturbation, or a
              validated digital twin.</p>
              <p><b>Status:</b> READY_FOR_METHODOLOGICAL_SUPERVISOR_REVIEW;
              NOT_READY_FOR_CONFIRMATORY_OR_BIOLOGICAL_CLAIMS.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        note(
            "The strongest defensible statement is in statistical_audit.json. Read it "
            "verbatim if asked for the claim."
        )
    elif index == 8:
        st.markdown(
            """
            <div class="body">
              <p><b>Independent animals</b> with documented IDs, a pre-registered split
              and endpoint, and the animal as the unit.</p>
              <p><b>A capacity-matched residual model</b> or an explicit common-input
              measurement that survives conditioning.</p>
              <p><b>A prospective, sham-controlled perturbation</b> with held-out targets
              and animals, and a pre-registered direction.</p>
              <p><b>A fresh test tail</b> that has not been reused across earlier phases.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        note(
            "If any of these fails, the honest outcome is to preserve the negative "
            "result. Do not re-tune on the same tail."
        )
    else:
        st.markdown(
            """
            <div class="body">
              <p><b>Next experiment:</b> independent-animal replication of the H20
              contrast, with the baseline family and the cross-neuron test fixed before
              looking at the new scores.</p>
              <p><b>Falsification:</b> the animal-level B3-minus-B1 energy interval
              includes zero, or the sign reverses in most animals.</p>
              <p>The Studio is a review instrument. It is not a validated digital twin,
              and it does not predict the outcome of the next experiment.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        note(
            "Close by returning to the distinction between a number, a result, a claim, "
            "and a mechanism. Invite Denis to attack the unit, the split, and the capacity "
            "confound first."
        )
    st.markdown("</div>", unsafe_allow_html=True)
    _nav(index)
