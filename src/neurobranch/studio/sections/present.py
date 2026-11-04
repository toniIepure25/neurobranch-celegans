"""Supervisor presentation mode: ten slides, no new inference.

The slide shell is a real Streamlit container.  Earlier versions opened a
``<div class="nb-slide">`` in one markdown call and closed it in another; Streamlit
closes each markdown document independently, so the title card became a 520px
empty box and all scientific content rendered below it.
"""

from __future__ import annotations

import streamlit as st

from neurobranch.studio.components import badge
from neurobranch.studio.figures import evidence_forest, trajectory_3d
from neurobranch.studio.formatting import bytes_human, fixed, interval, percent, signed
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

SLIDE_MESSAGES = [
    "A useful result can be bounded more tightly than it is usually sold.",
    "Five recordings, one experimental cohort, and 592 included predictors.",
    "The projection summarises population state; it is not anatomy.",
    "Four models isolate what the population model adds beyond a ten-factor control.",
    "At H20, B3 has lower mean energy; the recording interval still includes zero.",
    "The advantage is horizon-dependent, and the MSE improvement is not established.",
    "Poor calibration and unmatched capacity are part of the result, not footnotes.",
    "What survives is narrower than a general forecasting claim.",
    "The next study must be able to falsify the claim.",
    "Replicate at the animal level before asking for mechanism.",
]


def _notes(text: str) -> None:
    with st.expander("Presenter notes", expanded=False):
        st.markdown(text)


def _header(index: int) -> None:
    st.markdown(
        f"""
        <div class="nb-slide-head">
          <div class="num">SLIDE {index + 1:02d} / {len(SLIDES):02d}</div>
          <h2>{SLIDES[index]}</h2>
          <div class="nb-slide-message">{SLIDE_MESSAGES[index]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _nav(index: int) -> None:
    st.markdown('<div class="nb-slide-rule"></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2.4, 1])
    with c1:
        if st.button("← Previous", disabled=index == 0, use_container_width=True):
            st.session_state["studio_present_index"] = max(0, index - 1)
            st.rerun()
    with c2:
        st.progress(
            (index + 1) / len(SLIDES),
            text=f"{index + 1} / {len(SLIDES)} · {SLIDES[index]}",
        )
    with c3:
        if st.button("Next →", disabled=index == len(SLIDES) - 1, use_container_width=True):
            st.session_state["studio_present_index"] = min(len(SLIDES) - 1, index + 1)
            st.rerun()


def _body(html: str) -> None:
    st.markdown(f'<div class="nb-slide-body">{html}</div>', unsafe_allow_html=True)


def render(snapshot: StudioSnapshot, store: TrajectoryStore, *, light: bool = False) -> None:
    index = int(st.session_state.get("studio_present_index", 0))
    index = max(0, min(len(SLIDES) - 1, index))
    primary = snapshot.primary
    identity = snapshot.dataset_identity
    archive = identity["archive"]

    top = st.columns([1.25, 3.5, 1.45])
    with top[0]:
        if st.button("Exit presentation", use_container_width=True):
            st.session_state["studio_present"] = False
            st.rerun()
    with top[1]:
        st.caption("Methodological review · Kato 2015 WT_NoStim · no new inference")
    with top[2]:
        st.markdown(
            badge("EXPLORATORY EVIDENCE", "prediction"),
            unsafe_allow_html=True,
        )

    with st.container(key="nb_slide"):
        _header(index)

        if index == 0:
            _body(
                "<p><b>NeuroBranch Scientific Studio</b> is a reanalysis of five Kato 2015 "
                "<i>WT_NoStim</i> whole-brain calcium recordings. It asks whether a full "
                "population model retains predictive structure beyond a ten-factor "
                "common-state control.</p>"
                "<p>The answer is exploratory and horizon-dependent — not a validated "
                "digital twin and not a biological mechanism.</p>"
            )
            left, right = st.columns(2, gap="medium")
            with left:
                st.markdown(
                    """
                    <div class="nb-panel">
                      <h4>What this is</h4>
                      <div>A frozen, reproducible reanalysis of one public cohort:
                      five recordings, 15,662 frames, 592 included predictors, and
                      175 overlapping H20 forecast landmarks.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with right:
                st.markdown(
                    """
                    <div class="nb-panel">
                      <h4>What it is not</h4>
                      <div>Not confirmatory, animal-level, or biological. Not a
                      connectome, a digital twin, or a causal model. The unit is a
                      forecast landmark, not an animal.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown(
                '<div class="nb-status-strip">'
                + badge("EXPLORATORY", "prediction")
                + badge("TEST TAILS REUSED", "danger")
                + " Earlier research phases reused the evaluation tails and explored model "
                "choices. This release adds no new test-tail optimization; the finding "
                "remains exploratory."
                "</div>",
                unsafe_allow_html=True,
            )
            _notes(
                "Opening line: 'I want to show you a result that is useful because we can "
                "state exactly what it does not establish.' Start with the unit of "
                "inference, not the 3D plot."
            )

        elif index == 1:
            total_frames = sum(int(s["n_frames"]) for s in snapshot.sessions)
            total_predictors = sum(int(s["n_predictors"]) for s in snapshot.sessions)
            n_sessions = len(snapshot.sessions)
            _body(
                "<p>Five recordings from <b>one experimental cohort</b>. The archive is "
                "verified against the OSF object by size and SHA256. The publication "
                "reports n=5 animals, but the archived data expose no authoritative "
                "one-to-one animal key.</p>"
            )
            c1, c2, c3 = st.columns(3, gap="small")
            with c1:
                st.metric("Neural recordings", str(n_sessions))
            with c2:
                st.metric("Frames", f"{total_frames:,}")
            with c3:
                st.metric("Included predictors", f"{total_predictors:,}")

            prov, read = st.columns([1, 1.25], gap="medium")
            with prov:
                st.markdown(
                    f"""
                    <div class="nb-panel">
                      <h4>Provenance</h4>
                      <div style="font-size:0.88rem;line-height:1.65">
                        Kato et al. (2015) · <i>WT_NoStim</i><br>
                        OSF object · {bytes_human(archive["actual_size_bytes"])}<br>
                        SHA256 {archive["actual_sha256"][:12]}… · verified
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with read:
                st.markdown(
                    """
                    <div class="nb-panel">
                      <h4>Reading the cohort</h4>
                      <div style="font-size:0.88rem;line-height:1.6">
                        Recordings are analysed separately as repeated sessions from
                        one cohort. The five-to-five animal mapping is <b>inferred</b>,
                        not documented. This is not an independent cohort and not
                        five verified distinct animals.
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            _notes(
                "Say 'five recordings from one experimental cohort'. Do not say 'five "
                "preparations' or imply that one animal was recorded five times. The "
                "paper reports five animals; the archive does not expose the key."
            )

        elif index == 2:
            _body(
                "<p>PC1–PC3 are the first three axes of a <b>training-only</b> "
                "10-component PCA. They summarise population activity. They are "
                "<b>not</b> anatomical coordinates, a connectome, or a physical brain "
                "space.</p>"
                "<p>A tight predicted path means the model reproduces this projection; "
                "it does not mean the model has found the circuit.</p>"
            )
            sid = int(st.session_state.get("studio_session", 0))
            landmark = int(
                st.session_state.get(
                    "studio_landmark", store.session(sid)["showcase_landmarks"][1]
                )
            )
            case = store.case(sid, landmark)
            st.plotly_chart(
                trajectory_3d(
                    case,
                    show_truth=True,
                    show_ensemble=False,
                    light=light,
                    height=270,
                ),
                use_container_width=True,
                key=f"present3d_{sid}_{landmark}",
            )
            st.markdown(
                '<div class="nb-caption">Held-out continuation is display-only. '
                'PC1–PC3 explain about 52% of total training variance in recording 0 '
                '(about 73% of retained PCA-10 variance).</div>',
                unsafe_allow_html=True,
            )
            _notes(
                "Drag the view, then use Play. Point out the gap between the model paths "
                "and the dotted observed continuation. Do not call the axes anatomy."
            )

        elif index == 3:
            _body(
                "<p>All four are lag-one models at the calcium-imaging timescale. "
                "Coefficients are effective transition parameters, not synaptic "
                "weights.</p>"
            )
            row1 = st.columns(2, gap="medium")
            with row1[0]:
                st.markdown(
                    """
                    <div class="nb-panel">
                      <h4>B0 · persistence</h4>
                      <div>No transition model. The forecast repeats the last observed
                      state.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with row1[1]:
                st.markdown(
                    """
                    <div class="nb-panel">
                      <h4>B1m · matched own-history AR</h4>
                      <div>Independent AR(1), 200-frame window. A matched-capacity
                      baseline for each neuron.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            row2 = st.columns(2, gap="medium")
            with row2[0]:
                st.markdown(
                    f"""
                    <div class="nb-panel">
                      <h4>B3 · full population VAR</h4>
                      <div>500-frame ridge. {snapshot.capacity["ratio_min"]:.2f}–
                      {snapshot.capacity["ratio_max"]:.2f}× the nominal coefficients of
                      B5-10.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with row2[1]:
                st.markdown(
                    """
                    <div class="nb-panel">
                      <h4>B5-10 · common-state control</h4>
                      <div>Own-history AR plus ten train-only PCA factors; 500-frame
                      ridge. The control is deliberately lower capacity.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown(
                '<div class="nb-equation">Reported contrast = energy(B5) − '
                'energy(B3) · positive values favour B3.</div>',
                unsafe_allow_html=True,
            )
            _notes(
                "The comparison is not capacity-matched. That is a limitation, not a "
                "reason to hide the result. Ask whether B5 is the strongest fair control."
            )

        elif index == 4:
            # The one-line message states the direction and the interval caveat;
            # the metrics carry the values, so no extra paragraph is needed here.
            c1, c2, c3 = st.columns(3, gap="small")
            with c1:
                st.metric("B3 mean H20 energy", fixed(primary["b3_mean"]))
            with c2:
                st.metric("B5-10 mean H20 energy", fixed(primary["b5_mean"]))
            with c3:
                st.metric("B5 − B3", signed(primary["difference"], 4))

            left, right = st.columns([1.55, 1], gap="medium")
            with left:
                st.plotly_chart(
                    evidence_forest(snapshot, light=light, height=230),
                    use_container_width=True,
                    key="present_forest",
                )
            with right:
                st.markdown(
                    f"""
                    <div class="nb-panel">
                      <h4>Read the intervals, then the mean</h4>
                      <div style="font-size:0.88rem;line-height:1.55">
                        Conditional landmark interval
                        <b>{interval(primary["conditional_ci"]["ci_low"], primary["conditional_ci"]["ci_high"])}</b><br>
                        Recording resampling
                        <b>{interval(primary["recording_ci"]["ci_low"], primary["recording_ci"]["ci_high"])}</b><br>
                        The recording interval includes zero; 4 of 5 recordings are positive.
                      </div>
                      <div class="nb-divider"></div>
                      <div style="font-size:0.88rem;line-height:1.55">
                        <b>MSE is not established.</b>
                        Difference {signed(primary["mse_difference"], 4)};
                        interval {interval(*primary["mse_ci"])}.
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            _notes(
                "State the estimand before the number: a paired H20 energy difference at "
                "overlapping landmarks. The conditional interval is not animal-level "
                "precision."
            )

        elif index == 5:
            _body(
                "<p>The effect is not uniform across horizons. It reverses at H1, grows "
                "through H20, and does not establish an MSE advantage at any horizon "
                "shown.</p>"
            )
            c1, c2, c3, c4 = st.columns(4, gap="small")
            with c1:
                st.metric("H1", signed(-0.0636694611, 4))
            with c2:
                st.metric("H5", signed(0.0835331933, 4))
            with c3:
                st.metric("H10", signed(0.1471216612, 4))
            with c4:
                st.metric("H20", signed(0.2644379927, 4))

            # The interval/MSE panel already carries the caveat; no extra footnote row.
            _notes(
                "This is the slide that prevents overclaiming. If asked for a single "
                "number, give +0.2644 and then immediately give the recording interval."
            )

        elif index == 6:
            b3_cov = percent(
                snapshot.statistical["paired_effect"]["coverage_at_nominal_95"][
                    "B3_population"
                ]["95"],
                1,
            )
            b5_cov = percent(
                snapshot.statistical["paired_effect"]["coverage_at_nominal_95"][
                    "B5_common_k10"
                ]["95"],
                1,
            )
            _body(
                "<p>Calibration is marginal per dimension, not joint 10-D calibration. "
                "B5 is under-covered at the nominal 95% level, and B3 has substantially "
                "more parameters than the control.</p>"
            )
            c1, c2, c3 = st.columns(3, gap="small")
            with c1:
                st.metric("B3 marginal coverage", b3_cov)
            with c2:
                st.metric("B5 marginal coverage", b5_cov)
            with c3:
                st.metric(
                    "B3 / B5 coefficients",
                    f"{snapshot.capacity['ratio_min']:.2f}–{snapshot.capacity['ratio_max']:.2f}×",
                )
            st.markdown(
                """
                <div class="nb-panel">
                  <h4>Why this limits the claim</h4>
                  <div style="font-size:0.95rem;line-height:1.6">
                    A lower energy score can coexist with poor calibration. Capacity is
                    not matched, and the ordered 51/49 common/residual decomposition is
                    arithmetic, not a structural partition. The residual advantage is not
                    identifiable as cross-neuron information from this design.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            _notes(
                "Do not present calibration as a success. The B5 interval is too narrow "
                "for its nominal level; this is a property of the frozen forecast."
            )

        elif index == 7:
            _body(
                "<p>The result survives as a narrow, exploratory statement. It does not "
                "survive as a general claim about population models or mechanisms.</p>"
            )
            left, right = st.columns(2, gap="medium")
            with left:
                st.markdown(
                    """
                    <div class="nb-panel">
                      <h4>Can defend</h4>
                      <div style="line-height:1.65">
                        • Within these five recordings and at H20, B3 has lower mean
                        PCA-10 energy than B5-10.<br>
                        • The direction is positive in 4 of 5 recordings.<br>
                        • The pipeline is reproducible from frozen artifacts and the
                        test tails are disclosed as reused.
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with right:
                st.markdown(
                    """
                    <div class="nb-panel">
                      <h4>Cannot defend</h4>
                      <div style="line-height:1.65">
                        • Animal-level generalization or a five-animal effect.<br>
                        • A general forecasting advantage beyond H20 energy.<br>
                        • Joint calibration, a mechanistic cross-neuron effect, or a
                        validated digital twin.<br>
                        • Causal effects of virtual perturbations.
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown(
                '<div class="nb-status-strip">'
                + badge("SUPPORTED_EXPLORATORY", "prediction")
                + badge("NOT ESTABLISHED", "danger")
                + " The evidence tier is exploratory; there is no confirmed tier."
                "</div>",
                unsafe_allow_html=True,
            )
            _notes(
                "Keep the two columns visually separate. The left is the claim; the "
                "right is the boundary. Do not let the caveats become the headline."
            )

        elif index == 8:
            _body(
                "<p>These are the observations that would weaken or overturn the "
                "current reading. They are stated before the next experiment, not after "
                "it.</p>"
            )
            row1 = st.columns(2, gap="medium")
            with row1[0]:
                st.markdown(
                    """
                    <div class="nb-panel">
                      <h4>Independent animals</h4>
                      <div>The recording-level interval excludes zero and the direction
                      replicates in a new cohort.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with row1[1]:
                st.markdown(
                    """
                    <div class="nb-panel">
                      <h4>Capacity-matched residual</h4>
                      <div>A capacity-matched B3 retains an advantage over B5 on a
                      fresh test tail.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            row2 = st.columns(2, gap="medium")
            with row2[0]:
                st.markdown(
                    """
                    <div class="nb-panel">
                      <h4>Sham-controlled perturbation</h4>
                      <div>A prespecified target, dose, time window, and sham/control
                      produce a stable, target-specific response.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with row2[1]:
                st.markdown(
                    """
                    <div class="nb-panel">
                      <h4>Fresh test tail</h4>
                      <div>A pre-registered H20 test on data not used in earlier model
                      exploration reproduces the sign.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown(
                '<div class="nb-status-strip">A result that cannot fail is not a '
                'scientific result.</div>',
                unsafe_allow_html=True,
            )
            _notes(
                "Invite the supervisor to choose which falsification test is most "
                "informative. Do not imply any of these has already been run."
            )

        elif index == 9:
            _body(
                "<p>The next study should test the claim at the level where it matters: "
                "new animals, a capacity-matched baseline, and a pre-registered H20 "
                "endpoint.</p>"
            )
            left, right = st.columns(2, gap="medium")
            with left:
                st.markdown(
                    """
                    <div class="nb-panel">
                      <h4>Next experiment</h4>
                      <div style="line-height:1.65">
                        Independent-animal replication of the H20 contrast, with the
                        baseline family and cross-neuron test fixed before new scores
                        are seen.
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with right:
                st.markdown(
                    """
                    <div class="nb-panel">
                      <h4>Falsification</h4>
                      <div style="line-height:1.65">
                        The animal-level B3-minus-B1 energy interval includes zero, or
                        the sign reverses in most animals.
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown(
                '<div class="nb-status-strip">The Studio is a review instrument. It is '
                'not a validated digital twin, and it does not predict the outcome of '
                'the next experiment.</div>',
                unsafe_allow_html=True,
            )
            _notes(
                "Close by returning to the distinction between a number, a result, a "
                "claim, and a mechanism. Invite the supervisor to attack the unit, the "
                "split, and the capacity confound first."
            )

        _nav(index)
