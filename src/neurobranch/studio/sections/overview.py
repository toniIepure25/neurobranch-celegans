"""Overview: the claim, its unit, and its boundary."""

from __future__ import annotations

import streamlit as st

from neurobranch.studio.components import (
    badge,
    caption,
    evidence_badge,
    hero,
    note,
    panel,
    section_header,
    stat_block,
    warning,
)
from neurobranch.studio.formatting import bytes_human, fixed, interval, percent, signed
from neurobranch.studio.metadata import StudioSnapshot


def render(snapshot: StudioSnapshot, *, light: bool = False) -> None:
    primary = snapshot.primary
    identity = snapshot.dataset_identity
    archive = identity["archive"]
    hero(
        "NeuroBranch Scientific Studio · Kato 2015 WT_NoStim",
        "A bounded view of a real neural system.",
        "Five whole-brain calcium recordings, one frozen train-only projection, and a "
        "forecast comparison that is useful precisely because its limits are visible. "
        "This is a methodological reanalysis, not a digital twin and not a new "
        "biological experiment.",
        [
            ("EXPLORATORY", "prediction"),
            ("5 recordings · 15,662 frames", "observed"),
            ("1 laboratory / preparation", "neutral"),
            ("H20 PCA-10 energy", "control"),
            ("test tail reused", "danger"),
        ],
    )

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        section_header(
            "01 · THE QUESTION",
            "Does a population model retain predictive structure beyond a "
            "ten-factor common-state control?",
            "The prespecified contrast is B5 − B3: positive values mean the full "
            "population VAR has lower H20 energy than the ten-factor model. Energy is "
            "a proper score for a probabilistic forecast; it is not a biological "
            "effect size.",
        )
        st.markdown(
            f"""
            <div class="nb-panel">
              <h4>The result in one sentence</h4>
              <div style="font-size:1.1rem;line-height:1.55;color:inherit">
                Across <b>175 overlapping test landmarks</b> in <b>five recordings</b>,
                B3 had lower H20 energy than B5-10 by a weighted mean of
                <b style="color:inherit">{signed(primary["difference"])}</b>
                ({percent(primary["relative_to_b3"])} of the B3 mean).
                Four of five recordings were positive, but the recording-level
                interval includes zero and the sign test is not significant.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cols = st.columns(3, gap="small")
        with cols[0]:
            stat_block(fixed(primary["b3_mean"]), "B3 mean H20 energy")
        with cols[1]:
            stat_block(fixed(primary["b5_mean"]), "B5-10 mean H20 energy")
        with cols[2]:
            stat_block(signed(primary["difference"]), "B5 − B3 paired mean")
        st.markdown(
            f"""
            <div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin:0.35rem 0 1rem">
              {badge("conditional block-5 " + interval(primary["conditional_ci"]["ci_low"], primary["conditional_ci"]["ci_high"]), "prediction")}
              {badge("recording resampling " + interval(primary["recording_ci"]["ci_low"], primary["recording_ci"]["ci_high"]), "control")}
              {badge("session t " + interval(*primary["session_t_ci"]), "neutral")}
            </div>
            """,
            unsafe_allow_html=True,
        )
        note(
            "The conditional interval describes landmarks within these recordings. "
            "The recording-resampling interval is the appropriate check for extending "
            "to other animals from this preparation, and with five units it is weak. "
            "Neither interval is an animal-level generalization guarantee."
        )

    with right:
        st.markdown(
            f"""
            <div class="nb-panel">
              <h4>Evidence status</h4>
              <div style="font-size:1.9rem;font-weight:720;letter-spacing:-0.03em">EXPLORATORY</div>
              <div style="color:inherit;opacity:0.75;font-size:0.82rem;line-height:1.5">
                Ready for methodological review.<br>
                Not ready for confirmatory or biological claims.
              </div>
              <div class="nb-divider"></div>
              <div style="font-size:0.82rem;line-height:1.7">
                <div><b>H1 reverses</b> · B3 is worse than B5 at the shortest horizon.</div>
                <div><b>MSE is null</b> · mean difference {signed(primary["mse_difference"])}, interval crosses zero.</div>
                <div><b>Calibration is not established</b> · B5 95% marginal coverage {percent(snapshot.statistical["paired_effect"]["coverage_at_nominal_95"]["B5_common_k10"]["95"], 1)}.</div>
                <div><b>Capacity differs</b> · B3 has 7.96–9.95× the nominal coefficients.</div>
                <div><b>Unit is not 175 animals</b> · 175 overlapping landmarks from five recordings.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open the 8-minute supervisor tour", use_container_width=True):
            st.session_state["studio_present"] = True
            st.session_state["studio_present_index"] = 0
            st.rerun()
        st.caption("No claim is promoted by this interface. Use the Evidence section to inspect every boundary.")

    st.markdown("---")
    section_header(
        "02 · WHAT IS AND IS NOT ESTABLISHED",
        "Three categories, kept separate.",
        "The studio uses the Master Review claim vocabulary: SUPPORTED_EXPLORATORY, "
        "LIMITED, UNSUPPORTED, INVALIDATED, and NOT_EVALUABLE. There is no CONFIRMED tier.",
    )
    a, b, c = st.columns(3, gap="medium")
    with a:
        panel(
            "Supported exploratory",
            "Within these five recordings and at H20, B3 has lower mean PCA-10 energy "
            "than B5-10 in the paired landmark sample. The direction is horizon-dependent "
            "and not uniform across recordings.",
        )
        st.markdown(evidence_badge("SUPPORTED_EXPLORATORY"), unsafe_allow_html=True)
    with b:
        panel(
            "Limited",
            "The recording-level interval is compatible with zero. The five recordings "
            "are one laboratory/preparation, not an independent cohort. The fitted "
            "state is a calcium-imaging transition model, not a synaptic wiring diagram.",
        )
        st.markdown(evidence_badge("LIMITED"), unsafe_allow_html=True)
    with c:
        panel(
            "Not established",
            "A general forecasting advantage, a mechanistic cross-neuron effect, joint "
            "calibration, animal-level generalization, a validated digital twin, and "
            "causal effects of virtual perturbations.",
        )
        st.markdown(evidence_badge("UNSUPPORTED"), unsafe_allow_html=True)

    st.markdown("---")
    section_header(
        "03 · PROVENANCE",
        "The archive is verified; the documentation is not the source of truth.",
        "The raw file is checked against the OSF object by size and SHA256. "
        "The historical documentation contains the 17 corrections surfaced in the "
        "Research Notebook; frozen artifacts are preserved rather than rewritten.",
    )
    with st.expander("Dataset identity and field inventory", expanded=False):
        cols = st.columns(4)
        cols[0].metric("Archive size", bytes_human(archive["actual_size_bytes"]))
        cols[1].metric("Frames", "15,662")
        cols[2].metric("Predictors", "592")
        cols[3].metric("Named occurrences", "192")
        st.code(
            f"path      {archive['path']}\n"
            f"sha256    {archive['actual_sha256']}\n"
            f"OSF       {archive['osf_url']}\n"
            f"identity  {identity['identity']['answer']}",
            language="text",
        )
        st.markdown(
            f"**Recommended wording.** {identity['identity']['recommended_wording']}"
        )
        warning(
            "The OSF archive exposes no explicit animal key. The five distinct "
            "acquisition preparations are consistent with the publication's five "
            "animals; the one-to-one mapping is inferred, not documented by a primary "
            "key. They are still one laboratory/preparation and not an independent cohort."
        )
        st.markdown(
            "The `stateParams` field is unused by the model and does not align with all "
            "trace rows (109/109, 135/107, 131/131, 125/126, 129/129). Behaviour labels "
            "are derived using command/interneuron activity and are not independent "
            "ground truth. Numeric ROIs are not registered cell identities."
        )
    caption(
        "Data source: Kato et al. (2015), Cell 163:656–669, DOI 10.1016/j.cell.2015.09.034. "
        "Project artifacts: reports/master_review/dataset_identity_audit.json and "
        "reports/simulator/kato_common_state/model_per_landmark.csv."
    )
