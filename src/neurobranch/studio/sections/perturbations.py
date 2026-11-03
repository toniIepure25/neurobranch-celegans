"""Virtual perturbation lab: a model counterfactual, not a real intervention."""

from __future__ import annotations

import streamlit as st

from neurobranch.adaptive.var import WARNING as PERTURBATION_WARNING
from neurobranch.studio.activity import load_activity
from neurobranch.studio.components import (
    badge,
    caption,
    figure_caption,
    note,
    section_header,
    stat_block,
    warning,
)
from neurobranch.studio.controls import landmark_selector, session_selector
from neurobranch.studio.figures import perturbation_pair_3d, perturbation_response
from neurobranch.studio.formatting import fixed
from neurobranch.studio.metadata import ArtifactUnavailable, StudioSnapshot
from neurobranch.studio.perturbations import perturb
from neurobranch.studio.trajectories import TrajectoryStore


def render(
    snapshot: StudioSnapshot, store: TrajectoryStore, *, light: bool = False
) -> None:
    section_header(
        "06 · VIRTUAL PERTURBATIONS",
        "A model counterfactual, not an intervention.",
        "Choose one recorded fluorescence coordinate, edit it by a stated amount, and "
        "roll the fitted B3 model forward under zero innovation. This demonstrates "
        "model dependence. It does not stimulate a neuron, predict behaviour, or "
        "establish a cause.",
    )
    warning(PERTURBATION_WARNING)
    warning(
        "This is not optogenetics, electrophysiology, chemogenetics, or behaviour. The "
        "perturbed quantity is a standardized ΔF/F coordinate at one frame in a fitted "
        "linear model. The result has no biological effect size and must never be "
        "described as a real stimulation outcome."
    )
    c1, c2, c3 = st.columns([1.15, 1.45, 1.1], gap="small")
    with c1:
        session = session_selector(snapshot)
    with c2:
        landmark = landmark_selector(store, session)
    activity = load_activity(session)
    names = activity["names"]
    preferred = next((n for n in names if n.upper() == "AVAL"), None)
    preferred = preferred or next((n for n in names if any(ch.isalpha() for ch in n)), names[0])
    with c3:
        neuron = st.selectbox(
            "Edited coordinate",
            names,
            index=names.index(preferred),
            help="A predictor label in this recording. Labels are not registered one-to-one across recordings.",
        )
    magnitude = st.slider(
        "Initial-state edit (training z units)",
        -3.0,
        3.0,
        1.0,
        0.25,
        help="The same number is added to the standardized predictor at the origin frame.",
    )
    if abs(magnitude) < 1e-9:
        st.info("Choose a non-zero edit to compute a counterfactual.")
        return

    with st.spinner("Fitting the frozen B3 model at this origin and rolling both states…"):
        try:
            result = perturb(session, landmark, neuron, float(magnitude), 20)
        except (ArtifactUnavailable, KeyError, ValueError) as exc:
            st.error(f"Perturbation unavailable: {exc}")
            return

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        fig = perturbation_pair_3d(
            result.pca_baseline,
            result.pca_perturbed,
            result.pca_difference,
            result.observed_origin,
            light=light,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"perturb3d_{session}_{landmark}_{neuron}_{magnitude}")
        figure_caption(
            "PC1–PC3 projection of the B3 mean trajectory. The difference trajectory "
            "is perturbed − baseline; it is not an observed signal. The projection uses "
            "the same training-only PCA as every other Studio view."
        )
        st.plotly_chart(
            perturbation_response(
                result.future_time,
                result.neuron_response,
                result.population_response,
                light=light,
            ),
            use_container_width=True,
            key=f"perturbresponse_{session}_{landmark}_{neuron}_{magnitude}",
        )
        figure_caption(
            "Left axis: change in the edited neuron's state. Right axis: L2 norm of the "
            "full population-state difference. A large population norm does not identify "
            "a circuit path."
        )
    with right:
        stat_block(f"{magnitude:+.2f} z", f"edit to {neuron} at t={landmark}")
        st.markdown(
            f"""
            <div class="nb-panel" style="font-size:0.84rem;line-height:1.7">
              <div><b>H20 B3 vs B5 disagreement</b> {fixed(result.disagreement_h20)}</div>
              <div><b>B3 spectral radius</b> {fixed(result.b3_radius)}</div>
              <div><b>B5 factor spectral radius</b> {fixed(result.b5_radius)}</div>
              <div><b>Training-marginal OOD</b> {"yes" if result.ood else "no"}</div>
              <div><b>OOD fraction</b> {result.ood_fraction:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if result.ood:
            warning(
                "The edited coordinate lies outside the 0.5–99.5% training marginal for "
                "this neuron. The model is being evaluated outside its training support; "
                "the trajectory is a numerical extrapolation, not a credible biological "
                "counterfactual."
            )
        note(
            "Sensitivity to model family is shown by the B3–B5 disagreement. Agreement "
            "would not validate the perturbation; disagreement shows that the "
            "counterfactual is model-dependent."
        )
        st.markdown("---")
        st.markdown("#### What this demonstration does not do")
        st.markdown(
            "- It does not rank neurons or targets.\n"
            "- It does not estimate a biological effect or a therapeutic effect.\n"
            "- It does not transfer the label to another recording; there is no cross-session cell registration.\n"
            "- It does not interpret a VAR coefficient as a synapse or a directed connection.\n"
            "- It does not establish a validated digital twin."
        )
        st.markdown(
            badge("MODEL COUNTERFACTUAL", "prediction")
            + badge("NO CAUSAL CLAIM", "danger"),
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### Representation boundary")
    st.markdown(
        "The perturbation is applied in the native predictor coordinates of this "
        "recording. Named labels are cell-type labels; numeric ROIs are unregistered "
        "segments. The dataset has no cross-session one-to-one registration, so a "
        "perturbation of `AVAL` in recording 0 is not the same identified cell as "
        "`AVAL` in another recording. A request for a cell absent from this recording "
        "has no valid mapping and is refused by the selector."
    )
    if st.button("Open the recorded trace for this coordinate", use_container_width=False):
        st.session_state["studio_section"] = "activity"
        st.session_state["studio_session"] = int(session)
        st.session_state["studio_landmark"] = int(landmark)
        st.rerun()
    caption(
        "Implementation: fit_window from the frozen model module, ridge 1.0, window 500, "
        "zero innovation; "
        "B3 and B5 fitted at the same origin. The perturbation is reproducible for the "
        "stated origin, model, coordinate, and magnitude."
    )
