"""Read-only diagnostic dashboard for the generative-failure study."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "reports/simulator/failure_diagnostics"
WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED."


def _read_csv(relative: str) -> pd.DataFrame:
    path = DIAG / relative
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(relative: str) -> dict:
    path = DIAG / relative
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _forecast_tab() -> None:
    summary = _read_csv("forecastability/forecast_summary.csv")
    if summary.empty:
        st.info("Forecastability artifacts are not generated in this checkout.")
        return
    agg = _read_csv("forecastability/forecast_aggregate.csv")
    st.markdown(
        "**Skill = 1 − MSE_model / MSE_persistence** on identical chronological test starts. "
        "Positive is better than holding the last observation. Intervals are moving-block descriptive intervals."
    )
    fig = px.line(
        agg[agg.model != "persistence"],
        x="horizon",
        y="mean_skill",
        color="model",
        markers=True,
        labels={"horizon": "horizon (frames)", "mean_skill": "mean skill vs persistence"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(agg, width="stretch", hide_index=True)
    pop = _read_csv("forecastability/population_vs_independent.csv")
    if not pop.empty:
        st.markdown("**Population VAR versus independent per-neuron AR (same starts)**")
        fig2 = px.line(
            pop,
            x="horizon",
            y="population_minus_independent_mse",
            color="worm_id",
            markers=True,
            labels={"population_minus_independent_mse": "MSE(population) − MSE(independent)"},
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig2, width="stretch")
        st.caption("Negative values mean the population model has lower MSE than independent neurons; both may still lose to persistence.")
    neurons = _read_csv("forecastability/forecast_by_neuron.csv")
    if not neurons.empty:
        pick = neurons[(neurons.model == "population_var") & (neurons.horizon == 20)]
        st.markdown("**Per-neuron skill at 20 frames**")
        st.plotly_chart(
            px.histogram(pick, x="skill_vs_persistence", nbins=50, labels={"skill_vs_persistence": "skill"}),
            width="stretch",
        )
    period = _read_csv("forecastability/period_stability.csv")
    if not period.empty:
        st.markdown("**Stability across the three test thirds**")
        st.dataframe(period, width="stretch", hide_index=True)


def _observation_tab() -> None:
    synth = _read_csv("observation/synthetic_forecast.csv")
    real = _read_csv("observation/real_observation_aggregate.csv")
    if synth.empty and real.empty:
        st.info("Observation-model artifacts are not generated in this checkout.")
        return
    if not synth.empty:
        st.markdown("**Synthetic known-truth controls (not biological validation)**")
        st.dataframe(synth, width="stretch", hide_index=True)
    if not real.empty:
        st.markdown("**Real held-out data: observation-aware models**")
        st.dataframe(real, width="stretch", hide_index=True)
        st.caption(
            "The second-order and causal-calcium models were fitted only on training frames. "
            "A negative result is retained; it is not hidden by selecting a different test start."
        )


def _identifiability_tab() -> None:
    path = DIAG / "identifiability/identifiability_summary.json"
    if not path.exists():
        st.info("Identifiability artifacts are not generated in this checkout.")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    st.markdown(
        "Latent coordinates are defined only up to rotation/sign/permutation. "
        "Eigenvalues and subspace angles are the reported invariants; no anatomical connectivity is claimed."
    )
    for item in payload.get("results", []):
        st.markdown(f"**Worm {item['worm_id']}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Participation ratio", f"{item['pca']['participation_ratio']:.2f}")
        c2.metric("Components for 90% variance", item["pca"]["n_components_90pct"])
        c3.metric("Half-split subspace angle", f"{item['subspace']['half_subspace']['mean_angle_deg']:.1f}°")
        st.json(item["operator"], expanded=False)


def _oxygen_tab() -> None:
    payload = _read_json("oxygen/oxygen_summary.json")
    if not payload:
        st.info("Oxygen-response artifacts are not generated in this checkout.")
        return
    st.warning(
        "Oxygen is a controlled environmental sensory input, not targeted neuronal stimulation. "
        "The 30 s schedule is repeated across sessions; event and elapsed time are not randomized."
    )
    st.json(payload, expanded=False)
    window = _read_csv("oxygen/oxygen_window_aggregate.csv")
    if not window.empty:
        st.markdown("**Baseline-corrected event-window controls**")
        st.dataframe(window, width="stretch", hide_index=True)
    reliability = _read_csv("oxygen/oxygen_event_reliability.csv")
    if not reliability.empty:
        st.markdown("**Across-event response-vector reproducibility**")
        st.dataframe(reliability, width="stretch", hide_index=True)


def _perturbation_tab() -> None:
    payload = _read_json("perturbation/perturbation_summary.json")
    if not payload:
        st.info("Perturbation-consistency artifacts are not generated in this checkout.")
        return
    st.error(WARNING)
    st.json(payload, expanded=False)
    pairs = _read_csv("perturbation/pairwise_consistency.csv")
    if not pairs.empty:
        st.markdown("**Cross-model response agreement**")
        st.dataframe(pairs, width="stretch", hide_index=True)


def render() -> None:
    st.header("Generative-failure diagnostics")
    st.caption(
        "Read-only summary of why the existing simulators lose to persistence. "
        "No test-selected model and no biological causal claim."
    )
    tabs = st.tabs(
        ["Forecastability", "Observation model", "Identifiability", "Oxygen response", "Perturbation"]
    )
    with tabs[0]:
        _forecast_tab()
    with tabs[1]:
        _observation_tab()
    with tabs[2]:
        _identifiability_tab()
    with tabs[3]:
        _oxygen_tab()
    with tabs[4]:
        _perturbation_tab()
