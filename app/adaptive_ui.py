"""Read-only dashboard for the adaptive-dynamics stability experiment."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "reports/simulator/adaptive_dynamics"
WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED."


def _read_csv(name: str) -> pd.DataFrame:
    path = DIAG / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(name: str) -> dict:
    path = DIAG / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _overview() -> None:
    paired = _read_csv("model_paired_overall.csv")
    horizon = _read_csv("model_horizon_curves.csv")
    if paired.empty:
        st.info("Adaptive-dynamics artifacts are not generated in this checkout.")
        return
    st.markdown(
        "**Paired H20 improvement versus the validated frozen VAR.** Positive means the adaptive "
        "model has lower free-running error on the same landmarks. Intervals are moving-block "
        "bootstrap intervals over landmarks, not animal-level confidence intervals."
    )
    st.dataframe(paired, width="stretch", hide_index=True)
    if not horizon.empty:
        fig = px.line(
            horizon,
            x="horizon",
            y="mean_mse",
            color="model",
            markers=True,
            log_y=True,
            labels={"horizon": "horizon (frames)", "mean_mse": "mean free-running MSE (log)"},
        )
        st.plotly_chart(fig, width="stretch")
    st.warning(
        "Primary gate C passes in aggregate, but Gate D (distributional fidelity) fails: forecasts "
        "are strongly under-dispersed. Do not present the lower MSE as a generative reproduction."
    )


def _nonstationarity() -> None:
    windows = _read_csv("nonstationarity_windows.csv")
    summary = _read_csv("nonstationarity_summary.csv")
    null = _read_csv("nonstationarity_null.csv")
    if windows.empty:
        st.info("Nonstationarity artifacts are not generated in this checkout.")
        return
    fig = px.line(
        windows,
        x="window_center",
        y="reference_angle_mean_deg",
        color="worm_id",
        labels={"window_center": "window centre (frames)", "reference_angle_mean_deg": "mean angle (deg)"},
        title="Sliding-window angle to train-only reference subspace",
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(summary, width="stretch", hide_index=True)
    if not null.empty:
        st.caption(
            "Stationary surrogate 95th percentiles are in the summary table. Large absolute angles "
            "are not evidence of a biological state transition."
        )
    oxygen = _read_csv("oxygen_adaptive_summary.csv")
    if not oxygen.empty:
        st.markdown("**WT_Stim exploratory stratification by oxygen-transition proximity**")
        st.dataframe(oxygen, width="stretch", hide_index=True)
        st.caption(
            "Repeated sessions, fixed 30 s schedule. Transition proximity remains confounded with "
            "elapsed time; oxygen is an environmental input, not targeted stimulation."
        )


def _calcium() -> None:
    summary = _read_csv("calcium_summary.csv")
    diagnostics = _read_json("calcium_diagnostics.json")
    if summary.empty:
        st.info("Calcium stability artifacts are not generated in this checkout.")
        return
    st.markdown(
        "**Original feedback rollout versus corrected latent-drive recursion.** The corrected model "
        "is finite but underfits; stability alone is not predictive credibility."
    )
    h20 = summary[summary.horizon == 20]
    fig = px.bar(
        h20,
        x="case",
        y="mean_mse",
        color="model",
        barmode="group",
        log_y=True,
        labels={"mean_mse": "H20 mean MSE (log)"},
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(summary, width="stretch", hide_index=True)
    if diagnostics:
        st.json(diagnostics.get("version", "corrected"), expanded=False)


def _models() -> None:
    session = _read_csv("model_per_session_h20.csv")
    neurons = _read_csv("model_per_neuron_h20.csv")
    updates = _read_csv("model_update_summary.csv")
    if session.empty:
        st.info("Adaptive model artifacts are not generated in this checkout.")
        return
    st.markdown("**H20 error by session**")
    st.dataframe(session, width="stretch", hide_index=True)
    if not neurons.empty:
        pick = neurons[neurons.model.isin(["frozen_var", "rolling_w500", "ewma_h1000"])]
        fig = px.histogram(
            pick,
            x="skill_vs_persistence_h20",
            color="model",
            nbins=60,
            range_x=[-2, 1],
            barmode="overlay",
            opacity=0.55,
            labels={"skill_vs_persistence_h20": "per-neuron H20 skill"},
        )
        st.plotly_chart(fig, width="stretch")
        st.caption("Skill is clipped to the displayed range only for visualization. Extreme values are retained in the data.")
    if not updates.empty:
        st.markdown("**Update cost and realized spectral radius**")
        st.dataframe(updates, width="stretch", hide_index=True)


def _perturbation() -> None:
    payload = _read_json("perturbation_metadata.json")
    sensitivity = _read_csv("perturbation_sensitivity.csv")
    if not payload:
        st.info("Perturbation sensitivity artifacts are not generated in this checkout.")
        return
    st.error(WARNING)
    st.json(payload, expanded=False)
    if not sensitivity.empty:
        st.markdown("**Frozen-versus-adaptive impulse agreement (descriptive only)**")
        st.dataframe(sensitivity, width="stretch", hide_index=True)
        st.caption("No neuron is ranked and no causal intervention is implied.")


def render() -> None:
    st.header("Adaptive dynamics and stability")
    st.caption(
        "Read-only diagnostics for frozen versus causally adaptive dynamics on Kato WT_NoStim. "
        "Repeated sessions are not independent animals; no biological intervention is claimed."
    )
    tabs = st.tabs(["Overview", "Nonstationarity", "Calcium stability", "Adaptive models", "Perturbation"])
    with tabs[0]:
        _overview()
    with tabs[1]:
        _nonstationarity()
    with tabs[2]:
        _calcium()
    with tabs[3]:
        _models()
    with tabs[4]:
        _perturbation()
