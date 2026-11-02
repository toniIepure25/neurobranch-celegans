"""Read-only dashboard for joint neural dynamics and predictive calibration."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/simulator/joint_dynamics"
WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED."


def _csv(name: str) -> pd.DataFrame:
    path = OUT / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _json(name: str) -> dict:
    path = OUT / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _overview() -> None:
    s = _csv("score_summary.csv")
    gates = _json("gates.json")
    if s.empty:
        st.info("Joint-dynamics artifacts are not generated in this checkout.")
        return
    idx = s.set_index("model")
    p = idx.loc["J5_population_lead1_boot"] if "J5_population_lead1_boot" in idx.index else None
    i = idx.loc["J1_independent_diag"] if "J1_independent_diag" in idx.index else None
    z = idx.loc["J0_persistence"] if "J0_persistence" in idx.index else None
    c1, c2, c3, c4 = st.columns(4)
    if p is not None:
        c1.metric("Population H20 energy", f"{p.energy_h20:.3f}")
        c2.metric("Population trajectory energy", f"{p.trajectory_energy_h20:.3f}")
        c3.metric("Population 50% coverage", f"{p.coverage_h20_full_50:.3f}")
        c4.metric("Neural covariance error", f"{p.neural_cov_error:.3f}")
    st.error(gates.get("stop_label", "PROBABILISTIC CALIBRATION NOT ESTABLISHED"))
    st.markdown(
        "The population model improves the multivariate and trajectory energy scores over independent AR, "
        "but it does not calibrate at 50% and its neural covariance error remains above the frozen gate. "
        "The improvement is carried by conditional-mean dynamics, not by the full innovation covariance."
    )
    view = s[
        [
            "model",
            "energy_h20",
            "trajectory_energy_h20",
            "crps_h20_full",
            "coverage_h20_full_50",
            "coverage_h20_full_80",
            "coverage_h20_full_95",
            "neural_cov_error",
            "autocorr_error",
        ]
    ].sort_values("energy_h20")
    st.dataframe(view, width="stretch", hide_index=True)
    if p is not None and i is not None and z is not None:
        st.caption(
            f"Population vs independent AR: {i.energy_h20 - p.energy_h20:+.3f} H20 energy. "
            f"Population vs persistence: {z.energy_h20 - p.energy_h20:+.3f}. "
            "Positive values mean the population model is better."
        )
    st.caption("175 overlapping landmarks from five repeated sessions; not independent animals.")
    st.warning(WARNING)


def _trajectories() -> None:
    obs = _csv("example_observed.csv")
    bands = _csv("example_bands.csv")
    paths = _csv("example_trajectories.csv")
    if obs.empty or bands.empty:
        st.info("No executed example trajectories.")
        return
    labels = _json("model_cards.json")
    models = list(labels)
    model = st.selectbox("Model", models, index=models.index("J5_population_lead1_boot") if "J5_population_lead1_boot" in models else 0)
    fig = go.Figure()
    p = paths[paths.model == model]
    for _member, g in p.groupby("member"):
        fig.add_trace(go.Scatter(x=g.step, y=g.pc1, mode="lines", line={"width": 0.6, "color": "rgba(212,160,23,0.18)"}, showlegend=False, hoverinfo="skip"))
    b = bands[bands.model == model].sort_values("step")
    fig.add_trace(go.Scatter(x=b.step, y=b.q05, line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=b.step, y=b.q95, fill="tonexty", fillcolor="rgba(212,160,23,0.25)", line={"width": 0}, name="5-95%"))
    fig.add_trace(go.Scatter(x=b.step, y=b.q50, name="ensemble median", line={"color": "#8a5a00"}))
    fig.add_trace(go.Scatter(x=obs.step, y=obs.observed_pc1, name="observed PC1", line={"color": "black", "width": 2}))
    fig.update_layout(xaxis_title="step after landmark", yaxis_title="PC1 activity (training-scaled)", height=430, margin={"l": 10, "r": 10, "t": 30, "b": 10})
    st.plotly_chart(fig, width="stretch")
    st.caption("Held-out session 0, landmark 2690, chosen as the first prespecified test landmark; not score-selected.")


def _scores() -> None:
    s = _csv("score_summary.csv")
    f = _csv("factorial_decomposition.csv")
    if s.empty:
        st.info("No score artifacts.")
        return
    long = s.melt(id_vars=["model"], value_vars=["energy_h20", "trajectory_energy_h20"], var_name="score", value_name="value")
    fig = px.bar(long, x="model", y="value", color="score", barmode="group", title="H20 proper scores (lower better)")
    fig.update_xaxes(tickangle=18)
    st.plotly_chart(fig, width="stretch")
    if not f.empty:
        st.dataframe(f, width="stretch", hide_index=True)
        st.caption("Mean main effect > 0 means the population mean helps; covariance main effect < 0 means full covariance hurts.")


def _calibration() -> None:
    c = _csv("coverage_by_horizon.csv")
    cal = _csv("calibration_h20.csv")
    pit = _csv("calibration_curves.csv")
    if c.empty:
        st.info("No calibration artifacts.")
        return
    models = ["J0_persistence", "J1_independent_diag", "J2_population_diag", "J5_population_lead1_boot"]
    view = c[c.model.isin(models)]
    fig = px.line(view, x="horizon", y=["coverage_50", "coverage_80", "coverage_95"], color="model", markers=True, log_x=True, title="Coverage by horizon, 10-D PCA")
    for level in (0.5, 0.8, 0.95):
        fig.add_hline(y=level, line_dash="dot", opacity=0.3)
    st.plotly_chart(fig, width="stretch")
    if not cal.empty:
        st.dataframe(cal, width="stretch", hide_index=True)
    if not pit.empty:
        fig2 = px.line(pit, x="bin", y="density", color="model", markers=True, title="PIT, H20")
        fig2.add_hline(y=0.1, line_dash="dash")
        st.plotly_chart(fig2, width="stretch")
    st.caption("Full-neuron coverage is the reproduction endpoint; PCA coverage is the joint evaluation representation.")


def _covariance() -> None:
    cov = _csv("covariance_h20.csv")
    acf = _csv("autocorr_profile.csv")
    if cov.empty:
        st.info("No covariance artifacts.")
        return
    models = ["observed", "J1_independent_diag", "J5_population_lead1_boot", "J2_population_diag"]
    model = st.selectbox("Covariance model", [m for m in models if m in set(cov.model)], index=0)
    g = cov[cov.model == model]
    mat = g.pivot(index="row", columns="col", values="correlation")
    fig = px.imshow(mat, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Pooled PC correlation at H20", aspect="equal")
    st.plotly_chart(fig, width="stretch")
    if not acf.empty:
        fig2 = px.line(acf, x="lag", y="autocorrelation", color="model", markers=True, title="Temporal autocorrelation")
        st.plotly_chart(fig2, width="stretch")


def _uncertainty() -> None:
    d4 = _csv("d4_summary.csv")
    d4v = _csv("d4_validation_summary.csv")
    par = _csv("parameter_summary.csv")
    synth = _csv("parameter_synthetic_audit.csv")
    c1, c2 = st.columns(2)
    with c1:
        if not d4.empty:
            st.dataframe(d4, width="stretch", hide_index=True)
        if not d4v.empty:
            fig = px.line(d4v, x="scale", y=["coverage_50", "coverage_80", "coverage_95"], markers=True, title="D4 validation calibration")
            for level in (0.5, 0.8, 0.95):
                fig.add_hline(y=level, line_dash="dot", opacity=0.3)
            st.plotly_chart(fig, width="stretch")
    with c2:
        if not par.empty:
            st.dataframe(par, width="stretch", hide_index=True)
        if not synth.empty:
            st.dataframe(synth, width="stretch", hide_index=True)
    st.caption("D4 Student-t scale was fitted only on validation. Parameter draws use causal moving-block bootstrap; neither is a test-tuned result.")


def _gates() -> None:
    gates = _json("gates.json")
    if not gates:
        st.info("No gate record.")
        return
    rows = []
    for key, value in gates.items():
        if not isinstance(value, dict):
            continue
        rows.append({"gate": key, "status": value.get("status", value.get("stop_label", "n/a")), "detail": json.dumps({k: v for k, v in value.items() if k != "status"})[:240]})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.error(gates.get("stop_label", ""))
    st.warning(WARNING)


def render() -> None:
    st.header("JOINT NEURAL DYNAMICS & CALIBRATION")
    st.caption("Follow-up to the uncertainty audit: does population structure add information beyond independent-neuron forecasting?")
    tabs = st.tabs(["Overview", "Joint trajectories", "Scores", "Calibration", "Covariance & autocorrelation", "Uncertainty", "Gates"])
    with tabs[0]:
        _overview()
    with tabs[1]:
        _trajectories()
    with tabs[2]:
        _scores()
    with tabs[3]:
        _calibration()
    with tabs[4]:
        _covariance()
    with tabs[5]:
        _uncertainty()
    with tabs[6]:
        _gates()
