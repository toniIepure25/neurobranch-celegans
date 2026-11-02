"""Read-only dashboard for neural uncertainty and calibration."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/simulator/uncertainty_calibration"
WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED."


def _csv(name: str) -> pd.DataFrame:
    path = OUT / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _json(name: str) -> dict:
    path = OUT / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _overview() -> None:
    gates = _json("gates.json")
    summary = _csv("score_summary.csv")
    if summary.empty:
        st.info("Uncertainty-calibration artifacts are not generated in this checkout.")
        return
    p2 = summary[summary.model == "P2_independent_ar_lead1"]
    p6 = summary[summary.model == "P6_adaptive_block"]
    p6l = summary[summary.model == "P6_adaptive_lead1"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Primary H20 CRPS", f"{p2.crps_h20.iloc[0]:.3f}" if len(p2) else "n/a")
    c2.metric("Primary 95% coverage", f"{p2.coverage_95.iloc[0]:.3f}" if len(p2) else "n/a")
    c3.metric("Historical P6 energy", f"{p6.energy_h20.iloc[0]:.3f}" if len(p6) else "n/a")
    c4.metric("Lead-1 P6 energy", f"{p6l.energy_h20.iloc[0]:.3f}" if len(p6l) else "n/a")
    st.error(
        "The prespecified independent AR model improves marginal CRPS but under-covers at 95%, "
        "and its joint trajectory score is worse than persistence. "
        "PROBABILISTIC CALIBRATION NOT ESTABLISHED."
    )
    st.dataframe(
        summary[
            ["model", "energy_h20", "crps_h20", "coverage_50", "coverage_80", "coverage_95", "width_95", "ood_h20"]
        ].sort_values("energy_h20"),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "175 overlapping test landmarks from five repeated sessions. Landmarks are not independent animals. "
        "Lower energy and CRPS are better."
    )
    st.json(
        {
            "gates": {k: v for k, v in gates.items() if k[0].isupper() and "_" in k},
            "stop_label": gates.get("stop_label"),
            "warning": WARNING,
        }
    )


def _intervals() -> None:
    bands = _csv("example_bands.csv")
    obs = _csv("example_observed.csv")
    if bands.empty:
        st.info("No executed example ensembles in this checkout.")
        return
    models = sorted(bands.model.unique())
    model = st.selectbox("Model", models, index=0)
    series = st.selectbox("Series", sorted(bands.series.unique()), index=0)
    g = bands[(bands.model == model) & (bands.series == series)].sort_values("step")
    o = obs[obs.series == series].sort_values("step")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.step, y=g.q05, line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=g.step, y=g.q95, fill="tonexty", fillcolor="rgba(47,111,159,0.22)", line={"width": 0}, name="5–95%"))
    fig.add_trace(go.Scatter(x=g.step, y=g.q50, name="predictive median", line={"color": "#2f6f9f"}))
    fig.add_trace(go.Scatter(x=o.step, y=o.value, name="observed continuation", line={"color": "black", "width": 2}))
    fig.update_layout(xaxis_title="step after landmark", yaxis_title="standardized activity", height=420, margin={"l": 10, "r": 10, "t": 30, "b": 10})
    st.plotly_chart(fig, width="stretch")
    st.caption("Executed ensemble from the held-out example landmark; no decorative trajectories.")


def _variance() -> None:
    v = _csv("variance_summary.csv")
    c = _csv("coverage_by_horizon.csv")
    if v.empty:
        st.info("No variance artifacts.")
        return
    models = ["P2_independent_ar_lead1", "P0_persistence", "P6_adaptive_block", "P6_adaptive_lead1"]
    view = v[v.model.isin(models)]
    fig = px.line(view, x="horizon", y="within_var_mean", color="model", markers=True, log_x=True, log_y=True, title="Predictive variance growth")
    st.plotly_chart(fig, width="stretch")
    fig2 = px.line(view, x="horizon", y="spread_error_ratio", color="model", markers=True, log_x=True, title="Spread / squared error")
    fig2.add_hline(y=1.0, line_dash="dash")
    st.plotly_chart(fig2, width="stretch")
    if not c.empty:
        cov = c[c.model.isin(models)].groupby(["model", "horizon"], as_index=False)[["coverage_50", "coverage_80", "coverage_95"]].mean()
        fig3 = px.line(cov.melt(id_vars=["model", "horizon"], var_name="level", value_name="coverage"), x="horizon", y="coverage", color="model", line_dash="level", markers=True, log_x=True, title="Marginal interval coverage")
        for level in (0.5, 0.8, 0.95):
            fig3.add_hline(y=level, line_dash="dot", opacity=0.3)
        st.plotly_chart(fig3, width="stretch")


def _innovation() -> None:
    lead = _csv("lead_profile.csv")
    summary = _csv("innovation_summary.csv")
    if lead.empty:
        st.info("No innovation artifacts.")
        return
    fig = px.line(lead, x="lead", y="trace_cov", color="model", markers=True, title="Residual covariance by prequential lead")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "The historical block bootstrap admitted leads 1–50 from a fixed fit. Those are multi-step forecast errors, "
        "not one-step innovations; the variance rises with lead."
    )
    if not summary.empty:
        st.dataframe(summary, width="stretch", hide_index=True)
    st.warning(WARNING)


def _calibration() -> None:
    cal = _csv("calibration_h20.csv")
    pit = _csv("calibration_curves.csv")
    session = _csv("per_session_calibration.csv")
    if cal.empty:
        st.info("No calibration artifacts.")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(cal, width="stretch", hide_index=True)
    with c2:
        if not pit.empty:
            fig = px.line(pit, x="bin", y="density", color="model", markers=True, title="PIT histogram, H20")
            fig.add_hline(y=0.1, line_dash="dash")
            st.plotly_chart(fig, width="stretch")
    if not session.empty:
        st.dataframe(session, width="stretch", hide_index=True)
    st.caption("Coverage is averaged over neurons; per-session rows keep repeated sessions visible.")


def _trajectory() -> None:
    traj = _csv("trajectory_scores.csv")
    cov = _csv("covariance_h20.csv")
    if traj.empty:
        st.info("No trajectory artifacts.")
        return
    fig = px.box(traj, x="model", y="trajectory_energy_h20", title="Joint H20 trajectory energy (lower is better)")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(traj.groupby("model").trajectory_energy_h20.agg(["mean", "median", "count"]).reset_index(), width="stretch", hide_index=True)
    if not cov.empty:
        st.dataframe(cov[cov.worm_id == "mean"], width="stretch", hide_index=True)
    st.info(
        "A model can have calibrated marginals but a poor joint trajectory score. "
        "The independent AR model has the better marginal CRPS but worse H20 trajectory energy than persistence."
    )


def _gates() -> None:
    gates = _json("gates.json")
    repro = _json("reproduction.json")
    if not gates:
        st.info("No gate record.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {"gate": "A reproduction", "status": gates.get("A_reproduction"), "evidence": f"max relative diff {repro.get('max_relative_difference', float('nan')):.4f}"},
                {"gate": "B variance diagnosis", "status": gates.get("B_variance_failure_diagnosed"), "evidence": "lead pooling + operator amplification"},
                {"gate": "C innovation validity", "status": gates.get("C_innovation_assumptions"), "evidence": "primary residual ACF and mean predictability"},
                {"gate": "D proper score", "status": gates.get("D_proper_score_improvement"), "evidence": "CRPS versus P0 and P1"},
                {"gate": "E marginal calibration", "status": gates.get("E_marginal_calibration"), "evidence": "50/80/95 coverage, PIT, standardized errors"},
                {"gate": "F trajectory fidelity", "status": gates.get("F_joint_trajectory_fidelity"), "evidence": "joint energy, covariance, autocorrelation"},
                {"gate": "G generative", "status": gates.get("G_generative_credibility"), "evidence": "requires D, E and F"},
                {"gate": "H intervention", "status": gates.get("H_biological_intervention"), "evidence": "no interventions"},
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.error(gates.get("stop_label", ""))
    st.warning(WARNING)


def render() -> None:
    st.header("NEURAL UNCERTAINTY & CALIBRATION")
    st.caption("Diagnostic and corrective experiment on probabilistic forecasts of future C. elegans calcium activity.")
    tabs = st.tabs(["Overview", "Forecast intervals", "Variance growth", "Innovation validity", "Calibration", "Trajectory fidelity", "Gates"])
    with tabs[0]:
        _overview()
    with tabs[1]:
        _intervals()
    with tabs[2]:
        _variance()
    with tabs[3]:
        _innovation()
    with tabs[4]:
        _calibration()
    with tabs[5]:
        _trajectory()
    with tabs[6]:
        _gates()
