"""Read-only dashboard for the probabilistic adaptive-dynamics extension."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "reports/simulator/probabilistic_adaptive"
FIG = DIAG / "figures"
WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED."


def _read_csv(name: str) -> pd.DataFrame:
    path = DIAG / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(name: str) -> dict:
    path = DIAG / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _overview() -> None:
    gates = _read_json("prob_gates.json")
    summary = _read_csv("prob_summary.csv")
    if summary.empty:
        st.info("Probabilistic artifacts are not generated in this checkout.")
        return
    g = gates.get("gates", {})
    energy = summary[summary.metric == "energy_h20"].set_index("model")["mean"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Landmarks", 175)
    c2.metric("Primary energy H20", f"{energy.get('P6_adaptive_block', float('nan')):.3f}")
    c3.metric("Local-mean baseline", f"{energy.get('P1_local_mean', float('nan')):.3f}")
    c4.metric("Gate D", g.get("D_probabilistic_score_advantage", {}).get("status", "n/a"))
    st.error(
        "The prespecified stochastic adaptive VAR fails the probabilistic gate. Its energy score is worse "
        "than persistence, local mean and independent AR, and its 95% intervals cover 99.9% of outcomes. "
        "Do not read the lower deterministic MSE as a calibrated generative model."
    )
    view = summary[summary.metric.isin(
        ["energy_h20", "crps_h20_mean", "coverage_50", "coverage_80", "coverage_95", "spread_error_ratio_h20"]
    )]
    st.dataframe(view.pivot(index="model", columns="metric", values="mean").sort_values("energy_h20"), width="stretch")
    st.caption(
        "Energy score is the primary metric: 10-D training-fitted PCA projection, H20, 100 ensemble members. "
        "Lower is better. All five repeated sessions contribute; landmarks overlap, so intervals are not animal-level."
    )


def _baseline_ablation() -> None:
    local = _read_csv("baseline_paired_vs_local_mean.csv")
    ar = _read_csv("baseline_paired_vs_independent_ar.csv")
    session = _read_csv("baseline_per_session_h20.csv")
    if local.empty:
        st.info("Baseline ablation artifacts are not generated in this checkout.")
        return
    st.markdown(
        "**Does the adaptive population VAR beat simpler adaptive forecasters?** "
        "Improvement = reference MSE − model MSE; positive means the model is better."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("vs rolling local mean")
        st.dataframe(local, width="stretch", hide_index=True)
    with c2:
        st.markdown("vs independent per-neuron AR")
        st.dataframe(ar, width="stretch", hide_index=True)
    row_l = local[local.model == "B5_rolling_var_w500"]
    row_a = ar[ar.model == "B5_rolling_var_w500"]
    if not row_l.empty and not row_a.empty:
        st.warning(
            "B5 minus local mean = "
            f"{float(row_l.mean_improvement.iloc[0]):+.4f} "
            f"(95% CI {float(row_l.ci_low.iloc[0]):+.4f} to {float(row_l.ci_high.iloc[0]):+.4f}); "
            "B5 minus independent AR = "
            f"{float(row_a.mean_improvement.iloc[0]):+.4f} "
            f"(95% CI {float(row_a.ci_low.iloc[0]):+.4f} to {float(row_a.ci_high.iloc[0]):+.4f}). "
            "Both include zero: cross-neuron dynamics are not demonstrated."
        )
    if not session.empty:
        st.markdown("**H20 MSE by session**")
        st.dataframe(session, width="stretch", hide_index=True)
        fig = px.bar(
            session,
            x="worm_id",
            y="mean_mse",
            color="model",
            barmode="group",
            labels={"worm_id": "repeated session", "mean_mse": "mean H20 MSE"},
            title="Session heterogeneity: the adaptive advantage is not uniform",
        )
        st.plotly_chart(fig, width="stretch")


def _residuals() -> None:
    summary = _read_csv("residual_summary.csv")
    acf = _read_csv("residual_autocorrelation.csv")
    choice = _read_json("innovation_choice.json")
    if summary.empty:
        st.info("Residual artifacts are not generated in this checkout.")
        return
    st.markdown(
        "Residuals are block-prequential: the rolling operator is fitted at a block start and used "
        "one step ahead inside the block. This avoids treating in-sample residuals as out-of-sample innovations."
    )
    st.dataframe(summary, width="stretch", hide_index=True)
    if not acf.empty:
        fig = px.line(
            acf.groupby(["model", "lag"], as_index=False)["median_abs_acf"].mean(),
            x="lag",
            y="median_abs_acf",
            color="model",
            markers=True,
            labels={"lag": "residual lag (frames)", "median_abs_acf": "median |autocorrelation|"},
            title="Training residual dependence",
        )
        st.plotly_chart(fig, width="stretch")
    st.json(choice, expanded=False)
    st.caption(
        "The block bootstrap was selected by a training-only rule because lag-1 residual autocorrelation "
        "is high. That diagnostic choice is legitimate; the later failure is about variance propagation, not leakage."
    )


def _calibration() -> None:
    summary = _read_csv("prob_summary.csv")
    pit = _read_csv("prob_pit.csv")
    variance = _read_csv("prob_variance_summary.csv")
    if summary.empty:
        st.info("Calibration artifacts are not generated in this checkout.")
        return
    keep = ["P0_persistence", "P1_local_mean", "P2_independent_ar", "P6_adaptive_block", "P7_adaptive_diag"]
    metrics = ["coverage_50", "coverage_80", "coverage_95", "width_95", "spread_error_ratio_h20"]
    view = summary[(summary.model.isin(keep)) & (summary.metric.isin(metrics))]
    st.dataframe(view.pivot(index="model", columns="metric", values="mean"), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        if (FIG / "prob_calibration.png").exists():
            st.image(str(FIG / "prob_calibration.png"), caption="Nominal versus empirical central-interval coverage")
    with c2:
        if (FIG / "prob_variance_decomposition.png").exists():
            st.image(str(FIG / "prob_variance_decomposition.png"), caption="Within-ensemble and total implied variance")
    if not pit.empty:
        p6 = pit[pit.model == "P6_adaptive_block"]
        if not p6.empty:
            fig = go.Figure(go.Bar(x=p6["bin"], y=p6["density"], marker_color="#B55239"))
            fig.add_hline(y=0.1, line_dash="dash", line_color="#333")
            fig.update_layout(
                height=300,
                xaxis_title="PIT bin",
                yaxis_title="fraction",
                title="PIT, primary adaptive block model (uniform is ideal)",
            )
            st.plotly_chart(fig, width="stretch")
    if not variance.empty:
        st.dataframe(variance, width="stretch", hide_index=True)
    st.warning(
        "Coverage near 1.0 is not success when intervals are this wide. Width, spread-versus-error and "
        "total-variance ratio are reported to prevent a coverage-only pass."
    )


def _trajectories() -> None:
    bands = _read_csv("prob_example_bands.csv")
    trajectories = _read_csv("prob_example_trajectories.csv")
    meta = _read_json("prob_example_metadata.json")
    if bands.empty:
        st.info("Trajectory example artifacts are not generated in this checkout.")
        return
    p6 = bands[bands.model == "P6_adaptive_block"]
    fig = go.Figure()
    if not trajectories.empty:
        for _sample, group in trajectories.groupby("sample"):
            fig.add_trace(
                go.Scatter(
                    x=group.step,
                    y=group.population_mean,
                    mode="lines",
                    line=dict(color="#8A9AA8", width=0.7),
                    opacity=0.55,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    fig.add_trace(
        go.Scatter(x=p6.step, y=p6.q95, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    )
    fig.add_trace(
        go.Scatter(
            x=p6.step,
            y=p6.q05,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(59,126,161,0.18)",
            name="90% predictive band",
        )
    )
    fig.add_trace(go.Scatter(x=p6.step, y=p6.ensemble_population_mean, name="ensemble mean", line=dict(color="#1F4E66", width=2)))
    fig.add_trace(
        go.Scatter(
            x=p6.step,
            y=p6.deterministic_population_mean,
            name="deterministic conditional mean",
            line=dict(color="#D08A3E", width=1.6, dash="dash"),
        )
    )
    fig.add_trace(go.Scatter(x=p6.step, y=p6.truth_population_mean, name="observed continuation", line=dict(color="black", width=1.6)))
    fig.update_layout(
        height=420,
        xaxis_title="steps after landmark",
        yaxis_title="population-mean activity",
        title=f"One history, many simulated futures (session {meta.get('worm_id', '?')}, landmark {meta.get('landmark', '?')})",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width="stretch")
    if (FIG / "prob_horizon.png").exists():
        st.image(str(FIG / "prob_horizon.png"), caption="Energy-score growth with forecast horizon")
    if (FIG / "prob_example_rollouts.png").exists():
        st.image(str(FIG / "prob_example_rollouts.png"), caption="Static version of the trajectory example")
    st.caption(
        "These are model-generated trajectories, not data. Divergent paths are a property of the stochastic "
        "model and are not biological bifurcations. The observed continuation is shown only for descriptive comparison."
    )


def _decision() -> None:
    gates = _read_json("prob_gates.json")
    perturbation = _read_json("prob_perturbation_status.json")
    if not gates:
        st.info("Decision-gate artifacts are not generated in this checkout.")
        return
    rows = []
    for key, value in gates.get("gates", {}).items():
        rows.append(
            {
                "gate": key,
                "status": value.get("status"),
                "key evidence": value.get("interpretation", ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.warning(gates.get("overall", ""))
    secondary = gates.get("exploratory_secondary_finding", {})
    if secondary:
        st.markdown(
            f"**Post-hoc exploratory lead:** {secondary.get('model')} energy "
            f"{secondary.get('energy_h20'):.3f}, paired vs local mean "
            f"{secondary.get('paired_vs_local_mean'):+.3f}. This is not a gate pass: "
            "it does not reliably beat independent AR and remains overdispersed."
        )
    st.divider()
    st.error(WARNING)
    st.json(perturbation, expanded=False)
    st.caption(
        "The perturbation analysis was not launched because the stochastic model failed calibration and "
        "variance-fidelity gates. A virtual perturbation would not be causally interpretable."
    )


def render() -> None:
    st.header("Probabilistic adaptive dynamics")
    st.caption(
        "Read-only evaluation of whether the adaptive VAR conditional mean can support a calibrated "
        "probabilistic simulator. Exploratory repeated-session reanalysis; not animal-level inference."
    )
    tabs = st.tabs(["Overview", "Baseline ablation", "Residuals", "Calibration", "Trajectories", "Decision"])
    with tabs[0]:
        _overview()
    with tabs[1]:
        _baseline_ablation()
    with tabs[2]:
        _residuals()
    with tabs[3]:
        _calibration()
    with tabs[4]:
        _trajectories()
    with tabs[5]:
        _decision()
