"""Focused, read-only supervisor demo for the NeuroBranch handoff.

Run from the repository root:
    python -m streamlit run app/handoff_demo.py --server.port 8502

The app reads compact frozen artifacts only. It does not load checkpoints, raw
MATLAB files or any external data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "simulator" / "supervisor_handoff"
JOINT = ROOT / "reports" / "simulator" / "joint_dynamics"

INK = "#17202a"
MUTED = "#5d6d7e"
TEAL = "#117a65"
AMBER = "#b9770e"
RED = "#922b21"
GRAY = "#7f8c8d"

MODEL_LABELS = {
    "J0_persistence": "J0 persistence",
    "J1_independent_diag": "J1 independent AR",
    "J2_population_diag": "J2 population diag",
    "J3_population_full": "J3 population full",
    "J4_independent_full": "J4 independent full",
    "J5_population_lead1_boot": "J5 population lead-1",
}

SECTIONS = [
    "Start here",
    "Timeline",
    "Central result",
    "Real trajectory",
    "Mean vs stochastic",
    "Calibration",
    "Perturbation boundary",
    "Decision and next",
]

WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION \u2014 NOT CAUSALLY VALIDATED."


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}" if value != 0 else f"{value:.{digits}f}"


def chip(label: str, color: str) -> str:
    return (
        f"<span style='display:inline-block;padding:0.18rem 0.55rem;"
        f"border-radius:999px;background:{color};color:white;"
        f"font-size:0.72rem;font-weight:650;letter-spacing:0.02em;"
        f"margin:0 0.35rem 0.35rem 0'>{label}</span>"
    )


def header(summary: dict) -> None:
    st.markdown(
        f"""
        <div style="border-left:6px solid {TEAL};padding:0.2rem 0 0.2rem 1rem;margin-bottom:0.8rem">
        <div style="color:{MUTED};font-size:0.78rem;letter-spacing:0.12em;font-weight:700">
        NEUROBRANCH · MSc SUPERVISOR HANDOFF</div>
        <div style="font-size:2.0rem;line-height:1.12;font-weight:750;color:{INK};margin-top:0.3rem">
        Evaluating generative neural population dynamics and model-dependent perturbations</div>
        <div style="color:{MUTED};font-size:0.92rem;margin-top:0.4rem">
        Focused, read-only demonstration · exploratory repeated-session reanalysis · not a systematic review</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        chip("5 Kato sessions, not 5 animals", GRAY)
        + chip("Fixed 10-D PCA", "#1f618d")
        + chip("Calibration not established", AMBER)
        + chip("Biological validation NOT_EVALUABLE", RED)
        + chip("MODEL-DEPENDENT, NOT CAUSAL", RED),
        unsafe_allow_html=True,
    )


def footer(summary: dict) -> None:
    st.divider()
    st.caption(
        "Source: reports/simulator/supervisor_handoff/summary.json and the frozen "
        "joint-dynamics tables. Every perturbation statement is model-dependent. "
        + WARNING
    )


def section_start(summary: dict) -> None:
    st.subheader("What this is")
    st.markdown(
        "A bounded test of whether a generative model can roll forward from a measured "
        "*C. elegans* population state, and whether a modelled perturbation of a named "
        "neuron can support an experimentally testable hypothesis. The exact MSc brief "
        "is still provisional."
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sessions", "5", "not 5 animals")
    col2.metric("H20 landmarks", "175", "overlapping")
    col3.metric("Representation", "10-D", "train-only PCA")
    col4.metric("Ensemble", "100", "members")

    st.markdown("### The one result to discuss")
    st.success(
        "In a fixed 10-D PCA, the corrected population conditional mean predicts joint "
        "H20 trajectories better than independent per-neuron AR. It is an exploratory "
        "mean result, not a calibrated generative simulator."
    )
    central = summary["central_claim"]
    col1, col2 = st.columns(2)
    col1.metric(
        "J5 vs J1 · H20 energy",
        fmt(central["primary"]["improvement"]),
        f"95% CI [{central['primary']['ci_low']:+.3f}, {central['primary']['ci_high']:+.3f}]",
    )
    col2.metric(
        "J5 vs J1 · trajectory",
        fmt(central["trajectory"]["improvement"]),
        f"95% CI [{central['trajectory']['ci_low']:+.3f}, {central['trajectory']['ci_high']:+.3f}]",
    )
    st.info(
        "Use the sidebar to follow the 8-minute route. The demo deliberately does not "
        "start with the 56-tab audit dashboard."
    )
    st.markdown("### What the labels mean")
    rows = [
        ("Exploratory", "Directionally supported but exposed to test reuse or undocumented independence."),
        ("Negative", "The prespecified or natural comparison did not support the claim."),
        ("Withdrawn", "A previous number or interpretation is superseded by a documented correction."),
        ("Not evaluable", "The data or design cannot answer the question."),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Status", "Meaning"]), hide_index=True, width="stretch")


def section_timeline(summary: dict) -> None:
    st.subheader("Chronological account, not a phase-number parade")
    timeline = load_csv(str(OUT / "phase_timeline.csv"))
    st.markdown(
        "Each row is a workstream that changed the next question. The negative results "
        "are connected: representation leakage, then weak full-neuron predictability, "
        "then branching and specificity failures, then calibration failure."
    )
    display = timeline[["order", "workstream", "key_result", "status"]].rename(
        columns={
            "order": "#",
            "workstream": "Workstream",
            "key_result": "What it established",
            "status": "Status",
        }
    )
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        height=560,
        column_config={
            "#": st.column_config.TextColumn(width="small"),
            "Workstream": st.column_config.TextColumn(width="medium"),
            "What it established": st.column_config.TextColumn(width="large"),
            "Status": st.column_config.TextColumn(width="small"),
        },
    )
    counts = timeline["status"].value_counts().rename_axis("status").reset_index(name="count")
    fig = go.Figure(
        go.Bar(
            x=counts["count"],
            y=counts["status"],
            orientation="h",
            marker_color=[TEAL if s in {"supported", "technical"} else AMBER if s in {"exploratory", "mixed"} else RED for s in counts["status"]],
            text=counts["count"],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=10, r=30, t=10, b=10),
        xaxis_title="Workstreams",
        yaxis_title="",
        showlegend=False,
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Statuses summarize the handoff; they are not independent significance tests.")


def forest_figure(summary: dict) -> go.Figure:
    rows = [
        ("J5 vs J1 · energy", summary["central_claim"]["primary"]),
        ("J5 vs J1 · trajectory", summary["central_claim"]["trajectory"]),
    ]
    wanted = [
        ("J5 vs J0 · energy", "J5_population_lead1_boot", "J0_persistence", "energy_h20"),
        ("J5 vs J0 · trajectory", "J5_population_lead1_boot", "J0_persistence", "trajectory_energy_h20"),
        ("J2 vs J1 · energy", "J2_population_diag", "J1_independent_diag", "energy_h20"),
        ("J2 vs J1 · trajectory", "J2_population_diag", "J1_independent_diag", "trajectory_energy_h20"),
        ("J3 vs J1 · energy", "J3_population_full", "J1_independent_diag", "energy_h20"),
    ]
    for label, model, reference, metric in wanted:
        row = next(
            r for r in summary["paired"] if r["model"] == model and r["reference"] == reference and r["metric"] == metric
        )
        rows.append((label, row))

    labels = [r[0] for r in rows]
    vals = [r[1]["improvement"] for r in rows]
    lows = [r[1]["ci_low"] for r in rows]
    highs = [r[1]["ci_high"] for r in rows]
    colors = [TEAL if lo > 0 else GRAY for lo in lows]
    fig = go.Figure()
    for label, val, lo, hi, color in zip(labels, vals, lows, highs, colors, strict=True):
        fig.add_trace(
            go.Scatter(
                x=[val],
                y=[label],
                mode="markers",
                marker=dict(size=10, color=color, line=dict(color="white", width=1.5)),
                error_x=dict(type="data", symmetric=False, array=[hi - val], arrayminus=[val - lo], color=color, thickness=2),
                showlegend=False,
                hovertemplate=f"{label}<br>{val:+.3f} [{lo:+.3f}, {hi:+.3f}]<extra></extra>",
            )
        )
    fig.add_vline(x=0, line_dash="dash", line_color=RED, line_width=1.4)
    fig.update_layout(
        height=430,
        margin=dict(l=10, r=150, t=20, b=30),
        xaxis_title="Improvement (reference minus model; positive favours first model)",
        yaxis_title="",
        plot_bgcolor="white",
        xaxis=dict(range=[-0.30, 1.10], gridcolor="#e5e8e8"),
        yaxis=dict(autorange="reversed"),
    )
    for val, hi, label in zip(vals, highs, labels, strict=True):
        fig.add_annotation(
            x=hi + 0.025,
            y=label,
            text=f"{val:+.3f} [{hi:+.3f}]",
            showarrow=False,
            font=dict(size=10, color=INK),
            xanchor="left",
        )
    return fig


def section_central(summary: dict) -> None:
    st.subheader("The central contrast")
    st.markdown(
        "All models share the same 175 H20 landmarks, the same initial observed state, "
        "the same fixed 10-D PCA and 100 ensemble members. J1/J4 and J2/J3 form a "
        "2 × 2 factorial: independent vs population mean, crossed with diagonal vs full "
        "innovation covariance."
    )
    col1, col2 = st.columns([1.25, 1])
    with col1:
        st.plotly_chart(forest_figure(summary), width="stretch")
    with col2:
        scores = pd.DataFrame(summary["scores"])
        display = scores[["label", "energy_h20", "trajectory_h20", "coverage_full_50", "covariance_error"]].rename(
            columns={
                "label": "Model",
                "energy_h20": "H20 energy",
                "trajectory_h20": "Trajectory",
                "coverage_full_50": "Cov50",
                "covariance_error": "Cov err",
            }
        )
        st.dataframe(
            display,
            hide_index=True,
            width="stretch",
            height=300,
            column_config={
                "H20 energy": st.column_config.NumberColumn(format="%.3f"),
                "Trajectory": st.column_config.NumberColumn(format="%.3f"),
                "Cov50": st.column_config.NumberColumn(format="%.3f"),
                "Cov err": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.caption("Lower energy/trajectory is better. J2 and J5 are effectively tied.")
    st.warning(
        "Exploratory: the test split was reused, the 175 landmarks overlap, and the "
        "five sessions are repeated recordings, not documented independent animals. "
        "The bootstrap resamples landmarks, not animals."
    )
    st.markdown("### J2 is an internal replication, not a second headline")
    st.markdown(
        "J2 (parametric diagonal population innovations) gives energy 2.700 and "
        "trajectory 2.058, almost identical to J5. The mean advantage is therefore not "
        "an artefact of one innovation sampler. J2 still fails calibration."
    )


def trajectory_frame():
    observed = load_csv(str(JOINT / "example_observed.csv"))
    bands = load_csv(str(JOINT / "example_bands.csv"))
    return observed, bands


def section_trajectory(summary: dict) -> None:
    st.subheader("A real observed continuation and a forecast from measured history")
    observed, bands = trajectory_frame()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=observed["step"],
            y=observed["observed_pc1"],
            mode="lines+markers",
            name="Observed continuation",
            line=dict(color=INK, width=2.6),
            marker=dict(size=4),
        )
    )
    j5 = bands[bands["model"] == "J5_population_lead1_boot"]
    fig.add_trace(
        go.Scatter(
            x=pd.concat([j5["step"], j5["step"][::-1]]),
            y=pd.concat([j5["q95"], j5["q05"][::-1]]),
            fill="toself",
            fillcolor="rgba(17,122,101,0.16)",
            line=dict(color="rgba(255,255,255,0)"),
            name="J5 90% band",
            hoverinfo="skip",
        )
    )
    for model, color, width, label in [
        ("J5_population_lead1_boot", TEAL, 2.4, "J5 population mean"),
        ("J1_independent_diag", AMBER, 1.7, "J1 independent AR"),
        ("J0_persistence", GRAY, 1.3, "J0 persistence"),
    ]:
        frame = bands[bands["model"] == model]
        fig.add_trace(
            go.Scatter(
                x=frame["step"],
                y=frame["q50"],
                mode="lines",
                name=label,
                line=dict(color=color, width=width),
            )
        )
    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=20, b=40),
        xaxis_title="H20 step after the last observed frame",
        yaxis_title="PC1 of the fixed training PCA",
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#e5e8e8"),
        yaxis=dict(gridcolor="#e5e8e8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Session 0, landmark 2690: the first prespecified test landmark, not selected by score. "
        "The band is ensemble uncertainty, not a calibrated biological distribution."
    )
    st.info(
        "The model sees only the measured history. The observed continuation is plotted "
        "for evaluation, not supplied to the rollout."
    )


def section_stochastic(summary: dict) -> None:
    st.subheader("Conditional mean vs stochastic rollout")
    npz_path = JOINT / "trajectory_examples.npz"
    if not npz_path.exists():
        st.warning(
            "The optional 10-member visualization subset is absent. The CSV bands still "
            "show the conditional mean and 90% interval; no checkpoint is required."
        )
        section_trajectory(summary)
        return
    data = load_npz(str(npz_path))
    key = "0_2690_J5_population_lead1_boot"
    if key not in data:
        st.warning("The expected J5 example is absent from the visualization subset.")
        return
    ensemble = data[key]
    observed, bands = trajectory_frame()
    steps = np.arange(1, ensemble.shape[1] + 1)
    fig = go.Figure()
    for member in range(ensemble.shape[0]):
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=ensemble[member, :, 0],
                mode="lines",
                line=dict(color="rgba(17,122,101,0.20)", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=ensemble[:, :, 0].mean(axis=0),
            mode="lines",
            name="J5 ensemble mean",
            line=dict(color=TEAL, width=2.8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=observed["step"],
            y=observed["observed_pc1"],
            mode="lines+markers",
            name="Observed",
            line=dict(color=INK, width=2.4),
            marker=dict(size=4),
        )
    )
    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=20, b=40),
        xaxis_title="H20 step",
        yaxis_title="PC1 of the fixed training PCA",
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#e5e8e8"),
        yaxis=dict(gridcolor="#e5e8e8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Ten of 100 members, stored as float32 for visualization. The full scores use all "
        "100 members and are not recomputed here."
    )
    st.warning(
        "A fan of trajectories is not evidence of biological branching. The observed "
        "conditional outcomes are concentrated, and the simulator is worse than a Markov "
        "baseline on the branching test."
    )


def section_calibration(summary: dict) -> None:
    st.subheader("The mean can be useful while the distribution is not calibrated")
    calibration = pd.DataFrame(summary["calibration"])
    models = [
        "J0_persistence",
        "J1_independent_diag",
        "J2_population_diag",
        "J3_population_full",
        "J5_population_lead1_boot",
    ]
    colors = {
        "J0_persistence": GRAY,
        "J1_independent_diag": AMBER,
        "J2_population_diag": "#76d7c4",
        "J3_population_full": RED,
        "J5_population_lead1_boot": TEAL,
    }
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0.45, 1.0],
            y=[0.45, 1.0],
            mode="lines",
            name="Nominal",
            line=dict(color=INK, dash="dash", width=1.2),
        )
    )
    for model in models:
        row = calibration[calibration["model"] == model].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[0.50, 0.80, 0.95],
                y=[row["full_50"], row["full_80"], row["full_95"]],
                mode="lines+markers",
                name=MODEL_LABELS[model],
                line=dict(color=colors[model], width=2.6 if model == "J5_population_lead1_boot" else 1.5),
                marker=dict(size=6),
            )
        )
    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=20, b=40),
        xaxis_title="Nominal coverage",
        yaxis_title="Empirical coverage (full neuron)",
        xaxis=dict(range=[0.45, 1.0], tickvals=[0.50, 0.80, 0.95], ticktext=["50%", "80%", "95%"]),
        yaxis=dict(range=[0.45, 1.02], gridcolor="#e5e8e8"),
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, width="stretch")
    col1, col2, col3 = st.columns(3)
    col1.metric("J5 50% coverage", "0.588", "nominal 0.50")
    col2.metric("J5 covariance error", "0.604", "gate ≤ 0.50", delta_color="inverse")
    col3.metric("J5 95% coverage", "0.947", "nominal 0.95")
    st.error(
        "Stop label: PROBABILISTIC CALIBRATION NOT ESTABLISHED. Student-t and "
        "parameter-uncertainty corrections are both worse than J5."
    )
    st.caption(
        "10-D PCA 50% coverage is 0.546. Full-neuron and PCA coverage are different "
        "estimands; both fail the 50% level."
    )


def section_perturbation(summary: dict) -> None:
    st.subheader("A mathematical model edit is not a biological stimulation")
    st.error(WARNING)
    st.markdown(
        "The intervention engine edits a **model state** with additive, clamp, zero or "
        "scale operations. That is a valid counterfactual inside the fitted model. It is "
        "not optogenetics, a lesion, a dose, or a prediction of a wet-lab effect. The "
        "model's causal graph is assumed, not identified."
    )
    status = load_json(str(JOINT / "perturbation_status.json"))
    col1, col2, col3 = st.columns(3)
    col1.metric("Joint-study perturbation", status.get("status", "NOT_LAUNCHED"))
    col2.metric("Biological validation", status.get("biological_validation", "NOT_EVALUABLE"))
    col3.metric("Matched intervention data", "none")
    st.markdown("### What would be required for a causal reading")
    checks = pd.DataFrame(
        [
            ("Causal sufficiency", "No: 31–47 named cells of 302; no neuromodulatory/mechanical variables."),
            ("Correct functional form", "No: the selected free-running model is worse than persistence."),
            ("Stationarity / transportability", "No: strict-forward failure and ~60° train-half rotation."),
            ("Support overlap", "Partial: OOD fractions 8–38% in historical perturbation runs."),
            ("No unmeasured confounding", "No: arousal, oxygen, muscle state and stimulus are unrecorded."),
            ("Intervention fidelity", "No: no light power, opsin expression or cell-specific calibration."),
        ],
        columns=["Assumption", "Current status"],
    )
    st.dataframe(checks, hide_index=True, width="stretch")
    st.markdown("### The historical example is a cautionary tale")
    st.markdown(
        "The RIVL +2 SD example produced ΔP(TURN)=+0.333, but after correcting the "
        "random-control sampling, 4/19 non-candidate cells reached |Δ| ≥ 0.333 and the "
        "empirical one-sided p was about 0.21. The effect is not specific."
    )
    st.caption(
        "No perturbation trajectory is shown here as a biological prediction. The joint "
        "study did not launch perturbations because calibration/covariance failed and no "
        "matched intervention dataset exists."
    )


def section_decision(summary: dict) -> None:
    st.subheader("Do not run another architecture search on the reused test split")
    st.markdown(
        "The decision is between a clean replication and an honest intervention-design "
        "study. Both fit 150–180 hours. Neither promises a causal simulator."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div style="border:1px solid #d5dbdb;border-top:5px solid {TEAL};border-radius:6px;padding:1rem;height:100%">
            <div style="color:{TEAL};font-weight:750;letter-spacing:0.08em;font-size:0.78rem">PROPOSAL A · 168 h</div>
            <div style="font-size:1.25rem;font-weight:700;margin:0.35rem 0">Replicate the population mean</div>
            <div style="color:{MUTED};font-size:0.9rem;line-height:1.45">
            Independent cohort; fixed train-only representation; H20 joint energy as the
            single primary endpoint; J1 and persistence baselines; animal-level resampling.
            A negative result is still a result.
            </div></div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div style="border:1px solid #d5dbdb;border-top:5px solid {AMBER};border-radius:6px;padding:1rem;height:100%">
            <div style="color:{AMBER};font-weight:750;letter-spacing:0.08em;font-size:0.78rem">PROPOSAL B · 174 h</div>
            <div style="font-size:1.25rem;font-weight:700;margin:0.35rem 0">Intervention-aware design</div>
            <div style="color:{MUTED};font-size:0.9rem;line-height:1.45">
            Specify target, intervention, timing, sham and outcome; audit matched data; run a
            minimal held-out-animal/-target falsification test. Deliver a protocol, not a
            causal simulator.
            </div></div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("### Questions that change the scope")
    questions = [
        "What is the primary object: trajectory simulation, a behavioural event, or an intervention response?",
        "What counts as branching for your purpose?",
        "Is a model-dependent virtual perturbation useful even when not causally valid?",
        "Is there an independent cohort with documented animal IDs?",
        "Is there matched stimulation data with neuron identities and a sham control?",
        "What would make you say stop?",
    ]
    for question in questions:
        st.markdown(f"- {question}")
    st.markdown("### Strongest allowed sentence")
    st.info(
        "In a fixed 10-D PCA fitted on training frames only, the corrected population-mean "
        "model provides evidence of improved joint neural-trajectory prediction over an "
        "independent-neuron AR baseline at H20, but generative calibration and intervention "
        "validity remain unresolved."
    )


def main() -> None:
    st.set_page_config(
        page_title="NeuroBranch supervisor demo",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        f"""
        <style>
        .block-container {{ padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1400px; }}
        h1, h2, h3 {{ color: {INK}; }}
        div[data-testid="stMetricValue"] {{ font-size: 1.55rem; }}
        div[data-testid="stMetricLabel"] {{ color: {MUTED}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    if not (OUT / "summary.json").exists():
        st.error(
            "The compact handoff summary is missing. Run `python scripts/build_supervisor_handoff.py` "
            "from the repository root."
        )
        st.stop()
    summary = load_json(str(OUT / "summary.json"))
    header(summary)

    section = st.sidebar.radio("Presentation section", SECTIONS, index=0)
    st.sidebar.markdown(
        "---\n"
        "**Route:** Start → Timeline → Central → Real trajectory → Mean vs stochastic → "
        "Calibration → Perturbation → Decision.\n\n"
        "The sidebar does not change any data. All views are read-only."
    )
    st.sidebar.caption(f"Artifacts: `{OUT.relative_to(ROOT)}`")

    if section == "Start here":
        section_start(summary)
    elif section == "Timeline":
        section_timeline(summary)
    elif section == "Central result":
        section_central(summary)
    elif section == "Real trajectory":
        section_trajectory(summary)
    elif section == "Mean vs stochastic":
        section_stochastic(summary)
    elif section == "Calibration":
        section_calibration(summary)
    elif section == "Perturbation boundary":
        section_perturbation(summary)
    elif section == "Decision and next":
        section_decision(summary)
    footer(summary)


if __name__ == "__main__":
    main()
