"""Streamlit tab for free-running simulation and virtual perturbations."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from neurobranch.config import PROJECT_ROOT
from neurobranch.data.kato import MACRO_ORDER
from neurobranch.preprocessing import split_starts
from neurobranch.simulator.branching import tv
from neurobranch.simulator.perturbation import CANDIDATES, outcome_distribution
from neurobranch.simulator.train import load_simulator_for
from neurobranch.simulator.types import PerturbationSpec

REPORT_DIR = PROJECT_ROOT / "reports" / "simulator"


def _json(name: str, default=None):
    try:
        return json.loads((REPORT_DIR / name).read_text(encoding="utf-8"))
    except Exception:
        return default


@st.cache_resource(show_spinner="Loading simulator …")
def _sim(kind: str, worm_id: int, seed: int, root: str):
    from neurobranch.config import ExperimentConfig
    from neurobranch.dynamics.training import load_worm_data

    cfg = ExperimentConfig.load(PROJECT_ROOT / "configs/experiments/blocked_suite.yaml")
    data = load_worm_data(cfg)[worm_id]
    return load_simulator_for(data, kind, seed, root=root)


def render(data, pca, worm_id: int, model_name: str, seed: int, root: Path) -> None:
    st.subheader("Core dynamics simulator")
    st.warning(
        "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED. "
        "These are free-running model rollouts under stated assumptions; no stimulation-response "
        "model was fitted to Kato 2015. Category C is not_evaluable.",
        icon="⚠️",
    )
    summary = _json("summary.json", {})
    selection = _json("selection.json", {})
    g1 = summary.get("gate_1", {})
    g2 = summary.get("gate_2", {})
    g6 = summary.get("gate_6", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("G1 functioning", str(g1.get("status", "not_run")).upper())
    c2.metric("G2 dynamical advantage", str(g2.get("status", "not_run")).upper())
    c3.metric("Selected model", str(selection.get("selected_kind", model_name)))
    c4.metric("G6 biological validation", str(g6.get("status", "not_evaluable")))
    st.caption(
        "G2 compares free-running rollouts with a constant-initial-state baseline and one-step "
        "persistence; the selected checkpoint is chosen on validation one-step R² only."
    )

    with st.expander("Observed vs simulated trajectories", expanded=True):
        frames = [
            int(t)
            for t in split_starts(data, "test", history=1, horizon=1)
            if data.masks["test"][int(t) : int(t) + 31].all()
        ]
        frame = st.select_slider(
            "Initial frame (held-out)",
            options=frames[:80],
            value=frames[0],
            format_func=lambda t: f"frame {t} · {MACRO_ORDER[int(data.y_macro[t])]}",
        )
        horizon = st.slider("Horizon", 5, 30, 20, key="sim_horizon")
        n = st.slider("Ensemble size", 10, 60, 30, step=10, key="sim_n")
        noise = st.slider("Noise SD (standardised units)", 0.0, 0.2, 0.05, 0.01, key="sim_noise")
        run = st.button("Roll out simulator", type="primary")
        if run:
            try:
                sim = _sim(selection.get("selected_kind", model_name), int(worm_id), int(seed), str(root))
                ensemble = sim.rollout_ensemble(
                    data.X[frame], n, horizon, base_seed=1000 + frame, noise_std=noise
                )
                Zobs = pca.transform(data.X[frame : frame + horizon + 1])
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=Zobs[:, 0],
                        y=Zobs[:, 1],
                        mode="lines+markers",
                        name="observed",
                        line=dict(color="black", width=4),
                    )
                )
                for r in ensemble.rollouts:
                    Z = pca.transform(r.states)
                    fig.add_trace(
                        go.Scatter(
                            x=Z[:, 0],
                            y=Z[:, 1],
                            mode="lines",
                            line=dict(color="#2E86AB", width=1),
                            opacity=0.25,
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
                fig.update_layout(
                    height=420,
                    xaxis_title="PC1",
                    yaxis_title="PC2",
                    legend=dict(orientation="h", y=1.05),
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    f"Ensemble stochastic: {ensemble.stochastic}; OOD fraction: {ensemble.ood_fraction:.3f}; "
                    f"observed initial label: {MACRO_ORDER[int(data.y_macro[frame])]}."
                )
            except Exception as exc:
                st.error(f"Simulator unavailable: {exc}")

    with st.expander("Named-neuron virtual perturbation", expanded=False):
        target = st.selectbox("Target neuron", [n for n in CANDIDATES if n in data.predictor_names])
        protocol = st.radio(
            "Protocol",
            ["dynamical_input", "initial_state"],
            horizontal=True,
            help="Initial-state probes test mathematical sensitivity; they are not stimulation.",
        )
        amplitude = st.slider("Amplitude (training SD units)", -3.0, 3.0, 1.0, 0.5)
        onset = st.slider("Onset (frames)", 0, 10, 0)
        duration = st.slider("Duration (frames)", 0, 15, 10)
        run_p = st.button("Run virtual perturbation")
        if run_p:
            try:
                sim = _sim(selection.get("selected_kind", model_name), int(worm_id), int(seed), str(root))
                frames = [
                    int(t)
                    for t in split_starts(data, "test", history=1, horizon=1)
                    if data.masks["test"][int(t) : int(t) + 21].all()
                ]
                frame = frames[0]
                base = sim.rollout_ensemble(
                    data.X[frame], 20, 20, base_seed=500 + frame, noise_std=0.05
                )
                spec = PerturbationSpec(
                    target=target,
                    protocol=protocol,
                    mode="additive",
                    amplitude=float(amplitude),
                    onset=int(onset),
                    duration=int(duration),
                )
                pert = sim.rollout_ensemble(
                    data.X[frame], 20, 20, base_seed=500 + frame, noise_std=0.05, perturbation=spec
                )
                initial = int(data.y_macro[frame])
                p0 = outcome_distribution(base.rollouts, initial)
                p1 = outcome_distribution(pert.rollouts, initial)
                fig = go.Figure(
                    data=[
                        go.Bar(name="baseline", x=list(MACRO_ORDER), y=p0, marker_color="#7D8491"),
                        go.Bar(name="perturbed", x=list(MACRO_ORDER), y=p1, marker_color="#E4572E"),
                    ]
                )
                fig.update_layout(
                    barmode="group",
                    height=320,
                    yaxis_title="outcome probability",
                    margin=dict(l=10, r=10, t=30, b=10),
                )
                st.plotly_chart(fig, width="stretch")
                st.metric("TV(perturbed, baseline)", f"{tv(p1, p0):.3f}")
                st.caption(
                    f"{spec.protocol} on {target}; model seed {seed}; OOD fraction {pert.ood_fraction:.3f}. "
                    "A non-zero TV is a model prediction, not an experimental effect."
                )
            except Exception as exc:
                st.error(f"Perturbation unavailable: {exc}")

    st.divider()
    st.markdown("**Frozen reports**")
    spec = pd.read_csv(REPORT_DIR / "perturbation_specificity.csv") if (REPORT_DIR / "perturbation_specificity.csv").exists() else None
    if spec is not None:
        st.dataframe(spec, width="stretch", hide_index=True)
    for name in ["trajectory_pca.png", "branching.png", "perturbation.png"]:
        path = PROJECT_ROOT / "figures" / "simulator" / name
        if path.exists():
            st.image(str(path), caption=name.replace("_", " ").replace(".png", ""), width="stretch")
    st.caption(
        "G3 branching and G5 perturbation reports are in reports/simulator/. "
        "No report establishes a bifurcation, a causal effect, or whole-brain dynamics."
    )
