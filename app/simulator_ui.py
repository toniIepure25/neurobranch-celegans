"""Streamlit tab for Phase 6 free-running simulation and bounded perturbations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
PHASE6_DIR = REPORT_DIR / "phase6"
WARNING = "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED."


def _json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


@st.cache_resource(show_spinner="Loading Phase 6 simulator …")
def _sim(family: str, worm_id: int, root: str):
    from neurobranch.config import ExperimentConfig
    from neurobranch.dynamics.training import load_worm_data
    from neurobranch.simulator import NeuralSimulator
    from neurobranch.simulator.train import checkpoint_root

    cfg = ExperimentConfig.load(PROJECT_ROOT / "configs/experiments/blocked_suite.yaml")
    data = load_worm_data(cfg)[worm_id]
    if family == "phase5_mlp":
        return load_simulator_for(data, "mlp_residual", 17, root=str(root))
    path = checkpoint_root(cfg) / "phase6" / family / f"worm{worm_id}.pt"
    return NeuralSimulator.from_checkpoint(path, data)


def _test_frames(data, horizon: int) -> list[int]:
    starts = split_starts(data, "test", history=3, horizon=horizon)
    return [int(t) for t in starts[:120]]


def _support_text(sim) -> str:
    quant = sim.support.calibration_quantile
    return (
        f"support threshold={sim.support.threshold:.3f} "
        f"({'validation ' + str(quant) + ' quantile' if quant is not None else 'training LOO'})"
    )


def render(data, pca, worm_id: int, model_name: str, seed: int, root: Path) -> None:
    st.subheader("Core dynamics simulator")
    st.warning(
        f"{WARNING} These are free-running model rollouts under stated assumptions. "
        "No stimulation-response model was fitted to Kato 2015; Gate 7 is not_evaluable.",
        icon="⚠️",
    )
    gates = _json(PHASE6_DIR / "gates.json", {}) or {}
    gate_map = {g["gate"]: g for g in gates.get("gates", [])}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("G1 functioning", str(gate_map.get("G1", {}).get("status", "not_run")).upper())
    c2.metric("G2 autonomous skill", str(gate_map.get("G2", {}).get("status", "not_run")).upper())
    c3.metric("G3 distributional", str(gate_map.get("G3", {}).get("status", "not_run")).upper())
    c4.metric("G7 biological", str(gate_map.get("G7", {}).get("status", "not_evaluable")))
    st.caption(
        "Test block 2 was reused from Phase 5: every displayed test result is exploratory, "
        "not confirmatory. Gate details are in reports/simulator/phase6/gates.json."
    )

    family = st.selectbox(
        "Model family",
        ["var", "stochastic_var", "switching_var", "phase5_mlp"],
        index=0,
        help="VAR is deterministic; stochastic/switching families use native innovations.",
    )
    with st.expander("Observed vs simulated trajectories", expanded=True):
        frames = _test_frames(data, 20)
        frame = st.select_slider(
            "Initial frame (held-out, real 3-frame history)",
            options=frames,
            value=frames[0],
            format_func=lambda t: f"frame {t} · {MACRO_ORDER[int(data.y_macro[t])]}",
        )
        horizon = st.slider("Horizon", 5, 100, 20, key="sim_horizon")
        n = st.slider("Ensemble size", 10, 60, 30, step=10, key="sim_n")
        sim_preview = _sim(family, int(worm_id), str(root))
        native = bool(getattr(sim_preview.model, "native_stochastic", False))
        stochastic = st.checkbox(
            "Use native stochasticity",
            value=native,
            disabled=not native,
            help="Deterministic models have no innovation distribution; adding arbitrary noise is not native.",
        )
        noise = 0.0 if native else st.slider(
            "Extra noise SD (standardised units)", 0.0, 0.10, 0.0, 0.01, key="sim_noise"
        )
        run = st.button("Roll out simulator", type="primary")
        if run:
            try:
                sim = _sim(family, int(worm_id), str(root))
                hist = sim.initial_history(int(frame), history=int(getattr(sim.model, "history_len", 1)))
                ensemble = sim.rollout_ensemble(
                    None,
                    n,
                    horizon,
                    initial_history=hist,
                    initial_macro=int(data.y_macro[frame]),
                    base_seed=1000 + frame,
                    noise_std=float(noise),
                    stochastic=stochastic if native else False,
                )
                Zobs = pca.transform(data.X[frame : frame + horizon + 1])
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=Zobs[:, 0], y=Zobs[:, 1], mode="lines+markers",
                        name="observed", line=dict(color="black", width=4),
                    )
                )
                for r in ensemble.rollouts:
                    Z = pca.transform(r.states)
                    fig.add_trace(
                        go.Scatter(
                            x=Z[:, 0], y=Z[:, 1], mode="lines",
                            line=dict(color="#2E86AB", width=1), opacity=0.25,
                            showlegend=False, hoverinfo="skip",
                        )
                    )
                fig.update_layout(
                    height=420, xaxis_title="PC1", yaxis_title="PC2",
                    legend=dict(orientation="h", y=1.05), margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    f"{_support_text(sim)} · native stochastic={ensemble.stochastic} · "
                    f"OOD fraction={ensemble.ood_fraction:.3f} · "
                    f"readout support={np.mean([r.metadata['readout_supported_fraction'] for r in ensemble.rollouts]):.3f}."
                )
                st.info(
                    "A visually diverging trajectory is not a bifurcation. Native stochasticity "
                    "is sampled from fitted innovations, not added to force entropy."
                )
            except Exception as exc:
                st.error(f"Simulator unavailable: {exc}")

    with st.expander("Support-constrained named-neuron perturbation", expanded=False):
        target = st.selectbox("Target neuron", [n for n in CANDIDATES if n in data.predictor_names])
        j = data.predictor_names.index(target)
        train = split_starts(data, "train", history=1, horizon=1)
        delta = data.X[train + 1, j] - data.X[train, j]
        delta_sd = float(np.std(delta))
        lo, hi = np.quantile(data.X[:, j], [0.05, 0.95])
        cap = max(1e-6, 0.5 * (hi - lo))
        amplitude = float(min(delta_sd, cap))
        direction = st.radio("Direction", ["increase (+)", "decrease (−)"], horizontal=True)
        q = st.slider("Amplitude multiplier q (observed one-step Δ)", 0.5, 2.0, 1.0, 0.5)
        protocol = st.radio(
            "Protocol",
            ["initial_state", "dynamical_input"],
            horizontal=True,
            help="Initial-state probes test mathematical sensitivity; they are not stimulation.",
        )
        duration = 1 if protocol == "initial_state" else st.slider("Duration (frames)", 1, 5, 1)
        run_p = st.button("Run bounded virtual perturbation")
        if run_p:
            try:
                sim = _sim(family, int(worm_id), str(root))
                frame_h = int(_test_frames(data, 20)[0])
                hist = sim.initial_history(int(frame_h), history=int(getattr(sim.model, "history_len", 1)))
                base = sim.rollout_ensemble(
                    None, 20, 20, initial_history=hist,
                    initial_macro=int(data.y_macro[frame_h]), base_seed=500 + frame_h,
                    stochastic=native,
                )
                signed = (1.0 if direction.startswith("increase") else -1.0) * q * amplitude
                spec = PerturbationSpec(
                    target=target, protocol=protocol, mode="additive",
                    amplitude=float(signed), onset=0, duration=int(duration),
                )
                pert = sim.rollout_ensemble(
                    None, 20, 20, initial_history=hist,
                    initial_macro=int(data.y_macro[frame_h]), base_seed=500 + frame_h,
                    stochastic=native, perturbation=spec,
                )
                initial = int(data.y_macro[frame_h])
                p0 = outcome_distribution(base.rollouts, initial)
                p1 = outcome_distribution(pert.rollouts, initial)
                fig = go.Figure(
                    data=[
                        go.Bar(name="baseline", x=list(MACRO_ORDER), y=p0, marker_color="#7D8491"),
                        go.Bar(name="perturbed", x=list(MACRO_ORDER), y=p1, marker_color="#E4572E"),
                    ]
                )
                fig.update_layout(
                    barmode="group", height=320, yaxis_title="outcome probability",
                    margin=dict(l=10, r=10, t=30, b=10),
                )
                st.plotly_chart(fig, width="stretch")
                st.metric("TV(perturbed, baseline)", f"{tv(p1, p0):.3f}")
                st.caption(
                    f"{protocol} on {target}; signed amplitude {signed:+.3f} standardised units "
                    f"(q={q:g}, observed Δ SD={delta_sd:.3f}, cap={cap:.3f}); "
                    f"in-support={1 - pert.ood_fraction:.3f}; readout support="
                    f"{np.mean([r.metadata['readout_supported_fraction'] for r in pert.rollouts]):.3f}."
                )
                st.warning(WARNING)
            except Exception as exc:
                st.error(f"Perturbation unavailable: {exc}")

    st.divider()
    st.markdown("**Phase 6 frozen evidence**")
    col1, col2 = st.columns(2)
    for col, name in ((col1, "test_overview.json"), (col2, "branching_overview.json")):
        payload = _json(PHASE6_DIR / name)
        if payload:
            col.json(payload)
    table = PHASE6_DIR / "perturbation_summary.csv"
    if table.exists():
        st.dataframe(pd.read_csv(table), width="stretch", hide_index=True)
    st.caption(
        "No report establishes a bifurcation, a causal effect, or whole-brain dynamics. "
        "Gate 6/7 are not_evaluable without matched intervention data."
    )
