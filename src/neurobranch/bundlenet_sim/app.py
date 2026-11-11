"""Vienna Neuroinformatics Workbench — Streamlit UI.

Run from the repository root:
    streamlit run src/neurobranch/bundlenet_sim/app.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from neurobranch.bundlenet_sim.clamped import (  # noqa: E402
    ClampedParameters,
    ClampedSimulationError,
    SignalPulse,
)
from neurobranch.bundlenet_sim.dataset import BEHAVIOR_NAMES  # noqa: E402
from neurobranch.bundlenet_sim.premotor import PremotorParameters  # noqa: E402
from neurobranch.bundlenet_sim.workbench import ViennaWorkbench  # noqa: E402

st.set_page_config(page_title="Vienna Neuroinformatics Studio", page_icon="🧠", layout="wide")

SETUP_COMMAND = "python scripts/fetch_vienna_artifacts.py"

COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#EECA3B",
    "#B279A2",
    "#FF9DA6",
]


@st.cache_resource
def get_workbench() -> ViennaWorkbench:
    return ViennaWorkbench(ROOT)


def render_missing_artifact(error: FileNotFoundError) -> None:
    st.error(f"Missing local artifact: {error}")
    st.markdown(
        "This workbench uses third-party research artifacts that are not committed to Git. "
        "Prepare them from the checked-out reference repositories:"
    )
    st.code(SETUP_COMMAND, language="bash")
    st.caption(
        "The fetch script records SHA-256 hashes in data/processed/vienna_artifacts_manifest.json. "
        "The upstream licenses are MIT (Celegans_premotor), GPL-3.0 (NC-MCM), and unspecified "
        "(BunDLe-Net); review them before redistribution."
    )


def behavior_trace(x: np.ndarray, y: np.ndarray, z: np.ndarray, labels: np.ndarray, name: str) -> go.Scatter3d:
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker={
            "size": 2.5,
            "color": [COLORS[int(b) % len(COLORS)] for b in labels],
            "opacity": 0.65,
        },
        name=name,
        hovertemplate="%{x:.2f}, %{y:.2f}, %{z:.2f}<extra></extra>",
    )


def tier1_tab(wb: ViennaWorkbench) -> None:
    st.subheader("Tier 1 — BunDLe-Net latent-manifold simulator")
    st.caption(
        "Published 3-D BunDLe-Net embeddings; a data-driven manifold walk and a quadratic "
        "forecast are shown separately. This is not a biophysical or causal model."
    )

    worm_id = st.selectbox(
        "Worm",
        options=[t.worm_id for t in wb.trajectories],
        format_func=lambda w: f"worm {w}",
        key="t1_worm",
    )
    traj = next(t for t in wb.trajectories if t.worm_id == worm_id)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        start = st.slider("Start frame", 0, len(traj.latent) - 2, 0, key="t1_start")
    with c2:
        horizon = st.slider("Horizon (frames)", 10, 300, 80, step=10, key="t1_horizon")
    with c3:
        mode = st.selectbox(
            "Dynamics",
            options=["manifold_walk", "switching", "quadratic"],
            format_func=lambda m: {
                "manifold_walk": "Manifold walk (descriptive, stable)",
                "switching": "Switching linear + behaviour Markov",
                "quadratic": "Quadratic forecast (one-step strong, unstable)",
            }[m],
            key="t1_mode",
        )
    with c4:
        seed = st.number_input("Seed", 0, 10_000, 0, key="t1_seed")

    c5, c6 = st.columns(2)
    with c5:
        temperature = st.slider("Walk temperature", 0.05, 2.0, 0.4, 0.05, key="t1_temp")
    with c6:
        target = st.selectbox(
            "Optional behaviour target",
            options=[-1, *range(len(BEHAVIOR_NAMES))],
            format_func=lambda i: "none" if i < 0 else BEHAVIOR_NAMES[i],
            key="t1_target",
        )

    if st.button("Simulate", type="primary", key="t1_run"):
        sim, proba = wb.tier1_simulate(
            worm_id=worm_id,
            start=start,
            horizon=horizon,
            mode=mode,
            seed=int(seed),
            temperature=temperature,
            target_behavior=None if target < 0 else int(target),
        )
        st.session_state["t1_sim"] = sim
        st.session_state["t1_proba"] = proba
        st.session_state["t1_meta"] = (worm_id, start, horizon, mode)

    sim = st.session_state.get("t1_sim")
    proba = st.session_state.get("t1_proba")
    meta = st.session_state.get("t1_meta")
    if meta is not None and (meta[0] != worm_id or meta[1] != start):
        sim = None
    if sim is None:
        sim, proba = wb.tier1_simulate(
            worm_id=worm_id,
            start=start,
            horizon=min(horizon, 80),
            mode="manifold_walk",
            seed=int(seed),
            temperature=temperature,
            target_behavior=None,
        )

    fig = go.Figure()
    fig.add_trace(
        behavior_trace(
            traj.latent[:, 0],
            traj.latent[:, 1],
            traj.latent[:, 2],
            traj.behavior,
            f"observed worm {worm_id}",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=sim[:, 0],
            y=sim[:, 1],
            z=sim[:, 2],
            mode="lines+markers",
            line={"color": "#111111", "width": 5},
            marker={"size": 2, "color": "#111111"},
            name="simulated",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[traj.latent[start, 0]],
            y=[traj.latent[start, 1]],
            z=[traj.latent[start, 2]],
            mode="markers",
            marker={"size": 7, "color": "#00CC96", "symbol": "diamond"},
            name="start",
        )
    )
    fig.update_layout(
        height=560,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        scene={"xaxis_title": "latent 1", "yaxis_title": "latent 2", "zaxis_title": "latent 3"},
        legend={"orientation": "h", "y": 1.05},
    )
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("**Simulated behaviour probabilities**")
        heat = go.Figure(
            data=go.Heatmap(
                z=proba.T,
                x=np.arange(len(sim)),
                y=list(BEHAVIOR_NAMES),
                colorscale="Viridis",
                colorbar={"title": "p"},
            )
        )
        heat.update_layout(height=320, margin={"l": 0, "r": 0, "t": 10, "b": 0})
        st.plotly_chart(heat, width="stretch")
    with right:
        st.markdown("**Held-out-worm metrics**")
        metrics = wb.metrics()
        folds = metrics.get("folds", [])
        if folds:
            st.metric("Mean one-step latent R²", f"{metrics.get('mean_one_step_r2', float('nan')):.3f}")
            st.metric(
                "Behaviour read-out (true latent)",
                f"{metrics.get('mean_behavior_accuracy_true', float('nan')):.1%}",
            )
            st.metric(
                "Behaviour read-out (simulated quadratic)",
                f"{metrics.get('mean_behavior_accuracy_simulated', float('nan')):.1%}",
            )
            st.caption(
                "Metrics are leave-one-worm-out on 5 worms. The quadratic model has strong "
                "one-step skill but unstable free-run; the manifold walk is stable because it "
                "samples observed transitions."
            )
        else:
            st.info("Run `python -m neurobranch.bundlenet_sim.train` to create metrics.")

        benchmark = wb.tier1_benchmark()
        if benchmark:
            st.markdown("**Open-loop benchmark (held-out worm)**")
            rows = []
            for item in benchmark.get("models", []):
                coverage = item.get("interval_coverage_90_h10", {}).get("mean", float("nan"))
                rows.append(
                    {
                        "model": item["model"],
                        "latent MSE h10": round(item["latent_mse_h10"]["mean"], 3),
                        "CRPS h10": round(item.get("latent_crps_h10", {}).get("mean", float("nan")), 3),
                        "90% coverage": "n/a" if not np.isfinite(coverage) else f"{coverage:.1%}",
                        "behaviour acc h10": f"{item['behavior_accuracy_h10']['mean']:.1%}",
                        "behaviour acc h25": f"{item['behavior_accuracy_h25']['mean']:.1%}",
                        "behaviour NLL h10": round(item.get("behavior_nll_h10", {}).get("mean", float("nan")), 3),
                        "transition JS h25": round(item["behavior_transition_js"]["mean"], 3),
                    }
                )
            st.dataframe(rows, hide_index=True, width="stretch")
            st.caption(
                "Open-loop forecast from 75-frame-spaced starts, 25-frame horizon, 16 seeds. "
                "Coverage is the marginal 90% ensemble interval; CRPS and NLL are proper scores. "
                "Lower MSE/CRPS/NLL/JS is better. This is a descriptive forecast comparison, not mechanism."
            )
            if "horizon_curve" in benchmark:
                curve = benchmark["horizon_curve"]
                fig = go.Figure()
                for model_name, values in curve.items():
                    xs = [int(h) for h in values]
                    ys = [values[h] for h in values]
                    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=model_name))
                fig.update_layout(
                    height=260,
                    margin={"l": 0, "r": 0, "t": 10, "b": 0},
                    xaxis_title="forecast horizon (frames)",
                    yaxis_title="latent MSE",
                )
                st.markdown("**Error growth by horizon**")
                st.plotly_chart(fig, width="stretch")

            st.markdown("**Manifold-walk predictive spread (empirical, not posterior)**")
            ensemble = wb.tier1_ensemble(
                worm_id, start, min(horizon, 120), seeds=32, temperature=temperature
            )
            mean = ensemble.mean(axis=0)
            low, high = np.quantile(ensemble, [0.05, 0.95], axis=0)
            spread = go.Figure()
            for dim, color in enumerate(("#4C78A8", "#F58518", "#54A24B")):
                spread.add_trace(
                    go.Scatter(
                        x=np.arange(len(mean)),
                        y=mean[:, dim],
                        mode="lines",
                        line={"color": color},
                        name=f"latent {dim + 1} mean",
                    )
                )
                spread.add_trace(
                    go.Scatter(
                        x=np.arange(len(mean)),
                        y=high[:, dim],
                        mode="lines",
                        line={"width": 0},
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                spread.add_trace(
                    go.Scatter(
                        x=np.arange(len(mean)),
                        y=low[:, dim],
                        mode="lines",
                        line={"width": 0},
                        fill="tonexty",
                        fillcolor=color,
                        opacity=0.12,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            spread.update_layout(
                height=280,
                margin={"l": 0, "r": 0, "t": 10, "b": 0},
                xaxis_title="steps",
                yaxis_title="latent value",
            )
            st.plotly_chart(spread, width="stretch")
            st.caption(
                "32 stochastic manifold-walk samples from the selected start. The band is an "
                "empirical 5–95% interval, not a posterior over mechanisms."
            )

    with st.expander("What this tier does and does not claim"):
        st.markdown(
            """
- **Does:** encode a 15-frame whole-brain window into the published 3-D BunDLe-Net manifold;
  decode behaviour; sample plausible latent trajectories on the observed manifold.
- **Does not:** reconstruct individual neurons, simulate a circuit, establish mechanism, or
  support causal interventions. A free-running quadratic forecast diverges; it is shown for
  comparison, not as a validated simulator.
"""
        )


def tier2_tab(wb: ViennaWorkbench) -> None:
    st.subheader("Tier 2 — NC-MCM causal-state workbench")
    st.caption(
        "Precomputed NC-MCM models: behaviour probabilities → k-means cognitive states → "
        "Markov tests. The UI exposes the fitted descriptive model; it does not identify do()."
    )
    worm_id = st.selectbox(
        "Worm",
        options=[t.worm_id for t in wb.trajectories],
        format_func=lambda w: f"worm {w}",
        key="t2_worm",
    )
    model = wb.ncmcm(worm_id)
    k = st.slider("Number of cognitive states K", 2, model.clusters.shape[1], 6, key="t2_k")
    p_value = model.markov_p(k)
    seq = model.cluster_sequence(k)
    P = model.transition_matrix(k)
    behavior_dist = model.behavior_distribution_by_cluster(k)

    c1, c2, c3 = st.columns(3)
    c1.metric("Markov-chain p-value", f"{p_value:.3f}")
    c2.metric("Frames", f"{len(seq):,}")
    c3.metric("States visited", f"{len(np.unique(seq))}/{k}")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Cluster transition matrix**")
        fig = go.Figure(
            data=go.Heatmap(
                z=P,
                x=[f"S{i + 1}" for i in range(k)],
                y=[f"S{i + 1}" for i in range(k)],
                colorscale="Blues",
                colorbar={"title": "p(next)"},
            )
        )
        fig.update_layout(height=430, margin={"l": 0, "r": 0, "t": 10, "b": 0})
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("**Behaviour composition of each state**")
        fig = go.Figure()
        for bi, name in enumerate(BEHAVIOR_NAMES):
            fig.add_trace(go.Bar(x=[f"S{i + 1}" for i in range(k)], y=behavior_dist[:, bi], name=name))
        fig.update_layout(
            barmode="stack",
            height=430,
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            legend={"orientation": "h", "y": -0.25},
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("**State sequence**")
    line = go.Figure(
        go.Scatter(
            x=np.arange(len(seq)),
            y=seq,
            mode="lines",
            line={"color": "#4C78A8", "width": 1},
            name="cluster",
        )
    )
    line.update_layout(height=220, margin={"l": 0, "r": 0, "t": 10, "b": 0}, yaxis_title="state")
    st.plotly_chart(line, width="stretch")

    st.markdown("**Model-based intervention trajectory (Markov assumption)**")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        state = st.selectbox("Force state at t=0", [f"S{i + 1}" for i in range(k)], key="t2_state")
        idx = int(state[1:]) - 1
    with c2:
        horizon = st.slider("Horizon", 1, 50, 20, key="t2_horizon")
    with c3:
        st.caption(
            "Propagate the forced state through the fitted transition matrix. "
            "This is a counterfactual under the Markov assumption, not an experimentally identified effect."
        )

    trajectory = model.intervention_horizon(k, idx + 1, horizon)
    left, right = st.columns([1.2, 1])
    with left:
        fig = go.Figure()
        for bi, name in enumerate(BEHAVIOR_NAMES):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(horizon + 1),
                    y=trajectory["behavior_distribution"][:, bi],
                    mode="lines",
                    name=name,
                    line={"color": COLORS[bi]},
                )
            )
        fig.update_layout(
            height=330,
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            xaxis_title="steps after do(state)",
            yaxis_title="p(behaviour)",
            legend={"orientation": "h", "y": -0.35},
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = go.Figure(
            data=go.Heatmap(
                z=trajectory["state_distribution"].T,
                x=np.arange(horizon + 1),
                y=[f"S{i + 1}" for i in range(k)],
                colorscale="Blues",
                colorbar={"title": "p(state)"},
            )
        )
        fig.update_layout(
            height=330,
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            xaxis_title="steps after do(state)",
        )
        st.plotly_chart(fig, width="stretch")

    with st.expander("Transition uncertainty (Dirichlet bootstrap) and cross-worm variability"):
        c1, c2 = st.columns(2)
        with c1:
            boot = model.bootstrap_transition(k, n_boot=200, seed=0)
            rows = []
            for a in range(k):
                for b in range(k):
                    if P[a, b] > 0.05 or a == idx:
                        rows.append(
                            {
                                "from": f"S{a + 1}",
                                "to": f"S{b + 1}",
                                "mean": round(float(boot["mean"][a, b]), 3),
                                "95% CI": f"{boot['low'][a, b]:.3f}–{boot['high'][a, b]:.3f}",
                            }
                        )
            st.dataframe(rows, hide_index=True, width="stretch")
        with c2:
            other_rows = []
            for other in wb.trajectories:
                if other.worm_id == worm_id:
                    continue
                other_model = wb.ncmcm(other.worm_id)
                js = model.cross_worm_js(other_model)
                other_rows.append(
                    {
                        "other worm": f"worm {other.worm_id}",
                        "behaviour JS": round(js["behavior_js"], 3),
                        "behaviour transition JS": round(js["behavior_transition_js"], 3),
                    }
                )
            st.dataframe(other_rows, hide_index=True, width="stretch")
            st.caption("Only aligned behaviour labels are compared; cluster IDs are arbitrary.")

    audit = wb.tier2_audit(int(k))
    with st.expander("Assumption audit: is the Markov counterfactual supported?"):
        st.caption(
            "These tests audit the NC-MCM assumptions. A failure does not invalidate the "
            "descriptive model; it means the model-based `do()` should not be read as an "
            "identified causal effect."
        )
        rows = []
        for test_name, result in audit.get("tests", {}).items():
            rows.append(
                {
                    "test": test_name.replace("_", " "),
                    "verdict": result.get("verdict", "n/a"),
                    "detail": {
                        "markov_order": lambda r: f"p_MC={r.get('p_monte_carlo', float('nan')):.3f}, LR={r.get('lr', float('nan')):.1f}",
                        "stationarity": lambda r: f"p_MC={r.get('p_monte_carlo', float('nan')):.3f}, JS={r.get('js_first_second_half', float('nan')):.3f}",
                        "behavior_transport": lambda r: f"held-out JS={r.get('heldout_js', float('nan')):.3f}, split-half={r.get('split_half_js', float('nan')):.3f}",
                        "state_behavior_mi": lambda r: f"MI={r.get('mutual_information', float('nan')):.3f}, p={r.get('p_permutation', float('nan')):.3f}",
                        "k_sensitivity": lambda r: f"max TV={r.get('max_tv_vs_reference', float('nan')):.3f}",
                    }.get(test_name, lambda r: "")(result),
                }
            )
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(" | ".join(audit.get("caveats", [])))

    st.warning(
        "This is a model-based counterfactual, not an identified causal effect. "
        "A real do(state) claim requires the NC-MCM assumptions plus experimental intervention."
    )


def tier3_clamped_tab(wb: ViennaWorkbench) -> None:
    st.subheader("Tier 3 — Published clamped-core simulation")
    st.caption(
        "15 intrinsic neurons are integrated with the published cubic rate ODE; 69 signal "
        "neurons are clamped to an Atanas recording. This is the configuration used in the "
        "Morrison & Young paper, not the 205-node free-run."
    )
    catalog = wb.clamped_catalog()
    if not catalog:
        st.info("No Atanas recording with all 15 intrinsic neurons was found.")
        return
    ids = [str(item["recording_id"]) for item in catalog]
    default = "2023-01-23-15" if "2023-01-23-15" in ids else ids[0]
    recording_id = st.selectbox(
        "Recording",
        ids,
        index=ids.index(default),
        format_func=lambda r: next(
            f"{r} — {item['n_core']} core / {item['n_signal']} signals"
            for item in catalog
            if item["recording_id"] == r
        ),
        key="t3c_recording",
    )
    info = next(item for item in catalog if item["recording_id"] == recording_id)
    st.caption(
        f"{info['n_core']} core neurons · {info['n_signal']} clamped signals · "
        f"{info['n_frames']} frames · replacements: {info['replacement_map'] or 'none'}"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        beta = st.slider("Gap-junction gain β", 0.0, 20.0, 10.0, 0.5, key="t3c_beta")
    with c2:
        gamma = st.slider("Synaptic gain γ", 0.0, 3.0, 1.4, 0.1, key="t3c_gamma")
    with c3:
        tau = st.slider("Time constant τ", 0.05, 1.0, 0.2, 0.05, key="t3c_tau")
    with c4:
        pulse_enabled = st.checkbox("Add signal pulse", value=False, key="t3c_pulse_on")

    pulse = None
    if pulse_enabled:
        p1, p2, p3 = st.columns(3)
        with p1:
            pulse_neuron = st.selectbox(
                "Signal neuron", list(wb.clamped_recording(recording_id).signal_names), key="t3c_pulse_neuron"
            )
        with p2:
            pulse_amp = st.slider("Pulse amplitude", -10.0, 10.0, 3.0, 0.5, key="t3c_pulse_amp")
        with p3:
            pulse_start = st.slider("Pulse start (s)", 0.0, 10.0, 1.0, 0.1, key="t3c_pulse_start")
        pulse = SignalPulse(pulse_neuron, float(pulse_amp), float(pulse_start), 2.0)

    if st.button("Run clamped core", type="primary", key="t3c_run"):
        params = ClampedParameters(
            beta=float(beta),
            gamma=float(gamma),
            tau=float(tau),
            signal_pulses=(pulse,) if pulse is not None else (),
        )
        try:
            with st.spinner("Integrating 15 intrinsic neurons with 69 clamped signals..."):
                result = wb.tier3_clamped(recording_id, params)
        except (ClampedSimulationError, ValueError, FileNotFoundError) as error:
            st.error(str(error))
            return
        st.session_state["t3c_result"] = result
        st.session_state["t3c_meta"] = recording_id

    result = st.session_state.get("t3c_result")
    if result is None or st.session_state.get("t3c_meta") != recording_id:
        st.info("Press **Run clamped core** to integrate the published configuration.")
        return

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Status", result.status)
    m2.metric("max |u|", f"{result.metrics['max_abs']:.2f}")
    m3.metric("State agreement", f"{result.metrics['state_agreement']:.1%}")
    m4.metric("Behaviour JS", f"{result.metrics['behavior_distribution_js']:.3f}")
    m5.metric("Method", result.method)

    names = result.intrinsic_names
    selected = st.selectbox(
        "Intrinsic neuron", list(names), index=names.index("AVBL") if "AVBL" in names else 0,
        key="t3c_neuron",
    )
    idx = names.index(selected)
    trace = go.Figure()
    trace.add_trace(
        go.Scatter(
            x=result.time,
            y=result.truth_intrinsic[idx],
            mode="lines",
            name="recorded",
            line={"color": "#4C78A8", "width": 1.5},
        )
    )
    trace.add_trace(
        go.Scatter(
            x=result.time,
            y=result.intrinsic[idx],
            mode="lines",
            name="simulated",
            line={"color": "#E45756", "width": 1.5},
        )
    )
    trace.update_layout(
        height=330,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        xaxis_title="time (s)",
        yaxis_title="activity",
        legend={"orientation": "h", "y": 1.05},
    )
    st.plotly_chart(trace, width="stretch")

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("**Intrinsic residual (simulated − recorded)**")
        residual = result.intrinsic - result.truth_intrinsic
        heat = go.Figure(
            data=go.Heatmap(
                z=residual,
                x=np.round(result.time, 2),
                y=list(names),
                colorscale="RdBu",
                zmid=0,
                colorbar={"title": "Δu"},
            )
        )
        heat.update_layout(height=330, margin={"l": 0, "r": 0, "t": 10, "b": 0}, xaxis_title="time (s)")
        st.plotly_chart(heat, width="stretch")
    with right:
        st.markdown("**Behaviour readout**")
        beh = go.Figure()
        beh.add_trace(
            go.Scatter(
                x=result.time,
                y=result.behavior_truth,
                mode="lines",
                name="recorded",
                line={"color": "#4C78A8", "width": 1},
            )
        )
        beh.add_trace(
            go.Scatter(
                x=result.time,
                y=result.behavior_sim,
                mode="lines",
                name="simulated",
                line={"color": "#E45756", "width": 1},
            )
        )
        beh.update_layout(
            height=180,
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            yaxis={"tickmode": "array", "tickvals": [-1, 0, 1], "ticktext": ["reverse", "quiet", "forward"]},
            legend={"orientation": "h", "y": -0.5},
        )
        st.plotly_chart(beh, width="stretch")
        st.caption(
            "Neural proxy readout: mean(AVBL, AVBR, RIBL, RIBR, RID) − "
            "mean(AVAL, AVAR, RIML, RIMR, AVEL, AVER), threshold ±0.5. It is computed "
            "from the recorded traces, not from independent video behaviour labels."
        )
        st.markdown("**Validation report**")
        report = wb.tier3_clamped_validation()
        if report:
            agg = report.get("aggregate", {})
            st.metric("Mean state agreement", f"{agg.get('mean_state_agreement', float('nan')):.1%}")
            st.metric("Mean behaviour JS", f"{agg.get('mean_behavior_distribution_js', float('nan')):.3f}")
            st.caption(report.get("claim_level", ""))
        else:
            st.caption("Run `python scripts/build_tier3_validation.py` to create the report.")
        if recording_id == "2023-01-23-15":
            reference = wb.tier3_reference()
            if reference:
                summary = reference.get("summary", {})
                st.caption(
                    "Julia reference: mean trace r = "
                    f"{summary.get('mean_pearson_r', float('nan')):.7f}, "
                    f"max |Δ| = {summary.get('max_abs_difference', float('nan')):.4f}. "
                    "This checks the port, not the biology."
                )
        parity = wb.tier3_reference_all()
        if parity:
            rows = []
            for rid, item in parity.get("recordings", {}).items():
                summary = item.get("summary", {})
                rows.append(
                    {
                        "recording": rid,
                        "core / signals": f"{item.get('core_neurons', '?')} / "
                        f"{item.get('core_neurons', 0) - 15}",
                        "trace r": f"{summary.get('mean_pearson_r', float('nan')):.7f}",
                        "max |Δ|": f"{summary.get('max_abs_difference', float('nan')):.4f}",
                        "behaviour vs recording": f"{summary.get('behavior_agreement', float('nan')):.1%}",
                        "behaviour Python vs Julia": f"{summary.get('behavior_port_agreement', float('nan')):.2%}",
                    }
                )
            st.markdown("**Julia port parity — all validation recordings**")
            st.dataframe(rows, hide_index=True, width="stretch")
            agg = parity.get("aggregate", {})
            st.caption(
                f"{agg.get('n_recordings', '?')} recordings; mean trace r="
                f"{agg.get('mean_pearson_r', float('nan')):.7f}, max |Δ|="
                f"{agg.get('max_abs_difference', float('nan')):.4f}, behaviour-label parity="
                f"{agg.get('mean_behavior_port_agreement', float('nan')):.2%}. "
                "Julia reproduces the same behavioural agreement with the recording, so "
                "the remaining gap is a model/measurement limitation, not a port bug."
            )
        behaviour = wb.tier3_behavior_validation()
        if behaviour:
            def _num(value: object, digits: int = 3) -> str:
                if value is None:
                    return "n/a"
                return f"{float(value):.{digits}f}"

            def _pct(value: object) -> str:
                if value is None:
                    return "n/a"
                return f"{float(value):.1%}"

            def _ci(value: object) -> str:
                if not value or len(value) < 2:  # type: ignore[arg-type]
                    return "n/a"
                return f"{float(value[0]):.3f}–{float(value[1]):.3f}"  # type: ignore[index]

            st.markdown("**External reversal-event check — baseline controlled**")
            beh_rows = []
            for item in behaviour.get("recordings", []):
                readout = item.get("readout", {})
                model = item.get("model", {})
                baselines = item.get("baselines", {})
                persistence = baselines.get("persistence", {})
                beh_rows.append(
                    {
                        "recording": item.get("recording"),
                        "readout AUC [95% CI]": f"{_num(readout.get('auc'))} [{_ci(readout.get('auc_ci'))}]",
                        "readout agreement": _pct(readout.get("agreement")),
                        "model lag-0 AUC": _num(model.get("lag0_auc")),
                        "model retrospective AUC [95% CI]": f"{_num(model.get('retrospective_auc'))} "
                        f"[{_ci(model.get('retrospective_auc_ci'))}]",
                        "selected shift": f"{int(model.get('selected_shift_frames', 0)):+d} fr / "
                        f"{_num(model.get('selected_shift_ms'), 0)} ms",
                        "model prospective AUC": _num(model.get("prospective_auc")),
                        "persistence AUC": _num(persistence.get("auc")),
                    }
                )
            st.dataframe(beh_rows, hide_index=True, width="stretch")
            agg_beh = behaviour.get("aggregate", {})
            st.caption(
                "Neural readout is the paper's 5-forward/6-reverse proxy. "
                f"Mean readout AUC={_num(agg_beh.get('mean_readout_auc'))}; "
                f"model lag-0 AUC={_num(agg_beh.get('mean_model_lag0_auc'))}; "
                f"retrospective AUC={_num(agg_beh.get('mean_model_retrospective_auc'))}; "
                f"prospective AUC={_num(agg_beh.get('mean_model_prospective_auc'))} "
                f"(exploratory max over negative lags={_num(agg_beh.get('max_exploratory_prospective_auc'))}); "
                f"persistence AUC={_num(agg_beh.get('mean_persistence_auc'))}. "
                "A positive selected shift means the model response follows the event; this is "
                "retrospective alignment and is not a forecast. The model does not beat the "
                "one-frame persistence baseline."
            )

            with st.expander("Lag curve, uncertainty, controls and event-level checks"):
                lag_fig = go.Figure()
                for item in behaviour.get("recordings", []):
                    curve = [point for point in item.get("lag_curve", []) if point.get("auc") is not None]
                    if not curve:
                        continue
                    lag_fig.add_trace(
                        go.Scatter(
                            x=[point["shift_frames"] for point in curve],
                            y=[point["auc"] for point in curve],
                            mode="lines+markers",
                            name=str(item.get("recording")),
                        )
                    )
                lag_fig.add_vline(x=0, line_dash="dash", line_color="grey")
                lag_fig.update_layout(
                    height=330,
                    margin={"l": 0, "r": 0, "t": 10, "b": 0},
                    xaxis_title="score shift (frames; negative = model leads, positive = model lags)",
                    yaxis_title="AUC vs reversal events",
                )
                st.plotly_chart(lag_fig, width="stretch")

                ci_rows = []
                for item in behaviour.get("recordings", []):
                    readout = item.get("readout", {})
                    model = item.get("model", {})
                    ci_rows.append(
                        {
                            "recording": item.get("recording"),
                            "readout AUC [95% CI]": f"{_num(readout.get('auc'))} [{_ci(readout.get('auc_ci'))}]",
                            "readout agreement [95% CI]": f"{_pct(readout.get('agreement'))} "
                            f"[{_ci(readout.get('agreement_ci'))}]",
                            "retrospective AUC [95% CI]": f"{_num(model.get('retrospective_auc'))} "
                            f"[{_ci(model.get('retrospective_auc_ci'))}]",
                            "exploratory max negative-lag AUC": _num(
                                item.get("prospective_max_negative_auc")
                            ),
                        }
                    )

                control_rows = []
                event_rows = []
                sensitivity_rows = []
                for item in behaviour.get("recordings", []):
                    controls = item.get("controls", {})
                    beta = controls.get("beta0", {}) if isinstance(controls, dict) else {}
                    zero = controls.get("zero_signals", {}) if isinstance(controls, dict) else {}
                    control_rows.append(
                        {
                            "recording": item.get("recording"),
                            "beta=0 lag-0 AUC": _num(beta.get("lag0_auc")),
                            "beta=0 aligned AUC": _num(beta.get("aligned_auc")),
                            "zero signals lag-0 AUC": _num(zero.get("lag0_auc")),
                            "zero signals aligned AUC": _num(zero.get("aligned_auc")),
                        }
                    )
                    event_level = item.get("event_level", {})
                    readout_ev = event_level.get("readout", {})
                    model_ev = event_level.get("model_retrospective", {})
                    persistence_ev = event_level.get("persistence", {})
                    event_rows.append(
                        {
                            "recording": item.get("recording"),
                            "readout event recall": _pct(readout_ev.get("event_recall")),
                            "model event recall": _pct(model_ev.get("event_recall")),
                            "persistence event recall": _pct(persistence_ev.get("event_recall")),
                            "model false alarms / min": _num(
                                model_ev.get("false_alarm_episodes_per_min"), 2
                            ),
                        }
                    )
                    sensitivity = item.get("alignment_sensitivity", {})
                    sensitivity_rows.append(
                        {
                            "recording": item.get("recording"),
                            "lag-0 AUC range, offset 3–7": "–".join(
                                _num(v) for v in (sensitivity.get("model_lag0_auc_range") or [])
                            )
                            or "n/a",
                            "closed-interval lag-0 AUC": _num(
                                sensitivity.get("closed_interval_model_lag0_auc")
                            ),
                        }
                    )
                st.markdown("**Block-bootstrap uncertainty**")
                st.dataframe(ci_rows, hide_index=True, width="stretch")
                st.markdown("**Negative controls**")
                st.dataframe(control_rows, hide_index=True, width="stretch")
                st.markdown("**Event-level check**")
                st.dataframe(event_rows, hide_index=True, width="stretch")
                st.markdown("**Alignment sensitivity**")
                st.dataframe(sensitivity_rows, hide_index=True, width="stretch")
                st.caption(
                    "Event-level recall uses a ±10-frame tolerance. Controls are simulation "
                    "manipulations, not experimental interventions. Offset/interval conventions "
                    "are inferred from the release; the range is shown rather than selecting the "
                    "best alignment."
                )
                st.download_button(
                    "Download external validation JSON (v2)",
                    data=json.dumps(behaviour, indent=2),
                    file_name="vienna_tier3_behavior_validation_v2.json",
                    mime="application/json",
                    key="t3b_download_v2",
                )

    buffer = io.StringIO()
    np.savetxt(
        buffer,
        np.column_stack([result.time, result.truth_intrinsic.T, result.intrinsic.T]),
        delimiter=",",
        header="time_s," + ",".join(f"recorded_{n}" for n in names) + "," + ",".join(f"simulated_{n}" for n in names),
        comments="",
    )
    st.download_button(
        "Download clamped intrinsic traces (CSV)",
        data=buffer.getvalue(),
        file_name=f"vienna_clamped_{recording_id}.csv",
        mime="text/csv",
        key="t3c_download",
    )

    if result.status == "unstable":
        st.error(
            "Numerical warning: the clamped integration exceeded the configured clip. "
            "The trajectory is shown, but it must not be treated as a validated replay."
        )
    st.warning(
        "This is a published-parameter replay/sandbox with independently recovered weights. "
        "The remaining disagreement (about 38–46% of frames across the three validation recordings) "
        "is evidence against a pointwise digital-twin claim, not evidence for a mechanism. It is "
        "not causally validated."
    )


def tier3_tab(wb: ViennaWorkbench) -> None:
    st.subheader("Tier 3 — Premotor biophysical simulation")
    mode = st.radio(
        "Simulation mode",
        ["Published clamped core (15 intrinsic + 69 signals)", "205-node free-run sandbox"],
        horizontal=True,
        key="t3_mode",
    )
    if mode.startswith("Published"):
        tier3_clamped_tab(wb)
    else:
        tier3_sandbox_tab(wb)


def tier3_sandbox_tab(wb: ViennaWorkbench) -> None:
    st.subheader("Tier 3 — 205-node free-run sandbox")
    st.caption(
        "205-node fitted synaptic matrix from Celegans_premotor, integrated with the published "
        "cubic rate ODE. Neuron names and gap-junction matrix W were recovered from the release's "
        "own connectome and verified against the 913 non-zero A entries."
    )
    model = wb.premotor
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        duration = st.slider("Duration", 2.0, 40.0, 12.0, 1.0, key="t3_dur")
    with c2:
        gamma = st.slider("Synaptic gain γ", 0.0, 3.0, 1.4, 0.1, key="t3_gamma")
    with c3:
        beta = st.slider(
            "Gap-junction gain β",
            0.0,
            10.0,
            0.0,
            0.25,
            key="t3_beta",
            help=(
                "W is recovered electrical-junction counts. Full 205-node free-run becomes unstable "
                "near β≈5; the published simulation clamps 69 signal neurons to recordings."
            ),
        )
    with c4:
        tau = st.slider("Time constant τ", 0.05, 1.0, 0.2, 0.05, key="t3_tau")

    c5, c6, c7 = st.columns(3)
    node_options = list(range(model.n_nodes))
    with c5:
        stim_node = st.selectbox(
            "Stimulated node",
            node_options,
            index=min(len(node_options) - 1, 0),
            format_func=model.node_label,
            key="t3_stim",
        )
    with c6:
        stim_amp = st.slider("Stimulus amplitude", 0.0, 20.0, 5.0, 0.5, key="t3_amp")
    with c7:
        ablate_text = st.text_input(
            "Ablate nodes (names or indices, comma-separated)", "", key="t3_ablate"
        )

    ablate: tuple[int, ...] = ()
    if ablate_text.strip():
        try:
            parsed: list[int] = []
            for token in ablate_text.split(","):
                token = token.strip()
                if not token:
                    continue
                parsed.append(int(token) if token.isdigit() else model.node_index(token))
            ablate = tuple(parsed)
        except (ValueError, IndexError):
            st.error("Ablation list must contain valid neuron names or integer indices.")

    if st.button("Run ODE", type="primary", key="t3_run"):
        params = PremotorParameters(
            gamma=gamma,
            beta=beta,
            tau=tau,
            duration=duration,
            n_samples=300,
            stim_node=int(stim_node),
            stim_amplitude=float(stim_amp),
            ablate_nodes=ablate,
        )
        with st.spinner("Integrating 205-node ODE..."):
            t, y = wb.tier3_simulate(params, seed=0)
        st.session_state["t3_result"] = (t, y)
        st.session_state["t3_params"] = params

    result = st.session_state.get("t3_result")
    if result is None:
        params = PremotorParameters(gamma=gamma, beta=beta, tau=tau, duration=duration, n_samples=300)
        with st.spinner("Integrating 205-node ODE..."):
            t, y = wb.tier3_simulate(params, seed=0)
    else:
        t, y = result

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("**Population activity**")
        fig = go.Figure(
            data=go.Heatmap(
                z=y,
                x=np.round(t, 2),
                y=np.arange(model.n_nodes),
                colorscale="RdBu",
                zmid=0,
                colorbar={"title": "u"},
            )
        )
        fig.update_layout(height=430, margin={"l": 0, "r": 0, "t": 10, "b": 0}, xaxis_title="time")
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("**Selected node**")
        node = st.selectbox(
            "Node trace",
            node_options,
            index=min(len(node_options) - 1, 0),
            format_func=model.node_label,
            key="t3_node",
        )
        trace = go.Figure(
            go.Scatter(x=t, y=y[int(node)], mode="lines", line={"color": "#E45756"})
        )
        trace.update_layout(height=210, margin={"l": 0, "r": 0, "t": 10, "b": 0}, yaxis_title="u")
        st.plotly_chart(trace, width="stretch")
        st.markdown("**Synaptic matrix (nonzero support)**")
        support = (model.A != 0).astype(float)
        mat = go.Figure(data=go.Heatmap(z=support, colorscale="Greys", showscale=False))
        mat.update_layout(height=210, margin={"l": 0, "r": 0, "t": 10, "b": 0})
        st.plotly_chart(mat, width="stretch")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Top nodes by fitted input strength**")
        order = np.argsort(-model.node_strength())[:12]
        st.dataframe(
            [
                {
                    "neuron": model.node_label(int(i)),
                    "|d| + Σ|A|": round(float(model.node_strength()[i]), 2),
                    "gap degree": int((model.W[i] != 0).sum()) if model.W is not None else 0,
                }
                for i in order
            ],
            hide_index=True,
            width="stretch",
        )
    with right:
        st.markdown("**Export current simulation**")
        buffer = io.StringIO()
        np.savetxt(
            buffer,
            np.column_stack([t, y.T]),
            delimiter=",",
            header="time_s," + ",".join(model.node_label(i) for i in range(model.n_nodes)),
            comments="",
        )
        st.download_button(
            "Download activity CSV",
            data=buffer.getvalue(),
            file_name="vienna_premotor_activity.csv",
            mime="text/csv",
            key="t3_download",
        )
        st.caption(
            "Rows are time samples; columns are named neurons. The export includes the exact "
            "parameter values shown above only in the accompanying report, so record them."
        )

    with st.expander("Stability scan: where does the full 205-node free-run leave the basin?"):
        st.caption(
            "Runs a short 4 s simulation at β = 0…10 with the current γ and τ, no stimulus. "
            "The peak absolute activity is a diagnostic, not a biological measurement."
        )
        if st.button("Run β scan", key="t3_scan_run"):
            betas = np.linspace(0.0, 10.0, 11)
            peaks = []
            with st.spinner("Running 11 short integrations..."):
                for beta_value in betas:
                    _, scan_y = wb.tier3_simulate(
                        PremotorParameters(
                            gamma=float(gamma),
                            beta=float(beta_value),
                            tau=float(tau),
                            duration=4.0,
                            n_samples=80,
                        ),
                        seed=0,
                    )
                    peaks.append(float(np.max(np.abs(scan_y))))
            st.session_state["t3_scan"] = (betas, peaks)
        scan = st.session_state.get("t3_scan")
        if scan is not None:
            scan_betas, scan_peaks = scan
            fig = go.Figure(
                go.Scatter(
                    x=scan_betas,
                    y=scan_peaks,
                    mode="lines+markers",
                    line={"color": "#E45756"},
                )
            )
            fig.update_layout(
                height=280,
                margin={"l": 0, "r": 0, "t": 10, "b": 0},
                xaxis_title="gap-junction gain β",
                yaxis_title="peak |activity|",
                yaxis_type="log",
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "A sharp rise marks the free-run stability boundary at these parameters. "
                "It is not evidence that the biological network is unstable."
            )

    peak = float(np.max(np.abs(y)))
    if peak > 20:
        st.error(
            f"Numerical warning: |activity| reached {peak:.1f}. The full 205-node free-run is "
            "outside the stable regime at this β; this is not the published clamped-signal simulation."
        )
    st.warning(
        "This sandbox uses the published fitted chemical synaptic matrix, recovered W and names, "
        "and the published ODE form. The paper's simulation clamps 69 signal neurons to recorded "
        "activity; use the **Published clamped core** mode for that configuration. The 205-node "
        "free-run is exploratory and is not a validated digital twin."
    )


def provenance_tab(wb: ViennaWorkbench) -> None:
    st.subheader("Provenance, scope, and limitations")
    st.json(wb.provenance())
    st.markdown(
        """
**Primary sources**

- Kumar, Gilra, Meunier, Grosse-Wentrup. *BunDLe-Net: Learning minimal neural manifolds that
  represent only behaviourally relevant dynamics.* bioRxiv 2023, v5 accepted 2026-10-08.
  https://doi.org/10.1101/2023.08.08.551978
- Grosse-Wentrup, Kumar, Meunier, Zimmer. *Neuro-cognitive multilevel causal modeling.*
  PLoS Comput Biol 2024. https://doi.org/10.1371/journal.pcbi.1012674 (PMID 39680605)
- Morrison & Young. *A data-driven biology-based network model reproduces C. elegans premotor
  neural dynamics.* PLoS Comput Biol 2025. https://doi.org/10.1371/journal.pcbi.1013818
  (PMID 41460929; PMC12768384). Public reference code and fitted `avg_model_weights.jld2`.
- Data/code artifacts: NC-MCM `precomputed_results/`; BunDLe-Net repo.

**Status language**

- Tier 1: validated on held-out worms for one-step latent prediction and behaviour read-out;
  free-run generative use is not validated. Intervals are ensemble quantiles, not posteriors.
- Tier 2: descriptive causal-state model; `do()` is not identified by the precomputed artifact.
  The current assumption audit rejects strict first-order Markov and stationarity.
- Tier 3: the published 84/15/69 clamped-core configuration is implemented; the canonical
  recording has 84 core / 15 intrinsic / 69 signal neurons. Fresh Julia 1.11.2 reference runs
  on all three validation recordings match the Python port at mean trace r=0.9999998 and
  max |Δ|=0.0133; behaviour-label parity is 99.98%. Canonical behaviour agreement with the
  readout proxy is 60.2%. The readout reaches mean AUC 0.848 against independent reversal
  events. The clamped model reaches lag-0 AUC 0.633 and a retrospective aligned AUC 0.834 at
  a selected +7.7-frame shift, but prospective AUC is 0.388 and a one-frame persistence
  baseline reaches 0.939. This validates the implementation and the proxy, not forecasting,
  biology or causality; W/names are independently recovered. The 205-node free-run remains
  exploratory. This is not a validated digital twin.
"""
    )


def overview_tab(wb: ViennaWorkbench) -> None:
    st.subheader("What this product is")
    st.markdown(
        """
A lab-facing, source-traceable simulator stack for *C. elegans* whole-brain data. It gives
Denis three different objects, with different claim levels, rather than one overclaimed
"digital twin":

1. **Latent manifold** — generate and forecast behaviourally relevant 3-D BunDLe-Net trajectories.
2. **Causal-state workbench** — inspect NC-MCM cognitive states, Markov structure, uncertainty,
   and model-based intervention trajectories.
3. **Premotor connectome sandbox** — run the published clamped-core ODE (15 intrinsic neurons
   integrated, 69 signals clamped to recording) and, separately, the exploratory 205-node free-run.
"""
    )
    metrics = wb.metrics()
    benchmark = wb.tier1_benchmark()
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Tier 1 one-step R²",
        f"{metrics.get('mean_one_step_r2', float('nan')):.3f}" if metrics else "n/a",
    )
    c2.metric(
        "Behaviour read-out (true latent)",
        f"{metrics.get('mean_behavior_accuracy_true', 0):.1%}" if metrics else "n/a",
    )
    c3.metric("Clamped-core recordings", f"{len(wb.clamped_catalog())}")
    if benchmark:
        st.markdown("**Open-loop forecast, held-out worm**")
        rows = []
        for item in benchmark.get("models", []):
            rows.append(
                {
                    "model": item["model"],
                    "latent MSE h10": round(item["latent_mse_h10"]["mean"], 3),
                    "behaviour acc h10": f"{item['behavior_accuracy_h10']['mean']:.1%}",
                    "behaviour acc h25": f"{item['behavior_accuracy_h25']['mean']:.1%}",
                    "transition JS h25": round(item["behavior_transition_js"]["mean"], 3),
                }
            )
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(
            "These are descriptive forecasts from four training worms to a fifth. "
            "They do not establish mechanism or causality."
        )

    st.markdown("**How to use it in a first meeting**")
    st.markdown(
        """
1. Open **Tier 1**, pick a held-out worm, and compare *manifold walk* with *switching linear*;
   inspect the 90% ensemble coverage before trusting an interval.
2. Open **Tier 2**, choose `K`, inspect the Markov p-value, then open **Assumption audit**.
3. Open **Tier 3**, run the **Published clamped core** at β=10, then compare with β=0 and the
   205-node free-run.
4. Download the **Evidence report** and **Evidence & lab** material before the meeting.

**Claim boundaries**
"""
    )
    st.warning(
        "Tier 1 is a latent-manifold model, not a circuit. Tier 2 is descriptive and its do() is "
        "a Markov counterfactual; the current audit rejects strict first-order Markov and "
        "stationarity, so do not present it as an identified effect. Tier 3 reproduces the "
        "published clamped-core configuration and is checked against fresh Julia runs on all "
        "three validation recordings; that is implementation parity, not biological or causal "
        "validity. None of the three is a validated whole-brain digital twin."
    )


def report_tab(wb: ViennaWorkbench) -> None:
    st.subheader("Evidence report and run bundle")
    st.caption("A portable summary for Denis; the numbers come from the committed artifacts.")
    metrics = wb.metrics()
    benchmark = wb.tier1_benchmark()
    lines = [
        "# Vienna Neuroinformatics Studio — evidence report",
        "",
        "## Scope",
        "Three tiers with separate claim levels: latent manifold, descriptive NC-MCM states, "
        "and a premotor rate-ODE sandbox. None is a validated digital twin.",
        "",
        "## Tier 1 — BunDLe-Net latent manifold",
    ]
    if metrics:
        lines += [
            f"- Mean one-step latent R² (leave-one-worm-out): {metrics.get('mean_one_step_r2', float('nan')):.4f}",
            f"- Behaviour read-out on true latent: {metrics.get('mean_behavior_accuracy_true', float('nan')):.3%}",
            f"- Behaviour read-out on simulated quadratic latent: {metrics.get('mean_behavior_accuracy_simulated', float('nan')):.3%}",
            f"- Majority baseline: {metrics.get('mean_behavior_majority_accuracy', float('nan')):.3%}",
        ]
    if benchmark:
        lines += ["", "| model | latent MSE h10 | behaviour acc h10 | behaviour acc h25 | transition JS h25 |", "|---|---:|---:|---:|---:|"]
        for item in benchmark.get("models", []):
            lines.append(
                f"| {item['model']} | {item['latent_mse_h10']['mean']:.3f} | "
                f"{item['behavior_accuracy_h10']['mean']:.3%} | "
                f"{item['behavior_accuracy_h25']['mean']:.3%} | "
                f"{item['behavior_transition_js']['mean']:.3f} |"
            )
        lines += [
            "",
            "Benchmark: open-loop forecasts, 25-frame horizon, 16 seeds, held-out worm. "
            "This is a descriptive comparison, not mechanism.",
        ]
    audit = wb.tier2_audit(6)
    if audit:
        lines += [
            "",
            "## Tier 2 — NC-MCM assumption audit",
            "",
            "| test | verdict |",
            "|---|---|",
        ]
        for name, result in audit.get("tests", {}).items():
            lines.append(f"| {name.replace('_', ' ')} | {result.get('verdict', 'n/a')} |")
        lines += [
            "",
            "The audit tests first-order Markov structure, stationarity, behaviour transport, "
            "state-behaviour information, and K sensitivity. A fail means `do()` remains a "
            "model-based counterfactual, not an identified causal effect.",
        ]
    clamped = wb.tier3_clamped_validation()
    lines += [
        "",
        "## Tier 3 — Premotor",
        "Published clamped-core mode: 15 intrinsic neurons integrated, 69 signal neurons "
        "clamped to an Atanas recording. The canonical 2023-01-23-15 recording has 84/15/69 "
        "counts; fresh Julia runs on all three validation recordings match the port at mean "
        "r=0.9999998 and behaviour-label parity 99.98%. The 205-node free-run remains an "
        "exploratory sandbox.",
    ]
    if clamped:
        agg = clamped.get("aggregate", {})
        lines += [
            f"- Clamped recordings: {agg.get('n_recordings', 'n/a')}",
            f"- Mean behaviour-state agreement: {agg.get('mean_state_agreement', float('nan')):.1%}",
            f"- Mean behaviour-distribution JS: {agg.get('mean_behavior_distribution_js', float('nan')):.3f}",
            f"- Claim level: {clamped.get('claim_level', 'n/a')}",
            f"- Reference status: {clamped.get('reference_status', 'n/a')}",
        ]
    behaviour = wb.tier3_behavior_validation()
    if behaviour:
        agg_beh = behaviour.get("aggregate", {})
        lines += [
            f"- Neural readout vs independent reversal events: mean AUC "
            f"{agg_beh.get('mean_readout_auc', float('nan')):.3f}",
            f"- Clamped model vs events: lag-0 AUC {agg_beh.get('mean_model_lag0_auc', float('nan')):.3f}; "
            f"retrospective aligned AUC {agg_beh.get('mean_model_retrospective_auc', float('nan')):.3f}; "
            f"prospective AUC {agg_beh.get('mean_model_prospective_auc', float('nan')):.3f}",
            f"- One-frame persistence baseline: AUC {agg_beh.get('mean_persistence_auc', float('nan')):.3f}",
            "- Retrospective alignment uses future model output and is not a forecast. The model "
            "does not beat persistence; it is not a validated behavioural predictor.",
            "- The reversal-event check is a behavioural sanity check, not a causal test.",
        ]
    lines += [
        "",
        "## Provenance",
        "BunDLe-Net preprint DOI 10.1101/2023.08.08.551978; NC-MCM DOI 10.1371/journal.pcbi.1012674; "
        "premotor DOI 10.1371/journal.pcbi.1013818. Artifacts are third-party and restored by "
        "`python scripts/fetch_vienna_artifacts.py`.",
    ]
    report = "\n".join(lines) + "\n"
    st.download_button(
        "Download evidence report (Markdown)",
        data=report,
        file_name="vienna_neuroinformatics_evidence.md",
        mime="text/markdown",
        key="report_download",
    )
    st.markdown(report)
    st.markdown("**Machine-readable provenance**")
    st.json(wb.provenance())


def lab_tab(wb: ViennaWorkbench) -> None:
    st.subheader("Evidence & lab alignment")
    st.caption(
        "What the product claims, what the evidence supports, and how each tier maps to the "
        "Neuroinformatics group's published work. This is a lab-facing map, not a claim that "
        "Denis authored these papers."
    )
    st.markdown("**Claim ledger**")
    st.dataframe(
        [
            {
                "claim": "3-D latent manifold supports short-horizon behaviour forecasting",
                "tier": "Tier 1",
                "status": "supported descriptively",
                "evidence": "held-out-worm benchmark; CRPS/coverage reported",
            },
            {
                "claim": "NC-MCM states are first-order Markov and stationary",
                "tier": "Tier 2",
                "status": "rejected by current audit",
                "evidence": "tier2_assumption_audit.json",
            },
            {
                "claim": "do(state) is an identified causal effect",
                "tier": "Tier 2",
                "status": "not claimed",
                "evidence": "observational clusters; audit fails strict assumptions",
            },
            {
                "claim": "published 84/15/69 clamped core is reproduced",
                "tier": "Tier 3",
                "status": "implementation-faithful; 60.2% canonical readout agreement; not a digital twin",
                "evidence": "tier3_clamped_validation.json; tier3_julia_parity_all.json",
            },
            {
                "claim": "neural readout tracks independent reversal events",
                "tier": "Tier 3",
                "status": "supported as a proxy; mean AUC 0.848",
                "evidence": "tier3_behavior_validation.json (v2, bootstrap CI)",
            },
            {
                "claim": "clamped model forecasts reversal events",
                "tier": "Tier 3",
                "status": "not supported; prospective AUC 0.388",
                "evidence": "tier3_behavior_validation.json (v2)",
            },
            {
                "claim": "clamped model is a validated behavioural predictor",
                "tier": "Tier 3",
                "status": "not supported; lag-0 AUC 0.633, retrospective 0.834, persistence 0.939",
                "evidence": "tier3_behavior_validation.json (v2)",
            },
            {
                "claim": "whole-brain digital twin / mechanistic validation",
                "tier": "all",
                "status": "not claimed",
                "evidence": "no causal intervention; implementation parity is not biological validation",
            },
        ],
        hide_index=True,
        width="stretch",
    )

    st.markdown("**Lab map**")
    st.markdown(
        """
- **BunDLe-Net** → Tier 1. Kumar, Gilra, Meunier, Grosse-Wentrup.
  [bioRxiv preprint](https://doi.org/10.1101/2023.08.08.551978) (preprint, not peer-reviewed).
- **NC-MCM** → Tier 2. Grosse-Wentrup, Kumar, Meunier, Zimmer.
  [PLoS Comput Biol 2024](https://doi.org/10.1371/journal.pcbi.1012674).
- **Celegans_premotor** → Tier 3. Morrison & Young.
  [PLoS Comput Biol 2025](https://doi.org/10.1371/journal.pcbi.1013818).
- **Group research direction**: causal inference from observational neural data; neural manifolds;
  BCI translation. [Projects](https://ni.cs.univie.ac.at/research/projects) ·
  [Code](https://ni.cs.univie.ac.at/research/code).
- **Denis Kovacevic** is a member of the group's research team
  ([team page](https://ni.cs.univie.ac.at/about-us/team/person/unitid/81/denis-kovacevic)).
  No publications are attributed to him here.
"""
    )
    st.markdown("**Next experiments that would change the claim level**")
    st.markdown(
        """
1. **Reference fidelity gate:** run the authors' Julia notebook on `2023-01-23-15` and compare
   intrinsic traces and behaviour statistics with the Python port.
2. **Held-out sixth worm:** test the clamped and latent models on a recording not used to fit
   either the manifold or the recovered weights.
3. **Intervention dataset:** use optogenetic or ablation trials with a pre-registered target and
   a held-out animal; the current Randi atlas is evidence about propagation, not a Tier-1/2 test.
4. **Markov-order follow-up:** if second-order structure is real, fit an explicit higher-order
   model and compare held-out behaviour NLL before using `do()`.
"""
    )
    st.download_button(
        "Download claim ledger (CSV)",
        data="claim,tier,status,evidence\n"
        "3-D latent manifold forecasting,Tier 1,supported descriptively,held-out benchmark\n"
        "NC-MCM first-order Markov and stationary,Tier 2,rejected by current audit,tier2 audit\n"
        "do(state) identified causal effect,Tier 2,not claimed,observational clusters\n"
        "published 84/15/69 clamped core,Tier 3,implementation-faithful; 60.2% readout agreement,tier3 validation + all-recording Julia parity\n"
        "neural readout vs independent reversal events,Tier 3,proxy supported; mean AUC 0.848,tier3_behavior_validation.json v2\n"
        "clamped model forecasts reversal events,Tier 3,not supported; prospective AUC 0.388,tier3_behavior_validation.json v2\n"
        "clamped model validated behavioural predictor,Tier 3,not supported; persistence AUC 0.939,tier3_behavior_validation.json v2\n"
        "whole-brain digital twin,all,not claimed,no causal intervention\n",
        file_name="vienna_claim_ledger.csv",
        mime="text/csv",
        key="lab_download",
    )


def main() -> None:
    wb = get_workbench()
    st.title("🧠 Vienna Neuroinformatics Studio")
    st.caption(
        "A three-tier, source-traceable simulator stack for the University of Vienna "
        "Neuroinformatics group: BunDLe-Net latent manifold · NC-MCM causal states · premotor "
        "biophysical sandbox."
    )
    st.info(
        "Scope: research prototype, not a validated digital twin. Each section states its claim "
        "level; no tier establishes whole-brain mechanism or causality."
    )
    tabs = st.tabs(
        [
            "Overview",
            "Tier 1 · BunDLe-Net",
            "Tier 2 · NC-MCM",
            "Tier 3 · Premotor",
            "Evidence report",
            "Evidence & lab",
            "Provenance",
        ]
    )
    with tabs[0]:
        overview_tab(wb)
    with tabs[1]:
        try:
            tier1_tab(wb)
        except FileNotFoundError as error:
            render_missing_artifact(error)
    with tabs[2]:
        try:
            tier2_tab(wb)
        except FileNotFoundError as error:
            render_missing_artifact(error)
    with tabs[3]:
        try:
            tier3_tab(wb)
        except FileNotFoundError as error:
            render_missing_artifact(error)
    with tabs[4]:
        report_tab(wb)
    with tabs[5]:
        lab_tab(wb)
    with tabs[6]:
        provenance_tab(wb)


if __name__ == "__main__":
    main()
