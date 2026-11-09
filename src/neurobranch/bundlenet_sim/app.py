"""Vienna Neuroinformatics Workbench — Streamlit UI.

Run from the repository root:
    streamlit run src/neurobranch/bundlenet_sim/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from neurobranch.bundlenet_sim.dataset import BEHAVIOR_NAMES  # noqa: E402
from neurobranch.bundlenet_sim.premotor import PremotorParameters  # noqa: E402
from neurobranch.bundlenet_sim.workbench import ViennaWorkbench  # noqa: E402

st.set_page_config(page_title="Vienna Neuroinformatics Workbench", page_icon="🧠", layout="wide")

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
            options=["manifold_walk", "quadratic"],
            format_func=lambda m: "Manifold walk (stable)" if m == "manifold_walk" else "Quadratic forecast (unstable)",
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

    st.markdown("**Intervention read-out (observational, not identified)**")
    c1, c2 = st.columns(2)
    with c1:
        state = st.selectbox("Force state", [f"S{i + 1}" for i in range(k)], key="t2_state")
        idx = int(state[1:]) - 1
        st.write("Outgoing transition probabilities:")
        st.dataframe(
            {"next": [f"S{j + 1}" for j in range(k)], "p": np.round(P[idx], 3)},
            width="stretch",
            hide_index=True,
        )
    with c2:
        fig = go.Figure(
            go.Bar(
                x=list(BEHAVIOR_NAMES),
                y=behavior_dist[idx],
                marker_color=COLORS,
            )
        )
        fig.update_layout(
            height=300,
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            yaxis_title="p(behaviour | state)",
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig, width="stretch")
    st.warning(
        "This is a conditional read-out from the fitted NC-MCM, not an identified causal effect. "
        "A real do(state) claim requires the NC-MCM assumptions plus experimental intervention."
    )


def tier3_tab(wb: ViennaWorkbench) -> None:
    st.subheader("Tier 3 — Biophysical premotor sandbox")
    st.caption(
        "205-node fitted synaptic matrix from Celegans_premotor, integrated with the published "
        "cubic rate ODE. Gap-junction weights and neuron-name order are not in the public export."
    )
    model = wb.premotor
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        duration = st.slider("Duration", 2.0, 40.0, 12.0, 1.0, key="t3_dur")
    with c2:
        gamma = st.slider("Synaptic gain γ", 0.0, 3.0, 1.4, 0.1, key="t3_gamma")
    with c3:
        beta = st.slider("Gap-junction gain β", 0.0, 20.0, 10.0, 0.5, key="t3_beta")
    with c4:
        tau = st.slider("Time constant τ", 0.05, 1.0, 0.2, 0.05, key="t3_tau")

    c5, c6, c7 = st.columns(3)
    with c5:
        stim_node = st.number_input("Stimulated node", 0, model.n_nodes - 1, 0, key="t3_stim")
    with c6:
        stim_amp = st.slider("Stimulus amplitude", 0.0, 20.0, 5.0, 0.5, key="t3_amp")
    with c7:
        ablate_text = st.text_input("Ablate nodes (comma-separated)", "", key="t3_ablate")

    ablate: tuple[int, ...] = ()
    if ablate_text.strip():
        try:
            ablate = tuple(int(x.strip()) for x in ablate_text.split(",") if x.strip())
        except ValueError:
            st.error("Ablation list must be comma-separated integers.")

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
        node = st.number_input("Node trace", 0, model.n_nodes - 1, 0, key="t3_node")
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

    st.warning(
        "This sandbox uses the published fitted chemical synaptic matrix and ODE form, but the "
        "export omits W (gap junctions) and neuron names. It is not a validated digital twin and "
        "must not be used for causal claims without the missing mapping and experimental checks."
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
  free-run generative use is not validated.
- Tier 2: descriptive causal-state model; `do()` is not identified by the precomputed artifact.
- Tier 3: biophysical sandbox with a fitted chemical matrix; gap junctions and neuron identities
  are missing from the export.
"""
    )


def main() -> None:
    wb = get_workbench()
    st.title("🧠 Vienna Neuroinformatics Workbench")
    st.caption(
        "A three-tier, source-traceable workbench for the University of Vienna Neuroinformatics "
        "group: BunDLe-Net latent manifold · NC-MCM causal states · premotor biophysical sandbox."
    )
    st.info(
        "Scope: this is a research prototype, not a validated digital twin. Each tab states its "
        "claim level; no tier establishes whole-brain mechanism or causality."
    )
    tabs = st.tabs(["Tier 1 · BunDLe-Net", "Tier 2 · NC-MCM", "Tier 3 · Premotor", "Provenance"])
    with tabs[0]:
        try:
            tier1_tab(wb)
        except FileNotFoundError as error:
            render_missing_artifact(error)
    with tabs[1]:
        try:
            tier2_tab(wb)
        except FileNotFoundError as error:
            render_missing_artifact(error)
    with tabs[2]:
        try:
            tier3_tab(wb)
        except FileNotFoundError as error:
            render_missing_artifact(error)
    with tabs[3]:
        provenance_tab(wb)


if __name__ == "__main__":
    main()
