"""Vienna Neuroinformatics Workbench — Streamlit UI.

Run from the repository root:
    streamlit run src/neurobranch/bundlenet_sim/app.py
"""

from __future__ import annotations

import io
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
                "Open-loop forecast from 75-frame-spaced starts, 25-frame horizon, 4 seeds. "
                "Lower JS is better; this is a descriptive forecast comparison, not mechanism."
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

    st.warning(
        "This is a model-based counterfactual, not an identified causal effect. "
        "A real do(state) claim requires the NC-MCM assumptions plus experimental intervention."
    )


def tier3_tab(wb: ViennaWorkbench) -> None:
    st.subheader("Tier 3 — Biophysical premotor sandbox")
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
        "activity; this page does not yet implement that clamped mode. It is not a validated "
        "digital twin and must not be used for causal claims."
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
- Tier 3: biophysical sandbox with the fitted chemical matrix and recovered gap-junction matrix
  and neuron names. The 205-node free-run is exploratory; the paper's clamped-signal simulation
  is not implemented. This is not a validated digital twin.
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
3. **Premotor connectome sandbox** — run the published 205-node rate ODE with recovered neuron
   names and electrical-junction weights, and export the activity.
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
    c3.metric(
        "Premotor nodes",
        f"{wb.premotor.n_nodes}" if (wb.root / "data/processed/premotor/avg_model_weights.jld2").exists() else "n/a",
    )
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
1. Open **Tier 1**, pick a held-out worm, and compare *manifold walk* with *switching linear*.
2. Open **Tier 2**, choose `K`, inspect the Markov p-value and the transition uncertainty.
3. Open **Tier 3**, run the chemical-only ODE, then increase β only within the stable range.
4. Download the **Evidence report** before the meeting; it records every number and caveat.
"""
    )

    st.markdown("**Claim boundaries**")
    st.warning(
        "Tier 1 is a latent-manifold model, not a circuit. Tier 2 is descriptive and its do() is "
        "a Markov counterfactual, not an experimentally identified effect. Tier 3 is a free-run "
        "sandbox: the published paper clamps 69 signal neurons to recordings, and that clamped "
        "mode is not implemented here. None of the three is a validated whole-brain digital twin."
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
            "Benchmark: open-loop forecasts, 25-frame horizon, 4 seeds, held-out worm. "
            "This is a descriptive comparison, not mechanism.",
        ]
    lines += [
        "",
        "## Tier 2 — NC-MCM",
        "Fitted cognitive-state models for five worms. Markov p-values and transition matrices are "
        "descriptive; intervention trajectories are model-based counterfactuals under the Markov "
        "assumption, not identified causal effects.",
        "",
        "## Tier 3 — Premotor",
        "205-node fitted chemical matrix and cubic rate ODE. Neuron names and W were recovered from "
        "the release connectome; all 913 non-zero A entries lie inside the chemical support. "
        "The full free-run is exploratory; the published clamped-signal simulation is not implemented.",
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
        provenance_tab(wb)


if __name__ == "__main__":
    main()
