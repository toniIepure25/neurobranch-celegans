"""NeuroBranch interactive explorer.

Run with:  uv run neurobranch app
or:        streamlit run app/streamlit_app.py

Every simulated trajectory is a model output.  Virtual perturbations are not
experimental interventions and are not causally validated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from neurobranch import EPISTEMIC_WARNING  # noqa: E402
from neurobranch.branching import (  # noqa: E402
    candidate_frames,
    local_outcome_diversity,
    summarize_branching,
)
from neurobranch.config import ExperimentConfig  # noqa: E402
from neurobranch.data.kato import MACRO_ORDER  # noqa: E402
from neurobranch.dynamics.checkpointing import load_model  # noqa: E402
from neurobranch.dynamics.training import load_worm_data  # noqa: E402
from neurobranch.interventions import InterventionSpec, run_counterfactual  # noqa: E402
from neurobranch.interventions.search import available_candidates  # noqa: E402
from neurobranch.interventions.spec import make_perturbation  # noqa: E402
from neurobranch.preprocessing import split_starts  # noqa: E402

CFG_PATH = ROOT / "configs" / "experiments" / "blocked_suite.yaml"
COLORS = {"FWD": "#2E86AB", "REV": "#E4572E", "TURN": "#F3A712", "NOSTATE": "#7D8491"}

st.set_page_config(page_title="NeuroBranch", page_icon="🧠", layout="wide")
st.title("NeuroBranch — C. elegans population dynamics")
st.caption("Observational model explorer. Not a causal simulator of the animal.")
st.warning(EPISTEMIC_WARNING, icon="⚠️")


@st.cache_resource(show_spinner="Loading Kato 2015 recordings …")
def get_data():
    cfg = ExperimentConfig.load(CFG_PATH)
    return cfg, load_worm_data(cfg)


@st.cache_resource(show_spinner="Loading model checkpoint …")
def get_model(path: str):
    return load_model(path)


@st.cache_data(show_spinner=False)
def project_pca(worm_id: int, seed: int = 0):
    cfg, datasets = get_data()
    data = datasets[worm_id]
    tr = split_starts(data, "train", history=1, horizon=1)
    pca = PCA(n_components=3, random_state=seed).fit(data.X[tr])
    return pca.transform(data.X), pca


@st.cache_data(show_spinner=False)
def branching_cached(worm_id: int):
    cfg, datasets = get_data()
    data = datasets[worm_id]
    Z, _ = project_pca(worm_id)
    cand = candidate_frames(data, horizon=20)
    return local_outcome_diversity(Z, data, cand, k=40, horizon=20), cand


def load_optional_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_by_worm(rows, worm_id: int):
    """Look up a row by worm_id, never by list position."""
    if not rows:
        return None
    return next((r for r in rows if r.get("worm_id") == worm_id), None)


def behavior_trace(data, frames, prefix: str, color: bool = True):
    labels = [MACRO_ORDER[int(i)] for i in data.y_macro[frames]]
    return go.Scatter3d(
        x=frames,
        y=[0] * len(frames),
        z=labels,
        mode="markers",
        marker=dict(size=2, color=[COLORS.get(lab, "#999") for lab in labels] if color else "#555"),
        name=prefix,
        hovertext=labels,
        visible=False,
    )


def path3d(Z, name, color="#111111", width=4, dash=None, opacity=1.0):
    return go.Scatter3d(
        x=Z[:, 0],
        y=Z[:, 1],
        z=Z[:, 2],
        mode="lines",
        line=dict(color=color, width=width, dash=dash),
        opacity=opacity,
        name=name,
    )


def points3d(Z, labels, name, size=3, opacity=0.75, symbol="circle"):
    return go.Scatter3d(
        x=Z[:, 0],
        y=Z[:, 1],
        z=Z[:, 2],
        mode="markers",
        marker=dict(
            size=size,
            color=[COLORS.get(str(lab), "#999") for lab in labels],
            opacity=opacity,
            symbol=symbol,
            line=dict(width=0),
        ),
        text=[str(lab) for lab in labels],
        name=name,
        hovertemplate="%{text}<extra></extra>",
    )


cfg, datasets = get_data()
models_dir = ROOT / "artifacts" / "runs" / "blocked_suite"
available_models = sorted({p.name.split("__")[0] for p in models_dir.glob("*__worm*__seed*")}) if models_dir.exists() else []

with st.sidebar:
    st.header("Controls")
    worm_id = st.selectbox("Worm", list(range(len(datasets))), format_func=lambda i: f"worm {i} · {datasets[i].n_neurons} predictors")
    model_name = st.selectbox("Dynamics model", available_models or ["mlp"], index=0)
    seed = st.selectbox("Training seed", [17, 23, 31], index=0)
    st.divider()
    st.caption("Blocked contiguous holdout; windows never cross split boundaries. Not forward forecasting.")
    st.caption("Official BunDLe-Net embeddings were fitted on all frames and are shown as in-sample artifact geometry only.")

data = datasets[worm_id]
model_path = models_dir / f"{model_name}__worm{worm_id}__seed{seed}" / "model.pt"
model = None
if model_path.exists():
    model, meta = get_model(str(model_path))
else:
    st.sidebar.error(f"No checkpoint at {model_path.relative_to(ROOT)}")

Z_pca, pca = project_pca(worm_id)
test_frames = np.flatnonzero(data.masks["test"])
train_frames = np.flatnonzero(data.masks["train"])

tabs = st.tabs(
    [
        "Overview",
        "3-D latent",
        "Branching",
        "Virtual perturbation",
        "Optogenetic (Randi)",
        "Reproduction",
        "Audit",
    ]
)

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames", data.n_frames)
    c2.metric("Predictor neurons", data.n_neurons)
    c3.metric("Held-out frames", int(data.masks["test"].sum()))
    c4.metric("Behaviour classes", len(np.unique(data.y_macro)))
    st.markdown(
        """
        **What this app can show:** observed population activity, a fitted model's
        open-loop predictions, and *virtual* perturbations of that model.

        **What it cannot show:** that a perturbation would work in a living worm.
        No interventional data were used to fit the dynamics.  Behaviour labels are
        decoded from the same neural population, with eight behaviour-reporting
        neurons excluded from predictors to reduce circularity.
        """
    )
    counts = pd.Series([MACRO_ORDER[i] for i in data.y_macro[test_frames]]).value_counts()
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[COLORS.get(i, "#999") for i in counts.index]))
    fig.update_layout(height=260, title="Held-out behaviour composition", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Observed trajectory and model rollout")
    source = st.radio(
        "Observed projection",
        ["Model-state PCA (3-D)", "Official BunDLe artifact (in-sample)"],
        horizontal=True,
    )
    start_choices = test_frames[test_frames + 20 < data.n_frames][:: max(1, len(test_frames) // 40)]
    start = st.select_slider(
        "Initial frame (held-out)",
        options=start_choices.tolist(),
        value=int(start_choices[len(start_choices) // 2]),
        format_func=lambda t: f"frame {t} · {MACRO_ORDER[int(data.y_macro[t])]}",
    )
    horizon = st.slider("Rollout horizon (frames)", 3, 40, 15)
    run = st.button("Roll out model", type="primary")

    obs_segment = test_frames[(test_frames > start - 150) & (test_frames < start + 150)]
    fig = go.Figure()
    fig.add_trace(
        points3d(
            Z_pca[obs_segment],
            [MACRO_ORDER[int(i)] for i in data.y_macro[obs_segment]],
            "OBSERVED (held-out)",
            size=3,
        )
    )
    fig.add_trace(path3d(Z_pca[obs_segment], "observed path", "#222222", width=2, opacity=0.35))
    fig.add_trace(
        go.Scatter3d(
            x=[Z_pca[start, 0]], y=[Z_pca[start, 1]], z=[Z_pca[start, 2]],
            mode="markers", marker=dict(size=8, color="black"), name="initial state",
        )
    )
    if run and model is not None:
        with st.spinner("Simulating …"):
            seq = model.rollout(data.X[start], horizon, seed=seed, noise_std=0.04)
        Zsim = pca.transform(seq)
        labels = [MACRO_ORDER[int(v)] for v in model.behavior_proba(seq[:-1]).argmax(axis=1)]
        fig.add_trace(path3d(Zsim, "MODEL PREDICTION", "#6A4C93", width=6))
        fig.add_trace(points3d(Zsim[1:], labels, "predicted behaviour", size=4, symbol="diamond"))
        st.info("Purple path is an open-loop model prediction from a single observed state; it is not data.")
    fig.update_layout(
        height=680,
        scene=dict(xaxis_title="PC 1", yaxis_title="PC 2", zaxis_title="PC 3"),
        legend=dict(orientation="h"),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Observation colours: blue FWD, red REV, orange TURN, grey NOSTATE. Grey 3-D projections can create apparent branches that are not bifurcations.")

    off = load_optional_json(ROOT / "artifacts" / "reproduction_bundlenet_official.json")
    if source.startswith("Official") and off:
        st.warning("The official artifact was fitted on all frames, including those shown here. Geometry is descriptive and in-sample; do not read accuracy as generalization.")
        row = find_by_worm(off, worm_id)
        if row:
            st.json(row, expanded=False)

with tabs[2]:
    st.subheader("Empirical local outcome diversity")
    st.markdown(
        "For each held-out reversal state we retrieve its 40 nearest neighbours in a 3-D PCA projection "
        "(excluding ±12 frames in time), then count the next macro-behaviour reached. "
        "High entropy means similar-looking states had different futures *in this projection*."
    )
    if st.button("Compute branching summary", type="primary"):
        with st.spinner("Retrieving neighbours …"):
            res, cand = branching_cached(worm_id)
            summ = summarize_branching(res, top_n=8)
        c1, c2, c3 = st.columns(3)
        c1.metric("Reversal states", summ["n"])
        c2.metric("Mean entropy", f"{summ['mean_entropy_bits']:.2f} bits")
        c3.metric("Majority prob.", f"{summ['mean_majority_probability']:.2f}")
        st.caption(
            "Corrected audit: observed diversity was *below* geometry-preserving nulls in 4/5 worms "
            "(entropy deficit, not excess). No worm showed significant excess. See docs/BRANCHING_STATISTICAL_AUDIT.md."
        )
        top = summ["top_frames"][0]
        fig = go.Figure(
            go.Bar(
                x=list(top["probabilities"].keys()),
                y=list(top["probabilities"].values()),
                marker_color=[COLORS.get(k, "#999") for k in top["probabilities"]],
            )
        )
        fig.update_layout(height=280, title=f"Top-entropy frame {top['frame']} (entropy {top['entropy']:.2f} bits)", yaxis_title="neighbour fraction")
        st.plotly_chart(fig, use_container_width=True)
        t = top["frame"]
        Zt = Z_pca[t]
        dist = np.linalg.norm(Z_pca[cand] - Zt, axis=1)
        near = cand[np.argsort(dist)[:80]]
        fig2 = go.Figure(points3d(Z_pca[near], [MACRO_ORDER[int(i)] for i in data.y_macro[near]], "local neighbourhood", size=5))
        fig2.add_trace(go.Scatter3d(x=[Zt[0]], y=[Zt[1]], z=[Zt[2]], mode="markers", marker=dict(size=10, color="black"), name="query"))
        fig2.update_layout(height=500, scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"))
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(pd.DataFrame(summ["top_frames"])[["frame", "entropy", "majority_probability"]], use_container_width=True)

with tabs[3]:
    st.subheader("Virtual perturbation (model-dependent)")
    st.error("MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED. This edits model state, not a neuron in an animal.")
    cands = available_candidates(data)
    targets = st.multiselect("Target neuron(s) present in this worm's predictor set", cands, default=cands[:1])
    c1, c2, c3 = st.columns(3)
    mode = c1.selectbox("Mode", ["additive", "clamp", "zero", "scale"])
    amp = c2.slider("Amplitude (training SD)", -3.0, 3.0, 2.0, 0.25)
    duration = c3.slider("Duration (frames)", 0, 30, 12)
    n_initial = st.slider("Initial held-out reversal states", 5, 40, 20)
    run_int = st.button("Run paired counterfactual", type="primary", disabled=not (targets and model))
    if run_int and model is not None and targets:
        spec = InterventionSpec(
            name="interactive",
            targets=tuple(targets),
            mode=mode,
            amplitude=amp,
            onset=0,
            duration=duration,
            horizon=12,
            n_initial=n_initial,
            seeds=(0, 1, 2),
            initial_behavior="REV",
            target_behavior="TURN",
        )
        with st.spinner("Running paired baseline and intervention …"):
            res = run_counterfactual(model, data, spec, seed=17, noise_std=0.04)
            frames = split_starts(data, "test", history=1, horizon=1)
            frames = frames[data.y_macro[frames] == MACRO_ORDER.index("REV")]
            x0 = data.X[int(frames[len(frames) // 2])]
            perturb, _ = make_perturbation(data, spec)
            base_seq = model.rollout(x0, 12, seed=3, noise_std=0.04)
            int_seq = model.rollout(x0, 12, seed=3, noise_std=0.04, perturb=perturb)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baseline P(TURN)", f"{res.baseline_distribution['TURN']:.2f}")
        c2.metric("Virtual P(TURN)", f"{res.intervention_distribution['TURN']:.2f}", f"{res.delta_target:+.2f}")
        c3.metric("Bootstrap 95% CI", f"[{res.delta_ci[0]:+.2f}, {res.delta_ci[1]:+.2f}]")
        c4.metric("OOD states", "n/a" if res.ood_fraction is None else f"{res.ood_fraction:.0%}")
        fig = go.Figure()
        fig.add_trace(path3d(pca.transform(base_seq), "BASELINE model rollout", "#222222", width=5))
        fig.add_trace(path3d(pca.transform(int_seq), f"VIRTUAL: {','.join(targets)}", "#C1121F", width=6, dash="dash"))
        fig.update_layout(height=480, scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(
            go.Figure(
                [
                    go.Bar(name="baseline", x=list(res.baseline_distribution), y=list(res.baseline_distribution.values()), marker_color="#777"),
                    go.Bar(name="virtual", x=list(res.intervention_distribution), y=list(res.intervention_distribution.values()), marker_color="#C1121F"),
                ]
            ).update_layout(barmode="group", height=300, yaxis_title="outcome probability"),
            use_container_width=True,
        )
        if res.notes:
            for n in res.notes:
                st.warning(n)
        st.caption(
            "A large effect at ±2 SD is an off-support stress test. Matched random-neuron controls produced "
            "equal or larger effects in every audited architecture; this is not a specific RIVL result."
        )

with tabs[4]:
    st.subheader("Randi et al. optogenetic propagation atlas")
    st.warning(
        "Experimental data, but this benchmark is not a biological validation of NeuroBranch. "
        "It asks whether pre-stimulus state predicts independently measured responses."
    )
    cov_path = ROOT / "reports" / "phase2b_optogenetic_atlas" / "randi_coverage.json"
    rel_path = ROOT / "reports" / "phase2b_optogenetic_atlas" / "response_reliability.json"
    agg_path = ROOT / "reports" / "phase2b_optogenetic_atlas" / "benchmark_aggregate_response.csv"
    cov = load_optional_json(cov_path)
    rel = load_optional_json(rel_path)
    if cov:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Animals", cov["n_animals"])
        c2.metric("Valid stimulations", cov["n_events"])
        c3.metric("Stimulated cells", cov["n_stimulated_neurons"])
        c4.metric("Recorded cells", cov["n_recorded_neurons"])
    if rel:
        r = rel.get("response", {})
        st.metric("Repeated-stimulation response r (median)", f"{r.get('median_pairwise_r', float('nan')):.3f}")
        st.caption(
            f"{r.get('n_pairs', 0)} repeated event pairs; low repeatability bounds single-trial predictability."
        )
    if agg_path.exists():
        agg = pd.read_csv(agg_path)
        st.dataframe(
            agg[["task", "model", "mae_mean", "pearson_mean", "sign_accuracy_mean"]],
            use_container_width=True,
        )
        st.caption(
            "Every learned model was worse than a constant training-mean baseline on held-out events, "
            "targets, and animals (negative result retained)."
        )
    else:
        st.info("Run scripts/phase2a_optogenetic_benchmark.py after downloading the Randi OSF archive.")

with tabs[5]:
    st.subheader("Reproduction artifacts")
    st.markdown("**NC-MCM** — Python port of the lag-2 surrogate Markov test applied to the distributed cluster labels.")
    ncm = ROOT / "artifacts" / "reproduction_ncmcm.csv"
    if ncm.exists():
        st.dataframe(pd.read_csv(ncm), use_container_width=True)
        st.caption("K=3. p-values reproduce the distributed values closely; decoder accuracy is blocked 5-fold and much lower than in-sample numbers.")
    st.markdown("**BunDLe-Net** — official 3-D artifact vs our PyTorch reimplementation.")
    bl = load_optional_json(ROOT / "artifacts" / "reproduction_bundlenet_lite.json")
    bo = load_optional_json(ROOT / "artifacts" / "reproduction_bundlenet_official.json")
    c1, c2 = st.columns(2)
    row = find_by_worm(bo, worm_id)
    if row:
        c1.metric("Official kNN (leaky)", f"{row.get('official_knn_accuracy', float('nan')):.3f}")
        c1.caption("Embedding fitted upstream on all frames; test embeddings were seen in training.")
    if bl:
        c2.metric("Reimplementation kNN (blocked)", f"{bl['metrics'].get('knn_accuracy', float('nan')):.3f}")
        c2.caption("Honest holdout; architecture reimplementation underperforms the artifact.")
    if bl:
        st.json({"metrics": bl["metrics"], "deviations": bl["metadata"]["deviations"]}, expanded=False)

with tabs[6]:
    st.subheader("Dataset audit")
    path = ROOT / "reports" / "recording_summary.csv"
    if path.exists():
        st.dataframe(pd.read_csv(path), use_container_width=True)
    st.markdown(
        "Raw recordings are immutable. `WT_NoStim` is observational; no optogenetic or histamine "
        "intervention data were used for fitting. The Randi optogenetic atlas is used only in the "
        "external response benchmark, not to train the dynamics. See `docs/DATA_PROVENANCE.md`."
    )
    st.code(
        "uv run neurobranch data audit\nuv run neurobranch train dynamics --config configs/experiments/blocked_suite.yaml\nuv run neurobranch report",
        language="bash",
    )

st.divider()
st.caption(EPISTEMIC_WARNING)
