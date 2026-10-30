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
from neurobranch.reporting.status import load_measurement_status  # noqa: E402

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
available_models = (
    sorted({p.name.split("__")[0] for p in models_dir.glob("*__worm*__seed*")}) if models_dir.exists() else []
)

with st.sidebar:
    st.header("Controls")
    worm_id = st.selectbox(
        "Worm",
        list(range(len(datasets))),
        format_func=lambda i: f"worm {i} · {datasets[i].n_neurons} predictors",
    )
    model_name = st.selectbox("Dynamics model", available_models or ["mlp"], index=0)
    seed = st.selectbox("Training seed", [17, 23, 31], index=0)
    st.divider()
    st.caption("Blocked contiguous holdout; windows never cross split boundaries. Not forward forecasting.")
    st.caption(
        "Official BunDLe-Net embeddings were fitted on all frames and are shown as in-sample artifact geometry only."
    )

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
        "Prospective Behavior",
        "Turn Direction & Neural Sequences",
        "SAAV Mechanistic Validation",
        "Causal Response Operator",
        "Active Probing",
        "Adaptive Twin",
        "BoundedBench",
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
    fig = go.Figure(
        go.Bar(x=counts.index, y=counts.values, marker_color=[COLORS.get(i, "#999") for i in counts.index])
    )
    fig.update_layout(height=260, title="Held-out behaviour composition", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

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
            x=[Z_pca[start, 0]],
            y=[Z_pca[start, 1]],
            z=[Z_pca[start, 2]],
            mode="markers",
            marker=dict(size=8, color="black"),
            name="initial state",
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
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Observation colours: blue FWD, red REV, orange TURN, grey NOSTATE. Grey 3-D projections can create apparent branches that are not bifurcations."
    )

    off = load_optional_json(ROOT / "artifacts" / "reproduction_bundlenet_official.json")
    if source.startswith("Official") and off:
        st.warning(
            "The official artifact was fitted on all frames, including those shown here. Geometry is descriptive and in-sample; do not read accuracy as generalization."
        )
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
        fig.update_layout(
            height=280,
            title=f"Top-entropy frame {top['frame']} (entropy {top['entropy']:.2f} bits)",
            yaxis_title="neighbour fraction",
        )
        st.plotly_chart(fig, width="stretch")
        t = top["frame"]
        Zt = Z_pca[t]
        dist = np.linalg.norm(Z_pca[cand] - Zt, axis=1)
        near = cand[np.argsort(dist)[:80]]
        fig2 = go.Figure(
            points3d(
                Z_pca[near], [MACRO_ORDER[int(i)] for i in data.y_macro[near]], "local neighbourhood", size=5
            )
        )
        fig2.add_trace(
            go.Scatter3d(
                x=[Zt[0]],
                y=[Zt[1]],
                z=[Zt[2]],
                mode="markers",
                marker=dict(size=10, color="black"),
                name="query",
            )
        )
        fig2.update_layout(height=500, scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"))
        st.plotly_chart(fig2, width="stretch")
        st.dataframe(
            pd.DataFrame(summ["top_frames"])[["frame", "entropy", "majority_probability"]],
            width="stretch",
        )

with tabs[3]:
    st.subheader("Virtual perturbation (model-dependent)")
    st.error(
        "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED. This edits model state, not a neuron in an animal."
    )
    cands = available_candidates(data)
    targets = st.multiselect(
        "Target neuron(s) present in this worm's predictor set", cands, default=cands[:1]
    )
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
        c2.metric(
            "Virtual P(TURN)", f"{res.intervention_distribution['TURN']:.2f}", f"{res.delta_target:+.2f}"
        )
        c3.metric("Bootstrap 95% CI", f"[{res.delta_ci[0]:+.2f}, {res.delta_ci[1]:+.2f}]")
        c4.metric("OOD states", "n/a" if res.ood_fraction is None else f"{res.ood_fraction:.0%}")
        fig = go.Figure()
        fig.add_trace(path3d(pca.transform(base_seq), "BASELINE model rollout", "#222222", width=5))
        fig.add_trace(
            path3d(pca.transform(int_seq), f"VIRTUAL: {','.join(targets)}", "#C1121F", width=6, dash="dash")
        )
        fig.update_layout(
            height=480,
            scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, width="stretch")
        st.plotly_chart(
            go.Figure(
                [
                    go.Bar(
                        name="baseline",
                        x=list(res.baseline_distribution),
                        y=list(res.baseline_distribution.values()),
                        marker_color="#777",
                    ),
                    go.Bar(
                        name="virtual",
                        x=list(res.intervention_distribution),
                        y=list(res.intervention_distribution.values()),
                        marker_color="#C1121F",
                    ),
                ]
            ).update_layout(barmode="group", height=300, yaxis_title="outcome probability"),
            width="stretch",
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
        st.metric(
            "Repeated-stimulation response r (median)", f"{r.get('median_pairwise_r', float('nan')):.3f}"
        )
        st.caption(
            f"{r.get('n_pairs', 0)} repeated event pairs; low repeatability bounds single-trial predictability."
        )
    if agg_path.exists():
        agg = pd.read_csv(agg_path)
        st.dataframe(
            agg[["task", "model", "mae_mean", "pearson_mean", "sign_accuracy_mean"]],
            width="stretch",
        )
        st.caption(
            "Every learned model was worse than a constant training-mean baseline on held-out events, "
            "targets, and animals (negative result retained)."
        )
    else:
        st.info("Run scripts/phase2a_optogenetic_benchmark.py after downloading the Randi OSF archive.")

with tabs[5]:
    st.subheader("Reproduction artifacts")
    st.markdown(
        "**NC-MCM** — Python port of the lag-2 surrogate Markov test applied to the distributed cluster labels."
    )
    ncm = ROOT / "artifacts" / "reproduction_ncmcm.csv"
    if ncm.exists():
        st.dataframe(pd.read_csv(ncm), width="stretch")
        st.caption(
            "K=3. p-values reproduce the distributed values closely; decoder accuracy is blocked 5-fold and much lower than in-sample numbers."
        )
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
        st.dataframe(pd.read_csv(path), width="stretch")
    st.markdown(
        "Raw recordings are immutable. `WT_NoStim` is observational; no optogenetic or histamine "
        "intervention data were used for fitting. The Randi optogenetic atlas is used only in the "
        "external response benchmark, not to train the dynamics. Phase 3 adds the independent "
        "Baskoylu & Flavell (2026) freely-moving dataset (27 animals), used only for the prospective "
        "behaviour tab; its published whole-recording z-score is not used as a predictor. See "
        "`docs/DATA_PROVENANCE.md` and `docs/PHASE3_DATASET_AUDIT.md`."
    )
    st.code(
        "uv run neurobranch data audit\nuv run neurobranch train dynamics --config configs/experiments/blocked_suite.yaml\nuv run neurobranch report",
        language="bash",
    )

with tabs[7]:
    st.subheader("Prospective reversal prediction on independently observed behaviour")
    st.markdown(
        "Baskoylu & Flavell (2026), 27 freely-moving animals. The target is a reversal "
        "**onset in the next H seconds**, using only behaviour and neural activity at or "
        "before the current frame. Evaluation is leave-one-animal-out."
    )
    p3 = ROOT / "reports" / "phase3"
    summary = load_optional_json(p3 / "prospective_summary.json")
    audit = load_optional_json(p3 / "dataset_summary.json")
    if audit:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Animals", audit.get("n_animals_usable", "—"))
        c2.metric("Frames", f"{audit.get('n_frames_total', 0):,}")
        c3.metric("Reversals", f"{audit.get('n_reversal_events', 0):,}")
        c4.metric("Identified traces", f"{audit.get('n_labeled_traces', 0):,}")
    if not summary:
        st.info("Run `python scripts/phase3_prospective_benchmark.py` to generate this tab.")
    else:
        horizon = st.select_slider(
            "Prediction horizon", options=[3, 6, 12], value=6, format_func=lambda h: f"{h} s"
        )
        metrics_path = p3 / f"prospective_metrics_{horizon}s.csv"
        delta_path = p3 / f"prospective_deltas_{horizon}s.csv"
        if metrics_path.exists():
            m = pd.read_csv(metrics_path)
            agg = m.groupby("mode", as_index=False)[
                ["log_loss", "brier", "balanced_accuracy", "auprc"]
            ].mean()
            agg = agg.sort_values("log_loss")
            st.markdown(f"**Mean animal-level proper scores, H = {horizon} s**")
            st.dataframe(agg, width="stretch", hide_index=True)
            fig = go.Figure(
                go.Bar(
                    x=agg["mode"],
                    y=agg["log_loss"],
                    marker_color=[
                        "#2E86AB" if x == "behavior" else "#C1121F" if "neural" in x else "#999"
                        for x in agg["mode"]
                    ],
                )
            )
            fig.update_layout(
                height=320, yaxis_title="log loss (lower is better)", margin=dict(l=10, r=10, t=20, b=80)
            )
            st.plotly_chart(fig, width="stretch")
        if horizon == 6 and summary.get("6s"):
            d = summary["6s"]["aggregate"].get("delta_behavior_neural", {})
            if d:
                c1, c2, c3 = st.columns(3)
                c1.metric("Δ log loss (behaviour+neural − behaviour)", f"{d['mean']:+.4f}")
                c2.metric("95% animal bootstrap CI", f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")
                c3.metric("Animals improved", f"{d['n_negative']}/{d['n']}")
                if d["ci95"][1] < 0 and d["n_negative"] / d["n"] >= 0.7:
                    st.success("Prespecified primary neural-increment criterion met at 6 s.")
                else:
                    st.warning(
                        "Prespecified primary criterion **not met**: the animal-level CI includes zero. "
                        "This is an inconclusive/negative result, not evidence that neural activity is irrelevant."
                    )
        if delta_path.exists():
            de = pd.read_csv(delta_path)
            cols = [c for c in de.columns if c.startswith("delta_")]
            if cols:
                st.markdown("**Per-animal paired log-loss differences** (negative = neural helps)")
                st.dataframe(de[["animal_id", *cols]], width="stretch", hide_index=True)
        st.error(
            "Observational prediction only. No neuron was stimulated, no transition was caused, "
            "and no result here supports neuron-specific control."
        )
        st.caption(
            "Primary model: prespecified L2 logistic regression (C=0.1), causal features, "
            "train-only scaling/PCA, leave-one-animal-out. Neural population features are "
            "permutation-invariant ROI summaries; identified-neuron results are separate."
        )

with tabs[8]:
    st.subheader("Turn direction & neural sequences")
    st.markdown(
        "Animal-disjoint, prospective prediction of the post-reversal dorsal/ventral turn "
        "from the middle four frames of a reversal. Features end at least two frames before "
        "the reversal offset; direction is read from the first two post-offset frames. "
        "Positive `dv` is dorsal (the archive's orientation-corrected field)."
    )
    td_metrics_path = ROOT / "reports" / "phase4" / "turn_direction_metrics.csv"
    td_pred_path = ROOT / "reports" / "phase4" / "turn_direction_predictions.parquet"
    td_summary_path = ROOT / "reports" / "phase4" / "turn_direction_summary.json"
    td_cal_path = ROOT / "reports" / "phase4" / "turn_direction_calibration.csv"
    td_dyn_path = ROOT / "reports" / "phase4" / "turn_dynamics_timeresolved.csv"
    td_ext_path = ROOT / "reports" / "phase4" / "turn_direction_external.csv"
    td_abl_path = ROOT / "reports" / "phase4" / "turn_direction_neuron_ablation.csv"
    if not td_metrics_path.exists():
        st.info("Phase 4 turn-direction reports are not generated in this checkout.")
    else:
        tm = pd.read_csv(td_metrics_path)
        st.dataframe(
            tm[
                [
                    "mode",
                    "n",
                    "prevalence",
                    "auroc",
                    "log_loss",
                    "brier",
                    "balanced_accuracy",
                    "calibration_slope",
                ]
            ].sort_values("auroc", ascending=False),
            width="stretch",
            hide_index=True,
        )
        if td_summary_path.exists():
            ts = load_optional_json(td_summary_path)
            if ts:
                dec = ts.get("decision", {})
                d = dec.get("delta_population", {})
                if d:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Δ AUROC (behaviour+population − behaviour)", f"{d.get('mean', np.nan):+.4f}")
                    ci = d.get("ci95", [np.nan, np.nan])
                    c2.metric("95% animal-cluster CI", f"[{ci[0]:+.4f}, {ci[1]:+.4f}]")
                    c3.metric(
                        "Primary classification", str(dec.get("classification", "n/a")).replace("_", " ")
                    )
                st.caption(dec.get("reason", ""))
                st.caption(
                    "This is an observational, model-dependent result. A null increment does not "
                    "show that the circuit is uninvolved; it bounds what this representation and "
                    "split can decode prospectively."
                )
        if td_pred_path.exists():
            tp = pd.read_parquet(td_pred_path)
            mode = st.selectbox(
                "Per-animal held-out AUROC (both classes only)",
                sorted(tp["mode"].unique()),
                index=0,
            )
            from sklearn.metrics import roc_auc_score

            rows = []
            for animal, g in tp[tp["mode"] == mode].groupby("animal_id"):
                if g["y"].nunique() == 2:
                    rows.append({"animal_id": animal, "auroc": roc_auc_score(g["y"], g["p"]), "n": len(g)})
            if rows:
                pa = pd.DataFrame(rows).sort_values("auroc")
                fig = go.Figure(
                    go.Bar(
                        x=pa["auroc"],
                        y=pa["animal_id"],
                        orientation="h",
                        marker_color=np.where(pa["auroc"] >= 0.5, "#2E86AB", "#C1121F"),
                    )
                )
                fig.add_vline(x=0.5, line_dash="dash", line_color="#555")
                fig.update_layout(
                    height=max(300, 14 * len(pa)),
                    xaxis_title="held-out AUROC",
                    margin=dict(l=10, r=10, t=20, b=30),
                )
                st.plotly_chart(fig, width="stretch")
        if td_cal_path.exists():
            cal = pd.read_csv(td_cal_path)
            if not cal.empty:
                fig = go.Figure()
                for mode, g in cal.groupby("mode"):
                    fig.add_trace(
                        go.Scatter(
                            x=g["mean_predicted"], y=g["fraction_positive"], mode="lines+markers", name=mode
                        )
                    )
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line_dash="dash", name="ideal"))
                fig.update_layout(
                    height=320,
                    xaxis_title="predicted probability",
                    yaxis_title="observed fraction",
                    margin=dict(l=10, r=10, t=20, b=30),
                )
                st.plotly_chart(fig, width="stretch")
        if td_dyn_path.exists():
            dyn = pd.read_csv(td_dyn_path)
            fig = go.Figure()
            for mode, g in dyn.groupby("mode"):
                fig.add_trace(
                    go.Scatter(x=g["relative_frame"], y=g["auroc"], mode="lines+markers", name=mode)
                )
            fig.add_vline(x=0, line_dash="dash", line_color="#555")
            fig.update_layout(
                height=340,
                xaxis_title="frame relative to reversal offset (negative = prospective)",
                yaxis_title="held-out AUROC",
                margin=dict(l=10, r=10, t=20, b=30),
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Frames at or after 0 are retrospective positive controls, not prospective predictions."
            )
        if td_ext_path.exists():
            ext = pd.read_csv(td_ext_path)
            st.markdown("**External cohort shift** (fit on wild type, score tdc1 or the other odour cohort)")
            st.dataframe(
                ext[["train", "test", "mode", "n", "prevalence", "auroc", "log_loss", "balanced_accuracy"]],
                width="stretch",
                hide_index=True,
            )
        if td_abl_path.exists():
            abl = pd.read_csv(td_abl_path)
            st.markdown("**Model-dependent circuit-feature ablation** (not an experimental ablation)")
            st.dataframe(abl, width="stretch", hide_index=True)
        st.error(
            "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED. "
            "No trial-level optogenetic or mutant intervention is present in this archive, "
            "so this tab does not report an intervention effect."
        )
        st.caption(
            "The linear logistic benchmark is not the paper's RNN; the dataset does not include the "
            "pre-extracted SAAV CSVs. Results are an independent, animal-disjoint benchmark, not an "
            "exact reproduction of the published accuracy."
        )

with tabs[9]:
    st.subheader("SAAV mechanistic validation (Phase 5)")
    st.markdown(
        "This tab separates the published result, our independent reconstruction, the "
        "animal-disjoint protocol comparison, the causal online-prospective benchmark, and "
        "the intervention evidence map. None of the decoding results is an intervention."
    )
    p5 = ROOT / "reports" / "phase5"
    st.markdown("**Reproduction status**")
    st.info(
        "BLOCKED_MISSING_INPUTS for exact reproduction; PARTIAL_REPRODUCTION for the "
        "independent raw-Dryad reconstruction. The four pre-extracted CSVs used by the "
        "official notebook (SAAV_dorsal.csv, SAAV_ventral.csv, HC_dorsal.csv, "
        "HC_ventral.csv) are absent from Dryad."
    )
    prot_metrics = p5 / "protocol_comparison" / "protocol_metrics.csv"
    prot_deltas = p5 / "protocol_comparison" / "protocol_mode_deltas.csv"
    if not prot_deltas.exists():
        prot_deltas = p5 / "protocol_comparison" / "protocol_deltas.csv"
    prot_summary = p5 / "protocol_comparison" / "protocol_delta_summary.json"
    if prot_metrics.exists():
        pm = pd.read_csv(prot_metrics)
        show = [
            c
            for c in [
                "regime",
                "mode",
                "n",
                "prevalence",
                "log_loss",
                "auroc",
                "balanced_accuracy",
                "section_accuracy",
            ]
            if c in pm.columns
        ]
        st.markdown("**Offline protocol comparison** (same reconstructed events, different splits)")
        st.dataframe(pm[show].sort_values(["regime", "mode"]), width="stretch", hide_index=True)
        if prot_summary.exists():
            ps = load_optional_json(prot_summary)
            if ps:
                st.json({k: v for k, v in ps.items() if isinstance(v, dict)})
        if prot_deltas.exists():
            pdlt = pd.read_csv(prot_deltas)
            st.markdown("**Paired animal-cluster deltas: neural mode − behaviour**")
            st.dataframe(
                pdlt[
                    [
                        c
                        for c in ["regime", "comparison", "metric", "delta", "ci_low", "ci_high", "p"]
                        if c in pdlt.columns
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
    else:
        st.info("Offline protocol reports are not generated in this checkout.")

    online_metrics = p5 / "online_decoding" / "online_metrics.csv"
    online_deltas = p5 / "online_decoding" / "online_deltas.csv"
    online_cfg = p5 / "online_decoding" / "online_config.json"
    risk_sets = p5 / "online_decoding" / "online_risk_sets.csv"
    if online_metrics.exists():
        om = pd.read_csv(online_metrics)
        st.markdown("**Causal online-prospective benchmark** (fixed elapsed landmarks from reversal onset)")
        if online_cfg.exists():
            cfg = load_optional_json(online_cfg) or {}
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Primary events", cfg.get("n_primary_events", "n/a"))
            c2.metric("Animals", cfg.get("n_primary_animals", "n/a"))
            c3.metric("Landmarks (s)", ", ".join(str(x) for x in cfg.get("landmarks_s", [])))
            c4.metric("Onset", str(cfg.get("onset_mode", "n/a")).replace("_", " "))
        if not om.empty:
            lm = st.selectbox("Landmark (s)", sorted(om["landmark_s"].unique()), key="p5_landmark")
            model_name = st.selectbox("Model", sorted(om["model"].unique()), key="p5_model")
            view = om[(om["landmark_s"] == lm) & (om["model"] == model_name)]
            st.dataframe(
                view[
                    [
                        c
                        for c in [
                            "mode",
                            "n",
                            "prevalence",
                            "log_loss",
                            "auroc",
                            "brier",
                            "balanced_accuracy",
                        ]
                        if c in view.columns
                    ]
                ].sort_values("log_loss"),
                width="stretch",
                hide_index=True,
            )
        if risk_sets.exists():
            st.markdown("**Risk set by landmark and direction**")
            st.dataframe(pd.read_csv(risk_sets), width="stretch", hide_index=True)
        if online_deltas.exists():
            od = pd.read_csv(online_deltas)
            st.markdown("**Paired animal-cluster deltas**")
            st.dataframe(
                od[
                    [
                        c
                        for c in [
                            "landmark_s",
                            "model",
                            "comparison",
                            "metric",
                            "delta",
                            "ci_low",
                            "ci_high",
                            "p",
                        ]
                        if c in od.columns
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
    else:
        st.info("Online-prospective reports are not generated in this checkout.")

    mech = p5 / "mechanisms" / "mechanism_channel_tests.csv"
    mech_stability = p5 / "mechanisms" / "mechanism_stability.csv"
    if mech.exists():
        mm = pd.read_csv(mech)
        st.markdown("**Observational mechanism analyses** (encoding/decoding, not causation)")
        view = mm[mm.get("metric").eq("log_loss")] if "metric" in mm else mm
        if not view.empty:
            st.dataframe(
                view[
                    [
                        c
                        for c in [
                            "channel",
                            "analysis",
                            "behavior_only",
                            "with_channel",
                            "delta",
                            "ci_low",
                            "ci_high",
                            "n_events",
                            "n_animals",
                            "holm_p",
                        ]
                        if c in view.columns
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
        if mech_stability.exists():
            st.markdown("**Cross-animal stability**")
            st.dataframe(pd.read_csv(mech_stability), width="stretch", hide_index=True)
    else:
        st.info("Mechanism reports are not generated in this checkout.")

    interventions = p5 / "interventions" / "intervention_evidence.csv"
    if interventions.exists():
        st.markdown("**Intervention evidence map**")
        st.dataframe(pd.read_csv(interventions), width="stretch", hide_index=True)
        st.caption(
            "Published manipulations are separated from trial-level observations, simulator "
            "predictions and hypotheses. Group averages are never treated as pseudo-trials, and "
            "WT versus tdc-1 is not an acute cell-specific stimulation."
        )
    st.error(
        "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED. "
        "No virtual intervention optimisation was run because no model passed the Phase 5 validation gate."
    )
    st.caption(
        "Primary generalisation claim is animal-disjoint. Online landmarks use only causal onset "
        "and elapsed time; events ending before a landmark are excluded from that landmark's risk set."
    )

with tabs[10]:
    p6 = ROOT / "reports" / "phase6"
    cro_root = p6 / "cro_target5"
    if not cro_root.exists():
        cro_root = p6 / "cro_target"
    st.subheader("Held-out-target response operator")
    st.error(
        "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED. "
        "The operator predicts responses to optogenetic stimulation from observational "
        "Randi data. It is not an interventional effect estimate and must not be used "
        "to claim that a neuron causes a response."
    )
    pooled_path = cro_root / "metrics_pooled.csv"
    if pooled_path.exists():
        pooled = pd.read_csv(pooled_path)
        best = pooled.sort_values("pair_spearman", ascending=False).head(1)
        c1, c2, c3 = st.columns(3)
        c1.metric("Best pair Spearman", f"{float(best['pair_spearman'].iloc[0]):.3f}")
        c2.metric(
            "95% animal-cluster CI",
            f"{float(best['pair_spearman_ci_low'].iloc[0]):.3f} – {float(best['pair_spearman_ci_high'].iloc[0]):.3f}",
        )
        c3.metric("Evaluated pairs", f"{int(best['n_pairs'].iloc[0]):,}")
        st.dataframe(
            pooled[
                [
                    c
                    for c in [
                        "model",
                        "n_events",
                        "n_pairs",
                        "pair_spearman",
                        "pair_spearman_ci_low",
                        "pair_spearman_ci_high",
                        "pair_mae",
                        "interval_90_coverage",
                    ]
                    if c in pooled.columns
                ]
            ].sort_values("pair_spearman", ascending=False),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Primary endpoint: response_mean, unseen targets, pair-level Spearman. "
            "The CRO is compared against target/responder means, additive, context-only, "
            "connectome ridge, SVD matrix factorisation and an oracle pair mean."
        )
    else:
        st.info("Phase 6 CRO metrics are not generated in this checkout.")

    delta_path = cro_root / "paired_deltas.csv"
    if delta_path.exists():
        st.markdown("**Paired animal-cluster deltas versus baselines**")
        st.dataframe(pd.read_csv(delta_path), width="stretch", hide_index=True)

    profile_path = cro_root / "profile_summary.csv"
    if profile_path.exists():
        st.markdown("**Within-target profile recovery and measurement ceiling**")
        st.dataframe(pd.read_csv(profile_path), width="stretch", hide_index=True)
        st.caption(
            "Reliability is estimated by split-half profiles; the attenuation ceiling "
            "is the maximum correlation expected from an error-free model. Normalised "
            "values near 1 mean the model is close to the noise ceiling, not perfect."
        )

    edge_path = cro_root / "connectivity" / "edge_metrics.csv"
    if edge_path.exists():
        st.markdown("**Anatomical association (not causal discovery)**")
        edge = pd.read_csv(edge_path)
        view = edge[edge.get("score").eq("predicted_magnitude")] if "score" in edge else edge
        st.dataframe(view, width="stretch", hide_index=True)
        st.caption(
            "AUROC ranks directly connected pairs above unconnected pairs using the "
            "predicted response magnitude. This is an association benchmark with "
            "selection and connectome-coverage caveats."
        )

    design_path = p6 / "design" / "target_candidates.csv"
    if design_path.exists():
        st.markdown("**Model-based experiment candidates**")
        st.dataframe(pd.read_csv(design_path), width="stretch", hide_index=True)
        st.warning(
            "Candidates are hypotheses for a new experiment, not validated interventions. "
            "They are ranked by predicted effect, interval width and animal coverage."
        )
    else:
        st.caption("Run `python scripts/phase6_design.py` to generate candidates.")

with tabs[11]:
    p7 = ROOT / "reports" / "phase7"
    st.subheader("Pseudo-prospective active target acquisition")
    st.error(
        "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED. "
        "Active acquisition selects targets inside a computational simulation. "
        "It is not a wet-lab closed loop and does not estimate causal effects."
    )
    metrics_path = p7 / "metrics.csv"
    summary_path = p7 / "summary.json"
    if summary_path.exists():
        s7 = json.loads(summary_path.read_text(encoding="utf-8"))
        h1 = s7.get("primary_hybrid_vs_random", [{}])[0]
        if s7.get("h1_supported"):
            st.success("H1 supported at the preregistered 40% gate.")
        else:
            st.warning(
                "H1 NOT SUPPORTED: hybrid minus random = "
                f"{h1.get('delta', float('nan')):.4f} "
                f"(95% CI {h1.get('ci_low', float('nan')):.4f} to "
                f"{h1.get('ci_high', float('nan')):.4f}); the +0.02 gate was not met."
            )
    if metrics_path.exists():
        m = pd.read_csv(metrics_path)
        primary = m[m["budget"].round(2).eq(0.40)]
        if not primary.empty:
            pivot = (
                primary.groupby("strategy")[["pair_spearman", "profile_normalized", "coverage_calibrated"]]
                .mean()
                .reset_index()
                .sort_values("pair_spearman", ascending=False)
            )
            st.markdown("**Primary budget: 40% of all targets**")
            st.dataframe(pivot, width="stretch", hide_index=True)
        curve = (
            m.groupby(["strategy", "budget"])["pair_spearman"]
            .mean()
            .reset_index()
            .pivot(index="budget", columns="strategy", values="pair_spearman")
        )
        st.markdown("**Budget curves (mean pair Spearman)**")
        st.line_chart(curve)
        st.dataframe(
            m[
                [
                    c
                    for c in [
                        "strategy",
                        "budget",
                        "n_targets",
                        "pair_spearman",
                        "profile_normalized",
                        "coverage_calibrated",
                        "interval_score",
                    ]
                    if c in m.columns
                ]
            ].tail(40),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "The outer test targets are frozen before acquisition. The primary gate is "
            "hybrid versus random at the 40% budget; negative results are reported. "
            "Profile/ceiling values above 1 indicate an underestimated reliability ceiling; "
            "use pair Spearman for decisions."
        )
    else:
        st.info("Phase 7 active metrics are not generated in this checkout.")
    design_path = p7 / "design" / "target_candidates.csv"
    if design_path.exists():
        st.markdown("**Next experiment candidates**")
        st.dataframe(pd.read_csv(design_path), width="stretch", hide_index=True)
        st.warning(
            "Candidates are hypotheses for a future experiment, ranked by predicted effect, "
            "uncertainty and anatomy-unexplained discrepancy. They are not validated interventions."
        )
    else:
        st.caption("Run `python scripts/phase7_design.py` to generate candidates.")

with tabs[12]:
    p8 = ROOT / "reports" / "phase8"
    st.subheader("Phase 8 — Adaptive Perturbation Twin")
    st.error(
        "MODEL-DEPENDENT VIRTUAL PERTURBATION — NOT CAUSALLY VALIDATED. "
        "Support-set personalisation and sequence rollouts are computational probes, "
        "not wet-lab closed-loop interventions or causal estimates."
    )
    readout_path = p8 / "readout.json"
    metrics_path = p8 / "metrics.csv"
    if readout_path.exists():
        r8 = json.loads(readout_path.read_text(encoding="utf-8"))
        d = r8.get("decisions", {})
        h1 = d.get("H1_primary", {})
        h2 = d.get("H2_personalisation", {})
        h3 = d.get("H3_active_eig", {})
        h4 = d.get("H4_sequence", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("H1 APT vs CRO", "supported" if h1.get("supported") else "not supported")
        c2.metric("H2 k=8 vs k=0", "supported" if h2.get("supported") else "not supported")
        c3.metric("H3 EIG vs random", "supported" if h3.get("supported") else "not supported")
        c4.metric("H4 sequence", "supported" if h4.get("supported") else "not supported")
        st.markdown(
            f"**H1 pair Δ** {h1.get('pair_delta')} "
            f"(CI {h1.get('pair_ci')}); trajectory Δ {h1.get('trajectory_delta')}.  \n"
            f"**H2 Δ** {h2.get('delta')} (CI {h2.get('ci')}).  \n"
            f"**H3 Δ** {h3.get('delta')} (CI {h3.get('ci')}).  \n"
            f"**H4 trajectory Δ** {h4.get('trajectory_delta')}, "
            f"coverage {h4.get('coverage')}."
        )
        if not r8.get("audit_clean", False):
            st.error("Leakage audit did not pass; treat these results as invalid.")
        else:
            st.caption(
                "Leakage audit passed: no animal overlap, support/test overlap, or test-event leakage."
            )
        table = r8.get("deployment_readout", {}).get("strategy_table_k8", [])
        if table:
            st.markdown("**Support strategy at the primary budget (k=8)**")
            st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
        budget = r8.get("deployment_readout", {}).get("random_budget_curve", [])
        if budget:
            st.markdown("**Random-support budget curve**")
            st.dataframe(pd.DataFrame(budget), width="stretch", hide_index=True)
        st.warning(
            "A null H2/H3 is an informative negative result about animal-level "
            "calibration in this atlas; it is not evidence that no effect exists."
        )
    elif metrics_path.exists():
        st.info("Run `python scripts/phase8_readout.py` to generate the dashboard readout.")
    else:
        st.info("Phase 8 outputs are not generated in this checkout.")
        st.caption("Run `python scripts/phase8_run.py --config configs/phase8/apt.yaml`.")

with tabs[13]:
    st.subheader("BoundedBench: measurement ceiling and claim status")
    status = load_measurement_status(ROOT / "reports")
    if status is not None:
        label = {
            "phase14": "Phase 14 zero-inflated measurement audit",
            "phase13": "Phase 13 measurement audit",
            "phase12r": "Phase 12R historical hierarchical audit",
            "phase12": "Phase 12 historical hierarchical audit",
        }.get(status.phase, status.phase)
        st.markdown(f"**{label}**")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Current R(22)", f"{status.primary_estimate:.3f}")
        p2.metric(
            "Reference R(22)",
            f"{status.reference_estimate:.3f}" if status.reference_estimate is not None else "not reported",
        )
        p3.metric("Failed gates", len(status.failed_gates))
        p4.metric("Decision", status.decision)
        if not status.adopted:
            st.warning(
                "The hierarchical interval is not calibrated. The current measurement gate "
                f"failures are: {', '.join(status.failed_gates) or 'not reported'}. The 0.560 "
                "coverage-aware ceiling remains the only descriptive bound; no calibrated "
                "interval, target ranking, animal-budget guarantee, or causal effect is claimed."
            )
        if status.gates:
            gate_rows = [
                {
                    "gate": name,
                    "passed": bool(gate.get("passed", False)),
                    "value": str(gate.get("value", "")),
                    "requirement": str(gate.get("requirement", "")),
                }
                for name, gate in status.gates.items()
            ]
            st.dataframe(pd.DataFrame(gate_rows), width="stretch", hide_index=True)
        st.caption(
            f"Phase {status.phase} status {status.decision} | model {status.model} | git "
            f"{str(status.manifest.get('git_revision', 'unknown'))[:12]} | dirty "
            f"{status.manifest.get('git_dirty', 'unknown')} | manifest {status.checksum}"
        )
        if status.historical:
            st.info("No Phase 13/14 report is available; this is a historical fallback.")
    else:
        st.info("No measurement-audit report is available yet.")
    phase11 = ROOT / "reports" / "phase11"
    phase10 = ROOT / "reports" / "phase10"
    phase9 = ROOT / "reports" / "phase9"
    card_path = phase11 / "benchmark_card.json"
    sensitivity_path = phase11 / "coverage_sensitivity.csv"
    manifest_path = phase11 / "manifest.json"
    if card_path.exists() and sensitivity_path.exists():
        card = json.loads(card_path.read_text(encoding="utf-8"))
        sensitivity = pd.read_csv(sensitivity_path)
        row = sensitivity[sensitivity["min_responder_coverage"] == 0.8].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pooled reliability (80% coverage)", f"{row['pooled_reliability_at_fold']:.3f}")
        c2.metric("Pooled ceiling", f"{row['pooled_ceiling_at_fold']:.3f}")
        c3.metric("Targets", f"{int(row['n_targets'])}")
        c4.metric("Best pair Spearman", f"{card['leaderboard']['best_pair_spearman']:.3f}")
        st.caption(
            "Coverage-aware Phase 11 values. The Phase 10 sparse-coverage ceiling (0.833) "
            "is shown below only as a historical comparison; the extrapolation gate failed."
        )
        provenance = {
            "benchmark": card["benchmark"],
            "version": card["version"],
            "ceiling": card["ceiling"]["value"],
            "git_revision": "unknown",
            "git_dirty": "unknown",
            "manifest_sha256": "missing",
        }
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            provenance["git_revision"] = str(manifest.get("git_revision", "unknown"))[:12]
            provenance["git_dirty"] = str(manifest.get("git_dirty", "unknown"))
            checksum_path = phase11 / "manifest.sha256"
            if checksum_path.exists():
                provenance["manifest_sha256"] = checksum_path.read_text(encoding="utf-8").split()[0][:12]
        st.caption(
            "Card {version} | ceiling {ceiling:.3f} | git {git_revision} | dirty {git_dirty} | "
            "manifest {manifest_sha256}".format(**provenance)
        )
        if provenance["git_dirty"] == "True":
            st.warning("Artifacts were generated from a dirty worktree; treat the provenance as provisional.")
        summary = None
        curve_path = phase10 / "ceiling_curve.csv"
        table_path = phase10 / "reliability_by_target.csv"
        claims_path = phase10 / "claims.json"
    else:
        report_dir = phase10 if (phase10 / "reliability_summary.json").exists() else phase9
        summary_path = report_dir / "reliability_summary.json"
        claims_path = report_dir / "claims.json"
        table_path = report_dir / "reliability_by_target.csv"
        curve_path = report_dir / "ceiling_curve.csv"
        summary = None
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            st.warning(
                "Phase 11 artifacts are absent; showing the historical Phase 10/9 report. "
                "Its pooled ceiling is coverage-sensitive."
            )
    if (card_path.exists() and sensitivity_path.exists()) or summary is not None:
        if summary is not None:
            is_v2 = summary.get("schema_version") == 2
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Repeat reliability", f"{summary['mean_repeat_reliability']:.3f}")
            c2.metric("Single-animal reliability", f"{summary['mean_animal_reliability']:.3f}")
            c3.metric("Single-animal ceiling", f"{summary['mean_animal_ceiling']:.3f}")
            if is_v2:
                c4.metric(
                    f"Pooled ceiling (n={int(summary['fold_size'])})",
                    f"{summary['pooled_ceiling_at_fold']:.3f}",
                )
                st.caption(
                    "The pooled ceiling applies to profiles averaged over evaluation animals. "
                    "The single-animal ceiling applies only to per-animal claims."
                )
            else:
                c4.metric("Fold size", "-")
        if curve_path.exists():
            curve = pd.read_csv(curve_path)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=curve["n_eval"],
                    y=curve["mean_reliability"],
                    mode="lines+markers",
                    name="mean reliability",
                    line=dict(color="#2b6cb0"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=curve["n_eval"],
                    y=curve["ceiling"],
                    mode="lines+markers",
                    name="attenuation ceiling",
                    line=dict(color="#c05621", dash="dash"),
                )
            )
            fig.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="Animals pooled",
                yaxis_title="Reliability",
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig, width="stretch")
            if "n_extrapolated" in curve.columns and curve["n_extrapolated"].max() > 0:
                st.caption(
                    "Points beyond a target's observed animal count are model extrapolations "
                    "and are descriptive only."
                )
        if table_path.exists():
            df = pd.read_csv(table_path)
            column = "single_animal_reliability"
            if column in df.columns:
                hist = go.Figure()
                hist.add_trace(
                    go.Histogram(
                        x=df[column].dropna(),
                        nbinsx=25,
                        marker_color="#c05621",
                        name="single animal",
                    )
                )
                hist.add_vline(x=0.5, line_dash="dash", annotation_text="adequate = 0.5")
                hist.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=30, b=10),
                    xaxis_title="Reliability",
                    yaxis_title="Targets",
                    showlegend=False,
                )
                st.plotly_chart(hist, width="stretch")
            st.dataframe(
                df.sort_values(column, ascending=False).head(15),
                width="stretch",
                hide_index=True,
            )
        if claims_path.exists():
            claims = json.loads(claims_path.read_text(encoding="utf-8"))["claims"]
            st.markdown("**Claims ledger**")
            st.dataframe(pd.DataFrame(claims), width="stretch", hide_index=True)
        st.warning(
            "A reliability ceiling is a measurement property, not a causal effect. "
            "A difference below the design MDE is not interpreted, even if its "
            "interval excludes zero. Target-level reliability rankings do not "
            "replicate across animal halves."
        )
    else:
        st.info("No reliability reports are generated in this checkout.")
        st.caption("Run `python scripts/phase10_reliability.py` and `python scripts/phase10_claims.py`.")

st.divider()
st.caption(EPISTEMIC_WARNING)
