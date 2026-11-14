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
from neurobranch.bundlenet_sim.handoff import build_gap_packet  # noqa: E402
from neurobranch.bundlenet_sim.intake import (  # noqa: E402
    validate_annotation_uploads,
    validate_trial_upload,
)
from neurobranch.bundlenet_sim.premotor import PremotorParameters  # noqa: E402
from neurobranch.bundlenet_sim.scope import (  # noqa: E402
    ANSWER_SOURCES,
    AVAILABLE_DATA,
    PRIMARY_ENDPOINTS,
    PRIMARY_OBJECTS,
    SUCCESS_CRITERIA,
    TARGET_MODES,
    ScopeAnswers,
)
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

NAV_LABELS = [
    "Overview",
    "Scope & decisions",
    "Tier 1 · BunDLe-Net",
    "Tier 2 · NC-MCM",
    "Tier 3 · Premotor",
    "Evidence report",
    "Evidence & lab",
    "Provenance",
]

STUDIO_CSS = """
<style>
:root {
  --ink: #10233f;
  --muted: #5b6b82;
  --line: #dbe3ef;
  --paper: #ffffff;
  --canvas: #f4f7fb;
  --navy: #0d2137;
  --teal: #0f6b5f;
  --blue: #1f4e79;
  --amber: #9a620c;
}
html, body, [class*="css"] {
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.stApp {
  background:
    radial-gradient(1100px 420px at 8% -8%, rgba(31, 78, 121, 0.10), transparent 60%),
    radial-gradient(900px 380px at 96% 0%, rgba(15, 107, 95, 0.08), transparent 55%),
    var(--canvas);
  color: var(--ink);
}
header[data-testid="stHeader"] {
  background: rgba(244, 247, 251, 0.78);
  backdrop-filter: blur(14px);
  border-bottom: 1px solid rgba(219, 227, 239, 0.75);
}
.block-container {
  padding-top: 1.9rem;
  padding-bottom: 3.5rem;
  max-width: 1480px;
}
[data-testid="stSidebar"] {
  background: #edf2f8;
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div:first-child {
  padding-top: 1.1rem;
}
.brand-card {
  background: linear-gradient(145deg, #0d2137 0%, #173d5c 62%, #0f5b55 100%);
  color: #f5f9ff;
  border-radius: 16px;
  padding: 1rem 1rem 0.95rem;
  margin-bottom: 1rem;
  box-shadow: 0 10px 30px rgba(13, 33, 55, 0.18);
}
.brand-mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 2.2rem;
  height: 2.2rem;
  border-radius: 10px;
  border: 1px solid rgba(255,255,255,0.35);
  font-weight: 800;
  letter-spacing: 0.06em;
  font-size: 0.72rem;
}
.brand-word {
  margin-top: 0.6rem;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 1.05rem;
  line-height: 1.2;
}
.brand-word span { color: #9dd8cf; }
.brand-sub {
  margin-top: 0.35rem;
  color: #c8d6e5;
  font-size: 0.72rem;
  line-height: 1.35;
}
.side-kicker {
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.66rem;
  font-weight: 700;
  margin: 0.6rem 0 0.35rem;
}
.hero {
  background: linear-gradient(135deg, #0d2137 0%, #173d5c 58%, #0f5b55 100%);
  color: #f7fbff;
  border-radius: 24px;
  padding: clamp(1.6rem, 3vw, 2.8rem);
  margin-bottom: 1.2rem;
  box-shadow: 0 20px 55px rgba(13, 33, 55, 0.20);
  position: relative;
  overflow: hidden;
}
.hero:after {
  content: "";
  position: absolute;
  right: -8%;
  bottom: -55%;
  width: 55%;
  height: 150%;
  border-radius: 50%;
  border: 1px solid rgba(157, 216, 207, 0.25);
  box-shadow: 0 0 0 40px rgba(157, 216, 207, 0.05), 0 0 0 90px rgba(157, 216, 207, 0.035);
}
.hero-kicker {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  color: #a9dcd5;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
.hero-kicker:before {
  content: "";
  width: 0.45rem;
  height: 0.45rem;
  border-radius: 50%;
  background: #9dd8cf;
  box-shadow: 0 0 0 5px rgba(157, 216, 207, 0.12);
}
.hero h1 {
  font-family: Georgia, "Times New Roman", serif;
  font-size: clamp(2.05rem, 4.2vw, 3.35rem);
  line-height: 1.04;
  letter-spacing: -0.025em;
  margin: 0.7rem 0 0.7rem;
  color: #ffffff;
  max-width: 900px;
}
.hero p {
  color: #d9e6f2;
  font-size: clamp(0.95rem, 1.4vw, 1.08rem);
  line-height: 1.55;
  max-width: 850px;
  margin: 0;
}
.hero-chips { margin-top: 1.1rem; display: flex; flex-wrap: wrap; gap: 0.45rem; }
.chip {
  display: inline-block;
  border: 1px solid rgba(255,255,255,0.24);
  background: rgba(255,255,255,0.08);
  color: #e9f4f2;
  border-radius: 999px;
  padding: 0.25rem 0.62rem;
  font-size: 0.7rem;
  letter-spacing: 0.02em;
}
.page-head {
  border-bottom: 1px solid var(--line);
  padding-bottom: 1rem;
  margin-bottom: 1rem;
}
.eyebrow {
  color: var(--teal);
  font-size: 0.67rem;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
.page-head h2 {
  font-family: Georgia, "Times New Roman", serif;
  color: var(--ink);
  font-size: clamp(1.55rem, 2.6vw, 2.25rem);
  letter-spacing: -0.02em;
  margin: 0.25rem 0 0.35rem;
}
.page-head p {
  color: var(--muted);
  max-width: 980px;
  margin: 0 0 0.6rem;
  line-height: 1.5;
}
.badge {
  display: inline-block;
  border-radius: 999px;
  padding: 0.22rem 0.6rem;
  font-size: 0.67rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  border: 1px solid;
}
.badge-green { color: #136744; background: #eaf7ef; border-color: #b9e2ca; }
.badge-blue { color: #184d78; background: #edf5fc; border-color: #c4dced; }
.badge-amber { color: #84540b; background: #fdf5e5; border-color: #edd5a5; }
.badge-red { color: #8a2f34; background: #fdf0f1; border-color: #efc4c7; }
.card-eyebrow {
  color: var(--teal);
  font-size: 0.64rem;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.section-title {
  font-family: Georgia, "Times New Roman", serif;
  color: var(--ink);
  font-size: 1.3rem;
  margin: 0.35rem 0 0.2rem;
}
.section-copy { color: var(--muted); line-height: 1.5; }
.stMainBlockContainer div[data-testid="stVerticalBlock"][data-test-wrap="false"] {
  background: rgba(255,255,255,0.9);
  border: 1px solid var(--line) !important;
  border-radius: 16px;
  box-shadow: 0 8px 24px rgba(16, 35, 63, 0.045);
}
div[data-testid="stMetric"] {
  background: rgba(255,255,255,0.9);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 0.8rem 0.9rem;
  box-shadow: 0 6px 18px rgba(16, 35, 63, 0.04);
}
div[data-testid="stMetricLabel"] p {
  color: var(--muted);
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
div[data-testid="stMetricValue"] {
  color: var(--ink);
  font-family: Georgia, "Times New Roman", serif;
  font-size: 1.65rem;
}
div[data-testid="stDataFrame"] {
  border: 1px solid var(--line);
  border-radius: 12px;
  overflow: hidden;
  background: #fff;
}
div[data-testid="stExpander"] details {
  border: 1px solid var(--line);
  border-radius: 12px;
  background: rgba(255,255,255,0.72);
}
.stButton > button, .stDownloadButton > button {
  border-radius: 999px;
  border: 1px solid #cbd7e5;
  font-weight: 600;
  padding: 0.38rem 0.95rem;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--teal);
  color: var(--teal);
}
button[kind="primary"] {
  background: linear-gradient(135deg, #0f6b5f, #1f4e79);
  border: none;
}
[data-testid="stSidebar"] [role="radiogroup"] label {
  border-radius: 10px;
  padding: 0.38rem 0.55rem;
  margin-bottom: 0.08rem;
  transition: background 120ms ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
  background: rgba(255,255,255,0.7);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: #ffffff;
  border: 1px solid var(--line);
}
[data-testid="stSidebar"] hr { border-color: var(--line); }
.side-status { color: #31465f; font-size: 0.78rem; line-height: 1.55; }
.side-status b { color: var(--ink); }
.side-foot { color: #73839a; font-size: 0.68rem; line-height: 1.45; }
@media (max-width: 1120px) {
  .stMainBlockContainer div[data-testid="stHorizontalBlock"]:has(.card-eyebrow) {
    flex-direction: column;
  }
  .stMainBlockContainer div[data-testid="stHorizontalBlock"]:has(.card-eyebrow) > div[data-testid="stColumn"] {
    width: 100% !important;
    flex: 1 1 100% !important;
  }
}
@media (max-width: 900px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .hero { border-radius: 18px; }
  .stMainBlockContainer div[data-testid="stHorizontalBlock"]:has(div[data-testid="stMetric"]) {
    flex-wrap: wrap;
  }
  .stMainBlockContainer div[data-testid="stHorizontalBlock"]:has(div[data-testid="stMetric"]) > div[data-testid="stColumn"] {
    flex: 1 1 45% !important;
    width: 45% !important;
    min-width: 150px;
  }
}
</style>
"""


def inject_styles() -> None:
    st.markdown(STUDIO_CSS, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, description: str, badge: str, badge_kind: str = "blue") -> None:
    st.markdown(
        f"""
<section class="page-head">
  <div class="eyebrow">{eyebrow}</div>
  <h2>{title}</h2>
  <p>{description}</p>
  <span class="badge badge-{badge_kind}">{badge}</span>
</section>
""",
        unsafe_allow_html=True,
    )


def _navigate(label: str) -> None:
    st.session_state["workspace_page"] = label


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
<div class="brand-card">
  <div class="brand-mark">VN</div>
  <div class="brand-word">Vienna Neuroinformatics<br><span>Studio</span></div>
  <div class="brand-sub">Research Group Neuroinformatics<br>University of Vienna</div>
</div>
""",
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Workspace",
            NAV_LABELS,
            label_visibility="collapsed",
            key="workspace_page",
        )
        st.divider()
        st.markdown('<div class="side-kicker">Release status</div>', unsafe_allow_html=True)
        st.markdown(
            """
<div class="side-status">
<b>Prototype v2</b> · lab-facing<br>
✓ contract suite · local + pod<br>
✓ Julia trace parity 0.9999998<br>
✓ external reversal audit complete<br>
△ clamped replay is not a validated predictor<br>
△ scope unconfirmed · needs Denis
</div>
""",
            unsafe_allow_html=True,
        )
        st.divider()
        pdf_path = ROOT / "docs" / "VIENNA_PRODUCT_BRIEF.pdf"
        if pdf_path.exists():
            st.download_button(
                "Download product brief (PDF)",
                data=pdf_path.read_bytes(),
                file_name="Vienna_Neuroinformatics_Studio_brief.pdf",
                mime="application/pdf",
                key="sidebar_pdf",
                width="stretch",
            )
        st.markdown(
            '<div class="side-foot">Prepared for lab review; scope not yet confirmed by Denis. '
            "Papers remain attributed to their authors; no digital-twin claim.</div>",
            unsafe_allow_html=True,
        )
    return page


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


def behavior_trace(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, labels: np.ndarray, name: str
) -> go.Scatter3d:
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
    page_header(
        "Tier 1 · latent manifold",
        "BunDLe-Net simulator",
        "Published 3-D BunDLe-Net embeddings; a data-driven manifold walk and a quadratic "
        "forecast are shown separately. This is a behaviourally relevant compression, not a "
        "biophysical or causal model.",
        "Descriptive · held-out-worm benchmark",
        "green",
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
                coverage_cal = item.get("interval_coverage_90_h10_calibrated", {}).get("mean", float("nan"))
                scale90 = item.get("interval_scale_90_h10", {}).get("mean", float("nan"))
                rows.append(
                    {
                        "model": item["model"],
                        "latent MSE h10": round(item["latent_mse_h10"]["mean"], 3),
                        "CRPS h10": round(item.get("latent_crps_h10", {}).get("mean", float("nan")), 3),
                        "90% raw": "n/a" if not np.isfinite(coverage) else f"{coverage:.1%}",
                        "90% calibrated": "n/a" if not np.isfinite(coverage_cal) else f"{coverage_cal:.1%}",
                        "scale ×": "n/a" if not np.isfinite(scale90) else f"{scale90:.2f}",
                        "behaviour acc h10": f"{item['behavior_accuracy_h10']['mean']:.1%}",
                        "behaviour acc h25": f"{item['behavior_accuracy_h25']['mean']:.1%}",
                        "behaviour NLL h10": round(
                            item.get("behavior_nll_h10", {}).get("mean", float("nan")), 3
                        ),
                        "transition JS h25": round(item["behavior_transition_js"]["mean"], 3),
                    }
                )
            st.dataframe(rows, hide_index=True, width="stretch")
            st.caption(
                "Open-loop forecast from 75-frame-spaced starts, 25-frame horizon, 16 seeds. "
                "Raw 90% ensemble intervals under-cover (68–75%). Nested training-worm conformal "
                "widening raises held-out coverage to 80–87%, still below nominal; intervals remain "
                "diagnostic, not calibrated. CRPS and NLL are proper scores; lower is better."
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
            ensemble = wb.tier1_ensemble(worm_id, start, min(horizon, 120), seeds=32, temperature=temperature)
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
    page_header(
        "Tier 2 · cognitive states",
        "NC-MCM state workbench",
        "Precomputed NC-MCM models: behaviour probabilities → k-means cognitive states → "
        "Markov tests. The UI exposes the fitted descriptive model; it does not identify do().",
        "Assumption-dependent · strict Markov/stationarity rejected",
        "amber",
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
                        "markov_order": lambda r: (
                            f"p_MC={r.get('p_monte_carlo', float('nan')):.3f}, LR={r.get('lr', float('nan')):.1f}"
                        ),
                        "markov_order_forecast": lambda r: (
                            f"ΔNLL={r.get('mean_delta_nll_order1_minus_order2', float('nan')):.3f}, "
                            f"95% CI [{r.get('delta_ci95', [float('nan'), float('nan')])[0]:.3f}, "
                            f"{r.get('delta_ci95', [float('nan'), float('nan')])[1]:.3f}], "
                            f"{r.get('n_worms_positive', 0)}/{r.get('n_worms', 0)} worms positive"
                        ),
                        "stationarity": lambda r: (
                            f"p_MC={r.get('p_monte_carlo', float('nan')):.3f}, JS={r.get('js_first_second_half', float('nan')):.3f}"
                        ),
                        "behavior_transport": lambda r: (
                            f"held-out JS={r.get('heldout_js', float('nan')):.3f}, split-half={r.get('split_half_js', float('nan')):.3f}"
                        ),
                        "state_behavior_mi": lambda r: (
                            f"MI={r.get('mutual_information', float('nan')):.3f}, p={r.get('p_permutation', float('nan')):.3f}"
                        ),
                        "k_sensitivity": lambda r: f"max TV={r.get('max_tv_vs_reference', float('nan')):.3f}",
                    }.get(test_name, lambda r: "")(result),
                }
            )
        st.dataframe(rows, hide_index=True, width="stretch")
        forecast = audit.get("tests", {}).get("markov_order_forecast", {})
        if forecast:
            st.info(
                "Order-2 follow-up: positive ΔNLL means the second-order chain predicts the "
                "held-out half better. This is a predictive comparison, not a causal test; "
                "the fitted order-2 chain still needs external validation before any intervention."
            )
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
        "Intrinsic neuron",
        list(names),
        index=names.index("AVBL") if "AVBL" in names else 0,
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
                            "model false alarms / min": _num(model_ev.get("false_alarm_episodes_per_min"), 2),
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
        header="time_s,"
        + ",".join(f"recorded_{n}" for n in names)
        + ","
        + ",".join(f"simulated_{n}" for n in names),
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
    page_header(
        "Tier 3 · premotor replay",
        "Clamped-core and free-run sandbox",
        "The published 15-intrinsic / 69-signal clamped core and a separate 205-node free-run. "
        "The clamped replay is implementation-faithful; the external audit shows delayed, "
        "retrospective behavioural information, not a validated forecast.",
        "Implementation-faithful · not a digital twin",
        "blue",
    )
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
        ablate_text = st.text_input("Ablate nodes (names or indices, comma-separated)", "", key="t3_ablate")

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
        trace = go.Figure(go.Scatter(x=t, y=y[int(node)], mode="lines", line={"color": "#E45756"}))
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
    page_header(
        "Provenance · scope",
        "Provenance, scope and limitations",
        "The exact sources, artifact hashes and claim boundaries behind the studio. Use this "
        "page when a result needs to leave the room.",
        "Source-traceable",
        "green",
    )
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
    st.markdown(
        """
<section class="hero">
  <div class="hero-kicker">Research Group Neuroinformatics · University of Vienna</div>
  <h1>From neural manifolds toward tested mechanisms.</h1>
  <p>
    A lab-facing studio for <i>C. elegans</i> whole-brain data: a latent-manifold simulator,
    an NC-MCM state-and-assumption workbench, and an implementation-faithful premotor replay.
    Three objects, three claim levels, one evidence trail.
  </p>
  <div class="hero-chips">
    <span class="chip">Prototype v2</span>
    <span class="chip">Source-traceable</span>
    <span class="chip">Not a digital twin</span>
    <span class="chip">Scope unconfirmed</span>
    <span class="chip">Built for a first lab review</span>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Three objects, three claim levels</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Choose the object that matches the question; do not collapse '
        "them into a single “model of the worm”.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1, st.container(border=True):
        st.markdown('<div class="card-eyebrow">Tier 1 · descriptive</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Latent manifold</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">Generate 3-D BunDLe-Net trajectories and inspect a '
            "held-out-worm forecast against persistence and switching baselines.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<span class="badge badge-green">Behaviourally relevant compression</span>',
            unsafe_allow_html=True,
        )
        st.button(
            "Open Tier 1", key="ov_t1", on_click=_navigate, args=("Tier 1 · BunDLe-Net",), width="stretch"
        )
    with c2, st.container(border=True):
        st.markdown('<div class="card-eyebrow">Tier 2 · assumption-dependent</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">NC-MCM state workbench</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">Inspect cognitive states, Markov tests, transport, '
            "K sensitivity, and the difference between a counterfactual and an identified effect.</div>",
            unsafe_allow_html=True,
        )
        st.markdown('<span class="badge badge-amber">do() is not identified</span>', unsafe_allow_html=True)
        st.button("Open Tier 2", key="ov_t2", on_click=_navigate, args=("Tier 2 · NC-MCM",), width="stretch")
    with c3, st.container(border=True):
        st.markdown(
            '<div class="card-eyebrow">Tier 3 · implementation-faithful</div>', unsafe_allow_html=True
        )
        st.markdown('<div class="section-title">Premotor replay</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-copy">Run the published 84/15/69 clamped core, compare with '
            "β=0 and zero signals, and inspect the independent reversal-event audit.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<span class="badge badge-blue">Julia parity · not a digital twin</span>',
            unsafe_allow_html=True,
        )
        st.button(
            "Open Tier 3", key="ov_t3", on_click=_navigate, args=("Tier 3 · Premotor",), width="stretch"
        )

    st.write("")
    metrics = wb.metrics()
    parity = wb.tier3_reference_all()
    behaviour = wb.tier3_behavior_validation()
    parity_mean = (parity or {}).get("aggregate", {}).get("mean_pearson_r")
    beh_agg = (behaviour or {}).get("aggregate", {})
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    m1.metric(
        "Tier 1 one-step R²",
        f"{metrics.get('mean_one_step_r2', float('nan')):.3f}" if metrics else "n/a",
        help="Leave-one-worm-out; descriptive one-step predictability, not mechanism.",
    )
    m2.metric(
        "Julia trace parity",
        f"{parity_mean:.7f}" if parity_mean is not None else "n/a",
        help="Three validation recordings; max |Δ| ≈ 0.013. Validates the port, not the biology.",
    )
    m3.metric(
        "Held-out worms",
        "5",
        help="BunDLe-Net release; no sixth held-out worm. Not a population claim.",
    )
    m4.metric(
        "Reversal AUC: model / persistence",
        f"{beh_agg.get('mean_model_lag0_auc', float('nan')):.3f} / "
        f"{beh_agg.get('mean_persistence_auc', float('nan')):.3f}",
        help="The model does not beat a one-frame persistence baseline. See Tier 3 for the retrospective-vs-prospective audit.",
    )

    st.write("")
    st.markdown('<div class="section-title">Project status at a glance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">A defensible prototype is not the same as a finished scientific '
        "result. This is the current boundary.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    ready, missing = st.columns(2, gap="medium")
    with ready, st.container(border=True):
        st.markdown('<div class="card-eyebrow">Ready now</div>', unsafe_allow_html=True)
        st.markdown(
            """
- Three working objects with explicit claim levels.
- Held-out-worm Tier-1 benchmark with CRPS, coverage, NLL, baselines and nested conformal calibration audit.
- NC-MCM assumption audit, order-2 forecast follow-up, transport test and K sensitivity.
- Published 84/15/69 clamped core, β=0 and zero-signal controls.
- Python ↔ Julia parity on three validation recordings.
- Baseline-controlled, lag-aware external reversal audit with bootstrap CIs.
- Reproducible artifacts, contract suite, product brief and claim ledger.
"""
        )
    with missing, st.container(border=True):
        st.markdown('<div class="card-eyebrow">Still missing</div>', unsafe_allow_html=True)
        st.markdown(
            """
- A sixth held-out worm or an external cohort.
- An intervention dataset (optogenetic, ablation, microfluidic) for a real `do()` test.
- Independent labels for all eight behaviours, not only binary reversal.
- Fully calibrated Tier-1 intervals; conformal widening helps but still under-covers.
- A validated behavioural predictor; the clamped replay does not beat persistence.
- BCI/EEG/fMRI or MeDIL integration; these are not in this release.
- A causal, individually calibrated whole-brain digital twin.
"""
        )

    st.write("")
    st.markdown('<div class="section-title">A 20-minute first review</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">A suggested path that surfaces both the strongest result and '
        "the most important negative result.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown("**0 · Scope — what does Denis actually want?**")
    st.caption(
        "Start on Scope & decisions. No answer is preselected; a completed form is a transcription, "
        "not a signed requirement."
    )
    s1, s2 = st.columns(2, gap="medium")
    with s1, st.container(border=True):
        st.markdown("**1 · Tier 1 — what is predictable?**")
        st.caption(
            "Pick a held-out worm; compare manifold walk with switching linear; check the 90% coverage before trusting a band."
        )
        st.markdown("**2 · Tier 2 — what survives an assumption audit?**")
        st.caption(
            "Choose K; inspect the Markov p-value; then open the assumption audit and the do() caveat."
        )
    with s2, st.container(border=True):
        st.markdown("**3 · Tier 3 — what is faithful, and what is not?**")
        st.caption(
            "Run the clamped core at β=10; compare β=0 and zero signals; open the external audit and read the prospective result."
        )
        st.markdown("**4 · Evidence — what can leave the room?**")
        st.caption("Download the evidence report, product brief and claim ledger before the meeting.")

    st.write("")
    st.warning(
        "Claim boundaries: Tier 1 is a latent-manifold model, not a circuit. Tier 2 is descriptive; "
        "its do() is a Markov counterfactual and strict first-order Markov/stationarity are rejected. "
        "Tier 3 is implementation-faithful, not a digital twin; the readout is a proxy, and the "
        "clamped replay has retrospective but not validated predictive behavioural information. "
        "No tier establishes whole-brain mechanism or causality."
    )

    with st.expander("What to ask Denis in the first meeting"):
        st.markdown(
            """
The canonical surface is **Scope & decisions**. Record the six answers in writing; do not infer
them from the group page or from this product.
1. Primary object: causal mechanism, external generalization, reusable workbench, or BCI/NeuroAI?
2. Primary endpoint: released eight states, binary reversal, turn direction, continuous kinematics, or other?
3. Which data actually exist, and under what access conditions?
4. What would count as success: positive result, rigorous negative, benchmark, wet-lab design, or demo?
5. If an intervention is in scope, which circuit, modality and positive control?
6. What is the stop rule, budget and deadline?
"""
        )


def report_tab(wb: ViennaWorkbench) -> None:
    page_header(
        "Evidence · portable handoff",
        "Evidence report and run bundle",
        "A portable summary for Denis; every number comes from a committed artifact and every "
        "limitation stays attached to the number.",
        "Audit-ready · v2 schema",
        "green",
    )
    behaviour_preview = wb.tier3_behavior_validation()
    preview = (behaviour_preview or {}).get("aggregate", {})
    r1, r2, r3, r4 = st.columns(4, gap="medium")
    r1.metric(
        "Readout AUC",
        f"{preview.get('mean_readout_auc', float('nan')):.3f}",
        help="Neural proxy against independent reversal intervals.",
    )
    r2.metric(
        "Model lag-0 AUC",
        f"{preview.get('mean_model_lag0_auc', float('nan')):.3f}",
        help="Clamped replay at the inferred alignment.",
    )
    r3.metric(
        "Retrospective AUC",
        f"{preview.get('mean_model_retrospective_auc', float('nan')):.3f}",
        help="Uses future model output; not a forecast.",
    )
    r4.metric(
        "Persistence AUC",
        f"{preview.get('mean_persistence_auc', float('nan')):.3f}",
        help="A one-frame persistence baseline. The model does not beat it.",
    )
    st.write("")
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
        lines += [
            "",
            "| model | latent MSE h10 | behaviour acc h10 | behaviour acc h25 | transition JS h25 |",
            "|---|---:|---:|---:|---:|",
        ]
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
        type="primary",
    )
    with st.expander("Preview the full evidence report", expanded=False):
        st.markdown(report)
    with st.expander("Machine-readable provenance", expanded=False):
        st.json(wb.provenance())


def gap_closure_section(wb: ViennaWorkbench) -> None:
    """Show the four evidence gates without implying that any gap is closed."""
    st.markdown("**Gap closure**")
    st.caption(
        "No row below is a scientific result. A row closes only when its artifact passes the "
        "validator and a frozen analysis is written. The brief is a draft until Denis confirms it."
    )
    st.dataframe(
        [
            {
                "gate": "Written specification",
                "status": "draft — awaiting Denis",
                "blocker": "six scope decisions",
                "next action": "send one-page brief",
            },
            {
                "gate": "Sixth held-out worm",
                "status": "blocked on data",
                "blocker": "no public sixth embedding; n=5",
                "next action": "manifest + shape gate",
            },
            {
                "gate": "Independent 8-state labels",
                "status": "blocked on data",
                "blocker": "no independent video annotation",
                "next action": "two blinded annotators",
            },
            {
                "gate": "Randomized intervention",
                "status": "blocked on data",
                "blocker": "no matched target/sham cohort",
                "next action": "power + QC + analysis ready",
            },
            {
                "gate": "BCI / EEG / fMRI / MeDIL",
                "status": "not started — separate lane",
                "blocker": "no human data or ethics path",
                "next action": "scope only if chosen",
            },
            {
                "gate": "Digital twin",
                "status": "not claimed — L0 only",
                "blocker": "no L2 causal validation",
                "next action": "readiness levels in brief",
            },
        ],
        hide_index=True,
        width="stretch",
    )
    documents = [
        ("Denis brief (Markdown)", wb.root / "docs" / "DENIS_REQUIREMENTS_BRIEF.md"),
        ("Gap matrix (Markdown)", wb.root / "docs" / "GAP_CLOSURE_MATRIX.md"),
        ("Data request package (Markdown)", wb.root / "docs" / "DATA_REQUEST_PACKAGE.md"),
        ("Annotation protocol (Markdown)", wb.root / "docs" / "ANNOTATION_PROTOCOL.md"),
        ("Intervention power plan (Markdown)", wb.root / "docs" / "INTERVENTION_POWER_PLAN.md"),
        ("Public-data memo (Markdown)", wb.root / "docs" / "PUBLIC_DATA_GAP_MEMO.md"),
    ]
    for start in range(0, len(documents), 3):
        columns = st.columns(3)
        for column, (label, path) in zip(columns, documents[start : start + 3], strict=False):
            with column:
                if path.exists():
                    st.download_button(
                        label,
                        data=path.read_bytes(),
                        file_name=path.name,
                        mime="text/markdown",
                        width="stretch",
                    )
                else:
                    st.caption(f"missing: {path.name}")
    power_path = wb.root / "reports" / "vienna_workbench" / "intervention_power.json"
    if power_path.exists():
        st.download_button(
            "Intervention power design (JSON)",
            data=power_path.read_bytes(),
            file_name=power_path.name,
            mime="application/json",
        )
        try:
            power = json.loads(power_path.read_text(encoding="utf-8"))
            designs = power.get("designs", [])
            if designs:
                summary = "; ".join(
                    f"{item['animals_per_arm']}/arm MDE {item['mde']:.2f}"
                    for item in designs
                    if "animals_per_arm" in item and "mde" in item
                )
                if summary:
                    st.caption(
                        f"Simulated design only, at ICC 0.2: {summary}. "
                        "This is not a lab result or a promise about the assay."
                    )
        except (OSError, ValueError, TypeError):
            st.caption("Power report exists but could not be summarized.")

    st.divider()
    try:
        packet = build_gap_packet(wb.root)
    except (OSError, ValueError) as exc:
        st.caption(f"Handoff packet unavailable: {exc}")
    else:
        st.download_button(
            "Download all gap documents (ZIP)",
            data=packet,
            file_name="vienna-gap-closure-packet.zip",
            mime="application/zip",
            width="stretch",
        )
        st.caption(
            "The ZIP contains the brief, contracts and power report plus a SHA-256 manifest. "
            "It transfers the open questions; it does not close them."
        )


def _show_gate_report(label: str, report: dict) -> None:
    if report.get("passed"):
        st.success(f"{label}: gate passed")
    else:
        st.error(f"{label}: gate failed")
    for error in report.get("errors") or []:
        st.markdown(f"- {error}")
    for warning in report.get("warnings") or []:
        st.caption(f"warning: {warning}")
    st.json(report)


def incoming_check_section() -> None:
    """Run the same incoming-data gates as the CLI without leaving the app."""
    st.markdown("**Incoming data check**")
    st.caption(
        "Processed only by this local app process and discarded after the check. "
        "Passing a gate is necessary, not sufficient, for a scientific claim."
    )
    with st.expander("Validate annotation or intervention files", expanded=False):
        annotation_tab, trial_tab = st.tabs(["Annotations", "Intervention trials"])
        with annotation_tab:
            annotation_files = st.file_uploader(
                "Annotation CSVs (one per blinded annotator)",
                type=["csv"],
                accept_multiple_files=True,
                key="intake_annotation_files",
            )
            if st.button("Check annotations", key="intake_annotation_run"):
                if not annotation_files:
                    st.warning("Add at least one annotation CSV.")
                else:
                    try:
                        report = validate_annotation_uploads(
                            [(item.name, item.getvalue()) for item in annotation_files]
                        )
                    except (OSError, ValueError) as exc:
                        st.error(str(exc))
                    else:
                        _show_gate_report("Annotation agreement", report)
        with trial_tab:
            trial_file = st.file_uploader(
                "Pre-unblinding intervention trial CSV",
                type=["csv"],
                key="intake_trial_file",
            )
            if st.button("Check trial table", key="intake_trial_run"):
                if trial_file is None:
                    st.warning("Add a trial CSV.")
                else:
                    try:
                        report = validate_trial_upload(
                            trial_file.name,
                            trial_file.getvalue(),
                        )
                    except (OSError, ValueError) as exc:
                        st.error(str(exc))
                    else:
                        _show_gate_report("Intervention trial table", report)


def lab_tab(wb: ViennaWorkbench) -> None:
    page_header(
        "Lab alignment · University of Vienna",
        "Evidence and research fit",
        "What the product claims, what the evidence supports, and how each tier maps to the "
        "Neuroinformatics group's published work. This is a lab-facing map, not a claim that "
        "Denis authored these papers.",
        "Lab-facing · attribution preserved",
        "blue",
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
                "status": "rejected; order-2 predicts better in 5/5 worms",
                "evidence": "tier2_assumption_audit.json (order-2 forecast follow-up)",
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
  ([stable team page](https://ni.cs.univie.ac.at/about-us/team/) ·
  [UCris profile](https://ucrisportal.univie.ac.at/en/persons/denis-kovacevic/)).
  No publications are attributed to him here; the checked pages list none.
"""
    )
    st.markdown("**Next experiments that would change the claim level**")
    st.markdown(
        """
1. **Reference fidelity gate:** completed — fresh Julia 1.11.2 runs match the Python port at
   mean trace r=0.9999998 on all three validation recordings.
2. **Sixth worm:** ingestion gate is ready — put the released embedding in
   `data/processed/bundlenet_sim/` and run `python scripts/validate_new_worm.py --worm-id 5`.
   No sixth worm exists in this release; the gate prevents accidental reuse of a fifth-worm split.
3. **Intervention dataset:** the pre-registered request and analysis plan are in
   `docs/VIENNA_INTERVENTION_PROTOCOL.md`; the current Randi atlas is evidence about
   propagation, not a Tier-1/2 test.
4. **Markov-order follow-up:** completed provisionally — an explicit order-2 chain lowers
   held-out NLL in 5/5 worms, but only 2/5 individually; this is a predictive result, not
   a causal mechanism. External-worm replication is still required before using `do()`.
"""
    )
    st.markdown("**Delivery snapshot**")
    st.dataframe(
        [
            {
                "area": "Tier 1 · latent manifold",
                "status": "ready descriptively",
                "what is missing": "external cohort; fully calibrated intervals",
            },
            {
                "area": "Tier 2 · NC-MCM",
                "status": "audit-ready; order-2 follow-up positive",
                "what is missing": "external order-2 replication; intervention data for do()",
            },
            {
                "area": "Tier 3 · premotor replay",
                "status": "parity-ready; behavioural validation negative",
                "what is missing": "perturbation experiment; validated predictor",
            },
            {
                "area": "Product / handoff",
                "status": "demo-ready",
                "what is missing": "sixth worm; eight-behaviour labels; lab priority decision",
            },
            {
                "area": "Lab extensions",
                "status": "not started",
                "what is missing": "BCI/EEG/fMRI and MeDIL integration",
            },
            {
                "area": "Causal digital twin",
                "status": "not supported",
                "what is missing": "intervention-validated mechanism; individual calibration",
            },
        ],
        hide_index=True,
        width="stretch",
    )
    st.markdown("**Behaviour-label coverage**")
    st.dataframe(
        [
            {
                "source": "BunDLe-Net release",
                "labels": "8 canonical behaviours",
                "independent?": "released annotation, not an external re-annotation",
                "used for": "Tier 1 decode/forecast target",
            },
            {
                "source": "NC-MCM release",
                "labels": "8 canonical behaviours",
                "independent?": "same annotation family as BunDLe-Net",
                "used for": "Tier 2 state–behaviour MI",
            },
            {
                "source": "Tier 3 neural readout",
                "labels": "binary forward/reverse/quiet",
                "independent?": "proxy from the paper's neural readout",
                "used for": "internal agreement only",
            },
            {
                "source": "reversal_events",
                "labels": "binary reversal intervals",
                "independent?": "release annotation; interval/offset inferred",
                "used for": "external binary check",
            },
            {
                "source": "full eight-behaviour video",
                "labels": "not present",
                "independent?": "would be independent",
                "used for": "next external validity test",
            },
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "No independent eight-behaviour video labels exist in this release. Tier 1 agreement is with "
        "released labels; Tier 3 AUC is a binary reversal proxy check. See docs/VIENNA_LABEL_COVERAGE.md."
    )
    gap_closure_section(wb)
    incoming_check_section()
    st.markdown("**Open protocols and scope**")
    st.caption(
        "These are prepared handoff documents, not data. The sixth-worm gate fails loudly "
        "until a real released recording is present."
    )
    protocol_paths = [
        ("Sixth-worm gate (Markdown)", wb.root / "docs" / "VIENNA_NEXT_WORM.md"),
        ("Intervention protocol (Markdown)", wb.root / "docs" / "VIENNA_INTERVENTION_PROTOCOL.md"),
        ("BCI / EEG / MeDIL scope (Markdown)", wb.root / "docs" / "VIENNA_SCOPE.md"),
    ]
    protocol_cols = st.columns(len(protocol_paths))
    for column, (label, path) in zip(protocol_cols, protocol_paths, strict=True):
        with column:
            if path.exists():
                st.download_button(
                    label,
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="text/markdown",
                    width="stretch",
                )
            else:
                st.caption(f"missing: {path.name}")
    st.download_button(
        "Download claim ledger (CSV)",
        data="claim,tier,status,evidence\n"
        "3-D latent manifold forecasting,Tier 1,supported descriptively,held-out benchmark\n"
        "NC-MCM first-order Markov and stationary,Tier 2,rejected; order-2 lower held-out NLL in 5/5 worms,tier2 audit + order-2 follow-up\n"
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


def _scope_label_to_id(options: dict[str, str], label: str | None) -> str | None:
    if label is None:
        return None
    for item_id, item_label in options.items():
        if item_label == label:
            return item_id
    return None


def scope_tab(wb: ViennaWorkbench) -> None:
    page_header(
        "Denis review · scope",
        "What exactly should this project be?",
        "The public record does not contain a Denis-specific brief. This page records his six "
        "decisions; it does not infer them. Until the signed brief is returned, every answer "
        "here is an unconfirmed transcription.",
        "Unconfirmed · needs Denis",
        "amber",
    )

    st.markdown("**What the public record says**")
    source_1, source_2, source_3 = st.columns(3, gap="medium")
    with source_1, st.container(border=True):
        st.markdown("**Group profile**")
        st.markdown(
            "Denis Kovacevic is listed as research and teaching staff (BSc M.Sc.) on the "
            "[stable team page](https://ni.cs.univie.ac.at/about-us/team/). The page contains "
            "no research statement."
        )
    with source_2, st.container(border=True):
        st.markdown("**Publication record**")
        st.markdown(
            "No publication is listed on the checked group or "
            "[UCris profile](https://ucrisportal.univie.ac.at/en/persons/denis-kovacevic/). "
            "This is not evidence that no publication exists elsewhere."
        )
    with source_3, st.container(border=True):
        st.markdown("**Group scope**")
        st.markdown(
            "The group project runs 2024-07-01 to 2028-06-30: *An interdisciplinary approach "
            "to learn and test the causal mapping between neural network dynamics and behavior*. "
            "See [projects](https://ni.cs.univie.ac.at/research/projects)."
        )
    st.caption(
        "Public pages checked 2026-11-13. No Denis-specific written brief or research statement "
        "was found. Group-level alignment is not a Denis requirement."
    )
    audit_path = wb.root / "docs" / "DENIS_PUBLIC_SOURCE_AUDIT.md"
    if audit_path.exists():
        st.download_button(
            "Download public-source audit (Markdown)",
            data=audit_path.read_bytes(),
            file_name=audit_path.name,
            mime="text/markdown",
            key="scope_audit_download",
        )

    st.divider()
    st.markdown("**Record the six decisions**")
    st.caption(
        "Nothing is preselected. A blank answer is not a default; it means G0 is still open. "
        "Use the words Denis actually gave you, not the wording you would prefer."
    )

    primary_label = st.selectbox(
        "1. Primary object",
        list(PRIMARY_OBJECTS.values()),
        index=None,
        placeholder="Select the primary object",
        key="scope_primary_object",
    )
    endpoint_label = st.selectbox(
        "2. Primary endpoint",
        list(PRIMARY_ENDPOINTS.values()),
        index=None,
        placeholder="Select the primary endpoint",
        key="scope_primary_endpoint",
    )
    endpoint_id = _scope_label_to_id(PRIMARY_ENDPOINTS, endpoint_label)
    endpoint_detail = ""
    if endpoint_id == "other":
        endpoint_detail = st.text_input(
            "Endpoint detail (required for 'other')",
            key="scope_endpoint_detail",
            placeholder="Name the endpoint in Denis's words",
        )

    data_labels = st.multiselect(
        "3. Data actually available",
        list(AVAILABLE_DATA.values()),
        default=[],
        placeholder="Select every item that exists; choose 'none yet' if none",
        key="scope_available_data",
    )
    available_data = tuple(
        item_id
        for item_id, item_label in AVAILABLE_DATA.items()
        if item_label in data_labels
    )
    data_detail = ""
    if "none_yet" not in available_data and available_data:
        data_detail = st.text_input(
            "Data access conditions or caveats (optional)",
            key="scope_data_detail",
            placeholder="Access, identifiers, protocol, or contact path",
        )

    success_label = st.selectbox(
        "4. Definition of success",
        list(SUCCESS_CRITERIA.values()),
        index=None,
        placeholder="Select what would count as success",
        key="scope_success_criterion",
    )
    target_label = st.selectbox(
        "5. Biological target and intervention",
        list(TARGET_MODES.values()),
        index=None,
        placeholder="Select whether an intervention is in scope",
        key="scope_target_mode",
    )
    target_id = _scope_label_to_id(TARGET_MODES, target_label)
    target_detail = ""
    if target_id == "specified":
        target_detail = st.text_area(
            "Circuit, modality and positive control",
            key="scope_target_detail",
            placeholder="Name the circuit, perturbation modality and assay positive control",
            height=90,
        )

    stop_col, budget_col = st.columns(2, gap="medium")
    with stop_col:
        stop_rule = st.text_area(
            "6a. Stop rule",
            key="scope_stop_rule",
            placeholder="What result would make you stop or change direction?",
            height=90,
        )
    with budget_col:
        budget_deadline = st.text_area(
            "6b. Budget and deadline",
            key="scope_budget_deadline",
            placeholder="ECTS or time budget, and the hard deadline",
            height=90,
        )

    st.markdown("**Provenance of this transcription**")
    provenance_1, provenance_2, provenance_3 = st.columns(3, gap="medium")
    with provenance_1:
        recorded_by = st.text_input(
            "Recorded by",
            key="scope_recorded_by",
            placeholder="Name of the person transcribing",
        )
    with provenance_2:
        recorded_on = st.text_input(
            "Recorded on",
            key="scope_recorded_on",
            placeholder="YYYY-MM-DD",
        )
    with provenance_3:
        source_label = st.selectbox(
            "Answer source",
            list(ANSWER_SOURCES.values()),
            index=None,
            placeholder="Where did the answer come from?",
            key="scope_answer_source",
        )

    answers = ScopeAnswers(
        primary_object=_scope_label_to_id(PRIMARY_OBJECTS, primary_label),
        primary_endpoint=endpoint_id,
        endpoint_detail=endpoint_detail,
        available_data=available_data,
        data_detail=data_detail,
        success_criterion=_scope_label_to_id(SUCCESS_CRITERIA, success_label),
        target_mode=target_id,
        target_detail=target_detail,
        stop_rule=stop_rule,
        budget_deadline=budget_deadline,
        recorded_by=recorded_by,
        recorded_on=recorded_on,
        answer_source=_scope_label_to_id(ANSWER_SOURCES, source_label),
    )

    st.divider()
    st.markdown("**Scope status**")
    decision_errors = answers.decision_errors()
    metadata_errors = answers.metadata_errors()
    if decision_errors:
        st.warning("G0 open — scope unconfirmed. Do not start new data collection.")
        st.markdown("\n".join(f"- {error}" for error in decision_errors))
    else:
        st.success("Recorded, not signed. The signed brief remains the authoritative confirmation.")
        first, deferred = answers.impact()
        impact_1, impact_2 = st.columns(2, gap="medium")
        with impact_1, st.container(border=True):
            st.markdown("**First workstream**")
            st.write(first)
        with impact_2, st.container(border=True):
            st.markdown("**Deferred**")
            st.write(deferred)
        if metadata_errors:
            st.warning("Decisions recorded, but provenance is incomplete; exports are disabled.")
            st.markdown("\n".join(f"- {error}" for error in metadata_errors))
        else:
            payload = answers.to_payload()
            st.download_button(
                "Download scope transcription (JSON)",
                data=json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8"),
                file_name="denis_scope_response.json",
                mime="application/json",
                key="scope_json_download",
            )
            st.download_button(
                "Download scope transcription (Markdown)",
                data=answers.to_markdown().encode("utf-8"),
                file_name="denis_scope_response.md",
                mime="text/markdown",
                key="scope_md_download",
            )

    st.info(
        "A completed form is a transcription, not a signature. Do not say Denis confirmed this "
        "scope until the signed block in DENIS_REQUIREMENTS_BRIEF.md is returned."
    )
    brief_path = wb.root / "docs" / "DENIS_REQUIREMENTS_BRIEF.md"
    if brief_path.exists():
        st.download_button(
            "Download six-decision request (Markdown)",
            data=brief_path.read_bytes(),
            file_name=brief_path.name,
            mime="text/markdown",
            key="scope_brief_download",
        )
    request_col, matrix_col = st.columns(2, gap="medium")
    data_request = wb.root / "docs" / "DATA_REQUEST_PACKAGE.md"
    gap_matrix = wb.root / "docs" / "GAP_CLOSURE_MATRIX.md"
    with request_col:
        if data_request.exists():
            st.download_button(
                "Download data request package (Markdown)",
                data=data_request.read_bytes(),
                file_name=data_request.name,
                mime="text/markdown",
                key="scope_data_request_download",
            )
    with matrix_col:
        if gap_matrix.exists():
            st.download_button(
                "Download gap closure matrix (Markdown)",
                data=gap_matrix.read_bytes(),
                file_name=gap_matrix.name,
                mime="text/markdown",
                key="scope_gap_matrix_download",
            )


def main() -> None:
    inject_styles()
    wb = get_workbench()
    page = render_sidebar()

    if page == "Overview":
        overview_tab(wb)
    elif page == "Scope & decisions":
        scope_tab(wb)
    elif page == "Tier 1 · BunDLe-Net":
        try:
            tier1_tab(wb)
        except FileNotFoundError as error:
            render_missing_artifact(error)
    elif page == "Tier 2 · NC-MCM":
        try:
            tier2_tab(wb)
        except FileNotFoundError as error:
            render_missing_artifact(error)
    elif page == "Tier 3 · Premotor":
        try:
            tier3_tab(wb)
        except FileNotFoundError as error:
            render_missing_artifact(error)
    elif page == "Evidence report":
        report_tab(wb)
    elif page == "Evidence & lab":
        lab_tab(wb)
    elif page == "Provenance":
        provenance_tab(wb)


if __name__ == "__main__":
    main()
