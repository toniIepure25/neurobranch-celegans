"""Dashboard view for the context-conditioned dynamics research branch.

Every number shown here is read from a report produced by the scripts in
``scripts/context_*.py``.  The view never presents a model rollout as a measured
animal response.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from neurobranch.config import PROJECT_ROOT
from neurobranch.dynamics.context_models import StableLatentLDS
from neurobranch.preprocessing import split_starts
from neurobranch.simulator.adapters import NeuralSimulator
from neurobranch.simulator.types import PerturbationSpec

TARGETS = ("AVBL", "AVBR", "AVEL", "AVER", "RIVL", "RIVR")
REPORTS = PROJECT_ROOT / "reports" / "simulator" / "context_dynamics"
WARNING = (
    "MODEL-DEPENDENT VIRTUAL PERTURBATION - NOT CAUSALLY VALIDATED. "
    "This edits model state, not a neuron in an animal, and is not an optogenetic dose."
)


def _json(name: str, default=None):
    path = REPORTS / name
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _csv(name: str) -> pd.DataFrame:
    path = REPORTS / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _latent_model(data, selected: dict):
    key = (int(data.worm_id), int(selected["latent_dim"]), float(selected["spectral_cap"]))
    if "context_models" not in st.session_state:
        st.session_state["context_models"] = {}
    cache = st.session_state["context_models"]
    if key not in cache:
        with st.spinner("Fitting the selected latent LDS on the training block ..."):
            cache[key] = StableLatentLDS.fit(
                data.X,
                data.y_macro,
                split_starts(data, "train", history=1, horizon=1),
                train_mask=data.masks["train"],
                latent_dim=int(selected["latent_dim"]),
                ridge=float(selected["ridge"]),
                spectral_cap=float(selected["spectral_cap"]),
            )
    return cache[key]


def _project(pca, states):
    return pca.transform(np.asarray(states))


def _line(Z, name, color):
    return go.Scatter3d(x=Z[:, 0], y=Z[:, 1], z=Z[:, 2], mode="lines",
                        line=dict(color=color, width=4), name=name)


def render(data, pca, worm_id: int, seed: int) -> None:
    st.subheader("CONTEXT-AWARE NEURAL SIMULATOR")
    st.error(
        "Research view. Measured oxygen is a controlled sensory input, not targeted neuronal "
        "stimulation. Model perturbations are mathematical, not experimental interventions. "
        "No intervention dataset is compatible with Gate 8, so that gate is NOT_EVALUABLE."
    )
    selected = (_json("model_selection.json", {}) or {}).get("selected_latent")
    if not selected:
        st.warning("No context-dynamics model selection report found. Run scripts/context_latent_baseline.py.")
        return
    tabs = st.tabs(["Overview", "Latent model", "Oxygen context", "Branching", "Perturbations"])

    with tabs[0]:
        st.markdown(
            f"Selected latent model: **dim {selected['latent_dim']}**, ridge **{selected['ridge']}**, "
            f"spectral cap **{selected['spectral_cap']}**. Evidence tier: exploratory temporal holdout."
        )
        st.markdown(
            "Task labels: **A** reconstruction, **B** causal filtering, **C** autonomous rollout, "
            "**D** known-schedule-conditioned forecast. A high reconstruction score is not evidence "
            "for C or D."
        )
        for name, label in (
            ("free_running_latent.csv", "Autonomous latent free-running"),
            ("oxygen_model_aggregate.csv", "Oxygen-input ablation (known schedule)"),
            ("shared_dynamics_aggregate.csv", "Session-disjoint shared dynamics"),
        ):
            frame = _csv(name)
            if not frame.empty:
                st.caption(label)
                st.dataframe(frame, width="stretch", hide_index=True)
            else:
                st.info(f"{label}: report not generated yet.")
        st.caption("Negative results are preserved. No result here establishes a biological generative mechanism.")

    with tabs[1]:
        model = _latent_model(data, selected)
        sim = NeuralSimulator(model, data)
        starts = sim.test_starts(horizon=20)
        frame = int(starts[len(starts) // 2]) if len(starts) else int(np.flatnonzero(data.masks["test"])[0])
        history = model.history_len
        initial = sim.initial_history(frame, history=history)
        base = sim.rollout(None, 20, initial_history=initial, seed=int(seed))
        stoch = sim.rollout(None, 20, initial_history=initial, seed=int(seed), stochastic=True)
        observed = data.X[frame : frame + 21]
        fig = go.Figure()
        fig.add_trace(_line(_project(pca, observed), "observed", "#2E86AB"))
        fig.add_trace(_line(_project(pca, base.states), "deterministic rollout", "#111111"))
        fig.add_trace(_line(_project(pca, stoch.states), "process-covariance rollout", "#E4572E"))
        fig.update_layout(height=520, scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(_csv("free_running_latent.csv"), width="stretch", hide_index=True)
        st.caption(
            "The stochastic path samples the fitted latent process covariance. It is model uncertainty, "
            "not measured biological variability and not a bifurcation."
        )

    with tabs[2]:
        audit = _json("oxygen_input_audit.json", {}) or {}
        sessions = _csv("oxygen_session_audit.csv")
        transitions = _csv("oxygen_transition_times.csv")
        if sessions.empty:
            st.info("Oxygen audit not generated. Run scripts/context_oxygen_audit.py.")
        else:
            st.markdown(
                "Measured Kato WT_Stim oxygen is a **controlled environmental sensory input**. "
                "It is shown as measured schedule data, never as targeted neuronal stimulation."
            )
            st.dataframe(sessions, width="stretch", hide_index=True)
            sid = st.selectbox("Session", sessions["session_id"].tolist())
            row = transitions[transitions.get("session_id", pd.Series(dtype=str)) == sid] if "session_id" in transitions else pd.DataFrame()
            if not row.empty:
                st.caption("Actual transition frame times (not assumed exact 30 s alignment).")
                st.dataframe(row, width="stretch", hide_index=True)
            st.info(
                "Hypothetical oxygen schedules are visual only and are not used to simulate a future animal. "
                "The oxygen-conditioned model forecasts use the *measured* schedule for the scored window."
            )
            st.warning(
                "Held-out transitions are few (3 per session in the default split) and the seven sessions "
                "share one fixed schedule. Any oxygen effect is low-power and exploratory."
            )
            st.json(audit, expanded=False)

    with tabs[3]:
        frame = _csv("branching_aggregate.csv")
        if frame.empty:
            st.info("Branching report not generated. Run scripts/context_branching.py.")
        else:
            st.dataframe(frame, width="stretch", hide_index=True)
            st.warning(
                "Agreement or disagreement with observed local outcomes is a model diagnostic. It is not "
                "evidence that the living network contains a bifurcation."
            )

    with tabs[4]:
        st.error(WARNING)
        targets = [t for t in TARGETS if t in data.predictor_names]
        if not targets:
            st.warning("No requested named target is present in this recording; no perturbation is offered.")
            return
        target = st.selectbox("Named model target (present in this recording)", targets)
        j = data.predictor_names.index(target)
        starts = split_starts(data, "train", history=1, horizon=1)
        delta = data.X[starts + 1, j] - data.X[starts, j]
        cap = float(np.std(delta))
        amplitude = st.slider("Bounded amplitude (one-step SD units)", -1.0, 1.0, 0.5, 0.1)
        protocol = st.selectbox("Protocol", ["initial_state", "dynamical_input"])
        direction = 1 if amplitude >= 0 else -1
        spec = PerturbationSpec(
            target=target,
            protocol=protocol,  # type: ignore[arg-type]
            mode="additive",
            amplitude=float(direction * abs(amplitude) * cap),
            onset=0,
            duration=0 if protocol == "initial_state" else 5,
        )
        model = _latent_model(data, selected)
        sim = NeuralSimulator(model, data)
        test_starts = sim.test_starts(horizon=20)
        frame = int(test_starts[len(test_starts) // 2]) if len(test_starts) else int(np.flatnonzero(data.masks["test"])[0])
        initial = sim.initial_history(frame, history=model.history_len)
        base = sim.rollout(None, 20, initial_history=initial, seed=int(seed))
        pert = sim.rollout(None, 20, initial_history=initial, seed=int(seed), perturbation=spec)
        fig = go.Figure()
        fig.add_trace(_line(_project(pca, base.states), "unperturbed", "#111111"))
        fig.add_trace(_line(_project(pca, pert.states), f"{target} virtual", "#E4572E"))
        fig.update_layout(height=480, scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"))
        st.plotly_chart(fig, width="stretch")
        thresholds = sim.support.thresholds
        st.metric("Fraction inside q99 support", f"{float(np.mean(pert.support_distance <= thresholds['q99'])):.2f}")
        st.metric("Max support distance", f"{float(np.max(pert.support_distance)):.2f}")
        summary = _csv("perturbation_summary.csv")
        if not summary.empty:
            st.dataframe(summary, width="stretch", hide_index=True)
        st.caption(
            "Gate 7 is not evaluable without matched experimental interventions; Gate 8 is NOT_EVALUABLE. "
            "No dose equivalence and no causal circuit claim is made."
        )
