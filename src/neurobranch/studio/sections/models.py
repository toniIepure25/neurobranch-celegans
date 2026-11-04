"""Model workbench: equations, capacity, and matched forecasts."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from neurobranch.studio.components import (
    caption,
    figure_caption,
    note,
    section_header,
    stat_block,
    warning,
)
from neurobranch.studio.controls import (
    case_for,
    go_to,
    landmark_selector,
    model_selector,
    session_selector,
)
from neurobranch.studio.figures import (
    energy_bars,
    forecast_time_series,
    trajectory_2d,
    trajectory_3d,
)
from neurobranch.studio.formatting import signed
from neurobranch.studio.metadata import StudioSnapshot
from neurobranch.studio.trajectories import TrajectoryStore

MODEL_CARDS = {
    "B0_persistence": {
        "equation": r"x_{t+1} = x_t + \varepsilon_{t+1}",
        "body": "The reference that assumes no dynamics beyond the current state. It "
        "defines the scale of forecast difficulty, not a biological null.",
    },
    "B1_matched_own": {
        "equation": r"x_{t+1,j} = a_j x_{t,j} + b_j + \varepsilon_{t+1,j}",
        "body": "A per-neuron independent AR(1) with the same 200-frame window as the "
        "historical baseline. It retains each neuron's own history but no population "
        "coupling and no shared factors.",
    },
    "B3_population": {
        "equation": r"x_{t+1} = A x_t + b + \varepsilon_{t+1}",
        "body": "A full lag-one population VAR fitted by ridge regression. It is the "
        "widest model in the primary contrast and has n(n+1) nominal coefficients.",
    },
    "B5_common_k10": {
        "equation": r"f_{t+1}=A_f f_t+b_f+\eta_{t+1};\quad x_{t+1,j}=b_j+c_j x_{t,j}+f_{t+1}\ell_j+\varepsilon_{t+1,j}",
        "body": "The ten-factor control: a factor VAR plus each neuron's own lag. The "
        "factors are training-only PCA scores. It cannot represent arbitrary residual "
        "cross-neuron coordinates.",
    },
}


def render(snapshot: StudioSnapshot, store: TrajectoryStore, *, light: bool = False) -> None:
    section_header(
        "04 · MODELS & FORECASTING",
        "Four fitted forecasters, one matched comparison.",
        "The workbench exposes equations, parameter budgets, refit windows, and matched "
        "trajectories. It is a frozen reanalysis of the fitted model family, not an "
        "architecture search.",
    )
    c1, c2, c3 = st.columns([1.2, 1.5, 1.1], gap="small")
    with c1:
        session = session_selector(snapshot)
    with c2:
        landmark = landmark_selector(store, session)
    with c3:
        models = model_selector()
    case = case_for(store, session, landmark)

    tabs = st.tabs(["Model cards", "Matched trajectories", "Capacity & metrics", "Historical rows"])
    with tabs[0]:
        cols = st.columns(2, gap="medium")
        for idx, model in enumerate(("B3_population", "B5_common_k10", "B1_matched_own", "B0_persistence")):
            card = MODEL_CARDS[model]
            with cols[idx % 2]:
                st.markdown(
                    f"""
                    <div class="nb-panel">
                      <h4>{case["model_labels"][model]}</h4>
                      <div style="font-size:1.05rem;line-height:1.5;margin:0.4rem 0 0.55rem">
                        <code>{card["equation"]}</code>
                      </div>
                      <div style="font-size:0.86rem;line-height:1.55;opacity:0.85">{card["body"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("#### Configuration")
        cfg = pd.DataFrame(snapshot.model_config)
        cfg = cfg[cfg["model"].isin(["B0_persistence", "B1_matched_own", "B3_population", "B5_common_k10", "B4_matched_own_500"])]
        st.dataframe(
            cfg[
                [
                    "model",
                    "family",
                    "window",
                    "n_parameters_per_neuron_or_total",
                    "penalty",
                    "intercept",
                    "role",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        note(
            "B4m is the independent AR comparator with the window matched to B3/B5. "
            "B4m is not a second population model and is not a substitute for the "
            "reported contrast. The reported contrast remains B5 − B3; positive values "
            "favor B3."
        )

    with tabs[1]:
        c1, c2 = st.columns([1.25, 1], gap="large")
        with c1:
            st.plotly_chart(
                trajectory_2d(case, visible_models=models, light=light),
                use_container_width=True,
                key=f"models_2d_{session}_{landmark}",
            )
            figure_caption(
                "Matched PC1–PC2 trajectories from one origin. All curves begin at the "
                "same observed state and use the same frozen PCA projection."
            )
        with c2:
            st.plotly_chart(
                forecast_time_series(case, visible_models=models, component=0, light=light),
                use_container_width=True,
                key=f"models_ts_{session}_{landmark}",
            )
            figure_caption("PC1 over time; vertical line is the origin. No future observation enters the forecast.")
        st.plotly_chart(
            trajectory_3d(
                case,
                visible_models=models,
                show_truth=True,
                show_ensemble=False,
                show_context=True,
                context_tail=120,
                light=light,
            ),
            use_container_width=True,
            key=f"models_3d_{session}_{landmark}",
        )
        figure_caption(
            "The same comparison in 3D. PC1–PC3 explain "
            f"{case['explained_3d']*100:.1f}% of total training variance in this recording "
            f"({case['explained_3d_of_10d']*100:.1f}% of the retained PCA-10 variance). "
            f"The other {case['remaining_10d_of_10d']*100:.1f}% of PCA-10 is not shown."
        )

    with tabs[2]:
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.plotly_chart(energy_bars(snapshot), use_container_width=True, key="model_energy_bars")
            figure_caption(
                "Mean H20 PCA-10 energy over 175 landmarks. Lower is better. These are "
                "descriptive means; the inferential contrast and its intervals are in "
                "the Evidence section."
            )
        with c2:
            st.markdown("#### This landmark")
            scores = case["scores"]
            frame = pd.DataFrame(
                [
                    {
                        "model": case["model_labels"][m],
                        "H20 energy": scores[m],
                        "vs B3": (scores[m] - scores["B3_population"]) if scores[m] is not None else None,
                    }
                    for m in ("B0_persistence", "B1_matched_own", "B4_matched_own_500", "B5_common_k10", "B3_population")
                    if m in scores
                ]
            )
            st.dataframe(
                frame,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "H20 energy": st.column_config.NumberColumn(format="%.4f"),
                    "vs B3": st.column_config.NumberColumn(format="%+.4f"),
                },
            )
            st.markdown(
                f"<div class='nb-caption'>B5 − B3 at this landmark: "
                f"<b>{signed(scores['B5_common_k10'] - scores['B3_population'])}</b>. "
                "One landmark is not the aggregate result.</div>",
                unsafe_allow_html=True,
            )
            st.markdown("#### Primary result")
            primary = snapshot.primary
            stat_block(signed(primary["difference"]), "weighted paired mean, 175 landmarks")
            st.markdown(
                f"<div class='nb-caption'>Recording-resampling interval "
                f"[{primary['recording_ci']['ci_low']:+.3f}, {primary['recording_ci']['ci_high']:+.3f}]. "
                f"{primary['sessions_positive']} of {primary['n_sessions']} recording means "
                "are positive; the sign test is not significant.</div>",
                unsafe_allow_html=True,
            )
        st.markdown("---")
        capacity = snapshot.capacity
        ordered = capacity["ordered_shares"]["baseline_B4m"]
        common_pct = round(100 * ordered["common"])
        residual_pct = 100 - common_pct
        warning(
            "Capacity is not held constant in the primary contrast. B3 has "
            f"{capacity['ratio_min']:.2f}–{capacity['ratio_max']:.2f}× "
            "the nominal coefficients of B5-10 and can represent arbitrary residual "
            "cross-neuron coordinates that B5 cannot. The residual advantage is therefore "
            "not identifiable as cross-neuron information from this design. The ordered "
            f"{common_pct}/{residual_pct} decomposition is arithmetic, not a structural partition."
        )
        st.markdown("#### Refit protocol")
        st.markdown(
            "At each landmark the model is refitted on the previous 500 observed frames "
            "(B1m uses 200). This is an **online-refit** estimand: no future frame enters "
            "the design, but validation and earlier test frames do. The test tail was "
            "reused across earlier project phases; this is not a fresh confirmatory "
            "holdout."
        )
        if st.button("Inspect the leakage and protocol audit", use_container_width=False):
            go_to("notebook")

    with tabs[3]:
        st.markdown(
            "The historical J2/J5 and B1/B4r rows are preserved for reproduction. They "
            "are not additional evidence for the current B3/B5 contrast. J5 has the same "
            "population-VAR form as B3 and is included for reconstruction checks."
        )
        hist = snapshot.per_landmark[
            snapshot.per_landmark["model"].isin(
                ["B1_historical", "J2_population_diag", "J5_population_lead1_boot", "B4r_historical"]
            )
        ]
        hist_mean = (
            hist.groupby("model")["pca10_energy_h20"].mean().rename("mean H20 energy").reset_index()
        )
        st.dataframe(hist_mean, use_container_width=True, hide_index=True)
        st.caption(
            "Aggregate means are descriptive only. Historical artifacts are not rewritten "
            "and these models are not used to rank current hypotheses."
        )
        st.markdown(
            "A historical stochastic population row is **not** a validated digital twin. "
            "Where a historical artifact used a different innovation scheme, its original "
            "report remains the record; the current comparison uses the common innovation "
            "ensemble with the same rollout contract for every model."
        )

    st.markdown("---")
    caption(
        "Sources: reports/simulator/kato_common_state/model_config.csv; "
        "model_per_landmark.csv; reports/master_review/capacity_audit.json. "
        "Equations describe the fitted lag-one transition, not synaptic mechanisms."
    )
