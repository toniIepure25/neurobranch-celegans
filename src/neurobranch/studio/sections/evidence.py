"""Uncertainty, calibration, capacity, and the claims ledger."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from neurobranch.studio.components import (
    badge,
    caption,
    figure_caption,
    note,
    section_header,
    stat_block,
    warning,
)
from neurobranch.studio.figures import (
    capacity_chart,
    coverage_chart,
    evidence_forest,
    horizon_chart,
    paired_histogram,
)
from neurobranch.studio.formatting import fixed, interval, percent, signed
from neurobranch.studio.metadata import StudioSnapshot


def render(snapshot: StudioSnapshot, *, light: bool = False) -> None:
    section_header(
        "05 · UNCERTAINTY & EVIDENCE",
        "Every interval names the unit it resamples.",
        "The 175 landmarks are overlapping forecast windows from five recordings. "
        "Conditional intervals resample landmarks within recordings; the recording "
        "bootstrap resamples the five recordings. They answer different questions and "
        "must not be collapsed into one.",
    )
    primary = snapshot.primary
    warning(
        "The five recordings are consistent with five inferred animals from one "
        "laboratory/preparation. They are not an independent cohort. With five "
        "recording-level units, the recording bootstrap has very low power; a wide "
        "interval is not evidence of no effect, and a narrow conditional interval is "
        "not animal-level precision."
    )

    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        stat_block(signed(primary["difference"], 4), "weighted paired mean · 175 landmarks")
    with c2:
        stat_block(interval(primary["conditional_ci"]["ci_low"], primary["conditional_ci"]["ci_high"]), "conditional landmark block-5 interval")
    with c3:
        stat_block(interval(primary["recording_ci"]["ci_low"], primary["recording_ci"]["ci_high"]), "recording resampling block-5 interval")

    tabs = st.tabs(
        [
            "Primary contrast",
            "Horizon & proper scores",
            "Calibration",
            "Capacity & identifiability",
            "Claims ledger",
        ]
    )
    with tabs[0]:
        left, right = st.columns([1.55, 1], gap="large")
        with left:
            st.plotly_chart(evidence_forest(snapshot, light=light), use_container_width=True, key="evidence_forest")
            figure_caption(
                "Recording means and the three aggregate intervals. Positive means B3 "
                "has lower energy. The conditional and recording intervals are not "
                "interchangeable."
            )
        with right:
            st.markdown("#### Recording-level summary")
            rows = []
            for row in snapshot.session_rows:
                sid = int(row["session"])
                interval_row = snapshot.uncertainty["sessions"][sid]
                rows.append(
                    {
                        "session": sid,
                        "n": int(row["n_landmarks"]),
                        "mean Δ": round(float(row["mean_energy_improvement"]), 3),
                        "block-5 low": round(float(interval_row["within_session_block5_low"]), 3),
                        "block-5 high": round(float(interval_row["within_session_block5_high"]), 3),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.markdown(
                f"""
                <div class="nb-panel" style="font-size:0.85rem;line-height:1.6">
                  <div><b>Positive recordings</b> {primary["sessions_positive"]} / {primary["n_sessions"]}</div>
                  <div><b>Positive landmarks</b> {percent(primary["fraction_positive"], 1)}</div>
                  <div><b>Descriptive dz</b> {fixed(primary["dz"])}</div>
                  <div><b>Sign test, one-sided</b> p = {fixed(primary["sign_test_one_sided_p"], 4)}</div>
                  <div><b>Equal-session mean</b> {signed(primary["equal_session_mean"])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.plotly_chart(paired_histogram(snapshot, light=light), use_container_width=True, key="evidence_hist")
        figure_caption(
            "Paired Δ energy for all 175 landmarks. The distribution is wide; the mean "
            "is not a description of a typical landmark."
        )
        with st.expander("Bootstrap and effective-unit details", expanded=False):
            st.dataframe(snapshot.cluster_bootstrap, use_container_width=True, hide_index=True)
            st.markdown(
                f"Non-overlapping H100 proxies: **{snapshot.statistical['overall_nonoverlapping_proxy']}**; "
                f"landmark inflation factor: **{snapshot.statistical['landmark_inflation_factor']:.1f}×**. "
                "The 175 count is a compute grid, not an effective sample size."
            )
            loo = snapshot.uncertainty.get("leave_one_session_out_positive")
            st.markdown(
                "All leave-one-recording-out means remain positive: "
                f"**{loo}**. This is a sensitivity check, not new animal-level evidence."
            )

    with tabs[1]:
        left, right = st.columns([1.2, 1], gap="large")
        with left:
            st.plotly_chart(horizon_chart(snapshot, light=light), use_container_width=True, key="evidence_horizon")
            figure_caption(
                "Four horizons were examined. H1 reverses; H5–H20 are positive. These "
                "are descriptive, not four independent confirmatory tests."
            )
        with right:
            st.markdown("#### The null that matters")
            st.markdown(
                f"""
                <div class="nb-panel">
                  <div class="nb-stat"><div class="value">{signed(primary["mse_difference"])}</div>
                  <div class="label">mean MSE difference (B5 − B3)</div></div>
                  <div class="nb-caption">Block-5 interval {interval(*primary["mse_ci"])} · crosses zero</div>
                  <div class="nb-divider"></div>
                  <div style="font-size:0.85rem;line-height:1.6">
                    Energy rewards both calibration and sharpness. A lower energy score
                    can reflect a better-spread predictive distribution rather than a
                    better conditional mean. The MSE result does not establish a mean
                    advantage.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            note(
                "H1 is a real reversal, not a rounding artifact. Any statement about "
                "B3's advantage must be restricted to multi-step horizons and cannot be "
                "described as uniform."
            )
        st.markdown("#### Horizon table")
        h = pd.DataFrame(snapshot.uncertainty["horizon"])
        st.dataframe(
            h[["horizon", "n", "mean_advantage", "ci_low", "ci_high", "fraction_better"]],
            use_container_width=True,
            hide_index=True,
        )

    with tabs[2]:
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            st.plotly_chart(coverage_chart(snapshot, light=light), use_container_width=True, key="evidence_coverage")
            figure_caption(
                "Coverage is the average of marginal per-dimension interval coverage, "
                "not the probability that a whole 10-D vector is inside a joint region."
            )
        with right:
            cov = snapshot.statistical["paired_effect"]["coverage_at_nominal_95"]
            st.markdown("#### Calibration status: not established")
            st.markdown(
                f"""
                <div class="nb-panel" style="font-size:0.9rem;line-height:1.8">
                  <div><b>B3 95% marginal</b> {percent(cov["B3_population"]["95"], 1)}</div>
                  <div><b>B5 95% marginal</b> {percent(cov["B5_common_k10"]["95"], 1)}</div>
                  <div><b>B3 mean MSE</b> {fixed(snapshot.statistical["calibration_confound"]["b3_mean_mse"])}</div>
                  <div><b>B5 mean MSE</b> {fixed(snapshot.statistical["calibration_confound"]["b5_mean_mse"])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            warning(
                "B5 is severely undercovered at nominal 95% while its mean MSE is only "
                "about 3% worse. The energy gap is partly a spread/calibration contrast. "
                "Do not describe this as joint 10-D calibration, and do not describe B3 "
                "as calibrated."
            )
        st.markdown(
            "The intervals are ensemble quantiles evaluated per dimension. Correlations "
            "between dimensions are not represented by this coverage statistic. A "
            "joint calibration claim would require a different construction and test."
        )

    with tabs[3]:
        left, right = st.columns([1.1, 1], gap="large")
        with left:
            st.plotly_chart(capacity_chart(snapshot, light=light), use_container_width=True, key="evidence_capacity")
            figure_caption(
                "Nominal parameter counts. Ridge shrinkage reduces effective capacity but "
                "does not make the models capacity-matched."
            )
        with right:
            cap = snapshot.capacity
            stat_block(f"{cap['ratio_min']:.2f}–{cap['ratio_max']:.2f}×", "B3 / B5-10 nominal coefficients")
            st.markdown(
                f"""
                <div class="nb-panel" style="font-size:0.84rem;line-height:1.65">
                  <div><b>B5 training variance captured</b> {percent(cap["b5_training_variance_fraction_min"])}–{percent(cap["b5_training_variance_fraction_max"])}</div>
                  <div><b>B5 residual variance</b> up to {percent(cap["b5_residual_variance_fraction_max"])}</div>
                  <div><b>Ordered split from B4m</b> {percent(cap["ordered_shares"]["baseline_B4m"]["common"])} common / {percent(cap["ordered_shares"]["baseline_B4m"]["residual"])} residual</div>
                  <div><b>Same split from B1m</b> {percent(cap["ordered_shares"]["baseline_B1m"]["common"])} / {percent(cap["ordered_shares"]["baseline_B1m"]["residual"])}</div>
                  <div><b>Same split from B0</b> {percent(cap["ordered_shares"]["baseline_B0"]["common"])} / {percent(cap["ordered_shares"]["baseline_B0"]["residual"])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            warning(
                "The 51/49 split depends on the ordering and baseline. It is arithmetic, "
                "not a structural partition. B3's extra parameters and arbitrary residual "
                "coordinates mean the residual advantage is not identifiable as "
                "cross-neuron information. No VAR coefficient is a synaptic weight or a "
                "directed connection."
            )
        st.markdown("#### What would make the mechanism claim possible?")
        st.markdown(
            "A capacity-matched population model with an explicit residual structure; "
            "an intervention or perturbation with a known ground truth; independent "
            "animals; and a pre-registered test that does not reuse the same tail. None "
            "of those is supplied by re-labelling this association."
        )

    with tabs[4]:
        st.markdown(
            "The claim ledger is the authoritative wording. The Studio does not add a "
            "`CONFIRMED` tier. Filter by status to see what the evidence supports."
        )
        statuses = sorted({c["status"] for c in snapshot.claims})
        selected = st.multiselect("Status", statuses, default=statuses)
        rows = [
            {
                "id": c["id"],
                "status": c["status"],
                "claim": c["claim"],
                "why": c["why"],
            }
            for c in snapshot.claims
            if c["status"] in selected
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=420)
        counts = snapshot.claim_counts
        st.markdown(" ".join(badge(f"{k} {v}", "danger" if k in {"INVALIDATED", "UNSUPPORTED"} else "control") for k, v in counts.items()), unsafe_allow_html=True)
        st.caption(
            "Tier vocabulary is defined in reports/master_review/claims_ledger.json. "
            "LIMITED is bounded, not uncertain; INVALIDATED means the stated claim is "
            "not supported by the audit, not that every related number is wrong."
        )
    caption(
        "Sources: reports/master_review/statistical_audit.json, cluster_bootstrap.csv, "
        "session_level_estimates.csv, claims_ledger.json; "
        "reports/simulator/kato_final_audit/uncertainty_summary.json."
    )
