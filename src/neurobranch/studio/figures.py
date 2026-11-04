"""Plotly figure builders with a shared scientific visual language.

These functions prepare figures; they do not compute inferential statistics.
Every score shown in a figure comes from the frozen, machine-readable evidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from neurobranch.studio.theme import palette

FONT = '"Segoe UI Variable", "Segoe UI", "Inter", Arial, sans-serif'
MODEL_COLORS = {
    "B0_persistence": "baseline",
    "B1_matched_own": "baseline",
    "B3_population": "prediction",
    "B5_common_k10": "control",
}
MODEL_DASH = {
    "B0_persistence": "dot",
    "B1_matched_own": "dash",
    "B3_population": "solid",
    "B5_common_k10": "solid",
}


def _p(light: bool = False) -> dict[str, str]:
    return palette(light)


def style(
    fig: go.Figure,
    *,
    height: int = 420,
    title: str | None = None,
    light: bool = False,
    legend: bool = True,
    margin: tuple[int, int, int, int] = (20, 20, 35, 20),
) -> go.Figure:
    p = _p(light)
    fig.update_layout(
        template="plotly_dark" if not light else "plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=p["ink"], size=12),
        title=dict(text=title, font=dict(size=14, color=p["ink"])) if title else None,
        margin=dict(l=margin[0], r=margin[1], t=margin[2], b=margin[3]),
        hoverlabel=dict(
            bgcolor=p["panel2"],
            bordercolor=p["line"],
            font=dict(color=p["ink"], family=FONT),
        ),
        showlegend=legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=p["muted"]),
        ),
    )
    fig.update_xaxes(gridcolor=p["line"], zerolinecolor=p["line"], color=p["muted"])
    fig.update_yaxes(gridcolor=p["line"], zerolinecolor=p["line"], color=p["muted"])
    return fig


def trajectory_3d(
    case: dict[str, Any],
    *,
    visible_models: tuple[str, ...] | list[str] = (
        "B3_population",
        "B5_common_k10",
        "B1_matched_own",
        "B0_persistence",
    ),
    show_truth: bool = True,
    show_ensemble: bool = False,
    show_context: bool = True,
    context_tail: int = 160,
    light: bool = False,
    frame_duration: int = 140,
    camera: dict[str, Any] | None = None,
    height: int = 650,
) -> go.Figure:
    p = _p(light)
    context = np.asarray(case["context"])[-int(context_tail) :]
    origin = np.asarray(case["origin"])
    truth = np.asarray(case["truth"])
    means = case["means"]
    fig = go.Figure()
    if show_context and len(context) > 1:
        fig.add_trace(
            go.Scatter3d(
                x=context[:, 0],
                y=context[:, 1],
                z=context[:, 2],
                mode="lines",
                name="Observed history (measured)",
                line=dict(color=p["observed"], width=4),
                opacity=0.92,
                hovertemplate=(
                    "Observed history<br>PC1 %{x:.3f}<br>PC2 %{y:.3f}<br>"
                    "PC3 %{z:.3f}<extra></extra>"
                ),
            )
        )
    fig.add_trace(
        go.Scatter3d(
            x=[origin[0]],
            y=[origin[1]],
            z=[origin[2]],
            mode="markers",
            name="Forecast origin",
            marker=dict(size=6, color=p["observed"], symbol="diamond"),
            hovertemplate=(
                f"Forecast origin · t={case['landmark']} · {case['origin_time']:.1f}s"
                "<extra></extra>"
            ),
        )
    )
    if show_truth:
        xyz = np.vstack([origin[None, :], truth])
        fig.add_trace(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="lines",
                name="Observed continuation (evaluation only)",
                line=dict(color=p["observed"], width=3, dash="dot"),
                opacity=0.95,
                hovertemplate=(
                    "Observed continuation<br>PC1 %{x:.3f}<br>PC2 %{y:.3f}<br>"
                    "PC3 %{z:.3f}<extra></extra>"
                ),
            )
        )
    for model in visible_models:
        if model not in means:
            continue
        color = p[MODEL_COLORS.get(model, "baseline")]
        dash = MODEL_DASH.get(model, "solid")
        label = case.get("model_labels", {}).get(model, model)
        if show_ensemble and model in case.get("ensembles", {}):
            ens = np.asarray(case["ensembles"][model])
            for member in ens[:8]:
                xyz = np.vstack([origin[None, :], member])
                fig.add_trace(
                    go.Scatter3d(
                        x=xyz[:, 0],
                        y=xyz[:, 1],
                        z=xyz[:, 2],
                        mode="lines",
                        name=f"{label} · sampled path",
                        line=dict(color=color, width=1.5),
                        opacity=0.17,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
        xyz = np.vstack([origin[None, :], np.asarray(means[model])])
        fig.add_trace(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="lines",
                name=label,
                line=dict(color=color, width=4, dash=dash),
                opacity=0.98,
                hovertemplate=(
                    f"{label}<br>PC1 %{{x:.3f}}<br>PC2 %{{y:.3f}}<br>"
                    "PC3 %{z:.3f}<extra></extra>"
                ),
            )
        )

    all_xyz = [origin, truth]
    if show_context:
        all_xyz.append(context)
    for model in visible_models:
        if model in means:
            all_xyz.append(np.asarray(means[model]))
    stacked = np.vstack([np.asarray(a) for a in all_xyz if len(a)])
    lo = stacked.min(axis=0)
    hi = stacked.max(axis=0)
    pad = np.maximum((hi - lo) * 0.10, 1e-3)
    evr = np.asarray(case["explained_variance_ratio"])
    common = dict(
        backgroundcolor="rgba(0,0,0,0)",
        gridcolor=p["line"],
        zerolinecolor=p["line"],
        color=p["muted"],
        tickfont=dict(size=10),
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=f"PC1 · {evr[0] * 100:.1f}% total",
                range=[lo[0] - pad[0], hi[0] + pad[0]],
                **common,
            ),
            yaxis=dict(
                title=f"PC2 · {evr[1] * 100:.1f}% total",
                range=[lo[1] - pad[1], hi[1] + pad[1]],
                **common,
            ),
            zaxis=dict(
                title=f"PC3 · {evr[2] * 100:.1f}% total",
                range=[lo[2] - pad[2], hi[2] + pad[2]],
                **common,
            ),
            aspectmode="data",
            camera=camera
            or dict(eye=dict(x=1.55, y=1.35, z=1.05), center=dict(x=0, y=0, z=0)),
            dragmode="orbit",
        ),
        uirevision="studio-3d",
    )
    style(fig, height=int(height), light=light, legend=True, margin=(0, 0, 40, 0))
    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=1.55, y=1.35, z=1.05),
            center=dict(x=0, y=0, z=0),
        )
    )
    # A client-side playhead makes the observed continuation the temporal anchor.
    play_x = [float(origin[0])] + [float(v) for v in truth[:, 0]]
    play_y = [float(origin[1])] + [float(v) for v in truth[:, 1]]
    play_z = [float(origin[2])] + [float(v) for v in truth[:, 2]]
    play_idx = len(fig.data)
    fig.add_trace(
        go.Scatter3d(
            x=[play_x[0]],
            y=[play_y[0]],
            z=[play_z[0]],
            mode="markers",
            name="Playback",
            marker=dict(size=8, color=p["prediction"], line=dict(color=p["ink"], width=1)),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    frames = [
        go.Frame(
            data=[
                go.Scatter3d(
                    x=[play_x[k]],
                    y=[play_y[k]],
                    z=[play_z[k]],
                    mode="markers",
                    marker=dict(size=8, color=p["prediction"], line=dict(color=p["ink"], width=1)),
                )
            ],
            traces=[play_idx],
            name=str(k),
        )
        for k in range(len(play_x))
    ]
    fig.frames = frames
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                y=1.12,
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=int(frame_duration), redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(frame=dict(duration=0, redraw=False), mode="immediate"),
                        ],
                    ),
                ],
                font=dict(color=p["ink"]),
                bgcolor=p["panel2"],
                bordercolor=p["line"],
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Observed step: ", font=dict(color=p["muted"], size=11)),
                steps=[
                    dict(
                        method="animate",
                        label=str(k),
                        args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                    )
                    for k in range(len(play_x))
                ],
                x=0.0,
                len=0.55,
                y=-0.05,
                bgcolor=p["panel2"],
                bordercolor=p["line"],
                font=dict(color=p["muted"], size=10),
            )
        ],
    )
    return fig


def trajectory_2d(
    case: dict[str, Any],
    *,
    visible_models: tuple[str, ...] | list[str] = ("B3_population", "B5_common_k10"),
    light: bool = False,
) -> go.Figure:
    """Graceful fallback when WebGL is unavailable; same coordinates, PC1/PC2."""
    p = _p(light)
    fig = go.Figure()
    context = np.asarray(case["context"])
    origin = np.asarray(case["origin"])
    fig.add_trace(
        go.Scatter(
            x=context[:, 0],
            y=context[:, 1],
            mode="lines",
            name="Observed history",
            line=dict(color=p["observed"], width=2),
        )
    )
    truth = np.asarray(case["truth"])
    fig.add_trace(
        go.Scatter(
            x=np.r_[origin[0], truth[:, 0]],
            y=np.r_[origin[1], truth[:, 1]],
            mode="lines",
            name="Observed continuation",
            line=dict(color=p["observed"], width=2, dash="dot"),
        )
    )
    for model in visible_models:
        if model not in case["means"]:
            continue
        arr = np.asarray(case["means"][model])
        fig.add_trace(
            go.Scatter(
                x=np.r_[origin[0], arr[:, 0]],
                y=np.r_[origin[1], arr[:, 1]],
                mode="lines",
                name=case.get("model_labels", {}).get(model, model),
                line=dict(
                    color=p[MODEL_COLORS.get(model, "baseline")],
                    width=2.5,
                    dash=MODEL_DASH.get(model, "solid"),
                ),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[origin[0]],
            y=[origin[1]],
            mode="markers",
            name="Forecast origin",
            marker=dict(color=p["observed"], size=10, symbol="diamond"),
        )
    )
    evr = np.asarray(case["explained_variance_ratio"])
    fig.update_xaxes(title=f"PC1 · {evr[0] * 100:.1f}% total")
    fig.update_yaxes(title=f"PC2 · {evr[1] * 100:.1f}% total")
    style(fig, height=470, light=light, title="2D fallback · PC1–PC2 neural-state projection")
    return fig


def forecast_time_series(
    case: dict[str, Any],
    *,
    visible_models: tuple[str, ...] | list[str] = ("B3_population", "B5_common_k10"),
    component: int = 0,
    light: bool = False,
) -> go.Figure:
    p = _p(light)
    fig = go.Figure()
    ct = np.asarray(case["context_time"])
    ctx = np.asarray(case["context"])
    ft = np.asarray(case["future_time"])
    truth = np.asarray(case["truth"])
    fig.add_trace(
        go.Scatter(
            x=ct,
            y=ctx[:, component],
            mode="lines",
            name="Observed history",
            line=dict(color=p["observed"], width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ft,
            y=truth[:, component],
            mode="lines",
            name="Observed continuation",
            line=dict(color=p["observed"], width=2, dash="dot"),
        )
    )
    for model in visible_models:
        if model not in case["means"]:
            continue
        arr = np.asarray(case["means"][model])
        fig.add_trace(
            go.Scatter(
                x=ft,
                y=arr[:, component],
                mode="lines",
                name=case.get("model_labels", {}).get(model, model),
                line=dict(
                    color=p[MODEL_COLORS.get(model, "baseline")],
                    width=2,
                    dash=MODEL_DASH.get(model, "solid"),
                ),
            )
        )
    fig.add_vline(
        x=case["origin_time"],
        line_color=p["muted"],
        line_dash="dash",
        annotation_text="origin",
        annotation_font_color=p["muted"],
    )
    fig.update_xaxes(title="time from recording start (s)")
    fig.update_yaxes(title=f"PC{component + 1} score (training PCA)")
    style(fig, height=360, light=light)
    return fig


def activity_heatmap(
    matrix: np.ndarray,
    time: np.ndarray,
    names: list[str],
    *,
    split_bounds: dict[str, list[int]] | None = None,
    light: bool = False,
) -> go.Figure:
    p = _p(light)
    z = np.asarray(matrix, dtype=float)
    finite = z[np.isfinite(z)]
    lim = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
    lim = max(lim, 1e-6)
    fig = go.Figure(
        go.Heatmap(
            z=z.T,
            x=np.asarray(time),
            y=np.arange(len(names)),
            colorscale=[
                [0.0, p["bg"]],
                [0.35, "#123b47"],
                [0.62, p["observed"]],
                [1.0, "#f4e3b2"],
            ],
            zmin=-lim,
            zmax=lim,
            colorbar=dict(title="z-score", thickness=10, len=0.7, outlinewidth=0),
            hovertemplate="neuron %{y}<br>t=%{x:.1f}s<br>z=%{z:.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(0, len(names), max(1, len(names) // 14))),
        ticktext=[names[i] for i in range(0, len(names), max(1, len(names) // 14))],
        title="recorded trace",
    )
    fig.update_xaxes(title="time (s)")
    style(fig, height=430, light=light, legend=False)
    if split_bounds:
        for label, color in (("train", p["faint"]), ("validation", p["control"])):
            if label in split_bounds:
                start = split_bounds[label][0]
                if 0 <= start < len(time):
                    fig.add_vline(x=float(time[start]), line_color=color, line_dash="dot")
    return fig


def activity_traces(
    matrix: np.ndarray,
    time: np.ndarray,
    names: list[str],
    *,
    light: bool = False,
) -> go.Figure:
    p = _p(light)
    fig = go.Figure()
    colors = [p["observed"], p["prediction"], p["control"], p["good"]]
    for j, name in enumerate(names[:4]):
        fig.add_trace(
            go.Scatter(
                x=time,
                y=matrix[:, j],
                mode="lines",
                name=name,
                line=dict(width=1.8, color=colors[j % len(colors)]),
            )
        )
    fig.update_xaxes(title="time (s)")
    fig.update_yaxes(title="standardized ΔF/F (z)")
    style(fig, height=360, light=light)
    return fig


def evidence_forest(snapshot, *, light: bool = False, height: int = 410) -> go.Figure:
    p = _p(light)
    rows = []
    for row in snapshot.session_rows:
        session = int(row["session"])
        blocks = snapshot.uncertainty["sessions"][session]
        rows.append(
            (
                f"Recording {session} · n={int(row['n_landmarks'])}",
                float(row["mean_energy_improvement"]),
                float(blocks["within_session_block5_low"]),
                float(blocks["within_session_block5_high"]),
                "session",
            )
        )
    primary = snapshot.primary
    rows.extend(
        [
            (
                "Conditional landmark block-5",
                primary["difference"],
                primary["conditional_ci"]["ci_low"],
                primary["conditional_ci"]["ci_high"],
                "conditional",
            ),
            (
                "Recording resampling block-5",
                primary["recording_ci"]["mean"],
                primary["recording_ci"]["ci_low"],
                primary["recording_ci"]["ci_high"],
                "recording",
            ),
            (
                "Five-recording t interval",
                primary["equal_session_mean"],
                primary["session_t_ci"][0],
                primary["session_t_ci"][1],
                "recording",
            ),
        ]
    )
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    lo = [r[1] - r[2] for r in rows][::-1]
    hi = [r[3] - r[1] for r in rows][::-1]
    color_map = {"session": p["observed"], "conditional": p["prediction"], "recording": p["control"]}
    colors = [color_map[r[4]] for r in rows][::-1]
    fig = go.Figure(
        go.Scatter(
            x=vals,
            y=labels,
            mode="markers",
            error_x=dict(type="data", symmetric=False, array=hi, arrayminus=lo, color=p["muted"], thickness=1.3),
            marker=dict(size=10, color=colors, line=dict(color=p["bg"], width=1)),
            hovertemplate="%{y}<br>Δ energy %{x:+.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color=p["danger"], line_dash="dash", opacity=0.8)
    fig.update_xaxes(title="B5 − B3 H20 PCA-10 energy  ·  positive = B3 lower")
    style(fig, height=int(height), light=light, legend=False, margin=(10, 30, 25, 10))
    return fig


def paired_histogram(snapshot, *, light: bool = False) -> go.Figure:
    p = _p(light)
    pivot = snapshot.per_landmark.pivot_table(
        index=["session", "landmark"], columns="model", values="pca10_energy_h20"
    )
    diff = (pivot["B5_common_k10"] - pivot["B3_population"]).dropna()
    fig = go.Figure(
        go.Histogram(
            x=diff,
            nbinsx=36,
            marker=dict(color=p["observed"], line=dict(color=p["bg"], width=1)),
            opacity=0.85,
            hovertemplate="Δ %{x:.2f}<br>%{y} landmarks<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color=p["danger"], line_dash="dash")
    fig.update_xaxes(title="paired Δ energy (B5 − B3)")
    fig.update_yaxes(title="landmarks")
    style(fig, height=330, light=light, legend=False)
    return fig


def horizon_chart(snapshot, *, light: bool = False) -> go.Figure:
    p = _p(light)
    rows = snapshot.uncertainty["horizon"]
    x = [int(r["horizon"]) for r in rows]
    y = [float(r["mean_advantage"]) for r in rows]
    lo = [float(r["ci_low"]) for r in rows]
    hi = [float(r["ci_high"]) for r in rows]
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(color=p["prediction"], width=2),
            marker=dict(size=9, color=[p["danger"] if v < 0 else p["prediction"] for v in y]),
            error_y=dict(
                type="data",
                symmetric=False,
                array=[high - value for high, value in zip(hi, y)],
                arrayminus=[value - low for value, low in zip(y, lo)],
                color=p["muted"],
            ),
            hovertemplate="H%{x}<br>Δ %{y:+.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color=p["danger"], line_dash="dash")
    fig.update_xaxes(title="forecast horizon (frames; ~2.9 Hz)", tickmode="array", tickvals=x)
    fig.update_yaxes(title="mean Δ energy (B5 − B3)")
    style(fig, height=350, light=light, legend=False)
    return fig


def coverage_chart(snapshot, *, light: bool = False) -> go.Figure:
    p = _p(light)
    cov = snapshot.statistical["paired_effect"]["coverage_at_nominal_95"]
    levels = ["50", "80", "95"]
    fig = go.Figure()
    for model, color, label in (
        ("B3_population", p["prediction"], "B3 population"),
        ("B5_common_k10", p["control"], "B5 ten-factor"),
    ):
        fig.add_trace(
            go.Bar(
                x=[f"{v}% nominal" for v in levels],
                y=[cov[model][v] for v in levels],
                name=label,
                marker_color=color,
                opacity=0.85,
                hovertemplate="%{x}<br>coverage %{y:.3f}<extra></extra>",
            )
        )
    fig.update_yaxes(title="marginal per-dimension coverage", range=[0, 1.05])
    fig.update_xaxes(title="nominal level")
    style(fig, height=330, light=light)
    return fig


def capacity_chart(snapshot, *, light: bool = False) -> go.Figure:
    p = _p(light)
    rows = snapshot.capacity["per_session"]
    sessions = [f"S{int(r['session'])}" for r in rows]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=sessions,
            y=[r["b3_parameters"] for r in rows],
            name="B3 population VAR",
            marker_color=p["prediction"],
            hovertemplate="%{x}<br>%{y:,} coefficients<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=sessions,
            y=[r["b5_k10_parameters"] for r in rows],
            name="B5 ten-factor",
            marker_color=p["control"],
            hovertemplate="%{x}<br>%{y:,} coefficients<extra></extra>",
        )
    )
    fig.update_yaxes(title="nominal parameters (n(n+1) vs 12n+110)", type="log")
    style(fig, height=330, light=light)
    return fig


def energy_bars(snapshot, *, light: bool = False) -> go.Figure:
    p = _p(light)
    mapping = {
        "B0_persistence": "B0\npersistence",
        "B1_matched_own": "B1m\nindependent",
        "B4_matched_own_500": "B4m\nindependent",
        "B5_common_k10": "B5\n10 factors",
        "B3_population": "B3\npopulation",
    }
    means = snapshot.per_landmark.groupby("model")["pca10_energy_h20"].mean()
    labels = [mapping[key] for key in mapping]
    values = [float(means[key]) for key in mapping]
    colors = [
        p["baseline"],
        p["baseline"],
        p["baseline"],
        p["control"],
        p["prediction"],
    ]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            textfont=dict(color=p["ink"], size=10),
            hovertemplate="%{x}<br>energy %{y:.3f}<extra></extra>",
        )
    )
    fig.update_yaxes(title="mean H20 PCA-10 energy", range=[0, 4.3])
    style(fig, height=330, light=light, legend=False)
    return fig


def perturbation_pair_3d(
    baseline: np.ndarray,
    perturbed: np.ndarray,
    difference: np.ndarray,
    origin: np.ndarray,
    *,
    light: bool = False,
) -> go.Figure:
    p = _p(light)
    fig = go.Figure()
    for arr, name, color, dash in (
        (baseline, "Baseline model trajectory", p["observed"], "solid"),
        (perturbed, "Mathematically perturbed trajectory", p["prediction"], "dash"),
        (difference, "Difference trajectory (perturbed − baseline)", p["control"], "dot"),
    ):
        xyz = np.vstack([origin[None, :], np.asarray(arr)])
        fig.add_trace(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="lines",
                name=name,
                line=dict(color=color, width=3.5, dash=dash),
            )
        )
    fig.update_layout(
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            bgcolor="rgba(0,0,0,0)",
            aspectmode="data",
        )
    )
    style(fig, height=520, light=light)
    return fig


def perturbation_response(
    time: np.ndarray,
    neuron_response: np.ndarray,
    population_response: np.ndarray,
    *,
    light: bool = False,
) -> go.Figure:
    p = _p(light)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time,
            y=neuron_response,
            mode="lines",
            name="Edited neuron difference",
            line=dict(color=p["prediction"], width=2.2),
            hovertemplate="t=%{x:.1f}s<br>Δ state %{y:+.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=population_response,
            mode="lines",
            name="Population L2 difference",
            line=dict(color=p["control"], width=2.2, dash="dash"),
            yaxis="y2",
            hovertemplate="t=%{x:.1f}s<br>L2 %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis=dict(title="edited-neuron difference (z)", gridcolor=p["line"]),
        yaxis2=dict(
            title="population L2 difference",
            overlaying="y",
            side="right",
            gridcolor="rgba(0,0,0,0)",
            color=p["control"],
        ),
        xaxis=dict(title="forecast time (s)", gridcolor=p["line"]),
    )
    style(fig, height=340, light=light)
    return fig


def dataframe_style(frame: pd.DataFrame) -> pd.io.formats.style.Styler:
    return frame.style.format(precision=3)
