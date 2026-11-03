"""Small, reusable presentation components for the Scientific Studio."""

from __future__ import annotations

import html
from collections.abc import Iterable

import streamlit as st

from neurobranch.studio.theme import css


def inject_css(light: bool = False) -> None:
    st.markdown(css(light), unsafe_allow_html=True)


def badge(text: str, tone: str = "neutral") -> str:
    tone = tone if tone in {"observed", "prediction", "control", "danger"} else "neutral"
    return f'<span class="nb-chip {tone}">{html.escape(text)}</span>'


def badges(items: Iterable[tuple[str, str]]) -> str:
    return "".join(badge(text, tone) for text, tone in items)


def hero(kicker: str, title: str, subtitle: str, chips: Iterable[tuple[str, str]]) -> None:
    st.markdown(
        f"""
        <div class="nb-hero">
          <div class="nb-kicker">{html.escape(kicker)}</div>
          <div class="nb-title">{title}</div>
          <div class="nb-subtitle">{subtitle}</div>
          <div style="margin-top:1.1rem">{badges(chips)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(number: str, title: str, lead: str) -> None:
    st.markdown(
        f"""
        <div class="nb-section">
          <div class="nb-kicker">{html.escape(number)}</div>
          <h2>{title}</h2>
          <div class="nb-lead">{lead}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel(title: str, body: str, *, tone: str = "") -> None:
    cls = f"nb-panel {tone}".strip()
    st.markdown(
        f'<div class="{cls}"><h4>{title}</h4><div>{body}</div></div>',
        unsafe_allow_html=True,
    )


def stat_block(value: str, label: str, *, color: str | None = None) -> None:
    style = f" style='border-color:{color}'" if color else ""
    st.markdown(
        f'<div class="nb-stat"{style}><div class="value">{value}</div>'
        f'<div class="label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    st.markdown(f'<div class="nb-note">{text}</div>', unsafe_allow_html=True)


def warning(text: str) -> None:
    st.markdown(f'<div class="nb-warning">{text}</div>', unsafe_allow_html=True)


def caption(text: str) -> None:
    st.markdown(f'<div class="nb-caption">{text}</div>', unsafe_allow_html=True)


def mono(text: str) -> None:
    st.markdown(f'<div class="nb-mono">{html.escape(text)}</div>', unsafe_allow_html=True)


def divider() -> None:
    st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.markdown(
        """
        <div class="nb-side-brand">
          <span class="mark"></span>
          <span class="name">NeuroBranch</span>
          <div class="tag">SCIENTIFIC STUDIO · V2</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def unavailable(message: str) -> None:
    st.error(message)
    st.stop()


def figure_caption(text: str) -> None:
    caption(text)


def evidence_badge(status: str) -> str:
    status = status.upper()
    if status in {"SUPPORTED_EXPLORATORY", "LIMITED", "PASS"}:
        tone = "prediction" if status != "PASS" else "observed"
    elif status in {"UNSUPPORTED", "INVALIDATED", "FAIL", "NOT_ESTABLISHED"}:
        tone = "danger"
    else:
        tone = "control"
    return badge(status.replace("_", " "), tone)
