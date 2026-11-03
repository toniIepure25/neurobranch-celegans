"""Visual language for the Scientific Studio.

The palette is intentionally small: cyan for measured neural state, amber for
model output, violet for uncertainty/controls, neutral grey for baselines, and
red only for methodologically material warnings.
"""

from __future__ import annotations

DARK = {
    "bg": "#071117",
    "bg2": "#0a1720",
    "panel": "#0d1e27",
    "panel2": "#102630",
    "ink": "#e3eef0",
    "muted": "#8ca4ad",
    "faint": "#7f98a1",
    "line": "#1d3540",
    "observed": "#5eead4",
    "observed_soft": "rgba(94,234,212,0.14)",
    "prediction": "#f4b860",
    "prediction_soft": "rgba(244,184,96,0.12)",
    "control": "#b9a2ff",
    "control_soft": "rgba(185,162,255,0.12)",
    "baseline": "#9aa7ac",
    "danger": "#ff7a7a",
    "danger_soft": "rgba(255,122,122,0.10)",
    "good": "#7ee2a8",
}

LIGHT = {
    "bg": "#f5f7f6",
    "bg2": "#eef3f2",
    "panel": "#ffffff",
    "panel2": "#f7faf9",
    "ink": "#142226",
    "muted": "#5f7278",
    "faint": "#66777c",
    "line": "#d7e1df",
    "observed": "#087f72",
    "observed_soft": "rgba(8,127,114,0.10)",
    "prediction": "#a96312",
    "prediction_soft": "rgba(169,99,18,0.10)",
    "control": "#6447b8",
    "control_soft": "rgba(100,71,184,0.09)",
    "baseline": "#6d7b80",
    "danger": "#a52a2a",
    "danger_soft": "rgba(165,42,42,0.08)",
    "good": "#1c7c4a",
}


def palette(light: bool = False) -> dict[str, str]:
    return LIGHT if light else DARK


def css(light: bool = False) -> str:
    p = palette(light)
    return f"""
<style>
:root {{
  --nb-bg: {p["bg"]};
  --nb-panel: {p["panel"]};
  --nb-ink: {p["ink"]};
  --nb-muted: {p["muted"]};
  --nb-line: {p["line"]};
  --nb-observed: {p["observed"]};
  --nb-prediction: {p["prediction"]};
  --nb-control: {p["control"]};
  --nb-danger: {p["danger"]};
}}
html, body, [class*="css"] {{
  font-family: "Segoe UI Variable", "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif;
}}
.stApp {{
  background:
    radial-gradient(1100px 520px at 12% -12%, {p["observed_soft"]}, transparent 62%),
    radial-gradient(900px 480px at 96% 4%, {p["control_soft"]}, transparent 58%),
    linear-gradient(180deg, {p["bg"]} 0%, {p["bg2"]} 100%);
  color: {p["ink"]};
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stToolbar"] {{ right: 1rem; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ max-width: 1500px; padding: 1.3rem 2.2rem 3rem; }}
section[data-testid="stSidebar"] {{
  background: {p["panel"]};
  border-right: 1px solid {p["line"]};
}}
section[data-testid="stSidebar"] .block-container {{ padding: 1.1rem 1rem 2rem; }}
h1, h2, h3, h4 {{ color: {p["ink"]}; letter-spacing: -0.02em; }}
h1 {{ font-weight: 720; }}
p, li, label, .stMarkdown {{ color: {p["ink"]}; }}
small, .nb-muted {{ color: {p["muted"]}; }}
a {{ color: {p["observed"]}; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
hr {{ border-color: {p["line"]}; }}

.nb-kicker {{
  color: {p["observed"]};
  font-size: 0.72rem;
  font-weight: 750;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  margin-bottom: 0.45rem;
}}
.nb-title {{
  color: {p["ink"]};
  font-size: clamp(2.0rem, 3.4vw, 3.4rem);
  font-weight: 730;
  line-height: 1.02;
  letter-spacing: -0.045em;
  max-width: 1000px;
}}
.nb-subtitle {{
  color: {p["muted"]};
  font-size: 1.02rem;
  line-height: 1.55;
  max-width: 860px;
  margin-top: 0.75rem;
}}
.nb-hero {{
  padding: 2.1rem 2.3rem 1.7rem;
  border: 1px solid {p["line"]};
  border-radius: 20px;
  background: linear-gradient(145deg, {p["panel"]}, {p["panel2"]});
  box-shadow: 0 18px 70px rgba(0,0,0,0.18);
  margin-bottom: 1.3rem;
}}
.nb-chip {{
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.24rem 0.62rem;
  border-radius: 999px;
  border: 1px solid {p["line"]};
  background: {p["panel2"]};
  color: {p["muted"]};
  font-size: 0.71rem;
  font-weight: 650;
  letter-spacing: 0.035em;
  margin: 0 0.35rem 0.35rem 0;
  white-space: nowrap;
}}
.nb-chip.observed {{ color: {p["observed"]}; border-color: {p["observed"]}; }}
.nb-chip.prediction {{ color: {p["prediction"]}; border-color: {p["prediction"]}; }}
.nb-chip.control {{ color: {p["control"]}; border-color: {p["control"]}; }}
.nb-chip.danger {{ color: {p["danger"]}; border-color: {p["danger"]}; }}
.nb-section {{
  margin: 0.4rem 0 1.1rem;
  padding-bottom: 0.8rem;
  border-bottom: 1px solid {p["line"]};
}}
.nb-section h2 {{ margin: 0.1rem 0 0.25rem; font-size: 1.65rem; }}
.nb-lead {{ color: {p["muted"]}; font-size: 0.95rem; line-height: 1.55; max-width: 980px; }}
.nb-panel {{
  border: 1px solid {p["line"]};
  border-radius: 16px;
  background: {p["panel"]};
  padding: 1.05rem 1.15rem;
  margin: 0.55rem 0 0.9rem;
}}
.nb-panel h4 {{ margin: 0 0 0.45rem; font-size: 0.95rem; }}
.nb-stat {{
  border-left: 2px solid {p["observed"]};
  padding: 0.25rem 0 0.25rem 0.9rem;
  margin: 0.5rem 0;
}}
.nb-stat .value {{ font-size: 2.25rem; font-weight: 720; line-height: 1.0; color: {p["ink"]}; }}
.nb-stat .label {{ color: {p["muted"]}; font-size: 0.78rem; margin-top: 0.35rem; }}
.nb-note {{
  border-left: 2px solid {p["prediction"]};
  background: {p["prediction_soft"]};
  padding: 0.75rem 0.9rem;
  border-radius: 0 10px 10px 0;
  color: {p["ink"]};
  font-size: 0.86rem;
  line-height: 1.5;
}}
.nb-warning {{
  border: 1px solid {p["danger"]};
  background: {p["danger_soft"]};
  padding: 0.75rem 0.9rem;
  border-radius: 10px;
  color: {p["ink"]};
  font-size: 0.84rem;
  line-height: 1.5;
}}
.nb-mono {{
  font-family: "Cascadia Mono", "SFMono-Regular", Consolas, monospace;
  font-size: 0.82rem;
  color: {p["muted"]};
}}
.nb-caption {{
  color: {p["faint"]};
  font-size: 0.75rem;
  line-height: 1.45;
  margin-top: 0.25rem;
}}
.nb-divider {{ height: 1px; background: {p["line"]}; margin: 1.1rem 0; }}
.nb-side-brand {{ padding: 0.2rem 0 1rem; }}
.nb-side-brand .mark {{
  width: 34px; height: 34px; border-radius: 9px;
  background: conic-gradient(from 210deg, {p["observed"]}, {p["prediction"]}, {p["control"]}, {p["observed"]});
  display: inline-block; vertical-align: middle; margin-right: 0.6rem;
  box-shadow: 0 0 24px {p["observed_soft"]};
}}
.nb-side-brand .name {{ font-weight: 720; font-size: 1.03rem; color: {p["ink"]}; }}
.nb-side-brand .tag {{ color: {p["muted"]}; font-size: 0.7rem; letter-spacing: 0.08em; }}
.nb-slide {{
  min-height: 520px;
  border: 1px solid {p["line"]};
  border-radius: 18px;
  background: {p["panel"]};
  padding: 2rem 2.3rem;
  margin: 0.5rem 0 0.9rem;
}}
.nb-slide .num {{ color: {p["faint"]}; font-size: 0.72rem; letter-spacing: 0.14em; }}
.nb-slide h2 {{ font-size: 2rem; margin: 0.3rem 0 0.7rem; }}
.nb-slide .body {{ color: {p["muted"]}; font-size: 1rem; line-height: 1.6; max-width: 900px; }}

div[data-testid="stMetric"] {{
  border: 1px solid {p["line"]};
  border-radius: 14px;
  background: {p["panel"]};
  padding: 0.8rem 0.9rem;
}}
div[data-testid="stMetricLabel"] {{ color: {p["muted"]}; }}
div[data-testid="stMetricValue"] {{ color: {p["ink"]}; font-weight: 680; }}
.stButton > button, .stDownloadButton > button {{
  border-radius: 10px;
  border: 1px solid {p["line"]};
  background: {p["panel2"]};
  color: {p["ink"]};
  font-weight: 600;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  border-color: {p["observed"]};
  color: {p["observed"]};
}}
div[data-testid="stExpander"] {{
  border: 1px solid {p["line"]};
  border-radius: 12px;
  background: {p["panel"]};
}}
div[data-testid="stDataFrame"] {{ border: 1px solid {p["line"]}; border-radius: 10px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 0.25rem; border-bottom: 1px solid {p["line"]}; }}
.stTabs [data-baseweb="tab"] {{ color: {p["muted"]}; padding: 0.5rem 0.8rem; }}
.stTabs [aria-selected="true"] {{ color: {p["ink"]}; border-bottom: 2px solid {p["observed"]}; }}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{
  background: {p["panel"]}; border-color: {p["line"]}; color: {p["ink"]};
}}
.stSlider [data-baseweb="slider"] div {{ color: {p["observed"]}; }}
[data-testid="stSidebar"] .stRadio label {{ color: {p["muted"]}; }}
[data-testid="stSidebar"] .stRadio label:hover {{ color: {p["ink"]}; }}
code {{ color: {p["prediction"]}; background: {p["panel2"]}; }}
</style>
"""


def figure_colors(light: bool = False) -> dict[str, str]:
    return palette(light)
