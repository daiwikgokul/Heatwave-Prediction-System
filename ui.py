"""Look and feel: CSS, small HTML components, plotly styling."""
import pandas as pd
import streamlit as st

from .config import ALERT_LEVELS, INK, PAPER, SLATE, THERMAL

FONT = "Instrument Sans, system-ui, -apple-system, Segoe UI, sans-serif"
DISPLAY = "Bricolage Grotesque, Instrument Sans, system-ui, sans-serif"

_RIBBON = ", ".join(f"{c} {i * 100 / (len(THERMAL) - 1):.0f}%" for i, c in enumerate(THERMAL))

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Instrument+Sans:wght@400;500;600&display=swap');
html, body, .stApp, button, input, textarea, select {{ font-family: {FONT}; }}
/* never override the icon font, otherwise icon names print as text and overlap labels */
[data-testid="stIconMaterial"], .material-symbols-rounded, .material-icons, [class*="material-symbols"] {{
    font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important; }}
.stApp {{ background: {PAPER}; color: {INK}; }}
h1, h2, h3, h4 {{ font-family: {DISPLAY}; color: {INK}; letter-spacing: -0.01em; }}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer, [data-testid="stAppDeployButton"], [data-testid="stStatusWidget"] {{ visibility: hidden; }}
/* keep the sidebar open/close controls visible and above everything */
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"] {{
    visibility: visible !important; display: flex !important; z-index: 999999; }}
section[data-testid="stSidebar"] {{ min-width: 280px; }}
section[data-testid="stSidebar"] * {{ overflow-wrap: anywhere; }}
section[data-testid="stSidebar"] button p {{ white-space: normal; line-height: 1.2; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1320px; }}
section[data-testid="stSidebar"] {{ background: #E8EDF3; border-right: 1px solid #D5DDE7; }}
.ribbon {{ height: 6px; border-radius: 3px; background: linear-gradient(90deg, {_RIBBON}); margin: 0 0 14px 0; }}
.page-title {{ font-family: {DISPLAY}; font-size: 2.15rem; font-weight: 700; line-height: 1.1; margin: 0; color: {INK}; }}
.page-sub {{ color: {SLATE}; margin: 4px 0 16px 0; font-size: 1.0rem; max-width: 78ch; }}
.brand {{ font-family: {DISPLAY}; font-weight: 700; font-size: 1.25rem; line-height: 1.15; color: {INK}; }}
.brand small {{ display:block; font-family: {FONT}; font-weight: 400; font-size: .78rem; color: {SLATE}; margin-top: 2px; }}
.readout {{ display: flex; flex-wrap: wrap; background: #fff; border: 1px solid #D5DDE7; border-radius: 6px; margin: 4px 0 18px 0; }}
.readout .cell {{ flex: 1 1 150px; padding: 12px 16px; border-right: 1px solid #E5EAF1; min-width: 0; }}
.readout .cell:last-child {{ border-right: 0; }}
.readout .val {{ font-family: {DISPLAY}; font-size: 1.95rem; font-weight: 700; line-height: 1.05; }}
.readout .lab {{ font-size: .82rem; color: {SLATE}; margin-top: 3px; }}
.readout .cell.flag {{ border-left: 6px solid var(--c); }}
.chip {{ display: inline-flex; align-items: center; gap: 6px; padding: 1px 9px; border-radius: 4px;
        font-size: .78rem; font-weight: 600; border: 1px solid var(--c); background: var(--bg); color: {INK}; white-space: nowrap; }}
.chip i {{ width: 9px; height: 9px; border-radius: 50%; background: var(--c); display: inline-block; }}
.panel {{ background: #fff; border: 1px solid #D5DDE7; border-radius: 6px; padding: 14px 18px; }}
.alert-row {{ display: flex; align-items: center; gap: 12px; padding: 9px 4px 9px 0; border-bottom: 1px solid #EDF1F6; }}
.alert-row:last-child {{ border-bottom: 0; }}
.alert-row .bar {{ width: 6px; align-self: stretch; border-radius: 3px; background: var(--c); }}
.alert-row .nm {{ font-weight: 600; }}
.alert-row .sm {{ color: {SLATE}; font-size: .8rem; }}
.alert-row .rt {{ margin-left: auto; text-align: right; font-size: .85rem; }}
.note {{ color: {SLATE}; font-size: .85rem; }}
.advisory {{ background: #fff; border: 1px solid #D5DDE7; border-left: 6px solid var(--c); border-radius: 6px; padding: 16px 20px; }}
.advisory h4 {{ margin: 0 0 6px 0; }}
div[data-testid="stTabs"] button[role="tab"] {{ font-weight: 600; }}
div[data-testid="stMetric"] {{ background: #fff; border: 1px solid #D5DDE7; border-radius: 6px; padding: 10px 14px; }}
.legend-dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; }}
@media (max-width: 760px) {{ .page-title {{ font-size: 1.6rem; }} .readout .val {{ font-size: 1.5rem; }} }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def default_asof() -> pd.Timestamp:
    from . import data
    try:
        return pd.Timestamp(data.metrics()["default_asof"])
    except Exception:
        return pd.Timestamp("2026-05-14")


def setup() -> pd.Timestamp:
    """Called by every page: CSS + session defaults. Returns the active issue date."""
    inject_css()
    if "asof" not in st.session_state:
        st.session_state["asof"] = default_asof().date()
    if "lookahead" not in st.session_state:
        st.session_state["lookahead"] = 5
    return pd.Timestamp(st.session_state["asof"])


def header(title: str, sub: str = ""):
    st.markdown(f'<div class="ribbon"></div><div class="page-title">{title}</div>'
                f'<div class="page-sub">{sub}</div>', unsafe_allow_html=True)


def chip(level: int, text: str | None = None) -> str:
    a = ALERT_LEVELS[int(level)]
    label = text or f"{a['colour']} · {a['name']}"
    return f'<span class="chip" style="--c:{a["hex"]};--bg:{a["hex"]}22"><i></i>{label}</span>'


def readout(items: list[dict]):
    """items: [{val, lab, color(optional)}] rendered as one instrument strip."""
    cells = []
    for it in items:
        cls = "cell flag" if it.get("color") else "cell"
        style = f' style="--c:{it["color"]}"' if it.get("color") else ""
        cells.append(f'<div class="{cls}"{style}><div class="val">{it["val"]}</div>'
                     f'<div class="lab">{it["lab"]}</div></div>')
    st.markdown(f'<div class="readout">{"".join(cells)}</div>', unsafe_allow_html=True)


def style_fig(fig, height: int = 380, legend: bool = True, margin=None):
    fig.update_layout(
        height=height, margin=margin or dict(l=8, r=8, t=34, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
        font=dict(family=FONT, color=INK, size=13),
        showlegend=legend, legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_family=FONT), title_font=dict(family=DISPLAY, size=16, color=INK),
    )
    fig.update_xaxes(showgrid=False, linecolor="#C5CFDC", ticks="outside", tickcolor="#C5CFDC")
    fig.update_yaxes(gridcolor="#E6EBF2", zeroline=False)
    return fig


def fmt_day(d) -> str:
    return pd.Timestamp(d).strftime("%a %d %b")
