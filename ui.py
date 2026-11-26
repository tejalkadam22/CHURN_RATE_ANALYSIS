"""Shared styling + small UI helpers."""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

PALETTE = {"Decision Tree": "#22d3ee", "Random Forest": "#8b5cf6",
           "Logistic Regression": "#f59e0b", "KNN": "#ec4899"}
RED, GREEN, AMBER = "#ef4444", "#10b981", "#f59e0b"

pio.templates["churn"] = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e6e9f5"), colorway=list(PALETTE.values()) + ["#10b981", "#ef4444"],
    xaxis=dict(gridcolor="rgba(255,255,255,.07)"), yaxis=dict(gridcolor="rgba(255,255,255,.07)"),
    margin=dict(l=10, r=10, t=50, b=10)))
pio.templates.default = "churn"

CSS = """
<style>
.block-container{padding-top:1.5rem;max-width:1300px}
.hero{background:linear-gradient(120deg,#4c1d95 0%,#6d28d9 40%,#0891b2 100%);padding:26px 30px;border-radius:20px;
 margin-bottom:22px;box-shadow:0 10px 40px rgba(109,40,217,.35)}
.hero h1{margin:0;font-size:2rem;color:#fff}.hero p{margin:6px 0 0;color:#e9d5ff;font-size:1rem}
.kpi{background:linear-gradient(145deg,#161f3d,#0f1730);border:1px solid rgba(139,92,246,.25);border-radius:16px;
 padding:18px 20px;transition:.25s;height:100%}
.kpi:hover{transform:translateY(-4px);border-color:#8b5cf6;box-shadow:0 8px 26px rgba(139,92,246,.3)}
.kpi .l{font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;color:#9aa4c7}
.kpi .v{font-size:1.9rem;font-weight:700;color:#fff;margin-top:4px}
.kpi .s{font-size:.8rem;color:#7c86ab;margin-top:2px}
.pill{display:inline-block;padding:5px 14px;border-radius:999px;font-weight:600;font-size:.85rem}
.pill.high{background:rgba(239,68,68,.15);color:#fca5a5;border:1px solid #ef4444}
.pill.med{background:rgba(245,158,11,.15);color:#fcd34d;border:1px solid #f59e0b}
.pill.low{background:rgba(16,185,129,.15);color:#6ee7b7;border:1px solid #10b981}
.card{background:#121a33;border:1px solid rgba(255,255,255,.07);border-radius:16px;padding:18px 22px;margin-bottom:12px}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f1530,#0b1020);border-right:1px solid rgba(255,255,255,.06)}
div[data-testid="stTabs"] button[aria-selected="true"]{color:#c4b5fd}
.stButton>button{background:linear-gradient(90deg,#7c3aed,#06b6d4);color:#fff;border:0;border-radius:12px;font-weight:600;padding:.6rem 1.4rem}
.stButton>button:hover{filter:brightness(1.12);transform:translateY(-1px)}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi(col, label: str, value: str, sub: str = ""):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def risk_of(p: float):
    return ("High", "high", RED) if p >= 0.6 else ("Medium", "med", AMBER) if p >= 0.35 else ("Low", "low", GREEN)


def gauge(prob: float, title="Churn probability"):
    _, _, color = risk_of(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number=dict(suffix="%", font=dict(size=44)),
        title=dict(text=title, font=dict(size=16)),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=color, thickness=.3), bgcolor="rgba(255,255,255,.05)",
                   steps=[dict(range=[0, 35], color="rgba(16,185,129,.18)"),
                          dict(range=[35, 60], color="rgba(245,158,11,.18)"),
                          dict(range=[60, 100], color="rgba(239,68,68,.18)")])))
    fig.update_layout(height=290, margin=dict(l=20, r=20, t=60, b=10))
    return fig
