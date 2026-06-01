"""
Scotiabank RAG Lending Intelligence Platform
Streamlit + ChromaDB + OpenRouter
"""

import json
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

load_dotenv("data/.env")
OPENROUTER_KEY = os.environ.get("OpenrouterApiKey", "").strip().strip("'\"")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Scotiabank — Lending Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────────────────

PRIMARY = "#EC111A"
DARK    = "#0f172a"
SURFACE = "#ffffff"
BG      = "#f1f5f9"
MUTED   = "#64748b"
BORDER  = "#e2e8f0"
SUCCESS = "#10b981"
WARNING = "#f59e0b"
DANGER  = "#ef4444"

RISK_COLORS = {"Low": SUCCESS, "Medium": WARNING, "High": DANGER}
RISK_BG     = {
    "Low":    "rgba(16,185,129,0.12)",
    "Medium": "rgba(245,158,11,0.12)",
    "High":   "rgba(239,68,68,0.12)",
}

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────

# Inject Google Fonts via link tag (more reliable than @import inside <style>)
st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

st.markdown(f"""
<style>

*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}

/* ── APP SHELL ──────────────────────────────────────────────────── */
.stApp {{ background: {BG}; }}
.block-container {{
  padding: 2rem 2.5rem 4rem !important;
  max-width: 1440px !important;
}}

/* ── SIDEBAR ────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
  background: #07090f;
  border-right: 1px solid rgba(255,255,255,0.04);
  box-shadow: 2px 0 24px rgba(0,0,0,0.5);
}}
section[data-testid="stSidebar"] .block-container {{
  padding: 1.5rem 1.1rem 1.5rem !important;
}}
section[data-testid="stSidebar"] * {{ color: #e2e8f0 !important; }}

/* Nav radio — styled as icon-text nav items */
section[data-testid="stSidebar"] .stRadio > label {{ display: none !important; }}
section[data-testid="stSidebar"] .stRadio > div {{ gap: 2px !important; }}
section[data-testid="stSidebar"] .stRadio label {{
  padding: 10px 12px !important;
  border-radius: 9px !important;
  font-size: 0.84rem !important;
  font-weight: 500 !important;
  color: #475569 !important;
  cursor: pointer !important;
  margin: 0 !important;
  letter-spacing: 0.01em !important;
  transition: background 0.12s, color 0.12s !important;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{
  background: rgba(255,255,255,0.07) !important;
  color: #cbd5e1 !important;
}}
section[data-testid="stSidebar"] .stRadio [aria-checked="true"] ~ div label,
section[data-testid="stSidebar"] .stRadio [data-checked="true"] label,
section[data-testid="stSidebar"] .stRadio input:checked + div label {{
  background: rgba(236,17,26,0.14) !important;
  color: {PRIMARY} !important;
}}

/* Sidebar inputs */
section[data-testid="stSidebar"] .stSelectbox > div > div {{
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 9px !important;
}}
section[data-testid="stSidebar"] .stTextInput > div > div {{
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 9px !important;
  transition: border-color 0.15s !important;
}}
section[data-testid="stSidebar"] .stTextInput > div > div:focus-within {{
  border-color: {PRIMARY}80 !important;
  background: rgba(255,255,255,0.07) !important;
}}
section[data-testid="stSidebar"] input {{
  background: transparent !important;
  border: none !important;
  color: #e2e8f0 !important;
  border-radius: 9px !important;
}}
section[data-testid="stSidebar"] hr {{
  border-color: rgba(255,255,255,0.06) !important;
  margin: 0.85rem 0 !important;
}}

/* ── HIDE CHROME ────────────────────────────────────────────────── */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
[data-testid="stDecoration"] {{ display: none !important; }}

/* ── METRIC CARDS ───────────────────────────────────────────────── */
div[data-testid="metric-container"] {{
  background: {SURFACE};
  border: 1px solid {BORDER};
  border-radius: 16px;
  padding: 1.2rem 1.4rem 1rem;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 4px 14px rgba(0,0,0,0.03);
  position: relative;
  overflow: hidden;
}}
div[data-testid="metric-container"]::before {{
  content: '';
  position: absolute;
  inset: 0 0 auto 0;
  height: 3px;
  background: linear-gradient(90deg, {PRIMARY} 0%, #ff6b72 100%);
  border-radius: 16px 16px 0 0;
}}
div[data-testid="metric-container"] label {{
  color: {MUTED} !important;
  font-size: 0.67rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.09em !important;
}}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
  font-size: 1.55rem !important;
  font-weight: 700 !important;
  color: {DARK} !important;
  line-height: 1.15 !important;
  margin-top: 4px !important;
}}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {{
  font-size: 0.74rem !important;
  font-weight: 500 !important;
  margin-top: 2px !important;
}}

/* ── TEXT COLORS ────────────────────────────────────────────────── */
.block-container p, .block-container li, .block-container td,
.block-container th, .block-container span, .block-container label,
.block-container h1, .block-container h2, .block-container h3,
.block-container h4, .block-container div {{ color: {DARK}; }}
[data-testid="stChatMessage"] * {{ color: {DARK} !important; }}

/* ── CHART CONTAINERS ───────────────────────────────────────────── */
div[data-testid="stPlotlyChart"] {{
  background: {SURFACE};
  border: 1px solid {BORDER};
  border-radius: 16px;
  padding: 1.25rem 1rem 0.5rem;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 4px 14px rgba(0,0,0,0.03);
}}

/* ── DATAFRAME ──────────────────────────────────────────────────── */
.stDataFrame {{
  border-radius: 14px !important;
  border: 1px solid {BORDER} !important;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
  overflow: hidden !important;
}}

/* ── TABS ───────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
  background: {SURFACE};
  border: 1px solid {BORDER};
  border-radius: 12px;
  padding: 4px;
  gap: 2px;
  display: inline-flex;
  margin-bottom: 1.25rem;
}}
.stTabs [data-baseweb="tab"] {{
  background: transparent;
  border: none;
  border-radius: 8px;
  padding: 8px 18px;
  font-size: 0.84rem;
  font-weight: 500;
  color: {MUTED};
}}
.stTabs [aria-selected="true"] {{
  background: {PRIMARY} !important;
  color: white !important;
  border-radius: 8px !important;
}}

/* ── BUTTONS ────────────────────────────────────────────────────── */
.stButton > button {{
  border-radius: 10px;
  font-weight: 600;
  font-size: 0.855rem;
  border: 1px solid {BORDER};
  background: {SURFACE};
  color: {DARK};
  padding: 0.55rem 1rem;
  transition: all 0.15s;
}}
.stButton > button:hover {{
  border-color: {PRIMARY};
  color: {PRIMARY};
  background: #fff5f5;
}}
.stButton > button[kind="primary"] {{
  background: {PRIMARY};
  border-color: {PRIMARY};
  color: white !important;
  box-shadow: 0 2px 8px rgba(236,17,26,0.25);
}}
.stButton > button[kind="primary"]:hover {{
  background: #c90e16;
  box-shadow: 0 4px 16px rgba(236,17,26,0.35);
  transform: translateY(-1px);
}}
.stButton > button[kind="secondary"] {{
  background: rgba(255,255,255,0.07) !important;
  border: 1px solid rgba(255,255,255,0.15) !important;
  color: #94a3b8 !important;
}}
.stButton > button[kind="secondary"]:hover {{
  background: rgba(255,255,255,0.12) !important;
  color: white !important;
}}

/* ── INPUTS ─────────────────────────────────────────────────────── */
.stSelectbox > div > div {{
  border-radius: 10px !important;
  border-color: {BORDER} !important;
  font-size: 0.875rem !important;
}}
.stNumberInput > div > div > input {{
  border-radius: 10px !important;
  border-color: {BORDER} !important;
  font-size: 0.875rem !important;
}}
.stMultiSelect > div > div {{
  border-radius: 10px !important;
  border-color: {BORDER} !important;
}}

/* ── MULTISELECT TAGS ───────────────────────────────────────────── */
.stMultiSelect span[data-baseweb="tag"] {{
  background: {PRIMARY}18 !important;
  border: 1px solid {PRIMARY}35 !important;
  border-radius: 6px !important;
}}
.stMultiSelect span[data-baseweb="tag"] span {{
  color: {PRIMARY} !important;
  font-weight: 600 !important;
  font-size: 0.75rem !important;
}}
.stChatInput > div {{
  border-radius: 14px !important;
  border-color: {BORDER} !important;
  box-shadow: 0 2px 10px rgba(0,0,0,0.06) !important;
  background: {SURFACE} !important;
}}

/* ── FORCE INTER ON EVERYTHING ──────────────────────────────── */
input, select, textarea, button, label, p, span, div, h1, h2, h3, h4, h5 {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}}

/* ── FORCE LIGHT INPUTS IN MAIN CONTENT ─────────────────────── */
.block-container input,
.block-container textarea,
.block-container [data-baseweb="input"] input,
.block-container [data-baseweb="select"] > div:first-child,
.block-container .stSelectbox [data-baseweb="select"] > div,
.block-container .stNumberInput [data-baseweb="input"] > div,
.block-container .stNumberInput input {{
  background: {SURFACE} !important;
  background-color: {SURFACE} !important;
  color: {DARK} !important;
  border-color: {BORDER} !important;
  border-radius: 10px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.9rem !important;
}}

/* Selectbox value text + dropdown */
.block-container [data-baseweb="select"] span,
.block-container [data-baseweb="select"] div {{
  color: {DARK} !important;
  background: {SURFACE} !important;
  font-family: 'Inter', sans-serif !important;
}}

/* stSelectbox outer wrapper */
.block-container .stSelectbox > div > div {{
  background: {SURFACE} !important;
  border: 1px solid {BORDER} !important;
  border-radius: 10px !important;
}}

/* NumberInput +/- buttons */
.block-container .stNumberInput button {{
  background: {SURFACE} !important;
  border-color: {BORDER} !important;
  color: {DARK} !important;
}}

/* Dropdown list popup */
[data-baseweb="popover"] li,
[data-baseweb="menu"] li,
[data-baseweb="list"] li {{
  font-family: 'Inter', sans-serif !important;
  color: {DARK} !important;
  background: {SURFACE} !important;
}}
[data-baseweb="popover"],
[data-baseweb="menu"] {{
  background: {SURFACE} !important;
  border: 1px solid {BORDER} !important;
  border-radius: 12px !important;
  box-shadow: 0 8px 30px rgba(0,0,0,0.12) !important;
}}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_customers():
    if not os.path.exists("data/customers.json"):
        st.error("Run `python generate_data.py` first.")
        st.stop()
    with open("data/customers.json") as f:
        return json.load(f)

@st.cache_resource
def get_chroma():
    client = chromadb.PersistentClient(path="data/chroma_db")
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return client.get_collection("bank_customers", embedding_function=ef)

customers  = load_customers()
collection = get_chroma()
cust_index = {c["id"]: c for c in customers}

df = pd.DataFrame([{
    "id": c["id"], "name": c["name"], "age": c["age"],
    "credit_score": c["credit_score"], "annual_income": c["annual_income"],
    "total_debt": c["total_debt"], "risk_tier": c["risk_tier"],
    "risk_score": c["risk_score"], "employment": c["employment_status"],
    "checking": c["checking_balance"], "savings": c["savings_balance"],
    "total_assets": c["total_assets"], "dti": c["debt_to_income_ratio"],
    "bankruptcy": c["bankruptcy_history"], "member_since": c["member_since"],
    "active_loans": sum(1 for l in c["loans"] if l["status"] == "Active"),
} for c in customers])

# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def page_header(title, subtitle=None, badge=None, badge_color=None):
    bc = badge_color or PRIMARY
    badge_html = (
        f"<span style='background:{bc}18;color:{bc};font-size:0.7rem;font-weight:700;"
        f"padding:3px 10px;border-radius:20px;border:1px solid {bc}30;"
        f"margin-left:10px;vertical-align:middle;letter-spacing:0.04em'>{badge}</span>"
        if badge else ""
    )
    sub_html = (
        f"<div style='font-size:0.85rem;color:{MUTED};margin-top:5px'>{subtitle}</div>"
        if subtitle else ""
    )
    st.markdown(
        f"""<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:18px;
                        padding:1.4rem 1.75rem 1.2rem;margin-bottom:1.5rem;
                        box-shadow:0 1px 2px rgba(0,0,0,0.04);
                        border-left:4px solid {PRIMARY}'>
          <div style='display:flex;align-items:center'>
            <span style='font-size:1.35rem;font-weight:800;color:{DARK};
                         letter-spacing:-0.5px'>{title}</span>{badge_html}
          </div>
          {sub_html}
        </div>""",
        unsafe_allow_html=True,
    )


def section_title(text):
    st.markdown(
        f"<div style='font-size:0.78rem;font-weight:700;color:{DARK};letter-spacing:0.02em;"
        f"margin-bottom:0.7rem;padding-bottom:0.45rem;"
        f"border-bottom:2px solid {BORDER}'>{text}</div>",
        unsafe_allow_html=True,
    )


def kv_row(label, value, warn=False):
    color = DANGER if warn else DARK
    return (
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"padding:7px 0;border-bottom:1px solid {BORDER};font-size:0.855rem'>"
        f"<span style='color:{MUTED}'>{label}</span>"
        f"<span style='font-weight:600;color:{color}'>{value}</span></div>"
    )


def risk_pill(tier):
    c  = RISK_COLORS[tier]
    bg = RISK_BG[tier]
    return (
        f"<span style='background:{bg};color:{c};font-size:0.65rem;font-weight:700;"
        f"padding:3px 10px;border-radius:20px;border:1px solid {c}30;"
        f"letter-spacing:0.05em;text-transform:uppercase'>{tier}</span>"
    )


def loan_card_html(loan):
    flag  = loan["missed_payments"] > 0
    status_html = (
        f"<span style='color:{DANGER};font-size:0.75rem;font-weight:600'>"
        f"⚠ {loan['missed_payments']} missed</span>"
        if flag else
        f"<span style='color:{SUCCESS};font-size:0.75rem;font-weight:500'>✓ Current</span>"
    )
    border = f"1px solid {DANGER}40" if flag else f"1px solid {BORDER}"
    return f"""
    <div style='background:{SURFACE};border:{border};border-radius:13px;
                padding:1rem 1.25rem;margin-bottom:0.6rem;
                box-shadow:0 1px 3px rgba(0,0,0,0.04)'>
      <div style='display:flex;justify-content:space-between;align-items:center'>
        <div style='display:flex;align-items:center;gap:10px'>
          <span style='font-weight:700;font-size:0.9rem;color:{DARK}'>{loan["type"]} Loan</span>
          {status_html}
        </div>
        <span style='font-size:0.73rem;color:{MUTED}'>{loan["start_date"]} → {loan["end_date"]}</span>
      </div>
      <div style='display:flex;gap:1.75rem;margin-top:9px;font-size:0.82rem'>
        <span style='color:{MUTED}'>Principal <span style='font-weight:700;color:{DARK}'>${loan["amount"]:,}</span></span>
        <span style='color:{MUTED}'>APR <span style='font-weight:700;color:{DARK}'>{loan["rate"]}%</span></span>
        <span style='color:{MUTED}'>Monthly <span style='font-weight:700;color:{DARK}'>${loan["monthly_payment"]:,.0f}</span></span>
      </div>
    </div>"""

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # Brand
    st.markdown(
        f"""<div style='padding:0.25rem 0.25rem 1.25rem'>
          <div style='display:flex;align-items:center;gap:10px'>
            <div style='width:36px;height:36px;flex-shrink:0;
                        background:linear-gradient(135deg,{PRIMARY} 0%,#ff6b72 100%);
                        border-radius:10px;display:flex;align-items:center;
                        justify-content:center;font-size:1.15rem;
                        box-shadow:0 3px 10px rgba(236,17,26,0.4)'>🏦</div>
            <div>
              <div style='font-size:0.95rem;font-weight:800;letter-spacing:-0.3px;
                          color:#f1f5f9'>Scotiabank</div>
              <div style='font-size:0.68rem;color:#334155;font-weight:400;margin-top:1px'>
                Lending Intelligence</div>
            </div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # AI status pill
    if OPENROUTER_KEY:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:7px;padding:6px 10px;"
            f"background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.18);"
            f"border-radius:8px;margin-bottom:1rem'>"
            f"<div style='width:6px;height:6px;border-radius:50%;background:{SUCCESS};flex-shrink:0'></div>"
            f"<span style='font-size:0.7rem;font-weight:600;color:{SUCCESS} !important;"
            f"letter-spacing:0.03em'>AI Connected</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:7px;padding:6px 10px;"
            f"background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.18);"
            f"border-radius:8px;margin-bottom:1rem'>"
            f"<div style='width:6px;height:6px;border-radius:50%;background:{DANGER};flex-shrink:0'></div>"
            f"<span style='font-size:0.7rem;font-weight:600;color:{DANGER} !important;"
            f"letter-spacing:0.03em'>No API Key</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<div style='font-size:0.62rem;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:0.1em;color:#1e293b;margin-bottom:6px;padding:0 2px'>"
        f"Menu</div>",
        unsafe_allow_html=True,
    )

    view = st.radio(
        "nav",
        options=[
            "📊   Portfolio",
            "👤   Client Profile",
            "📋   Loan Assessment",
            "📈   Forecasting",
            "🤖   AI Assistant",
        ],
        label_visibility="collapsed",
    )
    view_map = {
        "📊   Portfolio":       "Portfolio Overview",
        "👤   Client Profile":  "Client Profile",
        "📋   Loan Assessment": "Loan Assessment",
        "📈   Forecasting":     "Forecasting",
        "🤖   AI Assistant":    "AI Assistant",
    }
    view = view_map[view]

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    st.divider()

    # Client selector with search
    st.markdown(
        f"<div style='font-size:0.62rem;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:0.1em;color:#1e293b;margin-bottom:8px;padding:0 2px'>"
        f"Active Client</div>",
        unsafe_allow_html=True,
    )

    search_query = st.text_input(
        "search",
        placeholder="🔍  Search by name or ID…",
        label_visibility="collapsed",
    )

    all_opts = sorted([(c["id"], c["name"]) for c in customers], key=lambda x: x[1])
    if search_query.strip():
        q = search_query.strip().lower()
        all_opts = [(cid, nm) for cid, nm in all_opts
                    if q in nm.lower() or q in cid.lower()]

    if all_opts:
        options_map = {f"{nm}  ({cid})": cid for cid, nm in all_opts}
        sel_label   = st.selectbox("client", list(options_map.keys()), label_visibility="collapsed")
        sel_id      = options_map[sel_label]
    else:
        st.markdown(
            f"<div style='font-size:0.75rem;color:#475569;padding:6px 2px'>"
            f"No clients found for <b style='color:#94a3b8'>{search_query}</b></div>",
            unsafe_allow_html=True,
        )
        sel_id = sorted(cust_index.keys())[0]

    sel = cust_index[sel_id]

    # Client mini-card
    rc  = RISK_COLORS[sel["risk_tier"]]
    rbg = RISK_BG[sel["risk_tier"]]
    st.markdown(
        f"""<div style='margin-top:0.6rem;background:rgba(255,255,255,0.04);
                        border:1px solid rgba(255,255,255,0.07);border-radius:13px;padding:1rem'>
          <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px'>
            <div style='font-weight:700;font-size:0.88rem;color:#f1f5f9;line-height:1.3'>
              {sel["name"]}</div>
            <div style='background:{rbg};color:{rc};font-size:0.6rem;font-weight:700;
                        padding:2px 8px;border-radius:20px;border:1px solid {rc}30;
                        letter-spacing:0.06em;text-transform:uppercase;flex-shrink:0;margin-left:6px'>
              {sel["risk_tier"]}</div>
          </div>
          <div style='font-size:0.7rem;color:#334155;margin-bottom:10px'>
            {sel["id"]} · Age {sel["age"]} · {sel["job_title"]}</div>
          <div style='display:grid;grid-template-columns:1fr 1fr;gap:6px'>
            <div style='background:rgba(255,255,255,0.05);border-radius:8px;padding:7px 9px'>
              <div style='font-size:0.57rem;color:#334155;text-transform:uppercase;
                          letter-spacing:0.07em;font-weight:600'>Credit Score</div>
              <div style='font-size:1rem;font-weight:700;color:#f1f5f9;margin-top:2px'>
                {sel["credit_score"]}</div>
            </div>
            <div style='background:rgba(255,255,255,0.05);border-radius:8px;padding:7px 9px'>
              <div style='font-size:0.57rem;color:#334155;text-transform:uppercase;
                          letter-spacing:0.07em;font-weight:600'>DTI</div>
              <div style='font-size:1rem;font-weight:700;color:#f1f5f9;margin-top:2px'>
                {sel["debt_to_income_ratio"]:.0%}</div>
            </div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# AI CLIENT — OpenRouter (OpenAI-compatible)
# ─────────────────────────────────────────────────────────────────────────────

def get_client():
    if not OPENROUTER_KEY:
        return None
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)

OR_MODEL      = "anthropic/claude-opus-4-5"
OR_FAST_MODEL = "openai/gpt-4o-mini"

def ai_chat(system: str, user: str, model: str = OR_MODEL, max_tokens: int = 1500) -> str:
    client = get_client()
    if not client:
        return "⚠️ No API key found. Add OpenrouterApiKey to data/.env"
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return resp.choices[0].message.content or "⚠️ Empty response from model."
    except Exception as e:
        return f"⚠️ API Error: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# RAG HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def rag_query(question: str, n: int = 5):
    try:
        results = collection.query(query_texts=[question], n_results=n)
        docs    = results["documents"][0]
        ids     = results["ids"][0]
    except Exception as e:
        return f"⚠️ Vector DB error: {e}", []

    context  = "\n\n---\n\n".join(docs)
    system   = (
        "You are a senior lending officer at Scotiabank with expertise in credit risk "
        "and financial analysis. You have customer profiles retrieved from the database. "
        "Respond professionally, cite specific numbers, flag risk factors clearly. "
        "For loan eligibility be direct: APPROVED / CONDITIONALLY APPROVED / DECLINED."
    )
    user_msg = f"Customer profiles:\n\n{context}\n\nQuestion: {question}"

    with st.spinner("Analysing..."):
        text = ai_chat(system, user_msg, model=OR_FAST_MODEL, max_tokens=1500)

    matched = [cust_index[i] for i in ids if i in cust_index]
    return text, matched


def loan_assessment(customer, amount, loan_type, term):
    r = {"Personal": 9.5, "Auto": 6.9, "Mortgage": 5.5,
         "Business": 8.5, "Student": 6.0, "Line of Credit": 10.5}[loan_type] / 100 / 12
    pmt       = round(amount * r / (1 - (1 + r) ** -term), 2)
    dti_after = round((customer["total_debt"] + amount) / max(customer["annual_income"], 1), 4)
    pti       = round((pmt / (customer["annual_income"] / 12)) * 100, 1)

    system = f"""You are a senior underwriter at Scotiabank. Write a structured loan assessment report.
The client data is provided. Use EXACTLY these numbers — do not make up values.

---

## Decision: [write APPROVED, CONDITIONALLY APPROVED, or DECLINED]
**[repeat your decision]** — [one sentence reason]

---

## Key Metrics

| Metric | Value | Benchmark | Pass? |
|--------|-------|-----------|-------|
| Credit Score | {customer['credit_score']} | ≥ 680 | [✅ or ❌] |
| Current DTI | {customer['debt_to_income_ratio']:.1%} | < 40% | [✅ or ❌] |
| Post-Loan DTI | {dti_after:.1%} | < 45% | [✅ or ❌] |
| Payment / Monthly Income | {pti}% | < 15% | [✅ or ❌] |
| Employment | {customer['employment_status']} – {customer['years_at_employer']} yrs | Stable | [✅ or ❌] |
| Bankruptcy History | {'Yes' if customer['bankruptcy_history'] else 'No'} | None | [✅ or ❌] |
| Late Payments | {customer['num_late_payments']} | 0 | [✅ or ❌] |

---

## Key Risk Factors
- [3-4 bullet points citing specific numbers from the client data above]

---

## Recommended Terms
- **Amount:** ${amount:,}
- **Rate:** [suggest a rate based on risk]% APR
- **Term:** {term} months
- **Monthly Payment:** ${pmt:,.2f}
- **Conditions:** [any conditions, or "None" if clean approval]

---

## Officer Notes
[2-3 sentences with your concrete recommendation]"""

    msg = (
        f"Client: {customer['name']} ({customer['id']})\n"
        f"Credit Score: {customer['credit_score']}\n"
        f"Annual Income: ${customer['annual_income']:,}\n"
        f"Employment: {customer['employment_status']} — {customer['job_title']} ({customer['years_at_employer']} yrs)\n"
        f"Total Debt: ${customer['total_debt']:,}  DTI: {customer['debt_to_income_ratio']:.2%}\n"
        f"Late payments: {customer['num_late_payments']}  Bankruptcy: {'Yes' if customer['bankruptcy_history'] else 'No'}\n"
        f"Assets: ${customer['total_assets']:,}  Risk: {customer['risk_tier']} ({customer['risk_score']}/100)\n\n"
        f"Requested: {loan_type} ${amount:,}  {term}mo  est. ${pmt:,.2f}/mo  DTI after: {dti_after:.2%}"
    )

    with st.spinner("Generating assessment..."):
        return ai_chat(system, msg, max_tokens=1200)

# ─────────────────────────────────────────────────────────────────────────────
# CHART THEME
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = [PRIMARY, "#3b82f6", SUCCESS, WARNING, "#8b5cf6", "#06b6d4"]

def chart(fig, h=360):
    fig.update_layout(
        height=h,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="Inter, -apple-system, sans-serif", size=12, color="#374151"),
        margin=dict(l=10, r=10, t=36, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11)),
    )
    fig.update_xaxes(
        gridcolor="#f1f5f9", showline=False,
        tickfont=dict(size=10, color=MUTED), zeroline=False,
    )
    fig.update_yaxes(
        gridcolor="#f1f5f9", showline=False,
        tickfont=dict(size=10, color=MUTED), zeroline=False,
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# VIEW 1 — PORTFOLIO OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

if view == "Portfolio Overview":
    page_header(
        "Portfolio Overview",
        f"Scotiabank lending book · {len(df)} clients",
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Clients",     f"{len(df):,}")
    m2.metric("Avg Credit Score",  f"{int(df.credit_score.mean()):,}")
    m3.metric("Loan Exposure",     f"${df.total_debt.sum()/1e6:.1f}M")
    m4.metric("High-Risk Clients", f"{len(df[df.risk_tier=='High'])} ({len(df[df.risk_tier=='High'])/len(df):.0%})")
    m5.metric("Total Deposits",    f"${(df.checking+df.savings).sum()/1e6:.1f}M")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Row 1
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        section_title("Risk Tier Distribution")
        rc = df.risk_tier.value_counts().reset_index()
        rc.columns = ["Tier", "Count"]
        fig = px.pie(rc, names="Tier", values="Count", hole=0.55,
                     color="Tier", color_discrete_map=RISK_COLORS)
        fig.update_traces(
            textposition="inside", textinfo="percent+label",
            textfont_size=12,
            marker=dict(line=dict(color=SURFACE, width=2.5)),
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(chart(fig, 320), use_container_width=True)

    with c2:
        section_title("Credit Score Distribution")
        fig = px.histogram(df, x="credit_score", nbins=28,
                           color_discrete_sequence=[PRIMARY],
                           labels={"credit_score": "Credit Score", "count": "Clients"})
        fig.update_traces(marker_line_width=0, opacity=0.8)
        fig.add_vline(x=750, line_dash="dash", line_color=SUCCESS, line_width=1.5,
                      annotation_text="Excellent", annotation_font_size=10)
        fig.add_vline(x=670, line_dash="dash", line_color=WARNING, line_width=1.5,
                      annotation_text="Fair", annotation_font_size=10)
        st.plotly_chart(chart(fig, 320), use_container_width=True)

    # Row 2
    c3, c4 = st.columns(2, gap="medium")
    with c3:
        section_title("Income vs Total Debt")
        fig = px.scatter(df, x="annual_income", y="total_debt",
                         color="risk_tier", color_discrete_map=RISK_COLORS,
                         hover_data=["name", "credit_score"],
                         labels={"annual_income": "Annual Income ($)",
                                 "total_debt": "Total Debt ($)", "risk_tier": "Risk"},
                         opacity=0.75)
        fig.update_traces(marker=dict(size=7))
        st.plotly_chart(chart(fig, 320), use_container_width=True)

    with c4:
        section_title("Employment Status")
        ec = df.employment.value_counts().reset_index()
        ec.columns = ["Status", "Count"]
        fig = px.bar(ec, x="Count", y="Status", orientation="h",
                     color_discrete_sequence=[PRIMARY], text="Count")
        fig.update_traces(textposition="outside", marker_line_width=0, opacity=0.85)
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(chart(fig, 320), use_container_width=True)

    # ── CLIENT DIRECTORY ─────────────────────────────────────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    section_title("Client Directory")

    ALL_EMP = sorted(df.employment.unique().tolist())
    fil_col1, fil_col2 = st.columns([1, 2])
    with fil_col1:
        st.markdown(
            f"<div style='font-size:0.68rem;font-weight:700;color:{MUTED};"
            f"text-transform:uppercase;letter-spacing:0.07em;margin-bottom:4px'>Risk Tier</div>",
            unsafe_allow_html=True,
        )
        risk_filter = st.multiselect(
            "Risk", ["Low", "Medium", "High"],
            default=["Low", "Medium", "High"],
            label_visibility="collapsed",
        )
    with fil_col2:
        st.markdown(
            f"<div style='font-size:0.68rem;font-weight:700;color:{MUTED};"
            f"text-transform:uppercase;letter-spacing:0.07em;margin-bottom:4px'>Employment</div>",
            unsafe_allow_html=True,
        )
        emp_filter = st.multiselect(
            "Employment", ALL_EMP, default=ALL_EMP,
            label_visibility="collapsed",
        )

    # Apply filters
    filtered = df[df.risk_tier.isin(risk_filter) & df.employment.isin(emp_filter)]

    # Result count
    st.markdown(
        f"<div style='font-size:0.78rem;color:{MUTED};margin-bottom:0.9rem;font-weight:500'>"
        f"Showing <b style='color:{DARK}'>{len(filtered)}</b> of {len(df)} clients</div>",
        unsafe_allow_html=True,
    )

    # Client card grid
    def client_card(row):
        rc          = RISK_COLORS[row.risk_tier]
        rbg         = RISK_BG[row.risk_tier]
        score_pct   = max(0, min(100, (row.credit_score - 300) / 550 * 100))
        score_color = SUCCESS if row.credit_score >= 720 else (WARNING if row.credit_score >= 650 else DANGER)
        dti_color   = SUCCESS if row.dti < 0.35 else (WARNING if row.dti < 0.50 else DANGER)
        initials    = "".join(n[0] for n in str(row["name"]).split()[:2]).upper()
        bk          = f"<span style='font-size:0.58rem;color:{DANGER};font-weight:700'>BK</span>" if row.bankruptcy else ""
        # Build card as a list of joined strings — no triple-quotes, no leading spaces that trigger markdown code blocks
        parts = [
            f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:16px;padding:1.1rem 1.15rem 0.95rem;box-shadow:0 1px 2px rgba(0,0,0,0.04),0 4px 14px rgba(0,0,0,0.03)'>",
            f"<div style='display:flex;align-items:center;gap:9px;margin-bottom:11px'>",
            f"<div style='width:38px;height:38px;border-radius:11px;background:{rc}18;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:0.88rem;font-weight:800;color:{rc}'>{initials}</div>",
            f"<div style='min-width:0;flex-grow:1'>",
            f"<div style='font-weight:700;font-size:0.875rem;color:{DARK};white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{row['name']}</div>",
            f"<div style='font-size:0.66rem;color:{MUTED};margin-top:1px'>{row['id']} · Age {row['age']}</div>",
            f"</div>{bk}</div>",
            f"<div style='margin-bottom:11px'>",
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px'>",
            f"<span style='font-size:0.58rem;color:{MUTED};text-transform:uppercase;letter-spacing:0.07em;font-weight:600'>Credit Score</span>",
            f"<span style='font-size:0.78rem;font-weight:800;color:{score_color}'>{int(row.credit_score)}</span>",
            f"</div>",
            f"<div style='height:5px;background:{BG};border-radius:5px;overflow:hidden'>",
            f"<div style='height:100%;width:{score_pct:.1f}%;background:linear-gradient(90deg,{score_color}99,{score_color});border-radius:5px'></div>",
            f"</div></div>",
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-bottom:11px'>",
            f"<div style='background:{BG};border-radius:9px;padding:6px 8px'><div style='font-size:0.55rem;color:{MUTED};text-transform:uppercase;letter-spacing:0.06em;font-weight:600'>Income</div><div style='font-size:0.82rem;font-weight:700;color:{DARK};margin-top:1px'>${row.annual_income/1000:.0f}k</div></div>",
            f"<div style='background:{BG};border-radius:9px;padding:6px 8px'><div style='font-size:0.55rem;color:{MUTED};text-transform:uppercase;letter-spacing:0.06em;font-weight:600'>DTI</div><div style='font-size:0.82rem;font-weight:700;color:{dti_color};margin-top:1px'>{row.dti:.0%}</div></div>",
            f"<div style='background:{BG};border-radius:9px;padding:6px 8px'><div style='font-size:0.55rem;color:{MUTED};text-transform:uppercase;letter-spacing:0.06em;font-weight:600'>Assets</div><div style='font-size:0.82rem;font-weight:700;color:{DARK};margin-top:1px'>${row.total_assets/1000:.0f}k</div></div>",
            f"<div style='background:{BG};border-radius:9px;padding:6px 8px'><div style='font-size:0.55rem;color:{MUTED};text-transform:uppercase;letter-spacing:0.06em;font-weight:600'>Loans</div><div style='font-size:0.82rem;font-weight:700;color:{DARK};margin-top:1px'>{int(row.active_loans)} active</div></div>",
            f"</div>",
            f"<div style='display:flex;justify-content:space-between;align-items:center;padding-top:9px;border-top:1px solid {BORDER}'>",
            f"<span style='background:{rbg};color:{rc};font-size:0.6rem;font-weight:700;padding:2px 9px;border-radius:20px;border:1px solid {rc}28;letter-spacing:0.06em;text-transform:uppercase'>{row.risk_tier} Risk</span>",
            f"<span style='font-size:0.66rem;color:{MUTED}'>{row.employment}</span>",
            f"</div></div>",
        ]
        return "".join(parts)

    if filtered.empty:
        st.markdown(
            f"<div style='text-align:center;padding:3rem;color:{MUTED}'>No clients match filters.</div>",
            unsafe_allow_html=True,
        )
    else:
        cards_html = "".join(client_card(row) for _, row in filtered.iterrows())
        st.markdown(
            "<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px'>"
            + cards_html + "</div>",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# VIEW 2 — CLIENT PROFILE
# ─────────────────────────────────────────────────────────────────────────────

elif view == "Client Profile":
    c = sel
    page_header(
        c["name"],
        f"{c['id']} · {c['job_title']} · Member since {c['member_since']}",
        badge=f"{c['risk_tier']} Risk",
        badge_color=RISK_COLORS[c["risk_tier"]],
    )

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Credit Score",  c["credit_score"])
    m2.metric("Annual Income", f"${c['annual_income']:,}")
    m3.metric("Checking",      f"${c['checking_balance']:,}")
    m4.metric("Savings",       f"${c['savings_balance']:,}")
    m5.metric("Total Debt",    f"${c['total_debt']:,}")
    m6.metric("DTI",           f"{c['debt_to_income_ratio']:.1%}")

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    tabs = st.tabs(["Overview", "Transactions", "Loans", "Analytics"])

    # ── OVERVIEW ──────────────────────────────────────────────────────────────
    with tabs[0]:
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            section_title("Personal & Employment")
            rows_html = "".join([
                kv_row("Age / Gender",     f"{c['age']} / {c['gender']}"),
                kv_row("Email",            c["email"]),
                kv_row("Phone",            c["phone"]),
                kv_row("Address",          c["address"]),
                kv_row("Employment",       f"{c['employment_status']} — {c['job_title']}"),
                kv_row("Employer",         c["employer"]),
                kv_row("Tenure",           f"{c['years_at_employer']} years"),
                kv_row("Monthly Expenses", f"${c['monthly_expenses']:,}"),
            ])
            st.markdown(
                f"<div style='background:{SURFACE};border:1px solid {BORDER};"
                f"border-radius:14px;padding:1rem 1.25rem;'>{rows_html}</div>",
                unsafe_allow_html=True,
            )

        with col2:
            section_title("Account Balances")
            bal = pd.DataFrame({
                "Account": ["Checking", "Savings", "Investments"],
                "Balance": [c["checking_balance"], c["savings_balance"], c["investment_balance"]],
            })
            fig = px.bar(bal, x="Account", y="Balance", color="Account",
                         text_auto="$.2s",
                         color_discrete_sequence=["#3b82f6", SUCCESS, PRIMARY])
            fig.update_traces(textposition="outside", marker_line_width=0, opacity=0.85)
            fig.update_layout(showlegend=False, yaxis_tickformat="$,.0f")
            st.plotly_chart(chart(fig, 240), use_container_width=True)

            section_title("Credit Risk Flags")
            flags = [
                ("Late Payments",            str(c["num_late_payments"]),           c["num_late_payments"] > 0),
                ("Months Since Delinquency", str(c.get("months_since_last_delinquency") or "—"), False),
                ("Open Accounts",            str(c["num_open_accounts"]),            False),
                ("Bankruptcy",               "Yes ⚠️" if c["bankruptcy_history"] else "No ✅", c["bankruptcy_history"]),
            ]
            flags_html = "".join([kv_row(k, v, w) for k, v, w in flags])
            st.markdown(
                f"<div style='background:{SURFACE};border:1px solid {BORDER};"
                f"border-radius:14px;padding:1rem 1.25rem;'>{flags_html}</div>",
                unsafe_allow_html=True,
            )

    # ── TRANSACTIONS ──────────────────────────────────────────────────────────
    with tabs[1]:
        txns = c.get("transactions", [])
        if txns:
            txn_df = pd.DataFrame(txns)
            txn_df["date"] = pd.to_datetime(txn_df["date"])

            c1, c2 = st.columns(2, gap="medium")
            with c1:
                section_title("Monthly Cash Flow")
                mo = txn_df.groupby(txn_df["date"].dt.to_period("M"))["amount"].sum().reset_index()
                mo["date"] = mo["date"].astype(str)
                mo["type"] = mo["amount"].apply(lambda x: "Inflow" if x >= 0 else "Outflow")
                fig = px.bar(mo, x="date", y="amount", color="type",
                             color_discrete_map={"Inflow": SUCCESS, "Outflow": PRIMARY},
                             labels={"date": "Month", "amount": "Net ($)", "type": ""})
                fig.update_traces(marker_line_width=0, opacity=0.85)
                st.plotly_chart(chart(fig, 300), use_container_width=True)

            with c2:
                section_title("Spending by Category")
                exp = txn_df[txn_df["amount"] < 0].copy()
                exp["amount"] = exp["amount"].abs()
                cs = exp.groupby("category")["amount"].sum().reset_index()
                fig = px.pie(cs, names="category", values="amount", hole=0.48,
                             color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_traces(textposition="inside", textinfo="percent+label",
                                  marker=dict(line=dict(color=SURFACE, width=1.5)))
                st.plotly_chart(chart(fig, 300), use_container_width=True)

            section_title("Recent Transactions")
            disp = txn_df.head(50).copy()
            disp["Date"]   = disp["date"].dt.strftime("%Y-%m-%d")
            disp["Amount"] = disp["amount"].apply(
                lambda x: f"+${x:,.2f}" if x > 0 else f"-${abs(x):,.2f}"
            )
            st.dataframe(
                disp[["Date", "category", "description", "Amount"]]
                .rename(columns={"category": "Category", "description": "Description"}),
                use_container_width=True, height=320, hide_index=True,
            )

    # ── LOANS ─────────────────────────────────────────────────────────────────
    with tabs[2]:
        loans  = c.get("loans", [])
        active = [l for l in loans if l["status"] == "Active"]
        closed = [l for l in loans if l["status"] == "Closed"]

        if active:
            section_title(f"Active Loans ({len(active)})")
            for l in active:
                st.markdown(loan_card_html(l), unsafe_allow_html=True)

            section_title("Amortization Schedule")
            fig = go.Figure()
            for i, l in enumerate(active):
                r   = l["rate"] / 100 / 12
                s   = datetime.strptime(l["start_date"], "%Y-%m-%d")
                e   = datetime.strptime(l["end_date"],   "%Y-%m-%d")
                n   = int((e - s).days / 30)
                bal = l["amount"]
                dates, bals = [], []
                for m in range(min(n, 360)):
                    dates.append(s + timedelta(days=30 * m))
                    bals.append(bal)
                    bal = max(0, bal - (l["monthly_payment"] - bal * r))
                fig.add_trace(go.Scatter(
                    x=dates, y=bals, name=l["type"], mode="lines",
                    line=dict(color=PALETTE[i % len(PALETTE)], width=2.5),
                ))
            fig.update_layout(yaxis_tickformat="$,.0f", xaxis_title="Date", yaxis_title="Balance")
            st.plotly_chart(chart(fig, 320), use_container_width=True)

        if closed:
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            section_title(f"Closed Loans ({len(closed)})")
            for l in closed:
                st.markdown(
                    f"<div style='background:{SURFACE};border:1px solid {BORDER};"
                    f"border-radius:13px;padding:0.9rem 1.25rem;margin-bottom:0.5rem;"
                    f"opacity:0.55;font-size:0.855rem'>"
                    f"<span style='font-weight:700'>{l['type']} Loan</span>"
                    f" — <span style='color:{MUTED}'>Closed · ${l['amount']:,} · "
                    f"{l['start_date']} → {l['end_date']}</span></div>",
                    unsafe_allow_html=True,
                )
        if not loans:
            st.info("No loan history for this client.")

    # ── ANALYTICS ─────────────────────────────────────────────────────────────
    with tabs[3]:
        def clamp(v, lo, hi): return max(lo, min(hi, v))
        def norm(v, lo, hi, inv=False):
            n = (v - lo) / (hi - lo) * 100
            return clamp(100 - n if inv else n, 0, 100)

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            section_title("Financial Health Radar")
            cats = ["Credit Score", "Low DTI", "Income", "Assets", "Job Tenure", "Clean Record"]
            vals = [
                norm(c["credit_score"], 300, 850),
                norm(c["debt_to_income_ratio"], 0, 1.5, inv=True),
                norm(c["annual_income"], 15_000, 250_000),
                norm(c["total_assets"], 0, 500_000),
                norm(c["years_at_employer"], 0, 30),
                0 if c["bankruptcy_history"] else 100,
            ]
            fig = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself", name="",
                fillcolor=f"rgba(236,17,26,0.12)",
                line=dict(color=PRIMARY, width=2.5),
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100],
                                    tickfont=dict(size=9), gridcolor="#e2e8f0"),
                    angularaxis=dict(tickfont=dict(size=11)),
                    bgcolor=SURFACE,
                ),
                showlegend=False, height=360,
                paper_bgcolor=SURFACE, margin=dict(l=40, r=40, t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            section_title(f"Peer Comparison — {c['risk_tier']} Risk Tier")
            peers = df[df.risk_tier == c["risk_tier"]]
            compare_items = [
                ("Credit Score",  c["credit_score"],             int(peers.credit_score.mean()),  None),
                ("Annual Income", c["annual_income"],            int(peers.annual_income.mean()), "${:,}"),
                ("Total Debt",    c["total_debt"],               int(peers.total_debt.mean()),    "${:,}"),
                ("DTI Ratio",     c["debt_to_income_ratio"],     float(peers.dti.mean()),         "{:.1%}"),
            ]
            rows = []
            for label, cval, pval, fmt in compare_items:
                cv = fmt.format(cval) if fmt else str(cval)
                pv = fmt.format(pval) if fmt else str(pval)
                pct = (cval - pval) / max(abs(pval), 1) * 100
                better = (label in ("Credit Score", "Annual Income") and cval >= pval) or \
                         (label in ("Total Debt", "DTI Ratio") and cval <= pval)
                arrow = "↑" if cval >= pval else "↓"
                delta_color = SUCCESS if better else DANGER
                rows.append(
                    f"<div style='padding:10px 0;border-bottom:1px solid {BORDER}'>"
                    f"<div style='font-size:0.7rem;color:{MUTED};text-transform:uppercase;"
                    f"letter-spacing:0.05em;font-weight:600'>{label}</div>"
                    f"<div style='display:flex;align-items:baseline;gap:8px;margin-top:3px'>"
                    f"<span style='font-size:1.15rem;font-weight:700;color:{DARK}'>{cv}</span>"
                    f"<span style='font-size:0.78rem;color:{delta_color};font-weight:500'>"
                    f"{arrow} {abs(pct):.0f}% vs avg {pv}</span>"
                    f"</div></div>"
                )
            st.markdown(
                f"<div style='background:{SURFACE};border:1px solid {BORDER};"
                f"border-radius:14px;padding:0.25rem 1.25rem;'>"
                + "".join(rows) + "</div>",
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# VIEW 3 — LOAN ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────

elif view == "Loan Assessment":
    c = sel

    # ── Compact client context bar ────────────────────────────────────────────
    rc  = RISK_COLORS[c["risk_tier"]]
    rbg = RISK_BG[c["risk_tier"]]
    score_color = (SUCCESS if c["credit_score"] >= 720
                   else WARNING if c["credit_score"] >= 650 else DANGER)
    dti_color   = SUCCESS if c["debt_to_income_ratio"] < 0.35 else (
                  WARNING if c["debt_to_income_ratio"] < 0.50 else DANGER)

    st.markdown(
        f"""<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:18px;
                        padding:1.25rem 1.75rem;margin-bottom:1.25rem;
                        box-shadow:0 1px 3px rgba(0,0,0,0.04);border-left:4px solid {PRIMARY}'>
          <div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem'>
            <div>
              <div style='display:flex;align-items:center;gap:10px'>
                <span style='font-size:1.25rem;font-weight:800;color:{DARK};letter-spacing:-0.4px'>
                  Loan Assessment</span>
                <span style='background:{rbg};color:{rc};font-size:0.68rem;font-weight:700;
                             padding:3px 10px;border-radius:20px;border:1px solid {rc}30;
                             letter-spacing:0.05em;text-transform:uppercase'>{c["risk_tier"]} Risk</span>
              </div>
              <div style='font-size:0.82rem;color:{MUTED};margin-top:4px'>
                {c["name"]} · {c["id"]} · {c["job_title"]}</div>
            </div>
            <div style='display:flex;gap:2rem'>
              <div style='text-align:center'>
                <div style='font-size:0.62rem;font-weight:700;text-transform:uppercase;
                            letter-spacing:0.08em;color:{MUTED}'>Credit Score</div>
                <div style='font-size:1.4rem;font-weight:800;color:{score_color};margin-top:1px'>
                  {c["credit_score"]}</div>
              </div>
              <div style='text-align:center'>
                <div style='font-size:0.62rem;font-weight:700;text-transform:uppercase;
                            letter-spacing:0.08em;color:{MUTED}'>Annual Income</div>
                <div style='font-size:1.4rem;font-weight:800;color:{DARK};margin-top:1px'>
                  ${c["annual_income"]:,}</div>
              </div>
              <div style='text-align:center'>
                <div style='font-size:0.62rem;font-weight:700;text-transform:uppercase;
                            letter-spacing:0.08em;color:{MUTED}'>Current DTI</div>
                <div style='font-size:1.4rem;font-weight:800;color:{dti_color};margin-top:1px'>
                  {c["debt_to_income_ratio"]:.1%}</div>
              </div>
              <div style='text-align:center'>
                <div style='font-size:0.62rem;font-weight:700;text-transform:uppercase;
                            letter-spacing:0.08em;color:{MUTED}'>Risk Score</div>
                <div style='font-size:1.4rem;font-weight:800;color:{rc};margin-top:1px'>
                  {c["risk_score"]:.0f}<span style='font-size:0.75rem;font-weight:500;color:{MUTED}'>/100</span></div>
              </div>
            </div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1.65], gap="large")

    # ── LEFT: Form panel ──────────────────────────────────────────────────────
    with col1:
        # Loan type selector — styled as a pill toggle group
        LOAN_TYPES = ["Personal", "Auto", "Mortgage", "Business", "Student", "Line of Credit"]
        LOAN_META  = {
            "Personal":       {"icon": "💳", "rate": 9.5,  "max": 100_000,   "terms": [12,24,36,48,60]},
            "Auto":           {"icon": "🚗", "rate": 6.9,  "max": 150_000,   "terms": [24,36,48,60,72,84]},
            "Mortgage":       {"icon": "🏠", "rate": 5.5,  "max": 1_500_000, "terms": [120,180,240,300,360]},
            "Business":       {"icon": "💼", "rate": 8.5,  "max": 500_000,   "terms": [12,24,36,60,84]},
            "Student":        {"icon": "🎓", "rate": 6.0,  "max": 150_000,   "terms": [60,84,120]},
            "Line of Credit": {"icon": "🔄", "rate": 10.5, "max": 100_000,   "terms": [12,24,36]},
        }

        # Loan type pill buttons
        st.markdown(
            f"<div style='font-size:0.7rem;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:0.08em;color:{MUTED};margin-bottom:8px'>Loan Type</div>",
            unsafe_allow_html=True,
        )
        if "loan_type_sel" not in st.session_state:
            st.session_state.loan_type_sel = "Personal"

        btn_cols = st.columns(3)
        for idx, lt in enumerate(LOAN_TYPES):
            meta = LOAN_META[lt]
            is_active = st.session_state.loan_type_sel == lt
            bg     = PRIMARY if is_active else SURFACE
            color  = "white" if is_active else MUTED
            border = f"1px solid {PRIMARY}" if is_active else f"1px solid {BORDER}"
            if btn_cols[idx % 3].button(
                f"{meta['icon']} {lt}",
                key=f"lt_{lt}",
                use_container_width=True,
            ):
                st.session_state.loan_type_sel = lt
                st.rerun()
            # Override button style via targeted CSS injection
            st.markdown(
                f"<style>div[data-testid='stButton'] button[kind='secondary'] {{}}</style>",
                unsafe_allow_html=True,
            )

        loan_type = st.session_state.loan_type_sel
        meta      = LOAN_META[loan_type]

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # Amount + Term in a clean white card
        st.markdown(
            f"<div style='background:{SURFACE};border:1px solid {BORDER};"
            f"border-radius:16px;padding:1.4rem 1.5rem;"
            f"box-shadow:0 1px 3px rgba(0,0,0,0.04);margin-top:4px'>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div style='font-size:0.7rem;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:0.08em;color:{MUTED};margin-bottom:6px'>Loan Amount</div>",
            unsafe_allow_html=True,
        )
        amount = st.number_input(
            "Loan Amount",
            min_value=1_000, max_value=meta["max"],
            value=min(25_000, meta["max"]), step=1_000,
            label_visibility="collapsed",
        )

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:0.7rem;font-weight:700;text-transform:uppercase;"
            f"letter-spacing:0.08em;color:{MUTED};margin-bottom:6px'>Term</div>",
            unsafe_allow_html=True,
        )
        term_options = {f"{t} months ({t//12}yr {t%12}mo)".replace(" 0mo","") if t >= 12 else f"{t} months": t
                        for t in meta["terms"]}
        term_label = st.selectbox("Term", list(term_options.keys()), label_visibility="collapsed")
        term = term_options[term_label]

        # Live calculations
        r       = meta["rate"] / 100 / 12
        est_pmt = round(amount * r / (1 - (1 + r) ** -term), 2)
        new_dti = round((c["total_debt"] + amount) / max(c["annual_income"], 1), 4)
        dti_delta = new_dti - c["debt_to_income_ratio"]
        pti       = round((est_pmt / (c["annual_income"] / 12)) * 100, 1)

        pmt_ok  = pti < 15
        dti_ok  = new_dti < 0.45

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""<div style='background:{BG};border-radius:12px;padding:1rem 1.1rem'>
              <div style='font-size:0.62rem;font-weight:700;text-transform:uppercase;
                          letter-spacing:0.09em;color:{MUTED};margin-bottom:10px'>
                Live Calculation</div>
              <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px'>
                <div style='background:{SURFACE};border-radius:10px;padding:0.75rem 0.9rem;
                            border:1px solid {BORDER}'>
                  <div style='font-size:0.6rem;color:{MUTED};text-transform:uppercase;
                              letter-spacing:0.07em;font-weight:600'>Monthly Payment</div>
                  <div style='font-size:1.2rem;font-weight:800;color:{DARK};margin-top:2px'>
                    ${est_pmt:,.0f}</div>
                  <div style='font-size:0.7rem;margin-top:2px;color:{"#16a34a" if pmt_ok else DANGER};
                              font-weight:500'>
                    {"✓" if pmt_ok else "⚠"} {pti:.1f}% of income</div>
                </div>
                <div style='background:{SURFACE};border-radius:10px;padding:0.75rem 0.9rem;
                            border:1px solid {BORDER}'>
                  <div style='font-size:0.6rem;color:{MUTED};text-transform:uppercase;
                              letter-spacing:0.07em;font-weight:600'>DTI After Loan</div>
                  <div style='font-size:1.2rem;font-weight:800;color:{DARK};margin-top:2px'>
                    {new_dti:.1%}</div>
                  <div style='font-size:0.7rem;margin-top:2px;color:{"#16a34a" if dti_ok else DANGER};
                              font-weight:500'>
                    {"✓" if dti_ok else "⚠"} {dti_delta:+.1%} change</div>
                </div>
                <div style='background:{SURFACE};border-radius:10px;padding:0.75rem 0.9rem;
                            border:1px solid {BORDER}'>
                  <div style='font-size:0.6rem;color:{MUTED};text-transform:uppercase;
                              letter-spacing:0.07em;font-weight:600'>Est. Rate</div>
                  <div style='font-size:1.2rem;font-weight:800;color:{DARK};margin-top:2px'>
                    {meta["rate"]}%</div>
                  <div style='font-size:0.7rem;margin-top:2px;color:{MUTED};font-weight:500'>
                    APR (base)</div>
                </div>
                <div style='background:{SURFACE};border-radius:10px;padding:0.75rem 0.9rem;
                            border:1px solid {BORDER}'>
                  <div style='font-size:0.6rem;color:{MUTED};text-transform:uppercase;
                              letter-spacing:0.07em;font-weight:600'>Total Cost</div>
                  <div style='font-size:1.2rem;font-weight:800;color:{DARK};margin-top:2px'>
                    ${est_pmt * term:,.0f}</div>
                  <div style='font-size:0.7rem;margin-top:2px;color:{MUTED};font-weight:500'>
                    over {term} months</div>
                </div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        run = st.button("Run AI Assessment →", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── RIGHT: Results panel ──────────────────────────────────────────────────
    with col2:
        key = f"assess_{c['id']}"
        if run:
            result = loan_assessment(c, amount, loan_type, term)
            st.session_state[key] = result

        if key in st.session_state:
            text     = st.session_state[key]
            approved = "APPROVED" in text.upper() and "DECLINED" not in text.upper()
            declined = "DECLINED" in text.upper()
            decision = "APPROVED" if approved else "DECLINED" if declined else "CONDITIONAL"
            bar_color = SUCCESS if approved else DANGER if declined else WARNING
            dec_label = (
                f"✓  Approved" if approved else
                f"✗  Declined" if declined else
                f"~  Conditional"
            )
            st.markdown(
                f"""<div style='background:{bar_color}10;border:1px solid {bar_color}30;
                                border-radius:14px;padding:0.85rem 1.25rem;margin-bottom:1rem;
                                display:flex;align-items:center;gap:10px'>
                  <div style='width:8px;height:8px;border-radius:50%;
                              background:{bar_color};flex-shrink:0'></div>
                  <span style='font-weight:700;font-size:0.85rem;color:{bar_color}'>{dec_label}</span>
                </div>""",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:16px;"
                f"padding:1.5rem 1.75rem;box-shadow:0 1px 3px rgba(0,0,0,0.04)'>",
                unsafe_allow_html=True,
            )
            st.markdown(text)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"""<div style='background:{SURFACE};border:2px dashed {BORDER};border-radius:18px;
                                padding:4rem 2rem;text-align:center;height:100%;
                                display:flex;flex-direction:column;align-items:center;
                                justify-content:center'>
                  <div style='width:52px;height:52px;background:{BG};border-radius:14px;
                              display:flex;align-items:center;justify-content:center;
                              font-size:1.5rem;margin:0 auto 1rem'>📋</div>
                  <div style='font-size:0.95rem;font-weight:600;color:{DARK};margin-bottom:6px'>
                    No assessment yet</div>
                  <div style='font-size:0.82rem;color:{MUTED};max-width:220px;line-height:1.5'>
                    Configure loan parameters and click <b>Run AI Assessment</b></div>
                </div>""",
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# VIEW 4 — FORECASTING
# ─────────────────────────────────────────────────────────────────────────────

elif view == "Forecasting":
    page_header("Forecasting & Analytics", "Client projections and portfolio trends")
    tabs_f = st.tabs(["Credit Score", "Income Projection", "Risk Evolution", "Portfolio Health"])

    with tabs_f[0]:
        c = sel
        section_title(f"Credit Score Forecast — {c['name']} (24 months)")
        np.random.seed(int(c["id"][1:]))
        imp    = {"High": 0.35, "Medium": 0.18, "Low": 0.05}[c["risk_tier"]]
        months = list(range(25))
        noise  = np.random.normal(0, 3, 25)
        scores = [min(850, c["credit_score"] + imp * m * 5 + noise[m]) for m in months]
        upper  = [min(850, s + 18) for s in scores]
        lower  = [max(300, s - 18) for s in scores]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months + months[::-1], y=upper + lower[::-1],
            fill="toself", fillcolor="rgba(236,17,26,0.07)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=months, y=scores, name="Projected",
            mode="lines+markers", line=dict(color=PRIMARY, width=2.5),
            marker=dict(size=5),
        ))
        fig.add_hline(y=750, line_dash="dot", line_color="#3b82f6", line_width=1.5,
                      annotation_text="Excellent (750)", annotation_position="right")
        fig.add_hline(y=700, line_dash="dot", line_color=SUCCESS, line_width=1.5,
                      annotation_text="Good (700)", annotation_position="right")
        fig.update_layout(xaxis_title="Months from Today", yaxis_title="Credit Score",
                          yaxis=dict(range=[300, 870]))
        st.plotly_chart(chart(fig, 380), use_container_width=True)

        n1, n2, n3 = st.columns(3)
        n1.metric("Current",   int(scores[0]))
        n2.metric("12-Month",  int(scores[12]), delta=f"{int(scores[12]-scores[0]):+d}")
        n3.metric("24-Month",  int(scores[24]), delta=f"{int(scores[24]-scores[0]):+d}")

    with tabs_f[1]:
        c = sel
        section_title("Income & Savings Projection — 5 Years")
        growth = {"Full-Time": 0.04, "Self-Employed": 0.06, "Part-Time": 0.02,
                  "Unemployed": 0.0, "Retired": 0.01}.get(c["employment_status"], 0.03)
        np.random.seed(int(c["id"][1:]) + 1)
        yrs     = list(range(6))
        income  = [c["annual_income"] * (1 + growth) ** y
                   + np.random.normal(0, c["annual_income"] * 0.015) for y in yrs]
        expense = [c["monthly_expenses"] * 12 * (1.025 ** y) for y in yrs]
        net     = [i - e for i, e in zip(income, expense)]
        labels  = [f"Year {y}" for y in yrs]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Income",   x=labels, y=income,
                             marker_color=SUCCESS, opacity=0.85))
        fig.add_trace(go.Bar(name="Expenses", x=labels, y=expense,
                             marker_color=PRIMARY, opacity=0.85))
        fig.add_trace(go.Scatter(name="Net Savings", x=labels, y=net,
                                 mode="lines+markers",
                                 line=dict(color="#3b82f6", width=2.5),
                                 marker=dict(size=7)))
        fig.update_layout(barmode="group", yaxis_tickformat="$,.0f")
        st.plotly_chart(chart(fig, 360), use_container_width=True)

    with tabs_f[2]:
        c = sel
        section_title(f"Risk Score Evolution — {c['name']} (8 Quarters)")
        np.random.seed(int(c["id"][1:]) + 10)
        qtrs = [f"Q{(i%4)+1} '2{'5' if i < 4 else '6'}" for i in range(9)]

        base_improve = {"High": 1.8, "Medium": 0.9, "Low": 0.3}[c["risk_tier"]]
        if c["bankruptcy_history"]: base_improve *= 0.4
        if c["num_late_payments"] > 2: base_improve *= 0.6

        noise       = np.random.normal(0, 0.6, 9)
        risk_scores = [min(100, max(0, c["risk_score"] + base_improve * i + noise[i])) for i in range(9)]
        tier_color  = RISK_COLORS[c["risk_tier"]]

        fig = go.Figure()
        upper = [min(100, s + 4) for s in risk_scores]
        lower = [max(0,   s - 4) for s in risk_scores]
        fig.add_trace(go.Scatter(
            x=qtrs + qtrs[::-1], y=upper + lower[::-1],
            fill="toself", fillcolor="rgba(236,17,26,0.07)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=qtrs, y=risk_scores, name="Risk Score",
            mode="lines+markers", line=dict(color=tier_color, width=2.5),
            marker=dict(size=6),
        ))
        fig.add_hrect(y0=72, y1=100, fillcolor="rgba(16,185,129,0.05)",
                      line_width=0, annotation_text="Low Risk", annotation_position="right")
        fig.add_hrect(y0=48, y1=72, fillcolor="rgba(245,158,11,0.05)",
                      line_width=0, annotation_text="Medium Risk", annotation_position="right")
        fig.add_hrect(y0=0,  y1=48, fillcolor="rgba(239,68,68,0.05)",
                      line_width=0, annotation_text="High Risk", annotation_position="right")
        fig.update_layout(xaxis_title="Quarter", yaxis_title="Risk Score (0–100)",
                          yaxis=dict(range=[0, 105]))
        st.plotly_chart(chart(fig, 360), use_container_width=True)

        r1, r2, r3 = st.columns(3)
        r1.metric("Current Score", f"{risk_scores[0]:.1f}/100")
        r2.metric("Q4 Projection", f"{risk_scores[4]:.1f}/100",
                  delta=f"{risk_scores[4]-risk_scores[0]:+.1f}")
        r3.metric("Q8 Projection", f"{risk_scores[8]:.1f}/100",
                  delta=f"{risk_scores[8]-risk_scores[0]:+.1f}")

    with tabs_f[3]:
        c = sel
        section_title(f"Financial Health Forecast — {c['name']} (5 Years)")
        np.random.seed(int(c["id"][1:]) + 20)
        months_range = list(range(0, 61, 3))
        labels_q     = [f"Q{i//3}" for i in months_range]

        active_loans      = [l for l in c["loans"] if l["status"] == "Active"]
        total_monthly_pmt = sum(l["monthly_payment"] for l in active_loans)
        debt_curve        = []
        d = float(c["total_debt"])
        for m in months_range:
            debt_curve.append(max(0, d - total_monthly_pmt * m * 0.85))

        growth_rate = {"Full-Time": 0.005, "Self-Employed": 0.006, "Part-Time": 0.003,
                       "Unemployed": -0.002, "Retired": 0.003}.get(c["employment_status"], 0.004)
        asset_curve = [
            c["total_assets"] * (1 + growth_rate) ** m
            + np.random.normal(0, c["total_assets"] * 0.01)
            for m in months_range
        ]
        net_worth = [a - dv for a, dv in zip(asset_curve, debt_curve)]

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            section_title("Debt Paydown Trajectory")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=labels_q, y=debt_curve, name="Total Debt",
                mode="lines+markers", fill="tozeroy",
                fillcolor="rgba(236,17,26,0.08)",
                line=dict(color=PRIMARY, width=2.5), marker=dict(size=4),
            ))
            fig.add_trace(go.Scatter(
                x=labels_q, y=asset_curve, name="Total Assets",
                mode="lines", line=dict(color=SUCCESS, width=2, dash="dot"),
            ))
            fig.update_layout(yaxis_tickformat="$,.0f",
                              xaxis_title="Quarter", legend=dict(x=0.01, y=0.99))
            st.plotly_chart(chart(fig, 320), use_container_width=True)

        with c2:
            section_title("Net Worth Projection")
            nw_colors = [SUCCESS if v >= 0 else DANGER for v in net_worth]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=labels_q, y=net_worth, name="Net Worth",
                marker_color=nw_colors, opacity=0.85,
            ))
            fig.add_hline(y=0, line_dash="dash", line_color=MUTED, line_width=1)
            fig.update_layout(yaxis_tickformat="$,.0f",
                              xaxis_title="Quarter", showlegend=False)
            st.plotly_chart(chart(fig, 320), use_container_width=True)

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Monthly Debt Payments",  f"${total_monthly_pmt:,.0f}")
        p2.metric("Projected Debt (5yr)",   f"${debt_curve[-1]:,.0f}",
                  delta=f"${debt_curve[-1]-debt_curve[0]:+,.0f}")
        p3.metric("Projected Assets (5yr)", f"${asset_curve[-1]:,.0f}",
                  delta=f"${asset_curve[-1]-asset_curve[0]:+,.0f}")
        p4.metric("Net Worth (5yr)",        f"${net_worth[-1]:,.0f}",
                  delta=f"${net_worth[-1]-net_worth[0]:+,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# VIEW 5 — AI ASSISTANT
# ─────────────────────────────────────────────────────────────────────────────

elif view == "AI Assistant":
    page_header(
        "AI Lending Assistant",
        "Ask anything about clients, risk, or loan eligibility",
    )

    if not OPENROUTER_KEY:
        st.error("No API key found. Add OpenrouterApiKey to data/.env")
        st.stop()

    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "pending_q" not in st.session_state:
        st.session_state.pending_q = None

    # Quick prompts
    st.markdown(
        f"<div style='font-size:0.7rem;font-weight:600;color:{MUTED};"
        f"text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.6rem'>"
        f"Quick prompts</div>",
        unsafe_allow_html=True,
    )
    prompts = [
        f"Is {sel['name']} eligible for a $30,000 personal loan?",
        "Which clients have the highest default risk?",
        "List clients with credit scores over 750 and low debt.",
        "Who has the worst debt-to-income ratio?",
    ]
    q1, q2, q3, q4 = st.columns(4)
    for col, prompt in zip([q1, q2, q3, q4], prompts):
        if col.button(prompt[:42] + "…", key=f"qp_{prompt[:20]}", use_container_width=True):
            st.session_state.pending_q = prompt

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    st.divider()

    def ask_rag(question: str) -> str:
        try:
            results = collection.query(query_texts=[question], n_results=5)
            docs    = results["documents"][0]
            context = "\n\n---\n\n".join(docs)
        except Exception as e:
            return f"⚠️ Database error: {e}"
        system_msg = (
            "You are a senior lending officer at Scotiabank with expertise in credit risk. "
            "Answer based on the customer profiles below. Be direct, cite specific numbers, "
            "and flag risk factors clearly. For loan eligibility give a clear APPROVED / "
            "CONDITIONALLY APPROVED / DECLINED verdict."
        )
        return ai_chat(system_msg,
                       f"Customer profiles:\n\n{context}\n\nQuestion: {question}",
                       model=OR_FAST_MODEL)

    # Chat history
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Quick prompt trigger
    if st.session_state.pending_q:
        question = st.session_state.pending_q
        st.session_state.pending_q = None
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Analysing..."):
                answer = ask_rag(question)
            st.markdown(answer)
        st.session_state.chat.append({"role": "assistant", "content": answer})

    # Typed input
    user_input = st.chat_input("Ask about any client, loan, or risk…")
    if user_input:
        st.session_state.chat.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Analysing..."):
                answer = ask_rag(user_input)
            st.markdown(answer)
        st.session_state.chat.append({"role": "assistant", "content": answer})

    if st.session_state.chat:
        if st.button("Clear conversation"):
            st.session_state.chat = []
            st.rerun()
