"""
Scotiabank RAG Lending Intelligence Platform
Streamlit + ChromaDB + OpenRouter
Streamlit + ChromaDB + OpenRouter
"""

import json
import os
import re
import hashlib
from html import escape
from datetime import datetime, date, timedelta

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
CUSTOMERS_JSON_PATH = os.environ.get("CustomersJsonPath", "data/customers.json").strip().strip("'\"")
CHROMA_DB_PATH = os.environ.get("ChromaDbPath", "data/chroma_db").strip().strip("'\"")
OFFER_JSON_PATH = os.environ.get("OfferJsonPath", "data/offer.json").strip().strip("'\"")


def stable_seed_from_customer_id(customer_id):
    customer_id = str(customer_id or "0")
    digest = hashlib.md5(customer_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)

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
  overflow: hidden !important;
  max-width: 100% !important;
  box-sizing: border-box !important;
}}
div[data-testid="stPlotlyChart"] > div,
div[data-testid="stPlotlyChart"] .js-plotly-plot,
div[data-testid="stPlotlyChart"] .plot-container {{
  width: 100% !important;
  max-width: 100% !important;
  overflow: hidden !important;
  box-sizing: border-box !important;
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

st.markdown(f"""
<style>
.pillar-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 14px;
  align-items: stretch;
}}
.pillar-link-card {{
  display: block;
  position: relative;
  background: {SURFACE};
  border: 1px solid {BORDER};
  border-radius: 16px;
  padding: 1rem 1.15rem;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 4px 14px rgba(0,0,0,0.03);
  text-decoration: none !important;
  transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
  height: 100%;
}}
.pillar-link-card::before {{
  content: '';
  position: absolute;
  inset: 0 0 auto 0;
  height: 3px;
  border-radius: 16px 16px 0 0;
  background: linear-gradient(90deg, {PRIMARY} 0%, #ff6b72 100%);
}}
.pillar-link-card:hover {{
  transform: translateY(-2px);
  border-color: rgba(236,17,26,0.35);
  box-shadow: 0 10px 28px rgba(15,23,42,0.08);
}}
.pillar-link-eyebrow {{
  font-size: 0.62rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: {PRIMARY};
  margin-bottom: 0.45rem;
}}
.pillar-link-title {{
  font-size: 0.93rem;
  font-weight: 700;
  color: {DARK};
  margin-bottom: 0.45rem;
}}
.pillar-link-copy {{
  font-size: 0.8rem;
  color: {MUTED};
  line-height: 1.55;
  margin-bottom: 0.7rem;
}}
.pillar-link-preview {{
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 7px 0;
  border-bottom: 1px solid {BORDER};
  font-size: 0.8rem;
}}
.pillar-link-preview span:first-child {{
  color: {MUTED};
}}
.pillar-link-preview span:last-child {{
  color: {DARK};
  font-weight: 600;
  text-align: right;
}}
.pillar-link-footer {{
  margin-top: 0.8rem;
  font-size: 0.75rem;
  font-weight: 700;
  color: {PRIMARY};
  letter-spacing: 0.03em;
}}
.pillar-breadcrumb {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 0.85rem;
  font-size: 0.78rem;
  font-weight: 600;
  color: {PRIMARY} !important;
  text-decoration: none !important;
}}
.pillar-breadcrumb:hover {{
  text-decoration: underline !important;
}}
.pillar-detail-header {{
  background: {SURFACE};
  border: 1px solid {BORDER};
  border-radius: 18px;
  padding: 1.2rem 1.45rem 1.1rem;
  margin-bottom: 1rem;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 4px 14px rgba(0,0,0,0.03);
  border-left: 4px solid {PRIMARY};
}}
.pillar-detail-kicker {{
  font-size: 0.68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: {PRIMARY};
}}
.pillar-detail-title {{
  font-size: 1.24rem;
  font-weight: 800;
  color: {DARK};
  letter-spacing: -0.03em;
  margin-top: 0.3rem;
}}
.pillar-detail-subtitle {{
  font-size: 0.84rem;
  color: {MUTED};
  margin-top: 0.35rem;
}}
.pillar-badge {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  border: 1px solid transparent;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_customers():
    if not os.path.exists(CUSTOMERS_JSON_PATH):
        st.error(f"Customer data not found at `{CUSTOMERS_JSON_PATH}`.")
        st.stop()
    with open(CUSTOMERS_JSON_PATH) as f:
        return json.load(f)


@st.cache_data
def load_offer_catalog():
    # Load the offer catalog from data/offer.json once and reuse it across reruns.
    if not os.path.exists(OFFER_JSON_PATH):
        return [], f"Offer catalog not found at `{OFFER_JSON_PATH}`."
    try:
        with open(OFFER_JSON_PATH) as f:
            raw_data = json.load(f)
    except Exception as exc:
        return [], f"Offer catalog could not be loaded from `{OFFER_JSON_PATH}`: {exc}"

    if not isinstance(raw_data, list):
        return [], f"Offer catalog at `{OFFER_JSON_PATH}` must be a JSON array."

    catalog = []
    for record in raw_data:
        if not isinstance(record, dict):
            continue
        raw_signals = record.get("client_signal", [])
        if isinstance(raw_signals, str):
            raw_signals = [raw_signals]
        normalized_signals = [
            re.sub(r"[^a-z0-9]+", "_", str(signal).strip().lower()).strip("_")
            for signal in raw_signals
            if not is_missing(signal)
        ]
        catalog.append(
            {
                **record,
                "client_signal": raw_signals,
                "normalized_client_signals": [signal for signal in normalized_signals if signal],
            }
        )
    return catalog, None


@st.cache_resource
def get_chroma():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return client.get_collection("bank_customers", embedding_function=ef)

customers  = load_customers()
collection = get_chroma()
cust_index = {c["id"]: c for c in customers}
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


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def customer_field(customer, *keys, default=None):
    source_data = customer.get("source_data", {})
    if not isinstance(source_data, dict):
        source_data = {}

    for key in keys:
        if key in customer and not is_missing(customer[key]):
            return customer[key]
        if key in source_data and not is_missing(source_data[key]):
            return source_data[key]
    return default


def parse_bool(value):
    if is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1", "active"}:
        return True
    if text in {"false", "f", "no", "n", "0", "inactive"}:
        return False
    return None


def safe_text(value, default="Not available"):
    if is_missing(value):
        return default
    if isinstance(value, (list, tuple, set)):
        vals = [str(v).strip() for v in value if not is_missing(v)]
        return ", ".join(vals) if vals else default
    return str(value)


def safe_currency(value, default="Not available"):
    if is_missing(value):
        return default
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return default


def safe_count(value, default="Not available"):
    if is_missing(value):
        return default
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return safe_text(value, default)


def bool_label(value, default="Not available", true_text="Yes", false_text="No"):
    parsed = parse_bool(value)
    if parsed is None:
        return default
    return true_text if parsed else false_text


def infer_primacy_flag(customer):
    primacy = customer_field(customer, "primacy_flag")
    if not is_missing(primacy):
        return bool_label(primacy)
    segment = safe_text(customer_field(customer, "primary_segment"), default="")
    if segment == "Primacy":
        return "Yes"
    if segment in {"Near Primacy", "Non-Primacy"}:
        return "No"
    return "Not available"


def infer_primacy_steps_away(customer):
    steps = customer_field(customer, "primacy_steps_away")
    if not is_missing(steps):
        return safe_count(steps)
    segment = safe_text(customer_field(customer, "primary_segment"), default="")
    if segment == "Primacy":
        return "0"
    return "Not available"


def infer_flow_metrics(customer):
    txns = customer.get("transactions", [])
    inflow = customer_field(customer, "total_inflow_amount_cad_dda")
    outflow = customer_field(customer, "total_outflow_amount_cad_dda")
    internal = customer_field(customer, "total_internal_amount_cad_dda")
    product = customer_field(customer, "fof_counterparty_product_dda")

    if is_missing(inflow):
        inflow = sum(t.get("amount", 0) for t in txns if t.get("amount", 0) > 0)
    if is_missing(outflow):
        outflow = sum(abs(t.get("amount", 0)) for t in txns if t.get("amount", 0) < 0)
    if is_missing(internal):
        internal = sum(t.get("amount", 0) for t in txns if t.get("category") == "Transfer")
    if is_missing(product):
        product = customer_field(customer, "payment_category_dda", "payment_subcategory_dda", default="Not available")

    return {
        "inflow": safe_currency(inflow),
        "outflow": safe_currency(outflow),
        "internal": safe_currency(internal),
        "product": safe_text(product),
    }


def infer_account_usage(customer):
    active_chequing = customer_field(customer, "has_active_chequing", default=customer.get("checking_balance", 0) > 0)
    active_savings = customer_field(customer, "has_active_savings", default=customer.get("savings_balance", 0) > 0)
    product_usage = customer_field(customer, "product_usage_flag")
    if is_missing(product_usage):
        product_usage = "Active" if (
            customer.get("checking_balance", 0) > 0
            or customer.get("savings_balance", 0) > 0
            or customer.get("investment_balance", 0) > 0
            or customer.get("loans")
        ) else "Not available"
    digital = customer_field(customer, "has_digital_engagement_last_30days", "digital_engagement_flag_30days")
    if is_missing(digital):
        digital = True if customer.get("transactions") else None

    recent_usage = customer_field(customer, "transaction_for_flag_bb", "payment_category_dda")
    if is_missing(recent_usage):
        cats = [t.get("category") for t in customer.get("transactions", [])[:3] if not is_missing(t.get("category"))]
        recent_usage = ", ".join(cats) if cats else "Not available"

    return {
        "chequing": bool_label(active_chequing),
        "savings": bool_label(active_savings),
        "product_usage": safe_text(product_usage),
        "digital": bool_label(digital),
        "recent_usage": safe_text(recent_usage),
        "active_savings_bool": parse_bool(active_savings),
    }


def infer_goal_summary(customer):
    goal_count = customer_field(customer, "financial_goal_count", default=len(customer.get("financial_goals", [])))
    completed = customer_field(customer, "completed_goal_count", default=customer.get("completed_goal_count"))
    incomplete = customer_field(customer, "incomplete_goal_count", default=customer.get("incomplete_goal_count"))
    summary = customer_field(customer, "financial_goals_summary")
    if is_missing(summary):
        goals = customer.get("financial_goals", [])
        summary = " | ".join(
            f"{g.get('purpose', 'Goal')}: {g.get('status', 'Status unknown')}"
            for g in goals
        )
    return {
        "goal_count": safe_count(goal_count),
        "completed": safe_count(completed),
        "incomplete": safe_count(incomplete),
        "summary": safe_text(summary),
    }


def has_investment_product(customer):
    explicit_flags = [
        customer_field(customer, "has_active_registered_retirement_savings_account"),
        customer_field(customer, "has_active_registered_retirement_income_fund_account"),
        customer_field(customer, "has_smart_investor_plan"),
    ]
    if any(parse_bool(v) is True for v in explicit_flags):
        return True
    return customer.get("investment_balance", 0) > 0


def normalize_offer_signal(signal):
    return re.sub(r"[^a-z0-9]+", "_", str(signal).strip().lower()).strip("_")


def infer_client_signals(customer):
    # Infer reusable client signals from customer data so they can be matched to offer.json.
    signals = {}

    def add(signal, reason):
        normalized = normalize_offer_signal(signal)
        if normalized and normalized not in signals:
            signals[normalized] = reason

    age = safe_number(customer_field(customer, "age"), None)
    checking_balance = safe_number(customer_field(customer, "checking_balance", "balance_cad_1month_bb_d2d", "balance_cad_lmonth_bb_d2d"), 0.0) or 0.0
    savings_balance = safe_number(customer_field(customer, "savings_balance", "balance_cad_1month_hisa"), 0.0) or 0.0
    avg_balance = safe_number(customer_field(customer, "avg_balance_cad_1month_bb_d2d", "avg_balance_cad_lmonth_bb_d2d"), None)
    monthly_fee = (safe_number(customer_field(customer, "monthly_fee_cad_bb_d2d"), 0.0) or 0.0) + (safe_number(customer_field(customer, "total_fee_cad_1month_hisa"), 0.0) or 0.0)
    goal_count = int(safe_number(customer_field(customer, "financial_goal_count"), 0) or 0)
    digital_flag = parse_bool(customer_field(customer, "has_digital_engagement_last_30days", "digital_engagement_flag_30days"))
    active_savings = parse_bool(customer_field(customer, "has_active_savings"))
    active_chequing = parse_bool(customer_field(customer, "has_active_chequing"))
    combined_deposits = max(checking_balance, 0) + max(savings_balance, 0)

    paywave_amt = abs(safe_number(customer_field(customer, "net_paywave_amt_30days"), 0.0) or 0.0)
    online_amt = abs(safe_number(customer_field(customer, "net_online_amt_30days"), 0.0) or 0.0)
    chip_pin_amt = abs(safe_number(customer_field(customer, "net_chip_pin_amt_30days"), 0.0) or 0.0)
    magnetic_amt = abs(safe_number(customer_field(customer, "net_magnetic_stripe_amt_30days"), 0.0) or 0.0)
    grocery_amt = abs(safe_number(customer_field(customer, "net_grocery_amt_30days"), 0.0) or 0.0)
    dining_amt = abs(safe_number(customer_field(customer, "net_dining_amt_30days"), 0.0) or 0.0)
    fuel_amt = abs(safe_number(customer_field(customer, "net_fuel_amt_30days"), 0.0) or 0.0)
    transit_amt = abs(safe_number(customer_field(customer, "net_daily_transit_amt_30days"), 0.0) or 0.0)
    travel_amt = abs(safe_number(customer_field(customer, "net_travel_amt_30days"), 0.0) or 0.0)
    recurring_amt = abs(safe_number(customer_field(customer, "net_recurring_payment_amt_30days"), 0.0) or 0.0)
    foreign_amt = abs(safe_number(customer_field(customer, "net_foreign_amt_30days", "transaction_amount_gic"), 0.0) or 0.0)
    foreign_cash_advance_amt = abs(safe_number(customer_field(customer, "foreign_cash_advance_amt_30days"), 0.0) or 0.0)
    apple_pay_amt = abs(safe_number(customer_field(customer, "net_apple_pay_amt_30days", "net_apple_total_amt_30days"), 0.0) or 0.0)
    google_pay_amt = abs(safe_number(customer_field(customer, "net_google_total_amt_30days"), 0.0) or 0.0)
    samsung_pay_amt = abs(safe_number(customer_field(customer, "net_samsung_total_amt_30days"), 0.0) or 0.0)
    total_internal_flow = abs(safe_number(customer_field(customer, "total_internal_amount_cad_dda", "total_internal_amount_cad_bb"), 0.0) or 0.0)

    debit_usage = paywave_amt + chip_pin_amt + magnetic_amt
    digital_payment_usage = online_amt + apple_pay_amt + google_pay_amt + samsung_pay_amt + paywave_amt
    activity_score = debit_usage + digital_payment_usage + grocery_amt + dining_amt + fuel_amt + transit_amt + recurring_amt

    if age is not None and age <= 23:
        add("age_under_23", f"Client age is {int(age)}, which fits youth/student-style eligibility signals.")
    if age is not None and age <= 30:
        add("young_client", f"Client age is {int(age)}, which supports youth-oriented or early-stage banking offers.")
    if age is not None and age >= 60:
        add("age_60_plus", f"Client age is {int(age)}, which may be relevant for senior account benefits.")

    if activity_score < 350:
        add("low_transaction_volume", f"Recent payment activity is light at about {format_currency_value(activity_score)} over 30 days.")
        add("basic_banking_needs", "Recent transaction activity appears relatively simple and low volume.")
    elif activity_score < 1200:
        add("moderate_transaction_volume", f"Recent payment activity is moderate at about {format_currency_value(activity_score)} over 30 days.")
    else:
        add("high_transaction_volume", f"Recent payment activity is elevated at about {format_currency_value(activity_score)} over 30 days.")

    if debit_usage <= 100:
        add("limited_monthly_debit_activity", f"Debit-style spend is limited at about {format_currency_value(debit_usage)} over 30 days.")
    if debit_usage > 250:
        add("uses_debit_card_regularly", f"Debit-style spend is active at about {format_currency_value(debit_usage)} over 30 days.")
    if debit_usage > 600:
        add("frequent_debit_card_usage", f"Debit card usage is high at about {format_currency_value(debit_usage)} over 30 days.")
        add("frequent_debit_usage", f"Debit activity is elevated at about {format_currency_value(debit_usage)} over 30 days.")

    if digital_payment_usage > 250:
        add("digital_banking_user", "Digital or card-not-present payment activity is visible in the last 30 days.")
        add("uses_online_banking", "Online or wallet-based payment activity is visible in the last 30 days.")
    if digital_flag is True:
        add("digital_banking_user", "Digital engagement was active in the last 30 days.")
    if digital_payment_usage > 500:
        add("high_digital_payment_usage", f"Digital payment activity is elevated at about {format_currency_value(digital_payment_usage)} over 30 days.")
    if apple_pay_amt + google_pay_amt + samsung_pay_amt > 100:
        add("mobile_app_user", "Mobile wallet activity suggests active mobile banking behaviour.")
        add("uses_mobile_banking", "Mobile wallet usage suggests comfort with mobile channels.")

    if monthly_fee > 0:
        add("cost_sensitive_client", f"Recent account-related fees total about {format_currency_value(monthly_fee)}.")
        add("fee_charged_recently", f"Recent account-related fees total about {format_currency_value(monthly_fee)}.")
        add("monthly_account_fee_charged", f"Monthly account or related fees were charged recently ({format_currency_value(monthly_fee)}).")
    if monthly_fee >= 10:
        add("monthly_transaction_fees_detected", f"Fee activity appears meaningful at about {format_currency_value(monthly_fee)}.")

    if avg_balance is not None and checking_balance > 0:
        if abs(avg_balance - checking_balance) / max(checking_balance, 1) <= 0.2:
            add("stable_monthly_balance", f"Chequing balances have been relatively stable around {format_currency_value(avg_balance)}.")
    if checking_balance >= 3000:
        add("balance_above_3000", f"Chequing balance is above $3,000 at {format_currency_value(checking_balance)}.")
    if checking_balance >= 4000:
        add("balance_above_4000", f"Chequing balance is above $4,000 at {format_currency_value(checking_balance)}.")
    if checking_balance >= 6000:
        add("balance_above_6000", f"Chequing balance is above $6,000 at {format_currency_value(checking_balance)}.")
    if combined_deposits >= 30000:
        add("combined_balance_above_30000", f"Combined chequing and savings balances are above $30,000 at {format_currency_value(combined_deposits)}.")
    if combined_deposits >= 10000:
        add("high_deposit_balance", f"Combined deposit balances are healthy at {format_currency_value(combined_deposits)}.")

    if active_chequing is True:
        add("eligible_chequing_or_savings_account", "Client has an active chequing relationship.")
    if active_savings is True:
        add("uses_savings_account", "Client has an active savings relationship.")
    if active_savings is False:
        add("no_existing_savings_account", "Client does not appear to have an active savings account.")
        add("needs_basic_savings_account", "Client may benefit from adding a basic savings relationship.")
    if savings_balance < 3000:
        add("low_savings_balance", f"Savings balance is relatively low at {format_currency_value(savings_balance)}.")
    if savings_balance >= 10000:
        add("high_savings_balance", f"Savings balance is elevated at {format_currency_value(savings_balance)}.")
        add("idle_cash_balance", f"Savings balance suggests idle cash of about {format_currency_value(savings_balance)}.")
    if savings_balance >= 25000:
        add("large_savings_balance", f"Savings balance is large at {format_currency_value(savings_balance)}.")

    deposit_hisa = safe_number(customer_field(customer, "deposit_amount_cad_1month_hisa"), 0.0) or 0.0
    withdrawal_hisa = safe_number(customer_field(customer, "withdrawal_amount_cad_1month_hisa"), 0.0) or 0.0
    if deposit_hisa > 0 and withdrawal_hisa <= deposit_hisa * 0.4:
        add("stable_savings_balance", f"Savings deposits of {format_currency_value(deposit_hisa)} exceed withdrawals of {format_currency_value(withdrawal_hisa)}.")
        add("low_withdrawal_frequency", "Savings withdrawals are relatively light compared with deposits.")
    if total_internal_flow > 500:
        add("frequent_internal_transfers", f"Internal transfers total about {format_currency_value(total_internal_flow)}.")
        add("savings_transfer_activity", f"Internal movement suggests ongoing transfer activity of about {format_currency_value(total_internal_flow)}.")

    if goal_count > 0:
        add("savings_goal", f"Client has {goal_count} financial goal(s) recorded.")
    if active_savings is False and debit_usage > 250:
        add("needs_automatic_savings", "Client has active day-to-day spend but no active savings relationship.")
        add("round_up_savings_opportunity", "Frequent spend activity could support an automatic round-up savings conversation.")

    if travel_amt > 150:
        add("travel_transactions", f"Travel-related spend is visible at about {format_currency_value(travel_amt)} over 30 days.")
    if transit_amt > 75:
        add("transit_spend", f"Transit spend is visible at about {format_currency_value(transit_amt)} over 30 days.")
        add("commuter_client", "Transit activity suggests a commuter payment pattern.")
    if foreign_amt > 100 or foreign_cash_advance_amt > 0:
        add("uses_international_money_transfer", f"Foreign or cross-border activity totals about {format_currency_value(foreign_amt + foreign_cash_advance_amt)}.")
        add("frequent_cross_border_payments", "Cross-border or foreign transaction activity is visible.")
        add("international_client", "The client shows signs of international or foreign-currency activity.")
    if foreign_cash_advance_amt > 0:
        add("foreign_cash_withdrawals", f"Foreign cash advance activity is visible at about {format_currency_value(foreign_cash_advance_amt)}.")

    usd_markers = [
        safe_text(customer_field(customer, "transaction_currency_mft"), ""),
        safe_text(customer_field(customer, "transaction_currency_gic"), ""),
        safe_text(customer_field(customer, "currency_gic"), ""),
        safe_text(customer_field(customer, "product_name_gic"), ""),
        safe_text(customer_field(customer, "product_category_gic"), ""),
    ]
    if any("usd" in marker.lower() for marker in usd_markers):
        add("usd_transactions", "USD-denominated product or transaction activity is present in the client record.")
        add("foreign_currency_need", "USD-denominated activity suggests a foreign-currency banking need.")
        add("travel_to_us", "USD-denominated activity may indicate U.S. travel or payment needs.")

    if has_investment_product(customer):
        add("multi_product_client", "Client appears to hold both deposit and investment-style products.")

    gic_value = safe_number(customer_field(customer, "gic_market_value_cad_gic"), 0.0) or 0.0
    gic_days = safe_number(customer_field(customer, "days_to_maturity_gic"), None)
    if gic_value > 0:
        add("gic_interest", f"Client has GIC holdings with market value around {format_currency_value(gic_value)}.")
        add("conservative_investment_profile", "Existing GIC holdings suggest a conservative savings or investment profile.")
    if gic_value >= 10000:
        add("large_term_deposit_balance", f"GIC holdings are meaningful at about {format_currency_value(gic_value)}.")
    if gic_days is not None and gic_days <= 120:
        add("maturing_gic", f"A GIC is maturing relatively soon in about {int(gic_days)} day(s).")

    if paywave_amt + online_amt + grocery_amt + dining_amt > 900:
        add("high_card_spend", f"Recent card-like spend is elevated at about {format_currency_value(paywave_amt + online_amt + grocery_amt + dining_amt)}.")
        add("high_card_spend_potential", "Recent spend levels suggest potential value from card-linked package benefits.")
    if travel_amt + grocery_amt + dining_amt > 600:
        add("travel_or_cashback_spend_pattern", "Spend mix includes categories often linked to travel or cashback value discussions.")

    return signals


def get_recommended_offers(customer, top_n=5):
    # Score offers by how many inferred client signals overlap with each offer's client_signal list.
    offer_catalog, catalog_warning = load_offer_catalog()
    inferred_signals = infer_client_signals(customer)
    inferred_keys = set(inferred_signals.keys())
    scored_offers = []

    for offer in offer_catalog:
        matched_signals = [signal for signal in offer.get("normalized_client_signals", []) if signal in inferred_keys]
        if not matched_signals:
            continue
        scored_offers.append(
            {
                **offer,
                "match_score": len(matched_signals),
                "matched_signals": matched_signals,
                "matched_reasons": [inferred_signals[signal] for signal in matched_signals],
            }
        )

    scored_offers.sort(
        key=lambda offer: (
            -offer["match_score"],
            safe_text(offer.get("product_category"), ""),
            safe_text(offer.get("product_name"), ""),
        )
    )

    if scored_offers:
        return scored_offers[:top_n], catalog_warning, inferred_signals

    if not offer_catalog:
        return [], catalog_warning, inferred_signals

    fallback_categories = []
    if parse_bool(customer_field(customer, "has_active_savings")) is False:
        fallback_categories.extend(["savings", "savings_tool"])
    if safe_number(customer_field(customer, "gic_market_value_cad_gic"), 0.0):
        fallback_categories.append("gic_bundle")
    if parse_bool(customer_field(customer, "has_digital_engagement_last_30days", "digital_engagement_flag_30days")) is True:
        fallback_categories.append("digital_banking")
    if safe_number(customer_field(customer, "net_foreign_amt_30days"), 0.0):
        fallback_categories.extend(["foreign_currency_account", "international_payments"])
    fallback_categories.extend(["chequing", "chequing_package", "banking_package"])

    fallback_offers = []
    used_offer_ids = set()
    for category in fallback_categories:
        for offer in offer_catalog:
            if offer.get("benefit_id") in used_offer_ids:
                continue
            if safe_text(offer.get("product_category"), "").lower() == category.lower():
                fallback_offers.append(
                    {
                        **offer,
                        "match_score": 0,
                        "matched_signals": [],
                        "matched_reasons": [
                            "No direct signal match was detected, so this offer is shown as a general benefit based on the client's broader profile."
                        ],
                    }
                )
                used_offer_ids.add(offer.get("benefit_id"))
                break
        if len(fallback_offers) >= top_n:
            break

    if not fallback_offers:
        fallback_offers = [
            {
                **offer,
                "match_score": 0,
                "matched_signals": [],
                "matched_reasons": [
                    "No direct signal match was detected, so this offer is shown as a general benefit based on the client's broader profile."
                ],
            }
            for offer in offer_catalog[:top_n]
        ]

    return fallback_offers[:top_n], catalog_warning, inferred_signals


def build_offer_suggestions(customer):
    matched_offers, _, _ = get_recommended_offers(customer, top_n=3)
    if matched_offers:
        suggestions = []
        for offer in matched_offers:
            reason = offer["matched_reasons"][0] if offer.get("matched_reasons") else "General profile fit."
            suggestions.append(
                f"{safe_text(offer.get('product_name'))}: {reason}"
            )
        return suggestions
    return ["No clear offer was detected from the catalog; review general banking, savings, or package benefits for fit."]


def build_advisor_summary(customer):
    segment = safe_text(customer_field(customer, "primary_segment"))
    primacy = infer_primacy_flag(customer)
    steps_away = infer_primacy_steps_away(customer)
    flow = infer_flow_metrics(customer)
    usage = infer_account_usage(customer)
    goals = infer_goal_summary(customer)
    offers = build_offer_suggestions(customer)

    relationship = f"{customer['name']} is currently in the {segment} segment"
    if primacy != "Not available":
        relationship += f" with primacy status {primacy.lower()}"
    if steps_away not in {"0", "Not available"}:
        relationship += f" and {steps_away} remaining primacy step(s)"

    account = (
        f"Account usage shows chequing {usage['chequing'].lower()}, savings {usage['savings'].lower()}, "
        f"product usage marked as {usage['product_usage']}, and digital engagement {usage['digital'].lower()}."
    )
    flow_text = (
        f" Recent observed flow indicates inflow of {flow['inflow']}, outflow of {flow['outflow']}, "
        f"internal movement of {flow['internal']}, with key counterparty product noted as {flow['product']}."
    )
    goal_text = (
        f" Goals currently show {goals['goal_count']} total, {goals['completed']} completed, and "
        f"{goals['incomplete']} still in progress."
    )
    offer_text = f" Recommended advisor focus: {offers[0]}"
    return relationship + ". " + account + flow_text + goal_text + offer_text


def pillar_row_html(label, value):
    return (
        f"<div style='display:flex;justify-content:space-between;gap:12px;padding:8px 0;"
        f"border-bottom:1px solid {BORDER};font-size:0.84rem'>"
        f"<span style='color:{MUTED};min-width:0;flex:0 0 42%'>{escape(str(label))}</span>"
        f"<span style='font-weight:600;color:{DARK};text-align:right;min-width:0;flex:1;"
        f"white-space:normal;overflow-wrap:anywhere'>{escape(str(value))}</span>"
        f"</div>"
    )


def pillar_card_html(title, rows=None, body_html=""):
    rows_html = "".join(pillar_row_html(label, value) for label, value in (rows or []))
    return (
        f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:16px;"
        f"padding:1rem 1.15rem;box-shadow:0 1px 2px rgba(0,0,0,0.04),0 4px 14px rgba(0,0,0,0.03);"
        f"height:100%'>"
        f"<div style='font-size:0.78rem;font-weight:700;color:{DARK};letter-spacing:0.02em;"
        f"margin-bottom:0.85rem'>{escape(title)}</div>"
        f"{rows_html}{body_html}</div>"
    )


PILLAR_ROUTE_KEY = "pillar"
PILLAR_DEFS = [
    {
        "slug": "bns-relationships",
        "title": "BNS Relationships",
        "eyebrow": "Pillar 1",
        "description": "Primacy, relationship depth, product holdings, and advisor context.",
    },
    {
        "slug": "flow-of-fund",
        "title": "Flow of Fund",
        "eyebrow": "Pillar 2",
        "description": "Movement of funds across inflows, outflows, and counterparties.",
    },
    {
        "slug": "account-usage",
        "title": "Account Usage",
        "eyebrow": "Pillar 3",
        "description": "Usage signals across deposits, digital engagement, and activity.",
    },
    {
        "slug": "client-goal",
        "title": "Client Goal",
        "eyebrow": "Pillar 4",
        "description": "Goal counts, progress, and advisory planning context.",
    },
    {
        "slug": "offer",
        "title": "Offer",
        "eyebrow": "Pillar 5",
        "description": "Next-best-product and activation opportunities for the advisor.",
    },
    {
        "slug": "summary",
        "title": "Summary",
        "eyebrow": "Pillar 6",
        "description": "Advisor-ready synthesis of relationship, usage, and opportunity signals.",
    },
]
PILLAR_DEF_MAP = {pillar["slug"]: pillar for pillar in PILLAR_DEFS}
CLIENT_PROFILE_TABS = ["Overview", "Transactions", "Loans", "Analytics", "6 Pillars"]
CURRENT_PAGE_KEY = "current_page"
MAIN_NAV_WIDGET_KEY = "main_nav_widget"
VIEW_OPTIONS = [
    "📊   Portfolio",
    "👤   Client Profile",
    "📋   Loan Assessment",
    "📈   Forecasting",
    "🤖   AI Assistant",
]
VIEW_MAP = {
    "📊   Portfolio": "Portfolio Overview",
    "👤   Client Profile": "Client Profile",
    "📋   Loan Assessment": "Loan Assessment",
    "📈   Forecasting": "Forecasting",
    "🤖   AI Assistant": "AI Assistant",
}
VIEW_MAP_REVERSE = {value: key for key, value in VIEW_MAP.items()}


def _query_param_value(key, default=None):
    value = st.query_params.get(key, default)
    if isinstance(value, list):
        return value[0] if value else default
    return value


def clear_pillar_query_param():
    try:
        del st.query_params[PILLAR_ROUTE_KEY]
    except Exception:
        pass


def on_main_nav_change():
    selected_label = st.session_state.get(MAIN_NAV_WIDGET_KEY, VIEW_OPTIONS[0])
    st.session_state[CURRENT_PAGE_KEY] = VIEW_MAP[selected_label]


def ensure_navigation_state():
    default_customer_id = customers[0]["id"] if customers else None
    legacy_page = st.session_state.get("page", "Portfolio Overview")
    st.session_state.setdefault(CURRENT_PAGE_KEY, legacy_page)
    st.session_state.setdefault(MAIN_NAV_WIDGET_KEY, VIEW_MAP_REVERSE.get(st.session_state[CURRENT_PAGE_KEY], VIEW_OPTIONS[0]))
    st.session_state.setdefault("active_client_tab", "Overview")
    st.session_state.setdefault("selected_pillar", "overview")
    st.session_state.setdefault("selected_customer_id", default_customer_id)

    query_pillar = _query_param_value(PILLAR_ROUTE_KEY)
    if query_pillar is not None:
        query_pillar = str(query_pillar).strip().lower()
        if query_pillar in PILLAR_DEF_MAP or query_pillar == "overview":
            st.session_state[CURRENT_PAGE_KEY] = "Client Profile"
            st.session_state["active_client_tab"] = "6 Pillars"
            st.session_state["selected_pillar"] = query_pillar
        clear_pillar_query_param()

    if st.session_state["active_client_tab"] != "6 Pillars":
        st.session_state["selected_pillar"] = "overview"

    if st.session_state["selected_customer_id"] not in cust_index and default_customer_id is not None:
        st.session_state["selected_customer_id"] = default_customer_id

    nav_label = VIEW_MAP_REVERSE.get(
        st.session_state.get(CURRENT_PAGE_KEY, "Portfolio Overview"),
        VIEW_OPTIONS[0],
    )
    if st.session_state.get(MAIN_NAV_WIDGET_KEY) != nav_label:
        st.session_state[MAIN_NAV_WIDGET_KEY] = nav_label


def set_main_page(page):
    st.session_state[CURRENT_PAGE_KEY] = page
    if page != "Client Profile":
        st.session_state["active_client_tab"] = "Overview"
        st.session_state["selected_pillar"] = "overview"
        clear_pillar_query_param()


def set_client_profile_tab(tab_name):
    st.session_state[CURRENT_PAGE_KEY] = "Client Profile"
    st.session_state["active_client_tab"] = tab_name
    if tab_name != "6 Pillars":
        st.session_state["selected_pillar"] = "overview"
        clear_pillar_query_param()


def set_selected_pillar(pillar_slug):
    st.session_state[CURRENT_PAGE_KEY] = "Client Profile"
    st.session_state["active_client_tab"] = "6 Pillars"
    st.session_state["selected_pillar"] = pillar_slug
    clear_pillar_query_param()


def get_pillar_route():
    selected = st.session_state.get("selected_pillar", "overview")
    return selected if selected in {"overview", *PILLAR_DEF_MAP.keys()} else "overview"


def pillar_href(slug):
    return f"?{PILLAR_ROUTE_KEY}={slug}"


def split_steps(value):
    if is_missing(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if not is_missing(v)]
    parts = [p.strip(" -") for p in re.split(r"\s*\|\s*|;\s*|\n+|,\s*(?=[A-Za-z])", str(value)) if p.strip()]
    return parts


def describe_bool(value, true_label="Yes", false_label="No", default="Not available"):
    parsed = parse_bool(value)
    if parsed is None:
        return default
    return true_label if parsed else false_label


def open_active_label(customer, open_key, active_key):
    opened = customer_field(customer, open_key)
    active = customer_field(customer, active_key)
    open_label = describe_bool(opened, "Open", "Not open")
    active_label = describe_bool(active, "Active", "Not active")
    if open_label == "Not available" and active_label == "Not available":
        return "Not available"
    if active_label == "Not available":
        return open_label
    if open_label == "Not available":
        return active_label
    return f"{open_label} / {active_label}"


def product_plan_status(customer, key):
    return describe_bool(customer_field(customer, key), "Active", "Not active")


def primacy_steps_meta(customer):
    raw = customer_field(customer, "primacy_steps_away")
    if is_missing(raw):
        return None, "Not available"
    try:
        steps = int(float(raw))
        steps = max(0, min(4, steps))
        return steps, str(steps)
    except (TypeError, ValueError):
        return None, safe_text(raw)


def badge_html(text, tone="neutral"):
    palette = {
        "success": ("rgba(16,185,129,0.12)", SUCCESS, f"{SUCCESS}30"),
        "danger": ("rgba(239,68,68,0.12)", DANGER, f"{DANGER}30"),
        "warning": ("rgba(245,158,11,0.12)", WARNING, f"{WARNING}30"),
        "neutral": ("rgba(100,116,139,0.12)", MUTED, f"{MUTED}30"),
    }
    bg, color, border = palette[tone]
    return (
        f"<span class='pillar-badge' style='background:{bg};color:{color};border-color:{border}'>"
        f"{escape(str(text))}</span>"
    )


def pillar_link_card_html(title, slug, eyebrow, description, rows):
    preview_html = "".join(
        f"<div class='pillar-link-preview'><span>{escape(str(label))}</span><span>{escape(str(value))}</span></div>"
        for label, value in rows
    )
    return (
        f"<a class='pillar-link-card' href='{pillar_href(slug)}' target='_self'>"
        f"<div class='pillar-link-eyebrow'>{escape(eyebrow)}</div>"
        f"<div class='pillar-link-title'>{escape(title)}</div>"
        f"<div class='pillar-link-copy'>{escape(description)}</div>"
        f"{preview_html}"
        f"<div class='pillar-link-footer'>Open detail →</div>"
        f"</a>"
    )


def pillar_detail_header_html(title, customer, effective_date=None):
    return (
        f"<div class='pillar-detail-header'>"
        f"<div class='pillar-detail-kicker'>Client Profile / 6 Pillars</div>"
        f"<div class='pillar-detail-title'>{escape(title)}</div>"
        f"<div class='pillar-detail-subtitle'>{escape(customer['name'])} · {escape(customer['id'])}"
        + (f" · Business effective date {escape(str(effective_date))}" if effective_date else "")
        + "</div></div>"
    )


def build_bns_advisor_insight(customer):
    primacy_flag = customer_field(customer, "primacy_flag")
    primacy_text = describe_bool(primacy_flag, "in the Primacy program", "not in the Primacy program")
    product_usage = safe_text(customer_field(customer, "product_usage_flag"))
    digital_30 = describe_bool(customer_field(customer, "has_digital_engagement_last_30days", "digital_engagement_flag_30days"))
    missing_steps = split_steps(customer_field(customer, "missing_primacy_steps"))

    insight = f"Client is {primacy_text} with product usage flagged as {product_usage.lower()} and digital engagement in the last 30 days marked {digital_30.lower()}."
    if missing_steps:
        insight += " Advisor action should focus on: " + "; ".join(missing_steps[:3]) + "."
    elif digital_30 == "No":
        insight += " Re-engage the client through digital activation and recurring activity setup."
    else:
        insight += " Relationship signals are stable; focus on deepening product usage and retention."
    return insight


def render_placeholder_pillar_detail(customer, title):
    if st.button("← Back to 6 Pillars Overview", key=f"back_to_pillars_{title}", type="secondary"):
        set_selected_pillar("overview")
        st.rerun()
    st.markdown(pillar_detail_header_html(title, customer), unsafe_allow_html=True)
    st.markdown(
        pillar_card_html(
            f"{title} Detail",
            body_html=(
                f"<div style='font-size:0.84rem;color:{DARK};line-height:1.6'>"
                f"Detailed view for {escape(title)} is reserved and ready for expansion. "
                f"For now, return to the 6 Pillars overview to navigate to another pillar.</div>"
            ),
        ),
        unsafe_allow_html=True,
    )


def render_bns_relationships_detail(customer):
    effective_date = safe_text(customer_field(customer, "business_effective_date"))
    primacy_flag = customer_field(customer, "primacy_flag")
    primacy_label = describe_bool(primacy_flag, "In Primacy Program", "Not in Primacy Program")
    primacy_program_copy = (
        "Client is in the Primacy program."
        if parse_bool(primacy_flag) is True else
        "Client is not currently in the Primacy program."
        if parse_bool(primacy_flag) is False else
        "Primacy program status is not available."
    )
    steps_num, steps_display = primacy_steps_meta(customer)
    missing_steps_list = split_steps(customer_field(customer, "missing_primacy_steps"))
    missing_steps_text = " • ".join(missing_steps_list) if missing_steps_list else safe_text(customer_field(customer, "missing_primacy_steps"))
    action_items_html = ""
    if steps_num is not None and steps_num > 0 and missing_steps_list:
        items = "".join(
            f"<li style='margin-bottom:4px'>{escape(item)}</li>"
            for item in missing_steps_list
        )
        action_items_html = (
            f"<div style='padding-top:10px'>"
            f"<div style='font-size:0.7rem;color:{MUTED};text-transform:uppercase;letter-spacing:0.06em;font-weight:700;margin-bottom:6px'>Advisor Action Items</div>"
            f"<ul style='margin:0 0 0 1rem;padding:0;color:{DARK};font-size:0.84rem;line-height:1.55'>{items}</ul></div>"
        )

    primacy_rows = [
        ("Primary Segment", safe_text(customer_field(customer, "primary_segment"))),
        ("Primacy Status", primacy_label),
        ("Primacy Steps Away", steps_display),
        ("Missing Primacy Steps", missing_steps_text),
    ]
    primacy_body = (
        f"<div style='font-size:0.84rem;color:{DARK};line-height:1.6;padding-top:10px'>{escape(primacy_program_copy)}</div>"
        f"{action_items_html}"
    )

    product_rows = [
        ("Product Usage Flag", safe_text(customer_field(customer, "product_usage_flag"))),
        ("Chequing", open_active_label(customer, "has_open_chequing", "has_active_chequing")),
        ("Savings", open_active_label(customer, "has_open_savings", "has_active_savings")),
        ("RRSP", open_active_label(customer, "has_open_registered_retirement_savings_account", "has_active_registered_retirement_savings_account")),
        ("RRIF", open_active_label(customer, "has_open_registered_retirement_income_fund_account", "has_active_registered_retirement_income_fund_account")),
        ("FHSA", open_active_label(customer, "has_open_registered_first_home_savings_account", "has_active_registered_first_home_savings_account")),
        ("RDSP", open_active_label(customer, "has_open_registered_disability_savings_account", "has_active_registered_disability_savings_account")),
        ("RESP", open_active_label(customer, "has_open_registered_education_savings_account", "has_active_registered_education_savings_account")),
        ("Advice Plus Plan", product_plan_status(customer, "has_advice_plus_plan")),
        ("Smart Investor Plan", product_plan_status(customer, "has_smart_investor_plan")),
    ]

    digital_rows = [
        ("Digital Engagement Flag 30 Days", describe_bool(customer_field(customer, "digital_engagement_flag_30days"))),
        ("Digital Engagement Last 30 Days", describe_bool(customer_field(customer, "has_digital_engagement_last_30days"))),
        ("Digital Engagement Last 90 Days", describe_bool(customer_field(customer, "has_digital_engagement_last_90days"))),
        ("PAC Last 30 Days", describe_bool(customer_field(customer, "has_pac_last_30days"), "Active", "Not active")),
        ("PAD Last 30 Days", describe_bool(customer_field(customer, "has_pad_last_30days"), "Active", "Not active")),
    ]

    advisor_name = safe_text(customer_field(customer, "note_advisor_name"))
    advisor_body = (
        f"<div style='padding-bottom:10px'>{pillar_row_html('Advisor', advisor_name)}</div>"
        f"<div style='font-size:0.84rem;color:{DARK};line-height:1.65'>{escape(build_bns_advisor_insight(customer))}</div>"
    )

    unsolicited = parse_bool(customer_field(customer, "unsolicited_order_flag_ip"))
    flag_badge = (
        badge_html("TRUE", "warning") if unsolicited is True else
        badge_html("FALSE", "neutral") if unsolicited is False else
        badge_html("NOT AVAILABLE", "neutral")
    )
    flags_body = (
        f"<div style='display:flex;justify-content:space-between;align-items:center;gap:12px;'>"
        f"<div><div style='font-size:0.84rem;font-weight:600;color:{DARK}'>Unsolicited Order Flag</div>"
        f"<div style='font-size:0.76rem;color:{MUTED};margin-top:3px'>Indicator for investment-initiated unsolicited order activity.</div></div>"
        f"{flag_badge}</div>"
    )

    if st.button("← Back to 6 Pillars Overview", key="back_to_pillars_bns_relationships", type="secondary"):
        set_selected_pillar("overview")
        st.rerun()
    st.markdown(pillar_detail_header_html("BNS Relationships", customer, effective_date), unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(pillar_card_html("Primacy Summary", primacy_rows, primacy_body), unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown(pillar_card_html("Digital Engagement", digital_rows), unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown(pillar_card_html("Relationship Flags", body_html=flags_body), unsafe_allow_html=True)

    with c2:
        st.markdown(pillar_card_html("Product Relationship", product_rows), unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown(pillar_card_html("Advisor Notes", body_html=advisor_body), unsafe_allow_html=True)


def safe_number(value, default=None):
    if is_missing(value):
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        try:
            if np.isnan(value):
                return default
        except TypeError:
            pass
        return float(value)
    text = str(value).strip().replace(",", "").replace("$", "")
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def format_currency_value(value, default="Not available"):
    numeric = safe_number(value, None)
    if numeric is None:
        return default
    return f"${numeric:,.2f}"


def format_display_value(value):
    parsed = parse_bool(value)
    if parsed is True:
        return "Yes"
    if parsed is False:
        return "No"
    numeric = safe_number(value, None)
    if numeric is not None and isinstance(value, str) and any(ch in value for ch in [".", ",", "$"]):
        return f"{numeric:,.2f}"
    if isinstance(value, str):
        return value if value.strip() else "Not available"
    if value is None:
        return "Not available"
    return str(value)


def flow_value(customer, key, default=None):
    return customer_field(customer, key, default=default)


def flow_amount(customer, key):
    return safe_number(flow_value(customer, key), 0.0)


def normalize_flow_direction(value):
    if is_missing(value):
        return "Unknown"
    text = str(value).strip().lower()
    if "internal" in text or "transfer" in text:
        return "Internal Transfer"
    if "inflow" in text or "in " == text or text == "in":
        return "Inflow"
    if "outflow" in text or "out " == text or text == "out":
        return "Outflow"
    if "buy" in text or "purchase" in text:
        return "Buy"
    if "sell" in text or "redeem" in text:
        return "Sell"
    if "debit" in text:
        return "Debit"
    if "credit" in text:
        return "Credit"
    return text.title() if text else "Unknown"


def normalize_counterparty_product(value):
    if is_missing(value):
        return "Unknown"
    text = str(value).strip().lower()
    if "chequ" in text:
        return "Chequing"
    if "saving" in text:
        return "Savings"
    if "credit" in text and "card" in text:
        return "Credit Card"
    if "mortgage" in text:
        return "Mortgage"
    if "loan" in text or "loc" in text:
        return "Loan"
    if "gic" in text:
        return "GIC"
    if "mutual" in text or "mft" in text:
        return "Mutual Fund"
    if "invest" in text or "rrsp" in text or "rrif" in text:
        return "Investment"
    if "external" in text:
        return "External Account"
    return text.title() if text else "Unknown"


def humanize_field_label(key):
    acronyms = {
        "dda": "DDA",
        "bb": "BB",
        "d2d": "D2D",
        "gic": "GIC",
        "ipcash": "IPCash",
        "mft": "MFT",
        "cad": "CAD",
        "fi": "FI",
        "fof": "Flow of Fund",
        "snc": "Source",
        "dst": "Destination",
        "txn": "Txn",
        "ip": "IP",
    }
    words = []
    for part in key.split("_"):
        words.append(acronyms.get(part.lower(), part.capitalize()))
    label = " ".join(words)
    replacements = {
        "Customer Id": "Customer ID",
        "Account Id": "Account ID",
        "Fi": "FI",
        "Txn": "Transaction",
        "Cad": "CAD",
    }
    for src, target in replacements.items():
        label = label.replace(src, target)
    return label


def detail_table_df(customer, keys):
    rows = []
    for key in keys:
        value = flow_value(customer, key)
        rows.append(
            {
                "Variable Label": humanize_field_label(key),
                "Value": format_display_value(value),
            }
        )
    return pd.DataFrame(rows)


def render_detail_section(label, key, table_df, height=420):
    expanded = st.toggle(label, key=key, value=False)
    if expanded:
        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
            height=height,
        )


def build_flow_advisor_insight(customer):
    dda_inflow = flow_amount(customer, "total_inflow_amount_cad_dda")
    dda_outflow = flow_amount(customer, "total_outflow_amount_cad_dda")
    dda_internal = flow_amount(customer, "total_internal_amount_cad_dda")
    bb_inflow = flow_amount(customer, "total_inflow_amount_cad_bb")
    bb_outflow = flow_amount(customer, "total_outflow_amount_cad_bb")
    bb_internal = flow_amount(customer, "total_internal_amount_cad_bb")
    net_total = (dda_inflow - dda_outflow) + (bb_inflow - bb_outflow)

    insights = []
    if net_total < 0:
        insights.append("Client shows net cash outflow; consider reviewing liquidity needs or upcoming large payments.")
    elif net_total > 0:
        insights.append("Client shows positive net cash inflow; consider savings, investment, or GIC opportunities.")

    products = [
        normalize_counterparty_product(flow_value(customer, "fof_counterparty_product_dda")),
        normalize_counterparty_product(flow_value(customer, "fof_counterparty_product_bb")),
        normalize_counterparty_product(flow_value(customer, "fof_counterparty_product_mft")),
        normalize_counterparty_product(flow_value(customer, "fof_counterparty_product_gic")),
    ]
    if any(product in {"GIC", "Mutual Fund", "Investment"} for product in products):
        insights.append("Client has investment-related flow activity.")

    if (dda_internal + bb_internal) > max(1000.0, (dda_inflow + bb_inflow + dda_outflow + bb_outflow) * 0.35):
        insights.append("Client has significant internal transfers; review product movement and consolidation opportunities.")

    payment_categories = " ".join(
        str(flow_value(customer, key, "") or "").lower()
        for key in ("payment_category_dda", "payment_category_bb", "payment_category_mft", "payment_category_gic")
    )
    if "payroll" in payment_categories:
        insights.append("Payroll activity detected; client may be suitable for Primacy relationship review.")

    if not insights:
        insights.append("Flow activity is limited or mixed; use the detailed flow sections to identify product movement opportunities.")
    return " ".join(insights)


def render_flow_of_fund_detail(customer):
    if st.button("← Back to 6 Pillars Overview", key="back_to_pillars_flow_of_fund", type="secondary"):
        set_selected_pillar("overview")
        st.rerun()

    st.markdown(
        pillar_detail_header_html(
            "Flow of Fund",
            customer,
            safe_text(flow_value(customer, "business_effective_date")),
        ),
        unsafe_allow_html=True,
    )

    dda_inflow = flow_amount(customer, "total_inflow_amount_cad_dda")
    dda_outflow = flow_amount(customer, "total_outflow_amount_cad_dda")
    dda_internal = flow_amount(customer, "total_internal_amount_cad_dda")
    bb_inflow = flow_amount(customer, "total_inflow_amount_cad_bb")
    bb_outflow = flow_amount(customer, "total_outflow_amount_cad_bb")
    bb_internal = flow_amount(customer, "total_internal_amount_cad_bb")
    net_dda = dda_inflow - dda_outflow
    net_bb = bb_inflow - bb_outflow

    kpi_data = [
        ("Total DDA Inflow", dda_inflow),
        ("Total DDA Outflow", dda_outflow),
        ("Total DDA Internal Movement", dda_internal),
        ("Total BB Inflow", bb_inflow),
        ("Total BB Outflow", bb_outflow),
        ("Total BB Internal Movement", bb_internal),
        ("Net DDA Flow", net_dda),
        ("Net BB Flow", net_bb),
    ]
    kpi_cols = st.columns(4)
    for idx, (label, value) in enumerate(kpi_data):
        with kpi_cols[idx % 4]:
            delta = None
            if label == "Net DDA Flow":
                delta = "Positive" if value > 0 else "Negative" if value < 0 else "Flat"
            if label == "Net BB Flow":
                delta = "Positive" if value > 0 else "Negative" if value < 0 else "Flat"
            st.metric(label, format_currency_value(value), delta=delta)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        section_title("Inflow vs Outflow vs Internal Movement")
        flow_compare = pd.DataFrame(
            [
                {"Account": "DDA", "Metric": "Inflow", "Amount": dda_inflow},
                {"Account": "DDA", "Metric": "Outflow", "Amount": dda_outflow},
                {"Account": "DDA", "Metric": "Internal Movement", "Amount": dda_internal},
                {"Account": "BB", "Metric": "Inflow", "Amount": bb_inflow},
                {"Account": "BB", "Metric": "Outflow", "Amount": bb_outflow},
                {"Account": "BB", "Metric": "Internal Movement", "Amount": bb_internal},
            ]
        )
        fig = px.bar(
            flow_compare,
            x="Account",
            y="Amount",
            color="Metric",
            barmode="group",
            color_discrete_map={
                "Inflow": SUCCESS,
                "Outflow": PRIMARY,
                "Internal Movement": "#3b82f6",
            },
            labels={"Amount": "Amount (CAD)", "Account": "", "Metric": ""},
        )
        fig.update_traces(marker_line_width=0, opacity=0.86)
        fig.update_layout(yaxis_tickformat="$,.0f")
        st.plotly_chart(chart(fig, 300), use_container_width=True)

    with c2:
        section_title("Flow Direction Breakdown")
        direction_values = [
            normalize_flow_direction(flow_value(customer, "flow_direction_d2d")),
            normalize_flow_direction(flow_value(customer, "flow_direction_gic_product")),
            normalize_flow_direction(flow_value(customer, "flow_direction_ipcash")),
            normalize_flow_direction(flow_value(customer, "transaction_direction_mft")),
            normalize_flow_direction(flow_value(customer, "transaction_direction_gic")),
        ]
        direction_values = [value for value in direction_values if value != "Unknown"]
        direction_counts = pd.Series(direction_values).value_counts().reset_index() if direction_values else pd.DataFrame(columns=["index", "count"])
        if len(direction_values) <= 1:
            summary_rows = [
                ("D2D", normalize_flow_direction(flow_value(customer, "flow_direction_d2d"))),
                ("GIC Product", normalize_flow_direction(flow_value(customer, "flow_direction_gic_product"))),
                ("IPCash", normalize_flow_direction(flow_value(customer, "flow_direction_ipcash"))),
                ("MFT", normalize_flow_direction(flow_value(customer, "transaction_direction_mft"))),
                ("GIC", normalize_flow_direction(flow_value(customer, "transaction_direction_gic"))),
            ]
            st.markdown(pillar_card_html("Flow Direction Summary", summary_rows), unsafe_allow_html=True)
        else:
            direction_counts.columns = ["Direction", "Count"]
            fig = px.pie(
                direction_counts,
                names="Direction",
                values="Count",
                hole=0.52,
                color_discrete_sequence=[PRIMARY, SUCCESS, "#3b82f6", WARNING, "#8b5cf6", "#06b6d4", "#94a3b8"],
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(chart(fig, 300), use_container_width=True)

    c3, c4 = st.columns(2, gap="medium")
    with c3:
        section_title("Counterparty Product Mix")
        products = [
            normalize_counterparty_product(flow_value(customer, "fof_counterparty_product_dda")),
            normalize_counterparty_product(flow_value(customer, "fof_counterparty_product_bb")),
            normalize_counterparty_product(flow_value(customer, "fof_counterparty_product_mft")),
            normalize_counterparty_product(flow_value(customer, "fof_counterparty_product_gic")),
        ]
        products = [product for product in products if product != "Unknown"]
        if len(set(products)) <= 1:
            st.markdown(
                pillar_card_html(
                    "Counterparty Product",
                    [
                        ("DDA", normalize_counterparty_product(flow_value(customer, "fof_counterparty_product_dda"))),
                        ("BB", normalize_counterparty_product(flow_value(customer, "fof_counterparty_product_bb"))),
                        ("MFT", normalize_counterparty_product(flow_value(customer, "fof_counterparty_product_mft"))),
                        ("GIC", normalize_counterparty_product(flow_value(customer, "fof_counterparty_product_gic"))),
                    ],
                ),
                unsafe_allow_html=True,
            )
        else:
            product_counts = pd.Series(products).value_counts().reset_index()
            product_counts.columns = ["Product", "Count"]
            fig = px.bar(
                product_counts,
                x="Count",
                y="Product",
                orientation="h",
                color_discrete_sequence=[PRIMARY],
                text="Count",
            )
            fig.update_traces(textposition="outside", marker_line_width=0, opacity=0.86)
            fig.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(chart(fig, 300), use_container_width=True)

    with c4:
        section_title("Payment Category / Subcategory")
        payment_rows = pd.DataFrame(
            [
                {
                    "Source": "DDA",
                    "Category": safe_text(flow_value(customer, "payment_category_dda")),
                    "Subcategory": safe_text(flow_value(customer, "payment_subcategory_dda")),
                },
                {
                    "Source": "BB",
                    "Category": safe_text(flow_value(customer, "payment_category_bb")),
                    "Subcategory": safe_text(flow_value(customer, "payment_subcategory_bb")),
                },
                {
                    "Source": "MFT",
                    "Category": safe_text(flow_value(customer, "payment_category_mft")),
                    "Subcategory": safe_text(flow_value(customer, "payment_subcategory_mft")),
                },
                {
                    "Source": "GIC",
                    "Category": safe_text(flow_value(customer, "payment_category_gic")),
                    "Subcategory": safe_text(flow_value(customer, "payment_subcategory_gic")),
                },
            ]
        )
        st.dataframe(payment_rows, use_container_width=True, hide_index=True, height=260)

    section_title("Advisor Flow Insight")
    st.markdown(
        pillar_card_html(
            "Advisor Flow Insight",
            body_html=f"<div style='font-size:0.84rem;color:{DARK};line-height:1.65'>{escape(build_flow_advisor_insight(customer))}</div>",
        ),
        unsafe_allow_html=True,
    )

    dda_keys = [
        "account_id_dda", "effective_date_dda", "payee_customer_id_dda", "payee_fi_id_dda",
        "payee_customer_type_dda", "payer_customer_id_dda", "payer_customer_type_dda",
        "payer_fi_id_dda", "transaction_amount_dda", "payment_category_dda",
        "payment_subcategory_dda", "fof_category_dda", "fof_counterparty_fi_dda",
        "fof_counterparty_product_dda", "fof_subcategory_dda", "counterparty_system_dda",
        "counterparty_txn_key_dda", "total_inflow_amount_cad_dda", "total_outflow_amount_cad_dda",
        "total_internal_amount_cad_dda",
    ]
    bb_keys = [
        "account_id_bb", "effective_date_bb", "payee_customer_id_bb", "payee_fi_id_bb",
        "payee_customer_type_bb", "payer_customer_id_bb", "payer_customer_type_bb",
        "payer_fi_id_bb", "transaction_amount_bb", "payment_category_bb",
        "payment_subcategory_bb", "fof_category_bb", "fof_counterparty_fi_bb",
        "fof_counterparty_product_bb", "fof_subcategory_bb", "counterparty_system_bb",
        "counterparty_txn_key_bb", "total_inflow_amount_cad_bb", "total_outflow_amount_cad_bb",
        "total_internal_amount_cad_bb", "transaction_for_flag_bb",
    ]
    d2d_keys = [
        "flow_direction_d2d", "customer_id_d2d", "week_end_date_d2d", "financial_record_type_d2d",
        "account_id_d2d", "source_system_d2d", "filtered_amount_d2d", "flow_direction_gic_product",
        "customer_id_gic_product", "week_end_date_gic_product", "financial_record_type_gic_product",
        "account_id_gic_product", "source_system_gic_product", "filtered_amount_gic_product",
        "business_segment_gic_product", "plan_type_gic_product", "gic_product_gic_product",
        "gic_redemption_type_gic_product", "amount_filtered_gic_product", "gross_profit_gic_product",
        "final_amount_gic_product", "flow_direction_ipcash", "customer_id_ipcash", "week_end_date_ipcash",
        "financial_record_type_ipcash", "account_id_ipcash", "source_system_ipcash", "filtered_amount_ipcash",
    ]
    mft_keys = [
        "account_id_mft", "adjustment_amount_mft", "average_cost_amount_mft", "effective_date_mft",
        "gross_amount_cad_mft", "debit_credit_flag_mft", "channel_id_mft", "fee_amount_mft",
        "financial_record_type_mft", "fof_category_mft", "fof_counterparty_fi_mft",
        "fof_counterparty_product_mft", "fof_counterparty_tag_mft", "fof_product_tag_mft",
        "fof_subcategory_mft", "dealer_code_mft", "fund_account_id_mft", "initial_purchase_flag_mft",
        "interest_amount_mft", "transaction_id_mft", "investment_code_mft", "investment_account_id_mft",
        "investment_record_type_mft", "is_investment_flag_mft", "last_updated_date_mft",
        "last_updated_system_mft", "last_updated_user_mft", "transit_number_mft",
        "is_online_batch_flag_mft", "order_date_mft", "order_time_mft", "transaction_direction_mft",
        "payment_category_mft", "payment_subcategory_mft", "payee_fi_name_mft", "payee_fi_id_mft",
        "payee_company_name_mft", "payee_company_code_mft", "transaction_amount_mft",
        "transaction_date_mft", "effective_transaction_date_mft", "trade_date_mft",
        "is_first_recurring_flag_mft", "payer_customer_id_mft", "payer_account_id_mft",
        "payer_system_mft", "transaction_currency_mft", "transaction_amount_cad_mft",
        "transaction_code_mft", "transaction_description_mft", "service_channel_mft",
    ]
    gic_keys = [
        "account_id_gic", "effective_date_gic", "gross_amount_cad_gic", "counterparty_system_gic",
        "counterparty_txn_key_gic", "debit_credit_flag_gic", "channel_id_gic", "financial_record_type_gic",
        "fof_global_id_gic", "fof_category_gic", "fof_counterparty_product_gic", "fof_counterparty_tag_gic",
        "fof_product_tag_gic", "fof_subcategory_gic", "initial_purchase_flag_gic", "interest_amount_gic",
        "index_interest_amount_gic", "index_interest_rate_gic", "investment_code_gic",
        "investment_account_id_gic", "investment_record_type_gic", "is_investment_flag_gic",
        "order_date_gic", "order_time_gic", "payment_category_gic", "payment_subcategory_gic",
        "primary_owner_customer_id_gic", "transaction_amount_gic", "transaction_currency_gic",
        "transaction_time_gic", "transaction_date_gic", "transaction_id_gic", "transaction_description_gic",
        "transaction_detail_description_gic", "transaction_direction_gic", "gross_amount_gic",
        "net_amount_gic", "settlement_method_gic", "transaction_source_gic", "transaction_source_desc_gic",
        "transfer_company_code_gic", "transfer_account_id_gic", "unsolicited_order_flag_gic",
    ]

    render_detail_section(
        label="DDA Flow Details",
        key="flow_detail_dda",
        table_df=detail_table_df(customer, dda_keys),
        height=420,
    )
    render_detail_section(
        label="BB Flow Details",
        key="flow_detail_bb",
        table_df=detail_table_df(customer, bb_keys),
        height=440,
    )
    render_detail_section(
        label="D2D / GIC Product / IPCash Details",
        key="flow_detail_d2d_gic_ipcash",
        table_df=detail_table_df(customer, d2d_keys),
        height=520,
    )
    render_detail_section(
        label="Mutual Fund Flow Details",
        key="flow_detail_mft",
        table_df=detail_table_df(customer, mft_keys),
        height=620,
    )
    render_detail_section(
        label="GIC Flow Details",
        key="flow_detail_gic",
        table_df=detail_table_df(customer, gic_keys),
        height=620,
    )


def format_percent_value(value, default="Not available"):
    numeric = safe_number(value, None)
    if numeric is None:
        return default
    return f"{numeric:.2f}%"


def detail_table_from_mapping(customer, mapping):
    rows = []
    for label, field_name, formatter in mapping:
        value = customer_field(customer, field_name)
        if formatter == "currency":
            display = format_currency_value(value)
        elif formatter == "number":
            numeric = safe_number(value, None)
            display = f"{numeric:,.0f}" if numeric is not None else "Not available"
        elif formatter == "boolean":
            display = describe_bool(value)
        elif formatter == "percent":
            display = format_percent_value(value)
        else:
            display = format_display_value(value)
        rows.append({"Variable Label": label, "Value": display})
    return pd.DataFrame(rows)


def build_account_usage_insight(customer, total_spend_30d):
    annual_income = safe_number(customer.get("annual_income"), None)
    hisa_balance = safe_number(customer_field(customer, "balance_cad_1month_hisa"), 0.0) or 0.0
    gic_market_value = safe_number(customer_field(customer, "gic_market_value_cad_gic"), 0.0) or 0.0
    days_to_maturity = safe_number(customer_field(customer, "days_to_maturity_gic"), None)
    online_wallet = sum(
        safe_number(customer_field(customer, key), 0.0) or 0.0
        for key in (
            "net_online_amt_30days",
            "net_paywave_amt_30days",
            "net_apple_pay_amt_30days",
            "net_apple_total_amt_30days",
            "net_google_total_amt_30days",
            "net_samsung_total_amt_30days",
        )
    )
    monthly_fees = safe_number(customer_field(customer, "total_fee_cad_1month_hisa", "monthly_fee_cad_bb_d2d"), 0.0) or 0.0
    hisa_deposits = safe_number(customer_field(customer, "deposit_amount_cad_1month_hisa"), 0.0) or 0.0
    hisa_withdrawals = safe_number(customer_field(customer, "withdrawal_amount_cad_1month_hisa"), 0.0) or 0.0
    is_new = parse_bool(customer_field(customer, "is_new_flag_bb_d2d", "is_new_flag_hisa"))
    is_closed = parse_bool(customer_field(customer, "is_closed_flag_bb_d2d", "is_closed_flag_hisa"))

    insights = []
    if annual_income and total_spend_30d > (annual_income / 12) * 0.55:
        insights.append("30-day spend appears high relative to income; review budgeting and short-term cash flow habits.")
    if hisa_balance > 10_000 and gic_market_value < hisa_balance * 0.35:
        insights.append("HISA balances are strong relative to GIC holdings; consider a GIC or savings optimization conversation.")
    if days_to_maturity is not None and days_to_maturity <= 90:
        insights.append("GIC maturity is approaching soon; prepare a renewal or reinvestment conversation.")
    if online_wallet > max(500.0, total_spend_30d * 0.35):
        insights.append("Digital and mobile wallet usage is strong, indicating a digitally engaged client.")
    if monthly_fees > 25:
        insights.append("Monthly account fees are elevated; review account package fit and fee optimization.")
    if hisa_deposits > 0 and hisa_withdrawals < hisa_deposits * 0.4:
        insights.append("Savings deposits exceed withdrawals, suggesting room for investment or structured savings discussions.")
    if is_closed is True:
        insights.append("One or more account relationships appear closed; prioritize retention and reactivation opportunities.")
    elif is_new is True:
        insights.append("A newer account relationship is present; focus on onboarding and deepening early usage.")

    if not insights:
        insights.append("Account usage looks balanced overall; use spending, savings, and maturity signals to identify the next best conversation.")
    return " ".join(insights)


def render_account_usage_detail(customer):
    if st.button("← Back to 6 Pillars Overview", key="back_to_pillars_account_usage", type="secondary"):
        set_selected_pillar("overview")
        st.rerun()

    st.markdown(
        pillar_detail_header_html(
            "Account Usage",
            customer,
            safe_text(customer_field(customer, "effective_date_bb_d2d", "effective_date_hisa", "business_effective_date")),
        ),
        unsafe_allow_html=True,
    )

    current_bb_balance = safe_number(customer_field(customer, "balance_cad_1month_bb_d2d", "balance_cad_lmonth_bb_d2d"), 0.0) or 0.0
    avg_bb_balance = safe_number(customer_field(customer, "avg_balance_cad_1month_bb_d2d", "avg_balance_cad_lmonth_bb_d2d"), 0.0) or 0.0
    hisa_balance = safe_number(customer_field(customer, "balance_cad_1month_hisa"), 0.0) or 0.0
    hisa_deposits = safe_number(customer_field(customer, "deposit_amount_cad_1month_hisa"), 0.0) or 0.0
    hisa_withdrawals = safe_number(customer_field(customer, "withdrawal_amount_cad_1month_hisa"), 0.0) or 0.0
    gic_market_value = safe_number(customer_field(customer, "gic_market_value_cad_gic"), 0.0) or 0.0
    days_to_maturity = safe_number(customer_field(customer, "days_to_maturity_gic"), None)

    spend_categories = {
        "Grocery": "net_grocery_amt_30days",
        "Dining": "net_dining_amt_30days",
        "Online": "net_online_amt_30days",
        "PayWave": "net_paywave_amt_30days",
        "Fuel": "net_fuel_amt_30days",
        "Travel": "net_travel_amt_30days",
        "Daily Transit": "net_daily_transit_amt_30days",
        "Pharma": "net_pharma_amt_30days",
        "Health": "net_health_amt_30days",
        "Automotive": "net_automotive_amt_30days",
        "Entertainment": "net_entertainment_amt_30days",
        "TV / Streaming": "net_tv_streaming_amt_30days",
        "Professional Service": "net_professional_service_amt_30days",
        "Retail Service": "net_retail_service_amt_30days",
        "Home Improvement": "net_home_improvement_amt_30days",
        "Telecom / Utilities": "net_telecom_utilities_amt_30days",
        "Other Spend": "net_merchant_category_other_spend_amt_30days",
        "Recurring Payment": "net_recurring_payment_amt_30days",
        "Foreign": "net_foreign_amt_30days",
        "Foreign Cash Advance": "foreign_cash_advance_amt_30days",
    }
    spend_values = {
        label: abs(safe_number(customer_field(customer, field), 0.0) or 0.0)
        for label, field in spend_categories.items()
    }
    total_spend_30d = sum(spend_values.values())

    kpi_data = [
        ("Current BB/D2D Balance", format_currency_value(current_bb_balance)),
        ("Average BB/D2D Balance", format_currency_value(avg_bb_balance)),
        ("HISA Balance", format_currency_value(hisa_balance)),
        ("HISA Deposits", format_currency_value(hisa_deposits)),
        ("HISA Withdrawals", format_currency_value(hisa_withdrawals)),
        ("30-Day Total Spend/Usage", format_currency_value(total_spend_30d)),
        ("GIC Market Value", format_currency_value(gic_market_value)),
        ("Days to GIC Maturity", f"{days_to_maturity:,.0f}" if days_to_maturity is not None else "Not available"),
    ]
    kpi_cols = st.columns(4)
    for idx, (label, value) in enumerate(kpi_data):
        with kpi_cols[idx % 4]:
            st.metric(label, value)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        section_title("30-Day Spend / Usage Category Breakdown")
        spend_df = pd.DataFrame(
            [{"Category": label, "Amount": amount} for label, amount in spend_values.items() if amount > 0]
        ).sort_values("Amount", ascending=False).head(10)
        if spend_df.empty:
            st.info("No 30-day spend/usage data available.")
        else:
            fig = px.bar(
                spend_df,
                x="Amount",
                y="Category",
                orientation="h",
                color_discrete_sequence=[PRIMARY],
                text="Amount",
            )
            fig.update_traces(texttemplate="$%{x:,.0f}", textposition="outside", marker_line_width=0, opacity=0.86)
            fig.update_layout(yaxis=dict(categoryorder="total ascending"), xaxis_tickformat="$,.0f")
            st.plotly_chart(chart(fig, 330), use_container_width=True)

    with c2:
        section_title("Payment Method / Channel Usage")
        channel_values = pd.DataFrame(
            [
                {"Channel": "PayWave", "Amount": abs(safe_number(customer_field(customer, "net_paywave_amt_30days"), 0.0) or 0.0)},
                {"Channel": "Online", "Amount": abs(safe_number(customer_field(customer, "net_online_amt_30days"), 0.0) or 0.0)},
                {"Channel": "Chip & PIN", "Amount": abs(safe_number(customer_field(customer, "net_chip_pin_amt_30days"), 0.0) or 0.0)},
                {"Channel": "Magnetic Stripe", "Amount": abs(safe_number(customer_field(customer, "net_magnetic_stripe_amt_30days"), 0.0) or 0.0)},
                {"Channel": "Apple Pay", "Amount": abs(max(
                    safe_number(customer_field(customer, "net_apple_pay_amt_30days"), 0.0) or 0.0,
                    safe_number(customer_field(customer, "net_apple_total_amt_30days"), 0.0) or 0.0,
                ))},
                {"Channel": "Google Pay", "Amount": abs(safe_number(customer_field(customer, "net_google_total_amt_30days"), 0.0) or 0.0)},
                {"Channel": "Samsung Pay", "Amount": abs(safe_number(customer_field(customer, "net_samsung_total_amt_30days"), 0.0) or 0.0)},
            ]
        )
        if channel_values["Amount"].sum() <= 0:
            st.info("No payment method/channel usage data available.")
        else:
            fig = px.bar(
                channel_values,
                x="Channel",
                y="Amount",
                color="Channel",
                color_discrete_sequence=PALETTE,
                text="Amount",
            )
            fig.update_traces(texttemplate="$%{y:,.0f}", textposition="outside", marker_line_width=0, opacity=0.86)
            fig.update_layout(showlegend=False, yaxis_tickformat="$,.0f")
            st.plotly_chart(chart(fig, 330), use_container_width=True)

    c3, c4 = st.columns(2, gap="medium")
    with c3:
        section_title("Balance Movement")
        bb_previous = safe_number(customer_field(customer, "previous_balance_cad_bb_d2d"), 0.0) or 0.0
        hisa_previous = safe_number(customer_field(customer, "previous_balance_cad_hisa"), 0.0) or 0.0
        avg_hisa_balance = safe_number(customer_field(customer, "avg_balance_cad_1month_hisa"), 0.0) or 0.0
        bb_change = current_bb_balance - bb_previous
        hisa_change = hisa_balance - hisa_previous

        movement_df = pd.DataFrame(
            [
                {"Metric": "BB Previous", "Amount": bb_previous},
                {"Metric": "BB Current", "Amount": current_bb_balance},
                {"Metric": "BB Average", "Amount": avg_bb_balance},
                {"Metric": "HISA Previous", "Amount": hisa_previous},
                {"Metric": "HISA Current", "Amount": hisa_balance},
                {"Metric": "HISA Average", "Amount": avg_hisa_balance},
            ]
        )
        fig = px.bar(
            movement_df,
            x="Metric",
            y="Amount",
            color="Metric",
            color_discrete_sequence=[PRIMARY, "#ef4444", SUCCESS, "#3b82f6", "#06b6d4", WARNING],
            text="Amount",
        )
        fig.update_traces(texttemplate="$%{y:,.0f}", textposition="outside", marker_line_width=0, opacity=0.86)
        fig.update_layout(showlegend=False, yaxis_tickformat="$,.0f")
        st.plotly_chart(chart(fig, 320), use_container_width=True)
        st.markdown(
            pillar_card_html(
                "Balance Change Summary",
                [
                    ("BB Balance Change", f"{format_currency_value(bb_change)} {'↑' if bb_change > 0 else '↓' if bb_change < 0 else '→'}"),
                    ("HISA Balance Change", f"{format_currency_value(hisa_change)} {'↑' if hisa_change > 0 else '↓' if hisa_change < 0 else '→'}"),
                    ("BB Balance Change Description", safe_text(customer_field(customer, "balance_change_desc_bb_d2d"))),
                ],
            ),
            unsafe_allow_html=True,
        )

    with c4:
        section_title("HISA Activity Summary")
        hisa_activity = [
            ("Monthly Deposits", hisa_deposits),
            ("Monthly Withdrawals", hisa_withdrawals),
            ("Transaction Count", safe_number(customer_field(customer, "total_txn_count_1month_hisa"), 0.0) or 0.0),
            ("Total Fees", safe_number(customer_field(customer, "total_fee_cad_1month_hisa"), 0.0) or 0.0),
            ("Interest Accrued", safe_number(customer_field(customer, "interest_accrued_cad_1month_hisa"), 0.0) or 0.0),
            ("Dividend Amount", safe_number(customer_field(customer, "dividend_amount_cad_1month_hisa"), 0.0) or 0.0),
        ]
        activity_cards = "".join(
            f"<div style='background:{BG};border-radius:12px;padding:0.8rem 0.9rem'>"
            f"<div style='font-size:0.62rem;color:{MUTED};text-transform:uppercase;letter-spacing:0.06em;font-weight:700'>{escape(label)}</div>"
            f"<div style='font-size:1rem;font-weight:700;color:{DARK};margin-top:4px'>"
            f"{f'{int(value):,}' if label == 'Transaction Count' else format_currency_value(value)}</div></div>"
            for label, value in hisa_activity
        )
        st.markdown(
            "<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px'>" + activity_cards + "</div>",
            unsafe_allow_html=True,
        )

    section_title("GIC Product Timeline / Maturity")
    days_text = "Not available"
    maturity_flag = "Not available"
    if days_to_maturity is not None:
        days_text = f"{days_to_maturity:,.0f} days"
        if days_to_maturity <= 90:
            maturity_flag = "Maturing soon"
        elif days_to_maturity > 365:
            maturity_flag = "Longer-term holding"
        else:
            maturity_flag = "Mid-term holding"
    redeemable = describe_bool(customer_field(customer, "is_redeemable_flag_gic"), "Redeemable", "Non-redeemable")
    gic_rows = [
        ("Product Name", safe_text(customer_field(customer, "product_name_gic"))),
        ("Issuer", safe_text(customer_field(customer, "product_issuer_gic"))),
        ("Plan Type", safe_text(customer_field(customer, "plan_type_gic"))),
        ("Issue Date", safe_text(customer_field(customer, "issue_date_gic"))),
        ("Maturity Date", safe_text(customer_field(customer, "maturity_date_gic"))),
        ("Days to Maturity", days_text),
        ("Face Value", format_currency_value(customer_field(customer, "gic_face_value_cad_gic"))),
        ("Market Value", format_currency_value(customer_field(customer, "gic_market_value_cad_gic"))),
        ("Interest Rate Paid", format_percent_value(customer_field(customer, "interest_rate_paid_gic"))),
        ("Redeemability", redeemable),
        ("Channel", safe_text(customer_field(customer, "channel_gic", "channel_group_gic"))),
        ("Holding Signal", maturity_flag),
    ]
    st.markdown(pillar_card_html("GIC Product Timeline / Maturity", gic_rows), unsafe_allow_html=True)

    section_title("Advisor Account Usage Insight")
    st.markdown(
        pillar_card_html(
            "Advisor Account Usage Insight",
            body_html=f"<div style='font-size:0.84rem;color:{DARK};line-height:1.65'>{escape(build_account_usage_insight(customer, total_spend_30d))}</div>",
        ),
        unsafe_allow_html=True,
    )

    bb_detail_map = [
        ("BB/D2D Customer ID", "customer_id_bb_d2d", None),
        ("BB/D2D Account ID", "account_id_bb_d2d", None),
        ("BB/D2D Effective Date", "effective_date_bb_d2d", None),
        ("BB Average Balance, Last Month", "avg_balance_cad_lmonth_bb_d2d", "currency"),
        ("BB Balance, Last Month", "balance_cad_lmonth_bb_d2d", "currency"),
        ("BB Average Balance, Current Month", "avg_balance_cad_1month_bb_d2d", "currency"),
        ("BB Current Balance", "balance_cad_1month_bb_d2d", "currency"),
        ("BB Previous Balance", "previous_balance_cad_bb_d2d", "currency"),
        ("BB Balance Change Description", "balance_change_desc_bb_d2d", None),
        ("BB Monthly Deposits", "deposit_amount_cad_1month_bb_d2d", "currency"),
        ("BB Monthly Fee", "monthly_fee_cad_bb_d2d", "currency"),
        ("BB Months on Book", "month_on_book_bb_d2d", "number"),
        ("BB New Account Flag", "is_new_flag_bb_d2d", "boolean"),
        ("BB Closed Account Flag", "is_closed_flag_bb_d2d", "boolean"),
        ("BB Secondary Priority Customer ID", "secondary_customer_id_priority_bb_d2d", None),
        ("BB Secondary Other Customer ID", "secondary_customer_id_other_bb_d2d", None),
    ]
    spend_detail_map = [(f"{label} Spend, Last 30 Days", field, "currency") for label, field in spend_categories.items()] + [
        ("PayWave Usage, Last 30 Days", "net_paywave_amt_30days", "currency"),
        ("Chip & PIN Usage, Last 30 Days", "net_chip_pin_amt_30days", "currency"),
        ("Magnetic Stripe Usage, Last 30 Days", "net_magnetic_stripe_amt_30days", "currency"),
        ("Apple Pay Usage, Last 30 Days", "net_apple_pay_amt_30days", "currency"),
        ("Apple Total Usage, Last 30 Days", "net_apple_total_amt_30days", "currency"),
        ("Google Pay Usage, Last 30 Days", "net_google_total_amt_30days", "currency"),
        ("Samsung Pay Usage, Last 30 Days", "net_samsung_total_amt_30days", "currency"),
    ]
    hisa_detail_map = [
        ("HISA Customer ID", "customer_id_hisa", None),
        ("HISA Effective Date", "effective_date_hisa", None),
        ("HISA Account ID", "account_id_hisa", None),
        ("HISA Average Balance", "avg_balance_cad_1month_hisa", "currency"),
        ("HISA Previous Balance", "previous_balance_cad_hisa", "currency"),
        ("HISA Current Balance", "balance_cad_1month_hisa", "currency"),
        ("HISA Monthly Deposits", "deposit_amount_cad_1month_hisa", "currency"),
        ("HISA Monthly Withdrawals", "withdrawal_amount_cad_1month_hisa", "currency"),
        ("HISA Transaction Count", "total_txn_count_1month_hisa", "number"),
        ("HISA Total Amount", "total_amount_cad_1month_hisa", "currency"),
        ("HISA Total Fees", "total_fee_cad_1month_hisa", "currency"),
        ("HISA Buy Amount", "buy_amount_cad_1month_hisa", "currency"),
        ("HISA Sell Amount", "sell_amount_cad_1month_hisa", "currency"),
        ("HISA Dividend Amount", "dividend_amount_cad_1month_hisa", "currency"),
        ("HISA Interest Accrued", "interest_accrued_cad_1month_hisa", "currency"),
        ("HISA Months on Book", "month_on_book_hisa", "number"),
        ("HISA Holding ID", "holding_id_hisa", None),
        ("HISA New Account Flag", "is_new_flag_hisa", "boolean"),
        ("HISA Closed Account Flag", "is_closed_flag_hisa", "boolean"),
        ("HISA Product Switch Flag", "is_product_switch_flag_hisa", "boolean"),
        ("HISA Personal Flag", "is_personal_flag_hisa", "boolean"),
        ("HISA Senior Flag", "is_senior_flag_hisa", "boolean"),
        ("HISA Staff Flag", "is_staff_flag_hisa", "boolean"),
        ("HISA Liability Amount", "liability_amount_cad_hisa", "currency"),
        ("HISA Asset Amount", "asset_amount_cad_hisa", "currency"),
        ("HISA FX Rate", "coa_to_cad_fx_rate_hisa", None),
        ("HISA Bonus Rate", "bonus_rate_pct_hisa", "percent"),
        ("HISA Bonus Period Length", "bonus_period_length_hisa", "number"),
        ("HISA Registered Type Code", "product_registered_type_code_hisa", None),
        ("HISA Registered Type Name", "product_registered_type_name_hisa", None),
        ("HISA Financial Record Type Name", "financial_record_type_name_hisa", None),
    ]
    gic_detail_map = [
        ("GIC Channel Group", "channel_group_gic", None),
        ("GIC Currency", "currency_gic", None),
        ("GIC Term Length Days", "term_length_days_gic", "number"),
        ("GIC Term Length Months", "term_length_months_gic", "number"),
        ("GIC Term Length Years", "term_length_years_gic", "number"),
        ("GIC Certificate ID", "gic_certificate_id_gic", None),
        ("GIC Account Amount", "gic_account_amount_cad_gic", "currency"),
        ("GIC Plan Type", "plan_type_gic", None),
        ("GIC Days to Maturity", "days_to_maturity_gic", "number"),
        ("GIC Product Category", "product_category_gic", None),
        ("GIC Redeemable Flag", "is_redeemable_flag_gic", "boolean"),
        ("GIC Face Value", "gic_face_value_cad_gic", "currency"),
        ("GIC Market Value", "gic_market_value_cad_gic", "currency"),
        ("GIC Term Length Code", "gic_term_length_code_gic", None),
        ("GIC Plan Number", "plan_number_gic", None),
        ("GIC Position ID", "position_id_gic", None),
        ("GIC Holding Class Code", "product_holding_class_code_gic", None),
        ("GIC Issuer", "product_issuer_gic", None),
        ("GIC Product Name", "product_name_gic", None),
        ("GIC Issue Date", "issue_date_gic", None),
        ("GIC Maturity Date", "maturity_date_gic", None),
        ("GIC Maturity Value", "gic_maturity_value_coa_gic", "currency"),
        ("GIC Issue Value", "gic_issue_value_coa_gic", "currency"),
        ("GIC Interest Rate Paid", "interest_rate_paid_gic", "percent"),
        ("GIC Rate Approval Level", "rate_approval_level_gic", None),
        ("GIC Rate Source", "rate_source_gic", None),
        ("GIC Total Interest", "interest_total_coa_gic", "currency"),
        ("GIC Bonus Interest Rate", "interest_bonus_rate_gic", "percent"),
        ("GIC Channel", "channel_gic", None),
    ]

    render_detail_section(
        label="BB / D2D Account Usage Details",
        key="account_usage_bb_details",
        table_df=detail_table_from_mapping(customer, bb_detail_map),
        height=440,
    )
    render_detail_section(
        label="30-Day Spend and Channel Usage Details",
        key="account_usage_spend_details",
        table_df=detail_table_from_mapping(customer, spend_detail_map),
        height=620,
    )
    render_detail_section(
        label="HISA Account Usage Details",
        key="account_usage_hisa_details",
        table_df=detail_table_from_mapping(customer, hisa_detail_map),
        height=640,
    )
    render_detail_section(
        label="GIC Product Holding Details",
        key="account_usage_gic_details",
        table_df=detail_table_from_mapping(customer, gic_detail_map),
        height=620,
    )


def parse_date_value(value):
    if is_missing(value):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


def format_percent_display(value, default="Not available"):
    numeric = safe_number(value, None)
    if numeric is None:
        return default
    return f"{numeric:.0f}%"


def goal_status_badge(status):
    text = safe_text(status, "Unknown")
    lowered = text.lower()
    tone = "neutral"
    if "complete" in lowered:
        tone = "success"
    elif "progress" in lowered or "active" in lowered:
        tone = "warning"
    elif "pause" in lowered or "cancel" in lowered:
        tone = "danger"
    return badge_html(text, tone)


def build_goal_records(customer):
    records = []
    raw_goals = customer.get("financial_goals", [])
    if isinstance(raw_goals, list):
        for idx, goal in enumerate(raw_goals, start=1):
            if not isinstance(goal, dict):
                continue
            records.append(
                {
                    "goal_id": goal.get("goal_id") or f"{customer['id']}-goal-{idx}",
                    "goal_sequence": goal.get("goal_sequence", idx),
                    "goal_purpose": goal.get("purpose") or goal.get("goal_purpose") or customer_field(customer, "goal_purpose"),
                    "target_savings_amount": safe_number(goal.get("target_amount", goal.get("target_savings_amount")), None),
                    "target_spending_amount": safe_number(goal.get("target_spending_amount", customer_field(customer, "target_spending_amount")), None),
                    "target_date": goal.get("target_date") or customer_field(customer, "target_date"),
                    "retirement_age": goal.get("retirement_age", customer_field(customer, "retirement_age")),
                    "monthly_contribution": safe_number(goal.get("monthly_contribution", customer_field(customer, "monthly_contribution")), None),
                    "completion_status": goal.get("status") or goal.get("completion_status") or customer_field(customer, "completion_status"),
                    "completion_date": goal.get("completion_date") or customer_field(customer, "completion_date"),
                    "goal_progress_pct": safe_number(goal.get("goal_progress_pct"), None),
                    "projected_savings_amount": safe_number(goal.get("projected_savings_amount", customer_field(customer, "projected_savings_amount")), None),
                    "projected_spending_amount": safe_number(goal.get("projected_spending_amount", customer_field(customer, "projected_spending_amount")), None),
                    "projection_status": goal.get("projection_status") or customer_field(customer, "projection_status"),
                    "projection_date": goal.get("projection_date") or customer_field(customer, "projection_date"),
                    "completion_amount": safe_number(goal.get("completion_amount", customer_field(customer, "completion_amount")), None),
                }
            )

    if not records:
        records.append(
            {
                "goal_id": customer_field(customer, "goal_id", default=f"{customer['id']}-goal-1"),
                "goal_sequence": customer_field(customer, "goal_sequence", default=1),
                "goal_purpose": customer_field(customer, "goal_purpose"),
                "target_savings_amount": safe_number(customer_field(customer, "target_savings_amount"), None),
                "target_spending_amount": safe_number(customer_field(customer, "target_spending_amount"), None),
                "target_date": customer_field(customer, "target_date"),
                "retirement_age": customer_field(customer, "retirement_age"),
                "monthly_contribution": safe_number(customer_field(customer, "monthly_contribution"), None),
                "completion_status": customer_field(customer, "completion_status"),
                "completion_date": customer_field(customer, "completion_date"),
                "goal_progress_pct": safe_number(customer_field(customer, "goal_progress_pct"), None),
                "projected_savings_amount": safe_number(customer_field(customer, "projected_savings_amount"), None),
                "projected_spending_amount": safe_number(customer_field(customer, "projected_spending_amount"), None),
                "projection_status": customer_field(customer, "projection_status"),
                "projection_date": customer_field(customer, "projection_date"),
                "completion_amount": safe_number(customer_field(customer, "completion_amount"), None),
            }
        )

    for goal in records:
        if goal["goal_progress_pct"] is None:
            target = goal.get("target_savings_amount")
            completed = goal.get("completion_amount")
            projected = goal.get("projected_savings_amount")
            if target and completed is not None and target > 0:
                goal["goal_progress_pct"] = clamp(completed / target * 100, 0, 100)
            elif target and projected is not None and target > 0:
                goal["goal_progress_pct"] = clamp(projected / target * 100, 0, 100)
    return records


def build_financial_goal_insight(customer, goals):
    goal_count = int(safe_number(customer_field(customer, "financial_goal_count"), len(goals)) or 0)
    completed = int(safe_number(customer_field(customer, "completed_goal_count"), 0) or 0)
    incomplete = int(safe_number(customer_field(customer, "incomplete_goal_count"), 0) or 0)
    projected_savings = safe_number(customer_field(customer, "projected_savings_amount"), None)
    target_savings = safe_number(customer_field(customer, "target_savings_amount"), None)
    monthly_contribution = safe_number(customer_field(customer, "monthly_contribution"), None)
    target_date = parse_date_value(customer_field(customer, "target_date"))
    completion_status = safe_text(customer_field(customer, "completion_status"), "Unknown")
    goal_purpose_text = " ".join(safe_text(goal.get("goal_purpose"), "").lower() for goal in goals)

    insights = []
    if goal_count == 0:
        insights.append("No financial goals are currently recorded. Consider starting a goal planning conversation.")
    if incomplete > 0:
        insights.append("Client has active or incomplete goals. Review contribution progress and target timeline.")
    if completed > 0:
        insights.append("Client has completed goals. Consider discussing next-stage planning or new goals.")
    if target_savings and projected_savings is not None and projected_savings < target_savings:
        insights.append("Projected savings are below target. Consider increasing monthly contribution or extending target date.")
    if not monthly_contribution:
        insights.append("No monthly contribution is recorded. Consider setting up automatic savings.")
    if "retirement" in goal_purpose_text:
        insights.append("Retirement planning goal detected. Review retirement age, contribution level, and registered account opportunities.")
    if "home" in goal_purpose_text:
        insights.append("Home purchase goal detected. Consider FHSA, savings plan, and mortgage readiness conversation.")
    if target_date and (target_date - date.today()).days <= 183 and "complete" not in completion_status.lower():
        insights.append("Goal deadline is approaching. Review progress and action plan.")

    if not insights:
        insights.append("Goal profile is stable overall. Use the target, contribution, and completion signals to guide the next planning conversation.")
    return " ".join(insights)


def render_financial_goal_detail(customer):
    if st.button("← Back to 6 Pillars Overview", key="back_to_pillars_financial_goal", type="secondary"):
        set_selected_pillar("overview")
        st.rerun()

    annual_income = safe_number(customer_field(customer, "annual_income"), None)
    subtitle_date = safe_text(customer_field(customer, "target_date"))
    st.markdown(
        pillar_detail_header_html(
            "Financial Goal",
            customer,
            subtitle_date if subtitle_date != "Not available" else None,
        ),
        unsafe_allow_html=True,
    )
    if annual_income is not None:
        st.markdown(
            f"<div style='margin-top:-0.6rem;margin-bottom:1rem;font-size:0.82rem;color:{MUTED}'>Annual income: {format_currency_value(annual_income)}</div>",
            unsafe_allow_html=True,
        )

    goals = build_goal_records(customer)
    total_goals = int(safe_number(customer_field(customer, "financial_goal_count"), len(goals)) or 0)
    completed_goals = int(safe_number(customer_field(customer, "completed_goal_count"), 0) or 0)
    incomplete_goals = int(safe_number(customer_field(customer, "incomplete_goal_count"), max(total_goals - completed_goals, 0)) or 0)
    monthly_contribution = safe_number(customer_field(customer, "monthly_contribution"), None)
    target_savings = safe_number(customer_field(customer, "target_savings_amount"), None)
    target_spending = safe_number(customer_field(customer, "target_spending_amount"), None)
    projected_savings = safe_number(customer_field(customer, "projected_savings_amount"), None)
    projected_spending = safe_number(customer_field(customer, "projected_spending_amount"), None)
    completion_amount = safe_number(customer_field(customer, "completion_amount"), None)

    kpi_items = [
        ("Total Goals", f"{total_goals:,}"),
        ("Completed Goals", f"{completed_goals:,}"),
        ("Incomplete Goals", f"{incomplete_goals:,}"),
        ("Monthly Contribution", format_currency_value(monthly_contribution)),
        ("Target Savings Amount", format_currency_value(target_savings)),
        ("Target Spending Amount", format_currency_value(target_spending)),
        ("Projected Savings Amount", format_currency_value(projected_savings)),
        ("Projected Spending Amount", format_currency_value(projected_spending)),
        ("Completion Amount", format_currency_value(completion_amount)),
    ]
    kpi_cols = st.columns(3)
    for idx, (label, value) in enumerate(kpi_items):
        with kpi_cols[idx % 3]:
            st.metric(label, value)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        section_title("Goal Completion Status")
        status_df = pd.DataFrame(
            [
                {"Status": "Completed", "Count": completed_goals},
                {"Status": "Incomplete", "Count": incomplete_goals},
            ]
        )
        if status_df["Count"].sum() > 0:
            fig = px.pie(
                status_df,
                names="Status",
                values="Count",
                hole=0.56,
                color="Status",
                color_discrete_map={"Completed": SUCCESS, "Incomplete": PRIMARY},
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(chart(fig, 300), use_container_width=True)
        else:
            st.info("No completion status data available.")

        if total_goals <= 1:
            st.markdown(
                pillar_card_html(
                    "Single Goal Status",
                    body_html=goal_status_badge(customer_field(customer, "completion_status")),
                ),
                unsafe_allow_html=True,
            )

    with c2:
        section_title("Target vs Projected vs Completed Amount")
        compare_rows = []
        if target_savings is not None:
            compare_rows.append({"Series": "Target Savings", "Amount": target_savings})
        if projected_savings is not None:
            compare_rows.append({"Series": "Projected Savings", "Amount": projected_savings})
        if completion_amount is not None:
            compare_rows.append({"Series": "Completion Amount", "Amount": completion_amount})
        if target_spending is not None:
            compare_rows.append({"Series": "Target Spending", "Amount": target_spending})
        if projected_spending is not None:
            compare_rows.append({"Series": "Projected Spending", "Amount": projected_spending})
        if compare_rows:
            compare_df = pd.DataFrame(compare_rows)
            fig = px.bar(
                compare_df,
                x="Series",
                y="Amount",
                color="Series",
                color_discrete_sequence=[PRIMARY, SUCCESS, WARNING, "#3b82f6", "#06b6d4"],
                text="Amount",
            )
            fig.update_traces(texttemplate="$%{y:,.0f}", textposition="outside", marker_line_width=0, opacity=0.86)
            fig.update_layout(showlegend=False, yaxis_tickformat="$,.0f")
            st.plotly_chart(chart(fig, 300), use_container_width=True)
        else:
            st.info("No target/projection amount data available.")

    c3, c4 = st.columns(2, gap="medium")
    with c3:
        section_title("Goal Timeline")
        projection_date = parse_date_value(customer_field(customer, "projection_date"))
        target_date = parse_date_value(customer_field(customer, "target_date"))
        completion_date = parse_date_value(customer_field(customer, "completion_date"))
        completion_status = safe_text(customer_field(customer, "completion_status"))
        timeline_rows = [
            ("Projection Date", projection_date.isoformat() if projection_date else "Not available"),
            ("Target Date", target_date.isoformat() if target_date else "Not available"),
            ("Completion Date", completion_date.isoformat() if completion_date else "Not available"),
        ]
        timeline_notes = []
        if target_date:
            remaining = (target_date - date.today()).days
            if completion_date:
                timeline_notes.append(f"Completed on {completion_date.isoformat()}.")
            elif remaining >= 0:
                timeline_notes.append(f"{remaining} day(s) remaining until target date.")
            elif "complete" not in completion_status.lower():
                timeline_notes.append("Past target date — review required.")
        st.markdown(
            pillar_card_html(
                "Goal Timeline",
                rows=timeline_rows,
                body_html=f"<div style='padding-top:10px;font-size:0.84rem;color:{DARK};line-height:1.6'>{escape(' '.join(timeline_notes) or 'Timeline information is limited for this goal.')}</div>",
            ),
            unsafe_allow_html=True,
        )

    with c4:
        section_title("Goal Purpose Breakdown")
        if len(goals) > 1:
            purpose_rows = []
            for goal in goals:
                purpose_rows.append(
                    {
                        "Goal Purpose": safe_text(goal.get("goal_purpose")),
                        "Target Savings": safe_number(goal.get("target_savings_amount"), 0.0) or 0.0,
                        "Monthly Contribution": safe_number(goal.get("monthly_contribution"), 0.0) or 0.0,
                        "Progress %": safe_number(goal.get("goal_progress_pct"), 0.0) or 0.0,
                        "Completion Status": safe_text(goal.get("completion_status")),
                    }
                )
            purpose_df = pd.DataFrame(purpose_rows)
            fig = px.bar(
                purpose_df,
                x="Goal Purpose",
                y="Target Savings",
                color="Completion Status",
                text="Progress %",
                color_discrete_sequence=PALETTE,
            )
            fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside", marker_line_width=0, opacity=0.86)
            fig.update_layout(yaxis_tickformat="$,.0f")
            st.plotly_chart(chart(fig, 300), use_container_width=True)
        else:
            goal = goals[0]
            st.markdown(
                pillar_card_html(
                    "Goal Purpose",
                    [
                        ("Goal Purpose", safe_text(goal.get("goal_purpose"))),
                        ("Completion Status", safe_text(goal.get("completion_status"))),
                        ("Target Savings Amount", format_currency_value(goal.get("target_savings_amount"))),
                        ("Monthly Contribution", format_currency_value(goal.get("monthly_contribution"))),
                        ("Progress", format_percent_display(goal.get("goal_progress_pct"))),
                    ],
                ),
                unsafe_allow_html=True,
            )

    section_title("Goal Detail Cards")
    goal_cards = []
    for idx, goal in enumerate(goals, start=1):
        card_body = (
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px'>"
            f"<div style='font-size:0.92rem;font-weight:700;color:{DARK}'>{escape(safe_text(goal.get('goal_purpose'), f'Goal {idx}'))}</div>"
            f"{goal_status_badge(goal.get('completion_status'))}</div>"
            f"{pillar_row_html('Target Savings Amount', format_currency_value(goal.get('target_savings_amount')))}"
            f"{pillar_row_html('Target Spending Amount', format_currency_value(goal.get('target_spending_amount')))}"
            f"{pillar_row_html('Monthly Contribution', format_currency_value(goal.get('monthly_contribution')))}"
            f"{pillar_row_html('Target Date', safe_text(goal.get('target_date')))}"
            f"{pillar_row_html('Completion Date', safe_text(goal.get('completion_date')))}"
            f"{pillar_row_html('Progress', format_percent_display(goal.get('goal_progress_pct')))}"
        )
        goal_cards.append(
            f"<div style='background:{SURFACE};border:1px solid {BORDER};border-radius:16px;padding:1rem 1.15rem;"
            f"box-shadow:0 1px 2px rgba(0,0,0,0.04),0 4px 14px rgba(0,0,0,0.03);height:100%'>{card_body}</div>"
        )
    st.markdown(
        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px'>"
        + "".join(goal_cards) + "</div>",
        unsafe_allow_html=True,
    )

    section_title("Advisor Financial Goal Insight")
    st.markdown(
        pillar_card_html(
            "Advisor Financial Goal Insight",
            body_html=f"<div style='font-size:0.84rem;color:{DARK};line-height:1.65'>{escape(build_financial_goal_insight(customer, goals))}</div>",
        ),
        unsafe_allow_html=True,
    )

    goal_summary_map = [
        ("Goal Purpose", "goal_purpose", None),
        ("Financial Goal Count", "financial_goal_count", "number"),
        ("Completed Goal Count", "completed_goal_count", "number"),
        ("Incomplete Goal Count", "incomplete_goal_count", "number"),
        ("Financial Goals Summary", "financial_goals_summary", None),
        ("Financial Goals Completed", "financial_goals_completed", None),
        ("Financial Goals Incomplete", "financial_goals_incomplete", None),
        ("Product Code", "product_code", None),
        ("Retirement Age", "retirement_age", "number"),
    ]
    target_projection_map = [
        ("Target Savings Amount", "target_savings_amount", "currency"),
        ("Target Spending Amount", "target_spending_amount", "currency"),
        ("Projected Savings Amount", "projected_savings_amount", "currency"),
        ("Projected Spending Amount", "projected_spending_amount", "currency"),
        ("Projection Status", "projection_status", None),
        ("Projection Date", "projection_date", None),
        ("Monthly Contribution", "monthly_contribution", "currency"),
        ("Target Date", "target_date", None),
    ]
    completion_map = [
        ("Completion Status", "completion_status", None),
        ("Completion Date", "completion_date", None),
        ("Completion Amount", "completion_amount", "currency"),
        ("Goal Progress %", "goal_progress_pct", "percent"),
    ]
    raw_field_map = [
        ("Goal Purpose", "goal_purpose", None),
        ("Target Savings Amount", "target_savings_amount", "currency"),
        ("Target Spending Amount", "target_spending_amount", "currency"),
        ("Projected Savings Amount", "projected_savings_amount", "currency"),
        ("Projected Spending Amount", "projected_spending_amount", "currency"),
        ("Projection Status", "projection_status", None),
        ("Projection Date", "projection_date", None),
        ("Completion Status", "completion_status", None),
        ("Completion Date", "completion_date", None),
        ("Completion Amount", "completion_amount", "currency"),
        ("Financial Goal Count", "financial_goal_count", "number"),
        ("Completed Goal Count", "completed_goal_count", "number"),
        ("Incomplete Goal Count", "incomplete_goal_count", "number"),
        ("Financial Goals Summary", "financial_goals_summary", None),
        ("Financial Goals Completed", "financial_goals_completed", None),
        ("Financial Goals Incomplete", "financial_goals_incomplete", None),
    ]

    render_detail_section(
        label="Goal Summary Details",
        key="financial_goal_summary_details",
        table_df=detail_table_from_mapping(customer, goal_summary_map),
        height=360,
    )
    render_detail_section(
        label="Target and Projection Details",
        key="financial_goal_target_projection_details",
        table_df=detail_table_from_mapping(customer, target_projection_map),
        height=360,
    )
    render_detail_section(
        label="Completion Details",
        key="financial_goal_completion_details",
        table_df=detail_table_from_mapping(customer, completion_map),
        height=300,
    )
    render_detail_section(
        label="Financial Goal Raw Fields",
        key="financial_goal_raw_fields",
        table_df=detail_table_from_mapping(customer, raw_field_map),
        height=520,
    )


def render_offer_detail(customer):
    # Render the Offer pillar using offer.json-backed matching instead of placeholder content.
    if st.button("← Back to 6 Pillars Overview", key="back_to_pillars_offer", type="secondary"):
        set_selected_pillar("overview")
        st.rerun()

    st.markdown(pillar_detail_header_html("Offer", customer), unsafe_allow_html=True)

    matched_offers, catalog_warning, inferred_signals = get_recommended_offers(customer, top_n=4)
    if catalog_warning:
        st.warning(catalog_warning)

    if not matched_offers:
        st.info("No clear offer was detected for this client from the current offer catalog.")
        return

    if max(offer.get("match_score", 0) for offer in matched_offers) < 2:
        st.info("No strong signal match was detected, so the offers below are broader relevant benefits for advisor review.")

    st.markdown(
        pillar_card_html(
            "Offer Matching Summary",
            rows=[
                ("Detected Client Signals", f"{len(inferred_signals):,}"),
                ("Recommended Offers Shown", f"{len(matched_offers):,}"),
                ("Top Match Score", str(max(offer.get("match_score", 0) for offer in matched_offers))),
            ],
            body_html=(
                f"<div style='padding-top:10px;font-size:0.84rem;color:{DARK};line-height:1.65'>"
                f"The offer recommendations below are matched from <code>{escape(OFFER_JSON_PATH)}</code> using inferred client signals from the selected customer's profile, balances, activity, goals, and product indicators."
                f"</div>"
            ),
        ),
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    card_cols = st.columns(2, gap="medium")
    for idx, offer in enumerate(matched_offers):
        reasons = offer.get("matched_reasons") or [
            "No direct signal match was detected; this benefit is shown as a broader relevant offer for review."
        ]
        reason_items = "".join(
            f"<li style='margin-bottom:6px'>{escape(reason)}</li>"
            for reason in reasons[:4]
        )
        body_html = (
            f"<div style='font-size:0.84rem;color:{DARK};line-height:1.65'>"
            f"<div style='font-weight:700;margin-bottom:4px'>Customer Benefit</div>"
            f"<div style='margin-bottom:12px'>{escape(safe_text(offer.get('customer_benefit')))}</div>"
            f"<div style='font-weight:700;margin-bottom:4px'>Why This Client May Be Relevant</div>"
            f"<ul style='margin:0 0 12px 18px;padding:0'>{reason_items}</ul>"
            f"<div style='font-weight:700;margin-bottom:4px'>Advisor / Model Use</div>"
            f"<div style='margin-bottom:12px'>{escape(safe_text(offer.get('model_use')))}</div>"
            f"<div style='font-size:0.76rem;color:{MUTED};padding-top:10px;border-top:1px solid {BORDER}'>"
            f"Advisor should verify current eligibility, fees, rates, and suitability before presenting to the client."
            f"</div></div>"
        )
        rows = [
            ("Product Category", safe_text(offer.get("product_category"))),
            ("Match Score", str(offer.get("match_score", 0))),
            ("Source Document", safe_text(offer.get("source_doc"))),
            ("Source Page", safe_text(offer.get("source_page"))),
        ]
        with card_cols[idx % 2]:
            st.markdown(
                pillar_card_html(safe_text(offer.get("product_name")), rows=rows, body_html=body_html),
                unsafe_allow_html=True,
            )


def render_client_profile_tab_nav():
    active_tab = st.session_state.get("active_client_tab", "Overview")
    cols = st.columns(len(CLIENT_PROFILE_TABS), gap="small")
    for idx, tab_name in enumerate(CLIENT_PROFILE_TABS):
        with cols[idx]:
            if st.button(
                tab_name,
                key=f"client_profile_tab_{tab_name}",
                type="primary" if tab_name == active_tab else "secondary",
                use_container_width=True,
            ):
                set_client_profile_tab(tab_name)
                st.rerun()


ensure_navigation_state()

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

    nav_label = st.radio(
        "nav",
        options=VIEW_OPTIONS,
        key=MAIN_NAV_WIDGET_KEY,
        on_change=on_main_nav_change,
        label_visibility="collapsed",
    )
    view = st.session_state.get(CURRENT_PAGE_KEY, VIEW_MAP[nav_label])

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
        labels = list(options_map.keys())
        current_customer_id = st.session_state.get("selected_customer_id")
        if current_customer_id not in options_map.values():
            current_customer_id = options_map[labels[0]]
            st.session_state["selected_customer_id"] = current_customer_id
        current_index = next(
            (idx for idx, label in enumerate(labels) if options_map[label] == current_customer_id),
            0,
        )
        sel_label = st.selectbox(
            "client",
            labels,
            index=current_index,
            key="selected_client_label",
            label_visibility="collapsed",
        )
        sel_id = options_map[sel_label]
        st.session_state["selected_customer_id"] = sel_id
    else:
        st.markdown(
            f"<div style='font-size:0.75rem;color:#475569;padding:6px 2px'>"
            f"No clients found for <b style='color:#94a3b8'>{search_query}</b></div>",
            unsafe_allow_html=True,
        )
        sel_id = sorted(cust_index.keys())[0]
        st.session_state["selected_customer_id"] = sel_id

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
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)

OR_MODEL      = "anthropic/claude-opus-4-5"
OR_FAST_MODEL = "openai/gpt-4o-mini"
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

PALETTE = [PRIMARY, "#3b82f6", SUCCESS, WARNING, "#8b5cf6", "#06b6d4"]

def chart(fig, h=360):
    fig.update_layout(
        autosize=True,
        height=h,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="Inter, -apple-system, sans-serif", size=12, color="#374151"),
        margin=dict(l=28, r=20, t=36, b=28),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=10),
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="center",
            x=0.5,
        ),
    )
    fig.update_xaxes(
        gridcolor="#f1f5f9", showline=False,
        tickfont=dict(size=10, color=MUTED), zeroline=False,
    )
    fig.update_yaxes(
        gridcolor="#f1f5f9", showline=False,
        tickfont=dict(size=10, color=MUTED), zeroline=False,
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
        fig.update_traces(marker_line_width=0, opacity=0.8)
        fig.add_vline(x=750, line_dash="dash", line_color=SUCCESS, line_width=1.5,
                      annotation_text="Excellent", annotation_font_size=10)
        fig.add_vline(x=670, line_dash="dash", line_color=WARNING, line_width=1.5,
                      annotation_text="Fair", annotation_font_size=10)
        st.plotly_chart(chart(fig, 320), use_container_width=True)

    # Row 2

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
            key="portfolio_client_directory_risk_filter",
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
            key="portfolio_client_directory_employment_filter",
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
    render_client_profile_tab_nav()
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    active_client_tab = st.session_state.get("active_client_tab", "Overview")

    # ── OVERVIEW ──────────────────────────────────────────────────────────────
    if active_client_tab == "Overview":
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

            section_title("Credit Risk Flags")
            flags = [
                ("Late Payments",            str(c["num_late_payments"]),           c["num_late_payments"] > 0),
                ("Months Since Delinquency", str(c.get("months_since_last_delinquency") or "—"), False),
                ("Open Accounts",            str(c["num_open_accounts"]),            False),
                ("Bankruptcy",               "Yes ⚠️" if c["bankruptcy_history"] else "No ✅", c["bankruptcy_history"]),
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
    elif active_client_tab == "Transactions":
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
                fig.update_layout(
                    margin=dict(l=24, r=16, t=20, b=44),
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.2,
                        xanchor="center",
                        x=0.5,
                        font=dict(size=10),
                    ),
                )
                st.plotly_chart(chart(fig, 280), use_container_width=True)

            with c2:
                section_title("Spending by Category")
                exp = txn_df[txn_df["amount"] < 0].copy()
                exp["amount"] = exp["amount"].abs()
                cs = exp.groupby("category")["amount"].sum().reset_index()
                fig = px.pie(cs, names="category", values="amount", hole=0.48,
                             color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_traces(textposition="inside", textinfo="percent",
                                  marker=dict(line=dict(color=SURFACE, width=1.5)))
                fig.update_layout(
                    margin=dict(l=16, r=16, t=20, b=54),
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.22,
                        xanchor="center",
                        x=0.5,
                        font=dict(size=9),
                    ),
                )
                st.plotly_chart(chart(fig, 280), use_container_width=True)

            section_title("Recent Transactions")
            disp = txn_df.head(50).copy()
            disp["Date"]   = disp["date"].dt.strftime("%Y-%m-%d")
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
    elif active_client_tab == "Loans":
        loans  = c.get("loans", [])
        active = [l for l in loans if l["status"] == "Active"]
        closed = [l for l in loans if l["status"] == "Closed"]

        if active:
            section_title(f"Active Loans ({len(active)})")
            section_title(f"Active Loans ({len(active)})")
            for l in active:
                st.markdown(loan_card_html(l), unsafe_allow_html=True)
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
    elif active_client_tab == "Analytics":
        def clamp(v, lo, hi): return max(lo, min(hi, v))
        def norm(v, lo, hi, inv=False):
            n = (v - lo) / (hi - lo) * 100
            return clamp(100 - n if inv else n, 0, 100)

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            section_title("Financial Health Radar")
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
                ("Credit Score",  c["credit_score"],             int(peers.credit_score.mean()),  None),
                ("Annual Income", c["annual_income"],            int(peers.annual_income.mean()), "${:,}"),
                ("Total Debt",    c["total_debt"],               int(peers.total_debt.mean()),    "${:,}"),
                ("DTI Ratio",     c["debt_to_income_ratio"],     float(peers.dti.mean()),         "{:.1%}"),
            ]
            rows = []
            rows = []
            for label, cval, pval, fmt in compare_items:
                cv = fmt.format(cval) if fmt else str(cval)
                pv = fmt.format(pval) if fmt else str(pval)
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

    # ── 6 PILLARS ────────────────────────────────────────────────────────────
    elif active_client_tab == "6 Pillars":
        pillar_route = get_pillar_route()

        if pillar_route == "overview":
            section_title("6 Pillars Overview")

            flow = infer_flow_metrics(c)
            usage = infer_account_usage(c)
            goals = infer_goal_summary(c)
            matched_offers, _, _ = get_recommended_offers(c, top_n=1)
            offers = build_offer_suggestions(c)
            summary = build_advisor_summary(c)
            lead_offer = matched_offers[0] if matched_offers else None

            overview_cards = [
                pillar_link_card_html(
                    "BNS Relationships",
                    "bns-relationships",
                    "Pillar 1",
                    "Primacy, relationship depth, product holdings, and advisor context.",
                    [
                        ("Primacy Status", infer_primacy_flag(c)),
                        ("Primary Segment", safe_text(customer_field(c, "primary_segment"))),
                        ("Steps Away", infer_primacy_steps_away(c)),
                    ],
                ),
                pillar_link_card_html(
                    "Flow of Fund",
                    "flow-of-fund",
                    "Pillar 2",
                    "Movement of funds across inflows, outflows, and counterparties.",
                    [
                        ("Inflow", flow["inflow"]),
                        ("Outflow", flow["outflow"]),
                        ("Counterparty", flow["product"]),
                    ],
                ),
                pillar_link_card_html(
                    "Account Usage",
                    "account-usage",
                    "Pillar 3",
                    "Usage signals across deposits, digital engagement, and activity.",
                    [
                        ("Active Chequing", usage["chequing"]),
                        ("Active Savings", usage["savings"]),
                        ("Digital", usage["digital"]),
                    ],
                ),
                pillar_link_card_html(
                    "Financial Goal",
                    "client-goal",
                    "Pillar 4",
                    "Goal counts, progress, and advisory planning context.",
                    [
                        ("Goals", goals["goal_count"]),
                        ("Completed", goals["completed"]),
                        ("Incomplete", goals["incomplete"]),
                    ],
                ),
                pillar_link_card_html(
                    "Offer",
                    "offer",
                    "Pillar 5",
                    "Next-best-product and activation opportunities for the advisor.",
                    [
                        ("Lead Offer", safe_text(lead_offer.get("product_name")) if lead_offer else offers[0]),
                        ("Category", safe_text(lead_offer.get("product_category")) if lead_offer else "General"),
                        ("Signals Matched", str(lead_offer.get("match_score", 0)) if lead_offer else "0"),
                    ],
                ),
                pillar_link_card_html(
                    "Summary",
                    "summary",
                    "Pillar 6",
                    "Advisor-ready synthesis of relationship, usage, and opportunity signals.",
                    [
                        ("Risk Tier", c["risk_tier"]),
                        ("Primary Segment", safe_text(customer_field(c, "primary_segment"))),
                        ("Advisor", safe_text(customer_field(c, "note_advisor_name", "advisor_name"))),
                    ],
                ),
            ]

            st.markdown(
                "<div class='pillar-grid'>" + "".join(overview_cards) + "</div>",
                unsafe_allow_html=True,
            )
        elif pillar_route == "bns-relationships":
            render_bns_relationships_detail(c)
        elif pillar_route == "flow-of-fund":
            render_flow_of_fund_detail(c)
        elif pillar_route == "account-usage":
            render_account_usage_detail(c)
        elif pillar_route == "client-goal":
            render_financial_goal_detail(c)
        elif pillar_route == "offer":
            # Offer pillar now renders catalog-backed recommendations from data/offer.json.
            render_offer_detail(c)
        elif pillar_route == "summary":
            render_placeholder_pillar_detail(c, "Summary Detail")

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

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([0.95, 1.25], gap="large")

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

        btn_cols = st.columns(3, gap="small")
        for idx, lt in enumerate(LOAN_TYPES):
            meta = LOAN_META[lt]
            is_active = st.session_state.loan_type_sel == lt
            if btn_cols[idx % 3].button(
                f"{meta['icon']} {lt}",
                key=f"loan_type_{lt.lower().replace(' ', '_')}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state.loan_type_sel = lt
                st.rerun()

        loan_type = st.session_state.loan_type_sel
        meta      = LOAN_META[loan_type]

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

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
                f"padding:1.25rem 1.35rem;box-shadow:0 1px 3px rgba(0,0,0,0.04);"
                f"min-height:260px;width:100%;max-width:100%;box-sizing:border-box;overflow:hidden'>",
                unsafe_allow_html=True,
            )
            st.markdown(text)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"""<div style='background:{SURFACE};border:2px dashed {BORDER};border-radius:18px;
                                padding:2.5rem 1.5rem;text-align:center;min-height:260px;
                                width:100%;max-width:100%;box-sizing:border-box;overflow:hidden;
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
    c = sel
    customer_seed = stable_seed_from_customer_id(c.get("id") or c.get("customer_id"))
    page_header("Forecasting & Analytics", "Client projections and portfolio trends")
    tabs_f = st.tabs(["Credit Score", "Income Projection", "Risk Evolution", "Portfolio Health"])

    with tabs_f[0]:
        section_title(f"Credit Score Forecast — {c['name']} (24 months)")
        np.random.seed(customer_seed)
        imp = {"High": 0.35, "Medium": 0.18, "Low": 0.05}[c["risk_tier"]]
        months = list(range(25))
        noise = np.random.normal(0, 3, 25)
        scores = [min(850, c["credit_score"] + imp * m * 5 + noise[m]) for m in months]
        upper = [min(850, s + 18) for s in scores]
        lower = [max(300, s - 18) for s in scores]

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
        fig.update_layout(
            xaxis_title="Months from Today",
            yaxis_title="Credit Score",
            yaxis=dict(range=[300, 870]),
        )
        st.plotly_chart(chart(fig, 380), use_container_width=True)

        n1, n2, n3 = st.columns(3)
        n1.metric("Current", int(scores[0]))
        n2.metric("12-Month", int(scores[12]), delta=f"{int(scores[12] - scores[0]):+d}")
        n3.metric("24-Month", int(scores[24]), delta=f"{int(scores[24] - scores[0]):+d}")

    with tabs_f[1]:
        section_title("Income & Savings Projection — 5 Years")
        growth = {"Full-Time": 0.04, "Self-Employed": 0.06, "Part-Time": 0.02,
                  "Unemployed": 0.0, "Retired": 0.01}.get(c["employment_status"], 0.03)
        np.random.seed(customer_seed + 1)
        yrs = list(range(6))
        income = [
            c["annual_income"] * (1 + growth) ** y
            + np.random.normal(0, c["annual_income"] * 0.015)
            for y in yrs
        ]
        expense = [c["monthly_expenses"] * 12 * (1.025 ** y) for y in yrs]
        net = [i - e for i, e in zip(income, expense)]
        labels = [f"Year {y}" for y in yrs]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Income", x=labels, y=income,
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
        section_title(f"Risk Score Evolution — {c['name']} (8 Quarters)")
        np.random.seed(customer_seed + 10)
        qtrs = [f"Q{(i % 4) + 1} '2{'5' if i < 4 else '6'}" for i in range(9)]

        base_improve = {"High": 1.8, "Medium": 0.9, "Low": 0.3}[c["risk_tier"]]
        if c["bankruptcy_history"]:
            base_improve *= 0.4
        if c["num_late_payments"] > 2:
            base_improve *= 0.6

        noise = np.random.normal(0, 0.6, 9)
        risk_scores = [min(100, max(0, c["risk_score"] + base_improve * i + noise[i])) for i in range(9)]
        tier_color = RISK_COLORS[c["risk_tier"]]
        upper = [min(100, s + 4) for s in risk_scores]
        lower = [max(0, s - 4) for s in risk_scores]

        fig = go.Figure()
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
        fig.add_hrect(y0=0, y1=48, fillcolor="rgba(239,68,68,0.05)",
                      line_width=0, annotation_text="High Risk", annotation_position="right")
        fig.update_layout(
            xaxis_title="Quarter",
            yaxis_title="Risk Score (0–100)",
            yaxis=dict(range=[0, 105]),
        )
        st.plotly_chart(chart(fig, 360), use_container_width=True)

        r1, r2, r3 = st.columns(3)
        r1.metric("Current Score", f"{risk_scores[0]:.1f}/100")
        r2.metric("Q4 Projection", f"{risk_scores[4]:.1f}/100",
                  delta=f"{risk_scores[4] - risk_scores[0]:+.1f}")
        r3.metric("Q8 Projection", f"{risk_scores[8]:.1f}/100",
                  delta=f"{risk_scores[8] - risk_scores[0]:+.1f}")

    with tabs_f[3]:
        section_title(f"Financial Health Forecast — {c['name']} (5 Years)")
        np.random.seed(customer_seed + 20)
        months_range = list(range(0, 61, 3))
        labels_q = [f"Q{i // 3}" for i in months_range]

        active_loans = [l for l in c["loans"] if l["status"] == "Active"]
        total_monthly_pmt = sum(l["monthly_payment"] for l in active_loans)
        d = float(c["total_debt"])
        debt_curve = [max(0, d - total_monthly_pmt * m * 0.85) for m in months_range]

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
            fig.update_layout(
                yaxis_tickformat="$,.0f",
                xaxis_title="Quarter",
                legend=dict(x=0.01, y=0.99),
            )
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
            fig.update_layout(
                yaxis_tickformat="$,.0f",
                xaxis_title="Quarter",
                showlegend=False,
            )
            st.plotly_chart(chart(fig, 320), use_container_width=True)

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Monthly Debt Payments", f"${total_monthly_pmt:,.0f}")
        p2.metric("Projected Debt (5yr)", f"${debt_curve[-1]:,.0f}",
                  delta=f"${debt_curve[-1] - debt_curve[0]:+,.0f}")
        p3.metric("Projected Assets (5yr)", f"${asset_curve[-1]:,.0f}",
                  delta=f"${asset_curve[-1] - asset_curve[0]:+,.0f}")
        p4.metric("Net Worth (5yr)", f"${net_worth[-1]:,.0f}",
                  delta=f"${net_worth[-1] - net_worth[0]:+,.0f}")

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
        return ai_chat(system_msg,
                       f"Customer profiles:\n\n{context}\n\nQuestion: {question}",
                       model=OR_FAST_MODEL)

    # Chat history
    # Chat history
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Quick prompt trigger
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
