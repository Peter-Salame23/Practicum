"""
Scotiabank RAG Lending Intelligence Platform
Streamlit + ChromaDB + Claude claude-opus-4-6
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

RED   = "#EC111A"
DARK  = "#16213e"
GRAY  = "#f7f8fa"
WHITE = "#ffffff"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {{ font-family: 'Inter', 'Segoe UI', sans-serif; }}
  .stApp {{ background: {GRAY}; color: {DARK}; }}
  .block-container {{ padding: 1.5rem 2rem 3rem; max-width: 1400px; color: {DARK}; }}

  /* force all text in main content to dark — overrides Streamlit dark-mode white text */
  .block-container p, .block-container li, .block-container td, .block-container th,
  .block-container h1, .block-container h2, .block-container h3, .block-container h4,
  .block-container span, .block-container label, .block-container div {{
    color: {DARK};
  }}
  /* chat messages */
  [data-testid="stChatMessage"] p,
  [data-testid="stChatMessage"] li,
  [data-testid="stChatMessage"] td,
  [data-testid="stChatMessage"] th {{ color: {DARK} !important; }}

  /* chart container */
  div[data-testid="stPlotlyChart"] {{
    background: {WHITE};
    border: 1px solid #e8ecf0;
    border-radius: 10px;
    padding: 0.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }}

  /* sidebar */
  section[data-testid="stSidebar"] {{ background: {DARK}; border-right: none; }}
  section[data-testid="stSidebar"] * {{ color: #e2e8f0 !important; }}
  section[data-testid="stSidebar"] input {{ background: #1e2d4a !important; border: 1px solid #2d4a7a !important; color: white !important; border-radius: 6px !important; }}
  section[data-testid="stSidebar"] .stSelectbox > div > div {{ background: #1e2d4a !important; border: 1px solid #2d4a7a !important; }}

  /* hide streamlit chrome */
  #MainMenu, footer, header {{ visibility: hidden; }}

  /* metric cards */
  div[data-testid="metric-container"] {{
    background: {WHITE};
    border: 1px solid #e8ecf0;
    border-top: 3px solid {RED};
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }}
  div[data-testid="metric-container"] label {{ color: #6b7280; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }}
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {{ font-size: 1.35rem; font-weight: 700; color: {DARK}; }}
  div[data-testid="metric-container"] [data-testid="stMetricDelta"] {{ font-size: 0.75rem; }}

  /* section label */
  .sec-label {{
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9ca3af;
    margin-bottom: 0.4rem;
  }}
  .sec-title {{
    font-size: 1rem;
    font-weight: 700;
    color: {DARK};
    padding-bottom: 0.5rem;
    border-bottom: 2px solid {RED};
    margin-bottom: 1rem;
  }}

  /* page title */
  .page-title {{
    font-size: 1.6rem;
    font-weight: 700;
    color: {DARK};
    margin-bottom: 0.15rem;
  }}
  .page-sub {{ font-size: 0.85rem; color: #6b7280; margin-bottom: 1.5rem; }}

  /* risk badges */
  .badge {{ border-radius: 20px; padding: 3px 12px; font-size: 0.75rem; font-weight: 600; display: inline-block; }}
  .badge-low    {{ background: #dcfce7; color: #16a34a; }}
  .badge-medium {{ background: #fef9c3; color: #ca8a04; }}
  .badge-high   {{ background: #fee2e2; color: #dc2626; }}

  /* chat */
  .chat-user {{
    background: {RED}; color: white; padding: 0.65rem 1rem;
    border-radius: 14px 14px 2px 14px; margin: 0.4rem 0;
    max-width: 75%; float: right; clear: both; font-size: 0.9rem;
  }}
  .chat-bot {{
    background: {WHITE}; color: {DARK}; padding: 0.65rem 1rem;
    border-radius: 14px 14px 14px 2px; border: 1px solid #e8ecf0;
    margin: 0.4rem 0; max-width: 80%; float: left; clear: both;
    font-size: 0.9rem; white-space: pre-wrap; line-height: 1.55;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}
  .chat-wrap {{ overflow: hidden; margin-bottom: 0.25rem; }}

  /* loan card */
  .loan-card {{
    background: {WHITE}; border: 1px solid #e8ecf0; border-radius: 8px;
    padding: 0.9rem 1.1rem; margin-bottom: 0.6rem;
    border-left: 3px solid {RED};
  }}

  /* table */
  .dataframe {{ border: none !important; }}
  .stDataFrame {{ border-radius: 8px; overflow: hidden; border: 1px solid #e8ecf0; }}

  /* tabs */
  .stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 2px solid #e8ecf0; }}
  .stTabs [data-baseweb="tab"] {{ padding: 0.6rem 1.2rem; font-size: 0.85rem; font-weight: 500; color: #6b7280; background: transparent; border: none; }}
  .stTabs [aria-selected="true"] {{ color: {RED} !important; border-bottom: 2px solid {RED}; margin-bottom: -2px; }}

  /* buttons */
  .stButton > button {{ border-radius: 7px; font-weight: 600; font-size: 0.85rem; }}
  .stButton > button[kind="primary"] {{ background: {RED}; border: none; }}
  .stButton > button[kind="primary"]:hover {{ background: #c90e16; }}
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

customers    = load_customers()
collection   = get_chroma()
cust_index   = {c["id"]: c for c in customers}

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
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<div style='padding:1rem 0 0.5rem'>"
        "<div style='font-size:1.4rem;font-weight:800;letter-spacing:-0.5px'>🏦 Scotiabank</div>"
        "<div style='font-size:0.75rem;color:#94a3b8;margin-top:2px'>Lending Intelligence Platform</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── AI status ─────────────────────────────────────────────────────────────
    if OPENROUTER_KEY:
        st.markdown("<div style='font-size:0.75rem;color:#4ade80'>● AI Ready (OpenRouter)</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:0.75rem;color:#f87171'>● No API key found in data/.env</div>", unsafe_allow_html=True)

    st.divider()

    # ── navigation ────────────────────────────────────────────────────────────
    view = st.radio(
        "nav", label_visibility="collapsed",
        options=["Portfolio Overview", "Client Profile", "Loan Assessment", "Forecasting", "AI Assistant"],
    )

    st.divider()

    # ── client selector ───────────────────────────────────────────────────────
    st.markdown("<div style='font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;color:#94a3b8;margin-bottom:6px'>Active Client</div>", unsafe_allow_html=True)
    sorted_opts = sorted([(c["id"], c["name"]) for c in customers], key=lambda x: x[1])
    options_map = {f"{nm}  ({cid})": cid for cid, nm in sorted_opts}
    sel_label   = st.selectbox("client", list(options_map.keys()), label_visibility="collapsed")
    sel_id      = options_map[sel_label]
    sel         = cust_index[sel_id]

    risk_color = {"Low": "#4ade80", "Medium": "#facc15", "High": "#f87171"}[sel["risk_tier"]]
    st.markdown(
        f"<div style='background:#1e2d4a;border-radius:8px;padding:0.85rem;margin-top:0.3rem'>"
        f"<div style='font-weight:700;font-size:0.95rem'>{sel['name']}</div>"
        f"<div style='font-size:0.75rem;color:#94a3b8;margin:2px 0'>{sel['id']} · Age {sel['age']}</div>"
        f"<div style='font-size:0.8rem;margin-top:4px'>"
        f"<span style='color:{risk_color}'>●</span> {sel['risk_tier']} Risk"
        f"  ·  Score <b>{sel['credit_score']}</b></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# AI CLIENT — OpenRouter (OpenAI-compatible)
# ─────────────────────────────────────────────────────────────────────────────

def get_client():
    if not OPENROUTER_KEY:
        return None
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_KEY,
    )

OR_MODEL      = "anthropic/claude-opus-4-5"    # loan assessment (detailed)
OR_FAST_MODEL = "openai/gpt-4o-mini"           # chat assistant (fast, cheap)

def ai_chat(system: str, user: str, model: str = OR_MODEL, max_tokens: int = 1500) -> str:
    """Call OpenRouter. Returns the reply string or an error message — never raises."""
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
    """Retrieve top-n customer docs then ask the LLM. Never raises."""
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

    pti = round((pmt / (customer["annual_income"] / 12)) * 100, 1)
    system = f"""You are a senior underwriter at Scotiabank. Write a structured loan assessment report.
The client data is provided. Use EXACTLY these numbers — do not make up values.

Here is the report template to fill out:

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

def chart(fig, h=360):
    fig.update_layout(
        height=h, paper_bgcolor=WHITE, plot_bgcolor=WHITE,
        font=dict(family="Inter, Segoe UI", size=12, color="#374151"),
        margin=dict(l=16, r=16, t=36, b=16),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )
    fig.update_xaxes(gridcolor="#f3f4f6", showline=False, tickfont=dict(size=11))
    fig.update_yaxes(gridcolor="#f3f4f6", showline=False, tickfont=dict(size=11))
    return fig

RISK_COLORS = {"Low": "#16a34a", "Medium": "#ca8a04", "High": "#dc2626"}

# ─────────────────────────────────────────────────────────────────────────────
# VIEW 1 — PORTFOLIO OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

if view == "Portfolio Overview":
    st.markdown("<div class='page-title'>Portfolio Overview</div><div class='page-sub'>75 clients · Scotiabank lending book</div>", unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Clients",      f"{len(df):,}")
    m2.metric("Avg Credit Score",   f"{int(df.credit_score.mean()):,}")
    m3.metric("Loan Exposure",      f"${df.total_debt.sum()/1e6:.1f}M")
    m4.metric("High-Risk Clients",  f"{len(df[df.risk_tier=='High'])} ({len(df[df.risk_tier=='High'])/len(df):.0%})")
    m5.metric("Total Deposits",     f"${(df.checking+df.savings).sum()/1e6:.1f}M")

    st.markdown("<br>", unsafe_allow_html=True)

    # row 1
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("<div class='sec-title'>Risk Tier Distribution</div>", unsafe_allow_html=True)
        rc = df.risk_tier.value_counts().reset_index()
        rc.columns = ["Tier", "Count"]
        fig = px.pie(rc, names="Tier", values="Count", hole=0.52,
                     color="Tier", color_discrete_map=RISK_COLORS)
        fig.update_traces(textposition="inside", textinfo="percent+label",
                          textfont_size=12, marker=dict(line=dict(color=WHITE, width=2)))
        fig.update_layout(showlegend=False)
        st.plotly_chart(chart(fig, 320), use_container_width=True)
        
    with c2:
        st.markdown("<div class='sec-title'>Credit Score Distribution</div>", unsafe_allow_html=True)
        fig = px.histogram(df, x="credit_score", nbins=28,
                           color_discrete_sequence=[RED],
                           labels={"credit_score": "Credit Score", "count": "Clients"})
        fig.update_traces(marker_line_width=0, opacity=0.85)
        fig.add_vline(x=750, line_dash="dash", line_color="#16a34a", line_width=1.5,
                      annotation_text="Excellent", annotation_font_size=11)
        fig.add_vline(x=670, line_dash="dash", line_color="#ca8a04", line_width=1.5,
                      annotation_text="Fair", annotation_font_size=11)
        st.plotly_chart(chart(fig, 320), use_container_width=True)
        
    # row 2
    c3, c4 = st.columns(2, gap="medium")
    with c3:
        st.markdown("<div class='sec-title'>Income vs Total Debt</div>", unsafe_allow_html=True)
        fig = px.scatter(df, x="annual_income", y="total_debt",
                         color="risk_tier", color_discrete_map=RISK_COLORS,
                         hover_data=["name", "credit_score"],
                         labels={"annual_income": "Annual Income ($)", "total_debt": "Total Debt ($)",
                                 "risk_tier": "Risk"},
                         opacity=0.7)
        fig.update_traces(marker=dict(size=7))
        st.plotly_chart(chart(fig, 320), use_container_width=True)
        
    with c4:
        st.markdown("<div class='sec-title'>Employment Status</div>", unsafe_allow_html=True)
        ec = df.employment.value_counts().reset_index()
        ec.columns = ["Status", "Count"]
        fig = px.bar(ec, x="Count", y="Status", orientation="h",
                     color_discrete_sequence=[RED], text="Count")
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(chart(fig, 320), use_container_width=True)
        
    # client table
        st.markdown("<div class='sec-title'>Client Directory</div>", unsafe_allow_html=True)
    c_filter1, c_filter2, _ = st.columns([1, 1, 3])
    risk_filter = c_filter1.multiselect("Risk", ["Low", "Medium", "High"], ["Low", "Medium", "High"])
    emp_filter  = c_filter2.multiselect("Employment", df.employment.unique().tolist(), df.employment.unique().tolist())

    tbl = df[df.risk_tier.isin(risk_filter) & df.employment.isin(emp_filter)].copy()
    tbl["Annual Income"]  = tbl.annual_income.map("${:,.0f}".format)
    tbl["Total Debt"]     = tbl.total_debt.map("${:,.0f}".format)
    tbl["DTI"]            = tbl.dti.map("{:.1%}".format)
    tbl["Assets"]         = tbl.total_assets.map("${:,.0f}".format)
    st.dataframe(
        tbl[["id","name","age","credit_score","Annual Income","Total Debt","DTI","Assets","risk_tier","active_loans"]]
        .rename(columns={"id":"ID","name":"Name","age":"Age","credit_score":"Score",
                         "risk_tier":"Risk","active_loans":"Loans"}),
        use_container_width=True, height=340, hide_index=True,
    )
    
# ─────────────────────────────────────────────────────────────────────────────
# VIEW 2 — CLIENT PROFILE
# ─────────────────────────────────────────────────────────────────────────────

elif view == "Client Profile":
    c = sel
    badge_cls = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high"}[c["risk_tier"]]

    st.markdown(
        f"<div class='page-title'>{c['name']}"
        f"  <span class='badge {badge_cls}'>{c['risk_tier']} Risk</span></div>"
        f"<div class='page-sub'>{c['id']} · {c['job_title']} · Member since {c['member_since']}</div>",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Credit Score",   c["credit_score"])
    m2.metric("Annual Income",  f"${c['annual_income']:,}")
    m3.metric("Checking",       f"${c['checking_balance']:,}")
    m4.metric("Savings",        f"${c['savings_balance']:,}")
    m5.metric("Total Debt",     f"${c['total_debt']:,}")
    m6.metric("DTI",            f"{c['debt_to_income_ratio']:.1%}")

    st.markdown("<br>", unsafe_allow_html=True)
    tabs = st.tabs(["Overview", "Transactions", "Loans", "Analytics"])

    # ── OVERVIEW ──
    with tabs[0]:
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown("<div class='sec-title'>Personal & Employment</div>", unsafe_allow_html=True)
            rows = [
                ("Age / Gender",    f"{c['age']} / {c['gender']}"),
                ("Email",           c["email"]),
                ("Phone",           c["phone"]),
                ("Address",         c["address"]),
                ("Employment",      f"{c['employment_status']} — {c['job_title']}"),
                ("Employer",        c["employer"]),
                ("Tenure",          f"{c['years_at_employer']} years"),
                ("Monthly Expenses",f"${c['monthly_expenses']:,}"),
            ]
            for k, v in rows:
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;padding:5px 0;"
                    f"border-bottom:1px solid #f3f4f6;font-size:0.85rem'>"
                    f"<span style='color:#6b7280'>{k}</span><span style='font-weight:500'>{v}</span></div>",
                    unsafe_allow_html=True,
                )
            
        with col2:
            st.markdown("<div class='sec-title'>Account Balances</div>", unsafe_allow_html=True)
            bal = pd.DataFrame({"Account": ["Checking", "Savings", "Investments"],
                                "Balance": [c["checking_balance"], c["savings_balance"], c["investment_balance"]]})
            fig = px.bar(bal, x="Account", y="Balance",
                         color="Account", text_auto="$.2s",
                         color_discrete_sequence=["#3b82f6", "#10b981", RED])
            fig.update_traces(textposition="outside", marker_line_width=0)
            fig.update_layout(showlegend=False, yaxis_tickformat="$,.0f")
            st.plotly_chart(chart(fig, 240), use_container_width=True)
            
            st.markdown("<div class='sec-title'>Credit Risk Flags</div>", unsafe_allow_html=True)
            flags = [
                ("Late Payments",           str(c["num_late_payments"]), c["num_late_payments"] > 0),
                ("Months Since Delinquency",str(c.get("months_since_last_delinquency") or "—"), False),
                ("Open Accounts",           str(c["num_open_accounts"]), False),
                ("Bankruptcy",              "Yes ⚠️" if c["bankruptcy_history"] else "No ✅", c["bankruptcy_history"]),
            ]
            for k, v, warn in flags:
                color = RED if warn else "#374151"
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;padding:5px 0;"
                    f"border-bottom:1px solid #f3f4f6;font-size:0.85rem'>"
                    f"<span style='color:#6b7280'>{k}</span>"
                    f"<span style='font-weight:600;color:{color}'>{v}</span></div>",
                    unsafe_allow_html=True,
                )
            
    # ── TRANSACTIONS ──
    with tabs[1]:
        txns = c.get("transactions", [])
        if txns:
            txn_df = pd.DataFrame(txns)
            txn_df["date"] = pd.to_datetime(txn_df["date"])

            c1, c2 = st.columns(2, gap="medium")
            with c1:
                st.markdown("<div class='sec-title'>Monthly Cash Flow</div>", unsafe_allow_html=True)
                mo = txn_df.groupby(txn_df["date"].dt.to_period("M"))["amount"].sum().reset_index()
                mo["date"] = mo["date"].astype(str)
                mo["type"] = mo["amount"].apply(lambda x: "Inflow" if x >= 0 else "Outflow")
                fig = px.bar(mo, x="date", y="amount", color="type",
                             color_discrete_map={"Inflow": "#10b981", "Outflow": RED},
                             labels={"date": "Month", "amount": "Net ($)", "type": ""})
                fig.update_traces(marker_line_width=0)
                st.plotly_chart(chart(fig, 300), use_container_width=True)
                
            with c2:
                st.markdown("<div class='sec-title'>Spending by Category</div>", unsafe_allow_html=True)
                exp = txn_df[txn_df["amount"] < 0].copy()
                exp["amount"] = exp["amount"].abs()
                cs = exp.groupby("category")["amount"].sum().reset_index()
                fig = px.pie(cs, names="category", values="amount", hole=0.45,
                             color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_traces(textposition="inside", textinfo="percent+label",
                                  marker=dict(line=dict(color=WHITE, width=1.5)))
                st.plotly_chart(chart(fig, 300), use_container_width=True)
                
                st.markdown("<div class='sec-title'>Recent Transactions</div>", unsafe_allow_html=True)
            disp = txn_df.head(50).copy()
            disp["Date"] = disp["date"].dt.strftime("%Y-%m-%d")
            disp["Amount"] = disp["amount"].apply(
                lambda x: f"+${x:,.2f}" if x > 0 else f"-${abs(x):,.2f}"
            )
            disp["_color"] = disp["amount"].apply(lambda x: "green" if x > 0 else RED)
            st.dataframe(
                disp[["Date", "category", "description", "Amount"]]
                .rename(columns={"category": "Category", "description": "Description"}),
                use_container_width=True, height=320, hide_index=True,
            )
            
    # ── LOANS ──
    with tabs[2]:
        loans = c.get("loans", [])
        active = [l for l in loans if l["status"] == "Active"]
        closed = [l for l in loans if l["status"] == "Closed"]

        if active:
            st.markdown(f"<div class='sec-title'>Active Loans ({len(active)})</div>", unsafe_allow_html=True)
            for l in active:
                flag = " ⚠️" if l["missed_payments"] > 0 else ""
                st.markdown(
                    f"<div class='loan-card'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                    f"<span style='font-weight:700'>{l['type']} Loan{flag}</span>"
                    f"<span style='font-size:0.8rem;color:#6b7280'>{l['start_date']} → {l['end_date']}</span>"
                    f"</div>"
                    f"<div style='display:flex;gap:2rem;margin-top:6px;font-size:0.85rem'>"
                    f"<span><b>${l['amount']:,}</b> principal</span>"
                    f"<span><b>{l['rate']}%</b> APR</span>"
                    f"<span><b>${l['monthly_payment']:,.2f}</b>/mo</span>"
                    f"<span style='color:{'#dc2626' if l['missed_payments']>0 else '#16a34a'}'>"
                    f"{'⚠ ' + str(l['missed_payments']) + ' missed' if l['missed_payments'] else '✓ No missed payments'}"
                    f"</span></div></div>",
                    unsafe_allow_html=True,
                )

            # amortization chart
            if active:
                st.markdown("<div class='sec-title'>Amortization Schedule</div>", unsafe_allow_html=True)
                fig = go.Figure()
                palette = [RED, "#3b82f6", "#10b981", "#f59e0b"]
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
                    fig.add_trace(go.Scatter(x=dates, y=bals, name=l["type"],
                                             mode="lines", line=dict(color=palette[i % 4], width=2.5)))
                fig.update_layout(yaxis_tickformat="$,.0f", xaxis_title="Date", yaxis_title="Balance")
                st.plotly_chart(chart(fig, 320), use_container_width=True)
                
        if closed:
            st.markdown(f"<div class='sec-title' style='margin-top:1rem'>Closed Loans ({len(closed)})</div>", unsafe_allow_html=True)
            for l in closed:
                st.markdown(
                    f"<div class='loan-card' style='opacity:0.6;border-left-color:#d1d5db'>"
                    f"<b>{l['type']} Loan</b> — Closed"
                    f"  <span style='color:#6b7280;font-size:0.82rem'>${l['amount']:,} · {l['start_date']} → {l['end_date']}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        if not loans:
            st.info("No loan history for this client.")

    # ── ANALYTICS ──
    with tabs[3]:
        def clamp(v, lo, hi): return max(lo, min(hi, v))
        def norm(v, lo, hi, inv=False):
            n = (v - lo) / (hi - lo) * 100
            return clamp(100 - n if inv else n, 0, 100)

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown("<div class='sec-title'>Financial Health Radar</div>", unsafe_allow_html=True)
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
                fillcolor="rgba(236,17,26,0.15)",
                line=dict(color=RED, width=2.5),
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10),
                                    gridcolor="#e5e7eb"),
                    angularaxis=dict(tickfont=dict(size=11)),
                    bgcolor=WHITE,
                ),
                showlegend=False, height=360,
                paper_bgcolor=WHITE, margin=dict(l=40, r=40, t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.markdown("<div class='sec-title'>Peer Comparison — Same Risk Tier</div>", unsafe_allow_html=True)
            peers = df[df.risk_tier == c["risk_tier"]]
            compare_items = [
                ("Credit Score",  c["credit_score"],              int(peers.credit_score.mean()),   None),
                ("Annual Income", c["annual_income"],             int(peers.annual_income.mean()),  "${:,}"),
                ("Total Debt",    c["total_debt"],                int(peers.total_debt.mean()),     "${:,}"),
                ("DTI Ratio",     c["debt_to_income_ratio"],      float(peers.dti.mean()),          "{:.1%}"),
            ]
            for label, cval, pval, fmt in compare_items:
                if fmt:
                    cv = fmt.format(cval)
                    pv = fmt.format(pval)
                else:
                    cv, pv = str(cval), str(pval)

                pct = (cval - pval) / max(abs(pval), 1) * 100
                better = (label in ("Credit Score", "Annual Income") and cval >= pval) or \
                         (label in ("Total Debt", "DTI Ratio") and cval <= pval)
                arrow = "↑" if cval >= pval else "↓"
                delta_color = "#16a34a" if better else "#dc2626"
                st.markdown(
                    f"<div style='padding:8px 0;border-bottom:1px solid #f3f4f6'>"
                    f"<div style='font-size:0.75rem;color:#9ca3af;text-transform:uppercase;letter-spacing:0.04em'>{label}</div>"
                    f"<div style='display:flex;align-items:baseline;gap:0.5rem;margin-top:2px'>"
                    f"<span style='font-size:1.1rem;font-weight:700'>{cv}</span>"
                    f"<span style='font-size:0.8rem;color:{delta_color}'>{arrow} {abs(pct):.0f}% vs avg {pv}</span>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
            
# ─────────────────────────────────────────────────────────────────────────────
# VIEW 3 — LOAN ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────

elif view == "Loan Assessment":
    c = sel
    badge_cls = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high"}[c["risk_tier"]]
    st.markdown(
        f"<div class='page-title'>Loan Assessment</div>"
        f"<div class='page-sub'>{c['name']} · {c['id']} · "
        f"<span class='badge {badge_cls}'>{c['risk_tier']} Risk</span></div>",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Credit Score",   c["credit_score"])
    m2.metric("Annual Income",  f"${c['annual_income']:,}")
    m3.metric("Current DTI",    f"{c['debt_to_income_ratio']:.1%}")
    m4.metric("Risk Score",     f"{c['risk_score']}/100")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.markdown("<div class='sec-title'>Loan Parameters</div>", unsafe_allow_html=True)
        loan_type = st.selectbox("Loan Type", ["Personal", "Auto", "Mortgage", "Business", "Student", "Line of Credit"])
        max_amt   = {"Personal": 100_000, "Auto": 150_000, "Mortgage": 1_500_000,
                     "Business": 500_000, "Student": 150_000, "Line of Credit": 100_000}[loan_type]
        amount    = st.number_input("Amount ($)", 1_000, max_amt, min(25_000, max_amt), step=1_000)
        terms     = {"Personal":[12,24,36,48,60],"Auto":[24,36,48,60,72,84],
                     "Mortgage":[120,180,240,300,360],"Business":[12,24,36,60,84],
                     "Student":[60,84,120],"Line of Credit":[12,24,36]}[loan_type]
        term      = st.selectbox("Term (months)", terms)

        rate_est  = {"Personal":9.5,"Auto":6.9,"Mortgage":5.5,"Business":8.5,"Student":6.0,"Line of Credit":10.5}[loan_type]
        r         = rate_est / 100 / 12
        est_pmt   = round(amount * r / (1 - (1 + r) ** -term), 2)
        new_dti   = round((c["total_debt"] + amount) / max(c["annual_income"], 1), 4)

        st.markdown("<br>", unsafe_allow_html=True)
        e1, e2 = st.columns(2)
        e1.metric("Est. Payment/mo", f"${est_pmt:,.0f}")
        e2.metric("DTI after loan",  f"{new_dti:.1%}", delta=f"{new_dti-c['debt_to_income_ratio']:+.1%}")

        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("Run AI Assessment →", type="primary", use_container_width=True)
        
    with col2:
        st.markdown("<div class='sec-title'>Underwriter Report</div>", unsafe_allow_html=True)

        key = f"assess_{c['id']}"
        if run:
            result = loan_assessment(c, amount, loan_type, term)
            st.session_state[key] = result

        if key in st.session_state:
            text = st.session_state[key]
            border_color = ("#16a34a" if "APPROVED" in text.upper() and "DECLINED" not in text.upper()
                            else "#dc2626" if "DECLINED" in text.upper() else "#ca8a04")
            st.markdown(
                f"<div style='border-left:4px solid {border_color};padding-left:1rem;margin-bottom:0.5rem'></div>",
                unsafe_allow_html=True,
            )
            st.markdown(text)
        else:
            st.markdown(
                "<div style='color:#9ca3af;font-size:0.9rem;padding:2rem 0;text-align:center'>"
                "Configure parameters and click <b>Run AI Assessment</b></div>",
                unsafe_allow_html=True,
            )
        
# ─────────────────────────────────────────────────────────────────────────────
# VIEW 4 — FORECASTING
# ─────────────────────────────────────────────────────────────────────────────

elif view == "Forecasting":
    st.markdown("<div class='page-title'>Forecasting & Analytics</div><div class='page-sub'>Projections and portfolio trends</div>", unsafe_allow_html=True)
    tabs_f = st.tabs(["Credit Score Forecast", "Income Projection", "Risk Evolution", "Portfolio Health"])

    with tabs_f[0]:
        c = sel
        st.markdown(f"<div class='sec-title'>Credit Score Forecast — {c['name']} (24 months)</div>", unsafe_allow_html=True)
        np.random.seed(int(c["id"][1:]))
        imp = {"High": 0.35, "Medium": 0.18, "Low": 0.05}[c["risk_tier"]]
        months = list(range(25))
        noise  = np.random.normal(0, 3, 25)
        scores = [min(850, c["credit_score"] + imp * m * 5 + noise[m]) for m in months]
        upper  = [min(850, s + 18) for s in scores]
        lower  = [max(300, s - 18) for s in scores]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months+months[::-1], y=upper+lower[::-1],
                                 fill="toself", fillcolor="rgba(236,17,26,0.08)",
                                 line=dict(color="rgba(0,0,0,0)"), showlegend=False))
        fig.add_trace(go.Scatter(x=months, y=scores, name="Projected",
                                 mode="lines+markers", line=dict(color=RED, width=2.5),
                                 marker=dict(size=5)))
        fig.add_hline(y=750, line_dash="dot", line_color="#3b82f6", line_width=1.5,
                      annotation_text="Excellent (750)", annotation_position="right")
        fig.add_hline(y=700, line_dash="dot", line_color="#10b981", line_width=1.5,
                      annotation_text="Good (700)", annotation_position="right")
        fig.update_layout(xaxis_title="Months from Today", yaxis_title="Credit Score",
                          yaxis=dict(range=[300, 870]))
        st.plotly_chart(chart(fig, 380), use_container_width=True)

        n1, n2, n3 = st.columns(3)
        n1.metric("Current",          int(scores[0]))
        n2.metric("12-Month",         int(scores[12]), delta=f"{int(scores[12]-scores[0]):+d}")
        n3.metric("24-Month",         int(scores[24]), delta=f"{int(scores[24]-scores[0]):+d}")
        
    with tabs_f[1]:
        c = sel
        st.markdown(f"<div class='sec-title'>Income & Savings Projection — 5 Years</div>", unsafe_allow_html=True)
        growth = {"Full-Time": 0.04, "Self-Employed": 0.06, "Part-Time": 0.02,
                  "Unemployed": 0.0, "Retired": 0.01}.get(c["employment_status"], 0.03)
        np.random.seed(int(c["id"][1:]) + 1)
        yrs     = list(range(6))
        income  = [c["annual_income"] * (1 + growth) ** y + np.random.normal(0, c["annual_income"] * 0.015) for y in yrs]
        expense = [c["monthly_expenses"] * 12 * (1.025 ** y) for y in yrs]
        net     = [i - e for i, e in zip(income, expense)]
        labels  = [f"Year {y}" for y in yrs]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Income",   x=labels, y=income,  marker_color="#10b981", opacity=0.85))
        fig.add_trace(go.Bar(name="Expenses", x=labels, y=expense, marker_color=RED,       opacity=0.85))
        fig.add_trace(go.Scatter(name="Net Savings", x=labels, y=net,
                                 mode="lines+markers", line=dict(color="#3b82f6", width=2.5),
                                 marker=dict(size=7)))
        fig.update_layout(barmode="group", yaxis_tickformat="$,.0f")
        st.plotly_chart(chart(fig, 360), use_container_width=True)
        
    with tabs_f[2]:
        c = sel
        st.markdown(f"<div class='sec-title'>Risk Score Evolution — {c['name']} (8 Quarters)</div>", unsafe_allow_html=True)

        # Simulate client-specific risk score over 8 quarters
        np.random.seed(int(c["id"][1:]) + 10)
        qtrs = [f"Q{(i%4)+1} '2{'5' if i < 4 else '6'}" for i in range(9)]

        # Improvement rate based on current risk tier and credit factors
        base_improve = {"High": 1.8, "Medium": 0.9, "Low": 0.3}[c["risk_tier"]]
        if c["bankruptcy_history"]: base_improve *= 0.4
        if c["num_late_payments"] > 2: base_improve *= 0.6

        noise = np.random.normal(0, 0.6, 9)
        risk_scores = [min(100, max(0, c["risk_score"] + base_improve * i + noise[i])) for i in range(9)]

        # Determine tier bands
        tier_color = {"Low": "#16a34a", "Medium": "#ca8a04", "High": "#dc2626"}[c["risk_tier"]]

        fig = go.Figure()
        # confidence band
        upper = [min(100, s + 4) for s in risk_scores]
        lower = [max(0,   s - 4) for s in risk_scores]
        fig.add_trace(go.Scatter(
            x=qtrs + qtrs[::-1], y=upper + lower[::-1],
            fill="toself", fillcolor=f"rgba(236,17,26,0.08)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=qtrs, y=risk_scores, name="Risk Score",
            mode="lines+markers", line=dict(color=tier_color, width=2.5),
            marker=dict(size=6),
        ))
        fig.add_hrect(y0=72, y1=100, fillcolor="rgba(22,163,74,0.06)", line_width=0, annotation_text="Low Risk Zone", annotation_position="right")
        fig.add_hrect(y0=48, y1=72, fillcolor="rgba(202,138,4,0.06)", line_width=0, annotation_text="Medium Risk Zone", annotation_position="right")
        fig.add_hrect(y0=0,  y1=48, fillcolor="rgba(220,38,38,0.06)", line_width=0, annotation_text="High Risk Zone", annotation_position="right")
        fig.update_layout(xaxis_title="Quarter", yaxis_title="Risk Score (0–100)", yaxis=dict(range=[0, 105]))
        st.plotly_chart(chart(fig, 360), use_container_width=True)

        r1, r2, r3 = st.columns(3)
        r1.metric("Current Score",  f"{risk_scores[0]:.1f}/100")
        r2.metric("Q4 Projection",  f"{risk_scores[4]:.1f}/100", delta=f"{risk_scores[4]-risk_scores[0]:+.1f}")
        r3.metric("Q8 Projection",  f"{risk_scores[8]:.1f}/100", delta=f"{risk_scores[8]-risk_scores[0]:+.1f}")

    with tabs_f[3]:
        c = sel
        st.markdown(f"<div class='sec-title'>Financial Health Forecast — {c['name']} (5 Years)</div>", unsafe_allow_html=True)

        np.random.seed(int(c["id"][1:]) + 20)
        months_range = list(range(0, 61, 3))  # every quarter for 5 years
        labels_q = [f"Q{i//3}" for i in months_range]

        # Debt paydown: each active loan reduces by its monthly payment
        active_loans = [l for l in c["loans"] if l["status"] == "Active"]
        total_monthly_pmt = sum(l["monthly_payment"] for l in active_loans)
        debt_curve = []
        d = float(c["total_debt"])
        for m in months_range:
            debt_curve.append(max(0, d - total_monthly_pmt * m * 0.85))  # 85% goes to principal

        # Asset growth
        growth_rate = {"Full-Time": 0.005, "Self-Employed": 0.006, "Part-Time": 0.003,
                       "Unemployed": -0.002, "Retired": 0.003}.get(c["employment_status"], 0.004)
        asset_curve = [c["total_assets"] * (1 + growth_rate) ** m + np.random.normal(0, c["total_assets"]*0.01)
                       for m in months_range]

        net_worth = [a - d for a, d in zip(asset_curve, debt_curve)]

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown("<div class='sec-title'>Debt Paydown Trajectory</div>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=labels_q, y=debt_curve, name="Total Debt",
                                     mode="lines+markers", fill="tozeroy",
                                     fillcolor="rgba(236,17,26,0.10)",
                                     line=dict(color=RED, width=2.5), marker=dict(size=4)))
            fig.add_trace(go.Scatter(x=labels_q, y=asset_curve, name="Total Assets",
                                     mode="lines", line=dict(color="#10b981", width=2, dash="dot")))
            fig.update_layout(yaxis_tickformat="$,.0f", xaxis_title="Quarter", legend=dict(x=0.01, y=0.99))
            st.plotly_chart(chart(fig, 320), use_container_width=True)

        with c2:
            st.markdown("<div class='sec-title'>Net Worth Projection</div>", unsafe_allow_html=True)
            nw_colors = ["#16a34a" if v >= 0 else "#dc2626" for v in net_worth]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=labels_q, y=net_worth, name="Net Worth",
                                  marker_color=nw_colors, opacity=0.85))
            fig.add_hline(y=0, line_dash="dash", line_color="#6b7280", line_width=1)
            fig.update_layout(yaxis_tickformat="$,.0f", xaxis_title="Quarter", showlegend=False)
            st.plotly_chart(chart(fig, 320), use_container_width=True)

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Monthly Debt Payments",  f"${total_monthly_pmt:,.0f}")
        p2.metric("Projected Debt (5yr)",   f"${debt_curve[-1]:,.0f}", delta=f"${debt_curve[-1]-debt_curve[0]:+,.0f}")
        p3.metric("Projected Assets (5yr)", f"${asset_curve[-1]:,.0f}", delta=f"${asset_curve[-1]-asset_curve[0]:+,.0f}")
        p4.metric("Net Worth (5yr)",        f"${net_worth[-1]:,.0f}", delta=f"${net_worth[-1]-net_worth[0]:+,.0f}")
            
# ─────────────────────────────────────────────────────────────────────────────
# VIEW 5 — AI ASSISTANT
# ─────────────────────────────────────────────────────────────────────────────

elif view == "AI Assistant":
    st.markdown(
        "<div class='page-title'>AI Lending Assistant</div>"
        "<div class='page-sub'>Ask anything about clients, risk, or loan eligibility</div>",
        unsafe_allow_html=True,
    )

    if not OPENROUTER_KEY:
        st.error("No API key found. Add OpenrouterApiKey to data/.env")
        st.stop()

    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "pending_q" not in st.session_state:
        st.session_state.pending_q = None

    # ── quick prompts ─────────────────────────────────────────────────────────
    st.markdown("<div style='font-size:0.75rem;font-weight:600;color:#6b7280;margin-bottom:0.5rem;text-transform:uppercase;letter-spacing:0.05em'>Quick prompts</div>", unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)
    prompts = [
        f"Is {sel['name']} eligible for a $30,000 personal loan?",
        "Which clients have the highest default risk?",
        "List clients with credit scores over 750 and low debt.",
        "Who has the worst debt-to-income ratio?",
    ]
    for col, prompt in zip([q1, q2, q3, q4], prompts):
        if col.button(prompt[:45] + "…", key=f"qp_{prompt[:20]}", use_container_width=True):
            st.session_state.pending_q = prompt

    st.divider()

    # ── helper: call AI and return answer string ──────────────────────────────
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
        return ai_chat(system_msg, f"Customer profiles:\n\n{context}\n\nQuestion: {question}", model=OR_FAST_MODEL)

    # ── render chat history ───────────────────────────────────────────────────
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── handle quick-prompt button click ─────────────────────────────────────
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

    # ── handle typed input ────────────────────────────────────────────────────
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
