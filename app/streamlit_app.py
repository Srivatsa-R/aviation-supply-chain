"""
streamlit_app.py — Aviation Supply Chain Analytics Dashboard
=============================================================
Phase 5 · Dashboard & Visualisation

ENTRY POINT for the multi-page Streamlit application.
Run with: streamlit run app/streamlit_app.py

This file acts as the router and shared frame for all 5 pages.
Uses st.navigation (Streamlit ≥1.28) — the modern multipage pattern.

PAGES
-----
  1. 🏠 Overview          — KPI summary, key findings, project intro
  2. 🕸  Network Graph     — Interactive Pyvis supply chain network
  3. 📈 Demand Forecast   — ARIMA vs XGBoost charts, feature importance
  4. 📦 Inventory         — EOQ optimisation, ABC heatmap, saving waterfall
  5. ⚠️  Risk & Routing   — Supplier risk scores, LP cost comparison

DEPLOY
------
  Streamlit Cloud (free):
    1. Push this repo to GitHub (public)
    2. Go to share.streamlit.io
    3. Connect repo → set main file: app/streamlit_app.py
    4. Live URL: your-app.streamlit.app

AUTHOR  : Aviation SC Analytics Project
VERSION : 1.0.0 (Phase 5)
"""

import sys
from pathlib import Path

import streamlit as st

# ── Make project root importable ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Page config (applies to all pages as the shared frame) ───────────────────
st.set_page_config(
    page_title  = "Aviation SC Analytics",
    page_icon   = "✈️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
    menu_items  = {
        "Get Help":     "https://github.com/srivatsa-R/aviation-supply-chain",
        "Report a bug": "https://github.com/srivatsa-R/aviation-supply-chain/issues",
        "About":        "Aviation MRO Supply Chain Analytics · Built with Python & Streamlit",
    },
)

# ── Global CSS — dark theme consistent with project style ────────────────────
st.markdown("""
<style>
  /* Import project fonts */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  /* Root variables */
  :root {
    --accent: #0ea5e9;
    --green:  #10b981;
    --amber:  #f59e0b;
    --red:    #ef4444;
    --purple: #8b5cf6;
  }

  /* Sidebar styling */
  [data-testid="stSidebar"] {
    background: #0d1530 !important;
    border-right: 1px solid #1e2d4a;
  }
  [data-testid="stSidebar"] * { font-family: 'Inter', sans-serif !important; }

  /* Main content font */
  .main * { font-family: 'Inter', sans-serif !important; }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 16px !important;
  }
  [data-testid="stMetricValue"] { color: #0ea5e9 !important; font-weight: 700 !important; }

  /* Code blocks */
  code { font-family: 'JetBrains Mono', monospace !important; }

  /* Tab styling */
  [data-testid="stTabs"] button { font-weight: 600 !important; }

  /* Remove Streamlit's default top padding */
  .block-container { padding-top: 1.5rem !important; }

  /* Divider colour */
  hr { border-color: #1e2d4a !important; }
</style>
""", unsafe_allow_html=True)

# ── Navigation — using st.navigation (Streamlit ≥ 1.28 recommended pattern) ─
pages = {
    "📊 Dashboard": [
        st.Page("pages/1_Overview.py",        title="Overview",          icon="🏠"),
        st.Page("pages/2_Network_Graph.py",   title="Network Graph",     icon="🕸"),
        st.Page("pages/3_Demand_Forecast.py", title="Demand Forecast",   icon="📈"),
        st.Page("pages/4_Inventory.py",       title="Inventory & EOQ",   icon="📦"),
        st.Page("pages/5_Risk_Routing.py",    title="Risk & Routing",    icon="⚠️"),
    ],
}

pg = st.navigation(pages)

# ── Shared sidebar elements (appear on every page) ────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0 20px;">
      <div style="font-size:22px;font-weight:800;color:#fff;letter-spacing:-.01em;">
        ✈ Aviation SC
      </div>
      <div style="font-size:11px;color:#64748b;margin-top:2px;font-family:'JetBrains Mono';">
        MRO Supply Chain Analytics
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Key stats
    st.markdown("**📊 Project Stats**")
    col1, col2 = st.columns(2)
    col1.metric("Nodes",    "32")
    col2.metric("Edges",    "46")
    col1.metric("SKUs",     "500")
    col2.metric("Months",   "24")

    st.divider()

    # Links
    st.markdown("**🔗 Links**")
    st.markdown("- [GitHub Repo](https://github.com/srivatsa-R/aviation-supply-chain)")
    st.markdown("- [Interactive Network](network.html)")
    st.markdown("- [PDF Report](https://srivatsa-r.github.io/aviation-supply-chain/assets/Aviation_SC_Report.pdf)")
    st.markdown("- [LinkedIn](https://www.linkedin.com/in/srivatsa--r)")

    st.divider()
    st.caption("Phase 5 · Aviation SC Analytics · 2025")

pg.run()
