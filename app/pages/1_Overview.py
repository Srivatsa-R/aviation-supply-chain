"""
Page 1: Overview — KPI Summary & Key Findings
Aviation Supply Chain Analytics Dashboard
"""
import sys
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Overview · Aviation SC", page_icon="🏠", layout="wide")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="font-size:2rem;font-weight:800;color:#fff;margin-bottom:4px;">
  ✈️ Aviation MRO Supply Chain Analytics
</h1>
<p style="color:#64748b;font-size:15px;margin-bottom:24px;">
  End-to-end network analysis, ML demand forecasting, and optimisation of an
  aviation MRO spare-parts supply chain · 2025–2026
</p>
""", unsafe_allow_html=True)

# ── Alert banner ──────────────────────────────────────────────────────────────
st.warning(
    "⚠️ **Industry Context 2025–2026:** Aviation supply chains face $11B+ disruption costs. "
    "Aircraft delivery backlogs hit 17,000 jets. This project quantifies and mitigates those risks."
)

# ── KPI metrics row ───────────────────────────────────────────────────────────
st.markdown("### 📊 Project Key Results")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Supply Chain Nodes",     "32",        "5 tiers modelled")
c2.metric("XGBoost MAPE",           "2.05%",     "↓ vs 13.11% ARIMA",    delta_color="inverse")
c3.metric("Inventory Saving/yr",    "$4.37M",    "+24% vs naive policy")
c4.metric("LP Route Saving",        "$13.1M",    "+60.7% vs current")
c5.metric("Supplier Risk Accuracy", "92%",       "RF OOB score")
c6.metric("Critical Path (days)",   "194d",      "Rolls-Royce → Qantas", delta_color="off")

st.divider()

# ── Two-column layout: findings + architecture ────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("### 🔬 Key Findings")

    findings = [
        ("🕸 Network Bottleneck",
         "Removing just 5 suppliers (15.6% of the network) disrupts **39% of all supply edges** "
         "and increases average path length by **45.3%** — confirming single-point-of-failure risk "
         "concentrated in a small number of high-betweenness nodes."),
        ("⚠️ Highest Risk Supplier",
         "**Parker Hannifin** carries the highest composite risk score (0.2351): single-source "
         "supplier to multiple MROs with below-benchmark OTD reliability. Dual-sourcing this node "
         "would reduce AOG exposure for 3 airlines simultaneously."),
        ("📈 ML Forecasting Uplift",
         "XGBoost achieves **2.05% MAPE** versus ARIMA's 13.11% baseline — an **84.4% accuracy "
         "improvement**. The top predictive feature is log(annual_demand), followed by lag-2 demand "
         "and 3-month rolling mean, confirming MRO demand is pattern-driven, not random."),
        ("📦 Hidden AOG Risk in Class-C Parts",
         "Class-C AOG-critical parts show **negative inventory savings** — they are currently "
         "under-buffered. Cheap fasteners and consumables (\\$10–500/unit) ground aircraft when "
         "missing, but are treated as low-priority. EOQ model correctly increases their safety stock."),
        ("💰 Route Optimisation",
         "LP optimisation identifies a **\\$13.1M saving (60.7%)** versus historical supplier routing. "
         "8 active routes replace 15 eligible routes — the LP concentrates volume on Parker Hannifin "
         "and Heico Corp (highest capacity, lowest unit cost per route)."),
        ("🤖 RF Blind Spot",
         "Random Forest assigns **GE Aviation a 0.675 risk probability** despite a MEDIUM rule-based "
         "score. The RF surfaces high lead-time variability (σ=15 days) that the rule-based model "
         "under-weights — a real analytical insight missed by traditional scoring."),
    ]

    for title, body in findings:
        with st.expander(title, expanded=True):
            st.markdown(body)

with right:
    st.markdown("### 🗺 Project Architecture")

    # Architecture flow diagram using plotly
    phases = [
        ("Phase 1", "Research &\nData Discovery",   "#0ea5e9"),
        ("Phase 2", "ETL Pipeline\n& Synthetic Data","#10b981"),
        ("Phase 3", "Network Graph\n& Disruption Sim","#8b5cf6"),
        ("Phase 4", "ML Models &\nOptimisation",     "#f59e0b"),
        ("Phase 5", "Dashboard &\nDeployment",        "#ef4444"),
    ]

    fig = go.Figure()
    for i, (ph, label, color) in enumerate(phases):
        fig.add_trace(go.Scatter(
            x=[0.5], y=[len(phases) - i],
            mode="markers+text",
            marker=dict(size=52, color=color, line=dict(color="white", width=2)),
            text=[ph], textposition="middle center",
            textfont=dict(color="white", size=10, family="Inter"),
            hovertext=label.replace("\n", " "),
            hoverinfo="text",
            showlegend=False,
        ))
        fig.add_annotation(
            x=0.75, y=len(phases) - i,
            text=f"<b>{label.replace(chr(10), '<br>')}</b>",
            showarrow=False, align="left",
            font=dict(size=12, color="#e2e8f0"),
            xanchor="left",
        )
        if i < len(phases) - 1:
            fig.add_shape(
                type="line", x0=0.5, y0=len(phases)-i-0.35,
                x1=0.5, y1=len(phases)-i-0.65,
                line=dict(color="#1e2d4a", width=2, dash="dot"),
            )

    fig.update_layout(
        plot_bgcolor="#0d1530", paper_bgcolor="#0d1530",
        xaxis=dict(visible=False, range=[0, 1.5]),
        yaxis=dict(visible=False, range=[0, len(phases)+0.5]),
        height=340, margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🗄 Data Sources")
    sources = {
        "Kaggle Aerospace SC": "Supplier KPIs, lead times, OTD%",
        "BTS On-Time Data":    "Maintenance-related delays",
        "FAA ASRS":            "Safety incident correlations",
        "supplychainpy":       "Synthetic MRO inventory (500 SKUs)",
        "OpenSky Network":     "Aircraft utilisation signals",
        "World Bank LPI":      "Country logistics risk scores",
    }
    for src, desc in sources.items():
        st.markdown(f"- **{src}** — {desc}")

st.divider()

# ── Sankey diagram: supply chain material flow ────────────────────────────────
st.markdown("### 🔀 Supply Chain Material Flow (Sankey)")

sankey_fig = go.Figure(go.Sankey(
    node=dict(
        pad=20, thickness=16, line=dict(color="#1e2d4a", width=0.5),
        label=["Boeing", "Airbus", "GE Aviation", "Pratt & Whitney", "Rolls-Royce",
               "Collins\nAerospace", "Honeywell", "Parker\nHannifin", "Safran",
               "Heico Corp", "Precision\nCastparts",
               "Lufthansa\nTechnik", "Air France\nKLM E&M", "Delta\nTechOps",
               "Emirates\nEngineering", "ST Engineering",
               "Delta Air Lines", "Emirates", "Lufthansa\nGroup",
               "Singapore\nAirlines", "IndiGo"],
        color=["#8b5cf6"]*5 + ["#0ea5e9"]*5 + ["#06b6d4"]*1 +
              ["#10b981"]*5 + ["#f59e0b"]*5,
        x=[0.0]*5 + [0.3]*5 + [0.3]*1 + [0.65]*5 + [1.0]*5,
        y=[0.1, 0.3, 0.5, 0.7, 0.9,
           0.1, 0.3, 0.5, 0.7, 0.15, 0.55,
           0.1, 0.3, 0.5, 0.7, 0.9,
           0.1, 0.3, 0.5, 0.7, 0.9],
    ),
    link=dict(
        source=[0,0,1,1,2,2,3,4, 5,6,7,8,9,10, 11,12,13,14,15],
        target=[5,7,5,8,9,10,9,10, 11,11,11,12,9,11, 16,18,16,17,19],
        value =[8,5,6,7,4, 5,4,6,  7,5,6, 5,8, 4,  9, 7, 8, 9, 6],
        color =["rgba(14,165,233,0.3)"]*8 + ["rgba(6,182,212,0.3)"]*6 +
               ["rgba(16,185,129,0.3)"]*5,
    )
))
sankey_fig.update_layout(
    plot_bgcolor="#0d1530", paper_bgcolor="#0d1530",
    font=dict(color="#e2e8f0", size=11),
    height=380, margin=dict(l=20, r=20, t=10, b=10),
)
st.plotly_chart(sankey_fig, use_container_width=True)
