"""
Page 5: Supplier Risk & Route Optimisation
Aviation Supply Chain Analytics Dashboard
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Risk · Aviation SC", page_icon="⚠️", layout="wide")

st.markdown("## ⚠️ Supplier Risk Scoring & Route Optimisation")
st.markdown(
    "**6-factor rule-based risk score** (0–100) + **Random Forest** validation. "
    "LP route optimisation delivers **\\$13.1M saving (60.7%)** vs current routing policy."
)

@st.cache_data(ttl=3600)
def load_risk_data():
    from src.network.build_graph import build_aviation_sc_graph, add_risk_attributes
    from src.models.risk_score   import score_all_suppliers
    from src.models.optimisation import build_transport_problem, solve_lp, compare_with_current_policy
    G = build_aviation_sc_graph()
    G = add_risk_attributes(G)
    risk     = score_all_suppliers(G)
    problem  = build_transport_problem()
    lp_res   = solve_lp(problem)
    comp_lp  = compare_with_current_policy(lp_res)
    return risk, lp_res, comp_lp

with st.spinner("Running risk scoring and LP optimisation..."):
    risk_res, lp_res, comp_lp = load_risk_data()

scores_df = risk_res["supplier_scores"]
rf        = risk_res["rf_result"]

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Suppliers Scored",   len(scores_df))
c2.metric("HIGH/CRITICAL Risk", len(scores_df[scores_df["risk_tier"].isin(["HIGH","CRITICAL"])]))
c3.metric("RF Accuracy (OOB)",  f"{rf['oob_accuracy']:.0%}")
c4.metric("LP Optimal Cost",    f"${comp_lp['optimal_cost']/1e6:.2f}M")
c5.metric("Route Saving",       f"${comp_lp['absolute_saving']/1e6:.1f}M",
          f"+{comp_lp['saving_pct']}%")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["🏷️ Risk Scores", "🤖 RF Analysis", "🛣️ LP Routing", "📋 Summary"])

# ── TAB 1: Risk Scores ────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Supplier Risk Dashboard")

    col_l, col_r = st.columns([3, 2])

    with col_l:
        # Risk score bar chart
        tier_colors = {"CRITICAL":"#ef4444","HIGH":"#f59e0b","MEDIUM":"#0ea5e9","LOW":"#10b981"}
        fig_risk = px.bar(
            scores_df.head(20),
            x="risk_score", y="supplier",
            color="risk_tier", orientation="h",
            color_discrete_map=tier_colors,
            title="Supplier Risk Scores (Top 20)",
            labels={"risk_score":"Composite Risk Score (0–100)","supplier":"Supplier"},
            template="plotly_dark",
        )
        fig_risk.add_vline(x=50, line_dash="dash", line_color="#f59e0b", line_width=1.5,
                           annotation_text="HIGH threshold", annotation_position="top")
        fig_risk.add_vline(x=75, line_dash="dash", line_color="#ef4444", line_width=1.5,
                           annotation_text="CRITICAL threshold", annotation_position="top")
        fig_risk.update_layout(
            plot_bgcolor="#0d1530", paper_bgcolor="#111827",
            yaxis=dict(autorange="reversed"), height=480,
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_r:
        # Risk tier donut
        tier_counts = scores_df["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["tier", "count"]
        fig_donut = px.pie(
            tier_counts, values="count", names="tier",
            color="tier", color_discrete_map=tier_colors,
            hole=0.55, title="Risk Tier Distribution",
            template="plotly_dark",
        )
        fig_donut.update_layout(
            plot_bgcolor="#0d1530", paper_bgcolor="#111827", height=260,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        # Risk factor radar for top risk supplier
        top_supplier = scores_df.iloc[0]["supplier"]
        st.markdown(f"**Risk breakdown — {top_supplier}:**")
        top_row = scores_df.iloc[0]

        factor_scores = {
            "OTD":             round((100 - top_row.get("on_time_delivery_pct", 90)) / 30 * 100, 1),
            "Lead Time":       round(max(0, (top_row.get("avg_lead_time_days", 30) - 14) / 76 * 100), 1),
            "Single Source":   top_row.get("single_source_flag", 0) * 100,
            "Lead Variability":round(min(100, top_row.get("lead_time_variability", 0) / 30 * 100), 1),
            "Financial":       round(max(0, (10 - min(10, top_row.get("revenue_bn", 1))) / 10 * 100), 1),
        }
        categories = list(factor_scores.keys())
        values     = list(factor_scores.values())

        fig_radar = go.Figure(go.Scatterpolar(
            r=values + [values[0]], theta=categories + [categories[0]],
            fill="toself", name=top_supplier,
            line_color="#ef4444", fillcolor="rgba(239,68,68,0.15)",
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(range=[0,100], visible=True, color="#64748b"),
                bgcolor="#0d1530",
            ),
            plot_bgcolor="#0d1530", paper_bgcolor="#111827",
            showlegend=False, height=260,
            title=f"Risk Factor Radar — {top_supplier}",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Full table
    st.markdown("### Full Supplier Risk Table")
    st.dataframe(
        scores_df.style
        .background_gradient(subset=["risk_score"], cmap="YlOrRd")
        .format({"risk_score":"{:.1f}","on_time_delivery_pct":"{:.1f}%",
                 "rf_risk_probability":"{:.3f}"} if "rf_risk_probability" in scores_df else {}),
        use_container_width=True, height=340,
    )

# ── TAB 2: RF Analysis ────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Random Forest — Feature Importance & Model Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fi_df = rf["feature_importance"]
        fig_fi = px.bar(
            fi_df, x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale="Oranges",
            title="RF Feature Importance (Gini Impurity Reduction)",
            template="plotly_dark",
        )
        fig_fi.update_layout(
            plot_bgcolor="#0d1530", paper_bgcolor="#111827",
            yaxis=dict(autorange="reversed"), coloraxis_showscale=False, height=360,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with col2:
        # RF vs rule-based comparison
        pred_df = rf["predictions"]
        if "rf_risk_probability" in pred_df.columns and "risk_score" in pred_df.columns:
            fig_vs = px.scatter(
                pred_df,
                x="risk_score", y="rf_risk_probability",
                color="risk_tier",
                color_discrete_map={"CRITICAL":"#ef4444","HIGH":"#f59e0b","MEDIUM":"#0ea5e9","LOW":"#10b981"},
                hover_data=["supplier"] if "supplier" in pred_df.columns else [],
                title="Rule-Based Score vs RF Probability",
                labels={"risk_score":"Rule-Based Score (0-100)","rf_risk_probability":"RF Risk Probability"},
                template="plotly_dark",
            )
            fig_vs.add_hline(y=0.5, line_dash="dash", line_color="#f59e0b", line_width=1.5)
            fig_vs.add_vline(x=50,  line_dash="dash", line_color="#f59e0b", line_width=1.5)
            fig_vs.update_layout(plot_bgcolor="#0d1530", paper_bgcolor="#111827", height=360)
            st.plotly_chart(fig_vs, use_container_width=True)

    col3, col4, col5 = st.columns(3)
    col3.metric("OOB Accuracy",  f"{rf['oob_accuracy']:.1%}")
    col4.metric("CV Accuracy",   f"{rf['cv_accuracy']:.1%}")
    col5.metric("CV Std Dev",    f"±{rf['cv_std']:.1%}")

    st.info(
        f"🔍 **Key insight:** `{fi_df.iloc[0]['feature']}` is the most important RF feature "
        f"(importance={fi_df.iloc[0]['importance']:.3f}). The RF weights lead-time **variability** "
        f"more heavily than the rule-based model — surfacing suppliers with unpredictable lead times "
        f"that traditional scorecards miss."
    )

# ── TAB 3: LP Route Optimisation ─────────────────────────────────────────────
with tab3:
    st.markdown("### LP Route Optimisation Results")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Cost Comparison — Current vs Optimal")
        comp_data = comp_lp["summary_df"]
        fig_bar = px.bar(
            comp_data, x="policy", y="total_cost",
            color="policy",
            color_discrete_map={
                "Current (Historical Routing)": "#ef4444",
                "LP Optimised (Our Model)":     "#10b981",
            },
            title="Total Annual Transportation Cost (USD)",
            labels={"total_cost":"Cost (USD)","policy":"Policy"},
            text_auto=".3s",
            template="plotly_dark",
        )
        fig_bar.update_layout(plot_bgcolor="#0d1530", paper_bgcolor="#111827",
                              height=340, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown("#### Active Shipping Routes (LP Solution)")
        active_flows = lp_res["flows"][lp_res["flows"]["active"]].copy()
        active_flows["route"] = active_flows["supplier"] + "\n→ " + active_flows["mro"]
        fig_routes = px.bar(
            active_flows.sort_values("line_cost", ascending=False),
            x="route", y="line_cost",
            color="lead_days",
            color_continuous_scale="Blues_r",
            title="LP Optimal Routes (Line Cost)",
            labels={"line_cost":"Line Cost (USD)","route":"Route"},
            text_auto=".3s",
            template="plotly_dark",
        )
        fig_routes.update_layout(
            plot_bgcolor="#0d1530", paper_bgcolor="#111827",
            height=340, xaxis_tickangle=-30,
        )
        st.plotly_chart(fig_routes, use_container_width=True)

    # Supplier utilisation
    st.markdown("#### Supplier Capacity Utilisation")
    util_df = lp_res["utilisation"].copy()
    fig_util = px.bar(
        util_df, x="supplier", y="utilisation_pct",
        color="utilisation_pct",
        color_continuous_scale="RdYlGn",
        title="LP Optimal Capacity Utilisation (%)",
        labels={"utilisation_pct":"Utilisation (%)","supplier":"Supplier"},
        template="plotly_dark",
        text_auto=".1f",
    )
    fig_util.add_hline(y=80, line_dash="dash", line_color="#f59e0b", line_width=1.5,
                       annotation_text="80% threshold", annotation_position="top right")
    fig_util.update_layout(plot_bgcolor="#0d1530", paper_bgcolor="#111827",
                           height=330, coloraxis_showscale=False)
    st.plotly_chart(fig_util, use_container_width=True)

    st.dataframe(
        active_flows[["supplier","mro","flow_units","unit_cost","lead_days","line_cost"]],
        use_container_width=True, hide_index=True,
    )

# ── TAB 4: Summary ────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Executive Summary — Risk & Routing")

    summary_points = {
        "🚨 TOP RISK: Pratt & Whitney":
            "Risk score 60.8 (HIGH). Below-benchmark OTD (84.0%) + single-source flag. "
            "Sole engine parts supplier to multiple MROs. Recommend dual-sourcing GTF engine component supply.",
        "⚠️ HIDDEN RISK: GE Aviation":
            "Rule-based score 38.7 (MEDIUM) but RF assigns 0.675 risk probability (near HIGH). "
            "High lead-time variability (σ=15d) signals procurement instability not captured by average OTD alone.",
        "💰 LP ROUTING SAVING: $13.1M (60.7%)":
            "LP concentrates volume on Parker Hannifin (lowest unit cost per route) and Heico Corp. "
            "Collins Aerospace routes eliminated in favour of cheaper alternatives with same lead-time SLA.",
        "🔧 MITIGATION PRIORITIES":
            "1. Dual-source Pratt & Whitney engine parts through MT Aero or additional Tier-1. "
            "2. Implement LP-optimal routing — switchable in 30 days (contractual notice periods). "
            "3. Increase safety stock for Class-C AOG parts (fasteners, seals) — identified in Phase 4.",
    }
    for title, body in summary_points.items():
        with st.expander(title, expanded=True):
            st.markdown(body)

    # Download button
    full_export = pd.concat([
        scores_df.rename(columns={"supplier":"entity"}),
    ], ignore_index=True)
    csv = full_export.to_csv(index=False)
    st.download_button("⬇️ Download Risk Scores CSV", csv,
                       "supplier_risk_scores.csv", "text/csv")
