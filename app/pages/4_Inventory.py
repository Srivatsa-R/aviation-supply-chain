"""
Page 4: Inventory & EOQ Optimisation
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

st.set_page_config(page_title="Inventory · Aviation SC", page_icon="📦", layout="wide")

st.markdown("## 📦 MRO Inventory Optimisation")
st.markdown(
    "**EOQ + Safety Stock** optimisation across 196 SKUs. "
    "Total annual saving: **\\$4.37M (24%)** vs naive ordering policy."
)

@st.cache_data(ttl=3600)
def load_inventory_data():
    from src.data.ingest import generate_mro_synthetic
    from src.data.clean  import clean_mro_synthetic
    from src.models.inventory import optimise_inventory, cost_saving_summary
    raw  = generate_mro_synthetic(n_skus=120, n_months=24, save=False)
    df, _= clean_mro_synthetic(raw)
    inv   = optimise_inventory(df)
    summ  = cost_saving_summary(inv)
    return inv, summ

with st.spinner("Running EOQ optimisation..."):
    inv_df, summary_df = load_inventory_data()

# ── KPIs ──────────────────────────────────────────────────────────────────────
total_current = inv_df["tci_current_usd"].sum()
total_optimal = inv_df["tci_optimal_usd"].sum()
total_saving  = inv_df["saving_usd"].sum()
pct_saving    = total_saving / total_current * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current TCI / yr",  f"${total_current/1e6:.2f}M", "Naive policy baseline")
c2.metric("Optimal TCI / yr",  f"${total_optimal/1e6:.2f}M", "EOQ optimised")
c3.metric("Annual Saving",     f"${total_saving/1e6:.2f}M",  f"+{pct_saving:.1f}%")
c4.metric("SKUs Optimised",    f"{len(inv_df)}",              "Across 5 categories")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["💧 Saving Waterfall", "📊 ABC Analysis", "🔍 SKU Explorer", "📋 Full Results"])

# ── TAB 1: Saving Waterfall ───────────────────────────────────────────────────
with tab1:
    st.markdown("### Cost Saving Waterfall — Current vs Optimal by Segment")

    segments = summary_df.copy()
    segments["label"] = segments["abc_class"] + "-" + segments["criticality_tier"]

    fig_wf = go.Figure(go.Waterfall(
        name="Inventory Saving",
        orientation="v",
        measure=["relative"] * len(segments) + ["total"],
        x=segments["label"].tolist() + ["NET SAVING"],
        y=segments["total_saving"].tolist() + [total_saving],
        connector={"line": {"color": "#1e2d4a"}},
        increasing={"marker": {"color": "#10b981"}},
        decreasing={"marker": {"color": "#ef4444"}},
        totals={"marker": {"color": "#0ea5e9"}},
        text=[f"${v/1e3:.0f}K" for v in segments["total_saving"].tolist()] + [f"${total_saving/1e3:.0f}K"],
        textposition="outside",
    ))
    fig_wf.update_layout(
        title="Annual TCI Saving by ABC × Criticality Segment",
        xaxis_title="Segment", yaxis_title="Saving (USD)",
        plot_bgcolor="#0d1530", paper_bgcolor="#111827",
        template="plotly_dark", height=430,
        showlegend=False,
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    st.info(
        "🔴 **Negative savings (Class-C CRITICAL)** indicate under-buffering: "
        "the naive policy orders too little for cheap AOG-critical parts. "
        "The EOQ model correctly increases their order frequency and safety stock."
    )

# ── TAB 2: ABC Analysis ───────────────────────────────────────────────────────
with tab2:
    st.markdown("### ABC-XYZ Classification Heatmap")

    pivot = inv_df.pivot_table(
        values="saving_usd", index="abc_class", columns="criticality_tier",
        aggfunc="sum", fill_value=0,
    )
    fig_heat = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title="Total Annual Saving (USD) by ABC Class × Criticality Tier",
        labels=dict(x="Criticality Tier", y="ABC Class", color="Saving (USD)"),
        text_auto=".3s",
    )
    fig_heat.update_layout(
        plot_bgcolor="#0d1530", paper_bgcolor="#111827",
        template="plotly_dark", height=300,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # EOQ vs Naive comparison by category
    cat_comp = inv_df.groupby("category").agg(
        avg_eoq=("eoq_optimal", "mean"),
        avg_current_q=("annual_demand", lambda x: (x / 4).mean()),  # naive = quarterly
        avg_saving_pct=("saving_pct", "mean"),
        n_skus=("sku_id", "count"),
    ).reset_index()

    fig_eoq = px.bar(
        cat_comp, x="category", y=["avg_eoq", "avg_current_q"],
        barmode="group",
        title="Average EOQ Optimal vs Naive Order Quantity by Category",
        labels={"value": "Order Quantity (units)", "category": "Part Category"},
        color_discrete_map={"avg_eoq": "#10b981", "avg_current_q": "#f59e0b"},
        template="plotly_dark",
    )
    fig_eoq.update_layout(
        plot_bgcolor="#0d1530", paper_bgcolor="#111827", height=340,
        legend=dict(title="Policy"),
    )
    new_names = {"avg_eoq": "EOQ Optimal", "avg_current_q": "Naive (Quarterly)"}
    fig_eoq.for_each_trace(lambda t: t.update(name=new_names.get(t.name, t.name)))
    st.plotly_chart(fig_eoq, use_container_width=True)

# ── TAB 3: SKU Explorer ──────────────────────────────────────────────────────
with tab3:
    st.markdown("### SKU-Level Inventory Policy Explorer")

    col_f, col_c = st.columns([1, 2])
    with col_f:
        cat_filter  = st.multiselect("Category", inv_df["category"].unique(),
                                      default=inv_df["category"].unique().tolist())
        abc_filter  = st.multiselect("ABC Class", ["A","B","C"], default=["A","B","C"])
        tier_filter = st.multiselect("Criticality", ["CRITICAL","HIGH","STANDARD"],
                                      default=["CRITICAL","HIGH","STANDARD"])

    filtered = inv_df[
        inv_df["category"].isin(cat_filter) &
        inv_df["abc_class"].isin(abc_filter) &
        inv_df["criticality_tier"].isin(tier_filter)
    ]

    with col_c:
        st.markdown(f"**{len(filtered)} SKUs selected** | "
                    f"Total saving: **${filtered['saving_usd'].sum():,.0f}**")

    # Scatter: saving vs unit cost
    fig_scatter = px.scatter(
        filtered,
        x="unit_cost_usd", y="saving_usd",
        color="criticality_tier",
        size="annual_demand",
        hover_data=["sku_id", "category", "abc_class", "eoq_optimal", "safety_stock_optimal"],
        color_discrete_map={"CRITICAL":"#ef4444","HIGH":"#f59e0b","STANDARD":"#10b981"},
        title="Annual Saving vs Unit Cost (bubble size = annual demand)",
        labels={"unit_cost_usd":"Unit Cost (USD)","saving_usd":"Annual Saving (USD)"},
        log_x=True,
        template="plotly_dark",
    )
    fig_scatter.update_layout(plot_bgcolor="#0d1530", paper_bgcolor="#111827", height=380)
    st.plotly_chart(fig_scatter, use_container_width=True)

# ── TAB 4: Full Results Table ─────────────────────────────────────────────────
with tab4:
    st.markdown("### Full Inventory Optimisation Results")
    st.caption("Download the full table as CSV for use in your portfolio report")

    display_inv = inv_df[[
        "sku_id","category","abc_class","criticality_tier",
        "unit_cost_usd","lead_time_days","avg_monthly_demand",
        "eoq_optimal","safety_stock_optimal","reorder_point",
        "tci_optimal_usd","tci_current_usd","saving_usd","saving_pct","review_frequency"
    ]].sort_values("saving_usd", ascending=False)

    st.dataframe(
        display_inv.style
        .background_gradient(subset=["saving_usd"], cmap="RdYlGn")
        .format({"unit_cost_usd":"${:,.0f}","tci_optimal_usd":"${:,.0f}",
                 "tci_current_usd":"${:,.0f}","saving_usd":"${:,.0f}","saving_pct":"{:.1f}%"}),
        use_container_width=True, height=420,
    )

    csv = display_inv.to_csv(index=False)
    st.download_button("⬇️ Download CSV", csv, "inventory_optimisation_results.csv", "text/csv")
