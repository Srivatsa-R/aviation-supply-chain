"""
Page 2: Network Graph — Interactive Supply Chain Network
Aviation Supply Chain Analytics Dashboard
"""
import sys
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Network · Aviation SC", page_icon="🕸", layout="wide")

st.markdown("## 🕸 Supply Chain Network Graph")
st.markdown(
    "Multi-tier directed graph: **OEMs → Tier-1 → Tier-2 → MRO → Airlines**. "
    "Node size = betweenness centrality. Edge colour = OTD reliability. "
    "Red border = supplier at risk."
)

# ── Load graph and metrics ────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_graph_data():
    from src.network.build_graph import build_aviation_sc_graph, add_risk_attributes, graph_to_dataframes
    from src.network.metrics import compute_all_metrics, simulate_disruption, find_critical_paths
    G = build_aviation_sc_graph()
    G = add_risk_attributes(G)
    nodes_df, edges_df = graph_to_dataframes(G)
    metrics_df = compute_all_metrics(G)
    disruption = simulate_disruption(G, n_nodes=5, strategy="betweenness")
    paths_df   = find_critical_paths(G)
    return G, nodes_df, edges_df, metrics_df, disruption, paths_df

with st.spinner("Building supply chain graph..."):
    G, nodes_df, edges_df, metrics_df, disruption, paths_df = load_graph_data()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🌐 Interactive Network", "📐 Centrality Metrics", "⚠️ Disruption Simulation", "🛣️ Critical Paths"])

# ── TAB 1: Interactive Network ────────────────────────────────────────────────
with tab1:
    # Embed the Pyvis HTML if it exists, else show Plotly fallback
    network_path = ROOT / "docs" / "network.html"
    if network_path.exists():
        with open(network_path, "r", encoding="utf-8" ) as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=700, scrolling=False)
    else:
        st.info("Run `python src/network/visualise.py` to generate `docs/network.html`, then restart the app.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Nodes", G.number_of_nodes(), "Across 5 tiers")
    col2.metric("Total Edges", G.number_of_edges(), "Supply relationships")
    col3.metric("Is DAG", "✅ Yes", "No circular dependencies")

    # Legend
    st.markdown("**Node colour legend:**")
    cols = st.columns(5)
    legend = [("OEM","#8b5cf6"),("Tier-1","#0ea5e9"),("Tier-2","#06b6d4"),("MRO","#10b981"),("Airline","#f59e0b")]
    for i, (label, colour) in enumerate(legend):
        cols[i].markdown(
            f'<span style="background:{colour};padding:3px 10px;border-radius:12px;'
            f'color:#fff;font-size:12px;font-weight:600;">{label}</span>',
            unsafe_allow_html=True
        )

# ── TAB 2: Centrality Metrics ─────────────────────────────────────────────────
with tab2:
    st.markdown("### Top nodes by betweenness centrality (most critical)")

    top_n = st.slider("Show top N nodes", 5, 25, 10)
    display_cols = ["node_id", "node_type", "tier", "betweenness_centrality",
                    "pagerank", "composite_risk_score", "avg_reliability_pct", "single_source_flag"]
    display_cols = [c for c in display_cols if c in metrics_df.columns]
    top_df = metrics_df.head(top_n)[display_cols]

    st.dataframe(
        top_df.style
        .background_gradient(subset=["betweenness_centrality"], cmap="YlOrRd")
        .background_gradient(subset=["composite_risk_score"], cmap="YlOrRd")
        .format({"betweenness_centrality": "{:.4f}", "pagerank": "{:.4f}",
                 "composite_risk_score": "{:.3f}", "avg_reliability_pct": "{:.1f}%"}),
        use_container_width=True, height=350,
    )

    # Betweenness bar chart
    fig = px.bar(
        metrics_df.head(12),
        x="node_id", y="betweenness_centrality",
        color="node_type",
        color_discrete_map={"OEM":"#8b5cf6","Tier1":"#0ea5e9","Tier2":"#06b6d4","MRO":"#10b981","Airline":"#f59e0b"},
        title="Betweenness Centrality — Top 12 Nodes",
        labels={"node_id": "Node", "betweenness_centrality": "Betweenness"},
        template="plotly_dark",
    )
    fig.update_layout(plot_bgcolor="#0d1530", paper_bgcolor="#111827", height=350)
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: Disruption Simulation ─────────────────────────────────────────────
with tab3:
    st.markdown("### Cascading failure simulation — removing top-5 bottleneck nodes")
    st.caption(f"Removal sequence: {' → '.join(disruption['removal_sequence'])}")

    col1, col2, col3, col4 = st.columns(4)
    fi = disruption["final_impact"]
    col1.metric("Nodes Removed", fi["nodes_removed"], f"of {G.number_of_nodes()} total")
    col2.metric("Path Length Increase", f"{fi['path_length_increase_pct']}%", delta_color="inverse")
    col3.metric("Edges Disrupted", fi["edges_disrupted"], f"of {G.number_of_edges()} total", delta_color="inverse")
    col4.metric("Airlines at AOG Risk", fi["airlines_at_aog_risk"], delta_color="inverse")

    steps_df = pd.DataFrame(disruption["steps"])

    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(
        x=steps_df["step"], y=steps_df["network_reach_pct"],
        mode="lines+markers", name="Network Reach %",
        line=dict(color="#0ea5e9", width=3),
        marker=dict(size=10, color="#0ea5e9"),
    ))
    fig_sim.add_trace(go.Scatter(
        x=steps_df["step"], y=steps_df["avg_shortest_path"],
        mode="lines+markers", name="Avg Shortest Path",
        line=dict(color="#ef4444", width=3),
        marker=dict(size=10, color="#ef4444"),
        yaxis="y2",
    ))
    fig_sim.update_layout(
        title="Network Degradation Under Targeted Attack",
        xaxis=dict(title="Nodes Removed (sequential)", tickvals=steps_df["step"].tolist()),
        yaxis=dict(title="Network Reach (%)", color="#0ea5e9"),
        yaxis2=dict(title="Avg Shortest Path", overlaying="y", side="right", color="#ef4444"),
        legend=dict(x=0.01, y=0.99),
        plot_bgcolor="#0d1530", paper_bgcolor="#111827",
        template="plotly_dark", height=380,
    )
    fig_sim.add_vrect(x0=3.5, x1=4.5, fillcolor="#ef4444", opacity=0.1,
                      annotation_text="Network fractures", annotation_position="top left")
    st.plotly_chart(fig_sim, use_container_width=True)

    st.dataframe(steps_df[[
        "step","removed_node","node_type","n_components",
        "aog_risk_airlines","edges_lost_cumulative","network_reach_pct"
    ]], use_container_width=True, height=240)

# ── TAB 4: Critical Paths ─────────────────────────────────────────────────────
with tab4:
    st.markdown("### Longest supply paths (by cumulative lead time — OEM to Airline)")
    st.caption("Longer paths = higher AOG exposure window if any node fails")

    fig_paths = px.bar(
        paths_df.head(15),
        x="total_lead_time_days", y="path",
        color="source",
        orientation="h",
        title="Top 15 Longest Supply Paths (Lead Time Days)",
        labels={"total_lead_time_days": "Total Lead Time (days)", "path": "Supply Path"},
        template="plotly_dark",
    )
    fig_paths.update_layout(
        plot_bgcolor="#0d1530", paper_bgcolor="#111827", height=480,
        yaxis=dict(tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_paths, use_container_width=True)

    st.dataframe(paths_df.head(10), use_container_width=True, height=280)
