"""
metrics.py — Aviation Supply Chain: Graph Metrics & Disruption Analysis
========================================================================
Phase 3 · Network Modelling & Analysis

PURPOSE
-------
Computes all graph-theoretic metrics needed to:
  1. Rank the most critical supply chain nodes (centrality)
  2. Identify structural bottlenecks
  3. Simulate cascading failures (node removal attacks)
  4. Measure network resilience before and after disruptions

KEY CONCEPTS USED
-----------------
Betweenness Centrality
  - Fraction of all shortest paths that pass THROUGH a node
  - HIGH betweenness = bottleneck supplier = single point of failure
  - Source: research shows airports/nodes with high betweenness are
    "major connecting hubs" whose failure causes disproportionate damage

PageRank
  - Measures how many important nodes point TO you
  - HIGH pagerank in supply chain = you are depended on by major players
  - Originally Google's algorithm; works perfectly for supply chain influence

Degree Centrality
  - How many direct connections does a node have?
  - HIGH degree = broad supplier / customer base = more redundancy

Cascading Failure Simulation
  - Based on academic literature (2024-2025): disruptions in upstream tiers
    cascade differently than downstream disruptions
  - We simulate by removing the top-N highest-betweenness nodes one at a time
    and measuring how network connectivity degrades each time

USAGE
-----
    from src.network.metrics import compute_all_metrics, simulate_disruption
    metrics_df = compute_all_metrics(G)
    results    = simulate_disruption(G, n_nodes=5)

AUTHOR  : Aviation SC Analytics Project
VERSION : 1.0.0 (Phase 3)
"""

import logging
import numpy as np
import pandas as pd
import networkx as nx
from typing import Any

log = logging.getLogger("aviation_sc.network.metrics")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CENTRALITY METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(G: nx.DiGraph) -> pd.DataFrame:
    """
    Compute all centrality metrics for every node in the supply chain graph.

    Metrics computed
    ----------------
    betweenness_centrality : how often is this node on the shortest path?
    in_degree_centrality   : how many suppliers does this node have?
    out_degree_centrality  : how many customers does this node serve?
    pagerank               : how influential is this node in the network?
    closeness_centrality   : how quickly can this node reach all others?

    Parameters
    ----------
    G : nx.DiGraph  The aviation supply chain graph

    Returns
    -------
    pd.DataFrame
        One row per node with all metrics plus node attributes.
        Sorted descending by betweenness_centrality (most critical first).
    """
    log.info(f"📐 Computing centrality metrics for {G.number_of_nodes()} nodes...")

    # Betweenness: fraction of shortest paths through this node
    # normalize=True divides by (n-1)(n-2) so values are in [0,1]
    betweenness = nx.betweenness_centrality(G, normalized=True, weight="lead_time_days")

    # Degree centrality (in and out separately — directed graph)
    in_degree  = nx.in_degree_centrality(G)
    out_degree = nx.out_degree_centrality(G)

    # PageRank: measures influence propagation through the supply chain
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=200)

    # Closeness: average distance to all other nodes (using undirected version)
    # Directed closeness can be 0 for many nodes; undirected gives better intuition
    G_undirected = G.to_undirected()
    closeness = nx.closeness_centrality(G_undirected)

    # Assemble into a DataFrame
    records = []
    for node in G.nodes():
        attrs = G.nodes[node]
        records.append({
            "node_id":               node,
            "tier":                  attrs.get("tier", -1),
            "node_type":             attrs.get("type", "Unknown"),
            "country":               attrs.get("country", "N/A"),
            "betweenness_centrality":round(betweenness.get(node, 0), 4),
            "in_degree_centrality":  round(in_degree.get(node, 0), 4),
            "out_degree_centrality": round(out_degree.get(node, 0), 4),
            "pagerank":              round(pagerank.get(node, 0), 4),
            "closeness_centrality":  round(closeness.get(node, 0), 4),
            "avg_reliability_pct":   attrs.get("avg_reliability_pct", 100.0),
            "avg_lead_time_days":    attrs.get("avg_lead_time_days", 0.0),
            "supplier_at_risk":      attrs.get("supplier_at_risk", False),
            "single_source_flag":    attrs.get("single_source_flag", False),
            "revenue_bn_usd":        attrs.get("revenue_bn", None),
        })

    df = pd.DataFrame(records)

    # Composite risk score: combines betweenness + low reliability + single-source flag
    df["composite_risk_score"] = (
        df["betweenness_centrality"] * 0.45
        + (1 - df["avg_reliability_pct"] / 100) * 0.35
        + df["single_source_flag"].astype(float) * 0.20
    ).round(4)

    df = df.sort_values("betweenness_centrality", ascending=False).reset_index(drop=True)

    log.info(f"✅ Metrics computed. Top bottleneck: {df.iloc[0]['node_id']} "
             f"(betweenness={df.iloc[0]['betweenness_centrality']:.4f})")
    return df


def get_top_bottlenecks(
    metrics_df: pd.DataFrame,
    n: int = 5,
    metric: str = "betweenness_centrality",
) -> pd.DataFrame:
    """
    Return the top-N most critical nodes by a chosen metric.

    Parameters
    ----------
    metrics_df : pd.DataFrame  Output of compute_all_metrics()
    n          : int           Number of top nodes to return (default 5)
    metric     : str           Column to rank by

    Returns
    -------
    pd.DataFrame  Subset of metrics_df with just the top-N rows
    """
    return (
        metrics_df
        .sort_values(metric, ascending=False)
        .head(n)
        [["node_id", "node_type", "tier", metric, "composite_risk_score",
          "avg_reliability_pct", "single_source_flag"]]
        .reset_index(drop=True)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DISRUPTION SIMULATION — CASCADING FAILURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_disruption(
    G: nx.DiGraph,
    n_nodes: int = 5,
    strategy: str = "betweenness",
) -> dict[str, Any]:
    """
    Simulate cascading supply chain failures by progressively removing
    the most critical nodes and measuring how the network degrades.

    This models a real-world scenario: a geopolitical event, factory fire,
    or insolvency that takes out the most-connected supplier nodes one by one.

    Methodology (based on academic network resilience literature 2024-2025)
    -----------------------------------------------------------------------
    - Targeted attack: remove nodes in order of betweenness/PageRank
    - Random failure: remove nodes randomly (for comparison)
    - After each removal, measure:
        * Number of weakly connected components (should stay = 1)
        * Average shortest path length (should stay low)
        * Number of airlines that lose ALL supply paths (AOG risk)
        * Fraction of graph still reachable

    Parameters
    ----------
    G        : nx.DiGraph  The aviation supply chain graph
    n_nodes  : int         How many nodes to remove sequentially (default 5)
    strategy : str         "betweenness" (targeted) or "random"

    Returns
    -------
    dict with keys:
        "removal_sequence"  : list of removed node names (in order)
        "steps"             : list of dicts — one per removal step
        "baseline"          : baseline network stats before any removal
        "final_impact"      : summary impact after removing all n_nodes
    """
    log.info(f"⚠️  Simulating disruption: removing top {n_nodes} nodes by {strategy}")

    # Baseline measurements
    G_base = G.copy()
    G_ud   = G_base.to_undirected()

    baseline = {
        "n_nodes":               G_base.number_of_nodes(),
        "n_edges":               G_base.number_of_edges(),
        "n_components":          nx.number_weakly_connected_components(G_base),
        "avg_shortest_path":     _safe_avg_path(G_ud),
        "graph_density":         round(nx.density(G_base), 4),
        "airlines_fully_served": _count_fully_served_airlines(G_base),
    }

    # Determine removal order
    if strategy == "betweenness":
        scores = nx.betweenness_centrality(G_base, normalized=True,
                                            weight="lead_time_days")
        # Only remove non-airline, non-OEM nodes (supply chain intermediaries)
        removal_candidates = [
            n for n, d in G_base.nodes(data=True)
            if d.get("type") not in ("Airline", "OEM")
        ]
        removal_sequence = sorted(
            removal_candidates,
            key=lambda n: scores.get(n, 0),
            reverse=True
        )[:n_nodes]
    else:  # random
        import random
        candidates = [
            n for n, d in G_base.nodes(data=True)
            if d.get("type") not in ("Airline", "OEM")
        ]
        removal_sequence = random.sample(candidates, min(n_nodes, len(candidates)))

    # Progressive removal simulation
    G_sim = G.copy()
    steps = []

    for i, node in enumerate(removal_sequence):
        G_sim.remove_node(node)
        G_ud_sim = G_sim.to_undirected()

        n_components = nx.number_weakly_connected_components(G_sim)
        avg_path     = _safe_avg_path(G_ud_sim)
        served       = _count_fully_served_airlines(G_sim)
        aog_risk     = baseline["airlines_fully_served"] - served
        edges_lost   = G_base.number_of_edges() - G_sim.number_of_edges()
        reach_pct    = round(G_sim.number_of_nodes() / G_base.number_of_nodes() * 100, 1)

        step = {
            "step":            i + 1,
            "removed_node":    node,
            "node_type":       G.nodes[node].get("type", "?"),
            "n_components":    n_components,
            "avg_shortest_path": avg_path,
            "airlines_served": served,
            "aog_risk_airlines": aog_risk,
            "edges_lost_cumulative": edges_lost,
            "network_reach_pct": reach_pct,
        }
        steps.append(step)
        log.info(f"   Step {i+1}: removed {node} → "
                 f"components={n_components}, AOG risk={aog_risk} airlines")

    final_impact = {
        "nodes_removed":          n_nodes,
        "strategy":               strategy,
        "components_delta":       steps[-1]["n_components"] - baseline["n_components"],
        "path_length_increase_pct": round(
            (steps[-1]["avg_shortest_path"] - baseline["avg_shortest_path"])
            / max(baseline["avg_shortest_path"], 0.001) * 100, 1
        ),
        "airlines_at_aog_risk":   steps[-1]["aog_risk_airlines"],
        "edges_disrupted":        steps[-1]["edges_lost_cumulative"],
        "network_reach_remaining_pct": steps[-1]["network_reach_pct"],
    }

    log.info(f"✅ Disruption simulation complete: "
             f"{final_impact['airlines_at_aog_risk']} airlines at AOG risk")

    return {
        "removal_sequence": removal_sequence,
        "steps":            steps,
        "baseline":         baseline,
        "final_impact":     final_impact,
    }


def _safe_avg_path(G_undirected: nx.Graph) -> float:
    """Average shortest path length — handles disconnected graphs gracefully."""
    if G_undirected.number_of_nodes() < 2:
        return 0.0
    try:
        # Use largest component if graph is disconnected
        largest = max(nx.connected_components(G_undirected), key=len)
        sub     = G_undirected.subgraph(largest)
        return round(nx.average_shortest_path_length(sub), 3)
    except Exception:
        return 0.0


def _count_fully_served_airlines(G: nx.DiGraph) -> int:
    """Count airlines that still have at least one upstream supply path."""
    airlines = [n for n, d in G.nodes(data=True) if d.get("type") == "Airline"]
    count = 0
    for airline in airlines:
        # Check if ANY predecessor (indirect) still exists
        try:
            ancestors = nx.ancestors(G, airline)
            if ancestors:
                count += 1
        except Exception:
            pass
    return count


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SHORTEST PATH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def find_critical_paths(
    G: nx.DiGraph,
    source_types: list[str] = ["OEM"],
    target_types: list[str] = ["Airline"],
    weight: str = "lead_time_days",
) -> pd.DataFrame:
    """
    Find the longest/riskiest paths from OEMs to Airlines.

    In supply chain terms: which OEM → Airline route has the most total
    lead time? That's where AOG exposure is highest.

    Parameters
    ----------
    G            : nx.DiGraph
    source_types : node types to treat as source (default: OEM)
    target_types : node types to treat as target (default: Airline)
    weight       : edge attribute to minimise (default: lead_time_days)

    Returns
    -------
    pd.DataFrame  Columns: source, target, path, total_lead_time, n_hops
    """
    sources = [n for n, d in G.nodes(data=True) if d.get("type") in source_types]
    targets = [n for n, d in G.nodes(data=True) if d.get("type") in target_types]

    records = []
    for src in sources:
        for tgt in targets:
            try:
                path   = nx.shortest_path(G, src, tgt, weight=weight)
                length = nx.shortest_path_length(G, src, tgt, weight=weight)
                records.append({
                    "source":         src,
                    "target":         tgt,
                    "path":           " → ".join(path),
                    "total_lead_time_days": length,
                    "n_hops":         len(path) - 1,
                })
            except nx.NetworkXNoPath:
                pass

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("total_lead_time_days", ascending=False).reset_index(drop=True)
    return df


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parents[2]))
    from src.network.build_graph import build_aviation_sc_graph, add_risk_attributes

    G = build_aviation_sc_graph()
    G = add_risk_attributes(G)

    print("\n=== CENTRALITY METRICS ===")
    metrics = compute_all_metrics(G)
    print(metrics[["node_id","node_type","betweenness_centrality",
                   "pagerank","composite_risk_score"]].head(10).to_string(index=False))

    print("\n=== TOP 5 BOTTLENECKS ===")
    print(get_top_bottlenecks(metrics, n=5).to_string(index=False))

    print("\n=== CRITICAL PATHS (OEM → Airline) ===")
    paths = find_critical_paths(G)
    print(paths.head(5).to_string(index=False))

    print("\n=== DISRUPTION SIMULATION (remove top 3 nodes) ===")
    result = simulate_disruption(G, n_nodes=3, strategy="betweenness")
    print(f"Removed: {result['removal_sequence']}")
    print(f"Final impact: {result['final_impact']}")
    for step in result["steps"]:
        print(f"  Step {step['step']}: -{step['removed_node']} → "
              f"AOG risk={step['aog_risk_airlines']} airlines, "
              f"components={step['n_components']}")

    print("\n✅ metrics.py self-test passed.")
