"""
test_graph.py — Unit Tests: Supply Chain Network Graph
=======================================================
Phase 6 · Polish, Tests & Showcase

Tests cover:
  - Graph construction (correct node/edge count, types, attributes)
  - Graph properties (DAG, connected, no self-loops)
  - Risk attribute enrichment
  - Centrality metrics shape and value bounds
  - Disruption simulation structure and logic
  - Critical path finder

Run with:  pytest tests/test_graph.py -v
           pytest tests/ --cov=src --cov-report=term-missing

PATTERN: Arrange → Act → Assert (AAA) on every test.
NAMING:  test_<what>_<condition>_<expected_outcome>
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import networkx as nx

# ── Make src/ importable ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.network.build_graph import (
    build_aviation_sc_graph,
    add_risk_attributes,
    graph_to_dataframes,
    NODES,
    EDGES,
)
from src.network.metrics import (
    compute_all_metrics,
    get_top_bottlenecks,
    simulate_disruption,
    find_critical_paths,
)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def graph():
    """Build the aviation supply chain graph once for all tests in this module."""
    G = build_aviation_sc_graph()
    return G


@pytest.fixture(scope="module")
def enriched_graph(graph):
    """Graph with risk attributes added."""
    return add_risk_attributes(graph)


@pytest.fixture(scope="module")
def metrics(enriched_graph):
    """Pre-computed centrality metrics."""
    return compute_all_metrics(enriched_graph)


# ══════════════════════════════════════════════════════════════════════════════
# 1. GRAPH CONSTRUCTION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestGraphConstruction:

    def test_graph_has_correct_node_count(self, graph):
        """Graph should have exactly as many nodes as defined in NODES list."""
        assert graph.number_of_nodes() == len(NODES), (
            f"Expected {len(NODES)} nodes, got {graph.number_of_nodes()}"
        )

    def test_graph_has_correct_edge_count(self, graph):
        """Graph should have exactly as many edges as defined in EDGES list."""
        assert graph.number_of_edges() == len(EDGES), (
            f"Expected {len(EDGES)} edges, got {graph.number_of_edges()}"
        )

    def test_graph_is_directed(self, graph):
        """Supply chain graph must be a directed graph (materials flow one-way)."""
        assert isinstance(graph, nx.DiGraph), "Graph must be a directed graph (DiGraph)"

    def test_graph_is_dag(self, graph):
        """Supply chain should have no circular dependencies."""
        assert nx.is_directed_acyclic_graph(graph), (
            "Graph contains cycles — supply chains should not have circular flows"
        )

    def test_all_tier_types_present(self, graph):
        """All 5 node types must be represented: OEM, Tier1, Tier2, MRO, Airline."""
        types = {d.get("type") for _, d in graph.nodes(data=True)}
        expected = {"OEM", "Tier1", "Tier2", "MRO", "Airline"}
        missing = expected - types
        assert not missing, f"Missing node types: {missing}"

    def test_all_nodes_have_tier_attribute(self, graph):
        """Every node must have a 'tier' attribute set."""
        missing = [n for n, d in graph.nodes(data=True) if "tier" not in d]
        assert not missing, f"Nodes missing 'tier' attribute: {missing[:5]}"

    def test_all_nodes_have_type_attribute(self, graph):
        """Every node must have a 'type' attribute set."""
        missing = [n for n, d in graph.nodes(data=True) if "type" not in d]
        assert not missing, f"Nodes missing 'type' attribute: {missing[:5]}"

    def test_all_edges_have_lead_time(self, graph):
        """Every edge must have a 'lead_time_days' attribute (used as path weight)."""
        bad = [(u, v) for u, v, d in graph.edges(data=True)
               if "lead_time_days" not in d]
        assert not bad, f"Edges missing lead_time_days: {bad[:3]}"

    def test_all_edges_have_reliability(self, graph):
        """Every edge must have 'reliability_pct' attribute (used for risk scoring)."""
        bad = [(u, v) for u, v, d in graph.edges(data=True)
               if "reliability_pct" not in d]
        assert not bad, f"Edges missing reliability_pct: {bad[:3]}"

    def test_lead_times_are_positive(self, graph):
        """All lead times must be strictly positive (no instant or negative delivery)."""
        bad = [(u, v, d["lead_time_days"])
               for u, v, d in graph.edges(data=True)
               if d.get("lead_time_days", 0) <= 0]
        assert not bad, f"Non-positive lead times found: {bad[:3]}"

    def test_reliability_within_valid_range(self, graph):
        """OTD reliability must be between 0% and 100% inclusive."""
        bad = [(u, v, d["reliability_pct"])
               for u, v, d in graph.edges(data=True)
               if not (0 <= d.get("reliability_pct", 50) <= 100)]
        assert not bad, f"Reliability out of [0, 100] range: {bad[:3]}"

    def test_no_self_loops(self, graph):
        """A company cannot supply to itself."""
        loops = list(nx.selfloop_edges(graph))
        assert not loops, f"Self-loops found: {loops}"

    def test_known_node_boeing_exists(self, graph):
        """Boeing (an OEM) must be present in the graph."""
        assert "Boeing" in graph.nodes, "Boeing node missing from graph"

    def test_known_edge_boeing_to_safran(self, graph):
        """Boeing → Safran edge (engine nacelle supply) must exist."""
        assert graph.has_edge("Boeing", "Safran"), (
            "Expected Boeing → Safran supply edge not found"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. RISK ATTRIBUTE ENRICHMENT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestRiskAttributes:

    def test_all_nodes_have_avg_reliability(self, enriched_graph):
        """After enrichment, every node must have avg_reliability_pct."""
        missing = [n for n, d in enriched_graph.nodes(data=True)
                   if "avg_reliability_pct" not in d]
        assert not missing, f"Missing avg_reliability_pct: {missing[:5]}"

    def test_avg_reliability_within_range(self, enriched_graph):
        """Average reliability per node must be 0–100."""
        bad = [n for n, d in enriched_graph.nodes(data=True)
               if not (0 <= d.get("avg_reliability_pct", 50) <= 100)]
        assert not bad, f"Avg reliability out of range: {bad[:5]}"

    def test_single_source_flag_is_boolean(self, enriched_graph):
        """single_source_flag must be boolean (True/False), not numeric."""
        bad = [n for n, d in enriched_graph.nodes(data=True)
               if not isinstance(d.get("single_source_flag", False), bool)]
        assert not bad, f"Non-boolean single_source_flag: {bad[:3]}"

    def test_supplier_at_risk_flag_exists(self, enriched_graph):
        """supplier_at_risk flag must be set on every node."""
        missing = [n for n, d in enriched_graph.nodes(data=True)
                   if "supplier_at_risk" not in d]
        assert not missing

    def test_at_risk_nodes_have_low_reliability(self, enriched_graph):
        """Nodes flagged supplier_at_risk must have avg_reliability < 88%."""
        for n, d in enriched_graph.nodes(data=True):
            if d.get("supplier_at_risk"):
                assert d["avg_reliability_pct"] < 88, (
                    f"{n} flagged at_risk but avg_reliability={d['avg_reliability_pct']:.1f}%"
                )


# ══════════════════════════════════════════════════════════════════════════════
# 3. CENTRALITY METRICS TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestCentralityMetrics:

    def test_metrics_dataframe_has_all_nodes(self, graph, metrics):
        """Metrics DataFrame must have one row per graph node."""
        assert len(metrics) == graph.number_of_nodes(), (
            f"Expected {graph.number_of_nodes()} rows, got {len(metrics)}"
        )

    def test_betweenness_is_normalised(self, metrics):
        """Betweenness centrality (normalised) must be in [0, 1]."""
        bc = metrics["betweenness_centrality"]
        assert bc.between(0.0, 1.0).all(), (
            f"Betweenness out of [0,1]: min={bc.min():.4f}, max={bc.max():.4f}"
        )

    def test_pagerank_sums_to_one(self, metrics):
        """PageRank scores must sum to approximately 1.0 (±0.01)."""
        total = metrics["pagerank"].sum()
        assert abs(total - 1.0) < 0.01, f"PageRank sums to {total:.4f}, expected ≈ 1.0"

    def test_composite_risk_score_non_negative(self, metrics):
        """Composite risk score must be non-negative."""
        assert (metrics["composite_risk_score"] >= 0).all(), (
            "Negative composite risk scores found"
        )

    def test_metrics_sorted_by_betweenness(self, metrics):
        """Returned DataFrame should be sorted descending by betweenness."""
        bc = metrics["betweenness_centrality"].values
        assert all(bc[i] >= bc[i+1] for i in range(len(bc)-1)), (
            "Metrics DataFrame not sorted by betweenness descending"
        )

    def test_top_bottleneck_returns_n_rows(self, metrics):
        """get_top_bottlenecks(n=5) should return exactly 5 rows."""
        top = get_top_bottlenecks(metrics, n=5)
        assert len(top) == 5, f"Expected 5 rows, got {len(top)}"

    @pytest.mark.parametrize("n", [1, 3, 5, 10])
    def test_top_bottleneck_various_n(self, metrics, n):
        """get_top_bottlenecks should handle various values of n."""
        top = get_top_bottlenecks(metrics, n=n)
        assert len(top) <= n


# ══════════════════════════════════════════════════════════════════════════════
# 4. DISRUPTION SIMULATION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestDisruptionSimulation:

    def test_simulation_returns_expected_keys(self, enriched_graph):
        """Simulation result dict must contain all expected keys."""
        result = simulate_disruption(enriched_graph, n_nodes=3)
        required = {"removal_sequence", "steps", "baseline", "final_impact"}
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}"
        )

    def test_removal_sequence_length(self, enriched_graph):
        """Removal sequence must have exactly n_nodes entries."""
        n = 4
        result = simulate_disruption(enriched_graph, n_nodes=n)
        assert len(result["removal_sequence"]) == n

    def test_steps_count_matches_n_nodes(self, enriched_graph):
        """Number of simulation steps must equal n_nodes."""
        n = 3
        result = simulate_disruption(enriched_graph, n_nodes=n)
        assert len(result["steps"]) == n

    def test_network_reach_decreases_or_stays(self, enriched_graph):
        """Network reach % should not increase after removing a node."""
        result = simulate_disruption(enriched_graph, n_nodes=5)
        reaches = [s["network_reach_pct"] for s in result["steps"]]
        for i in range(1, len(reaches)):
            assert reaches[i] <= reaches[i-1] + 0.1, (
                f"Network reach increased at step {i+1}: {reaches[i-1]} → {reaches[i]}"
            )

    def test_baseline_has_correct_node_count(self, enriched_graph):
        """Baseline node count must match the actual graph."""
        result = simulate_disruption(enriched_graph, n_nodes=2)
        assert result["baseline"]["n_nodes"] == enriched_graph.number_of_nodes()

    def test_simulation_does_not_mutate_original_graph(self, enriched_graph):
        """Simulation must not modify the original graph (should copy internally)."""
        original_nodes = enriched_graph.number_of_nodes()
        simulate_disruption(enriched_graph, n_nodes=5)
        assert enriched_graph.number_of_nodes() == original_nodes, (
            "Original graph was mutated during simulation!"
        )

    def test_final_impact_contains_saving_metric(self, enriched_graph):
        """Final impact dict must contain path length increase metric."""
        result = simulate_disruption(enriched_graph, n_nodes=3)
        assert "path_length_increase_pct" in result["final_impact"]


# ══════════════════════════════════════════════════════════════════════════════
# 5. CRITICAL PATH FINDER TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestCriticalPaths:

    def test_paths_dataframe_not_empty(self, enriched_graph):
        """Should find at least one OEM-to-Airline path."""
        paths = find_critical_paths(enriched_graph)
        assert not paths.empty, "No critical paths found — check graph connectivity"

    def test_paths_have_required_columns(self, enriched_graph):
        """Paths DataFrame must contain standard columns."""
        paths = find_critical_paths(enriched_graph)
        for col in ["source", "target", "total_lead_time_days", "n_hops"]:
            assert col in paths.columns, f"Missing column: {col}"

    def test_all_lead_times_positive(self, enriched_graph):
        """All cumulative lead times must be positive."""
        paths = find_critical_paths(enriched_graph)
        assert (paths["total_lead_time_days"] > 0).all()

    def test_paths_sorted_descending(self, enriched_graph):
        """Paths should be sorted descending by total lead time."""
        paths = find_critical_paths(enriched_graph)
        times = paths["total_lead_time_days"].values
        assert all(times[i] >= times[i+1] for i in range(len(times)-1)), (
            "Critical paths not sorted descending by lead time"
        )

    def test_dataframes_export_correct_shape(self, graph):
        """graph_to_dataframes must return DataFrames with correct row counts."""
        nodes_df, edges_df = graph_to_dataframes(graph)
        assert len(nodes_df) == graph.number_of_nodes()
        assert len(edges_df) == graph.number_of_edges()
