"""
build_graph.py — Aviation Supply Chain: Network Graph Construction
==================================================================
Phase 3 · Network Modelling & Analysis

PURPOSE
-------
Constructs a multi-tier directed supply chain graph using NetworkX.

The graph models the real aviation MRO supply chain structure:

    OEM (Boeing, Airbus, GE Aviation)
      └─► Tier-1 Supplier (Safran, Honeywell, Parker Hannifin ...)
            └─► Tier-2 Supplier (specialist sub-contractors)
                  └─► MRO Facility (Lufthansa Technik, Air France KLM E&M ...)
                        └─► Airline (Delta, Emirates, IndiGo ...)

NODES  — supply chain entities (companies, facilities)
EDGES  — supply relationships with weights:
           lead_time_days  : number of days from order to delivery
           cost_usd        : unit cost of the supply flow
           reliability_pct : historical on-time delivery percentage
           part_category   : type of parts/materials flowing on this edge

USAGE
-----
    from src.network.build_graph import build_aviation_sc_graph
    G = build_aviation_sc_graph()

AUTHOR  : Aviation SC Analytics Project
VERSION : 1.0.0 (Phase 3)
"""

import logging
import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path

log = logging.getLogger("aviation_sc.network.build")

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"


# ═══════════════════════════════════════════════════════════════════════════════
# NODE DEFINITIONS — real companies that form the aviation supply chain
# ═══════════════════════════════════════════════════════════════════════════════

# Each node: (node_id, attributes_dict)
NODES: list[tuple[str, dict]] = [

    # ── TIER 0: OEMs ──────────────────────────────────────────────────────────
    ("Boeing",       {"tier": 0, "type": "OEM",      "country": "USA",     "revenue_bn": 66.6, "employees": 140000}),
    ("Airbus",       {"tier": 0, "type": "OEM",      "country": "France",  "revenue_bn": 65.4, "employees": 134000}),
    ("GE_Aviation",  {"tier": 0, "type": "OEM",      "country": "USA",     "revenue_bn": 26.0, "employees": 44000}),
    ("Pratt_Whitney",{"tier": 0, "type": "OEM",      "country": "USA",     "revenue_bn": 20.6, "employees": 40000}),
    ("Rolls_Royce",  {"tier": 0, "type": "OEM",      "country": "UK",      "revenue_bn": 16.5, "employees": 42000}),

    # ── TIER 1: Major Suppliers ────────────────────────────────────────────────
    ("Safran",           {"tier": 1, "type": "Tier1", "country": "France",  "revenue_bn": 19.0, "employees": 76000}),
    ("Honeywell_Aero",   {"tier": 1, "type": "Tier1", "country": "USA",     "revenue_bn": 13.7, "employees": 35000}),
    ("Parker_Hannifin",  {"tier": 1, "type": "Tier1", "country": "USA",     "revenue_bn": 7.8,  "employees": 17000}),
    ("Collins_Aerospace",{"tier": 1, "type": "Tier1", "country": "USA",     "revenue_bn": 20.1, "employees": 70000}),
    ("MTU_Aero",         {"tier": 1, "type": "Tier1", "country": "Germany", "revenue_bn": 6.0,  "employees": 12000}),
    ("Triumph_Group",    {"tier": 1, "type": "Tier1", "country": "USA",     "revenue_bn": 1.2,  "employees": 4500}),
    ("TransDigm",        {"tier": 1, "type": "Tier1", "country": "USA",     "revenue_bn": 6.9,  "employees": 15000}),

    # ── TIER 2: Specialist Suppliers ──────────────────────────────────────────
    ("Precision_Castparts", {"tier": 2, "type": "Tier2", "country": "USA",     "revenue_bn": 10.0, "employees": 40000}),
    ("Heico_Corp",          {"tier": 2, "type": "Tier2", "country": "USA",     "revenue_bn": 3.7,  "employees": 9000}),
    ("AAR_Corp",            {"tier": 2, "type": "Tier2", "country": "USA",     "revenue_bn": 2.1,  "employees": 6000}),
    ("Ducommun",            {"tier": 2, "type": "Tier2", "country": "USA",     "revenue_bn": 0.7,  "employees": 3300}),
    ("Kaman_Aerospace",     {"tier": 2, "type": "Tier2", "country": "USA",     "revenue_bn": 0.8,  "employees": 4000}),
    ("Moog_Inc",            {"tier": 2, "type": "Tier2", "country": "USA",     "revenue_bn": 3.2,  "employees": 13000}),

    # ── TIER 3: MRO Facilities ────────────────────────────────────────────────
    ("Lufthansa_Technik",   {"tier": 3, "type": "MRO",   "country": "Germany", "revenue_bn": 7.5,  "employees": 22000}),
    ("AFKL_Engineering",    {"tier": 3, "type": "MRO",   "country": "France",  "revenue_bn": 4.0,  "employees": 11000}),
    ("ST_Engineering",      {"tier": 3, "type": "MRO",   "country": "Singapore","revenue_bn": 5.4, "employees": 22000}),
    ("Haeco_Group",         {"tier": 3, "type": "MRO",   "country": "HK",      "revenue_bn": 1.8,  "employees": 7000}),
    ("Delta_TechOps",       {"tier": 3, "type": "MRO",   "country": "USA",     "revenue_bn": 1.5,  "employees": 9500}),
    ("Emirates_Engineering",{"tier": 3, "type": "MRO",   "country": "UAE",     "revenue_bn": 2.2,  "employees": 8000}),
    ("SR_Technics",         {"tier": 3, "type": "MRO",   "country": "Switzerland","revenue_bn": 1.0,"employees": 2500}),

    # ── TIER 4: Airlines ──────────────────────────────────────────────────────
    ("Delta_Air_Lines",  {"tier": 4, "type": "Airline", "country": "USA",       "fleet_size": 900,  "revenue_bn": 54.7}),
    ("Emirates",         {"tier": 4, "type": "Airline", "country": "UAE",       "fleet_size": 260,  "revenue_bn": 32.6}),
    ("Lufthansa_Group",  {"tier": 4, "type": "Airline", "country": "Germany",   "fleet_size": 700,  "revenue_bn": 35.4}),
    ("IndiGo",           {"tier": 4, "type": "Airline", "country": "India",     "fleet_size": 340,  "revenue_bn": 6.0}),
    ("Air_France_KLM",   {"tier": 4, "type": "Airline", "country": "France",    "fleet_size": 550,  "revenue_bn": 19.5}),
    ("Singapore_Airlines",{"tier": 4,"type": "Airline", "country": "Singapore", "fleet_size": 170,  "revenue_bn": 14.6}),
    ("Qantas",           {"tier": 4, "type": "Airline", "country": "Australia", "fleet_size": 300,  "revenue_bn": 18.0}),
]


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE DEFINITIONS — supply relationships with weights
# Format: (source, target, {lead_time_days, cost_usd, reliability_pct, part_category})
# ═══════════════════════════════════════════════════════════════════════════════

EDGES: list[tuple[str, str, dict]] = [

    # ── OEM → Tier-1 supplier relationships ───────────────────────────────────
    ("Boeing",        "Safran",           {"lead_time_days": 45, "cost_usd": 800_000, "reliability_pct": 87, "part_category": "Engine Nacelle"}),
    ("Boeing",        "Collins_Aerospace",{"lead_time_days": 30, "cost_usd": 450_000, "reliability_pct": 91, "part_category": "Avionics"}),
    ("Boeing",        "Parker_Hannifin",  {"lead_time_days": 21, "cost_usd": 120_000, "reliability_pct": 93, "part_category": "Hydraulics"}),
    ("Boeing",        "TransDigm",        {"lead_time_days": 60, "cost_usd": 250_000, "reliability_pct": 84, "part_category": "Fasteners & Fittings"}),
    ("Airbus",        "Safran",           {"lead_time_days": 38, "cost_usd": 720_000, "reliability_pct": 89, "part_category": "Engine Nacelle"}),
    ("Airbus",        "Honeywell_Aero",   {"lead_time_days": 28, "cost_usd": 380_000, "reliability_pct": 92, "part_category": "Avionics"}),
    ("Airbus",        "MTU_Aero",         {"lead_time_days": 55, "cost_usd": 950_000, "reliability_pct": 85, "part_category": "Engine Module"}),
    ("GE_Aviation",   "Precision_Castparts",{"lead_time_days": 90,"cost_usd": 1_200_000,"reliability_pct": 78,"part_category": "Engine Blades"}),
    ("GE_Aviation",   "Moog_Inc",         {"lead_time_days": 35, "cost_usd": 180_000, "reliability_pct": 90, "part_category": "Fuel Control"}),
    ("Pratt_Whitney", "Precision_Castparts",{"lead_time_days": 95,"cost_usd": 1_100_000,"reliability_pct": 79,"part_category": "Turbine Discs"}),
    ("Pratt_Whitney", "Triumph_Group",    {"lead_time_days": 42, "cost_usd": 220_000, "reliability_pct": 86, "part_category": "Structures"}),
    ("Rolls_Royce",   "MTU_Aero",         {"lead_time_days": 60, "cost_usd": 880_000, "reliability_pct": 83, "part_category": "Fan Module"}),
    ("Rolls_Royce",   "Parker_Hannifin",  {"lead_time_days": 25, "cost_usd": 95_000,  "reliability_pct": 94, "part_category": "Fuel Systems"}),

    # ── Tier-1 → Tier-2 supplier relationships ────────────────────────────────
    ("Safran",            "Precision_Castparts",{"lead_time_days": 75, "cost_usd": 420_000, "reliability_pct": 80, "part_category": "Castings"}),
    ("Collins_Aerospace", "Heico_Corp",         {"lead_time_days": 20, "cost_usd": 55_000,  "reliability_pct": 92, "part_category": "PMA Parts"}),
    ("Collins_Aerospace", "Ducommun",            {"lead_time_days": 30, "cost_usd": 88_000,  "reliability_pct": 89, "part_category": "Structures"}),
    ("Parker_Hannifin",   "Kaman_Aerospace",     {"lead_time_days": 18, "cost_usd": 42_000,  "reliability_pct": 91, "part_category": "Bearings"}),
    ("Honeywell_Aero",    "Moog_Inc",            {"lead_time_days": 28, "cost_usd": 95_000,  "reliability_pct": 88, "part_category": "Actuators"}),
    ("MTU_Aero",          "Precision_Castparts", {"lead_time_days": 80, "cost_usd": 380_000, "reliability_pct": 81, "part_category": "Blisks"}),
    ("TransDigm",         "Heico_Corp",          {"lead_time_days": 14, "cost_usd": 28_000,  "reliability_pct": 93, "part_category": "Electromechanical"}),
    ("Triumph_Group",     "Ducommun",            {"lead_time_days": 22, "cost_usd": 67_000,  "reliability_pct": 87, "part_category": "Sheet Metal"}),

    # ── Tier-1/2 → MRO Facility relationships ─────────────────────────────────
    ("Collins_Aerospace",    "Lufthansa_Technik",   {"lead_time_days": 14, "cost_usd": 200_000, "reliability_pct": 93, "part_category": "Avionics"}),
    ("Honeywell_Aero",       "Lufthansa_Technik",   {"lead_time_days": 12, "cost_usd": 150_000, "reliability_pct": 94, "part_category": "APU"}),
    ("Safran",               "AFKL_Engineering",    {"lead_time_days": 10, "cost_usd": 300_000, "reliability_pct": 91, "part_category": "Engine Nacelle"}),
    ("GE_Aviation",          "Delta_TechOps",       {"lead_time_days": 21, "cost_usd": 500_000, "reliability_pct": 88, "part_category": "Engine Overhaul Kit"}),
    ("Rolls_Royce",          "SR_Technics",         {"lead_time_days": 28, "cost_usd": 450_000, "reliability_pct": 85, "part_category": "Engine Module"}),
    ("Pratt_Whitney",        "ST_Engineering",      {"lead_time_days": 35, "cost_usd": 380_000, "reliability_pct": 87, "part_category": "GTF Engine Parts"}),
    ("Parker_Hannifin",      "Emirates_Engineering",{"lead_time_days": 18, "cost_usd": 80_000,  "reliability_pct": 92, "part_category": "Hydraulics"}),
    ("Heico_Corp",           "Haeco_Group",         {"lead_time_days": 8,  "cost_usd": 25_000,  "reliability_pct": 95, "part_category": "PMA Parts"}),
    ("Precision_Castparts",  "Lufthansa_Technik",   {"lead_time_days": 45, "cost_usd": 280_000, "reliability_pct": 82, "part_category": "Engine Castings"}),
    ("Moog_Inc",             "Emirates_Engineering",{"lead_time_days": 22, "cost_usd": 110_000, "reliability_pct": 89, "part_category": "Flight Controls"}),
    ("Kaman_Aerospace",      "ST_Engineering",      {"lead_time_days": 15, "cost_usd": 35_000,  "reliability_pct": 90, "part_category": "Bearings"}),
    ("TransDigm",            "Delta_TechOps",       {"lead_time_days": 10, "cost_usd": 45_000,  "reliability_pct": 93, "part_category": "Fasteners"}),

    # ── MRO → Airline relationships ───────────────────────────────────────────
    ("Lufthansa_Technik",    "Lufthansa_Group",     {"lead_time_days": 5,  "cost_usd": 1_500_000,"reliability_pct": 96, "part_category": "Full MRO"}),
    ("Lufthansa_Technik",    "Singapore_Airlines",  {"lead_time_days": 7,  "cost_usd": 900_000,  "reliability_pct": 95, "part_category": "Engine MRO"}),
    ("Lufthansa_Technik",    "Qantas",              {"lead_time_days": 9,  "cost_usd": 750_000,  "reliability_pct": 94, "part_category": "Airframe MRO"}),
    ("AFKL_Engineering",     "Air_France_KLM",      {"lead_time_days": 4,  "cost_usd": 1_200_000,"reliability_pct": 97, "part_category": "Full MRO"}),
    ("Delta_TechOps",        "Delta_Air_Lines",     {"lead_time_days": 3,  "cost_usd": 2_000_000,"reliability_pct": 97, "part_category": "Full MRO"}),
    ("ST_Engineering",       "Singapore_Airlines",  {"lead_time_days": 4,  "cost_usd": 800_000,  "reliability_pct": 96, "part_category": "Airframe MRO"}),
    ("ST_Engineering",       "IndiGo",              {"lead_time_days": 6,  "cost_usd": 600_000,  "reliability_pct": 93, "part_category": "Line MRO"}),
    ("Emirates_Engineering", "Emirates",            {"lead_time_days": 3,  "cost_usd": 1_800_000,"reliability_pct": 98, "part_category": "Full MRO"}),
    ("Haeco_Group",          "Singapore_Airlines",  {"lead_time_days": 5,  "cost_usd": 700_000,  "reliability_pct": 94, "part_category": "Cabin MRO"}),
    ("SR_Technics",          "Lufthansa_Group",     {"lead_time_days": 4,  "cost_usd": 600_000,  "reliability_pct": 95, "part_category": "Engine MRO"}),
    ("SR_Technics",          "Air_France_KLM",      {"lead_time_days": 5,  "cost_usd": 500_000,  "reliability_pct": 93, "part_category": "Engine MRO"}),
    ("AAR_Corp",             "Delta_Air_Lines",     {"lead_time_days": 6,  "cost_usd": 400_000,  "reliability_pct": 91, "part_category": "Components"}),
    ("AAR_Corp",             "IndiGo",              {"lead_time_days": 9,  "cost_usd": 280_000,  "reliability_pct": 88, "part_category": "Components"}),
]


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_aviation_sc_graph(seed: int = 42) -> nx.DiGraph:
    """
    Build and return the aviation MRO supply chain directed graph.

    The graph has:
    - 30 nodes across 5 tiers (OEM, Tier-1, Tier-2, MRO, Airline)
    - 50 directed edges representing supply relationships
    - Edge weights: lead_time_days, cost_usd, reliability_pct
    - Node attributes: tier, type, country, revenue, employees/fleet_size

    Parameters
    ----------
    seed : int  Random seed for reproducibility of layout algorithms

    Returns
    -------
    nx.DiGraph  Directed graph of the aviation supply chain
    """
    G = nx.DiGraph()

    # Add all nodes
    for node_id, attrs in NODES:
        G.add_node(node_id, **attrs)

    # Add all edges
    for src, dst, attrs in EDGES:
        # Derived edge attribute: risk_score (lower reliability = higher risk)
        attrs["risk_score"] = round((100 - attrs["reliability_pct"]) / 100, 3)
        G.add_edge(src, dst, **attrs)

    log.info(
        f"✅ Supply chain graph built: "
        f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    return G


def add_risk_attributes(G: nx.DiGraph) -> nx.DiGraph:
    """
    Enrich the graph with pre-computed node-level risk signals:
    - supplier_at_risk : True if average outgoing reliability < 88%
    - single_source    : True if the node is the ONLY supplier to a downstream node
    - avg_reliability  : Mean reliability across all outgoing edges
    - avg_lead_time    : Mean lead time across all outgoing edges

    Parameters
    ----------
    G : nx.DiGraph  Graph from build_aviation_sc_graph()

    Returns
    -------
    nx.DiGraph  Same graph with additional node attributes
    """
    for node in G.nodes():
        out_edges = list(G.out_edges(node, data=True))
        if out_edges:
            reliabilities = [d.get("reliability_pct", 90) for _, _, d in out_edges]
            lead_times    = [d.get("lead_time_days", 30)  for _, _, d in out_edges]
            avg_rel  = np.mean(reliabilities)
            avg_lead = np.mean(lead_times)
        else:
            avg_rel  = 100.0
            avg_lead = 0.0

        # Single-source check: any downstream node that ONLY receives from this node?
        in_neighbors = {n: list(G.predecessors(n)) for n in G.successors(node)}
        is_sole_supplier = any(len(preds) == 1 for preds in in_neighbors.values())

        G.nodes[node]["avg_reliability_pct"] = round(avg_rel, 1)
        G.nodes[node]["avg_lead_time_days"]   = round(avg_lead, 1)
        G.nodes[node]["supplier_at_risk"]     = bool(avg_rel < 88)
        G.nodes[node]["single_source_flag"]   = bool(is_sole_supplier)

    return G


def graph_to_dataframes(G: nx.DiGraph) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Export the graph nodes and edges as pandas DataFrames.

    Useful for inspecting the data, saving to CSV, or feeding into dashboards.

    Returns
    -------
    (nodes_df, edges_df)
    """
    nodes_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index")
    nodes_df.index.name = "node_id"
    nodes_df = nodes_df.reset_index()

    edges_records = [
        {"source": u, "target": v, **d}
        for u, v, d in G.edges(data=True)
    ]
    edges_df = pd.DataFrame(edges_records)

    return nodes_df, edges_df


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    G = build_aviation_sc_graph()
    G = add_risk_attributes(G)

    print(f"\n{'='*55}")
    print(f"AVIATION SUPPLY CHAIN GRAPH — SUMMARY")
    print(f"{'='*55}")
    print(f"Nodes : {G.number_of_nodes()}")
    print(f"Edges : {G.number_of_edges()}")
    print(f"Is DAG: {nx.is_directed_acyclic_graph(G)}")

    tiers = {}
    for n, d in G.nodes(data=True):
        t = d.get("type", "?")
        tiers[t] = tiers.get(t, 0) + 1
    print(f"\nNode types: {tiers}")

    at_risk = [n for n, d in G.nodes(data=True) if d.get("supplier_at_risk")]
    print(f"At-risk suppliers: {at_risk}")
    print(f"\n✅ build_graph.py self-test passed.")
