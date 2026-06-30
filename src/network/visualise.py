"""
visualise.py — Aviation Supply Chain: Interactive Network Visualisation
=======================================================================
Phase 3 · Network Modelling & Analysis

PURPOSE
-------
Exports the supply chain graph as a self-contained interactive HTML file
using the Pyvis library (which wraps vis.js JavaScript).

The output docs/network.html can be:
  - Opened directly in any web browser
  - Embedded in the GitHub Pages site
  - Shared with anyone — no Python required to view

VISUAL DESIGN DECISIONS
-----------------------
Node colour   → tier/type (OEM=purple, Tier1=blue, Tier2=cyan, MRO=green, Airline=amber)
Node size     → proportional to betweenness_centrality (bigger = more critical)
Edge colour   → reliability_pct bucket (green=high, amber=medium, red=low)
Edge thickness→ proportional to cost_usd (thicker = more expensive flow)
Node border   → red thick border if supplier_at_risk=True
Tooltip       → hover over any node/edge for full data

USAGE
-----
    from src.network.visualise import export_pyvis_network
    export_pyvis_network(G, metrics_df, output_path="docs/network.html")

AUTHOR  : Aviation SC Analytics Project
VERSION : 1.0.0 (Phase 3)
"""

import logging
import pandas as pd
import networkx as nx
from pathlib import Path

log = logging.getLogger("aviation_sc.network.visualise")

ROOT     = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)


# ── Colour palettes ───────────────────────────────────────────────────────────
NODE_COLOURS: dict[str, str] = {
    "OEM":     "#8b5cf6",   # Purple
    "Tier1":   "#0ea5e9",   # Sky blue
    "Tier2":   "#06b6d4",   # Cyan
    "MRO":     "#10b981",   # Emerald green
    "Airline": "#f59e0b",   # Amber
}

RELIABILITY_EDGE_COLOUR: list[tuple[float, str]] = [
    (92.0, "#10b981"),   # >= 92%  → green   (high reliability)
    (87.0, "#f59e0b"),   # >= 87%  → amber   (medium)
    (0.0,  "#ef4444"),   # < 87%   → red     (at risk)
]


def _edge_colour(reliability_pct: float) -> str:
    for threshold, colour in RELIABILITY_EDGE_COLOUR:
        if reliability_pct >= threshold:
            return colour
    return "#ef4444"


def _node_size(betweenness: float) -> int:
    """Scale betweenness [0,1] → node diameter [20, 70] pixels."""
    return int(20 + betweenness * 500)


def _node_tooltip(node_id: str, attrs: dict, metrics_row: pd.Series | None) -> str:
    """Build an HTML tooltip string shown on node hover."""
    lines = [
        f"<b>{node_id}</b>",
        f"Type: {attrs.get('type','?')} | Tier: {attrs.get('tier','?')}",
        f"Country: {attrs.get('country','?')}",
        f"Revenue: ${attrs.get('revenue_bn','?')}B",
    ]
    if attrs.get("employees"):
        lines.append(f"Employees: {attrs['employees']:,}")
    if attrs.get("fleet_size"):
        lines.append(f"Fleet size: {attrs['fleet_size']} aircraft")
    if attrs.get("avg_reliability_pct"):
        lines.append(f"Avg OTD: {attrs['avg_reliability_pct']}%")
    if attrs.get("avg_lead_time_days"):
        lines.append(f"Avg Lead Time: {attrs['avg_lead_time_days']} days")
    if attrs.get("supplier_at_risk"):
        lines.append("<b style='color:#ef4444'>⚠ SUPPLIER AT RISK</b>")
    if attrs.get("single_source_flag"):
        lines.append("<b style='color:#f59e0b'>⚠ SINGLE SOURCE</b>")
    if metrics_row is not None:
        lines.append(f"Betweenness: {metrics_row.get('betweenness_centrality','?')}")
        lines.append(f"PageRank: {metrics_row.get('pagerank','?')}")
        lines.append(f"Risk Score: {metrics_row.get('composite_risk_score','?')}")
    return "<br>".join(lines)


def _edge_tooltip(src: str, dst: str, attrs: dict) -> str:
    """Build an HTML tooltip string shown on edge hover."""
    return (
        f"<b>{src} → {dst}</b><br>"
        f"Part Category: {attrs.get('part_category','?')}<br>"
        f"Lead Time: {attrs.get('lead_time_days','?')} days<br>"
        f"Cost: ${attrs.get('cost_usd',0):,.0f}<br>"
        f"OTD Reliability: {attrs.get('reliability_pct','?')}%<br>"
        f"Risk Score: {attrs.get('risk_score','?')}"
    )


def export_pyvis_network(
    G: nx.DiGraph,
    metrics_df: pd.DataFrame | None = None,
    output_path: str | Path | None = None,
    height: str = "750px",
    bgcolor: str = "#0a0f1e",
    font_color: str = "#e2e8f0",
) -> Path:
    """
    Export the supply chain graph as an interactive Pyvis HTML file.

    Parameters
    ----------
    G           : nx.DiGraph  The aviation supply chain graph
    metrics_df  : pd.DataFrame, optional
                  Output of compute_all_metrics() — used for node sizing
    output_path : str or Path, optional
                  Where to save the HTML (default: docs/network.html)
    height      : str   Canvas height (default "750px")
    bgcolor     : str   Background colour (matches the project dark theme)
    font_color  : str   Node label colour

    Returns
    -------
    Path  Path to the generated HTML file
    """
    try:
        from pyvis.network import Network
    except ImportError:
        log.error("pyvis not installed. Run: pip install pyvis")
        raise

    if output_path is None:
        output_path = DOCS_DIR / "network.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build metrics lookup dict for node sizing
    metrics_lookup: dict[str, pd.Series] = {}
    if metrics_df is not None:
        for _, row in metrics_df.iterrows():
            metrics_lookup[row["node_id"]] = row

    # ── Initialise Pyvis network ──────────────────────────────────────────────
    net = Network(
        height=height,
        width="100%",
        directed=True,
        bgcolor=bgcolor,
        font_color=font_color,
        notebook=False,
        cdn_resources="in_line",   # Self-contained HTML — no internet needed to view
    )

    # ── Add nodes ─────────────────────────────────────────────────────────────
    for node_id, attrs in G.nodes(data=True):
        node_type   = attrs.get("type", "Unknown")
        colour      = NODE_COLOURS.get(node_type, "#64748b")
        metrics_row = metrics_lookup.get(node_id)
        betweenness = float(metrics_row["betweenness_centrality"]) if metrics_row is not None else 0.0
        at_risk     = attrs.get("supplier_at_risk", False)

        node_opts = {
            "label":       node_id.replace("_", "\n"),
            "color": {
                "background": colour,
                "border":     "#ef4444" if at_risk else "#ffffff22",
                "highlight":  {"background": "#ffffff", "border": "#0ea5e9"},
            },
            "size":        _node_size(betweenness),
            "title":       _node_tooltip(node_id, attrs, metrics_row),
            "font": {
                "size":  11 if "_" in node_id else 12,
                "color": font_color,
                "bold":  True,
            },
            "borderWidth": 3 if at_risk else 1,
            "shadow":      True,
        }
        net.add_node(node_id, **node_opts)

    # ── Add edges ─────────────────────────────────────────────────────────────
    for src, dst, attrs in G.edges(data=True):
        reliability = attrs.get("reliability_pct", 90)
        cost        = attrs.get("cost_usd", 100_000)
        edge_colour = _edge_colour(reliability)

        # Edge thickness: log scale of cost (avoid dominance by engine overhauls)
        import math
        width = max(1.0, min(8.0, math.log10(cost / 10_000 + 1) * 2.5))

        net.add_edge(
            src, dst,
            title=_edge_tooltip(src, dst, attrs),
            color={"color": edge_colour, "highlight": "#ffffff", "opacity": 0.85},
            width=round(width, 1),
            arrows={"to": {"enabled": True, "scaleFactor": 0.8}},
            smooth={"type": "curvedCW", "roundness": 0.15},
        )

    # ── Physics settings — Barnes-Hut hierarchical layout ────────────────────
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -12000,
          "centralGravity": 0.25,
          "springLength": 160,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 0.6
        },
        "maxVelocity": 50,
        "minVelocity": 0.1,
        "solver": "barnesHut",
        "stabilization": { "iterations": 200 }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200,
        "navigationButtons": true,
        "keyboard": true,
        "zoomView": true
      },
      "edges": {
        "shadow": false
      },
      "nodes": {
        "shadow": true
      }
    }
    """)

    # ── Write HTML ────────────────────────────────────────────────────────────
    # Pyvis doesn't let us pass an encoding directly, so we write manually in UTF-8
    html_content = net.generate_html()
    output_path.write_text(html_content, encoding="utf-8")

    # Inject legend and title into the generated HTML
    _inject_legend(output_path)

    log.info(f"✅ Pyvis network exported → {output_path}")
    return output_path


def _inject_legend(html_path: Path) -> None:
    """
    Inject a floating legend panel into the Pyvis-generated HTML.
    This adds tier colour keys and AOG risk indicators into the page.
    """
    legend_html = """
<div style="
  position:fixed; top:20px; left:20px; z-index:9999;
  background:rgba(10,15,30,0.92); border:1px solid #1e2d4a;
  border-radius:12px; padding:16px 20px; font-family:Inter,sans-serif;
  font-size:12px; color:#e2e8f0; min-width:200px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.5);
">
  <div style="font-weight:700;font-size:14px;margin-bottom:12px;color:#fff;">
    ✈ Aviation SC Network
  </div>

  <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;
              color:#64748b;margin-bottom:8px;">Node Tiers</div>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
    <span style="width:12px;height:12px;border-radius:50%;
                 background:#8b5cf6;flex-shrink:0;"></span>OEM
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
    <span style="width:12px;height:12px;border-radius:50%;
                 background:#0ea5e9;flex-shrink:0;"></span>Tier-1 Supplier
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
    <span style="width:12px;height:12px;border-radius:50%;
                 background:#06b6d4;flex-shrink:0;"></span>Tier-2 Supplier
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
    <span style="width:12px;height:12px;border-radius:50%;
                 background:#10b981;flex-shrink:0;"></span>MRO Facility
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
    <span style="width:12px;height:12px;border-radius:50%;
                 background:#f59e0b;flex-shrink:0;"></span>Airline
  </div>

  <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;
              color:#64748b;margin-bottom:8px;">Edge Reliability</div>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
    <span style="width:24px;height:3px;background:#10b981;
                 border-radius:2px;flex-shrink:0;"></span>&ge; 92% OTD
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
    <span style="width:24px;height:3px;background:#f59e0b;
                 border-radius:2px;flex-shrink:0;"></span>87–92% OTD
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
    <span style="width:24px;height:3px;background:#ef4444;
                 border-radius:2px;flex-shrink:0;"></span>&lt; 87% OTD ⚠
  </div>

  <div style="font-size:10px;color:#64748b;line-height:1.6;">
    Node size = betweenness centrality<br>
    Red border = supplier at risk<br>
    Hover nodes/edges for details
  </div>
</div>
"""
    html = html_path.read_text(encoding="utf-8")
    html = html.replace("</body>", legend_html + "</body>")
    html_path.write_text(html, encoding="utf-8")


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from src.network.build_graph import build_aviation_sc_graph, add_risk_attributes
    from src.network.metrics    import compute_all_metrics

    G = build_aviation_sc_graph()
    G = add_risk_attributes(G)
    metrics = compute_all_metrics(G)

    out = export_pyvis_network(G, metrics, output_path="docs/network.html")
    print(f"\n✅ Network exported to: {out}")
    print(f"   Open in browser: open {out}")
    print("\n✅ visualise.py self-test passed.")
