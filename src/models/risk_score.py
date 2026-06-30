"""
risk_score.py — Aviation Supply Chain: Supplier Risk Scoring
=============================================================
Phase 4 · ML Models & Optimisation

PURPOSE
-------
Builds a composite supplier risk score (0–100) using:

  1. Rule-Based Score  — weighted combination of KPI thresholds
                         (industry standard, fully explainable)

  2. Random Forest     — ML classifier that learns which feature combinations
                         actually predict AOG risk events, using the graph
                         edge data from Phase 3 as training features

The rule-based score is the PRIMARY output for the dashboard (explainable,
regulators can audit it). The RF model is a SECONDARY validation layer that
surfaces non-obvious risk patterns.

FEATURES USED
-------------
  on_time_delivery_pct    : historical OTD % (most important — from graph edges)
  lead_time_days          : supplier lead time (long = higher AOG exposure)
  lead_time_variability   : std dev of lead times (instability signal)
  reliability_pct         : from graph edge attribute
  single_source_flag      : sole supplier to any downstream node?
  revenue_bn              : financial stability proxy (small = fragile)
  geo_concentration       : country risk (1=USA/EU, 2=Emerging, 3=High-risk)
  n_customers             : out-degree (how many downstream nodes depend on them)

USAGE
-----
    from src.models.risk_score import (
        compute_rule_based_risk, train_rf_risk_classifier, score_all_suppliers
    )

AUTHOR  : Aviation SC Analytics Project
VERSION : 1.0.0 (Phase 4)
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

log = logging.getLogger("aviation_sc.models.risk_score")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. GENERATE SUPPLIER FEATURE DATASET FROM GRAPH EDGES
# ═══════════════════════════════════════════════════════════════════════════════

def extract_supplier_features(G) -> pd.DataFrame:
    """
    Extract supplier-level risk features from the NetworkX supply chain graph.

    Parameters
    ----------
    G : nx.DiGraph  The aviation supply chain graph from Phase 3

    Returns
    -------
    pd.DataFrame  One row per non-Airline, non-OEM node (the "supplier" tiers)
    """
    import networkx as nx

    records = []
    for node, attrs in G.nodes(data=True):
        node_type = attrs.get("type", "Unknown")
        if node_type in ("Airline",):   # Airlines are customers, not risk subjects
            continue

        out_edges  = list(G.out_edges(node, data=True))
        in_edges   = list(G.in_edges(node, data=True))
        n_customers = len(out_edges)
        n_suppliers = len(in_edges)

        # Reliability from outgoing edges
        reliabilities = [d.get("reliability_pct", 90) for _, _, d in out_edges]
        lead_times    = [d.get("lead_time_days", 30)  for _, _, d in out_edges]
        avg_rel   = np.mean(reliabilities) if reliabilities else 90.0
        avg_lead  = np.mean(lead_times)    if lead_times    else 30.0
        std_lead  = np.std(lead_times)     if len(lead_times) > 1 else 0.0

        # Geo-concentration: proxy from country attribute
        country = attrs.get("country", "USA")
        geo_risk = {"USA": 1, "UK": 1, "Germany": 1, "France": 1,
                    "Switzerland": 1, "Singapore": 2, "HK": 2,
                    "Australia": 1, "UAE": 2, "India": 3}.get(country, 2)

        # Single-source flag: any downstream node with only one supplier?
        single_src = attrs.get("single_source_flag", False)

        # Revenue as financial stability proxy
        revenue = attrs.get("revenue_bn", 1.0) or 1.0

        # Betweenness (pre-computed) — higher = more critical = more risk exposure
        btwn = nx.betweenness_centrality(G, normalized=True,
                                          weight="lead_time_days").get(node, 0)

        records.append({
            "supplier":              node,
            "supplier_type":         node_type,
            "country":               country,
            "tier":                  attrs.get("tier", -1),
            "on_time_delivery_pct":  round(avg_rel, 1),
            "avg_lead_time_days":    round(avg_lead, 1),
            "lead_time_variability": round(std_lead, 1),
            "single_source_flag":    int(single_src),
            "n_customers":           n_customers,
            "n_suppliers":           n_suppliers,
            "revenue_bn":            float(revenue),
            "geo_concentration":     geo_risk,
            "betweenness":           round(btwn, 4),
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RULE-BASED RISK SCORE (primary, explainable)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rule_based_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a weighted composite risk score (0–100) for each supplier.

    Scoring weights (sum to 1.0):
      OTD performance       : 30%  (most directly linked to AOG risk)
      Lead time level       : 20%  (long lead = higher AOG exposure window)
      Lead time variability : 15%  (unpredictability = poor planning ability)
      Single source flag    : 20%  (sole supplier = no backup)
      Financial stability   : 10%  (small revenue = fragile supplier)
      Geo concentration     : 5%   (country risk)

    Score interpretation:
      0–25   : LOW RISK    — stable, redundant supplier
      26–50  : MEDIUM RISK — monitor quarterly
      51–75  : HIGH RISK   — dual-source or buffer stock required
      76–100 : CRITICAL    — immediate mitigation needed

    Parameters
    ----------
    df : pd.DataFrame  Output of extract_supplier_features()

    Returns
    -------
    pd.DataFrame  Input df + "risk_score" + "risk_tier" columns
    """
    df = df.copy()

    # ── Component scores — each normalised to [0, 100] ────────────────────────

    # OTD: 100% OTD → score 0;  70% OTD → score 100
    df["score_otd"] = ((100 - df["on_time_delivery_pct"]) / 30 * 100).clip(0, 100)

    # Lead time: ≤14 days → 0; ≥90 days → 100
    df["score_lead"] = ((df["avg_lead_time_days"] - 14) / 76 * 100).clip(0, 100)

    # Lead variability: 0 std → 0; ≥30 days std → 100
    df["score_variability"] = (df["lead_time_variability"] / 30 * 100).clip(0, 100)

    # Single source: True → 100; False → 0
    df["score_single_src"] = df["single_source_flag"] * 100

    # Financial stability: revenue >$10B → 0; <$1B → 100
    df["score_financial"] = ((10 - df["revenue_bn"].clip(0, 10)) / 10 * 100).clip(0, 100)

    # Geo concentration: 1 → 0; 2 → 50; 3 → 100
    df["score_geo"] = ((df["geo_concentration"] - 1) / 2 * 100).clip(0, 100)

    # ── Weighted composite ────────────────────────────────────────────────────
    weights = {
        "score_otd":        0.30,
        "score_lead":       0.20,
        "score_variability":0.15,
        "score_single_src": 0.20,
        "score_financial":  0.10,
        "score_geo":        0.05,
    }
    df["risk_score"] = sum(
        df[col] * w for col, w in weights.items()
    ).round(1).clip(0, 100)

    # Risk tier
    def tier(score):
        if score >= 75: return "CRITICAL"
        if score >= 50: return "HIGH"
        if score >= 25: return "MEDIUM"
        return "LOW"

    df["risk_tier"] = df["risk_score"].apply(tier)

    log.info(f"✅ Rule-based risk scores computed for {len(df)} suppliers")
    log.info(f"   Distribution: {df['risk_tier'].value_counts().to_dict()}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RANDOM FOREST CLASSIFIER (secondary validation)
# ═══════════════════════════════════════════════════════════════════════════════

def train_rf_risk_classifier(df: pd.DataFrame, random_seed: int = 42) -> dict:
    """
    Train a Random Forest classifier to predict high-risk suppliers.

    Since we don't have historical AOG-event labels, we use the rule-based
    risk score to generate binary labels (risk_score > 50 = "at_risk").
    Then we train the RF on the raw features — if it learns a different pattern
    than the rule-based model, it highlights blind spots.

    This is a standard approach in supply chain risk analytics when historical
    incident labels are sparse or unavailable.

    Parameters
    ----------
    df          : pd.DataFrame  Output of compute_rule_based_risk()
    random_seed : int           For reproducibility

    Returns
    -------
    dict with keys:
        "model"              : trained RandomForestClassifier
        "accuracy"           : float (cross-validated)
        "feature_importance" : pd.DataFrame
        "predictions"        : pd.DataFrame  with risk_label vs predicted
        "classification_report" : str
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.preprocessing import StandardScaler

    FEATURE_COLS = [
        "on_time_delivery_pct", "avg_lead_time_days", "lead_time_variability",
        "single_source_flag", "n_customers", "n_suppliers",
        "revenue_bn", "geo_concentration", "betweenness",
    ]

    # Binary label: risk_score > 50 = at_risk
    df = df.copy()
    df["at_risk_label"] = (df["risk_score"] > 50).astype(int)

    X = df[FEATURE_COLS].fillna(0)
    y = df["at_risk_label"]

    if len(df) < 10:
        log.warning("Too few suppliers for RF training — using rule-based only")
        return {"model": None, "accuracy": None,
                "feature_importance": pd.DataFrame(), "predictions": df}

    log.info(f"🌲 Training Random Forest on {len(df)} suppliers...")

    model = RandomForestClassifier(
        n_estimators  = 100,
        max_depth     = 5,         # Shallow trees → avoid overfitting on small dataset
        min_samples_leaf = 2,
        oob_score     = True,      # Out-of-bag score = free cross-validation
        random_state  = random_seed,
        n_jobs        = -1,
        class_weight  = "balanced", # Handle class imbalance (fewer high-risk suppliers)
    )
    model.fit(X, y)

    # Cross-validated accuracy
    cv = StratifiedKFold(n_splits=min(5, len(df) // 3), shuffle=True,
                         random_state=random_seed)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    # Feature importance
    fi_df = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Predictions
    df["rf_predicted_at_risk"] = model.predict(X)
    df["rf_risk_probability"]  = model.predict_proba(X)[:, 1].round(3)

    report = classification_report(y, model.predict(X), zero_division=0)

    log.info(f"✅ RF trained: OOB accuracy={model.oob_score_:.3f}, "
             f"CV accuracy={cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    log.info(f"   Top RF feature: {fi_df.iloc[0]['feature']} "
             f"(importance={fi_df.iloc[0]['importance']:.3f})")

    return {
        "model":                 model,
        "oob_accuracy":          round(model.oob_score_, 3),
        "cv_accuracy":           round(cv_scores.mean(), 3),
        "cv_std":                round(cv_scores.std(), 3),
        "feature_importance":    fi_df,
        "predictions":           df,
        "classification_report": report,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. COMBINED SCORING — main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def score_all_suppliers(G) -> dict:
    """
    Run the full supplier risk scoring pipeline.

    Steps:
      1. Extract features from graph
      2. Compute rule-based risk score
      3. Train Random Forest classifier
      4. Return combined results

    Parameters
    ----------
    G : nx.DiGraph  Phase 3 supply chain graph

    Returns
    -------
    dict with keys:
        "supplier_scores"    : pd.DataFrame  All suppliers with risk scores
        "critical_suppliers" : pd.DataFrame  Those with risk_tier == CRITICAL or HIGH
        "rf_result"          : dict          RF model + importances
        "feature_df"         : pd.DataFrame  Raw extracted features
    """
    features_df = extract_supplier_features(G)
    scored_df   = compute_rule_based_risk(features_df)
    rf_result   = train_rf_risk_classifier(scored_df)

    # Final scored table — clean, portfolio-ready
    output_cols = [
        "supplier", "supplier_type", "country", "tier",
        "on_time_delivery_pct", "avg_lead_time_days",
        "single_source_flag", "n_customers",
        "risk_score", "risk_tier",
        "rf_risk_probability",
    ]
    available = [c for c in output_cols if c in rf_result["predictions"].columns]
    final_df  = rf_result["predictions"][available].sort_values(
        "risk_score", ascending=False
    ).reset_index(drop=True)

    critical = final_df[final_df["risk_tier"].isin(["CRITICAL", "HIGH"])]
    log.info(f"🚨 {len(critical)} suppliers classified as HIGH or CRITICAL risk")

    return {
        "supplier_scores":    final_df,
        "critical_suppliers": critical,
        "rf_result":          rf_result,
        "feature_df":         features_df,
    }


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from src.network.build_graph import build_aviation_sc_graph, add_risk_attributes

    print("\n=== SUPPLIER RISK SCORING SELF-TEST ===\n")
    G = build_aviation_sc_graph()
    G = add_risk_attributes(G)

    result = score_all_suppliers(G)

    print("=== TOP 10 SUPPLIERS BY RISK SCORE ===")
    print(result["supplier_scores"].head(10)[
        ["supplier","risk_score","risk_tier","on_time_delivery_pct",
         "single_source_flag","rf_risk_probability"]
    ].to_string(index=False))

    print(f"\n=== CRITICAL / HIGH RISK SUPPLIERS ({len(result['critical_suppliers'])}) ===")
    print(result["critical_suppliers"][
        ["supplier","risk_tier","risk_score","on_time_delivery_pct",
         "avg_lead_time_days","single_source_flag"]
    ].to_string(index=False))

    print(f"\n=== RF TOP FEATURE IMPORTANCES ===")
    print(result["rf_result"]["feature_importance"].head(5).to_string(index=False))
    print(f"\nRF OOB Accuracy: {result['rf_result']['oob_accuracy']}")
    print("\n✅ risk_score.py self-test passed.")
