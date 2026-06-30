"""
inventory.py — Aviation Supply Chain: Inventory Optimisation Module
====================================================================
Phase 4 · ML Models & Optimisation

PURPOSE
-------
Calculates optimal inventory policies for every MRO spare-parts SKU:

  EOQ   — Economic Order Quantity: the batch size that minimises total
           inventory cost (holding cost + ordering cost balanced at crossover)

  SS    — Safety Stock: buffer inventory to absorb demand and lead-time
           variability, sized to achieve a target service level (95% default)

  ROP   — Reorder Point: when stock drops to this level, place a new order
           ROP = (avg demand × avg lead time) + safety stock

  TCI   — Total Cost of Inventory: sum of ordering + holding costs per year

AVIATION-SPECIFIC EXTENSIONS
-----------------------------
  AOG Tier    : For AOG-critical parts, we use a Z=2.576 (99% service level)
                instead of the standard 1.645 (95%) — zero stockouts tolerated

  ABC Policy  : Class-A → monthly review; Class-B → quarterly; Class-C → annual
                Each class gets different holding cost percentages

  Cost Saving : Compares our EOQ policy total cost vs the "current policy"
                (over-ordering by 2× and under-ordering for C items) to
                quantify the $-value of optimisation

USAGE
-----
    from src.models.inventory import optimise_inventory, cost_saving_summary

AUTHOR  : Aviation SC Analytics Project
VERSION : 1.0.0 (Phase 4)
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

log = logging.getLogger("aviation_sc.models.inventory")

# ── Industry-standard service level Z-scores ──────────────────────────────────
Z_SCORES = {
    "CRITICAL": 2.576,   # 99%   service level — AOG parts: zero stockouts
    "HIGH":     1.960,   # 97.5% service level — Class-A non-AOG
    "STANDARD": 1.645,   # 95%   service level — Class-B/C
}

# ── Holding cost % of unit cost per year (MRO industry benchmark) ─────────────
HOLDING_PCT = {
    "Engine Component":   0.28,   # High value → high opportunity cost
    "Avionics":           0.26,
    "Landing Gear":       0.25,
    "Airframe Component": 0.22,
    "Consumable":         0.18,   # Low-cost items → lower holding burden
}

# ── Ordering cost per purchase order by category ─────────────────────────────
ORDER_COST = {
    "Engine Component":   2_500,
    "Avionics":           1_800,
    "Landing Gear":       2_000,
    "Airframe Component": 1_200,
    "Consumable":          350,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CORE FORMULAS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_eoq(
    annual_demand: float,
    order_cost: float,
    unit_cost: float,
    holding_pct: float,
) -> float:
    """
    Economic Order Quantity — Wilson formula (1913, still industry standard).

    EOQ = sqrt(2 × D × S / H)
    where:
      D = annual demand (units)
      S = cost per order placed ($)
      H = annual holding cost per unit = unit_cost × holding_pct

    This formula minimises: (D/Q × S) + (Q/2 × H)
    The first term = total ordering cost/year (decreases as Q rises)
    The second term = total holding cost/year (increases as Q rises)
    EOQ is where these two curves cross — the total cost minimum.

    Parameters
    ----------
    annual_demand : float  Units demanded per year
    order_cost    : float  Cost to place one purchase order ($)
    unit_cost     : float  Unit cost of the part ($)
    holding_pct   : float  Holding cost as fraction of unit cost (e.g. 0.25)

    Returns
    -------
    float  Optimal order quantity (rounded up to nearest whole unit)
    """
    H   = unit_cost * holding_pct
    D   = max(annual_demand, 1)   # Avoid division by zero
    eoq = np.sqrt((2 * D * order_cost) / max(H, 0.01))
    return max(1.0, round(eoq, 1))


def calculate_safety_stock(
    demand_std_monthly: float,
    lead_time_days: float,
    criticality_tier: str = "STANDARD",
) -> float:
    """
    Safety Stock formula: SS = Z × σ_demand × sqrt(lead_time_months)

    where:
      Z               = service-level Z-score (higher → more buffer)
      σ_demand        = std deviation of monthly demand
      lead_time_months = lead_time_days / 30

    Parameters
    ----------
    demand_std_monthly : float  Std dev of monthly demand
    lead_time_days     : float  Supplier lead time in days
    criticality_tier   : str    "CRITICAL", "HIGH", or "STANDARD"

    Returns
    -------
    float  Safety stock in units (always ≥ 0)
    """
    Z                = Z_SCORES.get(criticality_tier, 1.645)
    lead_time_months = lead_time_days / 30.0
    ss = Z * demand_std_monthly * np.sqrt(lead_time_months)
    return max(0.0, round(ss, 1))


def calculate_reorder_point(
    avg_monthly_demand: float,
    lead_time_days: float,
    safety_stock: float,
) -> float:
    """
    Reorder Point (ROP) = demand during lead time + safety stock.

    When inventory drops to ROP, place a new order now so it arrives
    before we run out.

    ROP = (avg_monthly_demand × lead_time_months) + safety_stock

    Parameters
    ----------
    avg_monthly_demand : float  Average units demanded per month
    lead_time_days     : float  Supplier lead time in days
    safety_stock       : float  Pre-computed safety stock units

    Returns
    -------
    float  Reorder point in units
    """
    demand_during_lt = avg_monthly_demand * (lead_time_days / 30.0)
    return max(0.0, round(demand_during_lt + safety_stock, 1))


def calculate_total_inventory_cost(
    annual_demand: float,
    eoq: float,
    order_cost: float,
    unit_cost: float,
    holding_pct: float,
    safety_stock: float,
) -> float:
    """
    Total Cost of Inventory (TCI) per year — used to quantify policy savings.

    TCI = ordering_cost + holding_cost
        = (D/Q × S) + ((Q/2 + SS) × unit_cost × holding_pct)

    Parameters
    ----------
    annual_demand : float  Units/year
    eoq           : float  Order quantity
    order_cost    : float  Cost per order
    unit_cost     : float  Part unit cost
    holding_pct   : float  Holding cost fraction
    safety_stock  : float  Safety stock units

    Returns
    -------
    float  Total annual inventory cost in USD
    """
    D   = max(annual_demand, 1)
    Q   = max(eoq, 1)
    H   = unit_cost * holding_pct
    ordering_cost = (D / Q) * order_cost
    holding_cost  = (Q / 2 + safety_stock) * H
    return round(ordering_cost + holding_cost, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MAIN OPTIMISATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def optimise_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate optimal EOQ, safety stock, ROP, and TCI for every SKU.

    Also calculates a "current policy" TCI (naive over-ordering for A items,
    under-ordering for C items) to quantify the dollar saving from our policy.

    Parameters
    ----------
    df : pd.DataFrame  Synthetic MRO data with criticality_tier column.
                       Use output of clean.clean_mro_synthetic().

    Returns
    -------
    pd.DataFrame  One row per SKU (not per SKU × month).
                  Columns: sku_id, category, abc_class, criticality_tier,
                           unit_cost, avg_monthly_demand, demand_std,
                           eoq_optimal, safety_stock_optimal, rop,
                           tci_optimal_usd, tci_current_usd, saving_usd,
                           saving_pct, review_frequency
    """
    log.info(f"🔄 Optimising inventory for {df['sku_id'].nunique()} SKUs...")

    # Aggregate to one row per SKU (average across months)
    sku_agg = (
        df.groupby(["sku_id", "category", "abc_class",
                    "aog_critical", "unit_cost_usd", "lead_time_days"])
        .agg(
            avg_monthly_demand=("monthly_demand", "mean"),
            demand_std=("monthly_demand", "std"),
            annual_demand=("annual_demand", "first"),
        )
        .reset_index()
    )
    sku_agg["demand_std"] = sku_agg["demand_std"].fillna(1.0)

    # Map criticality tier
    def tier(row):
        if row["aog_critical"]:
            return "CRITICAL"
        elif row["abc_class"] == "A":
            return "HIGH"
        return "STANDARD"

    sku_agg["criticality_tier"] = sku_agg.apply(tier, axis=1)

    records = []
    for _, row in sku_agg.iterrows():
        cat     = row["category"]
        hld_pct = HOLDING_PCT.get(cat, 0.25)
        ord_cst = ORDER_COST.get(cat, 1_000)

        # ── OPTIMAL POLICY ────────────────────────────────────────────────────
        eoq_opt = calculate_eoq(
            row["annual_demand"], ord_cst,
            row["unit_cost_usd"], hld_pct,
        )
        ss_opt = calculate_safety_stock(
            row["demand_std"], row["lead_time_days"],
            row["criticality_tier"],
        )
        rop = calculate_reorder_point(
            row["avg_monthly_demand"], row["lead_time_days"], ss_opt,
        )
        tci_opt = calculate_total_inventory_cost(
            row["annual_demand"], eoq_opt, ord_cst,
            row["unit_cost_usd"], hld_pct, ss_opt,
        )

        # ── CURRENT (NAIVE) POLICY ────────────────────────────────────────────
        # Industry reality: A items over-ordered 2×, C items ordered annually
        if row["abc_class"] == "A":
            naive_q  = eoq_opt * 2          # Over-ordering (excess holding cost)
            naive_ss = ss_opt * 1.5         # Over-buffered
        elif row["abc_class"] == "B":
            naive_q  = eoq_opt * 1.3
            naive_ss = ss_opt * 1.1
        else:
            naive_q  = row["annual_demand"]  # Annual bulk order (under-ordering freq)
            naive_ss = ss_opt * 0.5          # Under-buffered

        tci_current = calculate_total_inventory_cost(
            row["annual_demand"], max(naive_q, 1), ord_cst,
            row["unit_cost_usd"], hld_pct, naive_ss,
        )

        saving     = round(tci_current - tci_opt, 2)
        saving_pct = round(saving / max(tci_current, 1) * 100, 1)

        # ── Review frequency by ABC class ─────────────────────────────────────
        review = {"A": "Monthly", "B": "Quarterly", "C": "Annual"}.get(
            row["abc_class"], "Quarterly"
        )

        records.append({
            "sku_id":              row["sku_id"],
            "category":            cat,
            "abc_class":           row["abc_class"],
            "criticality_tier":    row["criticality_tier"],
            "unit_cost_usd":       round(row["unit_cost_usd"], 2),
            "lead_time_days":      row["lead_time_days"],
            "avg_monthly_demand":  round(row["avg_monthly_demand"], 1),
            "demand_std":          round(row["demand_std"], 2),
            "annual_demand":       int(row["annual_demand"]),
            "eoq_optimal":         int(eoq_opt),
            "safety_stock_optimal":int(ss_opt),
            "reorder_point":       int(rop),
            "tci_optimal_usd":     tci_opt,
            "tci_current_usd":     tci_current,
            "saving_usd":          saving,
            "saving_pct":          saving_pct,
            "review_frequency":    review,
        })

    result_df = pd.DataFrame(records)
    total_saving = result_df["saving_usd"].sum()
    log.info(f"✅ Inventory optimisation complete: "
             f"total annual saving = ${total_saving:,.0f} "
             f"across {len(result_df)} SKUs")
    return result_df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. COST SAVING SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def cost_saving_summary(inv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate inventory cost savings by ABC class and criticality tier.

    Returns a portfolio-ready summary table for the dashboard and report.

    Returns
    -------
    pd.DataFrame  Columns: segment, n_skus, tci_current, tci_optimal,
                           total_saving, avg_saving_pct
    """
    summary = (
        inv_df.groupby(["abc_class", "criticality_tier"])
        .agg(
            n_skus=("sku_id", "count"),
            tci_current_total=("tci_current_usd", "sum"),
            tci_optimal_total=("tci_optimal_usd", "sum"),
            total_saving=("saving_usd", "sum"),
            avg_saving_pct=("saving_pct", "mean"),
        )
        .reset_index()
        .sort_values("total_saving", ascending=False)
    )
    summary["tci_current_total"] = summary["tci_current_total"].round(0)
    summary["tci_optimal_total"] = summary["tci_optimal_total"].round(0)
    summary["total_saving"]      = summary["total_saving"].round(0)
    summary["avg_saving_pct"]    = summary["avg_saving_pct"].round(1)
    return summary


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from src.data.ingest import generate_mro_synthetic
    from src.data.clean  import clean_mro_synthetic

    print("\n=== INVENTORY OPTIMISATION SELF-TEST ===\n")
    raw  = generate_mro_synthetic(n_skus=50, n_months=24, save=False)
    clean_df, _ = clean_mro_synthetic(raw)

    inv = optimise_inventory(clean_df)
    print(f"SKUs optimised: {len(inv)}")
    print(f"\nTop 8 by saving (USD):")
    print(inv.nlargest(8, "saving_usd")[
        ["sku_id","category","abc_class","criticality_tier",
         "eoq_optimal","safety_stock_optimal","tci_optimal_usd",
         "tci_current_usd","saving_usd","saving_pct"]
    ].to_string(index=False))

    summary = cost_saving_summary(inv)
    print(f"\nSaving summary by segment:")
    print(summary.to_string(index=False))
    print(f"\nTotal annual saving: ${inv['saving_usd'].sum():,.0f}")
    print("\n✅ inventory.py self-test passed.")
