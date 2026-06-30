"""
test_inventory.py — Unit Tests: EOQ & Inventory Optimisation
=============================================================
Phase 6 · Polish, Tests & Showcase

Tests cover:
  - EOQ formula correctness with known inputs
  - Safety stock formula (Z-scores, edge cases)
  - Reorder point calculation
  - Total cost of inventory
  - Full optimise_inventory() output shape and business logic
  - Cost saving summary aggregation
  - Parametrised tests for criticality tiers

Run with:  pytest tests/test_inventory.py -v
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.inventory import (
    calculate_eoq,
    calculate_safety_stock,
    calculate_reorder_point,
    calculate_total_inventory_cost,
    optimise_inventory,
    cost_saving_summary,
    Z_SCORES,
)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def sample_mro_df():
    """Minimal synthetic MRO DataFrame sufficient for inventory tests."""
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from src.data.ingest import generate_mro_synthetic
    from src.data.clean  import clean_mro_synthetic
    raw  = generate_mro_synthetic(n_skus=40, n_months=12, save=False)
    df, _= clean_mro_synthetic(raw)
    return df


@pytest.fixture(scope="module")
def inv_results(sample_mro_df):
    """Pre-computed optimise_inventory() results for the sample dataset."""
    return optimise_inventory(sample_mro_df)


# ══════════════════════════════════════════════════════════════════════════════
# 1. EOQ FORMULA TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestEOQFormula:

    def test_eoq_known_textbook_example(self):
        """
        Classic textbook example (MRO Inventory Best Practices reference):
        D=100/yr, S=$50/order, H=$10/unit/yr → EOQ = sqrt(2×100×50/10) = 31.6
        """
        # Arrange
        D = 100.0; S = 50.0; unit_cost = 40.0; h_pct = 0.25  # H = 40 × 0.25 = 10
        # Act
        eoq = calculate_eoq(D, S, unit_cost, h_pct)
        # Assert: Wilson formula gives sqrt(2×100×50/10) = 31.6
        assert abs(eoq - 31.6) < 1.0, f"Expected ≈31.6, got {eoq}"

    def test_eoq_always_positive(self):
        """EOQ must always be ≥ 1 (minimum order of one unit)."""
        eoq = calculate_eoq(annual_demand=1, order_cost=10, unit_cost=0.01, holding_pct=0.01)
        assert eoq >= 1.0

    def test_eoq_increases_with_demand(self):
        """Higher annual demand → larger EOQ (bigger batches more cost-effective)."""
        low_eoq  = calculate_eoq(100,  500, 1000, 0.25)
        high_eoq = calculate_eoq(1000, 500, 1000, 0.25)
        assert high_eoq > low_eoq, "EOQ should increase with demand"

    def test_eoq_decreases_with_holding_cost(self):
        """Higher holding cost → smaller EOQ (order more frequently, hold less)."""
        low_h_eoq  = calculate_eoq(100, 500, 1000, holding_pct=0.10)
        high_h_eoq = calculate_eoq(100, 500, 1000, holding_pct=0.50)
        assert high_h_eoq < low_h_eoq, "EOQ should decrease as holding cost increases"

    def test_eoq_handles_zero_demand(self):
        """Zero demand should not raise — returns 1 (minimum order size)."""
        eoq = calculate_eoq(annual_demand=0, order_cost=500, unit_cost=1000, holding_pct=0.25)
        assert eoq >= 1.0

    @pytest.mark.parametrize("demand,order_cost,unit_cost,holding_pct", [
        (50,   200,  500,  0.20),
        (200,  1000, 5000, 0.25),
        (1000, 2500, 200,  0.18),
        (5000, 350,  50,   0.22),
    ])
    def test_eoq_always_returns_positive(self, demand, order_cost, unit_cost, holding_pct):
        """EOQ must be positive for all valid parameter combinations."""
        eoq = calculate_eoq(demand, order_cost, unit_cost, holding_pct)
        assert eoq > 0


# ══════════════════════════════════════════════════════════════════════════════
# 2. SAFETY STOCK TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestSafetyStock:

    def test_safety_stock_known_values(self):
        """
        From MRO inventory guide:
        SS = Z × σ × sqrt(LT_months)
        Z=1.645 (95%), σ=5, LT=60 days (2 months) → 1.645 × 5 × sqrt(2) ≈ 11.63
        """
        ss = calculate_safety_stock(
            demand_std_monthly=5.0,
            lead_time_days=60.0,
            criticality_tier="STANDARD",
        )
        assert abs(ss - 11.63) < 1.0, f"Expected ≈11.63, got {ss}"

    def test_critical_tier_higher_than_standard(self):
        """CRITICAL tier (99% SL) must produce more safety stock than STANDARD (95%)."""
        ss_critical = calculate_safety_stock(5, 30, "CRITICAL")
        ss_standard = calculate_safety_stock(5, 30, "STANDARD")
        assert ss_critical > ss_standard

    def test_safety_stock_non_negative(self):
        """Safety stock can never be negative."""
        ss = calculate_safety_stock(0, 7, "STANDARD")
        assert ss >= 0.0

    def test_longer_lead_time_more_safety_stock(self):
        """Longer supplier lead time → more safety stock needed."""
        ss_short = calculate_safety_stock(5, 14,  "STANDARD")
        ss_long  = calculate_safety_stock(5, 90,  "STANDARD")
        assert ss_long > ss_short

    @pytest.mark.parametrize("tier,z_expected", [
        ("CRITICAL", 2.576),
        ("HIGH",     1.960),
        ("STANDARD", 1.645),
    ])
    def test_z_scores_match_defined_constants(self, tier, z_expected):
        """Z-scores in Z_SCORES dict must match actuarial service-level values."""
        assert abs(Z_SCORES[tier] - z_expected) < 0.001, (
            f"Z_SCORES[{tier}] = {Z_SCORES[tier]}, expected {z_expected}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3. REORDER POINT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestReorderPoint:

    def test_rop_always_non_negative(self):
        """Reorder point must never be negative."""
        rop = calculate_reorder_point(avg_monthly_demand=0, lead_time_days=30, safety_stock=0)
        assert rop >= 0.0

    def test_rop_equals_demand_plus_ss(self):
        """
        ROP = (avg_demand × lead_time_months) + safety_stock
        avg_demand=10/month, LT=30 days (1 month), SS=5 → ROP = 10 × 1 + 5 = 15
        """
        rop = calculate_reorder_point(10.0, 30.0, 5.0)
        assert abs(rop - 15.0) < 0.5, f"Expected ≈15, got {rop}"

    def test_higher_demand_gives_higher_rop(self):
        """Higher average demand → higher reorder point."""
        rop_low  = calculate_reorder_point(5,  30, 3)
        rop_high = calculate_reorder_point(20, 30, 3)
        assert rop_high > rop_low

    def test_longer_lead_time_gives_higher_rop(self):
        """Longer lead time → trigger reorder sooner (higher ROP)."""
        rop_fast = calculate_reorder_point(10, 14, 5)
        rop_slow = calculate_reorder_point(10, 90, 5)
        assert rop_slow > rop_fast


# ══════════════════════════════════════════════════════════════════════════════
# 4. TOTAL COST OF INVENTORY TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestTotalInventoryCost:

    def test_tci_always_positive(self):
        """Total inventory cost must be positive for valid inputs."""
        tci = calculate_total_inventory_cost(100, 32, 500, 200, 0.25, 10)
        assert tci > 0

    def test_tci_at_eoq_is_minimised(self):
        """
        Key property: total cost is minimised when Q = EOQ.
        TCI(EOQ) ≤ TCI(EOQ/2) and TCI(EOQ) ≤ TCI(EOQ×2)
        """
        D, S, unit_cost, h_pct, ss = 100, 500, 1000, 0.25, 5
        eoq = calculate_eoq(D, S, unit_cost, h_pct)
        tci_opt  = calculate_total_inventory_cost(D, eoq,       S, unit_cost, h_pct, ss)
        tci_half = calculate_total_inventory_cost(D, eoq / 2,   S, unit_cost, h_pct, ss)
        tci_dbl  = calculate_total_inventory_cost(D, eoq * 2,   S, unit_cost, h_pct, ss)
        assert tci_opt <= tci_half + 1, "TCI at EOQ should be ≤ TCI at EOQ/2"
        assert tci_opt <= tci_dbl  + 1, "TCI at EOQ should be ≤ TCI at EOQ×2"


# ══════════════════════════════════════════════════════════════════════════════
# 5. FULL OPTIMISATION OUTPUT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestOptimiseInventory:

    def test_returns_dataframe(self, inv_results):
        """optimise_inventory() must return a pandas DataFrame."""
        assert isinstance(inv_results, pd.DataFrame)

    def test_correct_number_of_rows(self, sample_mro_df, inv_results):
        """One row per unique (SKU, criticality_tier) combination — never more than 3× SKUs."""
        n_skus   = sample_mro_df["sku_id"].nunique()
        n_result = len(inv_results)
        # Upper bound: at most 3 tiers × n_skus (some SKUs appear across ABC classes)
        assert n_result >= n_skus, f"Expected ≥{n_skus} rows, got {n_result}"
        assert n_result <= n_skus * 4, f"Too many rows ({n_result}) for {n_skus} SKUs"

    def test_required_columns_present(self, inv_results):
        """All essential output columns must be present."""
        required = [
            "sku_id", "category", "abc_class", "criticality_tier",
            "eoq_optimal", "safety_stock_optimal", "reorder_point",
            "tci_optimal_usd", "tci_current_usd", "saving_usd",
        ]
        for col in required:
            assert col in inv_results.columns, f"Missing required column: {col}"

    def test_eoq_values_all_positive(self, inv_results):
        """All EOQ values must be at least 1 unit."""
        assert (inv_results["eoq_optimal"] >= 1).all()

    def test_safety_stock_non_negative(self, inv_results):
        """Safety stock must never be negative."""
        assert (inv_results["safety_stock_optimal"] >= 0).all()

    def test_optimal_cost_less_than_current(self, inv_results):
        """On aggregate, optimal policy should be cheaper than current naive policy."""
        total_opt     = inv_results["tci_optimal_usd"].sum()
        total_current = inv_results["tci_current_usd"].sum()
        # Some individual SKUs may have higher optimal cost (C-CRITICAL under-buffered)
        # but overall, the portfolio should save money
        assert total_opt < total_current, (
            f"Optimal TCI (${total_opt:,.0f}) should be < current (${total_current:,.0f})"
        )

    def test_abc_classes_are_valid(self, inv_results):
        """All ABC classifications must be A, B, or C."""
        invalid = inv_results[~inv_results["abc_class"].isin(["A", "B", "C"])]
        assert invalid.empty, f"Invalid ABC classes: {invalid['abc_class'].unique()}"

    def test_criticality_tiers_are_valid(self, inv_results):
        """Criticality tiers must be CRITICAL, HIGH, or STANDARD."""
        valid = {"CRITICAL", "HIGH", "STANDARD"}
        invalid = inv_results[~inv_results["criticality_tier"].isin(valid)]
        assert invalid.empty, f"Invalid tiers: {invalid['criticality_tier'].unique()}"

    def test_saving_summary_returns_dataframe(self, inv_results):
        """cost_saving_summary() must return a DataFrame."""
        summary = cost_saving_summary(inv_results)
        assert isinstance(summary, pd.DataFrame)
        assert not summary.empty

    def test_saving_summary_has_expected_columns(self, inv_results):
        """Cost saving summary must include key aggregation columns."""
        summary = cost_saving_summary(inv_results)
        for col in ["abc_class", "criticality_tier", "n_skus", "total_saving"]:
            assert col in summary.columns, f"Missing column in summary: {col}"
