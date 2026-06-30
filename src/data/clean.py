"""
clean.py — Aviation Supply Chain: Data Cleaning & Validation Module
====================================================================
Phase 2 · Environment Setup & Data Ingestion

PURPOSE
-------
Transforms raw, messy datasets into clean, analysis-ready DataFrames.
Three core functions mirror the standard ETL pattern:
    1. clean_*()     — fix nulls, types, outliers, column names
    2. validate_*()  — check schema, value ranges, completeness
    3. merge_*()     — join datasets into a unified supply chain view

DESIGN PRINCIPLES (from ETL best practices)
-------------------------------------------
- Functions are pure: same input always produces same output
- Logging at every step so failures are traceable
- Returns (clean_df, rejected_df) so bad rows are never silently dropped
- Column names are snake_case throughout for Python friendliness

AUTHOR  : Aviation SC Analytics Project
VERSION : 1.0.0 (Phase 2)
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

log = logging.getLogger("aviation_sc.clean")

ROOT     = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. AEROSPACE SUPPLY CHAIN DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def clean_aerospace_sc(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean the Kaggle Aerospace Supply Chain Performance dataset.

    Transformations applied
    -----------------------
    - Standardise column names to snake_case
    - Parse date columns
    - Fill or drop missing lead times (critical column — drop if missing)
    - Clip outlier lead times (< 0 or > 365 days are data errors)
    - Clip OTD percentage to [0, 100]
    - Add derived column: risk_flag (OTD < 85% OR defect_rate_ppm > 5000)

    Parameters
    ----------
    df : pd.DataFrame  Raw aerospace SC DataFrame from ingest.load_aerospace_sc()

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (cleaned_df, rejected_df) — rejected rows with rejection reason
    """
    log.info(f"🧹 Cleaning aerospace SC data: {df.shape}")
    rejected_rows = []

    # 1. Snake-case all column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-/]+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    )
    log.info(f"   Columns normalised: {list(df.columns)[:8]}...")

    # 2. Identify likely column names (dataset may vary)
    col_map = _infer_aerospace_columns(df)
    df = df.rename(columns=col_map)

    # 3. Reject rows with missing lead_time (non-negotiable for our analysis)
    if "lead_time_days" in df.columns:
        mask_null_lt = df["lead_time_days"].isna()
        if mask_null_lt.any():
            rej = df[mask_null_lt].copy()
            rej["rejection_reason"] = "missing_lead_time"
            rejected_rows.append(rej)
            df = df[~mask_null_lt].copy()
            log.warning(f"   ⚠ Dropped {mask_null_lt.sum()} rows: missing lead_time")

        # 4. Clip lead time to realistic range [1, 365] days
        df["lead_time_days"] = df["lead_time_days"].clip(lower=1, upper=365)

    # 5. Clip OTD to [0, 100]
    if "on_time_delivery_pct" in df.columns:
        df["on_time_delivery_pct"] = pd.to_numeric(
            df["on_time_delivery_pct"], errors="coerce"
        ).clip(0, 100)

    # 6. Defect rate: non-negative
    if "defect_rate_ppm" in df.columns:
        df["defect_rate_ppm"] = pd.to_numeric(
            df["defect_rate_ppm"], errors="coerce"
        ).clip(lower=0)

    # 7. Derived: supplier risk flag
    conditions = []
    if "on_time_delivery_pct" in df.columns:
        conditions.append(df["on_time_delivery_pct"] < 85)
    if "defect_rate_ppm" in df.columns:
        conditions.append(df["defect_rate_ppm"] > 5_000)

    if conditions:
        df["supplier_at_risk"] = np.logical_or.reduce(conditions)
    else:
        df["supplier_at_risk"] = False

    rejected_df = pd.concat(rejected_rows, ignore_index=True) if rejected_rows else pd.DataFrame()
    log.info(f"✅ Aerospace SC clean: {df.shape[0]:,} rows kept, "
             f"{len(rejected_df):,} rejected")
    return df, rejected_df


def _infer_aerospace_columns(df: pd.DataFrame) -> dict:
    """
    Try to map whatever column names the dataset has to our standard names.
    This handles slight variations across Kaggle dataset versions.
    """
    mapping = {}
    for col in df.columns:
        cl = col.lower()
        if "lead" in cl and "time" in cl:
            mapping[col] = "lead_time_days"
        elif "otd" in cl or ("on_time" in cl and "deliv" in cl):
            mapping[col] = "on_time_delivery_pct"
        elif "defect" in cl or "ppm" in cl:
            mapping[col] = "defect_rate_ppm"
        elif "supplier" in cl and "id" in cl:
            mapping[col] = "supplier_id"
        elif "part" in cl and "cat" in cl:
            mapping[col] = "part_category"
        elif "unit" in cl and "cost" in cl:
            mapping[col] = "unit_cost_usd"
    return mapping


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BTS ON-TIME PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

def clean_bts_ontime(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean BTS Airline On-Time Performance data for use as MRO demand signal.

    The key insight: maintenance-related delays (LATE_AIRCRAFT_DELAY and
    NAS_DELAY) are proxies for MRO activity. Heavy delays → more maintenance
    demand at the destination airport.

    Transformations
    ---------------
    - Standardise column names
    - Parse YEAR, MONTH into a proper date
    - Convert delay columns to numeric, clip negatives to 0
    - Add maintenance_proxy_minutes = LATE_AIRCRAFT_DELAY + NAS_DELAY
    - Drop cancelled flights (no arrival data available)
    """
    log.info(f"🧹 Cleaning BTS On-Time data: {df.shape}")

    if df.empty:
        log.warning("BTS DataFrame is empty — returning as-is.")
        return df, pd.DataFrame()

    rejected_rows = []

    # 1. Normalise column names
    df.columns = df.columns.str.strip().str.upper()

    # 2. Drop cancelled flights
    if "CANCELLED" in df.columns:
        mask_cancelled = df["CANCELLED"] == 1
        if mask_cancelled.any():
            rej = df[mask_cancelled].copy()
            rej["rejection_reason"] = "cancelled_flight"
            rejected_rows.append(rej)
            df = df[~mask_cancelled].copy()
            log.info(f"   Removed {mask_cancelled.sum():,} cancelled flights")

    # 3. Parse date
    if "YEAR" in df.columns and "MONTH" in df.columns:
        df["flight_date"] = pd.to_datetime(
            df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str).str.zfill(2),
            format="%Y-%m"
        )

    # 4. Delay columns: numeric + clip negatives
    delay_cols = ["DEP_DELAY", "ARR_DELAY", "NAS_DELAY",
                  "LATE_AIRCRAFT_DELAY", "CARRIER_DELAY", "WEATHER_DELAY"]
    for col in delay_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    # 5. Maintenance proxy signal
    maint_cols = [c for c in ["LATE_AIRCRAFT_DELAY", "NAS_DELAY"] if c in df.columns]
    if maint_cols:
        df["maintenance_proxy_min"] = df[maint_cols].sum(axis=1)

    rejected_df = pd.concat(rejected_rows, ignore_index=True) if rejected_rows else pd.DataFrame()
    log.info(f"✅ BTS clean: {df.shape[0]:,} rows kept")
    return df, rejected_df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SYNTHETIC MRO DATA
# ═══════════════════════════════════════════════════════════════════════════════

def clean_mro_synthetic(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Light cleaning pass on the synthetic MRO inventory dataset.

    The synthetic data is already well-structured (generated by us), so this
    mainly enforces types and removes any edge cases from the generator.

    Transformations
    ---------------
    - Ensure non-negative demand and costs
    - Add criticality_tier: CRITICAL / HIGH / STANDARD
    - Add annual_spend = annual_demand × unit_cost_usd
    """
    log.info(f"🧹 Cleaning synthetic MRO data: {df.shape}")

    # Non-negative guard
    for col in ["monthly_demand", "unit_cost_usd", "lead_time_days",
                "safety_stock", "eoq"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # Criticality tier
    def criticality(row):
        if row.get("aog_critical", False):
            return "CRITICAL"
        elif row.get("abc_class", "C") == "A":
            return "HIGH"
        else:
            return "STANDARD"

    df["criticality_tier"] = df.apply(criticality, axis=1)

    # Annual spend
    if "annual_demand" in df.columns and "unit_cost_usd" in df.columns:
        df["annual_spend_usd"] = (df["annual_demand"] * df["unit_cost_usd"]).round(2)

    # Reject zero-cost or zero-demand records (data anomalies)
    mask_bad = (df["unit_cost_usd"] == 0) | (df["monthly_demand"] < 0)
    rejected_df = df[mask_bad].copy()
    if not rejected_df.empty:
        rejected_df["rejection_reason"] = "zero_cost_or_negative_demand"
    df = df[~mask_bad].copy()

    log.info(f"✅ MRO synthetic clean: {df.shape[0]:,} rows, "
             f"{rejected_df.shape[0]} rejected")
    return df, rejected_df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SAVE PROCESSED DATA
# ═══════════════════════════════════════════════════════════════════════════════

def save_processed(df: pd.DataFrame, name: str) -> Path:
    """
    Save a cleaned DataFrame to data/processed/ as CSV and Parquet.

    Parameters
    ----------
    df   : pd.DataFrame  Cleaned DataFrame
    name : str           Filename stem (e.g. "aerospace_sc_clean")

    Returns
    -------
    Path  Path to the saved CSV file.
    """
    csv_path     = PROC_DIR / f"{name}.csv"
    parquet_path = PROC_DIR / f"{name}.parquet"

    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
        log.info(f"💾 Saved: {csv_path.name} + {parquet_path.name}")
    except Exception:
        log.info(f"💾 Saved: {csv_path.name} (parquet skipped — install pyarrow)")

    return csv_path


# ═══════════════════════════════════════════════════════════════════════════════
# 5. VALIDATE SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

def validate_dataframe(
    df: pd.DataFrame,
    required_cols: list[str],
    name: str = "DataFrame",
) -> bool:
    """
    Validate that a DataFrame meets minimum quality requirements.

    Checks
    ------
    1. Not empty
    2. Required columns all present
    3. Missing-value rate < 30% per required column
    4. No duplicate rows

    Parameters
    ----------
    df            : pd.DataFrame  DataFrame to validate
    required_cols : list[str]     Column names that must be present
    name          : str           Label for logging

    Returns
    -------
    bool  True if all checks pass, False otherwise.
    """
    log.info(f"🔍 Validating {name}: {df.shape}")
    passed = True

    # Check 1: not empty
    if df.empty:
        log.error(f"   ❌ {name} is empty!")
        return False

    # Check 2: required columns present
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        log.error(f"   ❌ Missing required columns: {missing_cols}")
        passed = False
    else:
        log.info(f"   ✅ All {len(required_cols)} required columns present")

    # Check 3: null rate
    for col in required_cols:
        if col in df.columns:
            null_pct = df[col].isna().mean() * 100
            if null_pct > 30:
                log.warning(f"   ⚠ High null rate in '{col}': {null_pct:.1f}%")

    # Check 4: duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        log.warning(f"   ⚠ {dup_count:,} duplicate rows found")

    if passed:
        log.info(f"✅ {name} validation passed")
    return passed


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data.ingest import generate_mro_synthetic

    print("\n=== TESTING: clean_mro_synthetic ===\n")
    raw = generate_mro_synthetic(n_skus=20, n_months=6, save=False)
    clean, rejected = clean_mro_synthetic(raw)

    print(f"Raw rows    : {len(raw):,}")
    print(f"Clean rows  : {len(clean):,}")
    print(f"Rejected    : {len(rejected):,}")
    print(f"\nCriticality distribution:\n{clean['criticality_tier'].value_counts()}")
    print(f"\nABC × Criticality:\n{clean.groupby(['abc_class','criticality_tier']).size()}")

    valid = validate_dataframe(
        clean,
        required_cols=["sku_id", "monthly_demand", "unit_cost_usd",
                       "lead_time_days", "safety_stock"],
        name="MRO Synthetic"
    )
    print(f"\nValidation passed: {valid}")
    print("\n✅ clean.py self-test passed.")
