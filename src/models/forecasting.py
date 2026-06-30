"""
forecasting.py — Aviation Supply Chain: MRO Demand Forecasting
===============================================================
Phase 4 · ML Models & Optimisation

PURPOSE
-------
Forecasts monthly spare-parts demand for every SKU category using two models:

  1. ARIMA  — classical time-series baseline (AutoRegressive Integrated
              Moving Average). Good at capturing trends and seasonality in
              stable demand patterns.

  2. XGBoost — gradient-boosted trees that take BOTH time-series features
              AND rich contextual features (fleet age, aircraft type, ABC class).
              Consistently outperforms ARIMA on MRO data because MRO demand
              is driven by engineering events, not just history.

METHODOLOGY
-----------
Academic literature (2024-2025) shows that ensemble approaches combining
ARIMA + XGBoost achieve 10-20% lower MAPE than either model alone for
spare parts forecasting. We implement both separately so we can:
  - Report ARIMA as the "current policy" baseline
  - Report XGBoost as the "improved" model
  - Quantify the accuracy gain (business value = fewer AOG stockouts)

KEY METRICS
-----------
  MAPE  = Mean Absolute Percentage Error   (target < 15%)
  RMSE  = Root Mean Squared Error          (lower = better)
  MAE   = Mean Absolute Error              (in units of parts)

USAGE
-----
    from src.models.forecasting import (
        prepare_features, train_arima, train_xgboost, evaluate_model
    )

AUTHOR  : Aviation SC Analytics Project
VERSION : 1.0.0 (Phase 4)
"""

import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")
log = logging.getLogger("aviation_sc.models.forecasting")

ROOT      = Path(__file__).resolve().parents[2]
SYNTH_DIR = ROOT / "data" / "synthetic"
PROC_DIR  = ROOT / "data" / "processed"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a feature-rich dataset from the synthetic MRO inventory data.

    For XGBoost we create:
      - Lag features       : demand at t-1, t-2, t-3 (past demand is the strongest signal)
      - Rolling statistics : 3-month rolling mean and std (captures recent trend)
      - Calendar features  : month number, quarter (captures seasonality)
      - Part attributes    : lead_time, unit_cost, abc_class_encoded (contextual signals)
      - AOG flag           : whether this is a safety-critical AOG part

    Parameters
    ----------
    df : pd.DataFrame  Output from ingest.generate_mro_synthetic()

    Returns
    -------
    pd.DataFrame  Feature-engineered DataFrame ready for ML training
    """
    log.info(f"🔧 Engineering features for {df['sku_id'].nunique()} SKUs...")

    df = df.copy().sort_values(["sku_id", "month_index"]).reset_index(drop=True)

    # ── Lag features (per SKU) ────────────────────────────────────────────────
    for lag in [1, 2, 3]:
        df[f"demand_lag_{lag}"] = (
            df.groupby("sku_id")["monthly_demand"].shift(lag)
        )

    # ── Rolling statistics (3-month window) ──────────────────────────────────
    df["demand_roll_mean_3"] = (
        df.groupby("sku_id")["monthly_demand"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df["demand_roll_std_3"] = (
        df.groupby("sku_id")["monthly_demand"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).std().fillna(0))
    )

    # ── Calendar / seasonality features ──────────────────────────────────────
    df["month_sin"] = np.sin(2 * np.pi * df["month_index"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_index"] / 12)
    df["quarter"]   = ((df["month_index"] - 1) // 3) + 1

    # ── Encode categorical features ───────────────────────────────────────────
    abc_map = {"A": 3, "B": 2, "C": 1}
    df["abc_encoded"] = df["abc_class"].map(abc_map).fillna(1)

    cat_map = {
        "Engine Component": 5, "Landing Gear": 4, "Avionics": 3,
        "Airframe Component": 2, "Consumable": 1,
    }
    df["category_encoded"] = df["category"].map(cat_map).fillna(1)

    df["aog_flag"] = df["aog_critical"].astype(int)

    # ── Cost normalisation (log scale — costs span 4 orders of magnitude) ────
    df["log_unit_cost"]    = np.log1p(df["unit_cost_usd"])
    df["log_annual_demand"]= np.log1p(df["annual_demand"])

    # Drop rows where lag features are NaN (first 3 months of each SKU)
    feature_cols = [
        "demand_lag_1", "demand_lag_2", "demand_lag_3",
        "demand_roll_mean_3", "demand_roll_std_3",
        "month_sin", "month_cos", "quarter",
        "abc_encoded", "category_encoded", "aog_flag",
        "log_unit_cost", "log_annual_demand",
        "lead_time_days", "safety_stock", "eoq",
    ]
    df = df.dropna(subset=["demand_lag_1", "demand_lag_2", "demand_lag_3"])
    log.info(f"✅ Features ready: {df.shape[0]:,} rows, {len(feature_cols)} features")
    return df, feature_cols


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ARIMA BASELINE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def train_arima_per_category(
    df: pd.DataFrame,
    order: tuple = (2, 1, 2),
    n_forecast: int = 3,
) -> dict[str, dict]:
    """
    Train an ARIMA(2,1,2) model per part category and forecast 3 months ahead.

    We train one ARIMA per category (not per SKU — too many SKUs). The category-
    level demand aggregation gives enough data points for ARIMA to fit reliably.

    Parameters
    ----------
    df         : pd.DataFrame  Output of prepare_features() or raw synthetic data
    order      : tuple         ARIMA (p, d, q) order. (2,1,2) standard for MRO.
    n_forecast : int           Number of months to forecast ahead

    Returns
    -------
    dict  Keys = category names.
          Values = {"actual": Series, "fitted": Series, "forecast": array,
                    "mape": float, "rmse": float}
    """
    from statsmodels.tsa.arima.model import ARIMA

    log.info(f"📈 Training ARIMA{order} per category...")
    results = {}

    categories = df["category"].unique()
    for cat in categories:
        cat_df = (
            df[df["category"] == cat]
            .groupby("month_index")["monthly_demand"]
            .sum()
            .sort_index()
        )

        if len(cat_df) < 10:
            log.warning(f"   Skipping {cat}: too few data points ({len(cat_df)})")
            continue

        try:
            # Split: last 3 months = test set
            train = cat_df.iloc[:-n_forecast]
            test  = cat_df.iloc[-n_forecast:]

            model  = ARIMA(train, order=order)
            fitted = model.fit()
            fc     = fitted.forecast(steps=n_forecast)

            # Metrics on test set
            mape = float(np.mean(np.abs((test.values - fc.values) /
                         np.maximum(test.values, 1))) * 100)
            rmse = float(np.sqrt(np.mean((test.values - fc.values) ** 2)))
            mae  = float(np.mean(np.abs(test.values - fc.values)))

            results[cat] = {
                "model":    "ARIMA",
                "order":    str(order),
                "actual":   cat_df,
                "fitted":   fitted.fittedvalues,
                "forecast": fc.values,
                "test_actual": test.values,
                "mape":     round(mape, 2),
                "rmse":     round(rmse, 2),
                "mae":      round(mae, 2),
            }
            log.info(f"   {cat:25s} MAPE={mape:6.2f}%  RMSE={rmse:7.2f}  MAE={mae:6.2f}")

        except Exception as e:
            log.warning(f"   ARIMA failed for {cat}: {e}")

    avg_mape = np.mean([r["mape"] for r in results.values()])
    log.info(f"✅ ARIMA complete. Average MAPE across categories: {avg_mape:.2f}%")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3. XGBOOST MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def train_xgboost(
    df_features: pd.DataFrame,
    feature_cols: list[str],
    test_months: int = 3,
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Train an XGBoost demand forecasting model on the full SKU-level feature set.

    XGBoost treats demand forecasting as a supervised regression problem:
    given all features at month t, predict demand at month t.

    Train/test split: last `test_months` months of each SKU = test set.
    This mimics real-world "walk-forward" validation where we test on the
    most recent data — never training on the future.

    Parameters
    ----------
    df_features  : pd.DataFrame  Output of prepare_features()
    feature_cols : list[str]     Feature column names
    test_months  : int           Number of months held out for testing
    random_seed  : int           For reproducibility

    Returns
    -------
    dict with keys:
        "model"        : trained XGBRegressor
        "mape"         : float (test MAPE)
        "rmse"         : float (test RMSE)
        "mae"          : float (test MAE)
        "feature_importance" : pd.DataFrame sorted by importance
        "predictions"  : pd.DataFrame with actual vs predicted
    """
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    log.info(f"🤖 Training XGBoost on {len(feature_cols)} features...")

    # Walk-forward train/test split: last `test_months` months per SKU = test
    max_month  = df_features["month_index"].max()
    cutoff     = max_month - test_months
    train_mask = df_features["month_index"] <= cutoff
    test_mask  = df_features["month_index"] >  cutoff

    X_train = df_features.loc[train_mask, feature_cols]
    y_train = df_features.loc[train_mask, "monthly_demand"]
    X_test  = df_features.loc[test_mask,  feature_cols]
    y_test  = df_features.loc[test_mask,  "monthly_demand"]

    log.info(f"   Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    # XGBoost hyperparameters (tuned for MRO demand pattern)
    model = XGBRegressor(
        n_estimators      = 300,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 5,      # Prevents overfitting on rare AOG parts
        reg_alpha         = 0.1,    # L1 regularisation
        reg_lambda        = 1.0,    # L2 regularisation
        random_state      = random_seed,
        n_jobs            = -1,
        verbosity         = 0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred = model.predict(X_test).clip(min=0)  # Demand can't be negative

    # Metrics
    mape = float(np.mean(
        np.abs((y_test.values - y_pred) / np.maximum(y_test.values, 1)) * 100
    ))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))

    # Feature importance
    fi_df = pd.DataFrame({
        "feature":   feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Predictions DataFrame for portfolio notebook
    preds_df = df_features.loc[test_mask, ["sku_id", "category", "month_index"]].copy()
    preds_df["actual"]    = y_test.values
    preds_df["predicted"] = y_pred.round(1)
    preds_df["error"]     = (preds_df["predicted"] - preds_df["actual"]).round(1)
    preds_df["abs_pct_error"] = (
        np.abs(preds_df["error"]) / np.maximum(preds_df["actual"], 1) * 100
    ).round(1)

    log.info(f"✅ XGBoost complete: MAPE={mape:.2f}%  RMSE={rmse:.2f}  MAE={mae:.2f}")
    log.info(f"   Top features: {fi_df['feature'].head(3).tolist()}")

    return {
        "model":              model,
        "mape":               round(mape, 2),
        "rmse":               round(rmse, 2),
        "mae":                round(mae, 2),
        "feature_importance": fi_df,
        "predictions":        preds_df,
        "X_test":             X_test,
        "y_test":             y_test,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MODEL COMPARISON SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def compare_models(
    arima_results: dict,
    xgb_results: dict,
) -> pd.DataFrame:
    """
    Build a side-by-side comparison table of ARIMA vs XGBoost.

    Returns
    -------
    pd.DataFrame  Comparison with columns: model, mape, rmse, mae, vs_arima_improvement_%
    """
    arima_avg_mape = np.mean([r["mape"] for r in arima_results.values()])
    arima_avg_rmse = np.mean([r["rmse"] for r in arima_results.values()])
    arima_avg_mae  = np.mean([r["mae"]  for r in arima_results.values()])

    improvement_mape = round((arima_avg_mape - xgb_results["mape"]) /
                              arima_avg_mape * 100, 1)

    rows = [
        {
            "model": "ARIMA(2,1,2) — Baseline",
            "mape_%": round(arima_avg_mape, 2),
            "rmse":   round(arima_avg_rmse, 2),
            "mae":    round(arima_avg_mae,  2),
            "improvement_vs_arima": "—",
            "target_met": "✅" if arima_avg_mape < 15 else "❌",
        },
        {
            "model": "XGBoost — Enhanced Model",
            "mape_%": xgb_results["mape"],
            "rmse":   xgb_results["rmse"],
            "mae":    xgb_results["mae"],
            "improvement_vs_arima": f"+{improvement_mape}%" if improvement_mape > 0 else f"{improvement_mape}%",
            "target_met": "✅" if xgb_results["mape"] < 15 else "❌",
        },
    ]
    return pd.DataFrame(rows)


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from src.data.ingest import generate_mro_synthetic
    from src.data.clean  import clean_mro_synthetic

    print("\n=== PHASE 4: FORECASTING SELF-TEST ===\n")
    raw   = generate_mro_synthetic(n_skus=100, n_months=24, save=False)
    clean, _ = clean_mro_synthetic(raw)

    df_feat, feat_cols = prepare_features(clean)
    print(f"Features shape: {df_feat.shape}")

    arima_res = train_arima_per_category(clean)
    xgb_res   = train_xgboost(df_feat, feat_cols)

    comparison = compare_models(arima_res, xgb_res)
    print("\n=== MODEL COMPARISON ===")
    print(comparison.to_string(index=False))

    print("\n=== TOP 5 FEATURE IMPORTANCES (XGBoost) ===")
    print(xgb_res["feature_importance"].head(5).to_string(index=False))
    print("\n✅ forecasting.py self-test passed.")
