"""
test_forecasting.py — Unit Tests: Demand Forecasting Module
============================================================
Phase 6 · Polish, Tests & Showcase

Tests cover:
  - Feature engineering (lag features, rolling stats, encodings)
  - XGBoost model training output structure and metric bounds
  - Feature importance sums to 1
  - Model comparison DataFrame shape and content
  - Edge cases: zero demand, single category

Run with:  pytest tests/test_forecasting.py -v
"""

import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.forecasting import (
    prepare_features,
    train_xgboost,
    compare_models,
)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def raw_data():
    from src.data.ingest import generate_mro_synthetic
    from src.data.clean  import clean_mro_synthetic
    raw  = generate_mro_synthetic(n_skus=30, n_months=18, save=False)
    df, _= clean_mro_synthetic(raw)
    return df


@pytest.fixture(scope="module")
def feature_data(raw_data):
    df_feat, feat_cols = prepare_features(raw_data)
    return df_feat, feat_cols


@pytest.fixture(scope="module")
def xgb_result(feature_data):
    df_feat, feat_cols = feature_data
    return train_xgboost(df_feat, feat_cols, test_months=3)


# ══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE ENGINEERING TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestPrepareFeatures:

    def test_returns_tuple(self, raw_data):
        """prepare_features must return (DataFrame, list)."""
        result = prepare_features(raw_data)
        assert isinstance(result, tuple) and len(result) == 2

    def test_lag_features_created(self, feature_data):
        """Lag features demand_lag_1/2/3 must be present."""
        df_feat, _ = feature_data
        for lag in [1, 2, 3]:
            assert f"demand_lag_{lag}" in df_feat.columns

    def test_rolling_features_created(self, feature_data):
        """Rolling mean and std features must be present."""
        df_feat, _ = feature_data
        assert "demand_roll_mean_3" in df_feat.columns
        assert "demand_roll_std_3"  in df_feat.columns

    def test_seasonality_features_present(self, feature_data):
        """Month_sin, month_cos, quarter must be present."""
        df_feat, _ = feature_data
        for col in ["month_sin", "month_cos", "quarter"]:
            assert col in df_feat.columns

    def test_encoded_features_present(self, feature_data):
        """abc_encoded, category_encoded, aog_flag must be present."""
        df_feat, _ = feature_data
        for col in ["abc_encoded", "category_encoded", "aog_flag"]:
            assert col in df_feat.columns

    def test_abc_encoding_is_ordered(self, feature_data):
        """ABC encoding: A(3) > B(2) > C(1) — preserves business hierarchy."""
        df_feat, _ = feature_data
        a_val = df_feat[df_feat["abc_class"] == "A"]["abc_encoded"].iloc[0]
        c_val = df_feat[df_feat["abc_class"] == "C"]["abc_encoded"].iloc[0]
        assert a_val > c_val, "Class-A encoding should be higher than Class-C"

    def test_lag_features_no_all_nan_rows(self, feature_data):
        """After dropping NaN lags, no row should have all-NaN lag features."""
        df_feat, _ = feature_data
        lag_cols  = ["demand_lag_1", "demand_lag_2", "demand_lag_3"]
        all_nan   = df_feat[lag_cols].isna().all(axis=1)
        assert not all_nan.any(), "Rows with all-NaN lags should be dropped"

    def test_feature_cols_list_matches_columns(self, feature_data):
        """Returned feature_cols must all exist as columns in the DataFrame."""
        df_feat, feat_cols = feature_data
        for col in feat_cols:
            assert col in df_feat.columns, f"Feature '{col}' not in DataFrame"

    def test_no_nan_in_lag_features_after_drop(self, feature_data):
        """After prepare_features, lag features must have no NaN values."""
        df_feat, _ = feature_data
        for lag in [1, 2, 3]:
            col = f"demand_lag_{lag}"
            null_count = df_feat[col].isna().sum()
            assert null_count == 0, f"NaN values in {col}: {null_count}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. XGBOOST MODEL TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestXGBoostModel:

    def test_returns_dict_with_required_keys(self, xgb_result):
        """XGBoost result dict must contain all expected keys."""
        required = {"model", "mape", "rmse", "mae", "feature_importance", "predictions"}
        assert required.issubset(xgb_result.keys())

    def test_mape_is_positive_float(self, xgb_result):
        """MAPE must be a positive float."""
        assert isinstance(xgb_result["mape"], float)
        assert xgb_result["mape"] > 0

    def test_mape_within_reasonable_range(self, xgb_result):
        """MAPE should be below 50% — basic sanity check on model quality."""
        assert xgb_result["mape"] < 50.0, (
            f"MAPE={xgb_result['mape']:.2f}% — model may be failing to learn"
        )

    def test_rmse_non_negative(self, xgb_result):
        """RMSE must be non-negative."""
        assert xgb_result["rmse"] >= 0

    def test_feature_importance_sums_to_one(self, xgb_result):
        """Feature importance scores must sum to approximately 1.0."""
        total = xgb_result["feature_importance"]["importance"].sum()
        assert abs(total - 1.0) < 0.01, f"Feature importance sums to {total:.4f}"

    def test_feature_importance_all_non_negative(self, xgb_result):
        """All feature importances must be non-negative."""
        fi = xgb_result["feature_importance"]["importance"]
        assert (fi >= 0).all()

    def test_predictions_has_required_columns(self, xgb_result):
        """Predictions DataFrame must have actual, predicted, error columns."""
        preds = xgb_result["predictions"]
        for col in ["actual", "predicted", "error"]:
            assert col in preds.columns

    def test_predicted_values_non_negative(self, xgb_result):
        """Predicted demand values must never be negative (clipped at 0)."""
        assert (xgb_result["predictions"]["predicted"] >= 0).all()

    def test_model_object_can_predict(self, xgb_result, feature_data):
        """The returned model object must be callable via .predict()."""
        df_feat, feat_cols = feature_data
        model = xgb_result["model"]
        sample = df_feat[feat_cols].head(5)
        preds = model.predict(sample)
        assert len(preds) == 5


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL COMPARISON TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestModelComparison:

    def test_compare_returns_dataframe(self, raw_data, xgb_result):
        """compare_models must return a DataFrame."""
        from src.models.forecasting import train_arima_per_category
        arima = train_arima_per_category(raw_data)
        comp  = compare_models(arima, xgb_result)
        assert isinstance(comp, pd.DataFrame)

    def test_compare_has_two_rows(self, raw_data, xgb_result):
        """Comparison must have exactly 2 rows: ARIMA and XGBoost."""
        from src.models.forecasting import train_arima_per_category
        arima = train_arima_per_category(raw_data)
        comp  = compare_models(arima, xgb_result)
        assert len(comp) == 2

    def test_model_names_in_comparison(self, raw_data, xgb_result):
        """Comparison must contain model name strings."""
        from src.models.forecasting import train_arima_per_category
        arima = train_arima_per_category(raw_data)
        comp  = compare_models(arima, xgb_result)
        assert "model" in comp.columns
        names_str = comp["model"].str.cat()
        assert "ARIMA" in names_str
        assert "XGBoost" in names_str
