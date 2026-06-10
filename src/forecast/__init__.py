"""Demand forecasting — classical baselines vs. machine learning.

Modules: data loading, leakage-safe feature engineering, models
(seasonal-naive, SARIMA, LightGBM), a rolling-origin backtest, and metrics.
"""
from .data import TARGET, SEASON, load_daily
from .features import LAGS, ROLLS, build_features, feature_columns
from .models import seasonal_naive, fit_predict_sarima, fit_lgbm
from .metrics import mae, rmse, mape, wape, all_metrics
from .backtest import rolling_origin_backtest

__all__ = ["TARGET", "SEASON", "LAGS", "ROLLS", "load_daily", "build_features",
           "feature_columns", "seasonal_naive", "fit_predict_sarima", "fit_lgbm",
           "mae", "rmse", "mape", "wape", "all_metrics", "rolling_origin_backtest"]
