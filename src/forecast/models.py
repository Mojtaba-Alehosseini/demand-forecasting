"""Models: seasonal-naive baseline, SARIMA benchmark, LightGBM on engineered features."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .data import SEASON, TARGET
from .features import feature_columns


def seasonal_naive(history: pd.Series, horizon: int) -> np.ndarray:
    last = history.values
    return np.array([last[-SEASON + (i % SEASON)] for i in range(horizon)])


def fit_predict_sarima(train_y: pd.Series, horizon: int) -> np.ndarray:
    res = SARIMAX(train_y, order=(1, 1, 1), seasonal_order=(1, 1, 1, SEASON),
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return np.asarray(res.forecast(steps=horizon))


def fit_lgbm(train_df: pd.DataFrame):
    feats = feature_columns(train_df)
    tr = train_df.dropna(subset=feats + [TARGET])
    model = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=42, verbosity=-1)
    model.fit(tr[feats], tr[TARGET])
    return model, feats
