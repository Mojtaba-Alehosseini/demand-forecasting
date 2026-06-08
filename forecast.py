"""
Demand forecasting — classical baselines vs. machine learning.

A single, self-contained module: data loading, leakage-safe feature engineering,
models (seasonal-naive, SARIMA, LightGBM), a rolling-origin backtest, and metrics.

Dataset: UCI Bike Sharing (daily). Swap `load_daily` to apply to any retail/marketplace
demand series.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX

TARGET = "cnt"
SEASON = 7                      # weekly seasonality in daily data
LAGS = [1, 7, 14]
ROLLS = [7, 14, 28]

# --------------------------------------------------------------------------- data
def load_daily(path: str = "data/day.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["dteday"]).sort_values("dteday").reset_index(drop=True)
    df = df.rename(columns={"dteday": "date"})
    keep = ["date", "season", "holiday", "weekday", "workingday",
            "weathersit", "temp", "atemp", "hum", "windspeed", TARGET]
    return df[keep]

# ----------------------------------------------------------------------- features
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar + weather + leakage-safe autoregressive features (shift >= 1)."""
    out = df.copy()
    d = out["date"].dt
    out["dow"], out["month"] = d.dayofweek, d.month
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    for l in LAGS:
        out[f"lag_{l}"] = out[TARGET].shift(l)
    for w in ROLLS:
        out[f"roll_mean_{w}"] = out[TARGET].shift(1).rolling(w).mean()
        out[f"roll_std_{w}"] = out[TARGET].shift(1).rolling(w).std()
    return out

def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in {"date", TARGET}]

# ------------------------------------------------------------------------- models
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

# ------------------------------------------------------------------------ metrics
def mae(y, p):  return float(np.mean(np.abs(y - p)))
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float); m = y != 0
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100)
def wape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sum(np.abs(y - p)) / np.sum(np.abs(y)) * 100)
def all_metrics(y, p):
    return {"MAE": mae(y, p), "RMSE": rmse(y, p), "MAPE_%": mape(y, p), "WAPE_%": wape(y, p)}

# ----------------------------------------------------------------------- backtest
def rolling_origin_backtest(df_raw: pd.DataFrame, horizon: int = 14, n_folds: int = 6) -> pd.DataFrame:
    """Expanding-window CV: each fold trains on the past and forecasts the next horizon."""
    feat = build_features(df_raw)
    n, rows = len(feat), []
    for k in range(n_folds, 0, -1):
        split = n - k * horizon
        if split < 120:
            continue
        train, test = feat.iloc[:split].copy(), feat.iloc[split:split + horizon].copy()
        sn = seasonal_naive(df_raw[TARGET].iloc[:split], len(test))
        model, feats = fit_lgbm(train)
        rows.append({"fold": n_folds - k + 1,
                     "train_end": str(train["date"].iloc[-1].date()),
                     "WAPE_seasonal_naive": wape(test[TARGET].values, sn),
                     "WAPE_lightgbm": wape(test[TARGET].values, model.predict(test[feats]))})
    return pd.DataFrame(rows)
