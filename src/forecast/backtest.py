"""Rolling-origin (expanding-window) cross-validation."""
import pandas as pd

from .data import TARGET
from .features import build_features
from .models import fit_lgbm, seasonal_naive
from .metrics import wape


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
