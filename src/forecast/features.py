"""Calendar + weather + leakage-safe lag/rolling features."""
import numpy as np
import pandas as pd

from .data import TARGET

LAGS = [1, 7, 14]
ROLLS = [7, 14, 28]


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
