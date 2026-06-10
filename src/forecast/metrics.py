"""Business-facing error metrics: MAE / RMSE / MAPE / WAPE."""
import numpy as np


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = y != 0
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100)


def wape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sum(np.abs(y - p)) / np.sum(np.abs(y)) * 100)


def all_metrics(y, p):
    return {"MAE": mae(y, p), "RMSE": rmse(y, p), "MAPE_%": mape(y, p), "WAPE_%": wape(y, p)}
