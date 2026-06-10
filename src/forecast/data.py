"""Load & prepare the daily demand series.

Dataset: UCI Bike Sharing (daily). Swap `load_daily` to apply to any
retail/marketplace demand series — the rest of the pipeline is agnostic.
"""
import pandas as pd

TARGET = "cnt"
SEASON = 7  # weekly seasonality in daily data


def load_daily(path: str = "data/day.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["dteday"]).sort_values("dteday").reset_index(drop=True)
    df = df.rename(columns={"dteday": "date"})
    keep = ["date", "season", "holiday", "weekday", "workingday",
            "weathersit", "temp", "atemp", "hum", "windspeed", TARGET]
    return df[keep]
