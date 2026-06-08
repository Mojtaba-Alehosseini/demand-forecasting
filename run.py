"""End-to-end: baselines vs SARIMA vs LightGBM, rolling-origin backtest, plot.
Run:  python run.py
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import forecast as F

HORIZON = 90
os.makedirs("reports", exist_ok=True)

def main():
    raw = F.load_daily("data/day.csv")
    print(f"Loaded {len(raw)} days: {raw['date'].min().date()} -> {raw['date'].max().date()}")
    split = len(raw) - HORIZON
    feat = F.build_features(raw)
    train, test = feat.iloc[:split].copy(), feat.iloc[split:].copy()
    y = test[F.TARGET].values

    preds = {"SeasonalNaive": F.seasonal_naive(raw[F.TARGET].iloc[:split], HORIZON),
             "SARIMA": F.fit_predict_sarima(raw[F.TARGET].iloc[:split], HORIZON)}
    model, feats = F.fit_lgbm(train)
    preds["LightGBM"] = model.predict(test[feats])

    table = pd.DataFrame({n: F.all_metrics(y, p) for n, p in preds.items()}).T.round(2)
    table.to_csv("reports/holdout_metrics.csv")
    print("\n== Holdout metrics (last 90 days) ==\n", table)

    bt = F.rolling_origin_backtest(raw, 14, 6).round(2)
    bt.to_csv("reports/backtest_folds.csv", index=False)
    print("\n== Rolling-origin backtest ==\n", bt.to_string(index=False))
    print(f"\nMean WAPE  seasonal-naive={bt['WAPE_seasonal_naive'].mean():.2f}%  "
          f"lightgbm={bt['WAPE_lightgbm'].mean():.2f}%")

    plt.figure(figsize=(11, 4.5))
    plt.plot(test["date"], y, label="Actual", color="#1A1A1A", lw=1.6)
    plt.plot(test["date"], preds["SeasonalNaive"], label="Seasonal-naive", color="#BBBBBB", lw=1.2)
    plt.plot(test["date"], preds["LightGBM"], label="LightGBM", color="#1F6F4A", lw=1.6)
    plt.title("Daily demand — actual vs forecast (90-day holdout)"); plt.legend(); plt.tight_layout()
    plt.savefig("reports/forecast_vs_actual.png", dpi=130)
    print("\nSaved reports/forecast_vs_actual.png")

if __name__ == "__main__":
    main()
