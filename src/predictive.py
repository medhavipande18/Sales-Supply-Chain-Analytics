import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

RAW = os.path.join("data", "raw")
OUT = os.path.join("reports", "outputs")
os.makedirs(OUT, exist_ok=True)

def load_shipments():
    df = pd.read_csv(os.path.join(RAW, "shipments.csv"), parse_dates=["planned_departure"])
    df["week"] = df["planned_departure"].dt.to_period("W").dt.start_time
    return df

def build_weekly_series(df: pd.DataFrame, region: str) -> pd.Series:
    w = (df[(df["lane"] == "W2S") & (df["region"] == region)]
         .groupby("week")["shipment_id"].count()
         .asfreq("W-MON", fill_value=0))
    return w

def fit_forecast(y: pd.Series, steps: int = 2) -> tuple[pd.DataFrame, float]:
    # Simple SARIMAX without exogenous vars for portability
    split = int(len(y) * 0.8)
    train, test = y.iloc[:split], y.iloc[split:]

    model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,0,1,4), enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)

    pred = res.get_forecast(steps=len(test)).predicted_mean.clip(lower=0)
    mape = float(mean_absolute_percentage_error(test, pred)) if len(test) > 0 else np.nan

    future = res.get_forecast(steps=steps).predicted_mean.clip(lower=0)
    out = pd.DataFrame({
        "week": list(test.index) + list(pd.date_range(start=y.index[-1] + pd.Timedelta(days=7), periods=steps, freq="W-MON")),
        "type": (["test_pred"] * len(test)) + (["future_forecast"] * steps),
        "value": list(pred.values) + list(future.values)
    })
    return out, mape

def main():
    df = load_shipments()
    rows = []
    for region in ["EU", "US"]:
        y = build_weekly_series(df, region)
        out, mape = fit_forecast(y, steps=2)
        out["region"] = region
        out["mape"] = mape
        rows.append(out)

    final = pd.concat(rows, ignore_index=True)
    final.to_csv(os.path.join(OUT, "predictive_forecast_workload.csv"), index=False)
    print("Wrote reports/outputs/predictive_forecast_workload.csv")

if __name__ == "__main__":
    main()