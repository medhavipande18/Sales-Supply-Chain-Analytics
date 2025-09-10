import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

RAW = os.path.join("data", "raw")
OUT = os.path.join("reports", "figures")
os.makedirs(OUT, exist_ok=True)

def load_shipments() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW, "shipments.csv"), parse_dates=[
        "planned_departure","planned_arrival","warehouse_receive_time",
        "pick_start","pick_end","pack_start","pack_end","ship_out_time","store_receive_time"
    ])
    return df

def kpis(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["week"] = pd.to_datetime(df2["planned_departure"]).dt.to_period("W-MON").dt.start_time
    out = (df2.groupby(["lane","region","week"])
             .agg(shipments=("shipment_id","count"),
                  late_shipments=("delivered_late","sum"),
                  avg_delay_mins=("delay_minutes","mean"),
                  p95_delay_mins=("delay_minutes", lambda x: x.quantile(0.95)))
             .reset_index())
    out["late_rate"] = out["late_shipments"] / out["shipments"]
    return out

def plot_late_rate(k: pd.DataFrame) -> None:
    # Simple plot: late rate over time by lane
    for lane in k["lane"].unique():
        tmp = k[k["lane"] == lane].sort_values("week")
        plt.figure()
        plt.plot(tmp["week"], tmp["late_rate"], marker="o")
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))   # show every 2 weeks
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))     # Jan 01, Jan 15...
        plt.xticks(rotation=0)
        plt.title(f"Late Delivery Rate Over Time ({lane})")
        plt.ylabel("Late Rate")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"late_rate_{lane}.png"), dpi=160)
        plt.close()

def main():
    df = load_shipments()
    k = kpis(df)
    os.makedirs(os.path.join("reports","outputs"), exist_ok=True)
    k.to_csv(os.path.join("reports","outputs","descriptive_kpis.csv"), index=False)
    plot_late_rate(k)
    print("Descriptive outputs written to reports/")

if __name__ == "__main__":
    main()