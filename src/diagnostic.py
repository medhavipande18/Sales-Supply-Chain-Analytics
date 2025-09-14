import os
import pandas as pd

RAW = os.path.join("data", "raw")
OUT = os.path.join("reports", "outputs")
os.makedirs(OUT, exist_ok=True)

def load_shipments() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW, "shipments.csv"), parse_dates=[
        "planned_departure","planned_arrival","warehouse_receive_time",
        "export_clearance_start","export_clearance_end",
        "import_clearance_start","import_clearance_end",
        "pick_start","pick_end","pack_start","pack_end",
        "ship_out_time","store_receive_time"
    ])
    return df

def compute_stage_durations(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Stage durations in minutes (only where timestamps exist)
    def dur(a, b):
        return (d[b] - d[a]).dt.total_seconds() / 60

    d["dur_export_clearance_mins"] = dur("export_clearance_start","export_clearance_end")
    d["dur_import_clearance_mins"] = dur("import_clearance_start","import_clearance_end")
    d["dur_pick_mins"] = dur("pick_start","pick_end")
    d["dur_pack_mins"] = dur("pack_start","pack_end")
    return d

def root_cause_summary(df: pd.DataFrame) -> pd.DataFrame:
    d = compute_stage_durations(df)
    late = d[d["delivered_late"] == True].copy()

    # Simple automated rule-based RCA based on highest stage duration z-score
    candidates = ["dur_import_clearance_mins","dur_export_clearance_mins","dur_pack_mins","dur_pick_mins"]
    stats = {}
    for c in candidates:
        x = late[c].dropna()
        if len(x) == 0:
            stats[c] = (None, None)
            continue
        stats[c] = (x.mean(), x.std() if x.std() > 0 else 1.0)

    def pick_rca(row):
        scores = {}
        for c in candidates:
            if pd.isna(row.get(c)) or stats[c][0] is None:
                continue
            mu, sd = stats[c]
            scores[c] = (row[c] - mu) / sd
        if not scores:
            return "UNKNOWN"
        best = max(scores, key=scores.get)
        mapping = {
            "dur_import_clearance_mins": "IMPORT_CLEARANCE",
            "dur_export_clearance_mins": "EXPORT_CLEARANCE",
            "dur_pack_mins": "PACKING",
            "dur_pick_mins": "PICKING"
        }
        return mapping.get(best, "OTHER")

    late["rca_stage"] = late.apply(pick_rca, axis=1)

    out = (late.groupby(["lane","region","rca_stage"])
              .agg(late_shipments=("shipment_id","count"),
                   avg_delay_mins=("delay_minutes","mean"))
              .reset_index()
              .sort_values(["lane","region","late_shipments"], ascending=[True,True,False]))
    return out

def main():
    df = load_shipments()
    out = root_cause_summary(df)
    out.to_csv(os.path.join(OUT, "diagnostic_root_cause.csv"), index=False)
    print("Wrote reports/outputs/diagnostic_root_cause.csv")

if __name__ == "__main__":
    main()