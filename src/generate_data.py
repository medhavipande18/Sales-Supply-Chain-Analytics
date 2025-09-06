from __future__ import annotations
import os
import math
import uuid
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

RNG = np.random.default_rng(42)

@dataclass(frozen=True)
class GenConfig:
    start_date: str = "2024-01-01"
    months: int = 6
    n_skus: int = 250
    n_factories: int = 3
    n_warehouses: int = 2
    n_stores: int = 80
    shipments_per_day: int = 120
    promo_weeks_ratio: float = 0.18

def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _rand_choice(a, size=1, p=None):
    return RNG.choice(a, size=size, replace=True, p=p)

def _clamp_int(x: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(x))))

def generate_master_data(cfg: GenConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    categories = ["GARMENT", "BAG", "ACCESSORY"]
    sku_ids = [f"SKU{str(i).zfill(4)}" for i in range(1, cfg.n_skus + 1)]
    sku_cat = _rand_choice(categories, size=cfg.n_skus, p=[0.62, 0.18, 0.20])
    unit_cost = []
    volume = []
    for c in sku_cat:
        if c == "GARMENT":
            unit_cost.append(RNG.normal(18, 6))
            volume.append(RNG.normal(0.004, 0.001))
        elif c == "BAG":
            unit_cost.append(RNG.normal(45, 14))
            volume.append(RNG.normal(0.012, 0.004))
        else:
            unit_cost.append(RNG.normal(9, 3))
            volume.append(RNG.normal(0.0015, 0.0006))

    sku_master = pd.DataFrame({
        "sku_id": sku_ids,
        "category": sku_cat,
        "unit_cost": np.clip(unit_cost, 2, None).round(2),
        "unit_volume_m3": np.clip(volume, 0.0005, None).round(5),
        "launch_week": RNG.integers(1, 18, size=cfg.n_skus)
    })

    # Locations with coordinates (rough, for plotting/routing style features)
    factories = []
    for i in range(cfg.n_factories):
        factories.append({
            "location_id": f"F{i+1}",
            "type": "FACTORY",
            "region": "ASIA",
            "lat": float(RNG.uniform(10, 30)),
            "lon": float(RNG.uniform(100, 120))
        })

    warehouses = [
        {"location_id": "W1", "type": "WAREHOUSE", "region": "EU", "lat": 50.1, "lon": 8.6},
        {"location_id": "W2", "type": "WAREHOUSE", "region": "US", "lat": 40.7, "lon": -74.0},
    ][:cfg.n_warehouses]

    stores = []
    for i in range(cfg.n_stores):
        region = _rand_choice(["EU", "US"], p=[0.52, 0.48])[0]
        if region == "EU":
            lat, lon = float(RNG.normal(48.8, 3.5)), float(RNG.normal(2.3, 6.0))
        else:
            lat, lon = float(RNG.normal(39.0, 4.0)), float(RNG.normal(-95.0, 10.0))
        stores.append({
            "location_id": f"S{str(i+1).zfill(3)}",
            "type": "STORE",
            "region": region,
            "lat": lat,
            "lon": lon
        })

    locations = pd.DataFrame(factories + warehouses + stores)
    return sku_master, locations

def _business_cutoff(dt0: datetime, hour: int = 18) -> datetime:
    return dt0.replace(hour=hour, minute=0, second=0, microsecond=0)

def _add_minutes(dt0: datetime, mins: int) -> datetime:
    return dt0 + timedelta(minutes=int(mins))

def generate_shipments(cfg: GenConfig, sku_master: pd.DataFrame, locations: pd.DataFrame) -> pd.DataFrame:
    start = _dt(cfg.start_date)
    end = start + relativedelta(months=cfg.months)
    days = (end - start).days

    factories = locations[locations["type"] == "FACTORY"]["location_id"].tolist()
    warehouses = locations[locations["type"] == "WAREHOUSE"][["location_id","region"]]
    stores = locations[locations["type"] == "STORE"][["location_id","region"]]

    # Weekly promo calendar
    all_weeks = pd.date_range(start=start, end=end, freq="W-MON")
    promo_weeks = set(_rand_choice(all_weeks, size=_clamp_int(len(all_weeks)*cfg.promo_weeks_ratio, 2, len(all_weeks))).tolist())

    rows = []
    for d in range(days):
        day = start + timedelta(days=d)
        n = cfg.shipments_per_day + int(RNG.normal(0, cfg.shipments_per_day * 0.12))
        n = max(30, n)

        for _ in range(n):
            shipment_id = str(uuid.uuid4())[:8]
            sku_id = _rand_choice(sku_master["sku_id"].values)[0]
            sku_cat = sku_master.loc[sku_master["sku_id"] == sku_id, "category"].values[0]

            # Choose lane: Factory->Warehouse or Warehouse->Store
            lane = _rand_choice(["F2W", "W2S"], p=[0.38, 0.62])[0]

            if lane == "F2W":
                origin = _rand_choice(factories)[0]
                wh = warehouses.sample(1, random_state=int(RNG.integers(0, 1e9))).iloc[0]
                destination = wh["location_id"]
                region = wh["region"]
                qty = _clamp_int(abs(RNG.normal(180, 90)), 20, 800)

                planned_departure = day.replace(hour=int(RNG.integers(6, 14)), minute=int(RNG.integers(0, 60)))
                base_transit_mins = 24*60*3  # ~3 days planned air+ground
                planned_arrival = planned_departure + timedelta(minutes=base_transit_mins + int(RNG.normal(0, 8*60)))

                # Delay drivers (customs + missed cutoff)
                export_clearance = _clamp_int(abs(RNG.normal(6*60, 3*60)), 60, 16*60)
                import_clearance = _clamp_int(abs(RNG.normal(8*60, 4*60)), 90, 24*60)
                customs_congestion = RNG.random() < (0.10 if region == "EU" else 0.12)
                if customs_congestion:
                    import_clearance += _clamp_int(abs(RNG.normal(10*60, 4*60)), 180, 22*60)

                pickup_time = planned_departure + timedelta(minutes=int(RNG.normal(0, 90)))
                export_start = pickup_time + timedelta(minutes=int(RNG.normal(60, 30)))
                export_end = export_start + timedelta(minutes=export_clearance)

                flight_dep = export_end + timedelta(minutes=_clamp_int(abs(RNG.normal(6*60, 2*60)), 60, 16*60))
                flight_arr = flight_dep + timedelta(minutes=_clamp_int(abs(RNG.normal(11*60, 2*60)), 6*60, 18*60))

                import_start = flight_arr + timedelta(minutes=_clamp_int(abs(RNG.normal(90, 60)), 20, 5*60))
                import_end = import_start + timedelta(minutes=import_clearance)

                warehouse_receive = import_end + timedelta(minutes=_clamp_int(abs(RNG.normal(4*60, 2*60)), 60, 12*60))

                # Determine late stage and delivered_late
                delay_minutes = int((warehouse_receive - planned_arrival).total_seconds() / 60)
                delivered_late = delay_minutes > 12*60  # >12 hours late at warehouse
                late_stage = "IMPORT_CLEARANCE" if customs_congestion else ("EXPORT_CLEARANCE" if export_clearance > 10*60 else "TRANSPORT")

                rows.append({
                    "shipment_id": shipment_id,
                    "lane": lane,
                    "sku_id": sku_id,
                    "category": sku_cat,
                    "origin": origin,
                    "destination": destination,
                    "region": region,
                    "qty": qty,
                    "planned_departure": planned_departure,
                    "planned_arrival": planned_arrival,
                    "pickup_time": pickup_time,
                    "export_clearance_start": export_start,
                    "export_clearance_end": export_end,
                    "flight_departure": flight_dep,
                    "flight_arrival": flight_arr,
                    "import_clearance_start": import_start,
                    "import_clearance_end": import_end,
                    "warehouse_receive_time": warehouse_receive,
                    "pick_start": pd.NaT,
                    "pick_end": pd.NaT,
                    "pack_start": pd.NaT,
                    "pack_end": pd.NaT,
                    "ship_out_time": pd.NaT,
                    "store_receive_time": pd.NaT,
                    "delivered_late": delivered_late,
                    "late_stage": late_stage,
                    "delay_minutes": delay_minutes,
                    "delay_reason": "CUSTOMS_CONGESTION" if customs_congestion else "NORMAL_VARIANCE"
                })

            else:
                # Warehouse -> Store replenishment
                store = stores.sample(1, random_state=int(RNG.integers(0, 1e9))).iloc[0]
                region = store["region"]
                origin = "W1" if region == "EU" else "W2"
                destination = store["location_id"]
                qty = _clamp_int(abs(RNG.normal(22, 12)), 1, 120)

                # Planned SLA: 24–36 hours
                planned_departure = day.replace(hour=int(RNG.integers(8, 18)), minute=int(RNG.integers(0, 60)))
                planned_transit_mins = int(abs(RNG.normal(30 * 60, 6 * 60)))
                planned_arrival = planned_departure + timedelta(minutes=planned_transit_mins)

                
                # Warehouse operations: pick/pack bottleneck + cutoff miss
                is_promo_week = (pd.Timestamp(day).to_period("W").start_time in promo_weeks)
                workload_multiplier = 1.0 + (0.7 if is_promo_week else 0.0)

                pick_dur = _clamp_int(abs(RNG.normal(45, 20))*workload_multiplier, 10, 240)
                pack_dur = _clamp_int(abs(RNG.normal(55, 25))*workload_multiplier, 10, 360)

                pick_start = planned_departure - timedelta(minutes=_clamp_int(abs(RNG.normal(140, 60)), 30, 420))
                pick_end = pick_start + timedelta(minutes=pick_dur)

                pack_start = pick_end + timedelta(minutes=_clamp_int(abs(RNG.normal(20, 15)), 0, 120))
                pack_end = pack_start + timedelta(minutes=pack_dur)

                cutoff = _business_cutoff(planned_departure, hour=18)
                missed_cutoff = pack_end > cutoff

                # If missed cutoff, shipment leaves next day morning
                if missed_cutoff:
                    ship_out = (planned_departure + timedelta(days=1)).replace(hour=9, minute=int(RNG.integers(0, 60)))
                    delay_reason = "MISSED_CUTOFF_PACKING"
                else:
                    ship_out = planned_departure + timedelta(minutes=_clamp_int(abs(RNG.normal(30, 25)), 0, 180))
                    delay_reason = "NORMAL_VARIANCE"

                store_receive = ship_out + timedelta(minutes=_clamp_int(abs(RNG.normal(18*60, 6*60)), 4*60, 48*60))
                delay_minutes = int((store_receive - planned_arrival).total_seconds() / 60)
                delivered_late = delay_minutes > 4 * 60   # 4-hour SLA breach
                late_stage = "PACKING" if missed_cutoff else ("TRANSPORT" if delay_minutes > 0 else "ON_TIME")

                rows.append({
                    "shipment_id": shipment_id,
                    "lane": lane,
                    "sku_id": sku_id,
                    "category": sku_cat,
                    "origin": origin,
                    "destination": destination,
                    "region": region,
                    "qty": qty,
                    "planned_departure": planned_departure,
                    "planned_arrival": planned_arrival,
                    "pickup_time": pd.NaT,
                    "export_clearance_start": pd.NaT,
                    "export_clearance_end": pd.NaT,
                    "flight_departure": pd.NaT,
                    "flight_arrival": pd.NaT,
                    "import_clearance_start": pd.NaT,
                    "import_clearance_end": pd.NaT,
                    "warehouse_receive_time": pd.NaT,
                    "pick_start": pick_start,
                    "pick_end": pick_end,
                    "pack_start": pack_start,
                    "pack_end": pack_end,
                    "ship_out_time": ship_out,
                    "store_receive_time": store_receive,
                    "delivered_late": delivered_late,
                    "late_stage": late_stage,
                    "delay_minutes": delay_minutes,
                    "delay_reason": delay_reason if not is_promo_week else (delay_reason + "_PROMO_SURGE")
                })

    df = pd.DataFrame(rows)
    # Normalize dtypes
    for c in df.columns:
        if "time" in c or "departure" in c or "arrival" in c or "start" in c or "end" in c:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def generate_sales_inventory(cfg: GenConfig, sku_master: pd.DataFrame, locations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = _dt(cfg.start_date)
    end = start + relativedelta(months=cfg.months)
    weeks = pd.date_range(start=start, end=end, freq="W-MON")

    stores = locations[locations["type"] == "STORE"][["location_id","region"]]
    sku_ids = sku_master["sku_id"].values

    # Promo calendar by week-store (some stores run promos)
    promo_weeks = set(_rand_choice(weeks, size=_clamp_int(len(weeks)*cfg.promo_weeks_ratio, 2, len(weeks))).tolist())

    sales_rows = []
    inv_rows = []
    for week_start in weeks:
        for _, store in stores.iterrows():
            store_id = store["location_id"]
            region = store["region"]
            # each store carries subset of SKUs
            carried = RNG.choice(sku_ids, size=_clamp_int(cfg.n_skus*0.12, 12, 60), replace=False)
            promo_flag = (week_start in promo_weeks) and (RNG.random() < 0.55)

            for sku in carried:
                cat = sku_master.loc[sku_master["sku_id"] == sku, "category"].values[0]
                base = 4.5 if cat == "GARMENT" else (1.4 if cat == "BAG" else 2.3)
                region_mult = 1.05 if region == "US" else 1.0
                promo_mult = 1.55 if promo_flag else 1.0
                seasonal = 1.0 + 0.18 * math.sin((week_start.timetuple().tm_yday / 365) * 2 * math.pi)

                mean_units = base * region_mult * promo_mult * seasonal
                units = _clamp_int(abs(RNG.normal(mean_units, mean_units*0.55)), 0, 80)

                price = 55 if cat == "BAG" else (30 if cat == "GARMENT" else 18)
                price = float(max(5, RNG.normal(price, price*0.12)))
                if promo_flag:
                    price *= float(RNG.uniform(0.75, 0.9))

                sales_rows.append({
                    "week_start": week_start,
                    "store_id": store_id,
                    "sku_id": sku,
                    "category": cat,
                    "units_sold": units,
                    "unit_price": round(price, 2),
                    "promo_flag": promo_flag
                })

                # crude inventory policy
                lead_time_weeks = 2 if region == "EU" else 3
                safety_stock = _clamp_int(mean_units * lead_time_weeks * 0.35, 2, 60)
                reorder_point = _clamp_int(mean_units * lead_time_weeks + safety_stock, 5, 200)
                on_hand = _clamp_int(abs(RNG.normal(reorder_point * 0.9, reorder_point * 0.35)), 0, 350)

                inv_rows.append({
                    "week_start": week_start,
                    "store_id": store_id,
                    "sku_id": sku,
                    "on_hand_units": on_hand,
                    "safety_stock": safety_stock,
                    "reorder_point": reorder_point
                })

    sales = pd.DataFrame(sales_rows)
    inventory = pd.DataFrame(inv_rows)
    return sales, inventory

def main() -> None:
    cfg = GenConfig()
    out_dir = os.path.join("data", "raw")
    _ensure_dir(out_dir)

    sku_master, locations = generate_master_data(cfg)
    shipments = generate_shipments(cfg, sku_master, locations)
    sales, inventory = generate_sales_inventory(cfg, sku_master, locations)

    sku_master.to_csv(os.path.join(out_dir, "sku_master.csv"), index=False)
    locations.to_csv(os.path.join(out_dir, "locations.csv"), index=False)
    shipments.to_csv(os.path.join(out_dir, "shipments.csv"), index=False)
    sales.to_csv(os.path.join(out_dir, "sales.csv"), index=False)
    inventory.to_csv(os.path.join(out_dir, "inventory.csv"), index=False)

    print("Generated:")
    print(f"- {len(sku_master):,} SKUs")
    print(f"- {len(locations):,} locations")
    print(f"- {len(shipments):,} shipments")
    print(f"- {len(sales):,} sales rows")
    print(f"- {len(inventory):,} inventory rows")

if __name__ == "__main__":
    main()