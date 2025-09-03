# Supply Chain Analytics with Python (Synthetic Fashion Retail Case Study)

## Context
Data Science project for a Supply Chain organization of a global fashion retailer.
Factories in Asia produce SKUs. Regional warehouses (EU/US) replenish stores.

The objective is to leverage operational timestamps and sales signals to improve:
1) Visibility (Descriptive)
2) Delay explanation (Diagnostic)
3) Near-term workload forecast (Predictive)
4) Workforce decision-making (Prescriptive)

## Data
All data is synthetically generated but built to mimic real supply-chain behavior:
- Shipment milestone timestamps across lanes (Factory→Warehouse, Warehouse→Store)
- Promo-driven demand surges
- Delays via customs congestion and packing bottlenecks
- Store-week sales and inventory snapshots

Generated files:
- data/raw/shipments.csv
- data/raw/sales.csv
- data/raw/inventory.csv
- data/raw/sku_master.csv
- data/raw/locations.csv

## How to run
```bash
pip install -r requirements.txt
python run_all.py