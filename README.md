# Sales Supply Chain Analytics (Synthetic Case Study)

## Overview
This project presents an **end-to-end Supply Chain Analytics case study** for a global fashion retailer, implemented entirely in **Python** using **synthetic but realistic operational data**.

The objective is to demonstrate how data scientists and analytics engineers can leverage operational data to:
- Create visibility across the supply chain
- Diagnose root causes of delivery delays
- Forecast near-term operational workload
- Support decision-making through optimization

The project follows the classical **Supply Chain Analytics maturity framework**:  
**Descriptive → Diagnostic → Predictive → Prescriptive**.

---

## Business Context
You are a data scientist working in the **Supply Chain Analytics** team of an international fashion retailer.

### Supply Chain Network
- **Factories (Asia)** manufacture garments, bags, and accessories  
- **Regional warehouses (EU & US)** receive bulk shipments  
- **Retail stores** are replenished from local warehouses  

### Flows
- **Physical flow**: Factory → Warehouse → Store  
- **Information flow**: ERP, WMS, planning systems generate operational timestamps and sales signals  

The challenge is to **exploit these data flows** to improve service levels, anticipate bottlenecks, and support operational planning.

---

## Analytics Scope

### 1. Descriptive Analytics – *What happened?*
Provides visibility into supply chain performance:
- Weekly shipment volumes
- Late delivery rates by lane (Factory → Warehouse, Warehouse → Store)
- Delay distributions and service levels

**Outputs**
- KPI tables  
- Time-series performance plots  

---

### 2. Diagnostic Analytics – *Why did it happen?*
Automated **root cause analysis (RCA)** using shipment milestone timestamps:
- Export clearance delays
- Import clearance congestion
- Warehouse picking and packing bottlenecks
- Missed cutoff times

**Outputs**
- RCA attribution tables by lane and region  
- Delay drivers ranked by impact  

---

### 3. Predictive Analytics – *What is likely to happen next?*
Forecasts near-term **warehouse-to-store workload** to support capacity planning:
- Weekly shipment volume forecasting
- Model evaluation using error metrics (MAPE)

**Outputs**
- Short-term workload forecasts  
- Forecast accuracy metrics  

---

### 4. Prescriptive Analytics – *What should we do?*
Translates forecasts into **actionable decisions** using optimization:
- Workforce planning (full-time vs part-time vs overtime)
- Cost minimization under service-level constraints

**Outputs**
- Optimal staffing plans per region and week  
- Estimated operational cost trade-offs  

---

## Data
All datasets are **synthetically generated** but designed to mimic real-world supply chain behavior.

### Key Characteristics
- Realistic shipment milestone timestamps
- Promotion-driven demand surges
- Customs congestion and warehouse bottlenecks
- Early and late deliveries relative to SLA

### Generated Datasets
- `shipments.csv` – shipment-level operational events  
- `sales.csv` – store-level weekly sales  
- `inventory.csv` – inventory snapshots  
- `sku_master.csv` – product master data  
- `locations.csv` – factories, warehouses, stores  

---

## Project Structure
```
Sales-Supply-Chain-Analytics/
├── data/
│ ├── raw/
│ └── processed/
├── src/
│ ├── generate_data.py
│ ├── descriptive.py
│ ├── diagnostic.py
│ ├── predictive.py
│ └── prescriptive.py
├── reports/
│ ├── figures/
│ └── outputs/
├── run_all.py
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Run the full pipeline
```bash
python run_all.py
```
This will:
- Generate synthetic data
- Compute KPIs
- Perform RCA
- Train forecasting models
- Solve optimization problems
- Save outputs to reports/

---
## Key Insights (Example)
- Warehouse-to-store deliveries exhibit higher delay volatility due to packing bottlenecks and cutoff misses.
- Factory-to-warehouse shipments are generally stable but impacted by customs congestion during peak periods.
- Forecasted workload surges during promotional weeks require proactive workforce adjustments.
- Prescriptive optimization reduces operational costs while maintaining service-level constraints.
---
## Notes
- This project uses synthetic data for demonstration purposes.
- The analytical logic, workflows, and insights are representative of real-world supply chain analytics implementations.
