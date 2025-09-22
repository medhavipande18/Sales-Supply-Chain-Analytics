import os
import pandas as pd
from ortools.linear_solver import pywraplp

OUT = os.path.join("reports", "outputs")
os.makedirs(OUT, exist_ok=True)

def load_forecast() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(OUT, "predictive_forecast_workload.csv"), parse_dates=["week"])
    # take future forecast rows only
    return df[df["type"] == "future_forecast"].copy()

def solve_staffing(forecast_units: int, cost_ft: float = 2200, cost_pt: float = 1200,
                   cap_ft: int = 420, cap_pt: int = 180, overtime_cost: float = 9.0) -> dict:
    """
    Decision variables:
      x_ft, x_pt : number of workers
      overtime_units : units handled via overtime
    Constraints:
      capacity + overtime >= forecast_units
      overtime_units <= 0.10 * forecast_units (policy)
    Objective:
      minimize staffing cost + overtime cost
    """
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        raise RuntimeError("OR-Tools solver not available")

    x_ft = solver.NumVar(0, solver.infinity(), "x_ft")
    x_pt = solver.NumVar(0, solver.infinity(), "x_pt")
    overtime = solver.NumVar(0, solver.infinity(), "overtime_units")

    solver.Add(cap_ft * x_ft + cap_pt * x_pt + overtime >= forecast_units)
    solver.Add(overtime <= 0.10 * forecast_units)

    solver.Minimize(cost_ft * x_ft + cost_pt * x_pt + overtime_cost * overtime)

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        return {"status": "NOT_OPTIMAL"}

    return {
        "status": "OPTIMAL",
        "forecast_units": forecast_units,
        "x_ft": x_ft.solution_value(),
        "x_pt": x_pt.solution_value(),
        "overtime_units": overtime.solution_value(),
        "total_cost": solver.Objective().Value()
    }

def main():
    fc = load_forecast()
    outputs = []
    for region, grp in fc.groupby("region"):
        for _, r in grp.iterrows():
            sol = solve_staffing(int(round(r["value"])))
            sol["region"] = region
            sol["week"] = r["week"]
            outputs.append(sol)

    out = pd.DataFrame(outputs)
    out.to_csv(os.path.join(OUT, "prescriptive_staffing_plan.csv"), index=False)
    print("Wrote reports/outputs/prescriptive_staffing_plan.csv")

if __name__ == "__main__":
    main()