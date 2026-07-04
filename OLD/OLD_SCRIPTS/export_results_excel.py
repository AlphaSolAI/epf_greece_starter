import json, sys
import pandas as pd
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(r"C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter")
OUT  = ROOT / "thesis_output" / "eval_results_summary.xlsx"

FILES = {
    ("Dec 2025","Price","Direct H24"):    "dashboard_data_hourly_price_direct_h24_monthly.json",
    ("Dec 2025","Price","MIMO H24"):      "dashboard_data_hourly_price_mimo_h24_monthly.json",
    ("Dec 2025","Price","Open Loop H24"): "dashboard_data_hourly_price_openloop_h24_monthly.json",
    ("Dec 2025","Price","Open Loop H168"):"dashboard_data_hourly_price_openloop_h168_monthly.json",
    ("Dec 2025","Price","Closed Loop"):   "dashboard_data_hourly_price_cl_monthly.json",
    ("Dec 2025","Load","Direct H24"):     "dashboard_data_hourly_load_direct_h24_monthly.json",
    ("Dec 2025","Load","MIMO H24"):       "dashboard_data_hourly_load_mimo_h24_monthly.json",
    ("Dec 2025","Load","Open Loop H24"):  "dashboard_data_hourly_load_openloop_h24_monthly.json",
    ("Dec 2025","Load","Open Loop H168"): "dashboard_data_hourly_load_openloop_h168_monthly.json",
    ("Dec 2025","Load","Closed Loop"):    "dashboard_data_hourly_load_cl_monthly.json",
    ("Q1 2026","Price","Direct H24"):     "dashboard_data_hourly_price_direct_h24_monthly_q1_2026.json",
    ("Q1 2026","Price","MIMO H24"):       "dashboard_data_hourly_price_mimo_h24_monthly_q1_2026.json",
    ("Q1 2026","Price","Open Loop H24"):  "dashboard_data_hourly_price_openloop_h24_monthly_q1_2026.json",
    ("Q1 2026","Price","Open Loop H168"): "dashboard_data_hourly_price_openloop_h168_monthly_q1_2026.json",
    ("Q1 2026","Price","Closed Loop"):    "dashboard_data_hourly_price_cl_monthly_q1_2026.json",
    ("Q1 2026","Load","Direct H24"):      "dashboard_data_hourly_load_direct_h24_monthly_q1_2026.json",
    ("Q1 2026","Load","MIMO H24"):        "dashboard_data_hourly_load_mimo_h24_monthly_q1_2026.json",
    ("Q1 2026","Load","Open Loop H24"):   "dashboard_data_hourly_load_openloop_h24_monthly_q1_2026.json",
    ("Q1 2026","Load","Open Loop H168"):  "dashboard_data_hourly_load_openloop_h168_monthly_q1_2026.json",
    ("Q1 2026","Load","Closed Loop"):     "dashboard_data_hourly_load_cl_monthly_q1_2026.json",
    # --- Walk-Forward Monthly Retrain (MR) ---
    ("Q1 2026","Price","WF TF-Rec"):      "dashboard_data_hourly_price_monthly_retrain.json",
    ("Q1 2026","Load","WF TF-Rec"):       "dashboard_data_hourly_load_monthly_retrain.json",
    ("Q1 2026","Price","WF MIMO (v2)"):   "dashboard_data_hourly_price_mimo_monthly_retrain_optuna.json",
    ("Q1 2026","Load","WF MIMO (v2)"):    "dashboard_data_hourly_load_mimo_monthly_retrain_optuna.json",
    ("Q1 2026","Price","WF MIMO (v1)"):   "dashboard_data_hourly_price_mimo_monthly_retrain.json",
    ("Q1 2026","Load","WF MIMO (v1)"):    "dashboard_data_hourly_load_mimo_monthly_retrain.json",
}

rows = []
for (period, task, strategy), fname in FILES.items():
    p = ROOT / fname
    if not p.exists():
        print(f"SKIP (not found): {fname}")
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    dates = d.get("dates", [])
    period_str = f"{dates[0][:10] if dates else '?'} -> {dates[-1][:10] if dates else '?'}"
    for m in d.get("metrics", []):
        rows.append({
            "Period":      period,
            "Task":        task,
            "Strategy":    strategy,
            "Test Period": period_str,
            "Model":       m.get("Model", "?"),
            "Type":        m.get("Type", "?"),
            "MAE":         m.get("MAE"),
            "RMSE":        m.get("RMSE"),
            "sMAPE":       m.get("sMAPE"),
        })

df = pd.DataFrame(rows)

SHEET_NAMES = {
    ("Dec 2025","Price"): "Dec25_Price",
    ("Dec 2025","Load"):  "Dec25_Load",
    ("Q1 2026","Price"):  "Q1_2026_Price",
    ("Q1 2026","Load"):   "Q1_2026_Load",
}

with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="All Results", index=False)
    for (period, task), sheet_name in SHEET_NAMES.items():
        sub = df[(df.Period == period) & (df.Task == task)].drop(columns=["Period","Task"])
        sub.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Saved: {OUT}")
print(f"Total rows: {len(df)}")
