import json, sys
sys.stdout.reconfigure(encoding="utf-8")

files = {
    "CL_price":   "dashboard_data_hourly_price.json",
    "CL_load":    "dashboard_data_hourly_load.json",
    "OL_price":   "dashboard_data_hourly_price_openloop.json",
    "OL_load":    "dashboard_data_hourly_load_openloop.json",
    "MIMO_price": "dashboard_data_hourly_price_mimo_h168.json",
    "MIMO_load":  "dashboard_data_hourly_load_mimo_h168.json",
}

for label, fname in files.items():
    try:
        with open(fname, encoding="utf-8") as f:
            d = json.load(f)
        keys = list(d.keys())
        n_metrics = len(d.get("metrics", []))
        n_dates   = len(d.get("dates", []))
        n_series  = len(d.get("series", {}))
        has_actual = "actual" in d
        print(f"{label:15} | keys={keys} | metrics={n_metrics} | dates={n_dates} | series={n_series} | actual={has_actual}")
    except FileNotFoundError:
        print(f"{label:15} | ❌ FILE NOT FOUND: {fname}")
    except Exception as e:
        print(f"{label:15} | ❌ ERROR: {e}")
