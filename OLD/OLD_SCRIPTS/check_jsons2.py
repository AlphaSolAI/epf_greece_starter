import json, sys
sys.stdout.reconfigure(encoding="utf-8")

files = {
    "OLD_hourly (παλιό)":       "dashboard_data_hourly.json",
    "OLD_ol (παλιό OL)":        "dashboard_data_hourly_openloop.json",
    "NEW_CL_price":              "dashboard_data_hourly_price.json",
    "NEW_CL_load":               "dashboard_data_hourly_load.json",
    "NEW_OL_price":              "dashboard_data_hourly_price_openloop.json",
    "NEW_OL_load":               "dashboard_data_hourly_load_openloop.json",
    "MIMO_price":                "dashboard_data_hourly_price_mimo_h168.json",
    "MIMO_load":                 "dashboard_data_hourly_load_mimo_h168.json",
}

for label, fname in files.items():
    try:
        with open(fname, encoding="utf-8") as f:
            d = json.load(f)
        has_dates   = len(d.get("dates", [])) > 0
        has_actual  = len(d.get("actual", [])) > 0
        has_series  = len(d.get("series", {})) > 0
        # Check all possible metrics keys
        metrics = (d.get("metrics") or d.get("metrics_test") or
                   [r for r in d.get("results", {}).values() if isinstance(r, list)])
        n_metrics = len(metrics) if metrics else 0
        metrics_key = ("metrics" if "metrics" in d else
                       "metrics_test" if "metrics_test" in d else
                       "results" if "results" in d else "❌ NONE")
        print(f"\n{label} ({fname})")
        print(f"  dates={has_dates}, actual={has_actual}, series={has_series}, metrics_key={metrics_key}, n_metrics={n_metrics}")
        if "metrics" in d and d["metrics"]:
            print(f"  Sample metric: {d['metrics'][0]}")
        elif "metrics_test" in d and d["metrics_test"]:
            print(f"  Sample metric: {d['metrics_test'][0]}")
        elif "results" in d:
            print(f"  Results keys: {list(d['results'].keys())[:4]}")
    except FileNotFoundError:
        print(f"\n{label}: ❌ FILE NOT FOUND")
    except Exception as e:
        print(f"\n{label}: ❌ ERROR: {e}")
