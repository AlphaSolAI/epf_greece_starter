import json, sys
sys.stdout.reconfigure(encoding="utf-8")

for fname in ["dashboard_data_hourly_price_mimo_h168.json",
              "dashboard_data_hourly_load_mimo_h168.json"]:
    print(f"\n=== {fname} ===")
    with open(fname, encoding="utf-8") as f:
        d = json.load(f)
    print("Top-level keys:", list(d.keys()))
    results = d.get("results", {})
    print("results type:", type(results).__name__)
    if isinstance(results, list):
        print("results[0]:", results[0] if results else "empty")
    elif isinstance(results, dict):
        print("results keys:", list(results.keys())[:8])
        for k, v in list(results.items())[:2]:
            print(f"  {k}: type={type(v).__name__}, sample={str(v)[:120]}")
