#!/bin/bash
set -e
cd "C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter"

echo "--- LEAR, Q1 static, default features ---"
conda run -n epf --no-capture-output python -X utf8 -m src.master_forecast \
  --algo lear --task price --market dam --strategy recursive --gate strict --retrain static \
  --train_end "2025-11-30 23:00" --test_start "2025-12-01 00:00" --test_end "2026-02-28 23:00" \
  --features default \
  --out_json "step9_out/lear_q1_static_default.json"

echo "--- XGB, Q1 static, default features (fresh, for ensemble) ---"
conda run -n epf --no-capture-output python -X utf8 -m src.master_forecast \
  --algo xgb --task price --market dam --strategy recursive --gate strict --retrain static \
  --train_end "2025-11-30 23:00" --test_start "2025-12-01 00:00" --test_end "2026-02-28 23:00" \
  --features default \
  --out_json "step9_out/xgb_q1_static_default.json"

echo "ALL_DONE_9A"
