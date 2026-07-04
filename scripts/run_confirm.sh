#!/bin/bash
set -e
cd "C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter"

echo "--- A: default, monthly retrain, Q1 ---"
conda run -n epf --no-capture-output python -X utf8 -m src.master_forecast \
  --algo lgbm --task price --market dam --strategy recursive --gate strict --retrain monthly \
  --test_start "2025-12-01 00:00" --test_end "2026-02-28 23:00" \
  --features default \
  --out_json "confirm_out/A_default_monthly_q1.json"

echo "--- B: default,xborder, monthly retrain, Q1 ---"
conda run -n epf --no-capture-output python -X utf8 -m src.master_forecast \
  --algo lgbm --task price --market dam --strategy recursive --gate strict --retrain monthly \
  --test_start "2025-12-01 00:00" --test_end "2026-02-28 23:00" \
  --features "default,xborder" \
  --out_json "confirm_out/B_default_xborder_monthly_q1.json"

echo "--- C: default,xborder,-loadlags + SS-linear, static, Q1 ---"
conda run -n epf --no-capture-output python -X utf8 -m src.master_forecast \
  --algo lgbm --task price --market dam --strategy recursive --gate strict --retrain static \
  --train_end "2025-11-30 23:00" --test_start "2025-12-01 00:00" --test_end "2026-02-28 23:00" \
  --features "default,xborder,-loadlags" --ss --ss_decay linear --ss_rounds 3 \
  --out_json "confirm_out/C_ssinformed_static_q1.json"

echo "ALL_DONE"
