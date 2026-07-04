#!/bin/bash
set -e
cd "C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter"

BASE="conda run -n epf --no-capture-output python -X utf8 -m src.master_forecast --algo lgbm --task price --market dam --strategy recursive --gate strict --retrain static --train_end \"2025-11-30 23:00\" --test_start \"2025-12-01 00:00\" --test_end \"2026-02-28 23:00\" --features default"

echo "--- no-SS baseline ---"
conda run -n epf --no-capture-output python -X utf8 -m src.master_forecast \
  --algo lgbm --task price --market dam --strategy recursive --gate strict --retrain static \
  --train_end "2025-11-30 23:00" --test_start "2025-12-01 00:00" --test_end "2026-02-28 23:00" \
  --features default --out_json "ss_out/nossf.json"

for decay in linear exp step; do
  echo "--- SS decay=$decay ---"
  conda run -n epf --no-capture-output python -X utf8 -m src.master_forecast \
    --algo lgbm --task price --market dam --strategy recursive --gate strict --retrain static \
    --train_end "2025-11-30 23:00" --test_start "2025-12-01 00:00" --test_end "2026-02-28 23:00" \
    --features default --ss --ss_decay "$decay" --ss_rounds 3 \
    --out_json "ss_out/ss_${decay}.json"
done

echo "ALL_DONE"
