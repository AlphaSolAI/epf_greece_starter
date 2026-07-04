#!/bin/bash
set -e
cd "C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter"

for grp in meteo loadlags fuel resfc loadfc; do
  echo "--- SS-linear default,-${grp} ---"
  conda run -n epf --no-capture-output python -X utf8 -m src.master_forecast \
    --algo lgbm --task price --market dam --strategy recursive --gate strict --retrain static \
    --train_end "2025-11-30 23:00" --test_start "2025-12-01 00:00" --test_end "2026-02-28 23:00" \
    --features "default,-${grp}" --ss --ss_decay linear --ss_rounds 3 \
    --out_json "ss_out/ss_linear_no_${grp}.json"
done

echo "ALL_DONE"
