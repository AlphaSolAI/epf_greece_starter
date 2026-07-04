#!/bin/bash
set -e
cd "C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter"

echo "=== E1: n_estimators x meteo (price, dam) ==="
for n in 400 800 1600; do
  for feat in "default" "default,-meteo"; do
    slug=$(echo "$feat" | tr ',' '_')
    echo "--- E1 n_estimators=$n features=$feat ---"
    conda run -n epf --no-capture-output python -X utf8 -m src.master_forecast \
      --algo lgbm --task price --market dam --strategy recursive --gate strict --retrain static \
      --train_end "2025-11-30 23:00" --test_start "2025-12-01 00:00" --test_end "2026-02-28 23:00" \
      --features "$feat" --n_estimators "$n" \
      --out_json "e1_e2_out/e1_n${n}_${slug}.json"
  done
done

echo "=== E2: meteo on load task (static Q1) ==="
for feat in "default" "default,-meteo" "lags,calendar,meteo"; do
  slug=$(echo "$feat" | tr ',' '_')
  echo "--- E2 features=$feat ---"
  conda run -n epf --no-capture-output python -X utf8 -m src.master_forecast \
    --algo lgbm --task load --market dam --strategy recursive --gate strict --retrain static \
    --train_end "2025-11-30 23:00" --test_start "2025-12-01 00:00" --test_end "2026-02-28 23:00" \
    --features "$feat" \
    --out_json "e1_e2_out/e2_load_${slug}.json"
done

echo "ALL_DONE"
