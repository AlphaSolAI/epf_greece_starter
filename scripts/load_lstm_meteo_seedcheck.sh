#!/usr/bin/env bash
# TRIAGE: LSTM +meteo summer edeixe Δ=+744.6 (πολυ χειροτερο), 84 στηλες -> μεγαλο
# input_dim jump, μεγαλυτερο ρισκο seed-variance απο το resfc (6 στηλες). 2 seeds
# ακομα στο meteo arm πριν γραφτει οτιδηποτε (base seed7 ηδη υπαρχει απο πριν).
# Τρεξιμο (detached): bash scripts/load_lstm_meteo_seedcheck.sh > logs/load_lstm_meteo_seedcheck.log 2>&1

cd "C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter" || exit 1

PY="conda run -n epf --no-capture-output python -X utf8"
OUT="runs/load_lstm"
mkdir -p "$OUT"
stamp() { date +"%Y-%m-%d %H:%M:%S"; }

mf() {
    local oj="$1"; shift
    if [ -f "$oj" ]; then echo "=== [$(stamp)] SKIP  $oj"; return 0; fi
    echo ""; echo "=== [$(stamp)] RUN -> $oj"
    local t0=$(date +%s)
    if $PY -m src.master_forecast "$@" --out_json "$oj"; then
        echo "--- [$(stamp)] OK  $oj  ($(( $(date +%s) - t0 ))s)"
    else
        echo "!!! [$(stamp)] FAILED  $oj  (συνεχιζω)"
    fi
}

echo "############ LSTM meteo SEED CHECK START [$(stamp)] ############"
COMMON=(--algo lstm --task load --market dam --strategy recursive --gate strict --retrain static --train_end "2025-05-31 23:00" --test_start "2025-06-01 00:00" --test_end "2025-08-31 23:00")

for seed in 7 123; do
  mf "$OUT/summer_lstm_recstatic_g12_meteo_seed${seed}.json" "${COMMON[@]}" --seed "$seed" --features "calendar,meteo"
done

echo ""; echo "############ LSTM meteo SEED CHECK END [$(stamp)] ############"
