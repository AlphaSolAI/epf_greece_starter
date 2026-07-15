#!/usr/bin/env bash
# TRIAGE: LSTM +resfc summer edeixe Δ=+188.5 (χειροτερο) seed42. LSTM ειναι torch-based
# οπως το MLP — προσθηκη στηλων αλλαζει input_dim -> διαφορετικη RNG αλυσιδα ακομα και
# με "ιδιο" seed (ιδιο μαθημα με MLP §7.17). 2 seeds ακομα σε resfc + 1 στο base για
# ελεγχο variance πριν γραφτει οτιδηποτε ως ευρημα.
# Τρεξιμο (detached): bash scripts/load_lstm_resfc_seedcheck.sh > logs/load_lstm_resfc_seedcheck.log 2>&1

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

echo "############ LSTM resfc SEED CHECK START [$(stamp)] ############"
COMMON=(--algo lstm --task load --market dam --strategy recursive --gate strict --retrain static --train_end "2025-05-31 23:00" --test_start "2025-06-01 00:00" --test_end "2025-08-31 23:00")

for seed in 7 123; do
  mf "$OUT/summer_lstm_recstatic_g12_resfc_seed${seed}.json" "${COMMON[@]}" --seed "$seed" --features "calendar,resfc"
done
mf "$OUT/summer_lstm_recstatic_g12_base_seed7.json" "${COMMON[@]}" --seed 7 --features "calendar,lags,roll"

echo ""; echo "############ LSTM resfc SEED CHECK END [$(stamp)] ############"
