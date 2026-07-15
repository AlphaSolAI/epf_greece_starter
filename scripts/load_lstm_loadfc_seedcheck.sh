#!/usr/bin/env bash
# TRIAGE (ιδιο πρωτοκολλο με resfc/meteo): LSTM +loadfc summer edeixe Δ=-166 (μεγαλη
# βελτιωση). Εχει λογικο μηχανισμο (load_fc = ΑΔΜΗΕ day-ahead forecast, ισχυρο by
# construction) αλλα το LSTM ειναι stochastic — 2 seeds ακομα πριν γραφτει ως finding.
# Τρεξιμο (detached): bash scripts/load_lstm_loadfc_seedcheck.sh > logs/load_lstm_loadfc_seedcheck.log 2>&1

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

echo "############ LSTM loadfc SEED CHECK START [$(stamp)] ############"
COMMON=(--algo lstm --task load --market dam --strategy recursive --gate strict --retrain static --train_end "2025-05-31 23:00" --test_start "2025-06-01 00:00" --test_end "2025-08-31 23:00")

for seed in 7 123; do
  mf "$OUT/summer_lstm_recstatic_g12_loadfc_seed${seed}.json" "${COMMON[@]}" --seed "$seed" --features "calendar,loadfc"
done

echo ""; echo "############ LSTM loadfc SEED CHECK END [$(stamp)] ############"
