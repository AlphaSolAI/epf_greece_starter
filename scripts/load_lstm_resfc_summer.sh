#!/usr/bin/env bash
# LSTM πραγματικο ablation axis (calendar/resfc/loadfc/meteo/fuel — OXI dense/lags,
# βλ. ABLATION §7.7 doc 2026-07-12). resfc = solar/wind/gen DA forecasts, μελλοντικα
# γνωστα, εγκυρο decoder input. fuel=0 cols στο load parquet (VOID, δεν τεσταρεται).
# base ηδη τρεξμενο (summer_lstm_recstatic_g12_base.json) — μονο +resfc arm εδω.
# Τρεξιμο (detached): bash scripts/load_lstm_resfc_summer.sh > logs/load_lstm_resfc.log 2>&1

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
    if $PY -m src.master_forecast "$@" --seed 42 --out_json "$oj"; then
        echo "--- [$(stamp)] OK  $oj  ($(( $(date +%s) - t0 ))s)"
    else
        echo "!!! [$(stamp)] FAILED  $oj  (συνεχιζω)"
    fi
}

echo "############ LSTM +resfc (summer) START [$(stamp)] ############"
echo "GIT: $(git rev-parse HEAD) | DIRTY: $(git status --porcelain src/ | tr '\n' ' ')"

echo ""; echo "######## BLOCK 0 — PREFLIGHT+POISON [$(stamp)] ########"
if ! $PY .claude/skills/energy-forecast/scripts/preflight_check.py --poison; then
    echo "!!!!! PREFLIGHT/POISON FAIL — ABORT"; exit 1
fi
echo "######## BLOCK 0 PASS ########"

mf "$OUT/summer_lstm_recstatic_g12_resfc.json" \
   --algo lstm --task load --market dam --strategy recursive --gate strict --retrain static \
   --train_end "2025-05-31 23:00" --test_start "2025-06-01 00:00" --test_end "2025-08-31 23:00" \
   --features "calendar,resfc"

echo ""; echo "############ LSTM +resfc (summer) END [$(stamp)] ############"
