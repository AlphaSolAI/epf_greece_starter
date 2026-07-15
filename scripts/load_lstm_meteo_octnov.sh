#!/usr/bin/env bash
# LSTM +meteo — 2ο window (octnov) για να κλεισει §2 πανω στο summer finding (19b).
# base ηδη τρεξμενο (octnov_lstm_recstatic_g12_base.json). static, seed42.
# Τρεξιμο (detached): bash scripts/load_lstm_meteo_octnov.sh > logs/load_lstm_meteo_octnov.log 2>&1

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

echo "############ LSTM +meteo (octnov, 2nd window) START [$(stamp)] ############"
echo "GIT: $(git rev-parse HEAD) | DIRTY: $(git status --porcelain src/ | tr '\n' ' ')"

echo ""; echo "######## BLOCK 0 — PREFLIGHT+POISON [$(stamp)] ########"
if ! $PY .claude/skills/energy-forecast/scripts/preflight_check.py --poison; then
    echo "!!!!! PREFLIGHT/POISON FAIL — ABORT"; exit 1
fi
echo "######## BLOCK 0 PASS ########"

mf "$OUT/octnov_lstm_recstatic_g12_meteo.json" \
   --algo lstm --task load --market dam --strategy recursive --gate strict --retrain static \
   --train_end "2025-08-31 23:00" --test_start "2025-10-01 00:00" --test_end "2025-11-30 23:00" \
   --features "calendar,meteo"

echo ""; echo "############ LSTM +meteo (octnov, 2nd window) END [$(stamp)] ############"
