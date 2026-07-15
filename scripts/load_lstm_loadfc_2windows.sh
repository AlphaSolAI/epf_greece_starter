#!/usr/bin/env bash
# LSTM +loadfc (1 στηλη, ΑΔΜΗΕ published day-ahead load forecast) — τελευταιο εξεταστεο
# axis (calendar/resfc/loadfc/meteo/fuel, βλ. §7.19a/19b). resfc/meteo βλαψαν και τα δυο.
# 0% NaN και στα 2 windows (ελεγχθηκε). Και τα 2 windows μαζι εδω (φθηνο, 1 στηλη).
# Τρεξιμο (detached): bash scripts/load_lstm_loadfc_2windows.sh > logs/load_lstm_loadfc.log 2>&1

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

echo "############ LSTM +loadfc (2 windows) START [$(stamp)] ############"
echo "GIT: $(git rev-parse HEAD) | DIRTY: $(git status --porcelain src/ | tr '\n' ' ')"

echo ""; echo "######## BLOCK 0 — PREFLIGHT+POISON [$(stamp)] ########"
if ! $PY .claude/skills/energy-forecast/scripts/preflight_check.py --poison; then
    echo "!!!!! PREFLIGHT/POISON FAIL — ABORT"; exit 1
fi
echo "######## BLOCK 0 PASS ########"

mf "$OUT/summer_lstm_recstatic_g12_loadfc.json" \
   --algo lstm --task load --market dam --strategy recursive --gate strict --retrain static \
   --train_end "2025-05-31 23:00" --test_start "2025-06-01 00:00" --test_end "2025-08-31 23:00" \
   --features "calendar,loadfc"

mf "$OUT/octnov_lstm_recstatic_g12_loadfc.json" \
   --algo lstm --task load --market dam --strategy recursive --gate strict --retrain static \
   --train_end "2025-08-31 23:00" --test_start "2025-10-01 00:00" --test_end "2025-11-30 23:00" \
   --features "calendar,loadfc"

echo ""; echo "############ LSTM +loadfc (2 windows) END [$(stamp)] ############"
