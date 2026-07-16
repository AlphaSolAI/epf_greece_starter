#!/usr/bin/env bash
# LSTM calendar-only base + calendar+loadfc — 3ο window (q1), συνεπεια με το ηδη
# κλεισμενο §2 finding (§7.19c, summer+octnov). static, seed42.
# Τρεξιμο (detached): bash scripts/load_lstm_loadfc_q1.sh > logs/load_lstm_loadfc_q1.log 2>&1

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

echo "############ LSTM loadfc q1 (3rd window) START [$(stamp)] ############"
echo "GIT: $(git rev-parse HEAD) | DIRTY: $(git status --porcelain src/ | tr '\n' ' ')"

echo ""; echo "######## BLOCK 0 — PREFLIGHT+POISON [$(stamp)] ########"
if ! $PY .claude/skills/energy-forecast/scripts/preflight_check.py --poison; then
    echo "!!!!! PREFLIGHT/POISON FAIL — ABORT"; exit 1
fi
echo "######## BLOCK 0 PASS ########"

COMMON=(--algo lstm --task load --market dam --strategy recursive --gate strict --retrain static --train_end "2025-11-30 23:00" --test_start "2025-12-01 00:00" --test_end "2026-02-28 23:00")

mf "$OUT/q1_lstm_recstatic_g12_base.json"   "${COMMON[@]}" --features "calendar,lags,roll"
mf "$OUT/q1_lstm_recstatic_g12_loadfc.json" "${COMMON[@]}" --features "calendar,loadfc"

echo ""; echo "############ LSTM loadfc q1 (3rd window) END [$(stamp)] ############"
