#!/usr/bin/env bash
# MLP/LEAR q1 window (3ο window) — ενισχυει το ηδη κλεισμενο §2 finding (dense βοηθαει,
# summer+octnov). Ιδια δομη/gate/cadence με τα ηδη τρεξμενα (weekly recursive, g12, seed42).
# Τρεξιμο (detached): bash scripts/load_mlp_lear_q1.sh > logs/load_mlp_lear_q1.log 2>&1

cd "C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter" || exit 1

PY="conda run -n epf --no-capture-output python -X utf8"
OUT_MLP="runs/load_mlp"
OUT_LEAR="runs/load_lear"
mkdir -p "$OUT_MLP" "$OUT_LEAR"

Q1_TS="2025-12-01 00:00";  Q1_TE="2026-02-28 23:00"
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

echo "############ MLP/LEAR q1 (3rd window) START [$(stamp)] ############"
echo "GIT: $(git rev-parse HEAD) | DIRTY: $(git status --porcelain src/ | tr '\n' ' ')"

echo ""; echo "######## BLOCK 0 — PREFLIGHT+POISON [$(stamp)] ########"
if ! $PY .claude/skills/energy-forecast/scripts/preflight_check.py --poison; then
    echo "!!!!! PREFLIGHT/POISON FAIL — ABORT"; exit 1
fi
echo "######## BLOCK 0 PASS ########"

COMMON=(--task load --market dam --strategy recursive --gate strict --retrain weekly --test_start "$Q1_TS" --test_end "$Q1_TE")

mf "$OUT_MLP/q1_mlp_recw_g12_base.json"    --algo mlp  "${COMMON[@]}" --features "calendar,lags,roll"
mf "$OUT_MLP/q1_mlp_recw_g12_dense.json"   --algo mlp  "${COMMON[@]}" --features "calendar,lags,roll,dense"
mf "$OUT_LEAR/q1_lear_recw_g12_base.json"  --algo lear "${COMMON[@]}" --features "calendar,lags,roll"
mf "$OUT_LEAR/q1_lear_recw_g12_dense.json" --algo lear "${COMMON[@]}" --features "calendar,lags,roll,dense"

echo ""; echo "############ MLP/LEAR q1 (3rd window) END [$(stamp)] ############"
