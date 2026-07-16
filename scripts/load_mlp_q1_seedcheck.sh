#!/usr/bin/env bash
# TRIAGE: MLP q1 edeixe Δ=+0.74 (σχεδον flat, αντιθετο απο summer/octnov). Σε αντιθεση
# με το LEAR (deterministic, seed-invariant, ελεγχθηκε), το MLP ΕΙΝΑΙ πραγματικα
# stochastic (torch weight-init) — 2 seeds ακομα στο q1 base+dense πριν αποφασιστει
# αν το flip ειναι πραγματικο (winter-specific) η seed-noise.
# Τρεξιμο (detached): bash scripts/load_mlp_q1_seedcheck.sh > logs/load_mlp_q1_seedcheck.log 2>&1

cd "C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter" || exit 1

PY="conda run -n epf --no-capture-output python -X utf8"
OUT="runs/load_mlp"
mkdir -p "$OUT"

Q1_TS="2025-12-01 00:00";  Q1_TE="2026-02-28 23:00"
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

echo "############ MLP q1 SEED CHECK START [$(stamp)] ############"
COMMON=(--algo mlp --task load --market dam --strategy recursive --gate strict --retrain weekly --test_start "$Q1_TS" --test_end "$Q1_TE")

for seed in 7 123; do
  mf "$OUT/q1_mlp_recw_g12_base_seed${seed}.json"  "${COMMON[@]}" --seed "$seed" --features "calendar,lags,roll"
  mf "$OUT/q1_mlp_recw_g12_dense_seed${seed}.json" "${COMMON[@]}" --seed "$seed" --features "calendar,lags,roll,dense"
done

echo ""; echo "############ MLP q1 SEED CHECK END [$(stamp)] ############"
