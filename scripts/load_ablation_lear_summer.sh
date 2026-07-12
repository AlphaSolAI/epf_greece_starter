#!/usr/bin/env bash
# LOAD ABLATION — LEAR (φθηνο baseline, δεν ειναι stochastic οπως το MLP — LassoCV
# deterministic given data/seed, ασφαλες με 1 seed). STAGE LEAR-1 SCREEN, summer/g12.
# Combo arms προς συγκριση: base -> dense -> dense+seas (seas ηδη ΑΠΟΡΡΙΦΘΗΚΕ στο MLP
# λογω seed-variance· εδω το δοκιμαζουμε ξανα γιατι το LEAR ειναι deterministic, αρα
# αν βοηθαει εδω ειναι πραγματικο σημα, οχι θορυβος αρχικοποιησης).
# Τρεξιμο (detached): bash scripts/load_ablation_lear_summer.sh > logs/load_lear_summer.log 2>&1

cd "C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter" || exit 1

PY="conda run -n epf --no-capture-output python -X utf8"
OUT="runs/load_lear"
mkdir -p "$OUT"

SU_TS="2025-06-01 00:00";  SU_TE="2025-08-31 23:00"
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

echo "############ LEAR-1 SCREEN START [$(stamp)] ############"
echo "GIT: $(git rev-parse HEAD) | DIRTY: $(git status --porcelain src/ | tr '\n' ' ')"

echo ""; echo "######## BLOCK 0 — PREFLIGHT+POISON [$(stamp)] ########"
if ! $PY .claude/skills/energy-forecast/scripts/preflight_check.py --poison; then
    echo "!!!!! PREFLIGHT/POISON FAIL — ABORT"; exit 1
fi
echo "######## BLOCK 0 PASS ########"

COMMON=(--algo lear --task load --market dam --strategy recursive --gate strict --retrain weekly --test_start "$SU_TS" --test_end "$SU_TE")

mf "$OUT/summer_lear_recw_g12_base.json"      "${COMMON[@]}" --features "calendar,lags,roll"
mf "$OUT/summer_lear_recw_g12_dense.json"     "${COMMON[@]}" --features "calendar,lags,roll,dense"
mf "$OUT/summer_lear_recw_g12_denseseas.json" "${COMMON[@]}" --features "calendar,lags,roll,dense,seas"

echo ""; echo "############ LEAR-1 SCREEN END [$(stamp)] ############"
