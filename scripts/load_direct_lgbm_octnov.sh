#!/usr/bin/env bash
# STAGE: direct LGBM octnov g12, 5 clean arms, direct weekly (5 runs, bounded 1 window)
# Συνέχεια του §9 σειράς μετά το g14+direct queue (summer πέρασμα, §7.21) — επόμενο
# window του "direct LGBM (1 window τη φορά)" plan. g12 = χωρις --delay (ΟΧΙ ΑΔΜΗΕ-aligned gate).
# Idempotent (SKIP αν υπαρχει το JSON), ΕΝΑ conda process, συνεχιζει σε FAILED run.
# Τρεξιμο (detached): bash scripts/load_direct_lgbm_octnov.sh > logs/load_direct_lgbm_octnov.log 2>&1

cd "C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter" || exit 1

PY="conda run -n epf --no-capture-output python -X utf8"
mkdir -p runs/load_direct

ON_TS="2025-10-01 00:00";  ON_TE="2025-11-30 23:00"
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

echo "############ DIRECT LGBM OCTNOV g12 START [$(stamp)] ############"
echo "GIT: $(git rev-parse HEAD) | DIRTY: $(git status --porcelain src/ | tr '\n' ' ')"

echo ""; echo "######## BLOCK 0 — PREFLIGHT+POISON [$(stamp)] ########"
if ! $PY .claude/skills/energy-forecast/scripts/preflight_check.py --poison; then
    echo "!!!!! PREFLIGHT/POISON FAIL — ABORT"; exit 1
fi
echo "######## BLOCK 0 PASS ########"

echo ""; echo "######## STAGE 1/1 — direct LGBM octnov g12, 5 clean arms [$(stamp)] ########"
dir_arm() {  # dir_arm <slug> <features>
  mf "runs/load_direct/octnov_lgbm_dirw_g12_$1.json" \
     --algo lgbm --task load --market dam --strategy direct --gate strict \
     --retrain weekly --test_start "$ON_TS" --test_end "$ON_TE" --features "$2"
}
dir_arm "base"    "calendar,lags,roll"
dir_arm "dense"   "calendar,lags,roll,dense"
dir_arm "genlags" "calendar,lags,roll,genlags"
dir_arm "noroll"  "calendar,lags"
dir_arm "loadfc"  "calendar,lags,roll,loadfc"
echo "######## STAGE 1/1 DONE [$(stamp)] ########"

echo ""; echo "############ DIRECT LGBM OCTNOV g12 COMPLETE [$(stamp)] ############"
echo "Harvest: python scripts/build_run_ledger.py + synthesize-ablation (baseline=octnov_lgbm_dirw_g12_base —"
echo "  ΠΟΤΕ συγκριση g14 vs g12 η direct vs recursive αναμεικτα σε αυτο το πινακα, ξεχωριστο βημα)."
