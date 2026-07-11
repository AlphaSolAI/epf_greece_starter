#!/usr/bin/env bash
# LOAD CONTEST (G7) — SS x weekly XGB confirm, dense μόνο (2026-07-11, goal χρήστη «light xgb ss»)
# ------------------------------------------------------------------------------------------
# Στόχος: cross-algo επιβεβαίωση του SS-βοηθάει ευρήματος (LGBM 8/8 overnight) με ΤΟΝ ΕΛΑΧΙΣΤΟ
# χρόνο. Τρέχει ΜΟΝΟ τα SS σκέλη — τα XGB non-SS dense baselines ΥΠΑΡΧΟΥΝ ήδη (Batch 2):
#   octnov_xgb_recw_g12_dense=137.46 · summer_xgb_recw_g12_dense=336.10 -> ΔMAE άμεσο.
# Grid: XGB weekly recursive --ss --ss_decay linear, dense, {octnov,summer}, g12 = 2 runs.
# g14/base/q1 ΕΞΩ (ο χρήστης ζήτησε «μονο τα καλα dense αρχικα»). Idempotent SKIP.
# Baseline σύγκρισης: runs/load_contest/{octnov,summer}_xgb_recw_g12_dense.json (ίδιο seed42/gate/spec).
#
# Τρέξιμο (detached): bash scripts/load_contest_ss_xgb_dense.sh > logs/load_contest_ss_xgb_dense.log 2>&1

cd "C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter" || exit 1

PY="conda run -n epf --no-capture-output python -X utf8"
OUT="runs/load_contest_ss"
mkdir -p "$OUT"

SU_TS="2025-06-01 00:00";  SU_TE="2025-08-31 23:00"
ON_TS="2025-10-01 00:00";  ON_TE="2025-11-30 23:00"
stamp() { date +"%Y-%m-%d %H:%M:%S"; }

mf() {  # mf <out_json> <extra args...>
    local oj="$1"; shift
    if [ -f "$oj" ]; then echo "=== [$(stamp)] SKIP  $oj"; return 0; fi
    echo ""; echo "=== [$(stamp)] RUN -> $oj"
    if $PY -m src.master_forecast "$@" --seed 42 --out_json "$oj"; then
        echo "--- [$(stamp)] OK  $oj"
    else
        echo "!!! [$(stamp)] FAILED  $oj  (συνεχιζω)"
    fi
}

echo "############ SS x WEEKLY XGB (dense) START [$(stamp)] ############"
echo "GIT: $(git rev-parse HEAD) | DIRTY: $(git status --porcelain src/ | tr '\n' ' ')"

echo ""; echo "######## BLOCK 0 — PREFLIGHT+POISON [$(stamp)] ########"
if ! $PY .claude/skills/energy-forecast/scripts/preflight_check.py --poison; then
    echo "!!!!! PREFLIGHT/POISON FAIL — ABORT"; exit 1
fi
echo "######## BLOCK 0 PASS ########"

mf "$OUT/octnov_xgb_recwss_g12_dense.json" \
   --algo xgb --task load --market dam --strategy recursive --gate strict \
   --retrain weekly --ss --ss_decay linear \
   --test_start "$ON_TS" --test_end "$ON_TE" --features "calendar,lags,roll,dense"
mf "$OUT/summer_xgb_recwss_g12_dense.json" \
   --algo xgb --task load --market dam --strategy recursive --gate strict \
   --retrain weekly --ss --ss_decay linear \
   --test_start "$SU_TS" --test_end "$SU_TE" --features "calendar,lags,roll,dense"

echo ""; echo "############ SS x WEEKLY XGB (dense) END [$(stamp)] ############"
