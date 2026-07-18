#!/usr/bin/env bash
# B4 CONFORMAL BATCH (last.md §2 B-σειρα: ≥2 μοντελα × ≥2 windows):
#   S1  split-conformal πανω σε 4 υπαρχοντα point-forecast JSONs (LGBM+XGB × Q1+summer,
#       weekly default_dense απο runs/overnight_20260705/a_cadence) — γρηγορο, no training
#   S2  quantile-LGBM baseline (α=0.1/0.5/0.9) × {Q1, summer}, weekly, default,dense
# Μετρικες ΠΑΝΤΑ μαζι: pinball + coverage + sharpness (SKILL.md §Conformal).
# Idempotent, ΕΝΑ conda process. Τρεξιμο (detached):
#   bash scripts/conformal_b4_batch.sh > logs/conformal_b4_batch.log 2>&1

cd "C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter" || exit 1

PY="conda run -n epf --no-capture-output python -X utf8"
OUT="runs/b4_conformal"
mkdir -p "$OUT"

SU_TS="2025-06-01 00:00";  SU_TE="2025-08-31 23:00"
Q1_TS="2025-12-01 00:00";  Q1_TE="2026-02-28 23:00"
stamp() { date +"%Y-%m-%d %H:%M:%S"; }

run_step() {  # run_step <out_json> <args...>
    local oj="$1"; shift
    if [ -f "$oj" ]; then echo "=== [$(stamp)] SKIP  $oj"; return 0; fi
    echo ""; echo "=== [$(stamp)] RUN -> $oj"
    local t0=$(date +%s)
    if $PY -m src.conformal "$@" --out_json "$oj"; then
        echo "--- [$(stamp)] OK  $oj  ($(( $(date +%s) - t0 ))s)"
    else
        echo "!!! [$(stamp)] FAILED  $oj  (συνεχιζω)"
    fi
}

echo "############ B4 CONFORMAL BATCH START [$(stamp)] ############"
echo "GIT: $(git rev-parse HEAD) | DIRTY: $(git status --porcelain src/ | tr '\n' ' ')"

echo ""; echo "######## BLOCK 0 — PREFLIGHT [$(stamp)] ########"
if ! $PY .claude/skills/energy-forecast/scripts/preflight_check.py; then
    echo "!!!!! PREFLIGHT FAIL — ABORT"; exit 1
fi
echo "######## BLOCK 0 PASS ########"

echo ""; echo "######## STAGE 1/2 — split-conformal σε 4 point JSONs [$(stamp)] ########"
A="runs/overnight_20260705/a_cadence"
run_step "$OUT/q1_lgbm_weekly_dense_splitconf.json"     split-conformal --forecast_json "$A/q1_lgbm_weekly_default_dense.json"
run_step "$OUT/summer_lgbm_weekly_dense_splitconf.json" split-conformal --forecast_json "$A/summer_lgbm_weekly_default_dense.json"
run_step "$OUT/q1_xgb_weekly_dense_splitconf.json"      split-conformal --forecast_json "$A/q1_xgb_weekly_default_dense.json"
run_step "$OUT/summer_xgb_weekly_dense_splitconf.json"  split-conformal --forecast_json "$A/summer_xgb_weekly_default_dense.json"
echo "######## STAGE 1/2 DONE [$(stamp)] ########"

echo ""; echo "######## STAGE 2/2 — quantile-LGBM baseline × 2 windows [$(stamp)] ########"
run_step "$OUT/q1_qlgbm_weekly_dense.json"     quantile-lgbm --test_start "$Q1_TS" --test_end "$Q1_TE" --retrain weekly --features "default,dense" --gate strict --market dam --task price --seed 42
run_step "$OUT/summer_qlgbm_weekly_dense.json" quantile-lgbm --test_start "$SU_TS" --test_end "$SU_TE" --retrain weekly --features "default,dense" --gate strict --market dam --task price --seed 42
echo "######## STAGE 2/2 DONE [$(stamp)] ########"

echo ""; echo "############ B4 CONFORMAL BATCH COMPLETE [$(stamp)] ############"
echo "Harvest: pinball+coverage+sharpness ΜΑΖΙ, συγκριση split-conformal vs quantile-LGBM"
echo "  ανα window — ΠΟΤΕ coverage μονο του (SKILL.md κανονας 4)."
