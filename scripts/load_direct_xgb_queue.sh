#!/usr/bin/env bash
# STAGE: direct XGB g12, 5 clean arms x 3 windows (15 runs) — cross-algo confirm του
# direct verdict (§7.23: LGBM direct dense 3/3 βοηθαει, genlags 3/3 βλαπτει, recursive
# κερδιζει octnov/q1). g12 = χωρις --delay. Σειρα windows: octnov (γρηγορο) → summer → q1.
# Idempotent (SKIP αν υπαρχει το JSON), ΕΝΑ conda process, συνεχιζει σε FAILED run.
# Τρεξιμο (detached): bash scripts/load_direct_xgb_queue.sh > logs/load_direct_xgb_queue.log 2>&1

cd "C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter" || exit 1

PY="conda run -n epf --no-capture-output python -X utf8"
mkdir -p runs/load_direct

SU_TS="2025-06-01 00:00";  SU_TE="2025-08-31 23:00"
ON_TS="2025-10-01 00:00";  ON_TE="2025-11-30 23:00"
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

echo "############ DIRECT XGB QUEUE (3 windows) START [$(stamp)] ############"
echo "GIT: $(git rev-parse HEAD) | DIRTY: $(git status --porcelain src/ | tr '\n' ' ')"

echo ""; echo "######## BLOCK 0 — PREFLIGHT+POISON [$(stamp)] ########"
if ! $PY .claude/skills/energy-forecast/scripts/preflight_check.py --poison; then
    echo "!!!!! PREFLIGHT/POISON FAIL — ABORT"; exit 1
fi
echo "######## BLOCK 0 PASS ########"

dir_win() {  # dir_win <winlabel> <ts> <te>
  local w="$1" ts="$2" te="$3"
  local C=(--algo xgb --task load --market dam --strategy direct --gate strict --retrain weekly --test_start "$ts" --test_end "$te")
  mf "runs/load_direct/${w}_xgb_dirw_g12_base.json"    "${C[@]}" --features "calendar,lags,roll"
  mf "runs/load_direct/${w}_xgb_dirw_g12_dense.json"   "${C[@]}" --features "calendar,lags,roll,dense"
  mf "runs/load_direct/${w}_xgb_dirw_g12_genlags.json" "${C[@]}" --features "calendar,lags,roll,genlags"
  mf "runs/load_direct/${w}_xgb_dirw_g12_noroll.json"  "${C[@]}" --features "calendar,lags"
  mf "runs/load_direct/${w}_xgb_dirw_g12_loadfc.json"  "${C[@]}" --features "calendar,lags,roll,loadfc"
}

echo ""; echo "######## STAGE 1/3 — direct XGB octnov [$(stamp)] ########"
dir_win "octnov" "$ON_TS" "$ON_TE"
echo "######## STAGE 1/3 DONE [$(stamp)] ########"

echo ""; echo "######## STAGE 2/3 — direct XGB summer [$(stamp)] ########"
dir_win "summer" "$SU_TS" "$SU_TE"
echo "######## STAGE 2/3 DONE [$(stamp)] ########"

echo ""; echo "######## STAGE 3/3 — direct XGB q1 [$(stamp)] ########"
dir_win "q1" "$Q1_TS" "$Q1_TE"
echo "######## STAGE 3/3 DONE [$(stamp)] ########"

echo ""; echo "############ DIRECT XGB QUEUE COMPLETE [$(stamp)] ############"
echo "Harvest: synthesize-ablation ανα window (baseline το dirw_g12_base του window) +"
echo "  direct-vs-recursive vs runs/load_contest/<w>_xgb_recw_g12_* — ΠΟΤΕ αναμεικτα."
