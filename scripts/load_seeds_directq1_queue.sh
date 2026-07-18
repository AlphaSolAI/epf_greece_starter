#!/usr/bin/env bash
# MEGA QUEUE 4 σταδιων (goal χρηστη 2026-07-17 "παμε καντα ολα, μην σταματησεις"):
#   S1  LGBM seeds summer  {g12,g14} x {dense,mv,densemv} x {7,123}   (12 runs, ~2-3h)
#   S2  LGBM seeds q1      {g12,g14} x {dense,mv,densemv} x {7,123}   (12 runs, ~3-4h)
#   S3  XGB  seeds dense   {summer,q1,octnov} x {g12,g14} x {7,123}   (12 runs, ~3-4h)
#   S4  direct LGBM q1 g12, 5 clean arms, direct weekly               (5 runs, ~10-14h — loadfc ΑΡΓΟ)
# Σειρα: seeds ΠΡΩΤΑ (γρηγορα, ξεκλειδωνουν headline-eligibility) — direct q1 ΤΕΛΕΥΤΑΙΟ (αργο).
# Mirror του scripts/load_contest_octnov_seeds.sh (idempotent SKIP, ιδιο naming contract).
# Idempotent, ΕΝΑ conda process, συνεχιζει σε FAILED run.
# Τρεξιμο (detached): bash scripts/load_seeds_directq1_queue.sh > logs/load_seeds_directq1_queue.log 2>&1

cd "C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter" || exit 1

PY="conda run -n epf --no-capture-output python -X utf8"
OUT="runs/load_contest_seeds"
mkdir -p "$OUT" runs/load_direct

SU_TS="2025-06-01 00:00";  SU_TE="2025-08-31 23:00"
ON_TS="2025-10-01 00:00";  ON_TE="2025-11-30 23:00"
Q1_TS="2025-12-01 00:00";  Q1_TE="2026-02-28 23:00"
stamp() { date +"%Y-%m-%d %H:%M:%S"; }

mf() {  # mf <out_json> <extra args...>  (το seed δινεται ΡΗΤΑ απο τον caller)
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

echo "############ SEEDS+DIRECTQ1 QUEUE (4 stages) START [$(stamp)] ############"
echo "GIT: $(git rev-parse HEAD) | DIRTY: $(git status --porcelain src/ | tr '\n' ' ')"

echo ""; echo "######## BLOCK 0 — PREFLIGHT+POISON [$(stamp)] ########"
if ! $PY .claude/skills/energy-forecast/scripts/preflight_check.py --poison; then
    echo "!!!!! PREFLIGHT/POISON FAIL — ABORT"; exit 1
fi
echo "######## BLOCK 0 PASS ########"

seed_lgbm() {  # seed_lgbm <winlabel> <ts> <te> <seed> <gslug> <delay_args> <slug> <features>
    local w="$1" ts="$2" te="$3" sd="$4" g="$5" dly="$6" s="$7" feats="$8"
    mf "$OUT/${w}_lgbm_recw_${g}_${s}_s${sd}.json" \
       --algo lgbm --task load --market dam --strategy recursive --gate strict $dly \
       --retrain weekly --test_start "$ts" --test_end "$te" --features "$feats" --seed "$sd"
}

lgbm_seed_stage() {  # lgbm_seed_stage <winlabel> <ts> <te>
  for sd in 7 123; do
    for gate in "g12:" "g14:--delay 14"; do
      gslug="${gate%%:*}"; dly="${gate#*:}"
      echo ""; echo "---- $1 seed $sd gate $gslug [$(stamp)] ----"
      seed_lgbm "$1" "$2" "$3" "$sd" "$gslug" "$dly" "dense"   "calendar,lags,roll,dense"
      seed_lgbm "$1" "$2" "$3" "$sd" "$gslug" "$dly" "mv"      "calendar,lags,roll,meteo_vintage"
      seed_lgbm "$1" "$2" "$3" "$sd" "$gslug" "$dly" "densemv" "calendar,lags,roll,dense,meteo_vintage"
    done
  done
}

echo ""; echo "######## STAGE 1/4 — LGBM seeds summer [$(stamp)] ########"
lgbm_seed_stage "summer" "$SU_TS" "$SU_TE"
echo "######## STAGE 1/4 DONE [$(stamp)] ########"

echo ""; echo "######## STAGE 2/4 — LGBM seeds q1 [$(stamp)] ########"
lgbm_seed_stage "q1" "$Q1_TS" "$Q1_TE"
echo "######## STAGE 2/4 DONE [$(stamp)] ########"

echo ""; echo "######## STAGE 3/4 — XGB dense seeds (3 windows x 2 gates) [$(stamp)] ########"
seed_xgb() {  # seed_xgb <winlabel> <ts> <te> <seed> <gslug> <delay_args>
    local w="$1" ts="$2" te="$3" sd="$4" g="$5" dly="$6"
    mf "$OUT/${w}_xgb_recw_${g}_dense_s${sd}.json" \
       --algo xgb --task load --market dam --strategy recursive --gate strict $dly \
       --retrain weekly --test_start "$ts" --test_end "$te" --features "calendar,lags,roll,dense" --seed "$sd"
}
for sd in 7 123; do
  for gate in "g12:" "g14:--delay 14"; do
    gslug="${gate%%:*}"; dly="${gate#*:}"
    echo ""; echo "---- XGB seed $sd gate $gslug [$(stamp)] ----"
    seed_xgb "summer" "$SU_TS" "$SU_TE" "$sd" "$gslug" "$dly"
    seed_xgb "q1"     "$Q1_TS" "$Q1_TE" "$sd" "$gslug" "$dly"
    seed_xgb "octnov" "$ON_TS" "$ON_TE" "$sd" "$gslug" "$dly"
  done
done
echo "######## STAGE 3/4 DONE [$(stamp)] ########"

echo ""; echo "######## STAGE 4/4 — direct LGBM q1 g12, 5 clean arms (loadfc ΑΡΓΟ ~5-6h) [$(stamp)] ########"
dir_arm() {  # dir_arm <slug> <features>
  mf "runs/load_direct/q1_lgbm_dirw_g12_$1.json" \
     --algo lgbm --task load --market dam --strategy direct --gate strict \
     --retrain weekly --test_start "$Q1_TS" --test_end "$Q1_TE" --features "$2" --seed 42
}
dir_arm "base"    "calendar,lags,roll"
dir_arm "dense"   "calendar,lags,roll,dense"
dir_arm "genlags" "calendar,lags,roll,genlags"
dir_arm "noroll"  "calendar,lags"
dir_arm "loadfc"  "calendar,lags,roll,loadfc"
echo "######## STAGE 4/4 DONE [$(stamp)] ########"

echo ""; echo "############ SEEDS+DIRECTQ1 QUEUE COMPLETE [$(stamp)] ############"
echo "Harvest: python scripts/build_run_ledger.py + synthesize-ablation:"
echo "  seeds → std ανα (window,gate,spec) μαζι με το seed42 απο runs/load_contest —"
echo "  direct q1 → baseline q1_lgbm_dirw_g12_base, ΠΟΤΕ αναμεικτα με recursive/g14."
