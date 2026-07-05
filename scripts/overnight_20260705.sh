#!/bin/bash
# overnight_20260705.sh — Ολονύχτιο leak-free batch (μετά το B3 core).
# Σχεδίαση: ΕΝΑ conda process, ΟΛΑ σειριακά, continue-on-error (πρωινό debug).
# Σειρά blocks = αξία/ώρα: αν κρασάρει στις 04:00, τα κρίσιμα έχουν ήδη βγει.
#
# Block 0: validity gate (preflight --poison + dense y-path) — FAIL = ABORT ΟΛΩΝ
# Block A: retrain cadence (weekly/monthly) top-3 specs → υποψήφιο ΝΕΟ HEADLINE
# Block B: 3ο window Μάρτιος 2026 (tie-break των MIXED: resfc/lean/genlags/dense×dir)
# Block C: LOAD full ablation (7 specs × lgbm+xgb × rec+dir × 2 windows)
# Block D: dense×direct fill (q1/summer) + seeds robustness
# Block E: SS × {default, default+dense}
# Block F: conformal smoke (split-conformal + quantile-lgbm) στο q1 weekly dense
#
# Τρέξιμο:  bash scripts/overnight_20260705.sh 2>&1 | tee logs/overnight_20260705_master.log
# Πρωί:     conda run -n epf --no-capture-output python -X utf8 scripts/overnight_summarize.py

cd "C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter" || exit 1

PY="conda run -n epf --no-capture-output python -X utf8"
OUT="runs/overnight_20260705"
mkdir -p "$OUT"/{a_cadence,b_march,c_load,d_fill,e_ss,f_conformal}

# Παράθυρα
Q1_TR_END="2025-11-30 23:00";  Q1_TS="2025-12-01 00:00";  Q1_TE="2026-02-28 23:00"
SU_TR_END="2025-05-31 23:00";  SU_TS="2025-06-01 00:00";  SU_TE="2025-08-31 23:00"
MR_TR_END="2026-02-28 23:00";  MR_TS="2026-03-01 00:00";  MR_TE="2026-03-20 23:00"

stamp() { date +"%Y-%m-%d %H:%M:%S"; }

# slug: ',' -> '_', '-' -> 'no'
slug() { echo "$1" | sed 's/,-/_no/g; s/,/_/g'; }

mf() {  # mf <out_json> <extra args...>
    local oj="$1"; shift
    echo ""
    echo "=== [$(stamp)] RUN → $oj"
    echo "    CMD: python -m src.master_forecast $*"
    if $PY -m src.master_forecast "$@" --out_json "$oj"; then
        echo "--- [$(stamp)] OK  $oj"
    else
        echo "!!! [$(stamp)] FAILED  $oj  (συνεχίζω στο επόμενο)"
    fi
}

echo "############ OVERNIGHT 2026-07-05 START [$(stamp)] ############"

# ---------------------------------------------------------------- Block 0
echo ""
echo "######## BLOCK 0 — VALIDITY GATE [$(stamp)] (αναμενόμενο ~5') ########"
if ! $PY .claude/skills/energy-forecast/scripts/preflight_check.py --poison; then
    echo "!!!!! PREFLIGHT/POISON FAIL — ABORT ΟΛΟΥ ΤΟΥ BATCH (κανένα νούμερο πάνω σε σπασμένο θεμέλιο)"
    exit 1
fi
if ! $PY -m src.check_crosslag_fairness --features default,dense --poison_y; then
    echo "!!!!! DENSE Y-PATH POISON FAIL — ABORT"
    exit 1
fi
echo "######## BLOCK 0 PASS ########"

# ---------------------------------------------------------------- Block A
echo ""
echo "######## BLOCK A — RETRAIN CADENCE [$(stamp)] (~3.5h) ########"
# weekly πρώτα (το πιο πολύτιμο), μετά monthly· lgbm πριν xgb
for cadence in weekly monthly; do
  for algo in lgbm xgb; do
    for spec in "default,dense" "default,-resfc" "default"; do
      s=$(slug "$spec")
      mf "$OUT/a_cadence/q1_${algo}_${cadence}_${s}.json" \
         --algo $algo --task price --market dam --strategy recursive --gate strict \
         --retrain $cadence --test_start "$Q1_TS" --test_end "$Q1_TE" --features "$spec"
      mf "$OUT/a_cadence/summer_${algo}_${cadence}_${s}.json" \
         --algo $algo --task price --market dam --strategy recursive --gate strict \
         --retrain $cadence --test_start "$SU_TS" --test_end "$SU_TE" --features "$spec"
    done
  done
done

# ---------------------------------------------------------------- Block B
echo ""
echo "######## BLOCK B — ΜΑΡΤΙΟΣ 2026 TIE-BREAK [$(stamp)] (~1h) ########"
for algo in lgbm xgb; do
  for strat in recursive direct; do
    for spec in "default" "default,-meteo" "default,-resfc" "default,-genlags,-loadlags" \
                "lags,calendar,genlags" "default,dense" "lags,calendar"; do
      s=$(slug "$spec")
      mf "$OUT/b_march/march_${algo}_${strat:0:3}_${s}.json" \
         --algo $algo --task price --market dam --strategy $strat --gate strict \
         --retrain static --train_end "$MR_TR_END" --test_start "$MR_TS" --test_end "$MR_TE" \
         --features "$spec"
    done
  done
done

# ---------------------------------------------------------------- Block C
echo ""
echo "######## BLOCK C — LOAD FULL ABLATION [$(stamp)] (~1.5-2h) ########"
for algo in lgbm xgb; do
  for strat in recursive direct; do
    for spec in "default" "default,-meteo" "default,-loadfc" "default,-loadlags" \
                "default,-genlags,-loadlags" "lags,calendar,loadfc" "default,dense"; do
      s=$(slug "$spec")
      mf "$OUT/c_load/q1_${algo}_${strat:0:3}_${s}.json" \
         --algo $algo --task load --market dam --strategy $strat --gate strict \
         --retrain static --train_end "$Q1_TR_END" --test_start "$Q1_TS" --test_end "$Q1_TE" \
         --features "$spec"
      mf "$OUT/c_load/summer_${algo}_${strat:0:3}_${s}.json" \
         --algo $algo --task load --market dam --strategy $strat --gate strict \
         --retrain static --train_end "$SU_TR_END" --test_start "$SU_TS" --test_end "$SU_TE" \
         --features "$spec"
    done
  done
done

# ---------------------------------------------------------------- Block D
echo ""
echo "######## BLOCK D — DENSE×DIRECT FILL + SEEDS [$(stamp)] (~25') ########"
for algo in lgbm xgb; do
  mf "$OUT/d_fill/q1_${algo}_dir_default_dense.json" \
     --algo $algo --task price --market dam --strategy direct --gate strict \
     --retrain static --train_end "$Q1_TR_END" --test_start "$Q1_TS" --test_end "$Q1_TE" \
     --features "default,dense"
  mf "$OUT/d_fill/summer_${algo}_dir_default_dense.json" \
     --algo $algo --task price --market dam --strategy direct --gate strict \
     --retrain static --train_end "$SU_TR_END" --test_start "$SU_TS" --test_end "$SU_TE" \
     --features "default,dense"
done
# seeds robustness (42 υπάρχει ήδη από B3)
for seed in 7 123; do
  mf "$OUT/d_fill/q1_lgbm_rec_default_dense_seed${seed}.json" \
     --algo lgbm --task price --market dam --strategy recursive --gate strict \
     --retrain static --train_end "$Q1_TR_END" --test_start "$Q1_TS" --test_end "$Q1_TE" \
     --features "default,dense" --seed $seed
  mf "$OUT/d_fill/summer_lgbm_rec_default_dense_seed${seed}.json" \
     --algo lgbm --task price --market dam --strategy recursive --gate strict \
     --retrain static --train_end "$SU_TR_END" --test_start "$SU_TS" --test_end "$SU_TE" \
     --features "default,dense" --seed $seed
done

# ---------------------------------------------------------------- Block E
echo ""
echo "######## BLOCK E — SCHEDULED SAMPLING [$(stamp)] (~20') ########"
# Σημ.: το "published actuals όσο επιτρέπεται" ισχύει ΗΔΗ by default για price/DAM
# (gap=0: cutoff=23:00 D-1, poisoning-proven 2026-07-05) — το SS θεραπεύει μόνο το
# same-day mismatch που απομένει.
for spec in "default" "default,dense"; do
  s=$(slug "$spec")
  mf "$OUT/e_ss/q1_lgbm_rec_ss_${s}.json" \
     --algo lgbm --task price --market dam --strategy recursive --gate strict \
     --retrain static --train_end "$Q1_TR_END" --test_start "$Q1_TS" --test_end "$Q1_TE" \
     --features "$spec" --ss --ss_rounds 3 --ss_decay linear
  mf "$OUT/e_ss/summer_lgbm_rec_ss_${s}.json" \
     --algo lgbm --task price --market dam --strategy recursive --gate strict \
     --retrain static --train_end "$SU_TR_END" --test_start "$SU_TS" --test_end "$SU_TE" \
     --features "$spec" --ss --ss_rounds 3 --ss_decay linear
done

# ---------------------------------------------------------------- Block F
echo ""
echo "######## BLOCK F — CONFORMAL SMOKE [$(stamp)] (~45') ########"
FJ="$OUT/a_cadence/q1_lgbm_weekly_default_dense.json"
if [ -f "$FJ" ]; then
    echo "=== [$(stamp)] split-conformal πάνω στο $FJ"
    $PY -m src.conformal split-conformal --forecast_json "$FJ" \
        --out_json "$OUT/f_conformal/q1_splitconformal_weekly_dense.json" \
        || echo "!!! FAILED split-conformal"
else
    echo "!!! SKIP split-conformal — δεν υπάρχει $FJ (κοίτα Block A failures)"
fi
echo "=== [$(stamp)] quantile-lgbm weekly default,dense Q1"
$PY -m src.conformal quantile-lgbm --test_start "$Q1_TS" --test_end "$Q1_TE" \
    --retrain weekly --features "default,dense" --gate strict --market dam --task price \
    --out_json "$OUT/f_conformal/q1_qlgbm_weekly_dense.json" \
    || echo "!!! FAILED quantile-lgbm"

echo ""
echo "############ OVERNIGHT DONE [$(stamp)] ############"
echo "Πρωί: conda run -n epf --no-capture-output python -X utf8 scripts/overnight_summarize.py"
