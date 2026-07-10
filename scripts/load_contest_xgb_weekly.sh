#!/usr/bin/env bash
# LOAD CONTEST (G7) — Batch 2: XGB weekly recursive, ΚΑΘΑΡΑ arms (goal χρήστη 2026-07-10)
# ------------------------------------------------------------------------------------------
# Cross-model confirmation του LGBM (Batch 1). ΙΔΙΟ grid, --algo xgb.
# Σειρά: LGBM(1) -> **XGB(2)** -> SS(3) -> direct(4). Τρέχει ΜΕΤΑ το LGBM (ΕΝΑ conda).
#
# Arms (5, καθαρά): base / +dense / +genlags / -roll / +loadfc  (χωρίς meteo=oracle, χωρίς lean)
# GATE: g12 (δικό μας, gap=12, cutoff 11:00 CET) · g14 (ΑΔΜΗΕ-aligned, --delay 14, cutoff 09:00 CET)
# Grid: 5 specs x XGB x recursive x weekly x 3 windows x 2 gates = 30 runs, idempotent.
#
# Τρέξιμο (detached): bash scripts/load_contest_xgb_weekly.sh > logs/load_contest_xgb_weekly.log 2>&1

cd "C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter" || exit 1

PY="conda run -n epf --no-capture-output python -X utf8"
OUT="runs/load_contest"
mkdir -p "$OUT"

Q1_TS="2025-12-01 00:00";  Q1_TE="2026-02-28 23:00"
SU_TS="2025-06-01 00:00";  SU_TE="2025-08-31 23:00"
ON_TS="2025-10-01 00:00";  ON_TE="2025-11-30 23:00"

stamp() { date +"%Y-%m-%d %H:%M:%S"; }

mf() {  # mf <out_json> <extra args...>
    local oj="$1"; shift
    if [ -f "$oj" ]; then
        echo "=== [$(stamp)] SKIP (υπάρχει ήδη)  $oj"
        return 0
    fi
    echo ""
    echo "=== [$(stamp)] RUN → $oj"
    echo "    CMD: $PY -m src.master_forecast $* --seed 42 --out_json $oj"
    if $PY -m src.master_forecast "$@" --seed 42 --out_json "$oj"; then
        echo "--- [$(stamp)] OK  $oj"
    else
        echo "!!! [$(stamp)] FAILED  $oj  (συνεχίζω στο επόμενο)"
    fi
}

echo "############ LOAD CONTEST (G7) Batch 2 — xgb weekly rec, 5 clean specs x 2 gates START [$(stamp)] ############"
echo "GIT BASE: $(git rev-parse HEAD)"
echo "GIT DIRTY: $(git status --porcelain src/ | tr '\n' ' ')"
git status --porcelain > "$OUT/git_status_xgb_at_launch.txt"
git diff src/ > "$OUT/git_diff_src_xgb_at_launch.patch"

echo ""
echo "######## BLOCK 0 — PREFLIGHT+POISON [$(stamp)] ########"
if ! $PY .claude/skills/energy-forecast/scripts/preflight_check.py --poison; then
    echo "!!!!! PREFLIGHT/POISON FAIL — ABORT"
    exit 1
fi
echo "######## BLOCK 0 PASS ########"

echo ""
echo "######## BLOCK 1 — XGB RECURSIVE WEEKLY, 5 specs x 3 windows x 2 gates [$(stamp)] ########"
# $dly αχώριστο επίτηδες (word-split: "" -> τίποτα, "--delay 14" -> 2 args).
run_spec() {  # run_spec <gslug> <delay_args> <slug> <features>
    local g="$1" dly="$2" s="$3" feats="$4"
    mf "$OUT/q1_xgb_recw_${g}_${s}.json" \
       --algo xgb --task load --market dam --strategy recursive --gate strict $dly \
       --retrain weekly --test_start "$Q1_TS" --test_end "$Q1_TE" --features "$feats"
    mf "$OUT/summer_xgb_recw_${g}_${s}.json" \
       --algo xgb --task load --market dam --strategy recursive --gate strict $dly \
       --retrain weekly --test_start "$SU_TS" --test_end "$SU_TE" --features "$feats"
    mf "$OUT/octnov_xgb_recw_${g}_${s}.json" \
       --algo xgb --task load --market dam --strategy recursive --gate strict $dly \
       --retrain weekly --test_start "$ON_TS" --test_end "$ON_TE" --features "$feats"
}

for gate in "g12:" "g14:--delay 14"; do
    gslug="${gate%%:*}"; dly="${gate#*:}"
    echo ""
    echo "---- GATE $gslug (delay='${dly:-<default 12>}') [$(stamp)] ----"
    run_spec "$gslug" "$dly" "base"    "calendar,lags,roll"
    run_spec "$gslug" "$dly" "dense"   "calendar,lags,roll,dense"
    run_spec "$gslug" "$dly" "genlags" "calendar,lags,roll,genlags"
    run_spec "$gslug" "$dly" "noroll"  "calendar,lags"
    run_spec "$gslug" "$dly" "loadfc"  "calendar,lags,roll,loadfc"
done

echo ""
echo "############ LOAD CONTEST Batch 2 (xgb) END [$(stamp)] ############"
echo "Harvest: python scripts/build_run_ledger.py && synthesize-ablation (baseline: calendar,lags,roll)"
