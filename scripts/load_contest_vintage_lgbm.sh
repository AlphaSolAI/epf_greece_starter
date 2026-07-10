#!/usr/bin/env bash
# LOAD CONTEST (G7) — Batch 3: meteo_vintage arms, LGBM weekly recursive (goal χρήστη 2026-07-10)
# ------------------------------------------------------------------------------------------
# Πρώτο ΕΝΤΙΜΟ meteo arm του contest: vintage D-1/D-2 forecast (ομάδα meteo_vintage,
# docs/features/meteo_vintage/design.md) αντί για oracle w_* (last.md §2 Α6).
# Wiring validation ΠΡΙΝ από αυτό το batch (2026-07-10): pytest 94 PASS · poison PASS
# (logs/mv_wiring_preflight_poison.log) · control base Q1 static = 254.9893 ΤΑΥΤΟΣΗΜΟ ·
# smoke static Q1 mv: #features 17->101, MAE=221.28 (runs/feat_meteo_vintage/).
#
# Arms (slug ΧΩΡΙΣ underscore — 5-part filename contract του load_contest_report.py):
#   mv       calendar,lags,roll,meteo_vintage
#   densemv  calendar,lags,roll,dense,meteo_vintage   (dense = ισχυρότερο καθαρό candidate)
# GATES: g12 (δικό μας, gap=12) · g14 (ΑΔΜΗΕ-aligned, --delay 14) — βλ. Batch 1 header.
# Grid: 2 specs x 3 windows x 2 gates = 12 runs -> runs/load_contest/ (idempotent SKIP).
#
# Τρέξιμο (detached): bash scripts/load_contest_vintage_lgbm.sh > logs/load_contest_vintage_lgbm.log 2>&1

cd "C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter" || exit 1

PY="conda run -n epf --no-capture-output python -X utf8"
OUT="runs/load_contest"
mkdir -p "$OUT"

# weekly retrain = expanding window έως κάθε cutoff -> ΧΩΡΙΣ --train_end (κανόνας 8).
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

echo "############ LOAD CONTEST (G7) Batch 3 — lgbm weekly rec, meteo_vintage arms x 2 gates START [$(stamp)] ############"
echo "GIT BASE: $(git rev-parse HEAD)"
echo "GIT DIRTY: $(git status --porcelain src/ | tr '\n' ' ')"
git status --porcelain > "$OUT/git_status_at_launch_vintage.txt"
git diff src/ > "$OUT/git_diff_src_at_launch_vintage.patch"

echo ""
echo "######## BLOCK 0 — PREFLIGHT+POISON [$(stamp)] ########"
if ! $PY .claude/skills/energy-forecast/scripts/preflight_check.py --poison; then
    echo "!!!!! PREFLIGHT/POISON FAIL — ABORT"
    exit 1
fi
echo "######## BLOCK 0 PASS ########"

echo ""
echo "######## BLOCK 1 — LGBM RECURSIVE WEEKLY, 2 vintage specs x 3 windows x 2 gates [$(stamp)] ########"
# $dly ΑΦΗΝΕΤΑΙ αχώριστο επίτηδες (word-split: "" -> τίποτα, "--delay 14" -> 2 args).
run_spec() {  # run_spec <gslug> <delay_args> <slug> <features>
    local g="$1" dly="$2" s="$3" feats="$4"
    mf "$OUT/q1_lgbm_recw_${g}_${s}.json" \
       --algo lgbm --task load --market dam --strategy recursive --gate strict $dly \
       --retrain weekly --test_start "$Q1_TS" --test_end "$Q1_TE" --features "$feats"
    mf "$OUT/summer_lgbm_recw_${g}_${s}.json" \
       --algo lgbm --task load --market dam --strategy recursive --gate strict $dly \
       --retrain weekly --test_start "$SU_TS" --test_end "$SU_TE" --features "$feats"
    mf "$OUT/octnov_lgbm_recw_${g}_${s}.json" \
       --algo lgbm --task load --market dam --strategy recursive --gate strict $dly \
       --retrain weekly --test_start "$ON_TS" --test_end "$ON_TE" --features "$feats"
}

for gate in "g12:" "g14:--delay 14"; do
    gslug="${gate%%:*}"; dly="${gate#*:}"
    echo ""
    echo "---- GATE $gslug (delay='${dly:-<default 12>}') [$(stamp)] ----"
    run_spec "$gslug" "$dly" "mv"      "calendar,lags,roll,meteo_vintage"
    run_spec "$gslug" "$dly" "densemv" "calendar,lags,roll,dense,meteo_vintage"
done

echo ""
echo "############ LOAD CONTEST Batch 3 (vintage) END [$(stamp)] ############"
echo "Harvest: scripts/load_contest_report.py --algo lgbm --mode recw (baseline: base arm Batch 1)"
echo "  + T8 έλεγχος: MAE(mv) πρέπει ≥ MAE(oracle meteo αντίστοιχου config) − noise."
