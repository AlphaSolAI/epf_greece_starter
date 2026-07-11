#!/usr/bin/env bash
# LOAD CONTEST (G7) — SS x weekly XGB πληρης εικονα (2026-07-11, goal χρήστη «6 runs»)
# ------------------------------------------------------------------------------------------
# Ολοκληρωση του XGB SS grid ωστε να ΑΝΤΙΣΤΟΙΧΕΙ 1:1 με το LGBM overnight grid -> καθαρη
# cross-algo συγκριση στα ΙΔΙΑ κελια. base+dense x octnov+summer x g12+g14 = 8 κελια·
# τα 2 (dense g12) ΥΠΑΡΧΟΥΝ ηδη -> idempotent SKIP -> 6 νεα runs.
# XGB non-SS baselines ολα υπαρχουν (Batch 2, runs/load_contest/*_xgb_recw_*) -> ΔMAE αμεσο.
# q1 ΕΞΩ (bounded-out, runtime — και το LGBM SS δεν το εχει, οποτε δεν χαλαει το mirror).
#
# Τρέξιμο (detached): bash scripts/load_contest_ss_xgb_weekly.sh > logs/load_contest_ss_xgb_weekly.log 2>&1

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

echo "############ SS x WEEKLY XGB (full grid) START [$(stamp)] ############"
echo "GIT: $(git rev-parse HEAD) | DIRTY: $(git status --porcelain src/ | tr '\n' ' ')"

echo ""; echo "######## BLOCK 0 — PREFLIGHT+POISON [$(stamp)] ########"
if ! $PY .claude/skills/energy-forecast/scripts/preflight_check.py --poison; then
    echo "!!!!! PREFLIGHT/POISON FAIL — ABORT"; exit 1
fi
echo "######## BLOCK 0 PASS ########"

run_spec() {  # run_spec <win> <ts> <te> <gslug> <delay_args> <slug> <features>
    local w="$1" ts="$2" te="$3" g="$4" dly="$5" s="$6" feats="$7"
    mf "$OUT/${w}_xgb_recwss_${g}_${s}.json" \
       --algo xgb --task load --market dam --strategy recursive --gate strict $dly \
       --retrain weekly --ss --ss_decay linear \
       --test_start "$ts" --test_end "$te" --features "$feats"
}

for gate in "g12:" "g14:--delay 14"; do
  gslug="${gate%%:*}"; dly="${gate#*:}"
  echo ""; echo "---- gate $gslug [$(stamp)] ----"
  run_spec "octnov" "$ON_TS" "$ON_TE" "$gslug" "$dly" "base"  "calendar,lags,roll"
  run_spec "octnov" "$ON_TS" "$ON_TE" "$gslug" "$dly" "dense" "calendar,lags,roll,dense"
  run_spec "summer" "$SU_TS" "$SU_TE" "$gslug" "$dly" "base"  "calendar,lags,roll"
  run_spec "summer" "$SU_TS" "$SU_TE" "$gslug" "$dly" "dense" "calendar,lags,roll,dense"
done

echo ""; echo "############ SS x WEEKLY XGB (full grid) END [$(stamp)] ############"
