#!/usr/bin/env bash
# LOAD CONTEST (G7) — SS x weekly probe, LGBM recursive (overnight 2026-07-11, goal χρήστη)
# ------------------------------------------------------------------------------------------
# Επόμενη G7 στάση (σειρά χρήστη LGBM->XGB->SS->direct). SS = scheduled sampling (linear
# decay), ΜΟΝΟ recursive (code constraint). Πρώτη φορά SS x weekly για LOAD.
# ΦΡΑΓΜΕΝΟ ΣΚΟΠΙΜΑ (οδηγία χρήστη «οχι 3ωρα runs», 2026-07-11): SS κανει ss_rounds=3
# self-retraining => ~3-4x χρονος καθε run. Γι' αυτο:
#   - specs ΜΟΝΟ {base, dense} (ΟΧΙ mv/densemv = +84 features -> πολυωρα runs)
#   - windows ΜΟΝΟ {octnov (γρηγορο), summer} = 2 ανεξαρτητα windows για §2 signal
#   - q1 (το πιο αργο) ΕΞΩ -> ξεχωριστο follow-up κομματι με OK χρηστη
# Grid: 2 specs x 2 windows x 2 gates = 8 runs. seed 42. Idempotent. outdir runs/load_contest_ss/.
# Baseline συγκρισης: τα ΑΝΤΙΣΤΟΙΧΑ non-SS runs στο runs/load_contest/ (ιδιο seed/gate/window/spec).
#
# Τρέξιμο (detached): bash scripts/load_contest_ss_lgbm_weekly.sh > logs/load_contest_ss_lgbm_weekly.log 2>&1

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

echo "############ SS x WEEKLY PROBE START [$(stamp)] ############"
echo "GIT: $(git rev-parse HEAD) | DIRTY: $(git status --porcelain src/ | tr '\n' ' ')"

echo ""; echo "######## BLOCK 0 — PREFLIGHT+POISON [$(stamp)] ########"
if ! $PY .claude/skills/energy-forecast/scripts/preflight_check.py --poison; then
    echo "!!!!! PREFLIGHT/POISON FAIL — ABORT"; exit 1
fi
echo "######## BLOCK 0 PASS ########"

run_spec() {  # run_spec <win> <ts> <te> <gslug> <delay_args> <slug> <features>
    local w="$1" ts="$2" te="$3" g="$4" dly="$5" s="$6" feats="$7"
    mf "$OUT/${w}_lgbm_recwss_${g}_${s}.json" \
       --algo lgbm --task load --market dam --strategy recursive --gate strict $dly \
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

echo ""; echo "############ SS x WEEKLY PROBE END [$(stamp)] ############"
