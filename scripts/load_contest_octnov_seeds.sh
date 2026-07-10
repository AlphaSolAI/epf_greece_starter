#!/usr/bin/env bash
# LOAD CONTEST (G7) — octnov seed hardening (overnight 2026-07-11, goal χρήστη)
# ------------------------------------------------------------------------------------------
# Σκοπός: κλείδωμα του φρέσκου ευρήματος «καθαρό/vintage μοντέλο κερδίζει ΑΔΜΗΕ στο octnov»
# (ΑΔΜΗΕ=146.81) με 2ο+3ο seed (7,123) — σήμερα μόνο seed 42. ΜΟΝΟ octnov (γρήγορο
# window, ~3-5'/run) ώστε κάθε run να είναι μικρό, ΟΧΙ πολύωρο (οδηγία χρήστη 2026-07-11).
# Δεν αλλάζει τίποτα ΔΕΚΤΟ — απλώς seed-variance evidence, harvest+κρίση με τον χρήστη.
#
# Grid: seeds {7,123} x specs {dense, mv, densemv} x gates {g12,g14} = 12 runs.
# Idempotent (SKIP σε υπάρχον JSON). Ξεχωριστό outdir (seed-suffixed ονόματα ΔΕΝ
# ταιριάζουν στο 5-part contract του load_contest_report.py).
#
# Τρέξιμο (detached): bash scripts/load_contest_octnov_seeds.sh > logs/load_contest_octnov_seeds.log 2>&1

cd "C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter" || exit 1

PY="conda run -n epf --no-capture-output python -X utf8"
OUT="runs/load_contest_seeds"
mkdir -p "$OUT"

ON_TS="2025-10-01 00:00";  ON_TE="2025-11-30 23:00"
stamp() { date +"%Y-%m-%d %H:%M:%S"; }

mf() {  # mf <out_json> <extra args...>
    local oj="$1"; shift
    if [ -f "$oj" ]; then echo "=== [$(stamp)] SKIP  $oj"; return 0; fi
    echo ""; echo "=== [$(stamp)] RUN -> $oj"
    if $PY -m src.master_forecast "$@" --out_json "$oj"; then
        echo "--- [$(stamp)] OK  $oj"
    else
        echo "!!! [$(stamp)] FAILED  $oj  (συνεχιζω)"
    fi
}

echo "############ octnov SEED HARDENING START [$(stamp)] ############"
echo "GIT: $(git rev-parse HEAD) | DIRTY: $(git status --porcelain src/ | tr '\n' ' ')"

echo ""; echo "######## BLOCK 0 — PREFLIGHT+POISON [$(stamp)] ########"
if ! $PY .claude/skills/energy-forecast/scripts/preflight_check.py --poison; then
    echo "!!!!! PREFLIGHT/POISON FAIL — ABORT"; exit 1
fi
echo "######## BLOCK 0 PASS ########"

run_spec() {  # run_spec <seed> <gslug> <delay_args> <slug> <features>
    local sd="$1" g="$2" dly="$3" s="$4" feats="$5"
    mf "$OUT/octnov_lgbm_recw_${g}_${s}_s${sd}.json" \
       --algo lgbm --task load --market dam --strategy recursive --gate strict $dly \
       --retrain weekly --test_start "$ON_TS" --test_end "$ON_TE" --features "$feats" --seed "$sd"
}

for sd in 7 123; do
  for gate in "g12:" "g14:--delay 14"; do
    gslug="${gate%%:*}"; dly="${gate#*:}"
    echo ""; echo "---- seed $sd gate $gslug [$(stamp)] ----"
    run_spec "$sd" "$gslug" "$dly" "dense"   "calendar,lags,roll,dense"
    run_spec "$sd" "$gslug" "$dly" "mv"      "calendar,lags,roll,meteo_vintage"
    run_spec "$sd" "$gslug" "$dly" "densemv" "calendar,lags,roll,dense,meteo_vintage"
  done
done

echo ""; echo "############ octnov SEED HARDENING END [$(stamp)] ############"
