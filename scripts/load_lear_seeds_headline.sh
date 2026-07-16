#!/usr/bin/env bash
# LEAR dense seeds {7,123} x {summer,octnov,q1} x {base,dense} — πλησιαζει headline-level
# για το πιο συνεπες μη-tree finding (3/3 windows ηδη, seed42). LEAR = deterministic/φθηνο,
# ασφαλες με πολλαπλα seeds χωρις το MLP failure mode.
# Τρεξιμο (detached): bash scripts/load_lear_seeds_headline.sh > logs/load_lear_seeds.log 2>&1

cd "C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter" || exit 1

PY="conda run -n epf --no-capture-output python -X utf8"
OUT="runs/load_lear"
mkdir -p "$OUT"

SU_TS="2025-06-01 00:00";  SU_TE="2025-08-31 23:00"
ON_TS="2025-10-01 00:00";  ON_TE="2025-11-30 23:00"
Q1_TS="2025-12-01 00:00";  Q1_TE="2026-02-28 23:00"
stamp() { date +"%Y-%m-%d %H:%M:%S"; }

mf() {
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

echo "############ LEAR SEEDS (headline) START [$(stamp)] ############"
echo "GIT: $(git rev-parse HEAD) | DIRTY: $(git status --porcelain src/ | tr '\n' ' ')"

echo ""; echo "######## BLOCK 0 — PREFLIGHT+POISON [$(stamp)] ########"
if ! $PY .claude/skills/energy-forecast/scripts/preflight_check.py --poison; then
    echo "!!!!! PREFLIGHT/POISON FAIL — ABORT"; exit 1
fi
echo "######## BLOCK 0 PASS ########"

run_win() {  # run_win <label> <ts> <te>
  local w="$1" ts="$2" te="$3"
  local COMMON=(--algo lear --task load --market dam --strategy recursive --gate strict --retrain weekly --test_start "$ts" --test_end "$te")
  for seed in 7 123; do
    mf "$OUT/${w}_lear_recw_g12_base_seed${seed}.json"  "${COMMON[@]}" --seed "$seed" --features "calendar,lags,roll"
    mf "$OUT/${w}_lear_recw_g12_dense_seed${seed}.json" "${COMMON[@]}" --seed "$seed" --features "calendar,lags,roll,dense"
  done
}

run_win "summer" "$SU_TS" "$SU_TE"
run_win "octnov" "$ON_TS" "$ON_TE"
run_win "q1"     "$Q1_TS" "$Q1_TE"

echo ""; echo "############ LEAR SEEDS (headline) END [$(stamp)] ############"
