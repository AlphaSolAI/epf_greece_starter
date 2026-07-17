#!/usr/bin/env bash
# QUEUE 4 σταδιων ΔΙΑΔΟΧΙΚΑ (ρητο OK χρηστη 2026-07-16 — "ΒΑΛΕ ΚΑΙ ΤΑ 4 ΔΙΑΔΟΧΙΚΑ"):
#   S1  LEAR g14  {base,dense} x 3 windows, recursive weekly            (6 runs, ~30-45min)
#   S2  MLP  g14  {base,dense} x 3 windows, recursive weekly            (6 runs, ~2-2.5h)
#   S3  LSTM g14  {base,loadfc} x 3 windows, recursive static           (6 runs, ~1.5-2.5h)
#   S4  direct LGBM summer g12, 5 clean arms, direct weekly             (5 runs, bounded 1 window)
# g14 = ΑΔΜΗΕ-aligned gate (--delay 14, gap=14, cutoff 09:00 CET) — g12 = χωρις --delay.
# Idempotent (SKIP αν υπαρχει το JSON), ΕΝΑ conda process, συνεχιζει σε FAILED run.
# Τρεξιμο (detached): bash scripts/load_g14_direct_queue.sh > logs/load_g14_direct_queue.log 2>&1

cd "C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter" || exit 1

PY="conda run -n epf --no-capture-output python -X utf8"
mkdir -p runs/load_lear runs/load_mlp runs/load_lstm runs/load_direct

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

echo "############ G14+DIRECT QUEUE (4 stages) START [$(stamp)] ############"
echo "GIT: $(git rev-parse HEAD) | DIRTY: $(git status --porcelain src/ | tr '\n' ' ')"

echo ""; echo "######## BLOCK 0 — PREFLIGHT+POISON [$(stamp)] ########"
if ! $PY .claude/skills/energy-forecast/scripts/preflight_check.py --poison; then
    echo "!!!!! PREFLIGHT/POISON FAIL — ABORT"; exit 1
fi
echo "######## BLOCK 0 PASS ########"

echo ""; echo "######## STAGE 1/4 — LEAR g14 [$(stamp)] ########"
lear_win() {  # lear_win <label> <ts> <te>
  local C=(--algo lear --task load --market dam --strategy recursive --gate strict --delay 14 --retrain weekly --test_start "$2" --test_end "$3")
  mf "runs/load_lear/$1_lear_recw_g14_base.json"  "${C[@]}" --features "calendar,lags,roll"
  mf "runs/load_lear/$1_lear_recw_g14_dense.json" "${C[@]}" --features "calendar,lags,roll,dense"
}
lear_win "summer" "$SU_TS" "$SU_TE"
lear_win "octnov" "$ON_TS" "$ON_TE"
lear_win "q1"     "$Q1_TS" "$Q1_TE"
echo "######## STAGE 1/4 DONE [$(stamp)] ########"

echo ""; echo "######## STAGE 2/4 — MLP g14 [$(stamp)] ########"
mlp_win() {
  local C=(--algo mlp --task load --market dam --strategy recursive --gate strict --delay 14 --retrain weekly --test_start "$2" --test_end "$3")
  mf "runs/load_mlp/$1_mlp_recw_g14_base.json"  "${C[@]}" --features "calendar,lags,roll"
  mf "runs/load_mlp/$1_mlp_recw_g14_dense.json" "${C[@]}" --features "calendar,lags,roll,dense"
}
mlp_win "summer" "$SU_TS" "$SU_TE"
mlp_win "octnov" "$ON_TS" "$ON_TE"
mlp_win "q1"     "$Q1_TS" "$Q1_TE"
echo "######## STAGE 2/4 DONE [$(stamp)] ########"

echo ""; echo "######## STAGE 3/4 — LSTM g14 (static, οπως τα g12 LSTM runs) [$(stamp)] ########"
lstm_win() {  # lstm_win <label> <train_end> <ts> <te>
  local C=(--algo lstm --task load --market dam --strategy recursive --gate strict --delay 14 --retrain static --train_end "$2" --test_start "$3" --test_end "$4")
  mf "runs/load_lstm/$1_lstm_recstatic_g14_base.json"   "${C[@]}" --features "calendar,lags,roll"
  mf "runs/load_lstm/$1_lstm_recstatic_g14_loadfc.json" "${C[@]}" --features "calendar,loadfc"
}
lstm_win "summer" "2025-05-31 23:00" "$SU_TS" "$SU_TE"
lstm_win "octnov" "2025-08-31 23:00" "$ON_TS" "$ON_TE"
lstm_win "q1"     "2025-11-30 23:00" "$Q1_TS" "$Q1_TE"
echo "######## STAGE 3/4 DONE [$(stamp)] ########"

echo ""; echo "######## STAGE 4/4 — direct LGBM summer g12, 5 clean arms [$(stamp)] ########"
dir_arm() {  # dir_arm <slug> <features>
  mf "runs/load_direct/summer_lgbm_dirw_g12_$1.json" \
     --algo lgbm --task load --market dam --strategy direct --gate strict \
     --retrain weekly --test_start "$SU_TS" --test_end "$SU_TE" --features "$2"
}
dir_arm "base"    "calendar,lags,roll"
dir_arm "dense"   "calendar,lags,roll,dense"
dir_arm "genlags" "calendar,lags,roll,genlags"
dir_arm "noroll"  "calendar,lags"
dir_arm "loadfc"  "calendar,lags,roll,loadfc"
echo "######## STAGE 4/4 DONE [$(stamp)] ########"

echo ""; echo "############ G14+DIRECT QUEUE COMPLETE [$(stamp)] ############"
echo "Harvest: python scripts/build_run_ledger.py + synthesize-ablation (baseline ανα gate/strategy ΧΩΡΙΣΤΑ —"
echo "  ΠΟΤΕ συγκριση g14 vs g12 η direct vs recursive αναμεικτα· baseline = το αντιστοιχο base arm)."
