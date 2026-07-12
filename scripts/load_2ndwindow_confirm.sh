#!/usr/bin/env bash
# LOAD ABLATION — 2ο ΑΝΕΞΑΡΤΗΤΟ WINDOW (octnov) confirm για §2 eligibility.
# ΣΤΑΔΙΟ: MLP dense, LEAR dense, LSTM base — τα arms που εδειξαν καθαρο σημα στο
# summer/g12 screen. Στοχος: ιδιο προσημο σε 2 ανεξαρτητα windows (§2 κριτηριο).
# ΦΘΗΝΟ/ΜΙΚΡΟ ΣΚΟΠΙΜΑ (goal χρηστη: οχι 10ωρο bash): octnov ειναι το γρηγοροτερο
# window ιστορικα σε αυτο το repo (μικροτερο εξ ορισμου). 3 arms, ενα-ενα, idempotent.
# Τρεξιμο (detached): bash scripts/load_2ndwindow_confirm.sh > logs/load_2ndwindow.log 2>&1

cd "C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter" || exit 1

PY="conda run -n epf --no-capture-output python -X utf8"
stamp() { date +"%Y-%m-%d %H:%M:%S"; }

ON_TS="2025-10-01 00:00";  ON_TE="2025-11-30 23:00"

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

echo "############ 2ND-WINDOW CONFIRM START [$(stamp)] ############"
echo "GIT: $(git rev-parse HEAD) | DIRTY: $(git status --porcelain src/ | tr '\n' ' ')"

echo ""; echo "######## BLOCK 0 — PREFLIGHT+POISON [$(stamp)] ########"
if ! $PY .claude/skills/energy-forecast/scripts/preflight_check.py --poison; then
    echo "!!!!! PREFLIGHT/POISON FAIL — ABORT"; exit 1
fi
echo "######## BLOCK 0 PASS ########"

mkdir -p runs/load_mlp runs/load_lear runs/load_lstm

# --- MLP: base + dense (octnov) ---
mf "runs/load_mlp/octnov_mlp_recw_g12_base.json"  --algo mlp --task load --market dam --strategy recursive --gate strict --retrain weekly --test_start "$ON_TS" --test_end "$ON_TE" --features "calendar,lags,roll"
mf "runs/load_mlp/octnov_mlp_recw_g12_dense.json" --algo mlp --task load --market dam --strategy recursive --gate strict --retrain weekly --test_start "$ON_TS" --test_end "$ON_TE" --features "calendar,lags,roll,dense"

# --- LEAR: base + dense (octnov) ---
mf "runs/load_lear/octnov_lear_recw_g12_base.json"  --algo lear --task load --market dam --strategy recursive --gate strict --retrain weekly --test_start "$ON_TS" --test_end "$ON_TE" --features "calendar,lags,roll"
mf "runs/load_lear/octnov_lear_recw_g12_dense.json" --algo lear --task load --market dam --strategy recursive --gate strict --retrain weekly --test_start "$ON_TS" --test_end "$ON_TE" --features "calendar,lags,roll,dense"

# --- LSTM: base (octnov, static — ιδιο retrain mode με το summer verify) ---
mf "runs/load_lstm/octnov_lstm_recstatic_g12_base.json" --algo lstm --task load --market dam --strategy recursive --gate strict --retrain static --train_end "2025-08-31 23:00" --test_start "$ON_TS" --test_end "$ON_TE" --features "calendar,lags,roll"

echo ""; echo "############ 2ND-WINDOW CONFIRM END [$(stamp)] ############"
