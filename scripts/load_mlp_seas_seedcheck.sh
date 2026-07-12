#!/usr/bin/env bash
# TRIAGE (triaging-suspicious-results): MLP summer/g12 dense->dense+seas edeixe Δ=-66MW,
# υπερβαινει το pre-registered T8 orio (>60MW = υποπτο, design.md §2). Μηχανισμος: η προσθηκη
# 3 στηλων αλλαζει το input_dim -> διαφορετικη τυχαια αρχικοποιηση βαρων (οχι ισοδυναμο seed
# state) — το MLP (σε αντιθεση με LGBM/XGB) ειναι στοχαστικο σε dimension change. 2 επιπλεον
# seeds στο ΙΔΙΟ arm (dense+seas) + 1 επιπλεον seed στο dense (baseline variance) για να
# φανει αν το -66 ειναι σταθερο σημα η θορυβος αρχικοποιησης.
# Τρεξιμο (detached): bash scripts/load_mlp_seas_seedcheck.sh > logs/load_mlp_seedcheck.log 2>&1

cd "C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter" || exit 1

PY="conda run -n epf --no-capture-output python -X utf8"
OUT="runs/load_mlp"
mkdir -p "$OUT"

SU_TS="2025-06-01 00:00";  SU_TE="2025-08-31 23:00"
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

echo "############ MLP SEED-VARIANCE CHECK START [$(stamp)] ############"
COMMON=(--algo mlp --task load --market dam --strategy recursive --gate strict --retrain weekly --test_start "$SU_TS" --test_end "$SU_TE")

for seed in 7 123; do
  mf "$OUT/summer_mlp_recw_g12_denseseas_seed${seed}.json" "${COMMON[@]}" --seed "$seed" --features "calendar,lags,roll,dense,seas"
done
mf "$OUT/summer_mlp_recw_g12_dense_seed7.json" "${COMMON[@]}" --seed 7 --features "calendar,lags,roll,dense"

echo ""; echo "############ MLP SEED-VARIANCE CHECK END [$(stamp)] ############"
