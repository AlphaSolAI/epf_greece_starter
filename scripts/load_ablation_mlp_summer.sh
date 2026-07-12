#!/usr/bin/env bash
# LOAD ABLATION — MLP (remaining models, ΟΧΙ tree-based) — STAGE MLP-1 SCREEN
# ------------------------------------------------------------------------------
# Goal χρήστη (2026-07-12): σειριακά overnight ablation για ΠΛΗΡΟΤΗΤΑ, ΚΑΙ τα non-tree
# μοντέλα (MLP/LEAR/LSTM). ΟΧΙ ενα 10ωρο bash — μικρα idempotent σταδια με decision points.
# Διορθωση χρηστη: το combo (dense/SS/summer-feature) μπαινει ως ARM προς ΣΥΓΚΡΙΣΗ —
# κρατιεται ΜΟΝΟ αν κερδισει (§2). summer feature = seas (seasonal_trend, wired διπλα).
#
# ΣΤΑΔΙΟ 1 = SCREEN: summer/g12 μονο, 5 arms σε ΑΥΞΟΥΣΑ σειρα κοστους. Αν το MLP ειναι
# αργο, σταματα μετα τα 3 non-SS (τα JSON μενουν, idempotent). Verdict θελει 2ο window
# (octnov/q1) = ΞΕΧΩΡΙΣΤΟ επομενο σταδιο με OK/harvest αναμεσα.
# Baseline συγκρισης: ΕΝΤΟΣ αυτου του batch (ιδιο seed/gate/window/cadence, μονο #features
# αλλαζει). Το #features ΠΡΕΠΕΙ να αλλαζει base->dense->seas (αλλιως VOID, hard-rule #12).
#
# Τρεξιμο (detached): bash scripts/load_ablation_mlp_summer.sh > logs/load_mlp_summer.log 2>&1

cd "C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter" || exit 1

PY="conda run -n epf --no-capture-output python -X utf8"
OUT="runs/load_mlp"
mkdir -p "$OUT"

SU_TS="2025-06-01 00:00";  SU_TE="2025-08-31 23:00"
stamp() { date +"%Y-%m-%d %H:%M:%S"; }

mf() {  # mf <out_json> <extra args...>
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

echo "############ MLP-1 SCREEN START [$(stamp)] ############"
echo "GIT: $(git rev-parse HEAD) | DIRTY: $(git status --porcelain src/ | tr '\n' ' ')"

echo ""; echo "######## BLOCK 0 — PREFLIGHT+POISON [$(stamp)] ########"
if ! $PY .claude/skills/energy-forecast/scripts/preflight_check.py --poison; then
    echo "!!!!! PREFLIGHT/POISON FAIL — ABORT"; exit 1
fi
echo "######## BLOCK 0 PASS ########"

# array (ΟΧΙ string) — τα timestamps εχουν space, το string re-expansion τα σπαει (bug 08:45)
COMMON=(--algo mlp --task load --market dam --strategy recursive --gate strict --retrain weekly --test_start "$SU_TS" --test_end "$SU_TE")

# MLP base arm μετρηθηκε 3620s (~60') @ weekly summer. dense/seas ~ιδιο. Τα SS arms
# (ss_rounds=3 self-retrain ~4h/arm) ΑΦΑΙΡΕΘΗΚΑΝ — θα εσπρωχναν το screen στις ~11h
# (οδηγια χρηστη «οχι 10ωρο bash»). SS value: trees (LGBM/XGB done) + LEAR (φθηνο).
# Το MLP screen testαρει: βοηθαει dense; προσθετει seas; (combo χωρις SS).
mf "$OUT/summer_mlp_recw_g12_base.json"       "${COMMON[@]}" --features "calendar,lags,roll"
mf "$OUT/summer_mlp_recw_g12_dense.json"      "${COMMON[@]}" --features "calendar,lags,roll,dense"
mf "$OUT/summer_mlp_recw_g12_denseseas.json"  "${COMMON[@]}" --features "calendar,lags,roll,dense,seas"

echo ""; echo "############ MLP-1 SCREEN END [$(stamp)] ############"
