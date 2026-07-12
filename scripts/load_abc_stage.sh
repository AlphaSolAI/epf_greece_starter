#!/usr/bin/env bash
# LOAD — Στάδιο ABC (2026-07-12, επιλογή χρήστη «Α Β Γ όσο πιο γρήγορα γίνεται»)
# ------------------------------------------------------------------------------
# Α (weekly confirm του seas, ABLATION §7.16 PENDING):
#   A1-A3  densemvseas (calendar,lags,roll,dense,meteo_vintage,seas) weekly rec g12,
#          3 windows → runs/load_contest/ (contest arm, baseline: densemv Batch 3
#          q1=216.53 / summer=250.45 / octnov=128.46)
#   A4     summer densemvseas weekly g12 + --train_start 2023-01-01 (trim probe υπό
#          weekly· σύγκριση με A2) → runs/feat_seas/
# Β (direct static probe, 1 window κατά completeness map):
#   B1     summer direct static densemvseas g12 → runs/feat_seas/
#          (σύγκριση με recursive static densemvseas = 271.10, runs/feat_seas/)
# Γ τρέχει ξεχωριστά (scripts/apply_bias_correction.py — κανένα training).
#
# ΟΥΡΑ: Block W περιμένει να αδειάσει ο ΕΝΑΣ conda slot (τρέχει LSTM verify 19:11).
# Τρέξιμο (detached): bash scripts/load_abc_stage.sh > logs/load_abc_stage.log 2>&1

cd "C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter" || exit 1

PY="conda run -n epf --no-capture-output python -X utf8"
OUTC="runs/load_contest"
OUTS="runs/feat_seas"
mkdir -p "$OUTC" "$OUTS"

Q1_TS="2025-12-01 00:00";  Q1_TE="2026-02-28 23:00"
SU_TS="2025-06-01 00:00";  SU_TE="2025-08-31 23:00"
ON_TS="2025-10-01 00:00";  ON_TE="2025-11-30 23:00"
FEATS="calendar,lags,roll,dense,meteo_vintage,seas"

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

echo "############ LOAD ABC STAGE START [$(stamp)] ############"
echo "GIT: $(git rev-parse HEAD) | DIRTY src: $(git status --porcelain src/ | tr '\n' ' ')"

echo ""
echo "######## BLOCK W — ΑΝΑΜΟΝΗ ΟΥΡΑΣ (ένας conda τη φορά) [$(stamp)] ########"
waited=0
while tasklist //FI "IMAGENAME eq python.exe" 2>/dev/null | grep -q "python.exe"; do
    if [ "$waited" -ge 240 ]; then
        echo "!!!!! [$(stamp)] python.exe ακόμα ενεργό μετά από 4h αναμονής — ABORT (δεν παραλληλίζω)"
        exit 1
    fi
    sleep 60
    waited=$((waited+1))
done
echo "-- [$(stamp)] Ουρά άδεια μετά από ${waited} λεπτά — προχωράω."

echo ""
echo "######## BLOCK 0 — PREFLIGHT+POISON [$(stamp)] ########"
if ! $PY .claude/skills/energy-forecast/scripts/preflight_check.py --poison; then
    echo "!!!!! PREFLIGHT/POISON FAIL — ABORT"
    exit 1
fi
echo "######## BLOCK 0 PASS ########"

echo ""
echo "######## BLOCK A — seas weekly confirm (densemvseas, g12, 3 windows + ts2023) [$(stamp)] ########"
mf "$OUTC/q1_lgbm_recw_g12_densemvseas.json" \
   --algo lgbm --task load --market dam --strategy recursive --gate strict \
   --retrain weekly --test_start "$Q1_TS" --test_end "$Q1_TE" --features "$FEATS"
mf "$OUTC/summer_lgbm_recw_g12_densemvseas.json" \
   --algo lgbm --task load --market dam --strategy recursive --gate strict \
   --retrain weekly --test_start "$SU_TS" --test_end "$SU_TE" --features "$FEATS"
mf "$OUTC/octnov_lgbm_recw_g12_densemvseas.json" \
   --algo lgbm --task load --market dam --strategy recursive --gate strict \
   --retrain weekly --test_start "$ON_TS" --test_end "$ON_TE" --features "$FEATS"
mf "$OUTS/summer_lgbm_recw_g12_densemvseas_ts2023.json" \
   --algo lgbm --task load --market dam --strategy recursive --gate strict \
   --retrain weekly --train_start "2023-01-01 00:00" \
   --test_start "$SU_TS" --test_end "$SU_TE" --features "$FEATS"

echo ""
echo "######## BLOCK B — direct static probe (summer, densemvseas, g12) [$(stamp)] ########"
mf "$OUTS/summer_lgbm_dirstatic_g12_densemvseas.json" \
   --algo lgbm --task load --market dam --strategy direct --gate strict \
   --retrain static --test_start "$SU_TS" --test_end "$SU_TE" --features "$FEATS"

echo ""
echo "############ LOAD ABC STAGE END [$(stamp)] ############"
echo "Harvest: MAE από τα 5 JSONs + scripts/apply_bias_correction.py στα νέα weekly."
touch logs/load_abc_stage.done
