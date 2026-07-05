# CONFORMAL / PROBABILISTIC LAYER — REPORT (Β4)

> Template προετοιμασμένο 2026-07-05 (πριν τα νούμερα) — γεμίζεται μόλις τρέξει το πλήρες
> Β4 (Block F του overnight είναι μόνο smoke test, όχι το πλήρες πείραμα).
> Κανόνες πηγής: `.claude/skills/energy-forecast/SKILL.md §Conformal` ·
> `ABLATION_PLAN.md §8.2` · κριτήρια αποδοχής αναλογικά με `last.md §2` (Α3/Α4).
> Μέθοδος υλοποιημένη σε `src/conformal.py` (δύο subcommands: `split-conformal`,
> `quantile-lgbm`) — καμία νέα βιβλιοθήκη, numpy quantiles πάνω σε residuals.

## 0. Ερευνητική ερώτηση

Πόσο καλά καλυμμένα (calibrated) και πόσο αιχμηρά (sharp) είναι τα p10/p50/p90 intervals
γύρω από το point forecast, όταν η βαθμονόμηση είναι αυστηρά αιτιατή (rolling, ποτέ από
το test); Συγκρίνεται σε ≥2 point μοντέλα (model-agnostic split-conformal) ΚΑΙ έναντι ενός
baseline (quantile-LGBM εκπαιδευμένο απευθείας με `objective=quantile`).

## 1. Setup (γέμισε πριν τρέξεις)

| | |
|---|---|
| Point μοντέλα υπό αξιολόγηση | LGBM weekly `--features default` (+ ______ , π.χ. XGB weekly) |
| Πηγή point forecast JSON | `master_forecast --out_json ...` (πρέπει να καλύπτει ≥8 εβδ. ΠΡΙΝ το πρώτο eval window, αλλιώς calibration window κόβεται) |
| Calibration window | rolling, causal, per hour-of-day, `min_weeks=4` .. `max_weeks=8` (defaults conformal.py) |
| Alphas | 0.1 / 0.5 / 0.9 |
| Eval windows | `q1_2026`: 2025-12-01 00:00 .. 2026-02-28 23:00 · `march_2026`: 2026-03-01 00:00 .. 2026-03-19 23:00 |
| Baseline | quantile-LGBM, ίδιο config (recursive, weekly, `--features default`) |
| Gate | strict (AEL crosslag_mode=freeze) |

**Εντολές αναπαραγωγής (αντίγραψε ΑΚΡΙΒΩΣ πριν βάλεις νούμερα):**

```bash
# 1) point forecast (πηγή για split-conformal) — extended JSON, ≥8 εβδ. πριν το Q1 window
conda run -n epf --no-capture-output python -X utf8 -m src.master_forecast \
  --algo lgbm --task price --market dam --strategy recursive --gate strict \
  --retrain weekly --features default \
  --test_start "2025-10-01 00:00" --test_end "2026-03-19 23:00" \
  --out_json runs/conformal/point_lgbm_weekly_default.json

# 2) split-conformal πάνω στο (1) — μοντέλο-άσχετο
conda run -n epf --no-capture-output python -X utf8 -m src.conformal split-conformal \
  --forecast_json runs/conformal/point_lgbm_weekly_default.json \
  --out_json runs/conformal/split_conformal_lgbm.json

# 3) επανάληψη (1)+(2) για δεύτερο μοντέλο, π.χ. XGB weekly
#    --algo xgb ... --out_json runs/conformal/point_xgb_weekly_default.json
#    --forecast_json runs/conformal/point_xgb_weekly_default.json --out_json runs/conformal/split_conformal_xgb.json

# 4) baseline quantile-LGBM (εκπαιδεύεται δικό του, δεν διαβάζει JSON)
conda run -n epf --no-capture-output python -X utf8 -m src.conformal quantile-lgbm \
  --test_start "2025-12-01 00:00" --test_end "2026-03-19 23:00" \
  --retrain weekly --features default --gate strict \
  --out_json runs/conformal/quantile_lgbm_baseline.json
```

⚠️ **ΕΝΑ conda process τη φορά** — δεν τρέχει ΤΙΠΟΤΑ από τα παραπάνω απόψε όσο τρέχει το
overnight batch (`scripts/overnight_20260705.sh`). Πρώτα `overnight_summarize.py` +
FAILED-check, ΜΕΤΑ αυτό.

## 2. Αποτελέσματα — split-conformal (model-agnostic)

Πηγή στηλών: `results["q1_2026"]` / `results["march_2026"]` του out_json κάθε μοντέλου
(κλειδιά: `n`, `avg_pinball`, `pinball_p10/p50/p90`, `mae_p50`, `coverage_80pct_nominal`,
`p10_empirical_pct`, `p90_empirical_pct`). Sharpness (μέσο πλάτος band, `mean(p90−p10)`)
**ΔΕΝ υπολογίζεται ήδη μέσα στο conformal.py** — υπολόγισέ το post-hoc από τα
`p10`/`p90` arrays του out_json (δεν χρειάζεται re-run):
`np.mean(np.array(d["p90"]) - np.array(d["p10"]))` ανά window (φιλτράρισε πρώτα με τις
ημερομηνίες του window, ίδιο μοτίβο με `evaluate_window`).

### 2.1 Μοντέλο Α — LGBM weekly (`--features default`)

| window | n | avg_pinball | MAE(p50) | coverage 80% (στόχος ~80%) | P10 emp% (στόχος 10%) | P90 emp% (στόχος 90%) | sharpness (mean p90−p10) |
|---|---|---|---|---|---|---|---|
| Q1 2026    | | | | | | | |
| Μάρτιος 26 | | | | | | | |

### 2.2 Μοντέλο Β — ______ weekly (π.χ. XGB, `--features default`)

| window | n | avg_pinball | MAE(p50) | coverage 80% | P10 emp% | P90 emp% | sharpness |
|---|---|---|---|---|---|---|---|
| Q1 2026    | | | | | | | |
| Μάρτιος 26 | | | | | | | |

## 3. Αποτελέσματα — quantile-LGBM baseline (direct quantile training)

| window | n | avg_pinball | MAE(p50) | coverage 80% | P10 emp% | P90 emp% | sharpness |
|---|---|---|---|---|---|---|---|
| Q1 2026    | | | | | | | |
| Μάρτιος 26 | | | | | | | |

## 4. Σύγκριση split-conformal vs quantile-LGBM baseline

- Ποιο έχει χαμηλότερο avg_pinball σε ΚΑΙ τα δύο windows (Α3-style: συμφωνία ≥2 συνθηκών);
- Ποιο έχει coverage πιο κοντά στο ονομαστικό 80% και στα δύο windows;
- Ποιο είναι πιο sharp (μικρότερο band) ΣΤΟ ΙΔΙΟ επίπεδο coverage — coverage χωρίς
  sharpness = κενό claim (SKILL.md §Conformal κανόνας 4).
- Συμφωνούν LGBM και το δεύτερο μοντέλο (§2.1 vs §2.2) ως προς ποιά μέθοδος είναι καλύτερη;
  Αν όχι → algo-dependent, ανοιχτό σημείο (ίδιο πρότυπο με ABLATION §7 PENDING items).

## 5. Verdict (γέμισε ΜΟΝΟ μετά τα νούμερα, κριτήρια §2/last.md αναλογικά)

- [ ] Coverage εντός αποδεκτού εύρους (π.χ. 75-85% για ονομαστικό 80%) και στα δύο windows
      και στα δύο μοντέλα.
- [ ] Split-conformal ΔΕΝ χειρότερο από quantile-LGBM baseline σε avg_pinball με σημαντικό
      περιθώριο (ή: ρητή καταγραφή trade-off αν είναι χειρότερο αλλά πιο sharp/simple).
- [ ] Συμπέρασμα κρατάει σε ≥2 point μοντέλα (model-agnostic claim όχι μόνο για LGBM).
- [ ] Καμία μέτρηση coverage χωρίς συνοδευτικό sharpness.

**Συμπέρασμα (1 παράγραφος):** ___________________________________________________

## 6. Traceability (Α5)

| artifact | path |
|---|---|
| point forecast JSON (μοντέλο Α) | `runs/conformal/point_lgbm_weekly_default.json` |
| point forecast JSON (μοντέλο Β) | `runs/conformal/point_____weekly_default.json` |
| split-conformal out (Α) | `runs/conformal/split_conformal_lgbm.json` |
| split-conformal out (Β) | `runs/conformal/split_conformal_____.json` |
| quantile-lgbm baseline out | `runs/conformal/quantile_lgbm_baseline.json` |
| αυτό το report | `reports/CONFORMAL_REPORT_TEMPLATE.md` → μετονόμασε σε `CONFORMAL_REPORT_20260705.md` (ή ημερομηνία πλήρους run) όταν γεμίσει |

## 7. Ανοιχτά / μελλοντικά (από SKILL.md §Conformal σημ. 6)

- Adaptive conformal (ACI) για regime shifts — μελλοντική ακαδημαϊκή επέκταση, όχι Β4.
- Sharpness δεν είναι πεδίο του `evaluate_window()` σήμερα — αν χρησιμοποιείται συχνά,
  σκέψου να προστεθεί μόνιμα στο `src/conformal.py` (μικρό, μη-breaking change) σε επόμενο
  session — ΟΧΙ απόψε (κώδικας ίδιου αρχείου μπορεί να τρέξει από το overnight Block F).
