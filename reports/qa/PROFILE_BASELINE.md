# Direct smoke — Profiling baseline (QA optimize σκέλος)

**Ημερομηνία:** 2026-07-08 · **Skill:** `optimizing-training-runs` · **Spec §4α / plan Task 9**
· Μηχάνημα: local (miniconda `epf`). **ΧΩΡΙΣ αλλαγές κώδικα** — μόνο baseline καταγραφή.

## Smoke config (σταθερό — για συγκρίσιμες μετρήσεις)

```
--algo lgbm --task price --market dam --strategy direct --gate strict --retrain static
--features "default,dense" --seed 42 --test_start "2026-02-01 00:00" --test_end "2026-02-07 23:00"
```
- Test window: 1 εβδομάδα (168 ώρες), static (1 fit block/model), seed 42 = anchor seed.
- Result: **MAE=16.8414 €/MWh** (RMSE=23.5252, sMAPE=21.4831%, n=168) — LGBM-direct-dam.
- Wall-clock: **~105.6s / run** (baseline & rerun σχεδόν ταυτόσημα).

> Σημ.: αυτό ΔΕΝ είναι το headline anchor (16.10 = recursive/default/Q1). Είναι direct/dense/
> 1-week Feb — δικό του σταθερό σημείο για perf συγκρίσεις, ΟΧΙ για claims.

## Equivalence acceptance (σενάριο 5)

| Case | Εντολή | Αποτέλεσμα | Exit |
|---|---|---|---|
| 5α PASS | `compare_runs base vs rerun` | **EQUIVALENT** (bit-exact, ακόμα και tol=1e-9) | 0 |
| 5β FAIL | `compare_runs base vs tampered` (+0.5 στο pred[0]) | **NOT EQUIVALENT** — `PRED-DIFF max|diff|=5.0e-01 @ index 0` | 1 |

Ο direct/static/seed-42 είναι πλήρως ντετερμινιστικός → το equivalence gate είναι αξιόπιστο
για μελλοντικές perf αλλαγές (καμία ανοχή δεν χρειάστηκε).

## Top hotspots (cProfile, cumulative)

| Function | cumtime | % | Σημείωση |
|---|---|---|---|
| `master_forecast.run_forecast` | 105.6s | 100% | όλο το run |
| `master_forecast.fit_direct` (:212) | **81.6s** | **76%** | το ακριβό path |
| `sklearn.multioutput.MultiOutputRegressor.fit` | 78.3s | 73% | **τα 24 μοντέλα/ώρα** |
| `joblib.parallel` (workers) | ~101s cum | — | το `time.sleep` 100s = main-thread poll των workers |

**Καΐάτ cProfile+joblib:** το CPU των worker processes εμφανίζεται ως `time.sleep` στο main
thread (5396 calls, 100.4s). Το macro picture είναι σαφές (fit_direct κυριαρχεί), αλλά για
βαθύτερο profiling ΜΕΣΑ στο fit χρειάζεται είτε single-model fit είτε sequential joblib backend.
Profile file: `reports/qa/profile_direct_smoke.prof` (regen από το §3 του skill).

## Πρώτες υποψίες hotspot (ΚΑΤΑΓΡΑΦΗ ΜΟΝΟ — καμία αλλαγή σε αυτό το task)

1. **MultiOutputRegressor = 24 ανεξάρτητα LGBM fits.** Πρώτη ιδέα (μελλοντικά, με equivalence
   proof): κοινό feature-matrix build μία φορά αντί ανά output· έλεγχος `n_jobs` στο
   MultiOutputRegressor vs στο LGBM (double-parallelism μπορεί να υπο-αποδίδει).
2. **joblib backend**: αξίζει να μετρηθεί sequential vs threading vs loky για n=24 μικρά μοντέλα
   (per-worker startup overhead μπορεί να τρώει το κέρδος σε μικρό smoke).

Κάθε τέτοια αλλαγή = ΠΛΗΡΗΣ ροή: `compare_runs` PASS + anchor + (αν πυρήνας) CORE-DIFF+poisoning.
