---
name: optimizing-training-runs
description: Επιτάχυνση των training runs του epf_greece_starter (ιδίως direct ablations — 24 μοντέλα/run × refits) με profiling και ΑΠΟΔΕΙΞΗ αριθμητικής ισοδυναμίας. Χρησιμοποίησέ το όταν ζητείται «πιο γρήγορα», όταν ένα batch αργεί, ή πριν από perf αλλαγή σε οποιοδήποτε src αρχείο. ΠΡΟΣΟΧΗ: αν κάτι έγινε ΞΑΦΝΙΚΑ πολύ γρηγορότερο χωρίς εξήγηση → triaging-suspicious-results, όχι αυτό.
---

# Optimize training runs — με equivalence gate

## 0. Σιδερένιος κανόνας

«Πιο γρήγορο» μετράει ΜΟΝΟ με απόδειξη ίδιων αποτελεσμάτων. Κάθε perf αλλαγή:
1. `scripts/qa/compare_runs.py` PASS σε smoke run πριν/μετά (ίδιες προβλέψεις).
2. Anchor εντός ±0.05 (LGBM default static Q1) αν η αλλαγή αγγίζει training path.
3. Αν αγγίζει leakage-sensitive αρχείο: ΠΛΗΡΗΣ ροή `epf-code-reviewer` CORE-DIFF +
   poisoning. Η ταχύτητα ΔΕΝ αγοράζει παράκαμψη κανόνων.

## 1. Ο βρόχος (μεθοδολογία codspeed-optimize, local)

measure → hotspot → ΜΙΑ στοχευμένη αλλαγή → re-measure → equivalence → deposit.
Ποτέ optimization χωρίς μέτρηση πριν ΚΑΙ μετά. Ποτέ δύο αλλαγές μαζί.

## 2. Smoke-bench config (σταθερό — για συγκρίσιμες μετρήσεις)

Direct (το ακριβό path), 1 εβδομάδα test, static, seed 42 — τελειώνει σε λεπτά:

```bash
conda run -n epf --no-capture-output python -X utf8 -m src.master_forecast --algo lgbm --task price --market dam --strategy direct --gate strict --retrain static --features "default,dense" --seed 42 --test_start "2026-02-01 00:00" --test_end "2026-02-07 23:00" --out_json runs/qa_smoke/direct_smoke_base.json
```

## 3. Profiling (ΘΕΛΕΙ conda — μπαίνει στη ΜΙΑ ουρά· >2-3 min => detached/user-run)

```bash
# cProfile πάνω στο smoke config:
conda run -n epf --no-capture-output python -X utf8 -m cProfile -o reports/qa/profile_direct_smoke.prof -m src.master_forecast --algo lgbm --task price --market dam --strategy direct --gate strict --retrain static --features "default,dense" --seed 42 --test_start "2026-02-01 00:00" --test_end "2026-02-07 23:00" --out_json runs/qa_smoke/direct_smoke_profiled.json

# Ανάλυση (system python — pstats είναι stdlib):
python -X utf8 -c "import pstats;pstats.Stats(r'reports/qa/profile_direct_smoke.prof').sort_stats('cumulative').print_stats(25)"
```

## 4. Γνωστά hotspots (αρχικό runbook — ενημερώνεται με κάθε εύρημα)

| Hotspot | Γιατί κοστίζει | Πρώτη ιδέα (πάντα με equivalence proof) |
|---|---|---|
| direct strategy | 24 ανεξάρτητα μοντέλα ανά run | κοινό feature matrix build μία φορά, slice ανά ώρα |
| weekly retrain | refit ανά εβδομάδα σε expanding window | επαναχρησιμοποίηση αμετάβλητων υπολογισμών μεταξύ refits |
| feature matrix rebuild | αν ξαναχτίζεται ανά ώρα/refit | cache + invalidation στο cutoff |
| LGBM/XGB threading | default n_jobs μπορεί να μην κορεστεί | ρητό n_jobs — ΠΡΟΣΟΧΗ: αλλαγή threading μπορεί να αλλάξει αριθμητική → compare_runs υποχρεωτικό |

## 5. Equivalence + deposit

```bash
python -X utf8 scripts/qa/compare_runs.py --a runs/qa_smoke/direct_smoke_base.json --b runs/qa_smoke/direct_smoke_after.json
```
- PASS → κατέγραψε κέρδος wall-clock σε `reports/qa/PROFILE_BASELINE.md` + νέα γραμμή
  στο §4. FAIL → η αλλαγή απορρίπτεται ή επανασχεδιάζεται· ΔΕΝ υπάρχει «αποδεκτά
  διαφορετικό» αποτέλεσμα για χάρη ταχύτητας.

## 6. CodSpeed — φάση 2 (connector συνδεδεμένος από τον χρήστη)

Micro-benchmarks ΜΟΝΟ σε pure-logic paths χωρίς training data (feature_availability
filtering, dense lag construction, split_utils) μέσω `codspeed:codspeed-setup-harness`
→ CI. Macro timing (πλήρη runs) μένει ΠΑΝΤΑ local με το §3. Ανάλυση cloud runs:
CodSpeed MCP tools (list_runs, query_flamegraph, compare_runs).
