# Deploy Checklist — <name>

> Συμπληρώνεται στο στάδιο 6 του feature-eng, ΜΕΤΑ το verdict του synthesize-ablation.
> Αντίγραψέ το σε `docs/features/<name>/deploy.md` και τρέξε τον validator
> (`scripts/validate_deploy_checklist.py`). Κανένα `<...>` placeholder δεν μένει.
> ΠΟΤΕ suspended νούμερα (15.17, 16.10, 19.17, 14.43, 15.02) ως τρέχοντα αποτελέσματα.

## 1. LEAKAGE-FREE PROOF

- Gate timing (T1): <πότε δημοσιεύεται vs 12:00 CET D-1 — μία πρόταση>
- Lagscan (T2): <path/απόσπασμα, peak k, |corr|>
- Hour-profile (T3): <παρατήρηση>
- Poisoning μετά το feature_availability.py edit (T5): <log path, PASS>
- crosslag_mode: freeze (default) — επιβεβαιωμένο στα run JSONs: <ναι>
- Πληρότητα ανά task parquet (T4): <price: ναι/όχι/ν.α. · load: ναι/όχι/ν.α.>

## 2. FEATURE IMPORTANCE

- FI run: <εντολή + path (feature_importance CSV / figure)>
- Rank νέας ομάδας: <παρατηρούμενο> vs pre-registered: <από design.md §2>
- Too-good flag (T8): <όχι — ή τι targeted poisoning έγινε και πού>

## 3. EXPECTED vs ACTUAL

| Pre-registered (design.md §2) | Μετρημένο |
|---|---|
| Πρόσημο: <> | <> |
| Μέγεθος: <> | ΔMAE = <ανά window/algo> |
| Πού δρα (στρατηγική/εποχή): <> | <> |

- Πίνακας ΔMAE: `results/feat_<name>.csv` · σύνθεση: <synthesize output>
- Αποκλίσεις από τις προσδοκίες & ερμηνεία: <->

## 4. KPI & VERDICT

- §2 pre-gate (T7): <|Δ|, πρόσημο, ποια ανεξάρτητα windows>
- validity-reviewer verdict: <ACCEPT/PENDING/REJECT + ημερομηνία>
- Απόφαση default set: <μπαίνει/δεν μπαίνει/flag-μόνο> — ΑΝΘΡΩΠΙΝΗ έγκριση: <ποιος/πότε>
- Headline impact: <καμία / υποψήφιο — τότε ≥3 seeds + 2ο window: status>
- KPI αναφοράς: MAE baseline <x> → με feature <y> (€/MWh ή MW, window: <>)

## 5. DANGERS & ROLLBACK

- Κίνδυνοι που παραμένουν: <εποχιακό flip; regime dependence; πηγή μπορεί να αλλάξει schedule;>
- Rollback: αφαίρεση από default spec (η ομάδα μένει διαθέσιμη ως flag) — εντολή/spec: <>
- Monitoring: <τι θα έδειχνε ότι το feature «σάπισε» (π.χ. rolling ΔMAE αλλάζει πρόσημο)>

## 6. TRACE

- Design doc: `docs/features/<name>/design.md`
- Runs: `runs/feat_<name>/` · CSV: `results/feat_<name>.csv` · logs: <>
- Εντολή αναπαραγωγής (πλήρης): `conda run -n epf --no-capture-output python -X utf8 -m src.<...>`
- ABLATION_PLAN εγγραφή: §<5.x/7> · last.md ενημερώθηκε: <ναι>
