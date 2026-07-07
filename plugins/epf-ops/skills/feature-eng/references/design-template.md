# Feature Design Doc — <name>

> Συμπληρώνεται ΟΛΟΚΛΗΡΟ πριν γραφτεί κώδικας feature (TDD pre-registration).
> Αντίγραψέ το σε `docs/features/<name>/design.md`. Ό,τι δεν προβλέφθηκε εδώ
> εκ των προτέρων, δεν μετράει ως «επιβεβαίωση» εκ των υστέρων.

## 1. Ταυτότητα

- **Feature/πηγή:** <τι είναι, ποιος το δημοσιεύει>
- **Ομάδα (flag) στο feature_availability.py:** `<group>` (νέα ή υπάρχουσα;)
- **Στήλες parquet:** `<col1>, <col2>` — σε ποιο task parquet πρέπει να υπάρχουν
  (price: `hourly.parquet` / load: `hourly_load.parquet` / και τα δύο);
- **Availability rule (πρόχειρο):** <fc day-ahead-known; actual μόνο ως lags με reporting delay;>
- **T0 — Κάλυψη δεδομένων ανά window (read-only, ΠΡΙΝ από όλα):**

| Window | Ώρες παρούσες | Αναμενόμενες | Κάλυψη |
|---|---|---|---|
| Q1 (Δεκ-Φεβ) | <> | 2160 | <>% |
| Καλοκαίρι (Ιουν-Αυγ) | <> | 2208 | <>% |
| Μάρτιος | <> | ~456 | <>% |

  <2 ΠΛΗΡΗ windows → **BLOCKED** εδώ (T7 δομικά αδύνατο)· backfill πριν από οτιδήποτε άλλο.

## 2. Μηχανισμός & pre-registered προσδοκίες

- **Γιατί να βοηθάει (φυσικός/αγοραίος μηχανισμός):** <1-3 προτάσεις>
- **Αναμενόμενο πρόσημο ΔMAE:** <βελτίωση/επιδείνωση> · **αναμενόμενο μέγεθος:** <π.χ. 0.15-0.5>
  (αν δεν δικαιολογείται >0.15, το πείραμα δεν αξίζει slot στην ουρά)
- **Πού αναμένεται να δρα:** στρατηγική <recursive/direct/και οι δύο> ·
  εποχή <χειμώνας/καλοκαίρι/όλο τον χρόνο> · task <price/load>
- **Αναμενόμενο lagscan peak:** k = <h> επειδή <θεωρία>
- **Αναμενόμενο hour-profile:** <σχήμα, π.χ. solar peak ~12:00>
- **Αναμενόμενο FI rank ομάδας:** <π.χ. κάτω από lags/calendar, πάνω από fuel>

## 3. Acceptance tests (κλειδώνουν ΤΩΡΑ — T0-T8)

| Test | Τι ελέγχει | Κριτήριο PASS | Στάδιο |
|---|---|---|---|
| T0 | Data coverage | ≥2 ΠΛΗΡΗ ανεξάρτητα windows διαθέσιμα (πίνακας §1) | 0 |
| T1 | Gate timing | **Πρωτογενής πηγή** (market rules/publication calendar): δημοσίευση ΠΡΙΝ το 12:00 CET D-1 (ή νόμιμη lagged εκδοχή με ρητό delay). Οικονομική λογική ΔΕΝ αρκεί. | 2 |
| T2 | Lagscan | Peak στο προβλεπόμενο k του §2 · ΟΧΙ «βολικό» k · |corr|<0.85 | 2 |
| T3 | Hour-profile | Σχήμα = προβλεπόμενο του §2 (shift ⇒ fetcher bug) | 2 |
| T4 | Non-VOID arm | `#features` ΑΛΛΑΖΕΙ baseline↔spec στο πρώτο log · στήλη υπάρχει στο parquet ΚΑΘΕ target task | 3-4 |
| T5 | Poisoning | `preflight_check.py --poison` PASS μετά την αλλαγή feature_availability.py | 3 |
| T6 | Reproducibility | Control run· anchor ±0.05 | 3 |
| T7 | §2 gate | |ΔMAE|>0.15 ΚΑΙ ίδιο πρόσημο σε ≥2 ΑΝΕΞΑΡΤΗΤΑ windows | 5 |
| T8 | Too-good-to-be-true | Δ όχι πολύ μεγαλύτερο του pre-registered· FI rank εύλογο· αλλιώς targeted poisoning πριν από claim | 5-6 |

## 4. Batch matrix (στάδιο 4)

- Windows: Q1 (2025-12-01→2026-02-28) + καλοκαίρι 2025 <+ Μάρτιος 2026 ως 3ο>
- Algos: LGBM + XGB · strategy: <recursive/+direct> · retrain: static · seed: 42
- Specs: `default` vs `default,<group>` <ή `all` vs `all,-<group>`>
- Outputs: `runs/feat_<name>/<window>_<algo>_static_<spec>.json` + `results/feat_<name>.csv`
- Εκτίμηση διάρκειας/ουράς: <λεπτά/ώρες — detached αν >2-3 λεπτά>

## 5. Γνωστοί κίνδυνοι για ΑΥΤΟ το feature

- <ίδιος μηχανισμός/auction με το target; (xborder lesson)>
- <εποχιακό flip; (resfc lesson)>
- <TZ πηγής vs CET/CEST-naive frame; (TZFIX lesson)>
- <κενή στήλη σε ένα από τα δύο task parquets; (loadfc lesson)>
- <ορφανός loader/λάθος raw path — ποιος ΠΡΑΓΜΑΤΙΚΑ καταναλώνει το module; (henex lesson)>
- <structural breaks που τέμνουν τα windows: SDAC 15-min MTU 2025-10-01 · lignite exit 2026 ·
  άνοδος negative-price hours 2026 — επηρεάζουν το εύρημα;>

## 6. Audit evidence (συμπληρώνεται στο στάδιο 2)

- T1 γραπτή απάντηση: <εδώ ή link>
- T2 lagscan output: <path/απόσπασμα>
- T3 hour-profile: <παρατήρηση>
- Verdict σταδίου 2: ΠΡΟΧΩΡΑ / ΑΠΟΡΡΙΦΘΗΚΕ (γράφτηκε στο ABLATION_PLAN §7: <ναι/όχι>)
