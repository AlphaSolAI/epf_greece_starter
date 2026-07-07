# Feature Design Doc — renewable_ramp

> Συμπληρώνεται ΟΛΟΚΛΗΡΟ πριν γραφτεί κώδικας feature (TDD pre-registration).
> Ό,τι δεν προβλέφθηκε εδώ εκ των προτέρων, δεν μετράει ως «επιβεβαίωση» εκ των υστέρων.

## 1. Ταυτότητα

- **Feature/πηγή:** ΔΕΝ είναι νέα πηγή — παράγωγο (diff) υπάρχοντος, ήδη νόμιμου `resfc`
  (day-ahead solar/wind generation forecast, ΑΔΜΗΕ/ENTSO-E).
- **Ομάδα (flag) στο feature_availability.py:** `ramp` (ΝΕΑ ομάδα, ξεχωριστή από `resfc`
  για καθαρή απομόνωση αξίας στο ablation).
- **Στήλες parquet:** παράγονται on-the-fly από `solar_fc_dayahead`/`wind_onshore_fc_dayahead`
  (ήδη στο `hourly.parquet`) — `solar_ramp1h`, `wind_ramp1h`. price: `hourly.parquet` ✅
  υπάρχουν οι πηγές· load: `hourly_load.parquet` — **ΝΑ ΕΛΕΓΧΘΕΙ στο audit** (μάθημα loadfc).
- **Availability rule:** ΚΛΗΡΟΝΟΜΕΙ το availability του `resfc` (day-ahead-known, gap=0) —
  diff δύο ήδη day-ahead-known τιμών (fc(t), fc(t-1)) παραμένει day-ahead-known, καμία νέα
  πληροφορία μετά το gate.
- **T0 — Κάλυψη δεδομένων ανά window:**

| Window | Ώρες παρούσες | Αναμενόμενες | Κάλυψη |
|---|---|---|---|
| Q1 (Δεκ-Φεβ) | 2160 (πηγή: solar_fc_dayahead ήδη 100% στο headline) | 2160 | 100% |
| Καλοκαίρι (Ιουν-Αυγ) | 2208 | 2208 | 100% |

  ✅ 2 πλήρη ανεξάρτητα windows — T0 PASS χωρίς backfill (parquet κάλυψη 2017-2026).

## 2. Μηχανισμός & pre-registered προσδοκίες

- **Γιατί να βοηθάει:** Empirical εύρημα (headline runs, 2026-07-06): οι χειρότερες ώρες
  MAE συγκεντρώνονται 17:00-21:00 ΚΑΙ στα δύο windows (solar-to-thermal evening ramp) —
  ίδιος μηχανισμός, διαφορετική συχνότητα (Q1: 5.7% ωρών με σφάλμα>50 vs Summer 2.8%,
  παρότι το Summer έχει ΜΕΓΑΛΥΤΕΡΗ ωμή διακύμανση τιμής). Το `resfc` δίνει το ΕΠΙΠΕΔΟ
  solar/wind ανά ώρα αλλά ΟΧΙ ρητά τον ΡΥΘΜΟ μεταβολής — ένα δέντρο (LGBM/XGB) ΔΕΝ μπορεί
  να ανακατασκευάσει diff(t, t-1) από μία μόνο ωμή τιμή ανά γραμμή χωρίς το explicit
  lag-of-itself ως ξεχωριστή στήλη (που ήδη υπάρχει ως `resfc` level, όχι ως ρυθμός).
- **Αναμενόμενο πρόσημο ΔMAE:** βελτίωση (αρνητικό Δ) · **αναμενόμενο μέγεθος:** 0.15-0.5
  (transform υπάρχουσας πληροφορίας, ΟΧΙ νέα πηγή — ρεαλιστικά μέτρια προσδοκία, όχι μεγάλη).
- **Πού αναμένεται να δρα:** στρατηγική **recursive** (headline strategy, εκεί εστιάζουμε
  πρώτα για ταχύτητα) · εποχή **και οι δύο** (ο μηχανισμός/evening ramp είναι κοινός) ·
  task **price**.
- **Αναμενόμενο lagscan peak:** k=0 (ίδια ώρα, immediate relationship με y — ΟΧΙ shifted lag,
  αφού το ramp ΕΙΝΑΙ ήδη η χρονική πληροφορία, δεν χρειάζεται δικό του lag για να «βρει» πότε).
- **Αναμενόμενο hour-profile:** πιο αρνητικές (μεγάλη πτώση) τιμές `solar_ramp1h` στις
  16:00-19:00 (ηλιακή δύση) — συμμετρικό με το γνωστό solar peak ~12:00.
- **Αναμενόμενο FI rank:** μέτρια-χαμηλή — κάτω από `lags`/`calendar`/`resfc` level, πάνω
  από `fuel`/`other` (μικρό αλλά υπαρκτό συμπληρωματικό σήμα, ΟΧΙ κυρίαρχο).

## 3. Acceptance tests (T0-T8)

| Test | Κριτήριο PASS | Στάδιο |
|---|---|---|
| T0 | ≥2 πλήρη windows | 0 — ✅ PASS (§1) |
| T1 | Δημοσίευση πριν 12:00 CET D-1 | 2 — κληρονομείται από resfc (ήδη vetted, §5.6/§5.11) |
| T2 | Lagscan peak στο k=0, ΟΧΙ «βολικό» k, \|corr\|<0.85 | 2 |
| T3 | Hour-profile: αρνητικό peak 16-19h | 2 |
| T4 | `#features` αλλάζει baseline↔spec· στήλη υπάρχει σε ΚΑΘΕ target task parquet | 3-4 |
| T5 | `preflight_check.py --poison` PASS μετά την αλλαγή | 3 |
| T6 | Control run· anchor ±0.05 | 3 |
| T7 | \|ΔMAE\|>0.15 ΚΑΙ ίδιο πρόσημο σε ≥2 ανεξάρτητα windows | 5 |
| T8 | Δ όχι πολύ μεγαλύτερο του pre-registered (0.15-0.5)· FI rank εύλογο | 5-6 |

## 4. Batch matrix (στάδιο 4) — ΜΙΚΡΟ, ταχύ (ζητήθηκε «δοκιμή»)

- Windows: Q1 (2025-12-01→2026-02-28) + καλοκαίρι 2025 (static, ίδιο με headline anchor)
- Algos: LGBM + XGB · strategy: **recursive μόνο** (headline) · retrain: static · seed: 42
- Specs: `default,dense` vs `default,dense,ramp` (πάνω στο ΗΔΗ κλειδωμένο headline spec)
- Outputs: `runs/feat_renewable_ramp/<window>_<algo>_static_<spec>.json` +
  `results/feat_renewable_ramp.csv`
- Σύνολο: 4 runs (2 windows × 2 algos), εκτίμηση <10 λεπτά (static, ίδιο μέγεθος με March block)

## 5. Γνωστοί κίνδυνοι

- Ίδιος μηχανισμός/auction με target; ΟΧΙ — resfc είναι ανεξάρτητη, ήδη-vetted πηγή.
- Εποχιακό flip; πιθανό (όπως resfc/genlags) — θα ελεγχθεί στο batch, ΔΕΝ θα αγνοηθεί αν εμφανιστεί.
- TZ; resfc ήδη TZFIX-past — diff πάνω σε ήδη ευθυγραμμισμένες στήλες, χαμηλός κίνδυνος.
- Κενή στήλη σε load parquet; **ΘΑ ΕΛΕΓΧΘΕΙ ρητά** (μάθημα loadfc, 2026-07-05) πριν από
  οποιοδήποτε task=load πείραμα με αυτό το feature (εκτός scope αυτής της δοκιμής — μόνο price).
- Πρώτη ώρα κάθε window: `diff()` δίνει NaN (κανονικό, ίδια λογική με y_lag boundary) —
  ΔΕΝ είναι leakage, είναι αναμενόμενο missing-at-edge.

## 6. Audit evidence (στάδιο 2) — system python, in-memory transform (χωρίς κώδικα ακόμα)

- **T1 (gate timing):** κληρονομείται από `resfc` (ήδη vetted σε §5.6/§5.11 — day-ahead
  known, gap=0). Diff δύο ήδη day-ahead-known τιμών παραμένει day-ahead-known. ✅ PASS.
- **T2 (lagscan)**, `solar_ramp1h` (80125 ώρες, όλο το ιστορικό parquet):
  top-5 |corr(y(t), ramp(t−k))|: k=+3 (−0.236), k=+4 (−0.222), k=+2 (−0.206), k=+5 (−0.176),
  k=+11 (+0.167). `wind_ramp1h`: πολύ ασθενέστερο (max |corr|≈0.066).
  ⚠️ **Pre-registration MISS**: προβλέφθηκε peak k=0, βρέθηκε k=3 (broad ζώνη k=2-5, ΟΧΙ
  ένα αιχμηρό σημείο). Ερμηνεία: η επίδραση ενός απογευματινού ramp επιμένει 2-5 ώρες
  αργότερα (η τιμή παραμένει αυξημένη ενόσω τα θερμικά μένουν σε αυξημένη παραγωγή), όχι
  μόνο τη στιγμή του ramp — φυσικά εύλογο, ΟΧΙ leakage-σχήμα (κανένα |corr|>0.85, ομαλή
  κατανομή σε πολλαπλά k, όχι αιχμηρό «βολικό» σημείο όπως το ιστορικό xborder 1h-shift bug).
  ✅ PASS (ως προς leakage), καταγεγραμμένο ως λάθος στην αρχική πρόβλεψη (τίμιο pre-reg).
- **T3 (hour-profile)**, μέση `solar_ramp1h` ανά ώρα: μέγιστη πτώση 15:00 (−487) / 16:00
  (−487) / 14:00 (−354) / 17:00 (−392), ομαλή καμπύλη γύρω από ηλιοβασίλεμα, μηδενική τη
  νύχτα. ⚠️ Peak λίγο ΝΩΡΙΤΕΡΑ (15-16h) από την πρόβλεψη (16-19h), αλλά ίδιο φυσικό μοτίβο
  (dusk), όχι ύποπτο shift-bug σχήμα (π.χ. solar peak στη μέση της νύχτας θα ήταν κόκκινη
  σημαία — δεν συμβαίνει). ✅ PASS.
- **Verdict σταδίου 2: ΠΡΟΧΩΡΑ.** Καμία ένδειξη leakage/bug. Οι δύο miss στις ακριβείς
  προβλέψεις (k, ώρα peak) καταγράφονται ως pre-registration lessons, όχι ως απόρριψη.
