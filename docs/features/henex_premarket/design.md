# Feature Design Doc — henex_premarket

> Στάδιο 1 (feature-eng). Συμπληρώθηκε 2026-07-06 με **πραγματική επιθεώρηση** του
> repo (όχι υποθέσεις) — δες §7 για τα ευρήματα. **Status: BLOCKED πριν από το Στάδιο 2**
> (βλ. §5). Κανένας κώδικας feature δεν γράφτηκε.

## 1. Ταυτότητα

- **Feature/πηγή:** HENEX (Ελληνικό Χρηματιστήριο Ενέργειας) pre-market bilateral
  nomination volumes. 3 στήλες σε cached parquet: `pm_buy_nom_mw`, `pm_sell_nom_mw`,
  `pm_net_nom_mw` (πιθανώς net = buy − sell).
- **Ομάδα (flag) στο feature_availability.py:** `premarket` — **ΔΕΝ υπάρχει ακόμα**.
  Καμία αναφορά σε `feature_availability.py` (`grep` = 0 hits). Η ομάδα πρέπει να
  δημιουργηθεί από την αρχή.
- **Στήλες parquet:** οι 3 παραπάνω· αφορούν αποκλειστικά **price/DAM** (καμία έννοια
  «premarket load nomination» δεν βρέθηκε) → πιθανώς μόνο task=price.
- **Availability rule (πρόχειρο):** ΑΓΝΩΣΤΟ ακόμα — εξαρτάται από το §2/T1 (πότε
  δημοσιεύεται vs gate). Working hypothesis: day-ahead-known (bilateral nominations
  πρέπει να «κλειδώσουν» πριν το DAM auction gate) — **ΑΝΕΠΙΒΕΒΑΙΩΤΟ**, βλ. §6.

## 2. Μηχανισμός & pre-registered προσδοκίες

- **Γιατί να βοηθάει (φυσικός/αγοραίος μηχανισμός):** οι διμερείς (bilateral/OTC)
  nominations για την ημέρα παράδοσης πρέπει να δηλωθούν/καθαρίσουν (netting) πριν
  κλείσει το DAM auction, αφού επηρεάζουν το καθαρό υπόλοιπο ζήτησης/προσφοράς που
  μπαίνει στο auction. Ένα ισχυρά αρνητικό `pm_net_nom_mw` (καθαρή θέση πωλητή/πλεόνασμα
  προσφοράς εκτός auction) θα αναμενόταν να συνδέεται με **χαμηλότερη** DAM τιμή (λιγότερη
  ζήτηση μένει για το auction) — αλλά αυτό είναι υπόθεση, όχι επιβεβαιωμένο.
- **Αναμενόμενο πρόσημο ΔMAE:** άγνωστο/ουδέτερο έως ελαφρώς θετικό (βελτίωση) — **δεν
  δικαιολογείται ακόμα ποσοτική εκτίμηση** χωρίς lagscan έναντι του y. ΔΕΝ γράφεται
  αριθμός «αναμενόμενο μέγεθος» μέχρι να τρέξει το πραγματικό lagscan (θα ήταν μαντεψιά).
- **Πού αναμένεται να δρα:** στρατηγική recursive (η προεπιλογή του project) · task=price
  (καμία ένδειξη load-relevance) · εποχικότητα άγνωστη.
- **Αναμενόμενο lagscan peak:** αν η υπόθεση δημοσίευσης ισχύει, peak σε k≥1 (η στήλη
  προηγείται) — ΔΕΝ μπορεί να δοκιμαστεί ακόμα (βλ. §5, blocker β).
- **Αναμενόμενο hour-profile:** άγνωστο a priori. Προκαταρκτική (μη-επίσημη) παρατήρηση
  σε ωμά δεδομένα Δεκ-2025 (§7): `pm_net_nom_mw` πιο αρνητικό στις ώρες αιχμής
  βραδιού (17-19:00, ≈ −372 έως −377) και πιο κοντά στο μηδέν το μεσημέρι (11-13:00,
  ≈ −208 έως −212). Υπάρχει πραγματικό ωριαίο pattern (όχι flat/dummy) — ενθαρρυντικό
  σημάδι ότι τα δεδομένα είναι αληθινά, ΟΧΙ απόδειξη νομιμότητας/χρησιμότητας.
- **Αναμενόμενη τάξη FI rank ομάδας:** άγνωστο — μόνο 3 στήλες, μικρή ομάδα, αναμένεται
  χαμηλά αν καθόλου χρήσιμη (σαν `fuel`/`other`), όχι στην κορυφή.

## 3. Acceptance tests (κλειδώνουν ΤΩΡΑ — T1-T8)

| Test | Τι ελέγχει | Κριτήριο PASS | Status |
|---|---|---|---|
| T1 | Gate timing | Γραπτή, **επιβεβαιωμένη από πρωτογενή πηγή** (HENEX Market Rules/publication calendar) απάντηση ότι η nomination δημοσιεύεται πριν το 12:00 CET D-1 | ❌ **OPEN** — καμία πρωτογενής πηγή βρέθηκε σε αυτό το session, μόνο η οικονομική υπόθεση του §2. ΔΕΝ περνάει ακόμα. |
| T2 | Lagscan | Peak στο προβλεπόμενο k · ΟΧΙ «βολικό» k · \|corr\|<0.85 | ⛔ **BLOCKED** — `lagscan.py` διαβάζει ΜΟΝΟ από `data/processed/hourly.parquet`· οι 3 στήλες ΔΕΝ υπάρχουν εκεί ακόμα (βλ. §5, blocker γ). |
| T3 | Hour-profile | Σχήμα λογικό, χωρίς ξαφνικό shift | ◐ Προκαταρκτικό OK (βλ. §2), επίσημος έλεγχος εκκρεμεί μαζί με T2 |
| T4 | Non-VOID arm | `#features` αλλάζει· στήλη υπάρχει στο parquet ΚΑΘΕ target task | ⛔ N/A ακόμα — η ομάδα δεν έχει καν οριστεί |
| T5 | Poisoning | `preflight_check.py --poison` PASS μετά feature_availability.py edit | Εκκρεμεί (Στάδιο 3) |
| T6 | Reproducibility | Control run, anchor ±0.05 | Εκκρεμεί (Στάδιο 3) |
| T7 | §2 gate | \|ΔMAE\|>0.15 ΚΑΙ ίδιο πρόσημο σε ≥2 ΑΝΕΞΑΡΤΗΤΑ windows | ⚠️ **ΡΙΣΚΟ**: με τα τρέχοντα δεδομένα υπάρχει ΜΟΝΟ 1 πλήρες window (καλοκαίρι 2025) — βλ. §5, blocker α. Q1/Μάρτιος δεν επαρκούν. |
| T8 | Too-good-to-be-true | Δ όχι πολύ μεγαλύτερο του pre-registered· FI rank εύλογο | Εκκρεμεί (Στάδιο 5-6) |

## 4. Batch matrix (στάδιο 4 — ΠΡΟΣΩΡΙΝΟ, εξαρτάται από λύση των blockers §5)

- Windows: **καλοκαίρι 2025 (πλήρες)** + ένα ΔΕΥΤΕΡΟ πλήρες window (χρειάζεται backfill,
  βλ. §5-α) — Q1 2026 ΔΕΝ αρκεί όπως έχει (μόνο Δεκέμβριος).
- Algos: LGBM + XGB · strategy: recursive (πρώτα) · retrain: static · seed: 42
- Specs: `default` vs `default,premarket`
- Outputs: `runs/feat_henex_premarket/<window>_<algo>_static_<spec>.json` + `results/feat_henex_premarket.csv`
- Εκτίμηση διάρκειας: όπως τυπικό ablation ζεύγος (λεπτά ανά config) — δεν χρειάζεται detached αν 1-2 configs.
- **ΔΕΝ ξεκινάει πριν λυθούν οι blockers §5.**

## 5. Γνωστοί κίνδυνοι — BLOCKERS (βρέθηκαν 2026-07-06, δεν είναι υποθέσεις)

**α) Ανεπαρκής κάλυψη δεδομένων σε σχέση με τα ενεργά windows.** Το cached parquet
(`data/processed/henex_premarket_hourly.parquet`, 615KB, χτισμένο 2026-02-27) καλύπτει
2020-11-01 → **2026-01-01 μόνο**. Μετρημένη κάλυψη (system python, εκτός conda ουράς):

| Window | Ώρες παρούσες | Αναμενόμενες | Κάλυψη |
|---|---|---|---|
| Q1 2026 (Δεκ-Φεβ) | 768 | 2160 | **35.6%** (μόνο Δεκέμβριος· Φεβρουάριος 100% λείπει) |
| Καλοκαίρι 2025 (Ιουν-Αυγ) | 2208 | 2208 | **100%** ✅ |
| Μάρτιος 2026 | 0 | ~456 | **0%** |

Με τα σημερινά δεδομένα υπάρχει **μόνο ΕΝΑ** πλήρες ανεξάρτητο window (καλοκαίρι) — το
T7 (≥2 windows) είναι δομικά αδύνατο να περάσει χωρίς backfill.

**β) Καμία ενεργή πηγή/fetcher.** `data/raw/henex/premarket_summary/` υπάρχει αλλά είναι
**εντελώς άδειος φάκελος** (0 αρχεία). Δεν υπάρχει `fetch_henex*.py` στο `src/` (μόνο
entsoe/weather fetchers). Το module `src/data_future.py` (που ορίζει `load_premarket_hourly()`)
ψάχνει μάλιστα σε **λάθος path** (`data/raw/henex_premarket`, όχι το πραγματικό
`data/raw/henex/premarket_summary`) — ακόμα κι αν υπήρχαν raw αρχεία εκεί, ο κώδικας δεν θα
τα έβρισκε. Άγνωστο από πού προήλθε το cached parquet του Φεβρουαρίου (πιθανώς
χειροκίνητο one-off build πριν αδειάσει/μετακινηθεί ο φάκελος raw).

**γ) `src/data_future.py` είναι ορφανό module — ΔΕΝ τροφοδοτεί το `hourly.parquet`.**
Επιβεβαιώθηκε: κανένα import/reference του `data_future` σε άλλο module εκτός από ένα
σχόλιο στο `split_utils.py`. Το `src/data.py` (ο πραγματικός builder του `hourly.parquet`
που διαβάζει το `lagscan.py`/`feature_availability.py`) δεν αναφέρει καθόλου premarket.
Άρα οι 3 στήλες δεν υπάρχουν σήμερα πουθενά στο ενεργό feature store — πρέπει να μπουν
στο `data.py` (Στάδιο 3.1), πριν καν τρέξει το lagscan (μη-τυπική σειρά σε σχέση με το
συνηθισμένο ingest-audit flow, όπου το lagscan συνήθως τρέχει σε ήδη-merged στήλη).

**δ) T1 ανεπιβεβαίωτο.** Δεν βρέθηκε πρωτογενής πηγή (HENEX market rules/publication
calendar) για την ΑΚΡΙΒΗ ώρα δημοσίευσης του premarket nomination έναντι του gate 12:00
CET D-1. Η οικονομική λογική (§2) είναι εύλογη αλλά ΔΕΝ αρκεί — μάθημα xborder:
«εύλογη λογική» χωρίς πρωτογενή απόδειξη ήταν ακριβώς πώς πέρασε το leakage πρώτη φορά.

## 6. Σύσταση (πριν προχωρήσει το Στάδιο 2)

**ΔΕΝ προτείνεται να προχωρήσει το lagscan/audit πριν:**
1. Βρεθεί/επιβεβαιωθεί πρωτογενής πηγή για το χρόνο δημοσίευσης (T1) — HENEX website/market rules.
2. Αποφασιστεί πώς θα καλυφθεί το data gap (backfill raw αρχείων ΚΑΙ διόρθωση του path
   bug στο `data_future.py`, ΕΙΤΕ νέος fetcher αν υπάρχει HENEX API/portal) ώστε να
   υπάρχουν ≥2 πλήρη windows.
3. Οι στήλες μπουν στο `src/data.py` (staging merge, ΧΩΡΙΣ ακόμα registration στο
   `feature_availability.py`) ώστε το `lagscan.py` να μπορεί να τρέξει.

Μέχρι τότε: **PENDING στο `ABLATION_PLAN.md §7`**, όχι ενεργό πείραμα.

## 7. Audit evidence (ό,τι μπόρεσε να ελεγχθεί ΧΩΡΙΣ conda, ενώ έτρεχε το followup batch)

- Εντοπισμός: `src/data_future.py:19` (path), `:227-251` (`load_premarket_hourly`).
- Parquet inspection (`Python311` + `fastparquet`, εκτός conda queue — δεν παραβίασε
  τον κανόνα «ΕΝΑ conda process»): shape (45264, 3), κάλυψη 2020-11-01→2026-01-01,
  βλ. πίνακα §5-α.
- `grep -rn henex src/` → 0 hits σε `feature_availability.py`, `data.py`, οποιοδήποτε
  `fetch_*.py`.
- Verdict σταδίου 2: **ΔΕΝ ΞΕΚΙΝΗΣΕ** (μπλοκάρεται από §5) — γράφτηκε στο `ABLATION_PLAN.md §7`: ναι.
