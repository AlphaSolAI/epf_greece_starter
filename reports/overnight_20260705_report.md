# OVERNIGHT 2026-07-05 — Ενδιάμεσο Report (Blocks 0/A/B πλήρη, C μερικό)

> Παράχθηκε 2026-07-05 ~11:30 ενώ το batch ΑΚΟΜΑ τρέχει (Block C 27/56).
> Πηγή αλήθειας: `ABLATION_PLAN.md §5.12` + `results/overnight_20260705.csv`.
> Αναπαραγωγή πινάκων: `python scripts/synthesize_ablation.py --dir runs/overnight_20260705/<block>`
> ή `conda run -n epf ... scripts/overnight_summarize.py` (πλήρες batch).

## Executive summary

1. **Weekly retrain > monthly — ACCEPTED** (12/12 συνθήκες, |Δ| έως −1.8 €/MWh).
2. **`default,dense` = καλύτερο spec** στο cadence stage — ACCEPTED (8/8 στο Block A,
   +4/4 Μάρτιος). **Νέο headline candidate: LGBM weekly `default,dense` → Q1 17.035,
   Summer 13.812** — κλειδώνει ΜΟΝΟ μετά τα seeds (Block D).
3. **2 ανοιχτές συγκρούσεις** (Μάρτιος vs §5.11): meteo και lean-core-direct — PENDING,
   χρειάζονται 3ο ανεξάρτητο window, ΔΕΝ αντικαθιστούν τα established verdicts.
4. **Bug στο LOAD ablation**: το `-loadfc` arm είναι VOID (το `hourly_load.parquet` δεν
   έχει καν στήλη `load_fc`) — εντοπίστηκε από το Δ=0.000 bit-for-bit red flag.
5. 0 FAILED runs σε όλο το batch μέχρι στιγμής.

## Setup (κοινό σε όλα)

`--gate strict`, AEL `crosslag_mode=freeze`, TZFIX parquet, price=€/MWh, load=MW.
Windows: Q1 = 2025-12-01→2026-02-28 (2160h) · Summer = 2025-06-01→08-31 (2208h) ·
Μάρτιος = 2026-03-01→03-20 (480h, static train_end 2026-02-28).
Batch: `scripts/overnight_20260705.sh`, detached (εκκίνηση 04:58:50), log:
`logs/overnight_20260705_master.log`.

## Αποτελέσματα ανά block

### Block A — Retrain cadence (✅ 24/24) → βλ. ABLATION §5.12α
Weekly κερδίζει monthly παντού. Το κέρδος του weekly είναι μεγαλύτερο στον χειμώνα
(−1.3..−1.8) από το καλοκαίρι (−0.0..−0.9) — λογικό: χειμερινό regime drift.
Το dense κερδίζει το default σε ΟΛΑ τα κελιά και στα δύο cadences.
Το `-resfc` βοηθάει ΜΟΝΟ χειμώνα-monthly (−1.6/−1.8) και βλάπτει καλοκαίρι — το γνωστό
εποχιακό interaction· ΣΗΜΑΝΤΙΚΟ: στο weekly-Q1 το όφελος του `-resfc` σχεδόν εξαφανίζεται
(−0.07/−0.50), δηλ. το συχνό retrain «θεραπεύει» εν μέρει την τοξικότητα του resfc.

### Block B — Μάρτιος tie-break (✅ 28/28) → βλ. ABLATION §5.12β
dense −0.56..−2.80 σε 4/4 (2ο window ✓). resfc: ταιριάζει με το interaction story.
Συγκρούσεις (PENDING): meteo φαίνεται να ΒΛΑΠΤΕΙ στον Μάρτιο (3/4 κελιά, έως −1.57)·
lean core ΚΕΡΔΙΖΕΙ στο March-direct (2/2, έως −1.42). Και τα δύο = 1 window, δεν
ανατρέπουν §5.11 (8/8 και 4/4 αντίστοιχα σε Q1+summer).
Παρατήρηση: το direct base MAE στον Μάρτιο (24.5/24.7) είναι πολύ χειρότερο από το
recursive (20.9/21.2) — συνεπές με το γνωστό «recursive > direct για DAM».

### Block C — LOAD ablation (⏳ 27/56, μόνο LGBM ως τώρα) → βλ. ABLATION §5.12γ
- ⚠️ `-loadfc` arm VOID (bug, βλ. παρακάτω) — καμία γραμμή loadfc δεν είναι εύρημα.
- Ύποπτο μέγεθος: αφαίρεση genlags+loadlags στο q1-load-direct = **−88 MW (26% του
  baseline!)** αλλά +27 στο summer-direct — πριν από οποιοδήποτε claim χρειάζεται
  targeted poisoning check στο crosslag family για task=load.
- meteo στο load: βλάπτει έντονα q1-recursive (+109 MW χωρίς αυτό;… όχι: ΤΟ αφαιρούμε
  και χειροτερεύει +109 → meteo ΒΟΗΘΑΕΙ πολύ το load-Q1-recursive), μικτό αλλού.
- Οριστικοί πίνακες/verdicts όταν κλείσει το 56/56 (χρειάζεται και XGB).

### Blocks D/E/F — δεν έχουν ξεκινήσει (seeds+dense×direct / SS / conformal smoke).

## Validation report (κατά /validate-data)

**Overall: Ready to share (για A/B) · Needs completion (C — μερικό + bug flagged).**

- Calculation spot-checks: 5 δείγματα από 3 blocks — recomputed MAE ≡ JSON metrics ≡
  summarizer CSV (4 δεκαδικά), n_hours σωστά (2160/2208/480) → κανένα aggregation bug.
- Methodology: ίδιο gate/AEL/windows παντού· συγκρίσεις μόνο εντός ίδιου window/
  strategy· καμία σύγκριση με προ-AEL νούμερα.
- Pitfalls που ελέγχθηκαν: incomplete-period (όχι — πλήρη windows), timezone (TZFIX
  guards στο preflight), «identical values» red flag (ΠΙΑΣΤΗΚΕ — loadfc VOID arm),
  correlated-cells-ως-ανεξάρτητα (ΠΙΑΣΤΗΚΕ — §2 independence rule στα March verdicts).
- Caveats για stakeholders/διπλωματική: (α) headline ΔΕΝ έχει κλειδώσει (εκκρεμούν
  seeds), (β) οι 2 συγκρούσεις Μαρτίου είναι ανοιχτές ερωτήσεις όχι ευρήματα,
  (γ) LOAD συμπεράσματα ΜΟΝΟ μετά 56/56 + poisoning check + parquet fix.

## Τι μένει (κατά /deploy-checklist) — Β-σειρά

### Άμεσα (αυτό το batch)
- [ ] Block C να κλείσει (56/56) → §5.12γ οριστικό (με XGB)
- [ ] Block D seeds 7/123 → std → **ΚΛΕΙΔΩΜΑ νέου headline** (§2 κανόνας 5)
- [ ] Block E SS × {default, dense} → επιβιώνει το SS leak-free; προσθέτει πάνω στο dense;
- [ ] Block F conformal smoke → πρώτη coverage/sharpness εικόνα
- [ ] Τελικό summarize + συμπλήρωση §5.12γ-στ + ενημέρωση last.md §1-3

### Νέα από σήμερα
- [ ] **Rebuild `hourly_load.parquet` με load_fc** (merge `load_forecast_hourly.parquet`)
      → μετά re-run του loadfc arm (Block C bis). Backup+σύγκριση πριν σβηστεί το παλιό.
- [ ] Poisoning check crosslag family για task=load (το −88 MW / 26% είναι ύποπτο)
- [ ] 3ο ανεξάρτητο window (π.χ. Δεκ-only ή Απρ 2026 αν υπάρχουν δεδομένα) για να
      λυθούν οι 2 συγκρούσεις meteo / lean-core-direct

### Β4-Β6 (επόμενα στάδια)
- [ ] Β4 conformal πλήρες: ≥2 μοντέλα × Q1+Μάρτιος, pinball+coverage+sharpness μαζί
- [ ] Β5: henex_premarket · xb_lag1_h0 · solar_fc 2h-shift check · SS×weekly ·
      weather forecast archive · Chronos/TimesFM · LSTM calibration
- [ ] Β6 προϊόν: daily runner, settle loop, delivery
- [ ] Infra: environment.yml · worktree prune (permission denied — θέλει restart/χειροκίνητο)
      · fetch_weather_2026 TZ align · figures (create-viz) όταν ελευθερωθεί το conda

## Rollback triggers (αναλογία deploy)
- Οποιοδήποτε poisoning test FAIL → πάγωμα όλων των claims του αντίστοιχου family.
- Reproducibility anchor εκτός ±0.05 → πάγωμα, control run.
- Δ=0.000 bit-for-bit σε ablation → arm VOID, όχι εύρημα (SKILL κανόνας 12).
