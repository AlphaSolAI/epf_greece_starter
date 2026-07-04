# VALIDITY CHECKLIST — DAM Price Forecasting (v1, 2026-07-04)

> Δύο λίστες: (Α) invariants που ΠΡΕΠΕΙ ΝΑ ΙΣΧΥΟΥΝ για να είναι κάθε αποτέλεσμα επιστημονικά
> έγκυρο, (Β) δουλειές που ΠΡΕΠΕΙ ΝΑ ΓΙΝΟΥΝ (με σειρά). Πηγές: SYSTEM_DESIGN §4.8-4.10,
> ABLATION_PLAN §1-2, §5.9-5.10, MASTER_PIPELINE §1-2.
> Κανόνας χρήσης: πριν γραφτεί ΟΠΟΙΟΔΗΠΟΤΕ νούμερο σε doc/διπλωματική, τσεκάρεται η λίστα Α.

## Α. ΠΡΕΠΕΙ ΝΑ ΙΣΧΥΟΥΝ (scientific invariants — έλεγχος πριν από κάθε claim)

### Α1. Information availability (anti-leakage)
- [ ] **A1.1** Κάθε feature family σέβεται το cutoff της στον πίνακα SYSTEM_DESIGN §4.8
      σε **eval ΚΑΙ train** (όχι μόνο y-lags). ⛔ ΣΗΜΕΡΑ ΠΑΡΑΒΙΑΖΕΤΑΙ (gen/load/residual
      lags) — μπλοκάρει κάθε νέο αποτέλεσμα μέχρι το Β1.
- [x] **A1.2** Same-day/same-auction πηγές (xborder same-day) δομικά ΕΚΤΟΣ parquet.
- [x] **A1.3** `tf` χρησιμοποιείται ΜΟΝΟ ως δηλωμένο oracle, ποτέ ως tradeable νούμερο.
- [ ] **A1.4** Poisoning tests περνούν: y-poisoning ΚΑΙ cross-actuals-poisoning (νέο),
      σε recursive ΚΑΙ direct, πριν από κάθε batch (θα αυτοματοποιηθεί στο preflight).
- [x] **A1.5** Νέα πηγή → πρώτα το 3-βήμα pre-flight πρωτόκολλο (πότε δημοσιεύεται /
      lag-scan / hour-profile) — ABLATION_PLAN §1.

### Α2. Χρονική συνέπεια (timezone contract)
- [x] **A2.1** Όλες οι πηγές σε CET/CEST-naive frame, μετατροπή ΜΟΝΟ στο data.py (§4.9).
- [x] **A2.2** TZ guards περνούν: preflight solar_fc DJF peak=11:00 · solar_shift_check
      peak_k=0 σε όλα τα έτη με corr ≥0.95.
- [ ] **A2.3** Κάθε fetcher δηλώνει ρητά το ρολόι του ΚΑΙ συμφωνεί με το αποθηκευμένο frame
      (εκκρεμεί: fetch_weather_2026.py ζητά UTC, parquet είναι UTC+1).

### Α3. Πειραματική μεθοδολογία
- [x] **A3.1** Κριτήρια αποδοχής: |ΔMAE| > 0.15 ΚΑΙ ίδιο πρόσημο σε ≥2 ανεξάρτητες συνθήκες
      (άλλο window Ή algo Ή στρατηγική) — αλλιώς PENDING, όχι claim (ABLATION §2).
- [x] **A3.2** Κάθε συμπέρασμα ανά feature-ομάδα = ζεύγος (χειμώνας, καλοκαίρι) — ποτέ 1 εποχή.
- [x] **A3.3** Headline μόνο με: ≥3 seeds (std γνωστό) + 2ο out-of-sample window.
- [x] **A3.4** Control-run πρωτόκολλο σε κάθε δομική αλλαγή (backup swap — το Δ αποδίδεται
      αποκλειστικά στην αλλαγή). Εφαρμόστηκε στο TZFIX (16.096 ✓).
- [x] **A3.5** Reproducibility anchor: γνωστό config αναπαράγει καταγεγραμμένο MAE ±0.05.
- [x] **A3.6** Ensembles/calibrated μέθοδοι: βάρη/παράμετροι ΜΟΝΟ από out-of-sample περίοδο
      πριν το eval (ποτέ από το eval window).
- [ ] **A3.7** Συγκρίσεις μεθόδων πάντα στο ΙΔΙΟ test window / ίδιο gate / ίδια δεδομένα
      (μετά από κάθε δομική αλλαγή: ΟΛΑ τα συγκρινόμενα ξανατρέχουν — τίποτα παλιό δίπλα σε νέο).

### Α4. Probabilistic layer (όταν τρέξει)
- [x] **A4.1** Conformal calibration αυστηρά αιτιατό: residuals ΜΟΝΟ πριν από κάθε test
      σημείο, trailing 4-8 εβδ., per hour-of-day (υλοποιημένο στο src/conformal.py ✓).
- [ ] **A4.2** Coverage αναφέρεται ΠΑΝΤΑ μαζί με sharpness (μέσο πλάτος) + pinball.
- [ ] **A4.3** Αξιολόγηση σε ≥2 μοντέλα × ≥2 windows πριν από συμπέρασμα.

### Α5. Αναπαραγωγιμότητα & ιχνηλασιμότητα
- [ ] **A5.1** Κάθε νούμερο σε doc δείχνει σε JSON/CSV στο runs/ ή results/ + εντολή αναπαραγωγής.
- [x] **A5.2** MD αρχεία = μόνο τρέχουσα αλήθεια, παλιά νούμερα σε ΑΝΑΣΤΟΛΗ με banner (έγινε).
- [x] **A5.3** Git: δουλειά commit-άρεται τακτικά + push σε remote — ✅ πρώτο push 2026-07-04
      (origin/FEB272026, commit a810137). Εφεξής: commit+push μετά από κάθε session με αλλαγές.
      ⚠️ Το branch `FEB272026_localhistory` (παλιό ιστορικό) ΔΕΝ pushάρεται ποτέ (4.3GB object).
- [ ] **A5.4** environment.yml υπάρχει και ενημερώνεται (conda env export).

## Β. ΠΡΕΠΕΙ ΝΑ ΓΙΝΟΥΝ (με σειρά — τίποτα δεν προσπερνά το προηγούμενο gate)

### Β1. 🚨 AEL — Availability Enforcement Layer (μπλοκάρει τα πάντα)
- [ ] Υλοποίηση freeze-at-cutoff για gen_*/residual_load/load lags σε recursive row-build,
      direct row@cutoff, ΚΑΙ training rows (train/serve συνέπεια). SYSTEM_DESIGN §4.8.
- [ ] Poisoning self-test για cross actuals (§4.10.2) + ένταξη στο preflight.
- [ ] NaN-variant ως sensitivity (trees) — 1 run σύγκρισης.
- [ ] Πρώτα leak-free νούμερα: static Q1 σε default / default,-meteo,-resfc /
      lags,calendar,genlags → ABLATION §5.10.

### Β2. Git & υποδομή (παράλληλα με Β1, μικρό)
- [x] Επισκευή .git (refs/ ξαναχτίστηκαν, FEB272026 → abcadb3, stale lock αφαιρέθηκε) — 2026-07-04.
- [x] Commit + push στο origin — ✅ 2026-07-04: clean orphan branch FEB272026 (a810137) →
      origin (LFS 108MB)· παλιό ιστορικό τοπικά ως FEB272026_localhistory (μη-pushable: 4.3GB
      sarima cache + missing blobs).
- [ ] `git worktree prune` (10 ορφανά worktrees) + αποκατάσταση main ref (d97a827) αν χρειάζεται.
- [ ] `conda env export -n epf > environment.yml`.
- [ ] fetch_weather_2026.py: ευθυγράμμιση timezone με το αποθηκευμένο frame (Α2.3).

### Β3. Re-ablation στα καθαρά δεδομένα (μετά το Β1)
- [ ] Core ablation: LGBM+XGB × (Q1, καλοκαίρι 2025) × (recursive, direct), static.
      Ανοιχτά ερωτήματα: resfc-βλάπτει-recursive (γενικεύεται;) · meteo strategy-effect ·
      αξία genlags ΧΩΡΙΣ leak · dense · λιτός πυρήνας vs default.
- [ ] Retrain cadence (static/monthly/weekly) στον νέο νικητή + 3 seeds + Μάρτιος 2026
      → **ΝΕΟ HEADLINE** (αντικαθιστά το 15.17 παντού).
- [ ] SS×(monthly, weekly) στο νέο config · weekly LGBM+XGB ensemble confirm.

### Β4. Probabilistic layer (μετά το Β3 — ο κώδικας ΥΠΑΡΧΕΙ ήδη)
- [ ] Extended point-forecast run (test_start ~8 εβδ. πριν το eval window για calibration).
- [ ] split-conformal + quantile-lgbm σε ≥2 μοντέλα × Q1+Μάρτιο → §Conformal-RESULTS.

### Β5. Πληρότητα έρευνας DAM price
- [ ] task=load πλήρες ablation (τροφοδοτεί και load_fc της τιμής).
- [ ] henex_premarket ομάδα (με Α1.5 πρωτόκολλο) · xb_lag1_h0 · summer xborder confirm.
- [ ] Weather forecast archive migration (reanalysis → πραγματικά ιστορικά forecasts).
- [ ] Προαιρετικά: Chronos/TimesFM zero-shot στήλη · LSTM calibration bug.

### Β6. Προϊόν (L3-L5) — μετά τα Β3/Β4
- [ ] Daily batch runner + fallback chain · settle/monitoring loop · delivery JSON.

## Ιστορικό ελέγχων (γεμίζει σε κάθε μεγάλο βήμα)

| Ημ/νία | Έλεγχος | Αποτέλεσμα |
|---|---|---|
| 2026-07-04 | TZ alignment (solar_shift_check, όλα τα έτη) | ✅ peak_k=0, corr 0.98 |
| 2026-07-04 | Control run TZFIX (backup swap) | ✅ 16.096 αναπαράχθηκε |
| 2026-07-04 | Cross-actuals poisoning | ⛔ ΔΕΝ ΥΠΑΡΧΕΙ ΑΚΟΜΑ (Β1) |
| 2026-07-04 | Git repair (refs rebuild, tip abcadb3) | ✅ git log/status ΟΚ |
