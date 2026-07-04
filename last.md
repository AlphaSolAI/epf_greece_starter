# SESSION HANDOFF — energy trading agent (GR EPF/STLF)

> **Ενημερώθηκε 2026-07-04** (μετά OVERNIGHT PROTOCOL P1-P6 + αναδιοργάνωση φακέλου).
> Περιέχει ΜΟΝΟ την τρέχουσα αλήθεια. Πλήρες ιστορικό sessions (πορεία, ανατροπές, xborder
> leakage post-mortem): `OLD/docs/last_history_20260704.md`.
> Διαβάζεται ΜΑΖΙ με: `.claude/skills/energy-forecast/SKILL.md` (κανόνες/εντολές) ·
> `ABLATION_PLAN.md` (έγκυρα ευρήματα + pending + επόμενα βήματα) ·
> `MASTER_PIPELINE_DESIGN.md` (anti-leakage θεωρία) · `SYSTEM_DESIGN_TRADING_AGENT.md` (προϊόν).

## 1. Πού είμαστε (μία ματιά — ΑΝΑΤΡΟΠΗ 2026-07-04 απόγευμα)

**⚠️ ΟΛΑ τα νούμερα του §5 του ABLATION_PLAN είναι ΣΕ ΑΝΑΣΤΟΛΗ** μετά από 2 ευρήματα:

1. **TZFIX (✅ διορθώθηκε + rebuild + guard)**: το parquet ανακάτευε 4 ρολόγια — index=CET/CEST,
   gen +1h αργά (Athens), resfc 1-2h νωρίς (UTC), meteo 1h νωρίς καλοκαίρι (UTC+1 σταθερό).
   Απόδειξη/επαλήθευση: corr(solar_fc, gen_solar) 0.65→**0.98** στο k=0, όλα τα peaks σωστά.
   Νέο baseline default static Q1 = **19.17** (control στο backup: 16.10 ✓ αναπαράγεται).
   Στα σωστά δεδομένα το **resfc βλάπτει το recursive-χειμώνα (~+2.4)** → το παλιό feature
   selection ήταν προσαρμοσμένο στα στραβά δεδομένα, ξαναγίνεται. `ABLATION_PLAN §5.9`.
2. **🚨 Engine leak (ΑΝΟΙΧΤΟ — πρώτη προτεραιότητα)**: `recursive_openloop.py` αντικαθιστά
   μόνο y-lags· τα gen/load/residual lags εντός ορίζοντα έρχονται από ACTUALS (άγνωστα στο
   gate) = leakage σε ΚΑΘΕ config με genlags/loadlags, και στο παλιό headline. `§5.10`.

**Επόμενο στάδιο: engine leak fix → re-ablation στα καθαρά δεδομένα → νέο headline →
conformal.** Πλήρες πλάνο: `ABLATION_PLAN §8.2`.

## 2. Κλειδωμένα συμπεράσματα (πλήρης λίστα: `ABLATION_PLAN.md §5`)

1. **xborder ΕΚΤΟΣ default** — same-day ήταν leakage (ακυρώθηκαν 14.43/15.02)· το νόμιμο lagged
   βλάπτει τον χειμώνα (3/3 cadences). Καλοκαίρι βοηθάει (1 σημείο) → PENDING.
2. **recursive > direct για DAM** (16.1 vs 19.5 στο Q1) — και το retrain δεν βοηθάει το direct.
3. **meteo = strategy-effect 4/4**: βοηθάει πάντα στο direct, βλάπτει πάντα στο recursive.
4. **genlags** πολύτιμο σε 3 αλγόριθμους × 2 στρατηγικές × 2 εποχές — το πιο στιβαρό feature.
5. **resfc** πολύτιμο ΤΟ ΚΑΛΟΚΑΙΡΙ (LGBM+XGB) — χειμωνιάτικα windows το υποεκτιμούν. **loadfc**
   άχρηστο/αρνητικό (2 αλγόριθμοι συμφωνούν).
6. **SS-linear ΔΕΝ είναι redundant με retrain** (−0.24 και σε monthly, καθαρό config) — το παλιό
   «redundant» ήταν artifact του xborder config. SS×weekly αδοκίμαστο.
7. **Ensembles: κανένα δεν κερδίζει στιβαρά το LGBM** — weighted-by-1/MAE απορρίφθηκε με τίμιο
   calibration· μόνο το weekly LGBM+XGB (−0.148) είναι οριακά ελπιδοφόρο (PENDING).
8. **LEAR θέλει δικό του feature set** (αφαίρεση resfc/fuel το ΒΕΛΤΙΩΝΕΙ — L1 effect)·
   MLP: bare core κερδίζει το default κατά 1.6· LSTM: calibration bug, όχι tradeable.

## 3. Οργάνωση φακέλου (ΝΕΑ, 2026-07-04 — μάθε την πριν ψάξεις οτιδήποτε)

```
runs/      ← ΟΛΟΙ οι φάκελοι εξόδου πειραμάτων (p1_out..p6_out, ablation_*, confirm_out*,
             master_grid_out*, ss_out, e1_e2_out, step9_out, preflight_out)
results/   ← όλα τα results_*.csv (ablation πίνακες, grids)
logs/      ← όλα τα _step*_log.txt + λοιπά logs
scripts/   ← run_*.sh (ιστορικά batch), export_appendix_fi.py, make_weather_viz.py
reports/   ← ablation_20260702/ (η οπτική αναφορά, ΔΙΟΡΘΩΜΕΝΗ + πλήρης με P1-P6),
             results_dashboard.html, weather_gr_hourly_viz.html
OLD/       ← αρχείο: docs/ (παλιές εκδόσεις MD με το πλήρες ιστορικό), notes/, OLD_SCRIPTS...
data/, src/, thesis/, dashboard(+.html)  ← ως είχαν
```
Τα default outdirs των εργαλείων (π.χ. `ablation_out/`) θα ξαναδημιουργούνται στο root σε νέα
τρεξίματα — είτε δώσε `--outdir runs/<όνομα>` είτε μετακίνησέ τα στο τέλος.

## 4. Workflow με τον χρήστη (ΝΕΟΣ ΚΑΝΟΝΑΣ από 2026-07-04)

- **Όταν ο χρήστης είναι online**: δίνε του τις έτοιμες εντολές (ένα copy-paste block, με
  `--out_json`/logs σε σωστά paths) να τις τρέχει ΕΚΕΙΝΟΣ στο VS Code terminal, ώστε να μη
  δεσμεύεται το παράθυρο του Claude. Μετά ο χρήστης επικολλά το RESULT line ή το path.
- **Σε autonomous/overnight**: όπως πριν (background+notifications, ΕΝΑ conda process τη φορά).
- Στο ΤΕΛΟΣ ΚΑΘΕ απάντησης με ανοιχτή δουλειά: γράφε ορατά (α) PENDING λίστα, (β) το γενικό
  πλάνο/στάδιο, (γ) το ΕΠΟΜΕΝΟ ακριβές prompt — ο χρήστης θέλει να τα βλέπει στη συνομιλία.

Λειτουργικά (αμετάβλητα): conda `epf` · `conda run -n epf --no-capture-output python -X utf8` ·
multi-line python ΜΟΝΟ σε αρχείο (ποτέ `-c`) · OneDrive πρέπει να τρέχει · rebuild parquet
πάντα με backup+σύγκριση · `--train_end` μόνο με static.

## 5. Επόμενο βήμα + ΕΤΟΙΜΟ PROMPT

**Κύριο (ΑΛΛΑΞΕ 2026-07-04 απόγευμα): engine leak fix ΠΡΙΝ από οτιδήποτε άλλο** — κανένα νέο
ablation/headline δεν έχει νόημα όσο τα gen/load/residual lags διαρρέουν actuals στο eval.
Έτοιμο prompt για νέο session:

```
Διάβασε last.md και ABLATION_PLAN.md §5.9-§5.10. Διόρθωσε το crosslag leakage στο engine:
στο recursive_openloop.py (και στο direct path του master_forecast) τα actual-derived non-y
lags (gen_solar_lag*, gen_wind_lag*, residual_load_lag*, load_lag*) πρέπει σε eval να
σέβονται το cutoff διαθεσιμότητας actuals (γνωστά έως ~11:00 D-1 στο strict): για κάθε
(target t, lag k) με t−k > cutoff χρησιμοποίησε freeze-at-cutoff (τελευταία νόμιμη τιμή)
— ΚΑΙ στο training εφάρμοσε το ίδιο σχήμα για να μην υπάρχει train/serve mismatch. Πρόσθεσε
poisoning self-test: δηλητηρίασε actuals εντός ορίζοντα → MAE αμετάβλητο. Μετά τρέξε
static Q1: default / default,-meteo,-resfc / lags,calendar,genlags στα καθαρά δεδομένα
και γράψε τα πρώτα leak-free νούμερα στο ABLATION_PLAN §5.10. Αν είμαι online δώσε μου
τις εντολές· αλλιώς background.
```

**Μετά (με σειρά)**: re-ablation (LGBM+XGB × Q1+καλοκαίρι × recursive+direct) → νέο headline
(cadence+seeds+Μάρτιος) → conformal (κανόνες: SKILL.md §Conformal — model-agnostic, ≥2 μοντέλα
× 2 windows, pinball+coverage+sharpness, vs quantile-LGBM).

**Μείζον ερευνητικό ανοιχτό**: **task=load πλήρες ablation** — κανένα feature engineering για
load πέρα από E2/meteo· ό,τι ξέρουμε για ομάδες ισχύει μόνο για price (`ABLATION_PLAN.md §8.2`).

**Λοιπά (με σειρά, `ABLATION_PLAN.md §7-8`)**: solar_fc 2h-shift — ΝΕΟ εργαλείο
`.claude/skills/energy-forecast/scripts/solar_shift_check.py` (fc vs actual ανά έτος· το
«peak 10:00» του lagscan είναι UTC = 12-13:00 τοπική, πιθανό red herring) · xb_lag1_h0 (§8.1) ·
SS×weekly · henex_premarket.

## 6. Ανοιχτά ζητήματα υποδομής (όχι μοντέλα)

1. ~~Git repo σπασμένο~~ ✅ **ΕΠΙΣΚΕΥΑΣΤΗΚΕ + ΠΡΩΤΟ PUSH 2026-07-04**: (α) έλειπε όλο το
   `.git/refs/` (+logs/, θύμα OneDrive) — ξαναχτίστηκε, tip ανακτήθηκε από objects (abcadb3)·
   (β) ο χρήστης έκανε commit όλης της δουλειάς· (γ) το push απέτυχε: το ΠΑΛΙΟ ιστορικό είχε
   `models/sarima_hourly_cache.pkl` 4.3GB (>2GiB GitHub LFS limit) + missing blobs → μη-pushable·
   (δ) λύση: παλιό ιστορικό κρατήθηκε ΤΟΠΙΚΑ ως branch `FEB272026_localhistory`, το `FEB272026`
   ξαναγεννήθηκε orphan με όλο το τρέχον δέντρο (commit a810137) και **πουσαρίστηκε: origin/FEB272026
   ✅ (LFS 108MB)** — πρώτο remote backup της διπλωματικής.
   Υπόλοιπα (μικρά): `git worktree prune` (10 ορφανά)· ΜΗΝ γίνει ποτέ push το
   `FEB272026_localhistory` (θα ξανασκάσει στο 4.3GB)· εξέτασε `.gitignore` για μελλοντικά
   μεγάλα caches (πχ *.pkl >100MB)· main ref αποκαταστάσιμο στο d97a827 αν χρειαστεί.
2. **Δεν υπάρχει environment.yml** — το env `epf` έχει πάρει πακέτα (entsoe-py, matplotlib)
   χωρίς καταγραφή. `conda env export -n epf > environment.yml` όποτε βρεθεί ευκαιρία.
3. **Ξεχωριστό `.venv/`** στο root παράλληλα με το conda — πηγή σύγχυσης, δεν αγγίχτηκε.
4. **solar_fc_dayahead ύποπτο 2h shift** στο παλιό (2024-25) κομμάτι — high-priority data-quality
   έλεγχος (επηρεάζει resfc). Βλ. `ABLATION_PLAN.md §7.8`.
