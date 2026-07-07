---
name: energy-forecast
description: Leakage-free προβλέψεις τιμής/φορτίου GR (DAM/IDM/Forward) με το master pipeline του epf_greece_starter. Χρησιμοποίησέ το ΠΑΝΤΑ πριν από οποιοδήποτε training run, backtest, ablation, grid, ensemble, conformal/probabilistic δουλειά, προσθήκη νέου feature/πηγής δεδομένων (ENTSO-E fetch, parquet rebuild), ή όταν αναφέρονται MAE/headline/gate/leakage/retrain — ακόμα κι αν ο χρήστης δεν πει τη λέξη «forecast». Περιέχει τους σκληρούς κανόνες (no Optuna, strict gate, ΕΝΑ conda process), το pre-flight πρωτόκολλο leakage, και έτοιμα scripts ελέγχου.
---

# Energy Forecast Pipeline — Κανόνες & Εντολές

Πλήρη design docs: `MASTER_PIPELINE_DESIGN.md` (ερευνητικό/anti-leakage επίπεδο) και
`SYSTEM_DESIGN_TRADING_AGENT.md` (προϊοντική αρχιτεκτονική, data inventory, roadmap).

## Σκληροί κανόνες (μη διαπραγματεύσιμοι)

1. **ΟΧΙ Optuna** — απόφαση χρήστη. Diversity μέσω ensemble αντί για tuning.
2. **`--gate strict` είναι το default και το μόνο tradeable.** `academic` μόνο για σύγκριση με papers (Lago 2021 convention).
3. **`--strategy tf` = διαγνωστικό oracle** (actual lags) — ΠΟΤΕ ως αποτέλεσμα trading. Το engine τυπώνει warning.
4. ~~ΜΗΝ κάνεις full rebuild~~ **ΛΥΘΗΚΕ (2026-07-02)**: `data/raw/generation/` έχει πλέον πλήρες ιστορικό
   2015-2026 (κατέβηκε μέσω `src/fetch_entsoe_generation.py`). Rebuild με `python -m src.data --task
   price` / `--task load` είναι πλέον ασφαλές — επαληθεύτηκε ότι τα `gen_*`/`residual_load_*` του
   2015-2022 ταιριάζουν με το προηγούμενο (backed-up) parquet.
5. ~~Data gaps~~ **ΛΥΘΗΚΑΝ (2026-07-02)**: `entsoe_extra_hourly.parquet` (DA RES/gen fc) καλύπτει τώρα
   έως **2026-07-02** (`src/fetch_entsoe_dayahead.py`). Το `load_forecast_hourly.parquet` δεν είναι
   πια συνθετικό (παλιά: recursive πρόβλεψη δικού μας LGBM) — είναι το **πραγματικό** δημοσιευμένο
   day-ahead load forecast ΑΔΜΗΕ/ENTSO-E, ήδη κρυμμένο σε στήλη μέσα στα raw load CSVs, εξαγόμενο με
   `src/build_real_load_forecast.py` (καμία λήψη, μόνο parsing) — καλύπτει 2015-2026.
   ⚠️ **Τα Q1-2026 backtests πριν το fix (LEAR winning, "no-forecast" winning) έτρεξαν πάνω σε
   ελλιπή δεδομένα — ξανατρέξε τα για να δεις αν αλλάζουν τα συμπεράσματα με πλήρη coverage.**
6. Νέες πηγές δεδομένων μπαίνουν ΜΟΝΟ ως νέα ομάδα στο `src/feature_availability.py` με ρητό availability rule — ποτέ κατευθείαν στο engine.
7. Backtests ΔΕΝ σώζουν pkl (μόνο JSON/CSV). Τα παλιά pkl είναι άχρηστα μόλις υπάρξει το eval JSON.
8. **Retrain semantics**: με `--retrain monthly/weekly` ΜΗΝ περνάς `--train_end` — αγνοείται
   (expanding window έως κάθε cutoff). Το `--train_end` έχει νόημα ΜΟΝΟ με `--retrain static`.
   (Ιστορικό: πριν τη διόρθωση, monthly+train_end έκανε refit σε παγωμένα δεδομένα = no-op.)
9. **Ensemble εύρημα (ΟΡΙΣΤΙΚΟ, P3 2026-07-04)**: με τίμιο out-of-sample calibration ΚΑΝΕΝΑ
   ensemble (mean/median/weighted-by-1/MAE/cross-strategy) δεν κερδίζει το καλύτερο μεμονωμένο
   μοντέλο — μην το θεωρείς ΠΟΤΕ δεδομένη βελτίωση. Μόνη οριακή υπόσχεση: weekly LGBM+XGB
   (2 σχεδόν-ισοδύναμα μέλη, Δ=−0.148, PENDING). Βλ. `ABLATION_PLAN.md §5.5`.
10. **LEAR — ΑΝΑΘΕΩΡΗΜΕΝΟ (v2, 2026-07-02)**: `--algo lear` υποστηρίζεται πλήρως. Με τα ΠΛΗΡΗ
   δεδομένα το LEAR είναι σαφώς ΧΕΙΡΟΤΕΡΟ από τα δέντρα σε Q1 2026 (19.0 vs 15.2-15.8 €/MWh)
   ΚΑΙ σε full-2025 (19.6 vs 17.0-17.2). Το παλιό εύρημα «LEAR κερδίζει σε regime shift» ήταν
   κυρίως ARTIFACT των ελλιπών 2026 covariates (κενές forecast/load_fc στήλες δηλητηρίαζαν τα
   δέντρα)· με πλήρη δεδομένα η «κατάρρευση Φεβρουαρίου» των δέντρων εξαφανίστηκε (LGBM static
   Φεβ: 32.4 → 17.6). Ομοίως ΝΕΚΡΟ το εύρημα «no-forecast καλύτερο»: με γεμάτες στήλες το
   default feature set κερδίζει. Αξία LEAR πλέον: φθηνό robustness baseline / fallback μόνο.
11. ⚠️ **ΑΝΑΣΤΟΛΗ ΟΛΩΝ ΤΩΝ ΝΟΥΜΕΡΩΝ (2026-07-04 απόγευμα — TZFIX + engine leak)**:
   (α) Βρέθηκε & διορθώθηκε **timezone misalignment 4 ρολογιών** στο parquet (index=CET/CEST·
   gen ήταν +1h αργά, resfc 1-2h νωρίς, meteo 1h νωρίς καλοκαίρι) — `ABLATION_PLAN §5.9`,
   rebuild έγινε, guard στο preflight. Νέο baseline: default static Q1 = **19.17** (το 16.10
   ήταν στα στραβά δεδομένα· control confirmed). Στα ευθυγραμμισμένα δεδομένα το resfc ΒΛΑΠΤΕΙ
   το recursive-χειμώνα (~+2.4) → **feature selection ξαναγίνεται από την αρχή**.
   (β) ✅ **ΛΥΘΗΚΕ (2026-07-04) — AEL (SYSTEM_DESIGN §4.8)**: το crosslag leak
   (gen/load/residual actual lags χωρίς cutoff enforcement) διορθώθηκε με
   **freeze-at-cutoff** σε recursive rollout + direct row@cutoff + training rows +
   conformal quantile path. Anchor-based cutoff ανά block (forward 168h: ΕΝΑ cutoff
   για όλη την εβδομάδα). Νέο CLI flag: `--crosslag_mode {freeze,nan}` (default
   freeze· nan = sensitivity, μόνο lgbm/xgb). Poisoning self-test:
   `python -m src.check_crosslag_fairness` — PASS σε recursive-dam/direct-dam/
   recursive-forward (A leak=0.000000). Αυτόματα: `preflight_check.py --poison`.
   ⚠️ Q1 νούμερα ΠΡΙΝ το AEL (και το 19.17 baseline) κουβαλούν το leak — τα πρώτα
   leak-free Q1 specs: ✅ Β1 έκλεισε (runs/b1_leakfree/), Β3 re-ablation σε εξέλιξη (ABLATION §5.11· validity gate: last.md §2).
   Το παλιό headline (προ-fix): LGBM recursive **weekly**, `--features default` → **15.17 €/MWh**
   (seed std≈0.11, Μάρτιος ✓) — ΣΕ ΑΝΑΣΤΟΛΗ, μόνο ως μεθοδολογική αναφορά.
   ⛔ **Τα «14.43/15.02 με xborder» ΠΑΡΑΜΕΝΟΥΝ ΑΚΥΡΑ — ήταν leakage**: οι same-day τιμές
   γειτόνων βγαίνουν από το ΙΔΙΟ SDAC auction με το target (δημοσίευση ~13:00 D-1, ΜΕΤΑ το gate
   12:00). Η νόμιμη lagged εκδοχή (xb_*_lag24/48/168) ΒΛΑΠΤΕΙ τον χειμώνα σε 3/3 retrain
   cadences (static/monthly/weekly) — **ΑΠΟΡΡΙΦΘΗΚΕ οριστικά από το default set**. Μόνιμος
   κανόνας πλέον (πριν από κάθε νέο feature): γραπτή απάντηση «πότε ΑΚΡΙΒΩΣ δημοσιεύεται σε
   σχέση με το gate;» + cross-correlation lag-scan + hour-profile sanity — ΠΡΙΝ μπει σε τεστ.

   **✅ OVERNIGHT PROTOCOL (P1-P6) ΟΛΟΚΛΗΡΩΘΗΚΕ 2026-07-04** — πλήρης επανεξέταση όλων των
   ανοιχτών ζητημάτων σε καθαρά (χωρίς xborder) configs, βλ. `ABLATION_PLAN.md §5` για πλήρεις
   πίνακες (ιστορικό/ωμές §RESULTS ενότητες: `OLD/docs/ABLATION_PLAN_full_20260704.md`).
   Σύνοψη νέων ευρημάτων:
   - **xborder-lagged**: βλάπτει χειμώνα (3/3 cadences), βοηθάει καλοκαίρι (1 σημείο, PENDING) —
     εποχιακό interaction, ΟΧΙ universal feature. Δεν μπαίνει στο default.
   - **Cross-model confirmation**: resfc/genlags επιβεβαιώθηκαν σε XGB (summer) και MLP (Q1) —
     πλέον 3/3 αλγόριθμοι συμφωνούν. LEAR αντιδράει ΑΝΤΙΘΕΤΑ (L1 regularization υποβαθμίζει
     resfc/fuel) — μοντελο-εξαρτώμενο εύρημα, όχι artifact.
   - **Ensembles με τίμιο calibration**: weighted-by-1/MAE (weights από Δεκ, eval Ιαν-Φεβ) ΔΕΝ
     κερδίζει κανένα μεμονωμένο μοντέλο — ΑΠΟΡΡΙΦΘΗΚΕ. weekly LGBM+XGB (σχεδόν ισοδύναμα μέλη)
     δείχνει PENDING υπόσχεση (Δ=−0.148, ακριβώς στο noise floor).
   - **SS ΔΕΝ είναι redundant με retrain** (ΔΙΟΡΘΩΣΗ) — SS-linear βοηθάει και στο monthly
     (−0.243, καθαρό config) όπως και στο static (−0.266). Το παλιό «redundant» συμπέρασμα ήταν
     artifact μέτρησης πάνω σε xborder-contaminated config.
   - **Direct στρατηγική στο πλήρες Q1** (πρώτη φορά, πριν μόνο Δεκ-only): παραμένει σαφώς
     χειρότερο από recursive (19.5 vs 16.1). **Νέο στιβαρό εύρημα: meteo βοηθάει ΠΑΝΤΑ στο direct
     (4/4 σημεία: LGBM-Δεκ/Q1, XGB-Q1) και βλάπτει ΠΑΝΤΑ στο recursive** — καθαρό strategy-effect.
     Retrain cadence ΔΕΝ βοηθάει στο direct (αντίθετα με recursive). Cross-strategy ensemble
     (direct+recursive) ΑΠΟΡΡΙΦΘΗΚΕ, χάνει (+0.523). **Απόφαση: recursive παραμένει η επιλογή.**

   Οπτική αναφορά `reports/ablation_20260702/index.html` **ΞΑΝΑΠΑΡΑΧΘΗΚΕ 2026-07-04** με τα
   διορθωμένα νούμερα + πλήρη κάλυψη P1-P6.
   Πλήρης ετυμηγορία xborder: `ABLATION_PLAN.md §5.7` (post-mortem: `OLD/docs/`).
   **Επόμενο βήμα (`ABLATION_PLAN.md §8.2`)**: probabilistic/conformal layer πάνω στο κλειδωμένο
   15.17 config· δευτερεύοντα: solar_fc-2h-shift έλεγχος, xb_lag1_h0 (§8.1), SS×weekly, task=load.

12. **Δ=0.000 bit-for-bit σε ablation flag = RED FLAG, ΟΧΙ «no effect» εύρημα (2026-07-05,
   βρέθηκε στο LOAD Block C)**: αν η αφαίρεση μιας feature ομάδας δίνει ΑΚΡΙΒΩΣ ίδιο MAE
   και `#features` με το baseline σε πολλαπλές συνθήκες, η ομάδα πιθανότατα ήταν ήδη κενή
   στο φορτωμένο dataframe (π.χ. λείπει η στήλη από το συγκεκριμένο parquet/task), όχι ότι
   το feature «δεν βοηθάει». Case: `task=load` φορτώνει `hourly_load.parquet` (μέσω
   `split_utils._pick_processed_path`) που ΔΕΝ έχει `load_fc` (σε αντίθεση με το
   `hourly.parquet` του price) — το `-loadfc` arm ήταν VOID σε όλο το Block C. Πριν γραφτεί
   οποιοδήποτε «X δεν βοηθάει» εύρημα: έλεγξε ότι το `#features` ΑΛΛΑΖΕΙ ανάμεσα σε
   baseline/ablated config στο log.

## Περιβάλλον εκτέλεσης

- Πάντα: `conda run -n epf --no-capture-output python -X utf8 -m src.<module> ...`
- Από το project root: `C:\Users\aggel\OneDrive\Υπολογιστής\ALPHA\ECE\ΔΙΠΛΩΜΑΤΙΚΗ\epf_greece_starter`
- Windows: ΕΝΑ conda process τη φορά (όχι παράλληλα).
- Multi-line python: γράψε σε αρχείο και τρέξε το — το `conda run python -c` σπάει με newlines.
- **Workflow με χρήστη (2026-07-04)**: όταν ο χρήστης είναι online, δώσε του τις εντολές ως
  copy-paste block να τις τρέξει ΕΚΕΙΝΟΣ στο VS Code terminal (δεν δεσμεύεται το Claude
  παράθυρο)· background+notifications μόνο σε autonomous/overnight sessions. Στο τέλος κάθε
  απάντησης με ανοιχτή δουλειά: γράφε ορατά PENDING + στάδιο πλάνου + επόμενο ακριβές prompt.
- **Οργάνωση εξόδων (2026-07-04)**: πειραματικοί φάκελοι → `runs/` · CSV πίνακες → `results/` ·
  logs → `logs/` · αναφορές → `reports/` · αρχειοθετημένα → `OLD/`. Σε νέα τρεξίματα δίνε
  `--outdir runs/<όνομα>` και `--csv results/<όνομα>.csv` (τα defaults γράφουν στο root).
- **Long runs = detached (2026-07-05)**: κάθε run >2-3 λεπτά ΔΕΝ τρέχει μέσα από το Bash του
  Claude (αν πέσει το session/tokens, χάνεται η πρόσβαση — τα runs να μην εξαρτώνται από
  κανένα account). Detached εκκίνηση με log αρχείο:
  `Start-Process -WindowStyle Hidden -FilePath "C:\Program Files\Git\bin\bash.exe"
  -WorkingDirectory "<repo>" -ArgumentList '-c', 'bash scripts/<runner>.sh > logs/<x>.log 2>&1'`
  ⚠️ ΠΑΝΤΑ με `-WorkingDirectory` + ASCII-only ArgumentList — το ελληνικό path ΜΕΣΑ στο
  ArgumentList αλλοιώνεται (απέτυχε σιωπηλά 2026-07-05). Παρακολούθηση χωρίς tokens:
  `Get-Content logs\<x>.log -Wait -Tail 20` σε σκέτο terminal. Ο κανόνας «ΕΝΑ conda» ισχύει
  ανεξαρτήτως ποιος εκκίνησε το process (Claude, χρήστης, script) — μία ουρά, σειριακά.

## Scripts του skill (scripts/) — χρησιμοποίησέ τα, ΜΗΝ τα ξαναγράφεις ad-hoc

Το lag-scan ως ad-hoc script έπιασε 2 πραγματικά bugs (1h shift στο xborder fetcher, same-day
leakage)· γι' αυτό έγιναν μόνιμα εργαλεία εδώ. Τρέχουν σε δευτερόλεπτα (κανένα training):

```bash
# 1. ΠΡΙΝ από κάθε batch πειραμάτων — έλεγχος περιβάλλοντος/parquet (exit 1 σε FAIL):
conda run -n epf --no-capture-output python -X utf8 .claude/skills/energy-forecast/scripts/preflight_check.py
# --baseline: τυπώνει την εντολή αναπαραγωγής baseline (δεν την τρέχει)
# --poison:   τρέχει ΟΝΤΩΣ το cross-actuals poisoning (AEL §4.10.2) σε recursive+direct (~10-20s/strategy)

# 2. ΠΡΙΝ μπει νέο feature σε τεστ — leakage lag-scan + hour-profile (βήματα β+γ του §1):
conda run -n epf --no-capture-output python -X utf8 .claude/skills/energy-forecast/scripts/lagscan.py --col <στήλη>

# 3. Poisoning self-test AEL χειροκίνητα (π.χ. μετά από αλλαγή στο engine/AEL):
conda run -n epf --no-capture-output python -X utf8 -m src.check_crosslag_fairness \
  --task price --algo lgbm --strategy {recursive|direct} --market {dam|forward} --gate strict
# 3 tests: A unsafe-poison≈0 (leak-free) · B safe-control>0 (νομιμοποίηση) · C training-freeze exact
```

**Pre-flight checklist νέου feature (με αυτή τη σειρά, ΠΡΙΝ γραφτεί κώδικας feature):**
1. ✍️ Γραπτή απάντηση: «πότε ΑΚΡΙΒΩΣ δημοσιεύεται η πηγή σε σχέση με το gate 12:00 CET D-1;»
2. 🔍 `lagscan.py --col ...` — peak εκεί που προβλέπει η θεωρία· peak σε «βολικό» k ή |corr|>0.85
   = ύποπτο (ίδιο auction/μηχανισμός με το target;)
3. ☀️ Hour-of-day profile λογικό (το τυπώνει το lagscan — π.χ. solar peak ~12:00)
4. ▶️ `preflight_check.py` PASS πριν ξεκινήσει το batch
4α. ▶️ QA pre-run: ΝΕΟ/αλλαγμένο script → `python -X utf8 scripts/qa/check_run_config.py --script <path>` + agent `epf-code-reviewer` (PRE-RUN, με δηλωμένο σκοπό)

## Conformal / probabilistic layer — κανόνες σταδίου (2026-07-04)

Έρευνα δείχνει (probabl-ai/skills, DataRobot skills, MAPIE): ΚΑΜΙΑ έτοιμη λύση δεν καλύπτει
rolling time-series split-conformal με per-hour-of-day quantiles — υλοποίηση δική μας
(`src/conformal.py`, numpy quantiles πάνω σε residuals· καμία νέα βιβλιοθήκη).

1. **Model-agnostic by design**: το conformal στρώμα διαβάζει ΟΠΟΙΟΔΗΠΟΤΕ forecast JSON του
   masterscript — αξιολογείται σε ≥2 point μοντέλα (π.χ. LGBM weekly + XGB weekly), ΟΧΙ μόνο
   στο headline. Η ακαδημαϊκή σύγκριση προηγείται του προϊόντος.
2. **Calibration αυστηρά rolling**: residuals ΜΟΝΟ από παράθυρο 4-8 εβδομάδων ΠΡΙΝ από κάθε
   test ημέρα — ποτέ από το test set (αυτό είναι το conformal-αντίστοιχο του strict gate).
3. **Quantiles ανά hour-of-day** (24 κατανομές) — η διακύμανση της τιμής είναι έντονα ωριαία.
4. **Μετρικές πάντα μαζί**: pinball loss (α=0.1/0.5/0.9) + empirical coverage (στόχος ~80%
   για το p10-p90 band) + μέσο πλάτος band (sharpness). Coverage χωρίς sharpness = κενό claim.
5. **Baseline σύγκρισης**: quantile-LGBM (`objective="quantile"`, 3 μοντέλα) στο ΙΔΙΟ setup.
6. **Κριτήρια §2 του ABLATION_PLAN αναλογικά**: συμπέρασμα μόνο με συμφωνία σε ≥2 windows
   (Q1 + Μάρτιος 2026). Μελλοντική ακαδημαϊκή επέκταση: adaptive conformal (ACI) για regime shifts.

## Εντολές

### Μεμονωμένο config (`src/master_forecast.py`)

```bash
python -m src.master_forecast \
  --algo {lgbm,xgb,mlp,lstm} --task {price,load} \
  --market {dam,idm,forward,custom} [--delay D --horizon H --stride S] \
  --strategy {recursive,direct,seq2seq,tf} \
  --gate {strict,academic} --retrain {static,monthly,weekly} \
  --train_end "YYYY-MM-DD HH:00" \
  --test_start "YYYY-MM-DD HH:00" --test_end "YYYY-MM-DD HH:00" \
  --features default|all|"all,-meteo"|"lags,calendar" \
  [--crosslag_mode freeze|nan] [--out_json path] [--quiet]
```

`--crosslag_mode` (AEL §4.8): freeze=default/deployable (τελευταία γνωστή τιμή στο cutoff)·
nan=sensitivity variant μόνο για lgbm/xgb. Καταγράφεται στο out_json (crosslag_mode/crosslag_gap).

Presets: dam=(H24,s24) · idm=(H6,s3) · forward=(H168,s24).
Περιορισμοί: `seq2seq` μόνο με `--algo lstm` · `direct` όχι για mlp/lstm.

### Grid (`src/run_master_grid.py`)

```bash
python -m src.run_master_grid \
  --algos lgbm,xgb --tasks price,load --markets dam \
  --strategies recursive,direct --retrain static --gate strict \
  --train_end "2025-11-30 23:00" \
  --test_start "2025-12-01 00:00" --test_end "2026-02-28 23:00" \
  --outdir master_grid_out --csv results_master_grid.csv
```

Γράφει `results_master_grid.csv` (μία γραμμή/config) + ένα dashboard-συμβατό JSON ανά config, και τυπώνει MAE ranking ανά task.

### Ablation (`src/run_ablation.py`)

```bash
python -m src.run_ablation            # defaults: lgbm/price/dam/recursive/static, Δεκ 2025
python -m src.run_ablation --algo xgb --strategy direct
python -m src.run_ablation --specs "all;all,-fuel;default"   # specs χωρίζονται με ';'
```

Default specs: `all` / `all,-forecast` / `all,-meteo` / `lags,calendar` (1ο = baseline).
Γράφει `results_ablation.csv` + JSON ανά spec στο `ablation_out/`, τυπώνει πίνακα ΔMAE.
Ο Δεκ 2025 είναι το σωστό default window (πλήρη forecast/load_fc/meteo, μηδέν downloads).

### Ensemble (`src/make_ensemble.py`)

```bash
python -m src.make_ensemble \
    master_grid_out/lgbm_price_dam_recursive_static.json \
    master_grid_out/xgb_price_dam_recursive_static.json \
    [--out_json master_grid_out/ensemble_price_dam.json]
```

Δέχεται 2+ forecast JSONs ΙΔΙΟΥ test window, ξανα-υπολογίζει MAE όλων στο κοινό
παράθυρο (complete-case) και βγάζει `ensemble_mean`/`ensemble_median` + πίνακα.

## Ομάδες features (ablation flags) — λεπτόκοκκες από 2026-07-02

`calendar` · `lags` (y_lag 1/2/3/6/12/24/48/168) · `dense` (y_lag4..23) · `roll` ·
`resfc` (DA solar/wind/gen fc) · `loadfc` (load_fc) · `engfc` (resload_fc, on-the-fly) ·
`xborder` (xb_*_lag24/48/168 ΜΟΝΟ — same-day = leakage, δομικά αποκλεισμένο· ΕΚΤΟΣ default,
βλάπτει χειμώνα) · `meteo` · `fuel` ·
`genlags` (gen actuals + residual_load lags) · `loadlags` (load lags) · `other`.
Umbrellas (backward compat): `forecast`=resfc+loadfc · `crosslags`=genlags+loadlags+other.
Σύνταξη: `all,-meteo` · `default,-resfc` · `default,engfc` · `lags,calendar,genlags`.
engfc/xborder ΔΕΝ είναι στο default.

Νέα flags στο master_forecast: `--ss --ss_decay {linear,exp,step} --ss_rounds N`
(scheduled sampling, μόνο recursive — υλοποιημένο στο `src/scheduled_sampling.py`) και
`--n_estimators N` (capacity tests). Πλήρες πρόγραμμα πειραμάτων: `ABLATION_PLAN.md`.
Cross-border fetch: `python -m src.fetch_entsoe_xborder [--start YYYY-MM-DD]`.

## Data fetchers (κατέβασμα νέων δεδομένων)

```bash
python -m src.fetch_entsoe_generation --start 2015-01-01 [--end YYYY-MM-DD]  # actual gen/type → data/raw/generation/*.csv
python -m src.fetch_entsoe_dayahead [--start YYYY-MM-DD] [--end YYYY-MM-DD]  # DA RES/gen fc → entsoe_extra_hourly.parquet (συνέχεια από το τέλος αν χωρίς --start)
python -m src.build_real_load_forecast                                       # ΤΟΠΙΚΟ μόνο: πραγματικό load_fc από τα ήδη κατεβασμένα raw/load CSVs
```
Χρειάζονται env var `ENTSOE_API_KEY` (μόνο τα πρώτα δύο — το 3ο δεν κάνει download).
Μετά από οποιοδήποτε fetch: `python -m src.data --task price` και `--task load` για rebuild
(κάνε πρώτα backup του `data/processed/*.parquet` — trivial: `cp` σε `_backup_*` φάκελο — και σύγκρινε
gen-related columns ανά έτος πριν σβήσεις το backup, όπως έγινε στο 2026-07-02 fix).

**Windows/OneDrive caveat**: αν raw CSV read σκάει με `OSError: [Errno 22] Invalid argument`, ή αν
`attrib` δείχνει flag `O` σε ένα αρχείο, το OneDrive client δεν τρέχει (cloud-only placeholder, όχι
πραγματικά κατεβασμένο τοπικά) — ξεκίνα το `"$env:PROGRAMFILES\Microsoft OneDrive\OneDrive.exe"`.

## Γρήγορη ερμηνεία gate (γιατί price ≠ load)

- **PRICE @ DAM**: οι τιμές της D-1 είναι ΟΛΕΣ γνωστές στο D-1 12:00 (auction D-2) → anchor 23:00 D-1, gap=0.
- **LOAD @ DAM strict**: actual load γνωστό μόνο ως ~11:00 D-1 → gap=12h, καλύπτεται recursive ή με load_fc.
