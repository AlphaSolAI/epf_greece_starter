# SYSTEM DESIGN — GR Energy Forecast Agent (προϊόν) — v1.1 (2026-07-04)

> **v1.1 review update**: μετά τα ευρήματα της 2026-07-04 (timezone misalignment 4 ρολογιών —
> διορθώθηκε· crosslag engine leakage — ανοιχτό) προστέθηκαν τα §4.8 (Availability Enforcement
> Layer), §4.9 (Time Canonicalization Contract), §4.10 (Validation Harness). Αυτά τα τρία
> είναι πλέον ΤΟ θεμέλιο του συστήματος: κανένα αποτέλεσμα δεν δημοσιεύεται/πωλείται αν δεν
> περνά τα §4.10 invariants. Βλ. `ABLATION_PLAN.md §5.9-§5.10` για το post-mortem.

> **Στόχος**: αναβάθμιση του υπάρχοντος συστήματος σε ανταγωνιστικό, deployable προϊόν πρόβλεψης
> τιμής (EPF) & φορτίου (STLF) για την ελληνική αγορά — DAM / Intraday / Forward (≤1 εβδομάδα) —
> τέτοιο που έμποροι ενέργειας, ΑΔΜΗΕ, παραγωγοί θα πλήρωναν συνδρομή για να το χρησιμοποιούν.
>
> **Σχέση με [MASTER_PIPELINE_DESIGN.md](MASTER_PIPELINE_DESIGN.md)**: εκείνο ορίζει το
> *ερευνητικό* επίπεδο (information clock, gate closure, στρατηγικές, ablations, εκτιμήσεις χρόνου).
> Αυτό εδώ ορίζει το *προϊοντικό* επίπεδο: αρχιτεκτονική, καθημερινή λειτουργία, αξιοπιστία,
> παράδοση, monitoring. Το ένα δεν αντικαθιστά το άλλο.

---

## 1. Requirements

### 1.1 Functional (τι κάνει)

| ID | Απαίτηση | Κατάσταση |
|---|---|---|
| FR1 | **DAM τιμή**: 24 ωριαίες προβλέψεις για την ημέρα D, έτοιμες **πριν** το gate closure D-1 12:00 CET, με πληροφορία αυστηρά ≤ cutoff | ✅ engine έτοιμο (`master_forecast --market dam --gate strict`) |
| FR2 | **DAM φορτίο**: ίδιο καθεστώς, με σωστό anchor (actual load γνωστό μόνο ως ~11:00 D-1 → gap 12h) | ✅ engine έτοιμο |
| FR3 | **Intraday**: rolling προβλέψεις ορίζοντα ~6h, re-anchor ανά 3h με φρέσκα actuals | ✅ preset `idm` (horizon=6, stride=3) |
| FR4 | **Forward**: 168h ορίζοντας για εβδομαδιαίο hedging, re-anchor ημερησίως | ✅ preset `forward` |
| FR5 | **Backtesting** με αυστηρό information clock, ανά στρατηγική/αγορά/retrain | ✅ `master_forecast` + `run_master_grid` |
| FR6 | **Ablations** ανά ομάδα feature (meteo/forecast/dense/fuel/…) | ✅ `--features` spec |
| FR7 | **Retrain scheduling**: static / monthly / weekly + retrain-on-drift | ✅ τα 3 πρώτα · ⬜ drift-trigger |
| FR8 | **Probabilistic έξοδος**: p10/p50/p90 ανά ώρα (risk bands για bidding) | ◐ **υλοποιημένο** (`src/conformal.py`: split-conformal + quantile-LGBM, causal rolling calibration) — ΔΕΝ έχει τρέξει έγκυρα ακόμα: μπλοκάρεται από FR13 |
| FR13 | **Availability Enforcement Layer (AEL)**: ΚΑΘΕ feature family σέβεται το δικό της information cutoff σε train ΚΑΙ eval (όχι μόνο τα y-lags) | ⬜ **ΝΕΟ — ΚΡΙΣΙΜΟ (§4.8)**: το recursive rollout αντικαθιστά μόνο y-lags· gen/load/residual lags διαρρέουν actuals |
| FR14 | **Time canonicalization**: κάθε πηγή δηλώνει ρολόι, μετατροπή ΜΟΝΟ στο data.py, εμπειρικοί guards | ✅ **ΝΕΟ (§4.9)** — υλοποιήθηκε 2026-07-04 (TZFIX + preflight guard) |
| FR9 | **Ensemble** πάνω από τα ατομικά μοντέλα | ⬜ δεν υπάρχει — φθηνότερο κέρδος ακρίβειας |
| FR10 | **Καθημερινή αυτόματη έκδοση** πρόβλεψης (batch job) + fallback ώστε να μη χάνεται ποτέ gate | ⬜ δεν υπάρχει |
| FR11 | **Settlement loop**: μόλις έρθουν actuals → σκοράρισμα, rolling MAE, champion/challenger | ⬜ δεν υπάρχει |
| FR12 | **Παράδοση**: dashboard + μηχαναγνώσιμο JSON (αργότερα API) | ◐ dashboard υπάρχει, όχι ως service |

### 1.2 Non-functional

- **Deadline-driven, όχι latency-driven**: batch σύστημα. Όλα τα DAM προϊόντα έτοιμα ≤ **D-1 10:30 CET** (buffer 1.5h πριν το gate). Compute budget: λεπτά, όχι ms.
- **Availability = "never miss the gate"**: πρόβλεψη εκδίδεται **κάθε** μέρα, ακόμα κι αν λείπουν φρέσκα data ή σκάσει μοντέλο (fallback chain, §4.5). Ένα χαμένο gate = χαμένος πελάτης.
- **Auditability**: κάθε πρόβλεψη συνοδεύεται από: model version, training window, features hash, data freshness ανά πηγή. (Απαραίτητο για θεσμικούς πελάτες τύπου ΑΔΜΗΕ.)
- **Reproducibility**: ίδιο config → ίδιο αποτέλεσμα (seeds, versioned data snapshots).
- **Accuracy στόχος**: να κερδάμε *σταθερά* (α) naive D-7/D-1 copy, (β) LEAR baseline, στο strict gate. Το GR σύστημα είναι πιο volatile από DE/ES — τα apples-to-apples νούμερα κρίνονται μόνο εντός strict.
- **Κόστος**: 1 workstation τώρα → μικρό VM αργότερα. Καμία GPU απαραίτητη (trees + μικρά NN).

### 1.3 Constraints

- Solo developer + Claude · Windows / conda env `epf` · sequential εκτέλεση (όχι πολλαπλά conda ταυτόχρονα).
- **Όχι Optuna** (απόφαση χρήστη — χρονοβόρο χωρίς ουσιαστικό όφελος). Αντ' αυτού: σταθερά λογικά hyperparams + **diversity μέσω ensemble** (η βιβλιογραφία στηρίζει ότι αποδίδει περισσότερο από per-model tuning).
- ~~Data gaps~~ **ΛΥΘΗΚΑΝ (2026-07-02)**: generation/DA-forecasts/load_fc πλήρη 2015-2026 — βλ. §3. Rebuild parquet πλέον ασφαλές (πάντα με backup+σύγκριση).
- Οι υπάρχουσες σειρές είναι ωριαίες. Δομικός κίνδυνος: η μετάβαση του SDAC σε **15-λεπτο MTU** — όταν/όπως ισχύσει πλήρως για το ελληνικό DAM, θέλει re-architecture σε τεταρτοωριαία ανάλυση (§7).

---

## 2. High-level architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│ L0 INGESTION (fetchers, append-only raw)                                 │
│   ENTSO-E: DAM prices GR · actual load · actual gen · DA load/RES fc     │
│   HEnEx premarket · Open-Meteo (weather FORECAST archive + reanalysis)   │
│   TTF gas / EUA CO2 settlements                                          │
└───────────────┬──────────────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ L1 FEATURE STORE (data.py → processed/*.parquet)                         │
│   hourly.parquet (price) · hourly_load.parquet (load) + βοηθητικά        │
│   feature_availability.py = ΤΟ ΣΥΜΒΟΛΑΙΟ διαθεσιμότητας                  │
│   (ομάδες: calendar/lags/dense/roll/forecast/meteo/fuel/crosslags        │
│    + GateSpec: strict/academic ανά task/market)                          │
└───────────────┬──────────────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ L2 MODEL LAYER (master_forecast.py — ενιαία μηχανή για ΟΛΟΥΣ)            │
│   algos: LGBM · XGB · MLP · LSTM(seq2seq=true MIMO) · LEAR(baseline)     │
│   strategies: recursive | direct | seq2seq   (tf = diagnostic ΜΟΝΟ)      │
│   retrain: static/monthly/weekly (+drift-trigger)                        │
│   → MODEL REGISTRY: versioned artifacts + meta, auto-prune (§4.2)        │
└───────────────┬──────────────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ L3 FORECAST SERVICE (daily batch — ⬜ νέο)                                │
│   products/day: DAM price D · DAM load D · IDM rolling · Forward W       │
│   ENSEMBLE (mean/median των healthy μοντέλων) → point                    │
│   CONFORMAL/QUANTILE layer → p10/p50/p90                                 │
│   FALLBACK CHAIN → ποτέ χαμένο gate                                      │
│   → forecasts/YYYY-MM-DD/{product}.json                                  │
└───────────────┬──────────────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ L4 DELIVERY: dashboard.html (τώρα) → static JSON feed → FastAPI (προϊόν) │
└───────────────┬──────────────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ L5 EVAL / MONITORING (settle job — ⬜ νέο)                                │
│   actuals έρχονται → score χθεσινών προβλέψεων → rolling MAE ανά μοντέλο │
│   champion/challenger πίνακας · drift alert · retrain trigger προς L2    │
└──────────────────────────────────────────────────────────────────────────┘
```

### Λειτουργικό ρολόι μιας ημέρας (DAM προϊόν)

| Ώρα (CET, D-1) | Βήμα |
|---|---|
| 06:00–08:30 | L0 fetch: χθεσινές τιμές DAM (γνωστές από D-2 auction), actual load/gen ως τώρα, weather forecast για D, ENTSO-E DA forecasts για D (ό,τι έχει δημοσιευτεί ≤ τώρα — **timestamp check, όχι υπόθεση**) |
| 08:30–09:00 | L1 incremental update parquet + freshness report ανά πηγή |
| 09:00–10:00 | L2/L3: inference όλων των healthy μοντέλων → ensemble → quantiles → fallback αν χρειαστεί |
| 10:00–10:30 | L4: δημοσίευση JSON + dashboard · sanity gates (bounds, ramp checks) |
| **12:00** | **DAM gate closure** (οι πελάτες έχουν ήδη υποβάλει με το προϊόν μας) |
| D+1 πρωί | L5 settle: actuals D → score, rolling metrics, alerts |

---

## 3. Data inventory — «αξιοποιούνται όλα τα αρχεία;»

| Πηγή / αρχείο | Κάλυψη | Χρησιμοποιείται; | Gap / ενέργεια |
|---|---|---|---|
| `data/raw/dam_prices/` (12 CSV) | έως τέλος 2026 | ✅ κύριο target price | — |
| `data/raw/load/` (12 CSV) | έως τέλος 2026 | ✅ κύριο target load | — |
| `data/raw/generation/` | **✅ ΛΥΘΗΚΕ (2026-07-02): 2015-2026** | `src/fetch_entsoe_generation.py` (ENTSO-E API) | Επαληθεύτηκε byte-for-byte συνέπεια με το προηγούμενο cached parquet για 2018-2022 πριν διαγραφεί το backup |
| `processed/hourly.parquet` (21 MB) | πλήρες, 2017-2026 | ✅ price feature store | — |
| `processed/hourly_load.parquet` (24 MB) | πλήρες, 2015-2026 | ✅ load feature store | — |
| `processed/entsoe_extra_hourly.parquet` (1.2 MB) | **✅ ΛΥΘΗΚΕ: έως 2026-07-02** | ✅ ομάδα `forecast` | `src/fetch_entsoe_dayahead.py` — τρέξε ξανά χωρίς `--start` για να συνεχίσει από το τέλος |
| `processed/load_forecast_hourly.parquet` (1.1 MB) | **✅ ΛΥΘΗΚΕ: 2015-2026, ΠΡΑΓΜΑΤΙΚΟ** | ✅ `load_fc` | Δεν είναι πια συνθετικό (δικό μας μοντέλο) — parsed από τη στήλη "Day-ahead Total Load Forecast" που ήδη υπήρχε στα raw CSVs. `src/build_real_load_forecast.py`, καμία λήψη. |
| `processed/weather_gr_hourly.parquet` (6.3 MB) | πλήρες | ✅ ομάδα `meteo` (42 στήλες) | ⚠️ Είναι **reanalysis** (perfect-forecast proxy) — αποδεκτό στη βιβλιογραφία, ελαφρώς αισιόδοξο. Για προϊόν: μετάβαση σε Open-Meteo **historical forecast** archive (ίδιο API, έντιμη αξιολόγηση). |
| `processed/henex_premarket_hourly.parquet` (601 KB) | — | ◐ **υποαξιοποιημένο** | Περιέχει HEnEx pre-market πληροφορία — υποψήφιο ισχυρό feature για DAM· να μπει ως ξεχωριστή ομάδα `premarket` με δικό της availability rule. |
| Gas/CO2 (`_fetch_gas_history.py`) | πλήρες | ✅ ομάδα `fuel` | — |
| `processed/xborder_hourly.parquet` (BG/IT-SUD DAM) | 2017-2026 | ◐ ομάδα `xborder` (lagged ΜΟΝΟ, **εκτός default**) | ⛔ **ΔΙΟΡΘΩΣΗ 2026-07-04**: το αρχικό «−0.7 όφελος» ήταν **leakage** (same-day τιμές = ίδιο SDAC auction με το target). Δομικό fix στο `data.py`: μόνο `xb_*_lag{24,48,168}` μπαίνουν στο parquet. Το νόμιμο lagged xborder ΒΛΑΠΤΕΙ τον χειμώνα (3/3 cadences) → εκτός default· βοηθάει το καλοκαίρι (1 σημείο, PENDING). Βλ. `ABLATION_PLAN.md §5.7`. |
| **Ακόμα καθόλου** | — | — | gas forward curve (όχι μόνο spot), υδραυλικά αποθέματα/reservoir levels. Χαμηλή προτεραιότητα — δεν έχουν προταθεί από κανένα ablation finding ακόμα. |

**Απάντηση (ενημερωμένη 2026-07-04)**: το `henex_premarket` παραμένει υποαξιοποιημένο (καμία δουλειά πάνω του ακόμα). Όλα τα forecast/generation κενά που καταγράφηκαν αρχικά **έχουν κλείσει**. Πλήρη ablation ευρήματα ανά ομάδα: `ABLATION_PLAN.md §5`, οπτική σύνοψη: `reports/ablation_20260702/index.html`.

---

## 4. Deep dives

### 4.1 Feature store & availability contract (υπάρχει)

Το `feature_availability.py` είναι το **μοναδικό σημείο αλήθειας** για το τι επιτρέπεται πότε.
Κανόνας επέκτασης: **κάθε νέα πηγή** (π.χ. premarket, γείτονες) μπαίνει ΜΟΝΟ ως νέα ομάδα εδώ,
με ρητό availability rule — ποτέ κατευθείαν στο engine. Έτσι το anti-leakage μένει αποδείξιμο.

### 4.2 Model registry & αποθήκευση (λύνει και το «τεράστια pkl»)

```
models_v2/{algo}_{task}_{market}_{strategy}/
    {train_key}/model.pkl        # train_key: "static" | "2026-01" | "2026-W05"
    {train_key}/meta.json        # features hash, train window, git commit, seed, metrics@val
```

- **Retention policy**: κρατάμε μόνο (α) τον τρέχοντα champion, (β) τα Ν=2 τελευταία train_keys ανά config. Auto-prune στο τέλος κάθε retrain. Τα παλιά pkl **δεν χρειάζονται** αφού υπάρχει το eval JSON — μόνο το meta.json κρατιέται για ιστορικό (KB, όχι MB).
- Οι backtests (run_master_grid) **δεν** σώζουν pkl καθόλου — μόνο JSON/CSV αποτελέσματα. Pkl σώζει μόνο το L3 (production inference).

### 4.3 Ensemble — δοκιμάστηκε εμπειρικά, ΔΕΝ επιβεβαιώθηκε ακόμα (⚠️ αναθεωρημένο)

- **Αρχική υπόθεση** (από βιβλιογραφία, Lago et al. 2021): ο απλός ισοβαρής μέσος όρος
  (mean/median) ανόμοιων μοντέλων κερδίζει σταθερά κάθε μεμονωμένο μοντέλο.
- **Εμπειρικός έλεγχος στα δικά μας δεδομένα (LGBM+XGB+MLP+LEAR, DAM price, recursive, static)**:
  **ΔΕΝ επιβεβαιώθηκε.** Σε 3 ελέγχους (Q1 2026 4-μελές, Δεκ-2025-μόνο 4-μελές, Δεκ-2025-μόνο
  2-μελές LGBM+XGB) ο ισοβαρής μέσος **έχανε** από το καλύτερο μεμονωμένο μοντέλο (συνήθως LGBM).
  Αιτία: όταν ένα μέλος είναι σαφώς καλύτερο (εδώ LGBM 14.24 έναντι MLP 20.62 €/MWh στο ίδιο test),
  ο ισοβαρής μέσος με τα αδύναμα μέλη τραβάει το αποτέλεσμα προς τα κάτω αντί να βοηθάει. Η
  βιβλιογραφική συνταγή προϋποθέτει μέλη **συγκρίσιμης ποιότητας** — δεν ισχύει αυτόματα εδώ.
- **Επόμενο βήμα πριν το κρίνουμε προϊοντικό feature**: weighted average (βάρος ∝ 1/MAE) αντί
  ισοβαρούς — «τιμωρεί» αυτόματα τα αδύναμα μέλη. Ή ensemble μόνο μεταξύ μελών με κοντινό MAE
  (π.χ. LGBM+XGB, αποκλείοντας MLP/LEAR όταν υστερούν καθαρά). Χρειάζεται έλεγχος σε >1 μήνα/task
  πριν βγει τελικό συμπέρασμα — προς το παρόν ΜΗΝ θεωρείται δεδομένο ότι το ensemble βελτιώνει.
- **ΤΕΛΙΚΟ Update 2026-07-04** (`ABLATION_PLAN.md §5.5`, P3/P4 του overnight protocol):
  **Weighted-by-1/MAE ΔΟΚΙΜΑΣΤΗΚΕ με τίμιο out-of-sample calibration** (βάρη από Δεκ, eval
  Ιαν-Φεβ) → **χάνει και από mean και από median και από το σκέτο LGBM — ΑΠΟΡΡΙΦΘΗΚΕ, κλειστό.**
  Στο ίδιο honest split ούτε ο median κερδίζει (το παλιό «median οριακά κερδίζει» ήταν στο
  ευκολότερο πλήρες window, κάτω από noise floor). Cross-strategy ensemble (direct+recursive)
  επίσης ΑΠΟΡΡΙΦΘΗΚΕ (+0.52 χειρότερο, ανόμοια ποιότητα μελών). Το ΜΟΝΟ ελπιδοφόρο: **weekly
  LGBM+XGB** (2 σχεδόν-ισοδύναμα μέλη) 15.023 vs 15.171 (Δ=−0.148, στο όριο του noise floor) —
  PENDING, θέλει 2ο window/seeds. Συμπέρασμα προϊόντος: το ensemble ΔΕΝ είναι δεδομένο κέρδος·
  headline = μεμονωμένο LGBM.
- **Πώς** (υλοποιημένο, `src/make_ensemble.py`): διαβάζει per-model JSON, ευθυγραμμίζει timestamps
  στο κοινό complete-case παράθυρο, βγάζει `ensemble_mean`/`ensemble_median` + πίνακα σύγκρισης.
  Health-gate (εξαίρεση μέλους με rolling MAE > 1.5× του median) παραμένει ⬜ ανοιχτό.

### 4.4 Probabilistic layer — product differentiator

Ένας trader δεν θέλει μόνο p50 — θέλει «πόσο σίγουρο είναι» για να διαστασιολογήσει θέση. Δύο δρόμοι, συμπληρωματικοί:

1. **Quantile LGBM**: ίδιο feature set, `objective="quantile"`, α∈{0.1,0.5,0.9} → 3 μοντέλα. Απλό, native.
2. **Split-conformal** πάνω από **οποιοδήποτε** point μοντέλο (και το ensemble): residuals από
   calibration window (τελευταίες ~4–8 εβδομάδες), quantiles ανά hour-of-day → bands με εγγύηση
   κάλυψης, **χωρίς κανένα retrain**. Προτεινόμενο default για το προϊόν.

Αξιολόγηση: pinball loss + empirical coverage (π.χ. το 80% band να πιάνει ~80%). Μπαίνει στο settle job.

### 4.5 Fallback chain — never miss the gate

Σειρά προτίμησης ανά προϊόν, το L3 κατεβαίνει μέχρι να βρει υγιή έξοδο:

1. **Ensemble** όλων των healthy μοντέλων (full features).
2. Ensemble με **degraded feature set**: αν πηγή δεν είναι φρέσκια (timestamp check), χρησιμοποιείται
   variant μοντέλου εκπαιδευμένο **χωρίς** εκείνη την ομάδα (τα ablation variants ΕΙΝΑΙ τα degraded
   μοντέλα — το `--features` μηχάνημα ήδη το επιτρέπει· απλώς κρατάμε 2–3 variants στο registry).
3. **Μοναδικό champion μοντέλο** από το registry (τελευταίο γνωστό καλό).
4. **Naive seasonal**: y(D) = y(D-7) με διόρθωση επιπέδου από D-1 — υπολογίζεται πάντα, από τίποτα.

Κάθε επίπεδο σημειώνεται στο JSON (`fallback_level: 0..3`) ώστε ο πελάτης να ξέρει τι ποιότητας πρόβλεψη πήρε.

### 4.6 Delivery — API contract

Τώρα: static JSON ανά ημέρα/προϊόν (το dashboard τα διαβάζει). Προϊόν: FastAPI read-only πάνω από τα ίδια αρχεία.

```
GET /v1/forecast/dam-price?date=2026-03-15
{
  "product": "dam_price", "zone": "GR", "unit": "EUR/MWh",
  "issue_time": "2026-03-14T09:55:00+01:00",
  "cutoff":     "2026-03-14T08:00:00+01:00",     // πραγματικό info cutoff
  "gate":       "strict", "fallback_level": 0,
  "model": {"version": "ens-2026-03", "members": ["lgbm","xgb","mlp","lear"]},
  "data_freshness": {"entsoe_load": "2026-03-14T07:00", "weather_fc": "...", ...},
  "series": [ {"ts": "2026-03-15T00:00+01:00", "p10": 61.2, "p50": 74.5, "p90": 96.1}, ... 24 ]
}
```

Ίδιο σχήμα για `dam-load`, `idm-price`, `forward-price` (168 σημεία).

### 4.7 Settlement / monitoring loop

Καθημερινό `settle` job (L5):
- Διαβάζει actuals που ήρθαν → σκοράρει κάθε χθεσινό forecast JSON (MAE/RMSE/pinball/coverage).
- Ενημερώνει `metrics_history.csv` (μία γραμμή ανά μοντέλο×ημέρα) → dashboard panel «rolling 7/30-day MAE».
- **Drift rule**: rolling 7d MAE μοντέλου > 1.5× του δικού του 90d baseline → σημαία retrain (L2) + εξαίρεση από ensemble μέχρι να αναρρώσει.
- **Champion/challenger**: νέο retrain γίνεται champion ΜΟΝΟ αν κερδίζει τον τρέχοντα σε 2 εβδομάδες shadow scoring.

### 4.8 Availability Enforcement Layer (AEL) — ⬜ ΚΡΙΣΙΜΟ ΝΕΟ (v1.1)

**Πρόβλημα που λύνει (βρέθηκε 2026-07-04)**: το information-availability contract υπήρχε στα
docs (MASTER §2) αλλά υλοποιούνταν ΜΟΝΟ για y-lags/y-rolls. Τα gen/load/residual actual lags
διαβάζονται αυτούσια από το prebuilt dataframe → στο eval η ώρα 14:00 της D «βλέπει» actual
παραγωγή 13:00 της D (recursive), και το direct@cutoff 23:00 D-1 «βλέπει» actuals 12:00-22:00
D-1 — όλα αδημοσίευτα στο gate 12:00 D-1.

**Σχεδίαση**: πίνακας cutoff ΑΝΑ FEATURE FAMILY (όχι μόνο ανά task), στο `feature_availability.py`:

| Family | Cutoff στο DAM strict (gate 12:00 D-1) | Enforcement |
|---|---|---|
| y (price) lags/rolls | 23:00 D-1 | ✅ ήδη (running substitution) |
| y (load) lags/rolls | 11:00 D-1 | ✅ ήδη (gap=12) |
| gen_*/residual_load/load actual lags | **11:00 D-1** | ⬜ freeze-at-cutoff (default) ή NaN |
| resfc/loadfc (D-1-published fc) | όλος ο ορίζοντας | ✅ by construction |
| meteo | όλος ο ορίζοντας (fc proxy) | ✅ by construction |
| fuel (D-1 settlement) | lag≥1 ημέρα | ✅ by construction |
| xborder lagged | lag≥24h | ✅ by construction (same-day δομικά εκτός parquet) |

**Μηχανισμός**: στο row-build (recursive: κάθε t· direct: row@cutoff· ΚΑΙ στο training row
construction — ίδιο σχήμα, αλλιώς train/serve mismatch): για στήλη οικογένειας F με lag k,
αν `t−k > cutoff_F(anchor)` → αντικατάσταση με **freeze-at-cutoff** (τιμή στο cutoff_F —
deployable: «τελευταία γνωστή τιμή») ή NaN (trees, sensitivity variant). Το `tf` μένει ρητά
oracle (εξαίρεση με warning). SS: παραμένει y-only (τα frozen cross-lags δεν έχουν rollout
noise). **Κανένα νέο πείραμα δεν μετράει πριν μπει το AEL + περάσει το poisoning test (§4.10).**

### 4.9 Time Canonicalization Contract — ✅ (v1.1, υλοποιημένο 2026-07-04)

Κανονικό frame ΟΛΟΥ του pipeline: **CET/CEST-naive** (το frame των price/load exports, ταυτίζεται
με το gate 12:00 CET και τη βιβλιογραφία). Κανόνες:
1. Κάθε πηγή έχει ΔΗΛΩΜΕΝΟ ρολόι (generation: Europe/Athens · entsoe_extra fc: UTC ·
   weather: σταθερό UTC+1 · price/load/load_fc: CET/CEST · fuel: ημερήσιο).
2. Μετατροπή ΜΟΝΟ στους loaders του `data.py` (helpers `_utc_to_cet_naive_index`,
   `_fixed_utc1_to_cet_naive_index`, gen −1h) — ποτέ στο engine, ποτέ ad-hoc.
3. Εμπειρικοί guards: preflight (solar_fc DJF peak=11:00), `solar_shift_check.py`
   (corr fc↔gen actual: peak_k=0 ανά έτος, ≥0.95). Κάθε νέος fetcher/rebuild ξαναπερνά.
4. ⚠️ Εκκρεμότητα συνέπειας: `fetch_weather_2026.py` ζητά timezone=UTC ενώ το parquet είναι
   UTC+1 — να ευθυγραμμιστεί πριν το επόμενο weather append (ο guard θα το πιάσει ούτως ή άλλως).

### 4.10 Validation Harness — poisoning tests ως μόνιμη πύλη (v1.1)

Invariants που πρέπει να ισχύουν ΠΡΙΝ δημοσιευτεί οποιοδήποτε νούμερο:
1. **Poisoning y**: αντικατάσταση των actual y εντός κάθε scored block με τρελές τιμές →
   προβλέψεις ΑΜΕΤΑΒΛΗΤΕΣ (υπάρχει ήδη: `check_openloop_fairness.py`).
2. **Poisoning cross actuals (ΝΕΟ)**: αντικατάσταση gen/load/residual actuals ΜΕΤΑ το
   cutoff_F κάθε anchor → προβλέψεις ΑΜΕΤΑΒΛΗΤΕΣ. Τρέχει σε recursive ΚΑΙ direct.
3. **TZ guards** (§4.9.3) σε κάθε preflight.
4. **Reproducibility anchor**: γνωστό config αναπαράγει το καταγεγραμμένο MAE ±0.05
   (preflight --baseline).
5. **Control-run πρωτόκολλο**: κάθε δομική αλλαγή δεδομένων/engine συνοδεύεται από control
   run στο πριν-state (backup swap) ώστε το Δ να αποδίδεται αποκλειστικά στην αλλαγή.

---

## 5. Reliability & scaling

- **Φόρτος**: 4 προϊόντα/ημέρα × ~5 μοντέλα inference = λεπτά/ημέρα σε 1 μηχάνημα. Το βαρύ κομμάτι (backtests, retrains) είναι offline και μπαίνει βράδυ/σαββατοκύριακο. Δεν χρειάζεται distributed τίποτα.
- **Scheduling**: Windows Task Scheduler τώρα · cron σε φθηνό Linux VM όταν γίνει προϊόν (το workstation-κλειστό-στις-09:00 είναι ο #1 SLA κίνδυνος).
- **Idempotency**: κάθε job ξανατρέχει ακίνδυνα (per-day artifacts, atomic write σε temp→rename).
- **Data outage playbook**: ENTSO-E down → χθεσινά DA forecasts + degraded variant + σημείωση freshness. Open-Meteo down → forecast της προηγούμενης έκδοσης (persisted). Όλα down → fallback level 3, ποτέ σιωπή.
- **Alerting**: το settle job στέλνει notification όταν (α) χάθηκε deadline, (β) fallback_level ≥ 2, (γ) drift alert.

---

## 6. Trade-offs (ρητά)

| Απόφαση | Επιλογή | Κόστος | Γιατί αξίζει |
|---|---|---|---|
| Στρατηγική DAM | **Recursive** (⚠️ ΑΝΑΘΕΩΡΗΘΗΚΕ 2026-07-04 — το αρχικό σκεπτικό υπέρ direct ΔΙΑΨΕΥΣΤΗΚΕ εμπειρικά) | Rollout error accumulation στα μακρινά offsets (μετριέται· το SS το μετριάζει) | P4 test στο ΠΛΗΡΕΣ Q1: direct 19.5 vs recursive 16.1 €/MWh — το direct υποφέρει δυσανάλογα σε μεγάλα/ασταθή windows και το retrain ΔΕΝ το βοηθάει. Το «χωρίς error accumulation» δεν αρκεί να αντισταθμίσει την απώλεια των y-lags. Βλ. `ABLATION_PLAN.md §5.6`. |
| Tuning | **Όχι Optuna** — σταθερά params + ensemble diversity | Ίσως 1–3% ακρίβειας ανά μεμονωμένο μοντέλο | Το ensemble ανακτά περισσότερο απ' όσο χάνει το μη-tuning· τεράστια εξοικονόμηση χρόνου· απόφαση χρήστη βάσει εμπειρίας |
| Trees vs DL/linear | Trees πρώτο βιολί, LEAR ως robustness fallback ΜΟΝΟ, DL (LSTM) ανοιχτό | Λιγότερο «εντυπωσιακό» | **Αναθεωρήθηκε 2026-07-02**: με πλήρη v2 δεδομένα το LEAR είναι σαφώς χειρότερο (19.5 vs 14.4-16.1 €/MWh, `ABLATION_PLAN.md §Βήμα 9`) — η αρχική του «υπεροχή» ήταν artifact ελλιπών δεδομένων, όχι εγγενές πλεονέκτημα extrapolation. LSTM seq2seq τρέχει end-to-end αλλά με calibration bug (§9 honesty section) — δεν είναι tradeable ακόμα. |
| Weather | Reanalysis τώρα → forecast archive αργότερα | Τα τωρινά νούμερα ελαφρώς αισιόδοξα | Αποδεκτό ακαδημαϊκά· για το προϊόν όμως η μετάβαση είναι υποχρεωτική (έντιμο MAE) |
| Αποθήκευση | Parquet + JSON + CSV registry, όχι DB | Όχι SQL queries | Μηδέν operational βάρος· τα volumes είναι MB, όχι GB |
| Compute | 1 μηχάνημα, sequential | Backtest grids αργούν (ημέρες για weekly) | Αρκεί για MVP· το §7 του MASTER doc δίνει την πρακτική στρατηγική (weekly μόνο για νικητές) |
| Gate default | `strict` παντού, `academic` μόνο για σύγκριση με papers | Χειρότερα (αλλά αληθινά) νούμερα | Προϊόν που πουλάει academic νούμερα και παραδίδει strict απόδοση = νεκρό προϊόν |

---

## 7. Roadmap

**P0 — ✅ ΟΛΟΚΛΗΡΩΘΗΚΕ πλήρως (2026-07-02/03)**
1. ✅ Bulletproof ablation (18+18+10+10+10 specs, LGBM/XGB/direct, Q1+καλοκαίρι) — πολύ πέρα από
   το αρχικό Δεκ-2025 mini-ablation. `ABLATION_PLAN.md §RESULTS`.
2. ✅ Ensemble στάδιο τρεγμένο ΚΑΙ κλεισμένο (P3 2026-07-04) — κανένα ensemble δεν κερδίζει το LGBM με τίμιο calibration (§4.3)· μόνο weekly LGBM+XGB PENDING.
3. ✅ LSTM seq2seq ολοκληρώθηκε end-to-end (πρώτη φορά χωρίς exception) — ΑΛΛΑ αποκάλυψε
   calibration bug (bias, ποτέ αρνητικές τιμές) — ⬜ debugging ανοιχτό, βλ. `ABLATION_PLAN.md §9`.
4. ✅ LEAR μέσα στο master engine (`--algo lear`, `master_forecast.py`) — leakage-free, με GateSpec.
   Verdict: σαφώς χειρότερο από trees με πλήρη δεδομένα, ρόλος μόνο fallback/robustness.

**P1 — data completeness: ✅ ΟΛΟΚΛΗΡΩΘΗΚΕ (2026-07-02)**
5. generation 2015-2026, entsoe_extra έως 2026-07-02, load_fc πραγματικό 2015-2026, xborder
   (BG/IT-SUD) 2017-2026 (lagged μόνο, εκτός default — §3). Q1-2026 backtests ξανατρέχτηκαν —
   headline έπεσε 23.84→**15.17** €/MWh, κυρίως από τη συμπλήρωση δεδομένων + weekly retrain
   (`ABLATION_PLAN.md §0`, `last.md`).
6. Ομάδα `premarket` (henex_premarket_hourly.parquet) — ⬜ ΠΑΡΑΜΕΝΕΙ ανοιχτό, καμία δουλειά ακόμα.

**Ανοιχτά (μετά το OVERNIGHT PROTOCOL 2026-07-04 — πλήρης λίστα: `ABLATION_PLAN.md §7`)**:
- ~~xborder generalization/DEFAULT_GROUPS απόφαση~~ **ΚΛΕΙΣΤΟ**: xborder εκτός default (leakage
  post-mortem + lagged βλάπτει χειμώνα)· μόνο το summer σήμα (1 σημείο) παραμένει PENDING.
- LSTM calibration bug · weekly LGBM+XGB ensemble confirmation · `premarket` ομάδα ·
  solar_fc ύποπτο 2h shift · xb_lag1_h0 candidate (`ABLATION_PLAN.md §8.1`).

**P2 — προϊοντικά χαρακτηριστικά**
8. Conformal quantiles + pinball/coverage στο settle loop.
9. Daily batch runner (L3) + fallback chain + Task Scheduler.
10. Settle/monitoring job (L5) + champion/challenger.

**P3 — διαφοροποίηση**
11. Γείτονες (IT/BG DAM τιμές, SDAC coupling), gas forward curve, flows.
12. Foundation TS models (Chronos/TimesFM) ως zero-shot benchmark στήλη.
13. Weather forecast archive migration (§3).

**Τι θα ξαναδούμε καθώς μεγαλώνει**: (α) 15-λεπτο MTU στο SDAC → τεταρτοωριαία re-architecture των targets/features· (β) αν αποκτήσει πελάτες: VM + auth στο API + SLA· (γ) probabilistic → bidding optimization layer (από forecast σε θέση/προσφορά) — εκεί είναι το πραγματικό trading προϊόν.

---

## 8. Cutting-edge positioning (σύνοψη — τι υιοθετούμε)

| Τεχνική | Status στον κόσμο | Εδώ |
|---|---|---|
| LEAR + DNN **ensembles** (Lago 2021) | Το reference SOTA για DAM point forecast | ◐ δοκιμάστηκε εξαντλητικά (P0.2, P3) — στα δικά μας δεδομένα ΔΕΝ κερδίζει το καλύτερο μεμονωμένο μοντέλο· κρατιέται μόνο το LEAR ως fallback baseline |
| Trees (LGBM/XGB) με σωστά features | Κερδίζουν συστηματικά σε tabular/μικρά-μεσαία data | ✅ ήδη ο πυρήνας |
| Transformers (TFT/PatchTST/…) | Σπάνια κερδίζουν σε point MAE· αξία σε probabilistic/long-horizon | ⬜ όχι προτεραιότητα |
| **Probabilistic/conformal** | Εκεί έχει μετατοπιστεί η έρευνα — και εκεί είναι η προϊοντική αξία | ✅ P2 πυλώνας |
| Foundation TS (Chronos/TimesFM/Moirai) | Zero-shot benchmarks, ωριμάζουν | ◐ μόνο ως benchmark στήλη |
| HPO (Optuna κ.λπ.) | Marginal κέρδη σε καλά baselines | ⛔ εκτός, by design |
