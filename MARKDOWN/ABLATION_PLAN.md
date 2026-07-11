# ABLATION PLAN — Έγκυρα ευρήματα & ενεργό πρόγραμμα πειραμάτων (v2 δεδομένα)

> **Ενημερώθηκε 2026-07-04** μετά την ολοκλήρωση του OVERNIGHT PROTOCOL (P1→P6).
> Αυτό το αρχείο περιέχει **ΜΟΝΟ ό,τι ισχύει τώρα**. Το πλήρες ιστορικό (μαζί με τα
> ακυρωμένα xborder same-day αποτελέσματα και την πορεία ανακάλυψης του leakage):
> `OLD/docs/ABLATION_PLAN_full_20260704.md`. Ωμά αρχεία runs: `runs/` · CSV: `results/` ·
> logs: `logs/` (νέα οργάνωση 2026-07-04).

---

## 0. TL;DR — Τρέχουσα κατάσταση

> ⚠️ **ΑΝΑΣΤΟΛΗ HEADLINE (2026-07-04, μετά §5.9/§5.10)**: το 15.17 και ΟΛΑ τα νούμερα του §5
> μετρήθηκαν (α) σε timezone-misaligned δεδομένα (διορθώθηκε — §5.9) και (β) με το crosslag
> engine leak ανοιχτό (§5.10 — ✅ **διορθώθηκε πλέον, AEL freeze-at-cutoff**, βλ. πρώτα
> leak-free B1 νούμερα εκεί). Παραμένουν εσωτερικά συγκρίσιμα ως προς τη μεθοδολογία, αλλά
> ΚΑΝΕΝΑ δεν είναι το τρέχον «αληθινό» headline — αυτό ορίζεται μόνο μετά το πλήρες Β3
> re-ablation (LGBM+XGB × Q1+καλοκαίρι × recursive+direct) στα καθαρά (TZFIX+AEL) δεδομένα.

**🏆 Παλιό headline (προ-tzfix, ΣΕ ΑΝΑΣΤΟΛΗ): LGBM recursive weekly, `--features default` = 15.17 €/MWh**
(Q1 2026 = 2025-12-01→2026-02-28). Robustness: seeds 42/43/44 → 15.171/15.429/15.218
(std≈0.11) · 2ο window Μάρτιος 2026: weekly 17.666 < monthly 17.968 (ίδιο πρόσημο υπεροχής).

| Απόφαση | Κατάσταση |
|---|---|
| Αλγόριθμος | **LGBM** (XGB ισοδύναμο σχεδόν· LEAR/MLP/LSTM σαφώς πίσω) |
| Στρατηγική DAM | **recursive** (direct 19.5 vs 16.1 στο Q1 — απορρίφθηκε ως κύρια) |
| Retrain | **weekly** (μονότονο όφελος 16.10→15.80→15.17, επιβεβαιωμένο και σε Μάρτιο) |
| Features | **`default`** (χωρίς xborder — βλ. §5.7) |
| SS | Χρήσιμο (−0.24 και σε monthly) αλλά ΔΕΝ έχει δοκιμαστεί SS×weekly — δεν είναι στο headline |
| Ensembles | Κανένα δεν κερδίζει στιβαρά το καλύτερο μεμονωμένο μοντέλο (βλ. §5.5) |

---

## 1. Μόνιμο pre-flight πρωτόκολλο νέου feature (κανόνας — ΠΡΙΝ από κάθε τεστ)

Για κάθε νέα πηγή, ΠΡΙΝ γραφτεί κώδικας feature:
1. **Πότε ΑΚΡΙΒΩΣ δημοσιεύεται;** — γραπτή απάντηση σε σχέση με το gate 12:00 CET D-1.
2. **Cross-correlation lag-scan** κατά του y — το peak πρέπει να είναι εκεί που προβλέπει η
   θεωρία· peak σε «βολικό» σημείο = ύποπτο leakage.
3. **Hour-of-day profile sanity** (π.χ. solar peak μεσημέρι).

Δίδαγμα-καταλύτης: το xborder same-day «όφελος» −0.72 ήταν εξ ολοκλήρου leakage (ίδιο SDAC
auction με το target, δημοσίευση ~13:00 D-1 > gate 12:00). «Feature που βελτιώνει θεαματικά
και αμέσως» = πρώτος ύποπτος.

Pre-flight πριν από κάθε batch: OneDrive τρέχει · conda `epf` ΟΚ · parquet έχει ΜΟΝΟ
`xb_*_lag{24,48,168}` (ποτέ same-day xb) · LGBM default static Q1 αναπαράγει 16.10±0.05.

## 2. Κριτήρια αποδοχής ευρήματος (αμετάβλητα)

Εύρημα ΓΙΝΕΤΑΙ ΔΕΚΤΟ ⟺ **|ΔMAE| > 0.15** (noise floor· seed std μετρήθηκε 0.05-0.11) **ΚΑΙ
ίδιο πρόσημο σε ≥2 ανεξάρτητες συνθήκες** (άλλο window Ή άλλος αλγόριθμος Ή άλλη στρατηγική).
Αλλιώς: PENDING — όχι συμπέρασμα, όχι στη διπλωματική ως claim.

## 3. Σταθερό setup πειραμάτων

**Q1 2026** (2025-12-01→2026-02-28), v2 πλήρη δεδομένα, `--gate strict`, `--market dam`,
task=price, seed=42, **retrain=static** στα ablations (απομονώνει την αξία των features από
την πολιτική retrain· 1 fit/config). Confirmation νικητών με monthly/weekly στο τέλος.
Δεύτερα windows: Δεκ-2025-μόνο, καλοκαίρι 2025 (Ιουν-Αυγ), Μάρτιος 2026 (1-19).

## 4. Ομάδες features (λεπτόκοκκες)

`calendar` · `lags` (y_lag 1/2/3/6/12/24/48/168) · `dense` (y_lag4..23) · `roll` ·
`resfc` (DA solar/wind/gen fc) · `loadfc` (load_fc) · `engfc` (resload_fc on-the-fly) ·
`xborder` (xb_*_lag24/48/168 — ΜΟΝΟ lagged, εκτός default) · `meteo` · `fuel` ·
`genlags` (gen actuals + residual_load lags) · `loadlags` · `other`.
Umbrellas (backward compat): `forecast`=resfc+loadfc · `crosslags`=genlags+loadlags+other.
Σύνταξη: `default,-meteo` · `lags,calendar,genlags` · `default,xborder`.

---

## 5. ΕΓΚΥΡΑ ΑΠΟΤΕΛΕΣΜΑΤΑ (ό,τι ισχύει σήμερα)

### 5.1 Core ablation — LGBM-Q1 / XGB-Q1 / direct-Δεκ / recursive-Δεκ (2026-07-02)

Ωμά: `results/results_ablation_{q1_lgbm,q1_xgb,dec_direct,dec_recursive}.csv`.
Baselines: LGBM-Q1=16.19 · XGB-Q1=16.39 · direct-Δεκ=16.10 · recursive-Δεκ=13.88 €/MWh.

**🟢 Στιβαρά (συμφωνία σε πολλαπλές συνθήκες):**
- **`genlags` = η πιο πολύτιμη ομάδα.** ΔMAE αφαίρεσης: LGBM +1.45 · XGB +1.42 · direct +1.10 ·
  recursive-Δεκ +2.63 · **MLP +1.36 (P2)** → 3 αλγόριθμοι × 2 στρατηγικές, οριστικό.
- Λιτός πυρήνας `lags,calendar,resfc,genlags` (39-48 feat) κερδίζει το πλήρες `default` (152)
  στον χειμώνα σε recursive (LGBM 16.06 / XGB 16.01 vs 16.19/16.39) — kitchen-sink ≠ βέλτιστο.
  ⚠️ ΑΛΛΑ: αντιστρέφεται στο direct (§5.6) και το καλοκαίρι είναι οριακά χειρότερος.
- **`dense` βοηθάει σταθερά**: LGBM/XGB ≈−0.74 (recursive Q1), direct-Q1 −0.49 → 2 στρατηγικές.

**Strategy-effect (λύθηκε το confound, ίδιο window Δεκ, 2 στρατηγικές):**

| Ομάδα | Direct-Δεκ | Recursive-Δεκ | Συμπέρασμα |
|---|---|---|---|
| `meteo` | +0.548 (βοηθάει) | −0.228 (βλάπτει) | **STRATEGY effect** — βλ. §5.6 για την πλήρη 4/4 επιβεβαίωση |
| `fuel` | −0.682 | −0.376 | Βλάπτει σε Δεκ και στις 2 στρατηγικές (αλλά βλ. §5.6: ουδέτερο σε direct-Q1 — window confound, PENDING) |
| `loadlags` | +0.198 | +0.340 | Θετικό και στις 2 — η LGBM(+)/XGB(−) Q1 ασυμφωνία είναι algo effect |
| `genlags` | +1.104 | +2.626 | Θετικό παντού, μεγαλύτερο στο recursive (σταθεροποιεί rollout) |
| `resfc` | −0.316 | −0.043 | Δεν βοηθάει σε χειμωνιάτικο window — βλ. §5.2 για την εποχικότητα |
| `loadfc` | −0.160 | +0.170 | Στο noise floor — αδιάγνωστο στον χειμώνα· καλοκαίρι καθαρά αρνητικό (§5.2) |

### 5.2 Εποχικότητα — καλοκαίρι 2025 (LGBM 2026-07-02 + XGB P2 2026-07-04)

`results/results_ablation_summer2025.csv` + `runs/p2_out/results_ablation_xgb_summer.csv`.
Baselines: LGBM 14.43 · XGB 15.03. Test 2025-06-01→08-31, static.

| Ομάδα (leave-one-out) | LGBM summer | XGB summer | Ετυμηγορία |
|---|---|---|---|
| `resfc` | **+1.085** | **+1.218** | 🟢 ΔΕΚΤΟ cross-model: το resfc είναι από τις πιο πολύτιμες ομάδες ΤΟ ΚΑΛΟΚΑΙΡΙ (ενώ χειμώνα ~άχρηστο). Το χειμωνιάτικο test window ΥΠΟΕΚΤΙΜΑ συστηματικά solar-ευαίσθητες ομάδες. |
| `genlags` | +0.431 | +1.201 | 🟢 Πολύτιμο και το καλοκαίρι (αναλογικά πιο κρίσιμο τον χειμώνα για LGBM). |
| `loadfc` | −0.278 | −0.300 | 🟢 (αρνητικό) ΔΕΚΤΟ: το loadfc ΔΕΝ προσφέρει — 2 αλγόριθμοι συμφωνούν καλοκαίρι + χειμωνιάτικα σημεία οριακά/αρνητικά. Υποψήφιο για αφαίρεση από core. |
| `loadlags` | −0.055 | +0.421 | 🟡 Ασυνεπές μεταξύ αλγορίθμων — PENDING. |
| resfc additive πάνω σε core | −2.45 MAE | (combo 14.70 κερδίζει default) | Το μεγαλύτερο additive κέρδος όλης της μελέτης. |

**Συμπέρασμα-αρχή**: κάθε συμπέρασμα ανά ομάδα = ζεύγος (χειμώνας, καλοκαίρι) — ποτέ μόνο ένα.

### 5.3 E1 (capacity×meteo) & E2 (meteo στο φορτίο) — 2026-07-02

`runs/e1_e2_out/*.json`.
- **E1 — ΑΠΟΡΡΙΦΘΗΚΕ η υπόθεση «περισσότερα δέντρα ξεκλειδώνουν το meteo»**: Δ(meteo) στο price
  μονότονα χειρότερο με capacity (400: −0.02 → 800: +0.26 → 1600: +0.44) — overfit στο
  reanalysis noise. Και το σκέτο default χειροτερεύει με >800 δέντρα → n_estimators=800 σωστό.
- **E2 — ΕΠΙΒΕΒΑΙΩΘΗΚΕ**: στο task=load το meteo ΔMAE=−33 MW (~26%) — η πιο πολύτιμη ομάδα σε
  όλο το study στο σωστό target. Το meteo δρα στην τιμή ΕΜΜΕΣΑ (μέσω load/RES).
  ⚠️ Reanalysis = perfect-forecast proxy → αισιόδοξο άνω όριο.

### 5.4 Scheduled Sampling — σχήματα, interactions, retrain (2026-07-02 + P5 2026-07-04)

⚠️ **ΟΛΟΚΛΗΡΟΣ ο πίνακας παρακάτω είναι ΠΡΟ-AEL/ΠΡΟ-TZFIX, ΣΕ ΑΝΑΣΤΟΛΗ** (baseline 16.096
ανήκει στα suspended νούμερα του §5.9/§5.10 fix — validity-reviewer, 2026-07-05, στο πλαίσιο
του overnight audit). ΔΕΝ χρησιμοποιείται πλέον ως σύγκριση/αντίφαση για νέα ευρήματα.
Νέο leak-free SS σήμα: §5.12ε παρακάτω (static-retrain recursive, μόνο LGBM ακόμα, PENDING).

`runs/ss_out/*.json` + `runs/p5_out/lgbm_ss_linear_default_monthly_q1.json`.

| Εύρημα | Νούμερα | Κατάσταση |
|---|---|---|
| SS-linear = το μόνο σχήμα που βελτιώνει | static Q1: 16.096→15.830 (−0.266), far-offsets 17.59→17.36· exp/step χειρότερα | ⚠️ ΣΕ ΑΝΑΣΤΟΛΗ (προ-AEL) |
| **SS ΔΕΝ είναι redundant με retrain** | monthly Q1: 15.795→15.552 (**−0.243**) σε καθαρό `default` | ⚠️ ΣΕ ΑΝΑΣΤΟΛΗ (προ-AEL) — η μεθοδολογική διόρθωση (xborder-contamination) παραμένει έγκυρη ως μάθημα, οι τιμές όχι |
| SS×features interaction | υπό SS: resfc −0.04→**+1.10**, loadfc →+0.46, loadlags +0.22→**−0.32** (αναστροφή) | ⚠️ ΣΕ ΑΝΑΣΤΟΛΗ (προ-AEL) |
| SS×weekly | — | ⬜ ΔΕΝ έχει τρέξει ποτέ leak-free (SS καλωδιωμένο μόνο static στο overnight batch — βλ. §5.12ε) |

### 5.5 Μοντέλα & Ensembles (Βήμα 9 έγκυρα μέρη + P2 + P3, 2026-07-02/04)

**Κατάταξη (Q1 static, default, recursive)**: LGBM 16.10 < XGB 16.21 < LEAR 19.49 < MLP 20.49
< LSTM-s2s 44.16 (calibration bug: corr 0.69 αλλά bias +41, ποτέ αρνητικές τιμές — ανοιχτό).

**LEAR έχει ΔΙΚΟ του βέλτιστο feature set (P2 mini-ablation, 6 specs)**: αφαίρεση resfc
**βελτιώνει** το LEAR κατά −1.00 και αφαίρεση fuel κατά −0.51 (αντίθετα από δέντρα) — L1
shrinkage effect. Η υπόθεση «τα LGBM-καλύτερα features βοηθούν παντού» είναι ΛΑΘΟΣ. Αν το LEAR
χρησιμοποιηθεί ως fallback, θέλει δικό του spec (π.χ. `default,-resfc,-fuel`).

**MLP mini-ablation (P2, 3 specs)**: genlags πολύτιμο (+1.36) · bare core `lags,calendar`
**κερδίζει** το πλήρες default κατά −1.58 (18.91 vs 20.49) — το MLP πάσχει περισσότερο απ' όλους
από το kitchen-sink.

**Ensembles — όλα τα τίμια τεστ (P3/P4)**:

| Τεστ | Αποτέλεσμα | Ετυμηγορία |
|---|---|---|
| Weighted-by-1/MAE (3 μέλη, βάρη από Δεκ, eval Ιαν-Φεβ) | weighted 17.78 > median 17.32 > **LGBM 17.26** | 🔴 ΑΠΟΡΡΙΦΘΗΚΕ — με out-of-sample calibration κανένα ensemble δεν κερδίζει το LGBM |
| mean/median (3 μέλη LGBM+XGB+LEAR, full Q1) | median 16.042 vs LGBM 16.096 (−0.055) | 🟡 κάτω από noise floor — όχι συμπέρασμα |
| **Weekly LGBM+XGB (2 σχεδόν-ισοδύναμα μέλη)** | ens 15.023 vs LGBM 15.171 (**−0.148**) | 🟡 PENDING — ακριβώς στο όριο, 1 σημείο· το πιο ελπιδοφόρο ensemble εύρημα, θέλει 2ο window/seeds |
| Cross-strategy (direct+recursive LGBM) | ens 16.62 vs recursive 16.10 (+0.52) | 🔴 ΑΠΟΡΡΙΦΘΗΚΕ — ανόμοια ποιότητα μελών χαλάει τον μέσο |

Μοτίβο: ensemble βοηθάει ΜΟΝΟ με μέλη συγκρίσιμης ποιότητας — ποτέ δεν το θεωρούμε δεδομένο.

### 5.6 Direct πυλώνας & το meteo strategy-effect (P4, 2026-07-04)

`runs/p4_out/*`. Direct-LGBM 18 specs + direct-XGB 10 specs, static Q1 + direct monthly.

- **Direct πολύ χειρότερο από recursive στο Q1**: 19.50 (LGBM) / 19.59 (XGB) vs recursive 16.10.
  Στο Δεκ-μόνο η διαφορά ήταν μικρότερη (16.10 vs 13.88) — το direct υποφέρει δυσανάλογα σε
  μεγάλα/ασταθή windows. **Απόφαση: recursive για DAM προϊόν.**
- **Retrain ΔΕΝ βοηθάει το direct**: monthly 19.589 vs static 19.495 (+0.09, ουδέτερο) — αντίθετα
  με το recursive (retrain λύνει staleness του rollout, το direct δεν έχει rollout).
- **🎯 meteo strategy-effect — το πιο στιβαρό εύρημα strategy όλης της μελέτης (4/4)**:
  βοηθάει ΠΑΝΤΑ στο direct (LGBM-Δεκ +0.55, LGBM-Q1 +0.49, XGB-Q1 +0.29) και βλάπτει ΠΑΝΤΑ στο
  recursive (LGBM-Q1 −0.20, XGB-Q1 −0.52, LGBM-Δεκ −0.23). Μηχανισμός: στο direct κάθε offset
  είναι ανεξάρτητο μοντέλο χωρίς φρέσκα y-lags → αξιοποιεί exogenous· στο recursive τα y-lags
  κυριαρχούν και το meteo προσθέτει θόρυβο.
- Στο direct-Q1 το πλήρες default ΚΕΡΔΙΖΕΙ το bare core (αντίθετα από recursive) — 2 αλγόριθμοι
  συμφωνούν (LGBM +0.74, XGB +0.32 ζημιά του bare core).
- `fuel` σε direct: Δεκ −0.68 vs Q1 ~0 → window-size confound ΜΕΣΑ στο direct, PENDING.

### 5.7 xborder — τελική ετυμηγορία (post-mortem σύνοψη)

**Same-day xb τιμές = leakage** (ίδιο SDAC auction με το target). Δομικό fix στο `data.py`:
same-day στήλες δεν μπαίνουν καν στο parquet — μόνο `xb_price_{bg,itsud}_lag{24,48,168}`.
Τα «14.43 / 15.02 / −0.72 όφελος» ΑΚΥΡΑ — μην αναφερθούν πουθενά ως αποτελέσματα.

**Το έντιμο (lagged) xborder — P1 closure test:**

| Συνθήκη | ΔMAE (xborder−default) | |
|---|---|---|
| static Q1 | +0.422 | βλάπτει |
| monthly Q1 | +0.134 | βλάπτει (εντός noise) |
| weekly Q1 | +0.406 | βλάπτει |
| **summer static** | **−0.519** | **βοηθάει** (1 σημείο) |

🔴 **Χειμώνας: ΔΕΚΤΟ ότι βλάπτει** (3/3 cadences) → **εκτός default set**.
🟡 **Καλοκαίρι: PENDING** — 1 σημείο· πιθανό γνήσιο season-interaction (όπως το resfc), θέλει
cadence-robustness/seed πριν δηλωθεί. (Στο LEAR: −0.16, οριακό, 1 σημείο — επίσης PENDING.)

**Bonus διορθωμένο bug**: 1h timestamp shift στο `fetch_entsoe_xborder.py` βρέθηκε με
cross-correlation lag-scan και διορθώθηκε (peak τώρα lag=0, BG 0.824 / IT-SUD 0.913).

### 5.9 ⏰ TIMEZONE ALIGNMENT FIX (2026-07-04) — δομική διόρθωση δεδομένων

**Εύρημα (ξεκίνησε από τον §7.8 solar έλεγχο — η υποψία ήταν σωστή στην ουσία, με ανάποδη
απόδοση αιτίας)**: το hourly parquet ανακάτευε ΤΕΣΣΕΡΑ ρολόγια. Το index (από τα price/load
GUI exports) είναι **CET/CEST** — όχι UTC όπως υπέθεταν τα docstrings. Ως προς αυτό:

| Οικογένεια | Πραγματικό frame | Offset vs y (πριν το fix) | Απόδειξη |
|---|---|---|---|
| y, load, load_fc | CET/CEST | 0 ✓ | header «MTU (CET/CEST)» |
| gen_* → genlags, residual_load | Europe/Athens | **+1h αργά σταθερά** | gen_solar peak 12/13 αντί 11/12 |
| resfc (solar/wind/gen fc) | UTC | **1h νωρίς DJF / 2h νωρίς JJA** | solar_fc peak 10/10 |
| meteo w_* | σταθερό UTC+1 | 0 DJF / **1h νωρίς JJA** | radiation peak 11/11 όλα τα έτη |

Απόδειξη-κλειδί: corr(solar_fc, gen_solar actual) peak σε **k=3** (JJA-κυριαρχούμενα έτη) /
**k=2** (2026, μόνο χειμώνας) με 0.97 — ακριβώς το UTC↔EET/EEST χάσμα. Το «shift detection»
του `_align_and_join` ήταν no-op (μετράει index overlap, τυφλό σε label misalignment
συνεχών ωριαίων σειρών).

**Fix (δομικό, `src/data.py`)**: κανονικό frame = CET/CEST-naive· gen −1h σταθερά
(Athens=Brussels+1 πάντα)· entsoe_extra UTC→Europe/Brussels· weather Etc/GMT-1→Europe/Brussels.
xborder/load_fc δεν άλλαξαν (ήδη ευθυγραμμισμένα). Rebuild 2026-07-04 (backup:
`data/processed/_backup_tzfix_20260704/`). **Μετά το fix: peak_k=0 σε ΟΛΑ τα έτη, corr@0
0.65→0.98** (`solar_shift_check.py`), fc/gen/meteo όλα 11 DJF / 12 JJA ✓.

**Καμία διαρροή δεν εισήχθη ποτέ** από τα shifts (τα fc είναι D-1-published για όλη τη μέρα·
τα gen lags κουβαλούσαν ΠΑΛΑΙΟΤΕΡΗ πληροφορία απ' ό,τι δήλωναν) — μόνο χαμένο σήμα στα
genlags/resfc/meteo (τις κορυφαίες ομάδες!) και θολό residual_load (load(t)−gen(t−1)).
⚠️ **Όλα τα προ-fix νούμερα (και το headline 15.17) αφορούν τα μη ευθυγραμμισμένα δεδομένα**
— συγκρίσιμα μεταξύ τους, όχι με τα νέα. Νέο baseline: `runs/tzfix_out/`.

**Πρώτα post-fix νούμερα (LGBM recursive static Q1, `runs/tzfix_out/`)** — control στο backup
αναπαρήγαγε 16.096 ✓ (η μεταβολή οφείλεται αποκλειστικά στην ευθυγράμμιση):

| Spec | προ-fix | post-fix |
|---|---|---|
| `default` | 16.10 | **19.17** |
| `default,-meteo` | — | 18.90 |
| `default,-meteo,-resfc` | — | **16.54** |
| `lags,calendar,resfc,genlags` | 16.06 | 19.37 |
| `lags,calendar,genlags` | — | 17.16 |

**Συμπέρασμα**: στο ευθυγραμμισμένο resfc το recursive-χειμώνας παθαίνει ~+2.4 MAE (η §5.6
παθολογία των exogenous-in-recursive, που πριν κρυβόταν γιατί το misaligned resfc ήταν
αγνοήσιμος θόρυβος). **Η επιλογή feature set ΠΡΕΠΕΙ να ξαναγίνει από την αρχή στα
ευθυγραμμισμένα δεδομένα** — ΑΦΟΥ πρώτα κλείσει και το §5.10.

### 5.10 ✅ ΛΥΘΗΚΕ — Engine leakage στα cross actual lags (βρέθηκε & διορθώθηκε 2026-07-04, AEL)

Το `recursive_openloop.py` αντικαθιστά από το running series ΜΟΝΟ τα **y-lags** (regex
`y_lag*`). Όλα τα άλλα actual-derived lags — `gen_solar_lag1/2`, `gen_wind_lag1/2`,
`residual_load_lag1..12`, `load_lag*` — διαβάζονται **αυτούσια από τα actuals** και στο eval:
για την ώρα 14:00 της D, το `gen_solar_lag1` περιέχει πραγματική παραγωγή 13:00 της D —
άγνωστη στο gate (actuals γνωστά έως ~11:00 D-1, βλ. MASTER §1). Το MASTER §2 απαιτούσε
`h−k ≤ t0_cutoff` για crosslags· υλοποιήθηκε μόνο για y-lags. **Συνέπεια: κάθε config με
genlags/loadlags (και το παλιό headline 15.17) είναι αισιόδοξο** — μέρος του «genlags = η πιο
πολύτιμη ομάδα (+1.45)» είναι πιθανόν leakage, ίδιας φύσης με το xborder incident.

**Fix — AEL (υλοποιήθηκε 2026-07-04, βλ. `SYSTEM_DESIGN §4.8`)**: freeze-at-cutoff (default,
deployable) ή NaN (`--crosslag_mode nan`, sensitivity, μόνο lgbm/xgb) σε recursive rollout +
direct row@cutoff + training rows + conformal quantile path, με anchor-based cutoff ανά block.
Poisoning self-test `src/check_crosslag_fairness.py`: PASS (A leak=0.000000) σε recursive-dam /
direct-dam / recursive-forward· αυτόματο μέσω `preflight_check.py --poison`.

**Πρώτα leak-free νούμερα (B1, 2026-07-04) — LGBM recursive static Q1, `runs/b1_leakfree/`:**

| Spec | post-TZFIX προ-AEL (§5.9) | **leak-free (AEL freeze)** | Δ (κόστος leak) |
|---|---|---|---|
| `default` | 19.17 | **20.79** | +1.62 |
| `default,-meteo,-resfc` | 16.54 | **19.90** | +3.36 |
| `lags,calendar,genlags` | 17.16 | **18.94** | +1.78 |
| `default` + `--crosslag_mode nan` | — | 22.18 | (sensitivity· χειρότερο από freeze +1.39) |

Αναπαραγωγή: `python -m src.master_forecast --algo lgbm --task price --market dam
--strategy recursive --gate strict --retrain static --train_end "2025-11-30 23:00"
--test_start "2025-12-01 00:00" --test_end "2026-02-28 23:00" --features <spec>
[--crosslag_mode nan] --out_json runs/b1_leakfree/<name>.json`.

**Πρώτη ανάγνωση (static-only, 1 window — οριστικοποίηση στο Β3 re-ablation):**
- Το crosslag leak κόστιζε +1.6 έως +3.4 MAE ανάλογα με το spec — μεγαλύτερο εκεί που τα
  genlags/loadlags σήκωναν περισσότερο βάρος (χωρίς meteo/resfc). Ίδια τάξη με το xborder incident.
- **Νέα leak-free κατάταξη: ο λιτός πυρήνας `lags,calendar,genlags` (18.94) κερδίζει** και το
  default (20.79) και το default,-meteo,-resfc (19.90) — το «λιτό > kitchen-sink» στο
  recursive-χειμώνα επιβιώνει και μετά το AEL.
- **freeze > nan** (20.79 vs 22.18): η «τελευταία γνωστή τιμή» κρατά χρήσιμο σήμα· το NaN
  πετά και το νόμιμο μέρος της στήλης (και ρίχνει 152→138 usable features). Το freeze
  παραμένει το deployable default.
- ⚠️ Τα παλιά static baselines (16.10 προ-TZFIX, 19.17 προ-AEL) είναι πλέον ΜΟΝΟ ιστορικές
  αναφορές — κάθε νέα σύγκριση ξεκινά από τα leak-free του πίνακα (Α3.7).

Διευκρινίσεις από το πλήρες code review 2026-07-04 (βλ. `SYSTEM_DESIGN §4.8-4.10`,
`VALIDITY_CHECKLIST.md`): (α) το leak αφορά ΚΑΙ το **direct** (row@cutoff 23:00 D-1 περιέχει
gen/load actuals 12:00-22:00 D-1, αδημοσίευτα στο gate 12:00 — ηπιότερο αλλά υπαρκτό)·
(β) το **task=load** δεν έχει contemporaneous price feature (ελέγχθηκε — καθαρό)·
(γ) το **`src/conformal.py` ΥΠΑΡΧΕΙ ήδη υλοποιημένο** (split-conformal causal + quantile-LGBM
με aux_models στο rollout) — τα docs ήταν πίσω από τον κώδικα· έτοιμο να τρέξει ΜΕΤΑ το AEL.

### 5.11 Β3 LEAK-FREE RE-ABLATION (2026-07-05) — η νέα βάση feature selection

**Setup**: static/strict/DAM/price, TZFIX+AEL (crosslag_mode=freeze), LGBM+XGB ×
(Q1 2026, καλοκαίρι 2025) × (recursive· direct σε 5 specs). Sanity: default Q1
LGBM = 20.7926 ≡ B1 (Α3.5 ✓). Πηγές: `results/b3_*.csv`, `runs/b3_ablation/`,
summariser: `scripts/b3_summarize.py`.

**Recursive — ΔMAE vs default (θετικό = χειρότερο χωρίς/με το variant):**

| spec | Q1/LGBM | Q1/XGB | Summer/LGBM | Summer/XGB |
|---|---|---|---|---|
| default (MAE) | 20.79 | 21.59 | 15.14 | 16.11 |
| −resfc | −3.07 | −3.50 | +0.49 | −0.34 |
| −meteo | +2.24 | +0.37 | +0.66 | +0.75 |
| −genlags,−loadlags | −0.56 | +0.38 | +0.71 | +0.51 |
| +dense | **−1.08** | **−1.86** | **−0.98** | **−0.62** |
| lean (lags,cal,genlags) | −1.86 | −2.71 | +2.67 | +1.36 |
| bare (lags,cal) | −1.28 | −2.14 | +2.28 | +1.19 |

**Verdicts (§2: |ΔMAE|>0.15 & ίδιο πρόσημο ≥2 συνθήκες):**
1. 🟢 **meteo ΒΟΗΘΑΕΙ το recursive — ACCEPTED (4/4)**. ΑΝΑΤΡΟΠΗ του §5.6 «meteo
   βλάπτει πάντα στο recursive» — εκείνο μετρήθηκε πάνω σε leaked genlags configs.
2. 🟢 **dense ΒΟΗΘΑΕΙ — ACCEPTED (4/4)** (−0.6..−1.9). Νέο εύρημα (πριν: «μπορεί
   να βλάψει OL»).
3. 🔵 **resfc: εποχιακό flip — PENDING-seasonal**. Χειμώνα ΤΟΞΙΚΟ (αφαίρεση −3.1/−3.5,
   2 algos ✓), καλοκαίρι ουδέτερο/ελαφρά χρήσιμο (mixed). Όπως το xborder: εποχιακό
   interaction, ΟΧΙ universal.
4. 🔵 **genlags+loadlags: ΟΡΙΑΚΟ/MIXED** — το προ-AEL «πιο πολύτιμη ομάδα (+1.45)»
   ήταν σε μεγάλο βαθμό το leak. Καθαρή αξία: −0.6..+0.7, ασυνεπές πρόσημο.
5. 🔵 **lean core: winter-only** — κερδίζει Q1 (−1.9/−2.7) και καταρρέει καλοκαίρι
   (+2.7/+1.4), consistent σε 2 algos → όχι universal default.
6. 💡 Αδοκίμαστοι συνδυασμοί-υποψήφιοι για το headline stage: `default,dense,-resfc`
   (χειμώνας) · `default,dense` (universal). Q1-καλύτερα observed: `−resfc` 17.72/18.09.

**Direct — ΔMAE vs default (ολοκληρώθηκε, πηγή: `scripts/b3_summarize.py` στα ίδια CSVs):**

| spec | Q1/LGBM | Q1/XGB | Summer/LGBM | Summer/XGB |
|---|---|---|---|---|
| default (MAE) | 19.34 | 19.21 | 20.21 | 20.08 |
| −resfc | +0.43 | +0.37 | +0.75 | +0.27 |
| −meteo | +0.12 | +0.43 | +0.96 | +1.49 |
| −genlags,−loadlags | +0.19 | +0.34 | −0.40 | −0.51 |
| lean (lags,cal,genlags) | +1.53 | +1.83 | +1.20 | +1.78 |

**Συμπληρωματικά verdicts με το direct (σύνολο 8 συνθήκες):**
- **meteo → ACCEPTED 8/8** (η αφαίρεση χειροτερεύει παντού).
- **resfc → interaction επιβεβαιωμένο**: recursive-χειμώνα ΤΟΞΙΚΟ (−3.1/−3.5), direct
  ΒΟΗΘΑΕΙ σταθερά (+0.27..+0.75 όταν αφαιρεθεί, 4/4) → strategy×season effect, όχι universal.
- **lean core → χάνει στο direct 4/4** (+1.2..+1.8): winter-recursive-only φαινόμενο.
- **genlags+loadlags → MIXED και στο direct** (±0.5) — παραμένει ΟΡΙΑΚΟ.
- Σημ.: dense×direct + bare×direct ΔΕΝ έτρεξαν στο Β3 → γεμίζουν στο overnight
  (Blocks B/D, 2026-07-05 νύχτα).

### 5.12 OVERNIGHT 2026-07-05 + follow-up 2026-07-06 — 🏆 HEADLINE ΚΛΕΙΔΩΣΕ (LGBM default,dense weekly recursive, Q1=16.96/Summer=13.74) — C(LOAD)/E(SS)/F(conformal) παραμένουν PENDING

**Batch**: `scripts/overnight_20260705.sh` (εκκίνηση 2026-07-05 04:58:50, detached, τέλος
13:34:56). Outputs: `runs/overnight_20260705/{a_cadence,b_march,c_load,d_fill,e_ss,f_conformal}/`
(24/28/56/8/4/2 = 122 runs), log: `logs/overnight_20260705_master.log`. **0 FAILED** στο log.
Verdicts §5.12α/β περάσαν από `validity-reviewer` (ανεξάρτητος επανυπολογισμός Δ, confirmed).
Verdicts §5.12γ-στ (Blocks C/D/E/F, δεύτερος γύρος validity-reviewer 2026-07-05 μετά την
ολοκλήρωση του batch) — **καμία headline lock ακόμα**: βρέθηκε ότι το seed-check του Block D
δοκιμάζει ΛΑΘΟΣ config (§5.12δ), το Block C χρειάζεται poisoning test που ποτέ δεν έτρεξε για
task=load, και το Block F έχει μόνο 1 window. Σύνοψη: `conda run -n epf --no-capture-output
python -X utf8 scripts/overnight_summarize.py` → `results/overnight_20260705.csv`.

**§5.12α Cadence (Block A, recursive, static, price/DAM) — weekly vs monthly:**

| spec | Q1/LGBM (m→w) | Q1/XGB (m→w) | Summer/LGBM (m→w) | Summer/XGB (m→w) |
|---|---|---|---|---|
| default | 19.222→17.431 (Δ−1.79) | 19.486→17.672 (Δ−1.81) | 15.080→14.147 (Δ−0.93) | 15.419→14.573 (Δ−0.85) |
| default,dense | 18.351→17.035 (Δ−1.32) | 18.571→16.924 (Δ−1.65) | 14.063→13.812 (Δ−0.25) | 14.406→14.400 (Δ−0.01) |
| default,−resfc | 17.644→17.362 (Δ−0.28) | 17.702→17.169 (Δ−0.53) | 15.546→15.050 (Δ−0.50) | 15.522→14.959 (Δ−0.56) |

🟢 **weekly > monthly — ACCEPTED** (12/12 συνθήκες αρνητικό Δ, |Δ|>0.15 σε 11/12· η μόνη
οριακή summer/xgb/dense Δ=−0.006 αγνοείται ως θόρυβος). Directional finding, ΟΧΙ ακόμα
headline (Κανόνας 5 — χρειάζεται seeds, Block D).

Στο weekly cadence, `default,dense` έναντι `default`: Q1/LGBM −0.40, Q1/XGB −0.75,
Summer/LGBM −0.34, Summer/XGB −0.17 → 🟢 **dense καλύτερο spec στο cadence stage — ACCEPTED
(4/4, |Δ|>0.15 σε 3/4)**, συνεπές με §5.11. `default,−resfc` έναντι `default`: μικτό στο Q1
(θετικό, βοηθάει) αλλά χειρότερο στο καλοκαίρι (+0.5/+0.9) — επιβεβαιώνει το ήδη γνωστό
εποχιακό flip, όχι νέο εύρημα. **Καλύτερο observed μέχρι στιγμής**: `default,dense`
weekly-LGBM Q1=17.035, Summer=13.812 (υποψήφιο headline, εκκρεμεί seeds+2ο window — Block D/§5.12δ).

**§5.12β Μάρτιος tie-break (Block B, static train_end=2026-02-28, test 03-01→03-20) — ΔMAE
vs default — ΟΛΟΚΛΗΡΩΘΗΚΕ 28/28:**

| spec effect | lgbm/rec (base 20.934) | lgbm/dir (base 24.479) | xgb/rec (base 21.187) | xgb/dir (base 24.690) |
|---|---|---|---|---|
| dense | −0.557 | −2.795 | −0.636 | −2.576 |
| nometeo (αφαίρεση meteo) | −0.667 | −0.979 | −0.003 | −1.569 |
| noresfc (αφαίρεση resfc) | −0.271 | +0.233 | −0.609 | +0.837 |
| no genlags/loadlags | +0.948 | +0.704 | +0.435 | +1.847 |
| lean (lags,cal,genlags) | −0.280 | −1.419 | −1.466 | −1.013 |
| lags,calendar (bare) | +0.553 | −1.748 | +0.417 | −0.827 |

Verdicts (τελικά, περασμένα από validity-reviewer σε 2 γύρους — 1ος με 3/4, 2ος με 4/4
συνθήκες):
1. 🟢 **dense ΑΝΤΕΧΕΙ πλήρως στον Μάρτιο — ACCEPTED** (4/4 αρνητικό, |Δ|>0.15) — ενισχύει
   §5.11, τώρα 2ο ανεξάρτητο window υπέρ του dense σε ΟΛΕΣ τις 4 συνθήκες.
2. 🔵 **resfc: επιβεβαιώνει το ήδη γνωστό strategy×season interaction** (recursive: resfc
   τοξικό −0.27/−0.61, direct: βοηθάει +0.23/+0.84) — απλή ενίσχυση της §5.11 ετυμηγορίας.
3. ⚠️ **ΑΝΟΙΧΤΗ ΣΥΓΚΡΟΥΣΗ — meteo (ΠΑΡΑΜΕΝΕΙ PENDING μετά το 2ο validity pass)**: με
   4/4 πλέον, 3/4 δείχνουν meteo ΝΑ ΒΛΑΠΤΕΙ (rec −0.67, dir −0.98, xgb/dir −1.57· xgb/rec
   ουδέτερο −0.003, κάτω από seed-noise floor) — αντίθετο πρόσημο από το §5.11 ACCEPTED
   «meteo βοηθάει» (8/8). Ρητή ετυμηγορία validity-reviewer: ΔΕΝ αρκεί να ανατρέψει το
   §5.11 — είναι 1 window (Μάρτιος, όσα κελιά κι αν έχει) έναντι 2 ανεξάρτητων windows
   (Q1+summer) στο §5.11· τα κελιά ΕΝΤΟΣ του Μαρτίου είναι συσχετισμένα (ίδιο test
   διάστημα), όχι ισοδύναμα με ανεξάρτητα δείγματα. Χρειάζεται 3ο ανεξάρτητο window
   (π.χ. μόνο Δεκέμβριος) για να ανοίξει επίσημη επανεξέταση.
4. ⚠️ **ΑΝΟΙΧΤΗ ΣΥΓΚΡΟΥΣΗ — lean core στο direct (ΠΑΡΑΜΕΝΕΙ PENDING)**: με xgb/dir
   συμπληρωμένο, 2/2 direct κελιά Μαρτίου δείχνουν lean core ΝΑ ΚΕΡΔΙΖΕΙ (lgbm −1.42,
   xgb −1.01) — αντίθετο από τα 4 κελιά §5.11 direct (lean πάντα έχανε, +1.2..+1.8 σε
   Q1/summer). Ίδια λογική με το meteo: 2 συσχετισμένα κελιά ενός window δεν αρκούν να
   ανατρέψουν 4 κελιά δύο ανεξάρτητων windows. Χρειάζεται 3ο window με direct strategy.
5. 🔵 **genlags+loadlags**: 4/4 στον Μάρτιο συνεπές (βοηθάει, +0.44..+1.85) αλλά τα 8
   παλιότερα Q1/summer κελιά ήταν μικτά (±0.5) — **Παραμένει ΟΡΙΑΚΟ/MIXED** (§5.11
   αμετάβλητο)· χρειάζεται Block C ή 3ο window.
6. 💡 **bare core (lags,calendar)**: μικτό στο Μάρτιο (rec: χειρότερο +0.55/+0.42· dir:
   καλύτερο −1.75/−0.83) — strategy-dependent, συνεπές με το ήδη γνωστό winter-recursive
   pattern του §5.11 (bare/lean κερδίζουν winter-recursive, όχι απαραίτητα winter-direct).

**§5.12γ LOAD ablation (Block C, static, task=load, MAE σε MW) — ✅ 56/56 ΟΛΟΚΛΗΡΩΘΗΚΕ
(LGBM+XGB × dir/rec × Q1/summer × 7 specs) — validity-reviewer 2ος γύρος 2026-07-05:**

🟢 **`-loadfc` arm ΑΚΥΡΟ — ACCEPTED ως γνωστό VOID (bug confirmed, ΟΧΙ εύρημα)**:
bit-for-bit Δ=0.000 σε 8/8 κελιά τώρα (πλήρες 56/56, LGBM+XGB, dir+rec, Q1+summer) —
`data/processed/hourly_load.parquet` δεν έχει καθόλου στήλη `load_fc` (σε αντίθεση με το
`hourly.parquet` του price). Καμία γραμμή «loadfc» δεν είναι εύρημα. Fix (μελλοντικό,
ΟΧΙ τώρα, `data/processed/` προστατευμένο): rebuild με merge του `load_forecast_hourly.parquet`.

🔴 **Όλα τα υπόλοιπα arms (genlags/loadlags, loadlags-only, nometeo, dense) — BLOCKED, ΟΧΙ
ούτε καν PENDING με directional claim.** Με το πλήρες 8-κελιό (LGBM+XGB × dir+rec × Q1+summer)
το πρόσημο αντιστρέφεται σε ΚΑΘΕ arm — καμία ομάδα δεν πιάνει το κριτήριο §2 (ίδιο πρόσημο σε
≥2 ανεξάρτητες συνθήκες):

| spec effect | q1/lgbm/dir | q1/lgbm/rec | q1/xgb/dir | q1/xgb/rec | summer/lgbm/dir | summer/lgbm/rec | summer/xgb/dir | summer/xgb/rec |
|---|---|---|---|---|---|---|---|---|
| no genlags/loadlags | −87.857 | +2.213 | −101.587 | −5.176 | +26.708 | −8.278 | +21.399 | −22.084 |
| no loadlags (μόνο) | −84.159 | −2.582 | −90.241 | −9.842 | +13.061 | +26.158 | −8.277 | +32.101 |
| nometeo | +48.472 | +109.482 | +41.400 | +101.643 | +13.423 | −7.014 | −14.410 | −23.106 |
| dense | −64.591 | +0.220 | −72.035 | −1.275 | −10.568 | −10.762 | −22.323 | +15.971 |

Επιπλέον, **καμία poisoning/crosslag-fairness δοκιμή δεν έχει τρέξει ΠΟΤΕ για task=load**
(το `src/check_crosslag_fairness.py` υποστηρίζει `--task load` στον κώδικα αλλά μόνο
`--task price` έχει εκτελεστεί μέχρι σήμερα — βλ. Block 0 στο log). Effect sizes έως 26-30%
του baseline (genlags/loadlags στο dir) είναι ακριβώς η υπογραφή που το AEL fix έπιασε στο
price task — **χρειάζεται targeted poisoning check στο crosslag family για task=load πριν
γραφτεί οποιοδήποτε από τα παραπάνω arms ως εύρημα** (leakage-sensitive core rule, CLAUDE.md).
Εντολή: `conda run -n epf --no-capture-output python -X utf8 -m src.check_crosslag_fairness
--task load --algo lgbm --strategy {recursive,direct} --market dam --gate strict`.

**§5.12δ Seeds robustness / headline lock (Block D + follow-up Block G, 2026-07-06) —
🟢 ΚΛΕΙΔΩΣΕ:**

Το αρχικό Block D seed-loop έτρεξε με λάθος cadence (`--retrain static` αντί `weekly`,
βλ. ιστορικό παρακάτω) — διορθώθηκε με `scripts/followup_20260705.sh` Block G, σωστά
`--retrain weekly`, seeds 7+123 (μαζί με το ήδη υπάρχον seed=42 από το Block A):

| seed | Q1 | Summer |
|---|---|---|
| 42 | 17.0354 | 13.8123 |
| 7  | 16.9397 | 13.6910 |
| 123 | 16.8913 | 13.7081 |
| **mean** | **16.956** | **13.737** |
| **std** | **0.060** | **0.054** |

🟢 **ΔΕΚΤΟ — headline κλειδωμένο** (validity-reviewer, 2026-07-06): 3 seeds + 2 ανεξάρτητα
windows (Α3 κανόνας 5), std 0.054-0.060 — **πιο σφιχτό** από το μεθοδολογικό πρότυπο P6
(seeds 42/43/44, std≈0.112, §5.8). Poisoning self-test (Tests A/B/D1/D2) έτρεξε φρέσκο
μέσα στο ίδιο batch, PASS.

### 🏆 ΝΕΟ HEADLINE (leak-free, 2026-07-06)
**LGBM, `default,dense`, `--retrain weekly`, recursive, DAM/price, gate strict:**
**Q1 ≈ 16.96 €/MWh · Summer ≈ 13.74 €/MWh** (std ≤0.06, 3 seeds έκαστο).
Πηγές: `runs/overnight_20260705/a_cadence/{q1,summer}_lgbm_weekly_default_dense.json`
(seed 42) + `runs/followup_20260705/{q1,summer}_lgbm_rec_weekly_default_dense_seed{7,123}.json`.
Παλιό προ-AEL headline 15.17 παραμένει ΣΕ ΑΝΑΣΤΟΛΗ (leaked, ασύγκριτο — §5.8).

**Ιστορικό διόρθωσης (για traceability)**: το αρχικό seed-loop
(`scripts/overnight_20260705.sh:127-137`) έτρεξε λάθος με `--retrain static` — τα seed
MAE (Q1 19.9699/19.8542, Summer 14.1263/14.0797) ταίριαζαν με το static anchor του Β3
(19.713/14.166), όχι με το weekly 17.035/13.812. Εντοπίστηκε 2026-07-05/06, διορθώθηκε
με το follow-up batch.

**§5.12ε Scheduled Sampling (Block E, static-retrain recursive LGBM) — 🟡 PENDING:**

Σωστό baseline (static, ΟΧΙ τα weekly numbers του Block A) από `runs/b3_ablation/`:
q1/default=20.7926, q1/default,dense=19.7130, summer/default=15.1421, summer/default,dense=14.1656.

| spec | Q1 SS (Δ vs baseline) | Summer SS (Δ vs baseline) |
|---|---|---|
| default | 21.3171 (**+0.524**) | 15.2953 (**+0.153**) |
| default,dense | 19.8234 (+0.110, κάτω από floor) | 14.2881 (+0.122, κάτω από floor) |

🟡 **`default` (χωρίς dense): SS ΒΛΑΠΤΕΙ — clears το §2 κριτήριο** (2/2 ανεξάρτητα windows,
ίδιο πρόσημο, |Δ|>0.15) — αλλά μόνο 1 algorithm (LGBM) δοκιμασμένο, θέλει XGB ή 3ο window
πριν ΔΕΚΤΟ οριστικά. Αυτό **ανατρέπει το προ-AEL §5.4 «SS-linear ΔΕΚΤΟ, −0.266»** — εκείνο
είναι πλέον suspended (βλ. §5.4 πάνω), άρα καμία πραγματική σύγκρουση, μόνο ενημέρωση.
`default,dense`: και τα δύο Δ κάτω από το 0.15 noise floor — καμία ετυμηγορία, θα χρειαστεί
repeated-seed run για να ξεχωρίσει από θόρυβο.

**§5.12στ Conformal smoke (Block F, LGBM weekly default,dense, Q1 μόνο) — 🔵 δεδομένα ΥΠΑΡΧΟΥΝ
αλλά ΟΧΙ αρκετά για headline probabilistic claim (Rule 6: ≥2 μοντέλα × ≥2 windows):**

⚠️ Ο summarizer ανέφερε "NO-MAE" — αυτό είναι **σφάλμα του `scripts/overnight_summarize.py`**
(δεν ξέρει να διαβάσει το σχήμα `results.<window>.mae_p50` των conformal JSONs), ΟΧΙ αποτυχία
run. Πραγματικοί αριθμοί (από τα JSON απευθείας):

| μέθοδος | n | avg_pinball | MAE(p50) | coverage 80% (nominal) | p10 emp% | p90 emp% |
|---|---|---|---|---|---|---|
| quantile-LGBM | 2160 | 6.075 | 16.4454 | **43.19** | 38.94 | 82.13 |
| split-conformal | 1488 | 5.8544 | 17.238 | 72.38 | — | — |

🔴 **quantile-LGBM coverage 43.19% έναντι 80% nominal είναι σοβαρό miscalibration** (λιγότερο
από το μισό του στόχου) — πρέπει να αναφέρεται ΠΑΝΤΑ μαζί με το pinball/MAE αν αυτή η μέθοδος
γραφτεί οπουδήποτε, όχι να αποσιωπάται (κανόνας Α4/§2 σημείο 6: coverage πάντα μαζί με
sharpness). Το split-conformal (72.38%) είναι πιο κοντά στο 80% αλλά σε μικρότερο n (1488 vs
2160 — τα πρώτα ~672 rows είναι το αναμενόμενο causal trailing-calibration warm-up του
`src/conformal.py`, ΟΧΙ leak). `march_2026` κλειδί υπάρχει και στα δύο JSON αλλά n=0 (καμία
πραγματική δεδομένα Μαρτίου για conformal ακόμα) — άρα μόνο 1 πραγματικό window, 1 μοντέλο
έναντι 1 εναλλακτικής. **Καμία headline probabilistic claim δεν μπορεί να γραφτεί πριν
τρέξει conformal σε 2ο window (Μάρτιος) και σε ≥2 point-μοντέλα (π.χ. + XGB weekly).**
Μικρό fix ανοιχτό: patch `overnight_summarize.py` ώστε να διαβάζει το conformal schema
σωστά (ώστε να μη ξαναγράψει "NO-MAE" σε επόμενο batch).

**§5.12ζ Recursive vs Direct — επανεξέταση σε 3 windows (2026-07-06) — ΔΙΟΡΘΩΝΕΙ σφάλμα μου:**

Σε προηγούμενη απάντηση δηλώθηκε λανθασμένα «το direct είναι σαφώς χειρότερο» βασισμένο σε
σύγκριση ΔΙΑΦΟΡΕΤΙΚΩΝ retrain cadences (weekly-recursive έναντι static-direct) — άκυρη
σύγκριση (last.md §2/Α3: «συγκρίσεις πάντα ίδιο window/gate/data»). Διορθώθηκε με σύγκριση
ΙΔΙΟΥ cadence (static), ΙΔΙΟΥ spec (`default,dense`), ΙΔΙΟΥ gate, σε 3 windows:

| window | recursive LGBM | direct LGBM | recursive XGB | direct XGB | νικητής |
|---|---|---|---|---|---|
| Q1 (Δεκ-Φεβ) | 19.970 | 18.627 | 19.728 | 18.220 | **direct** (Δ≈−1.3/−1.5) |
| Μάρτιος | 20.378 | 21.685 | 20.551 | 22.114 | **recursive** (Δ≈+1.3/+1.6) |
| Καλοκαίρι | 14.126 | 18.873 | 15.490 | 19.288 | **recursive** (Δ≈+3.8/+4.7) |

Πηγές: `runs/overnight_20260705/d_fill/{q1,summer}_{lgbm,xgb}_dir_default_dense.json` +
`d_fill/{q1,summer}_lgbm_rec_default_dense_seed7.json` + `runs/b3_ablation/{q1,summer}_xgb_rec/
xgb_price_dam_recursive_default-dense.json` (recursive XGB static anchor) +
`runs/overnight_20260705/b_march/march_{lgbm,xgb}_{rec,dir}_default_dense.json`.

**Validity-reviewer ετυμηγορία (όλοι οι 12 αριθμοί επαληθεύτηκαν απευθείας από τα JSON,
ίδιο context: `gate=strict`, `retrain=static`, `features=[calendar,lags,dense]` παντού):**

🔵 **ΟΥΤΕ «recursive > direct» ΟΥΤΕ «direct > recursive» μπορεί να γραφτεί ως καθολικό
εύρημα — PENDING, regime-dependent.** Το πρόσημο είναι συνεπές ΜΕΣΑ σε κάθε window (2
αλγόριθμοι συμφωνούν σε καθένα από τα 3), αλλά ΑΝΑΣΤΡΕΦΕΤΑΙ ανάμεσα σε windows (Q1 vs
Μάρτιος+Καλοκαίρι) — ακριβώς η ίδια δομή με το ήδη τεκμηριωμένο strategy×season interaction
του resfc/xborder (§5.6/§5.7), ΟΧΙ ένα καθαρό 2-στα-3 majority vote. Τίποτα στα ωμά actual
prices του Q1 δεν το ξεχωρίζει ως ανωμαλία (min/max/mean/negative-hours συγκρίσιμα με τα
άλλα windows — Καλοκαίρι έχει μάλιστα περισσότερες αρνητικές ώρες και υψηλότερο spike) —
άρα ΔΕΝ είναι data artifact, είναι πραγματικό regime-dependent φαινόμενο. Ενημερώνει και
διορθώνει το `last.md §3` σημείο 2 (βλ. εκεί).

**ΛΥΘΗΚΕ (follow-up batch, 2026-07-06) — validity-reviewer, 2 ξεχωριστά ευρήματα, ΟΧΙ ένα
συγχωνευμένο claim:**

**(1) 🟢 ΔΕΚΤΟ — στο weekly (deployable) cadence, recursive κερδίζει το direct ΠΑΝΤΟΥ:**

| window | recursive (μέση seeds) | direct LGBM | direct XGB | Δ (LGBM/XGB) |
|---|---|---|---|---|
| Q1 | 16.956 | 18.176 | 18.029 | −1.22 / −1.07 |
| Summer | 13.737 | 16.313 | 16.593 | −2.58 / −2.86 |

4/4 ανεξάρτητες συνθήκες (2 windows × 2 algos), ίδιο πρόσημο, |Δ|≫0.15 — καθαρό ACCEPTED.
**Αυτό αναιρεί το Q1-static «direct κερδίζει» ΓΙΑ ΤΟ ΠΡΑΓΜΑΤΙΚΟ deployable config**: στο
σωστό (weekly) retrain cadence, recursive είναι η σαφής επιλογή και στα δύο windows.
Πηγές: `runs/followup_20260705/{q1,summer}_{lgbm,xgb}_dir_weekly_default_dense.json`.

**(2) 🔵 Static cadence — παραμένει strategy×cadence interaction, ΔΕΝ συγχωνεύεται με το (1):**
Με το 4ο ανεξάρτητο static window (Οκτ-Νοε 2025: recursive 18.029/18.609, direct
22.545/22.805, Δ≈−4.2/−4.5) η καταμέτρηση static windows γίνεται 3-στα-4 υπέρ recursive
(Οκτ-Νοε, Μάρτιος, Καλοκαίρι) — μόνο το Q1-static ευνοεί direct. Ρητή ετυμηγορία
validity-reviewer: αυτό ΔΕΝ γράφεται ως «recursive > direct, καθολικό» — static και weekly
είναι διαφορετικά retrain regimes, και το Q1-static countersignal είναι αλγοριθμικά
επιβεβαιωμένο (2/2 algos, |Δ|>1.0), όχι θόρυβος. Μένει καταγεγραμμένο ως strategy×cadence
interaction (ίδια λογική με §5.6 resfc), ΟΧΙ ως εξαίρεση προς παράβλεψη.
Πηγές: `runs/followup_20260705/octnov_{lgbm,xgb}_{rec,dir}_default_dense.json`.

**Πρακτικό συμπέρασμα**: επειδή το headline χρησιμοποιεί weekly cadence (§5.12δ), το (1)
είναι αυτό που μετράει για production — recursive, χωρίς επιφύλαξη. Το (2) μένει ως
μεθοδολογική σημείωση/μελλοντικό ερώτημα (γιατί το static-Q1 συμπεριφέρεται διαφορετικά),
όχι ως κάτι εκμεταλλεύσιμο (καμία regime-switching πρόταση — δεν υπάρχει deployable use
case όπου θα χρησιμοποιούσαμε static-Q1 direct αντί για weekly recursive).

### 5.8 Robustness τελικού νικητή (P6, 2026-07-04) — ⚠️ ΠΡΟ-AEL, ΣΕ ΑΝΑΣΤΟΛΗ

`runs/p6_out/*`. Seeds 42/43/44 (weekly default Q1): 15.171/15.429/15.218 → std≈0.11.
Μάρτιος 2026: weekly 17.666 < monthly 17.968 (Δ=−0.302, 2ο ανεξάρτητο window).
~~🟢 Headline ΔΕΚΤΟ~~ → **ΑΝΑΚΛΗΘΗΚΕ**: όλα μετρήθηκαν ΠΡΙΝ το AEL (leaked crosslags).
Κρατιέται ΜΟΝΟ ως μεθοδολογικό υπόδειγμα (seeds+2ο window)· το νέο headline ορίζεται
από το §5.12α (leak-free cadence + seeds).

---

## 6. Κάλυψη ανά μοντέλο/στρατηγική (honesty — ενημερώθηκε 2026-07-04)

- **LGBM**: πλήρης κάλυψη (18 specs Q1 recursive + direct-Q1 + Δεκ×2 + καλοκαίρι + E1/E2 + SS +
  P1-P6). Το μόνο μοντέλο με πλήρη ablation.
- **XGB**: δική του 18-spec Q1 + καλοκαίρι 10-spec (P2) + direct-Q1 10-spec (P4) + weekly run
  (P3) — καλή κάλυψη, συμφωνεί με LGBM σχεδόν παντού.
- **LEAR**: mini-ablation 6 specs (P2) — αρκετή για να ξέρουμε ότι θέλει ΔΙΚΟ του feature set.
  Όχι πλήρης ablation (δεν χρειάζεται — ρόλος fallback μόνο).
- **MLP**: mini-ablation 3 specs (P2). recursive μόνο (direct δεν υποστηρίζεται στον κώδικα).
- **LSTM**: καμία ablation — πρώτα το calibration bug (§5.5), μετά οτιδήποτε άλλο.
- **SS**: καλωδιωμένο ΜΟΝΟ για recursive στον κώδικα. SS×weekly αδοκίμαστο.
- **Seeds**: μόνο το headline έχει 3 seeds· όλα τα οριακά ΔMAE αλλού είναι single-seed.
- **Meteo**: reanalysis proxy παντού (αισιόδοξο άνω όριο για την ομάδα meteo).

## 7. PENDING (ρητή λίστα — τίποτα εδώ δεν είναι συμπέρασμα)

1. **Summer xborder-lagged θετικό σήμα** (−0.52, 1 σημείο) — θέλει monthly/weekly καλοκαίρι ή 2ο seed.
2. **Weekly LGBM+XGB ensemble** (−0.148, 1 σημείο, στο όριο) — θέλει Μάρτιο ή seeds.
3. **resfc-in-direct**: LGBM ουδέτερο (−0.04) vs XGB βλάπτει (−0.32) — algo-dependent, ανοιχτό.
4. **fuel window-confound στο direct** (Δεκ −0.68 vs Q1 ~0) — αδιευκρίνιστο.
5. **loadlags καλοκαίρι**: LGBM ουδέτερο vs XGB +0.42 — ασυνεπές.
6. **SS×weekly** — αδοκίμαστο (το SS οφέλη ίσως στοιβάζονται και με weekly).
7. **LSTM calibration bug** (bias +41, ποτέ αρνητικές τιμές· ύποπτο: hardcoded teacher forcing
   στο train + μικρό capacity) — θέλει στοχευμένο debugging, όχι ξανατρέξιμο.
8. ~~solar_fc_dayahead ύποπτο 2h shift~~ ✅ **ΕΛΥΘΗ 2026-07-04 → §5.9**: ήταν πραγματικό
   συστημικό timezone misalignment (4 ρολόγια στο ίδιο parquet), διορθώθηκε δομικά στο
   `data.py` + rebuild. ΝΕΟ ανοιχτό: re-run confirmτο headline στα ευθυγραμμισμένα δεδομένα.
9. **`henex_premarket` ομάδα — BLOCKED, design doc 2026-07-06** (`docs/features/henex_premarket/design.md`,
   feature-eng Στάδιο 1, ενώ έτρεχε το followup batch — καμία conda χρήση). Ευρήματα (όχι
   υποθέσεις): (α) cached parquet (`henex_premarket_hourly.parquet`, 615KB) καλύπτει
   2020-11-01→**2026-01-01 μόνο** — Q1 2026 κάλυψη 35.6% (μόνο Δεκ, Φεβ 0%), Μάρτιος 2026 0%,
   καλοκαίρι 2025 100% (μόνο ΕΝΑ πλήρες window διαθέσιμο σήμερα — T7 ≥2 windows αδύνατο χωρίς
   backfill)· (β) `data/raw/henex/premarket_summary/` άδειος φάκελος (0 αρχεία), κανένα
   `fetch_henex*.py`, και το `src/data_future.py` (μοναδικό module με τη λογική) ψάχνει σε
   ΛΑΘΟΣ path (`data/raw/henex_premarket` αντί `data/raw/henex/premarket_summary`)· (γ)
   `src/data_future.py` είναι ορφανό — ΔΕΝ τροφοδοτεί το `hourly.parquet`, άρα οι 3 στήλες
   (`pm_buy/sell/net_nom_mw`) δεν υπάρχουν πουθενά στο ενεργό feature store· `lagscan.py`
   δεν μπορεί να τρέξει πριν από staging merge στο `data.py`· (δ) T1 (gate timing vs 12:00
   CET D-1) ΑΝΕΠΙΒΕΒΑΙΩΤΟ — μόνο οικονομική υπόθεση, καμία πρωτογενής πηγή. **PENDING —
   δεν ξεκίνησε το Στάδιο 2 (lagscan)**, χρειάζεται πρώτα: primary-source gate-timing proof
   + data backfill/fetcher + path fix + staging merge στο data.py.
10. **xb_lag1_h0** — vetted νέο feature candidate (βλ. §8.1).
11. **Headline seed-lock (§5.12δ)** — το Block D seed-check έτρεξε στο ΛΑΘΟΣ retrain cadence
    (static αντί για weekly)· χρειάζεται re-run πριν κλειδώσει το `default,dense` weekly headline.
12. **Block C LOAD ablation arms (§5.12γ)** — genlags/loadlags/meteo/dense αλλάζουν πρόσημο σε
    κάθε συνθήκη· χρειάζεται `check_crosslag_fairness.py --task load` (ποτέ δεν έτρεξε) πριν
    από οποιοδήποτε claim.
13. **Conformal 2ο window/2ο μοντέλο (§5.12στ)** — μόνο Q1 quantile-LGBM/split-conformal
    υπάρχουν· quantile-LGBM coverage 43.19% έναντι 80% nominal είναι μη διορθωμένο miscalibration.
14. **`meteo_vintage` — ingest-audit PASS, ΠΡΟΧΩΡΑ σε wiring/batch (2026-07-10,
    design doc: `docs/features/meteo_vintage/design.md`)**. Open-Meteo Previous Runs
    D-1/D-2 buckets, task=load ΜΟΝΟ (168 `wv_*` στο hourly_load.parquet, 0 στο price).
    T1 ✅ (day1 issue = valid−24h ⇒ νόμιμο ⟺ h ≤ 23−gap· g12: h≤11, g14: h≤9· day2
    πάντα νόμιμο dam) · T2 ✅ (lagscan: |corr|≤0.099 temp / ≤0.18 swr, επίπεδο-συμμετρικό,
    καμία spike — `logs/g5_lagscan_wv_{temp,swr}.log`) · T3 ✅ (hour-profile: swr peak
    11:00, temp 13:00) · βήμα-4 πληρότητα ✅ (0 NaN/0 missing στα 3 contest windows,
    ζεύγη day1/day2 πλήρη). ⚠️ Μάθημα: unclassified στήλες πέφτουν στο `other` ⊂ default
    — τα ωμά `wv_*` χρειάζονται ΡΗΤΟ δομικό αποκλεισμό στο classify_columns (μπαίνει με
    το wiring). Pre-registered (T7/T8): Δ<0 σε ≥2 windows, μέγεθος 50-100% του oracle
    (Q1 −94.9 / summer −19.1 static)· vintage ΔΕΝ επιτρέπεται να κερδίζει το oracle
    πέρα από noise. Verdict μετά το G7 vintage batch — ΤΙΠΟΤΑ εδώ δεν είναι ΔΕΚΤΟ ακόμα.
    **Batch αποτέλεσμα (2026-07-10 23:02, 12/12 runs 0 FAILED, LGBM weekly rec seed 42,
    `runs/load_contest/*_{mv,densemv}.json`):** T7 pre-gate **PASS 6/6** — mv ΔMAE vs base
    ίδιου gate/window: g12 q1 −41.2 / summer −94.4 / octnov −20.6 · g14 −33.3 / −107.8 /
    −24.6 (3 ανεξάρτητα windows, ίδιο πρόσημο, |Δ|≫0.15)· T8 PASS (vintage ≤ oracle
    παντού: q1 214.8 > oracle-weekly 157.4· summer 264.3 > 237.2 — κρατά 42%/78% του
    oracle οφέλους q1/summer). **vs ΑΔΜΗΕ: octnov ΝΙΚΗ 4/4 κελιά** (g12 mv 133.09 /
    densemv 128.46 · g14 mv 132.38 / densemv 132.87 έναντι 146.81) — 1 window μόνο,
    window-specific claim· q1/summer ο ΑΔΜΗΕ κρατά (175.6/170.8 vs 214.8+/250.5+).
    Εκκρεμούν πριν από ΔΕΚΤΟ: ανθρώπινη αποδοχή + (προαιρετικά) validity-reviewer ·
    XGB confirm (Batch 2 τρέχει) · seeds για headline-level claim. 1 algo / 1 seed ακόμα.
    **Επιβεβαίωση XGB (Batch 2, 2026-07-11, `runs/load_contest/*_xgb_recw_*.json`)**: dense
    §2 6/6 σε XGB → 12/12 σε 2 αλγορίθμους· XGB dense κερδίζει ΑΔΜΗΕ στο octnov ΧΩΡΙΣ
    καιρό (137.46 g12 / 143.81 g14). **Επιβεβαίωση seeds (overnight, `runs/load_contest_seeds/`)**:
    octnov ΟΛΑ 6 configs (dense/mv/densemv × g12/g14) < ΑΔΜΗΕ και στους 3 seeds {42,7,123}·
    densemv std 0.14-0.17 MW. Το octnov claim seed-robust· παραμένει window-specific.
15. **SS×weekly για load — PENDING (πρώτο σήμα, overnight 2026-07-11, `runs/load_contest_ss/`)**.
    LGBM weekly recursive `--ss --ss_decay linear` vs non-SS ίδιο seed42/gate/window/spec:
    **SS ΒΟΗΘΑΕΙ 8/8** — octnov ΔMAE −4.9 έως −7.0, summer −5.1 έως −12.5 (2 windows ×
    2 gates × 2 specs {base,dense}). Πιάνει §2 (ίδιο πρόσημο ≥2 ανεξάρτητα windows, |Δ|≫0.15).
    Σημ.: octnov g12 base+SS=146.67 ≈ ΑΔΜΗΕ ακόμα και χωρίς dense/meteo.
    **XGB confirm (dense, g12, 2026-07-11, `runs/load_contest_ss/*_xgb_recwss_g12_dense.json`):
    MIXED — ΟΧΙ «παντού».** summer 336.10→330.60 (**−5.50**, βοηθάει, cross-algo με LGBM −10.6)·
    octnov 137.46→**137.42** (**−0.04**, ΚΑΤΩ από noise floor → SS ΔΕΝ βοηθάει στο XGB, ενώ
    στο LGBM βοηθούσε −5.1). Δηλαδή το LGBM 8/8 ΔΕΝ γενικεύεται: το octnov-SS όφελος ήταν
    algo-specific· μόνο το **summer-SS** αντέχει cross-algo (§2 met εκεί). Ετυμηγορία:
    SS×weekly = ΒΟΗΘΑΕΙ ΚΑΛΟΚΑΙΡΙ (2 algos), algo-dependent στο octnov, q1 αδοκίμαστο.
    ΠΑΡΑΜΕΝΕΙ PENDING (όχι «universal SS win»). 1 seed ακόμα.

## 8. Επόμενα βήματα

### 8.1 xb_lag1_h0 — vetted candidate (πέρασε τον §1 έλεγχο στα χαρτιά, 2026-07-04)

Ιδέα: `xb_lag1` ΜΟΝΟ για ώρα-στόχο 00:00 — το lag1 της 00:00 της D είναι η 23:00 της D-1,
δημοσιευμένη ~13:00 D-2, δηλαδή ΠΡΙΝ το gate 12:00 D-1 → νόμιμο. Για ώρες 01:00-23:00 το lag1
πέφτει εντός της D → παράνομο → NaN (τα δέντρα χειρίζονται NaN φυσικά).

- **§1(α) δημοσίευση**: ✅ νόμιμο μόνο για h==0, τεκμηριωμένο παραπάνω.
- **§1(β) lag-scan**: αναμενόμενο peak στο lag1 στο υποσύνολο h==0 (ήδη ξέρουμε same-hour
  coupling 0.91 IT-SUD → το lag1 cross-hour θα είναι ασθενέστερο αλλά υπαρκτό).
- **Υλοποίηση**: στο `data.py`, στο σημείο που φτιάχνονται τα xb lags:
  `xb_price_bg_lag1_h0 = xb_price_bg.shift(1).where(hour==0)` (ομοίως itsud), μπαίνουν στην
  ομάδα `xborder`. Rebuild parquet με backup+σύγκριση (καμία υπάρχουσα στήλη δεν αλλάζει).
- **Μέτρηση**: ΔMAE συνολικό ΚΑΙ ειδικά στο υποσύνολο ώρας-0 (εκεί ζει το σήμα — 1/24 των
  γραμμών, το συνολικό ΔMAE θα είναι εκ κατασκευής μικρό). Κριτήρια §2 κανονικά.
- **Προσδοκία (έντιμη)**: μικρό effect· αξίζει ως καθαρό επιστημονικό σημείο για τη διπλωματική
  («πώς αξιοποιείς νόμιμα ένα κατά τα άλλα leakage-prone feature»), όχι ως headline mover.

### 8.2 Κύριο επόμενο στάδιο (μεγάλη εικόνα)

Data ✅ (TZFIX) → Engine ✅ (AEL + dense y-path poison 2026-07-05) → Feature selection
leak-free ✅ (§5.11) → **Cadence/headline + Μάρτιος + LOAD + SS — ΕΔΩ ΕΙΜΑΣΤΕ (overnight
2026-07-05, §5.12)** → Probabilistic layer (conformal, smoke απόψε/πλήρες Β4) → Product.
(Το παλιό «Robustness ✅» του §5.8 ανακλήθηκε — προ-AEL.)

> **Διευκρίνιση (2026-07-04, οδηγία χρήστη)**: το «κλείδωμα» αφορά ΜΟΝΟ το headline/product
> config — ΔΕΝ κλειδώνουμε 1 μοντέλο. Η ακαδημαϊκή έρευνα προηγείται του προϊόντος: το
> conformal στρώμα αξιολογείται σε ≥2 μοντέλα, και το ερευνητικό πρόγραμμα (load ablation,
> SS×weekly, LSTM, henex_premarket) συνεχίζεται παράλληλα.

Επόμενο: **split-conformal intervals**, model-agnostic (πάνω σε forecast JSONs — LGBM weekly
ΚΑΙ τουλάχιστον ένα ακόμα μοντέλο, π.χ. XGB weekly): residuals από rolling calibration window
(4-8 εβδ., ποτέ από test), quantiles ανά hour-of-day, αξιολόγηση με pinball + empirical
coverage + sharpness σε Q1 + Μάρτιο. Baseline: quantile-LGBM (α=0.1/0.5/0.9). Πλήρεις κανόνες:
`.claude/skills/energy-forecast/SKILL.md §Conformal`.

**Μείζον ερευνητικό ανοιχτό (όχι «δευτερεύον»): task=load πλήρες ablation** — για το load δεν
έχει γίνει ΚΑΝΕΝΑ feature engineering/ablation πέρα από το E2 (meteo). Ό,τι ξέρουμε για ομάδες
features ισχύει μόνο για price. **→ ΤΡΕΧΕΙ στο overnight 2026-07-05 (Block C, §5.12γ):
7 specs × LGBM+XGB × rec+dir × Q1+summer.**

Λοιπά (με σειρά): solar_fc 2h-shift έλεγχος (§7.8 — νέο εργαλείο `solar_shift_check.py`,
πιθανό timezone red herring: index=UTC, peak 10:00 UTC = 12-13:00 τοπική) · xb_lag1_h0 (§8.1) ·
SS×weekly · henex_premarket.
