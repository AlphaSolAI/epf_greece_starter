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

`runs/ss_out/*.json` + `runs/p5_out/lgbm_ss_linear_default_monthly_q1.json`.

| Εύρημα | Νούμερα | Κατάσταση |
|---|---|---|
| SS-linear = το μόνο σχήμα που βελτιώνει | static Q1: 16.096→15.830 (−0.266), far-offsets 17.59→17.36· exp/step χειρότερα | 🟢 ΔΕΚΤΟ (υπογραφή exposure-bias, Bengio 2015) |
| **SS ΔΕΝ είναι redundant με retrain** | monthly Q1: 15.795→15.552 (**−0.243**) σε καθαρό `default` | 🟢 ΔΙΟΡΘΩΣΗ προηγούμενου συμπεράσματος — το «redundant» είχε μετρηθεί σε xborder-μολυσμένο config |
| SS×features interaction | υπό SS: resfc −0.04→**+1.10**, loadfc →+0.46, loadlags +0.22→**−0.32** (αναστροφή) | 🟢 Επιβεβαιωμένο — το μοντέλο ακουμπά στα πάντα-αξιόπιστα day-ahead exogenous όταν τα y-lags γίνονται θορυβώδη |
| SS×weekly | — | ⬜ ΔΕΝ έχει τρέξει ποτέ (επόμενος λογικός έλεγχος) |

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

### 5.12 OVERNIGHT 2026-07-05 — cadence/Μάρτιος/LOAD/SS/conformal (⏳ τρέχει)

**Batch**: `scripts/overnight_20260705.sh` (εκκίνηση 2026-07-05 βράδυ, detached).
Outputs: `runs/overnight_20260705/{a_cadence,b_march,c_load,d_fill,e_ss,f_conformal}/`,
log: `logs/overnight_20260705_master.log`.
Σύνοψη το πρωί: `conda run -n epf --no-capture-output python -X utf8 scripts/overnight_summarize.py`
→ `results/overnight_20260705.csv`.

Γέμισμα το πρωί (ΜΟΝΟ από το CSV/summarizer, κριτήρια §2):
- **§5.12α Cadence (Block A)**: weekly/monthly × {default,dense · default,−resfc · default}
  × LGBM+XGB × Q1+summer, recursive. → πίνακας + απόφαση **ΝΕΟΥ HEADLINE** (μαζί με seeds
  του Block D). _[πίνακας εδώ]_
- **§5.12β Μάρτιος tie-break (Block B)**: 7 specs × LGBM+XGB × rec+dir static
  (2026-03-01→03-20). Λύνει: resfc/lean/genlags/bare MIXED + πρώτα dense×direct κελιά.
  _[Δ-πίνακας + τελικά verdicts εδώ]_
- **§5.12γ LOAD ablation (Block C)**: πρώτο πλήρες load ablation (7 specs × 2 algos ×
  rec+dir × 2 windows, MAE σε MW). Ερωτήματα: κυριαρχεί το loadfc (TSO forecast);
  meteo; loadlags; dense; _[πίνακας εδώ]_
- **§5.12δ Seeds + dense×direct (Block D)**: seeds 7/123 στο default,dense (το 42 υπάρχει
  από Β3) → std για headline κριτήριο Α3. _[εδώ]_
- **§5.12ε SS (Block E)**: SS-linear×3 σε default & default,dense (Q1+summer, static).
  Ερώτημα: το SS κέρδος επιβιώνει leak-free; προσθέτει πάνω στο dense; _[εδώ]_
- **§5.12στ Conformal smoke (Block F)**: split-conformal + quantile-LGBM στο q1 weekly
  default,dense. Coverage/sharpness πρώτη εικόνα (πλήρες Β4 χωριστά). _[εδώ]_

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
9. **`henex_premarket` ομάδα** — παραμένει εντελώς υποαξιοποιημένη (κανένα τεστ ποτέ).
10. **xb_lag1_h0** — vetted νέο feature candidate (βλ. §8.1).

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
