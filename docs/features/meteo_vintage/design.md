# Feature Design Doc — meteo_vintage (D-1 vintage weather forecast, task=load)

> TDD pre-registration (feature-eng Στάδιο 1) — γράφτηκε 2026-07-10, ΠΡΙΝ το wiring
> του feature group. Το data ingest (fetcher/merge/leakage checks) είχε ήδη γίνει
> νωρίτερα την ίδια μέρα (GOALS.md G5 «DATA DONE») — αυτό το doc κλειδώνει το
> availability rule + τα acceptance tests ΠΡΙΝ το `feature_availability.py` wiring
> και ΠΡΙΝ από κάθε training run με την ομάδα.

## 1. Ταυτότητα

- **Feature/πηγή:** Open-Meteo **Previous Runs API** (`previous-runs-api.open-meteo.com`) —
  αρχειοθετημένες προγνώσεις καιρού με σταθερά lead-time buckets:
  `<var>_previous_day1` = η τιμή που προβλέφθηκε **24h πριν το valid time**,
  `_previous_day2` = **48h πριν**. 5 πόλεις (Αθήνα/Θεσσαλονίκη/Πάτρα/Λάρισα/Ηράκλειο)
  × 7 μεταβλητές (temperature_2m, relative_humidity_2m, precipitation, cloud_cover,
  wind_speed_10m, wind_gusts_10m, shortwave_radiation) + `gr_mean` aggregate.
  Fetcher: `src/fetch_open_meteo_vintage.py` → `data/processed/weather_vintage_hourly.parquet`.
  **ΔΙΟΡΘΩΣΗ σχεδίου (καταγεγραμμένη στο G5):** το αρχικά προτεινόμενο «Historical
  Forecast API» ΔΕΝ δίνει vintage δεδομένα (stitches τις πρώτες ώρες κάθε run) — το
  Previous Runs είναι το σωστό API.
- **Ομάδα (flag) στο feature_availability.py:** `meteo_vintage` — **ΝΕΑ**, ΕΚΤΟΣ
  `DEFAULT_GROUPS` (όπως τα xborder/engfc/ramp).
- **Στήλες parquet:** 168 στήλες `wv_*` **ΜΟΝΟ στο `hourly_load.parquet`** (task=load
  join στο `src/data.py::process_hourly` — το price pipeline ΑΘΙΚΤΟ, 0 wv στήλες στο
  `hourly.parquet`, επιβεβαιωμένο 2026-07-10):
  42 `wv_<loc>_<var>_day1` + 42 `wv_<loc>_<var>_day2` + 84 `_missing` flags
  (σύμβαση `_fill_weather`: ffill limit=6 → 0-fill + flag· coverage από 2024-02-01,
  0 NaN / 0 missing στα 3 contest windows q1/summer/octnov).
- **Availability rule (ρητό):** Τα ΩΜΑ buckets ΔΕΝ γίνονται ΠΟΤΕ features. Το feature
  set της ομάδας είναι **gate-aware blend** σε effective στήλες, χτισμένες on-the-fly
  (pattern `engfc`/`ramp`):
  `wveff_<x> = wv_<x>_day1 αν issue(day1) ≤ cutoff, αλλιώς wv_<x>_day2`
  όπου για στόχο την ώρα h της ημέρας D: issue(day1) = D-1 h:00, issue(day2) = D-2 h:00,
  cutoff = D-1 (23−gap):00. Άρα **day1 νόμιμο ⟺ h ≤ 23 − gap**:
  - **g12** (δικό μας deployable, gap=12, cutoff 11:00 CET D-1): h ≤ 11 → day1, h ≥ 12 → day2
  - **g14** (ΑΔΜΗΕ-aligned, `--delay 14`, cutoff 09:00 CET D-1): h ≤ 9 → day1, h ≥ 10 → day2
  Το day2 είναι ΠΑΝΤΑ νόμιμο για dam (issue D-2 < κάθε D-1 cutoff). Το gap έρχεται
  από `GateSpec.gap_hours()` → κληρονομεί το `--delay` override αυτόματα (ίδιο pattern
  με το AEL crosslag gap). **ΜΟΝΟ market=dam/idm**: σε multi-day blocks (forward) θα
  χρειάζονταν day3+ buckets που δεν υπάρχουν στο parquet → ρητό ValueError, όχι σιωπηλό leak.
- **Δομικός αποκλεισμός των ωμών στηλών:** τα `wv_*` (day1/day2/missing) ταξινομούνται
  σε ΚΑΜΙΑ ομάδα (ούτε `other` — εκεί πέφτουν σήμερα ως unclassified, και το `other`
  είναι ΜΕΣΑ στο default!). Χωρίς αυτόν τον κανόνα, ένα μελλοντικό `--features default`
  ή `all` σε task=load θα σέρβιρε ωμό day1 και σε ώρες h > 23−gap = leak ~1-12h.
  Ίδια φιλοσοφία με το «same-day xborder δομικά εκτός parquet».

## 2. Μηχανισμός & pre-registered προσδοκίες

- **Γιατί να βοηθάει:** θερμοκρασία → heating/cooling load (η ισχυρότερη εξωγενής
  σχέση του STLF)· shortwave → behind-the-meter PV που «τρώει» το μετρούμενο φορτίο.
  Το oracle meteo (observed, Archive API) το έχει ήδη δείξει εμπειρικά στο Phase 1 —
  εδώ μετράμε πόσο από το όφελος επιβιώνει με ΠΡΑΓΜΑΤΙΚΗ D-1 πρόγνωση (έντιμο gate).
- **Baselines αναφοράς (Phase 1, static, seed 42, ίδιο snapshot base στηλών):**
  Q1 base 254.99 → +meteo(oracle) **160.11** (Δ=−94.9) · summer base 372.01 →
  +meteo(oracle) **352.86** (Δ=−19.1). Weekly oracle: Q1 157.37 / summer 237.19.
- **Αναμενόμενο πρόσημο ΔMAE (vintage vs ίδιο base χωρίς meteo):** ΒΕΛΤΙΩΣΗ και στα
  δύο windows. **Αναμενόμενο μέγεθος:** 50-100% του oracle οφέλους — Q1: Δ ∈ [−95, −50]·
  summer: Δ ∈ [−19, −5] (το vintage forecast error είναι μικρό: MAE(day1 vs actual)
  = 0.86 °C / 13.1 W/m², corr 0.986-0.993). Υπερκαλύπτει το |Δ|>0.15 (βλ. σημείωση
  κλίμακας load στο GOALS G7 — το ουσιαστικό τεστ είναι το ίδιο πρόσημο σε ≥2 windows).
- **Δεσμευτικό άνω φράγμα (T8):** MAE(vintage) ≥ MAE(oracle) − noise ανά
  window/config. Αν το vintage ΚΕΡΔΙΣΕΙ το oracle πέρα από seed noise (~1-3 MW σε
  αυτή την κλίμακα) → red flag, targeted poisoning πριν από οποιοδήποτε claim.
- **Contest προσδοκία (weekly, vs ΑΔΜΗΕ MAE 175.56/170.76/146.81 q1/summer/octnov):**
  Q1: oracle 157.4 < ΑΔΜΗΕ 175.6 → το vintage ΜΠΟΡΕΙ να πέσει εκατέρωθεν του ΑΔΜΗΕ —
  ανοιχτό εμπειρικό ερώτημα (αυτό απαντά το batch). Summer: ακόμα και το oracle χάνει
  (237 vs 171) → ΔΕΝ αναμένεται νίκη vs ΑΔΜΗΕ το καλοκαίρι. octnov: άγνωστο (κανένα
  oracle σημείο ακόμα)· το καθαρό dense ήδη κερδίζει εκεί (142.97 vs 146.81 g12).
- **Πού αναμένεται να δρα:** task=load ΜΟΝΟ · recursive (G7 grid) · ισχυρότερα χειμώνα
  (heating) απ' ό,τι καλοκαίρι — ίδιο εποχιακό pattern με το oracle Phase 1.
- **Αναμενόμενο lagscan peak:** ομαλή, ΧΑΜΗΛΗ |corr| (≤~0.3) συμμετρική γύρω από k=0,
  ΧΩΡΙΣ αιχμηρό spike — αργά μεταβαλλόμενο εξωγενές covariate. Το k=0 ΔΕΝ είναι
  «βολικό leak» εδώ: η νομιμότητα κρίνεται από το issue time του bucket (24/48h πριν),
  όχι από τη στιγμή του valid time. Anti-pattern αναφοράς: xborder same-day spike 0.91.
- **Αναμενόμενο hour-profile:** shortwave → μεσημεριανό peak ~11:00-13:00 (ηλιακή
  τροχιά)· temperature → απογευματινό max. Shift ≥2h από αυτά ⇒ fetcher/TZ bug.
- **Αναμενόμενο FI rank ομάδας:** κάτω από lags/calendar, στην τάξη του dense·
  temperature πρώτη μεταξύ των wv μεταβλητών.

## 3. Acceptance tests (κλειδωμένα — T1-T8)

| Test | Τι ελέγχει | Κριτήριο PASS | Στάδιο | Κατάσταση |
|---|---|---|---|---|
| T1 | Gate timing | day1 issue = valid−24h (ορισμός API + εμπειρικός έλεγχος) · κανόνας h ≤ 23−gap ⇒ issue ≤ cutoff, ΠΡΙΝ το 12:00 CET D-1 | 2 | ✅ (βλ. §6) |
| T2 | Lagscan | Χαμηλό \|corr\| χωρίς spike, συμμετρικό γύρω από k=0 · \|corr\|<0.85 | 2 | ✅ (βλ. §6) |
| T3 | Hour-profile | shortwave peak ~11:00-13:00 · temperature απογευματινό | 2 | ✅ (βλ. §6) |
| T4 | Non-VOID arm | `#features` ΑΛΛΑΖΕΙ base↔mv στο πρώτο log (17 → 17+84 wveff) · wv στήλες ΜΟΝΟ στο load parquet | 3-4 | εκκρεμεί (smoke) |
| T5 | Poisoning | `preflight_check.py --poison` PASS ΜΕΤΑ το feature_availability.py wiring | 3 | εκκρεμεί |
| T6 | Reproducibility | Control run base spec: MAE = 254.9893 ΤΑΥΤΟΣΗΜΟ (ομάδα opt-in ⇒ μηδενική επίδραση σε υπάρχοντα configs) | 3 | εκκρεμεί |
| T7 | §2 gate | ΔMAE < 0 (βελτίωση) σε ≥2 ΑΝΕΞΑΡΤΗΤΑ windows, \|Δ\|>0.15 | 5 | εκκρεμεί (batch) |
| T8 | Too-good-to-be-true | MAE(vintage) ≥ MAE(oracle)−noise ανά window · FI rank εύλογο · Δ όχι >> pre-registered | 5-6 | εκκρεμεί |

Επιπλέον δομικό τεστ (unit, tests/test_feature_availability.py): ωμά `wv_*` σε ΚΑΜΙΑ
ομάδα (ούτε `other`) · `wveff_*` → `meteo_vintage` · `meteo_vintage` ΕΚΤΟΣ default ·
day1_ok(h,gap): (12,h=11)=True, (12,h=12)=False, (14,h=9)=True, (14,h=10)=False.

## 4. Batch matrix (στάδιο 4 — κλιμακούμενη εκτέλεση)

Κλίμακα 1 → 2 → 3, κάθε σκαλί gate για το επόμενο:

1. **Smoke (static, ~2'/run):** Q1 lgbm rec static — `base` (control T6, αναμένεται
   254.9893 ΤΑΥΤΟΣΗΜΟ) + `base,meteo_vintage` (T4 #features + πρώτο Δ σημείο +
   σύγκριση με oracle 160.11 για T8). → `runs/feat_meteo_vintage/`
2. **Κύριο batch (G7 override — απόφαση χρήστη: weekly ΜΟΝΟ, όχι το generic static
   του feature-eng):** LGBM weekly recursive × {`mv` = calendar,lags,roll,meteo_vintage ·
   `densemv` = calendar,lags,roll,dense,meteo_vintage} × {q1, summer, octnov} ×
   {g12, g14} = **12 runs** → `runs/load_contest/` (ίδια ονοματολογία
   `<win>_lgbm_recw_<gate>_<slug>.json` με το Batch 1 για το load_contest_report.py).
3. **Βαρύτερο (chained, ήδη εγκεκριμένο):** `scripts/load_contest_xgb_weekly.sh`
   Batch 2 (30 runs) — XGB confirm του dense candidate + XGB βάση για μελλοντικό mv.

- Εκτίμηση: weekly run ≈ 12-25' (Batch 1: 30 runs / ~4.5h) → σκαλί 2 ≈ 2.5-5h,
  σκαλί 3 ≈ 4-5h. Detached (Start-Process, ASCII args), ΕΝΑ conda process, σειριακά.
- Benchmark στήλες πίνακα (κανόνας G7): ΑΔΜΗΕ = MAE(load_fc,y) ανά window ·
  naive floor = MAE(y_lag168,y) · καθαρό δικό μας baseline = `calendar,lags,roll`.

## 5. Γνωστοί κίνδυνοι για ΑΥΤΟ το feature

- **Bucket ≠ ώρα έκδοσης:** το API δίνει ΜΟΝΟ 24ωρα buckets — η ανά-ώρα επιλογή
  day1/day2 είναι Η υλοποίηση του availability rule· λάθος όριο (π.χ. h ≤ 12 αντί
  h ≤ 11 στο g12) = 1h leak, αόρατο σε poisoning (exogenous covariate — τα AEL tests
  ΔΕΝ το πιάνουν). Γι' αυτό το όριο κλειδώνει με unit tests ΠΡΙΝ το run.
- **TZ quirk πηγής:** ίδιο fixed-UTC+1 με το Archive API — αν ο loader δεν εφάρμοζε
  `_fixed_utc1_to_cet_naive_index` ⇒ σιωπηλό 1-2h misalignment. Επιβεβαιωμένο
  εμπειρικά (cross-corr vs `w_gr_mean_*`: DJF 0.969 / JJA 0.988 ΑΚΡΙΒΩΣ σε k=0).
- **`other`-fallthrough leak (νέο μάθημα, 2026-07-10):** unclassified στήλες πέφτουν
  στο `other` που είναι ΜΕΣΑ στο default → κάθε νέα ωμή στήλη που μπαίνει στο parquet
  χωρίς ταυτόχρονο classification rule είναι ΣΙΩΠΗΛΑ επιλέξιμη. Το G7 γλίτωσε επειδή
  τρέχει explicit specs. Δομικό fix εδώ: ρητός αποκλεισμός `wv_*` στο classify_columns.
- **Εποχιακό flip (resfc lesson):** το oracle όφελος είναι 5× μικρότερο το καλοκαίρι —
  αν το vintage φλιπάρει πρόσημο σε ένα window, ΔΕΝ γίνεται claim (κριτήριο §2 ούτως ή άλλως).
- **Pre-2024-02 κενό coverage:** ΟΧΙ ffill πάνω από το κενό (θα κατασκεύαζε πρόγνωση
  που δεν υπήρξε) — 0-fill + `_missing` flag· τα trees μαθαίνουν το regime από το flag.
  Τα 3 eval windows είναι 2025-26 (0 missing rows) — ο κίνδυνος αφορά ΜΟΝΟ το training
  tail και είναι flagged.
- **Ένα window ΔΕΝ αρκεί** ούτε για τη νίκη-ΑΔΜΗΕ (αν έρθει): κριτήριο §2 + ρητή
  δήλωση caveat ΑΔΜΗΕ (μέθοδος άγνωστη, πιθανές post-gate αναθεωρήσεις load_fc ≥10%).

## 6. Audit evidence (στάδιο 2 — ingest-audit βήματα 1-4)

- **T1 γραπτή απάντηση (βήμα 1):** Ο πάροχος (Open-Meteo) ΔΕΝ «δημοσιεύει σε ώρα
  ημέρας» — σερβίρει per-hour buckets με ορισμό lead time: day1 = προβλέφθηκε 24h πριν
  το valid time ⇒ για στόχο (D,h) issue = D-1 h:00. Στο gate μας (απόφαση 12:00 CET D-1,
  data cutoff 11:00 CET για g12 / 09:00 για g14): day1 νόμιμο ⟺ h ≤ 23−gap (g12: h≤11 ✓
  πριν το gate· g14: h≤9)· day2 (issue D-2 h:00) νόμιμο για ΟΛΕΣ τις ώρες dam.
  Πηγή ΔΙΑΦΟΡΕΤΙΚΟΥ μηχανισμού από το target (NWP μοντέλα vs ΑΔΜΗΕ auction/SCADA) —
  κανένα xborder-style same-auction coupling. Actuals ΔΕΝ χρησιμοποιούνται (μόνο
  forecast buckets) ⇒ εκτός AEL crosslag οικογενειών by construction.
  Εμπειρική επιβεβαίωση ότι είναι ΟΝΤΩΣ forecast (όχι relabeled actuals):
  MAE(wv_day1, actual) = 0.86 °C / 13.1 W/m² με corr 0.986-0.993 — πραγματικό,
  μη-μηδενικό forecast error (oracle θα έδινε 0.00).
  Υπολειπόμενη αβεβαιότητα (δηλωμένη): ο ακριβής χρόνος του underlying model run —
  αν ήταν αργότερα από valid−24h, το όριο h≤23−gap θα δεχόταν issue έως ~1h μετά το
  cutoff στο οριακό h. Mitigation: το head-to-head vs ΑΔΜΗΕ γίνεται στο g14 (2h
  επιπλέον περιθώριο)· καταγράφεται ως ρητή υπόθεση στη διπλωματική.
- **T2 lagscan (βήμα 2):** `lagscan.py --task load --col wv_gr_mean_temperature_2m_day1`
  και `--col wv_gr_mean_shortwave_radiation_day1` → logs:
  `logs/g5_lagscan_wv_temp.log`, `logs/g5_lagscan_wv_swr.log` (2026-07-10, φρέσκα —
  επαναλαμβάνουν το G5 αποτέλεσμα: χαμηλό |corr|≤~0.2, ομαλό γύρω από k=0, κανένα spike).
- **T3 hour-profile (βήμα 2):** στο ίδιο lagscan output — shortwave peak μεσημέρι
  (~11:00-13:00), temperature απόγευμα. Συνεπές με φυσική, όχι shift.
- **Βήμα 3 (ένταξη ΜΟΝΟ μέσω feature_availability):** αυτό το doc κλειδώνει το rule·
  wiring σε ΞΕΧΩΡΙΣΤΟ βήμα με CORE-DIFF πρωτόκολλο (poison + control + unit tests).
- **Βήμα 4 (πληρότητα ανά task):** ✅ 2026-07-10 — `hourly_load.parquet`: 168 wv
  (42+42+84 flags), 0 NaN/0 missing σε q1/summer/octnov· `hourly.parquet` (price): 0 wv
  στήλες (by design — το price ΔΕΝ αγγίχτηκε)· ζεύγη day1/day2 πλήρη (0 unpaired).
  Backup+compare rebuild: `data/processed/_backup_vintageweather_20260710/` +
  `scripts/compare_vintage_weather_load.py` 4/4 PASS (135 υπάρχουσες στήλες bit-identical).
- **Leakage checks δεδομένων (προ-wiring, G5):** `preflight_check.py --poison` PASS
  (`logs/g5_vintage_preflight_poison.log`) · `check_crosslag_fairness --task load` PASS ·
  control run base Q1 static = 254.9893 ταυτόσημο με pre-merge (#features=17 αμετάβλητο).
- **Verdict σταδίου 2: ΠΡΟΧΩΡΑ** (γράφτηκε στο ABLATION_PLAN §7: ναι — σημείο 14).
