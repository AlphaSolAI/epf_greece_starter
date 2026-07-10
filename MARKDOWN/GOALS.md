# GOALS — ουρά στόχων (ALPHA agent queue)

> **Πώς δουλεύει**: ο agent παίρνει ΠΑΝΤΑ τον πρώτο στόχο με status `QUEUED` του οποίου
> τα depends_on είναι όλα `DONE`. ΜΟΝΟ ο χρήστης προσθέτει/αναδιατάσσει/διαγράφει στόχους.
> Ο agent αλλάζει μόνο `status` και γράφει στο πεδίο «σημειώσεις agent» (+ link σε report).
> Status: `QUEUED → RUNNING → DONE / BLOCKED`. BLOCKED = γράφει γιατί και ΣΤΑΜΑΤΑΕΙ —
> δεν αυτοσχεδιάζει εναλλακτικό πείραμα.
>
> **Κανόνας σύγκρισης**: τρέχοντα ευρήματα/headline (π.χ. 16.96/13.74) = baselines προς
> ξεπέρασμα, ΟΧΙ κλειδωμένες επιλογές. Κάθε νέο αποτέλεσμα περνά από ABLATION_PLAN §2 +
> validity-reviewer πριν γραφτεί ΔΕΚΤΟ. ΠΟΤΕ σύγκριση across data snapshots (βλ. G2 σημείωση).

---

## G1 — Crosslag poisoning check για task=load [status: DONE 2026-07-09 — PASS rec+dir, επιβεβαίωση χρήστη στο terminal (stdout δεν αποθηκεύτηκε σε log)]

- **Τι**: `check_crosslag_fairness` για load — ΔΕΝ έχει τρέξει ΠΟΤΕ (μόνο price). Προϋπόθεση
  για ΚΑΘΕ load εύρημα (last.md §5 blocker #2, ABLATION_PLAN §5.12γ).
- **Εντολές** (σειριακά, ΕΝΑ conda process):
  ```
  conda run -n epf --no-capture-output python -X utf8 -m src.check_crosslag_fairness --task load --algo lgbm --strategy recursive --market dam --gate strict
  conda run -n epf --no-capture-output python -X utf8 -m src.check_crosslag_fairness --task load --algo lgbm --strategy direct --market dam --gate strict
  ```
- **Κριτήριο επιτυχίας**: PASS (leak diff = 0.000000, controls > 0) και στα δύο strategies.
- **Αν FAIL**: status → BLOCKED. Το engine θέλει fix για load ΠΡΙΝ από οτιδήποτε άλλο load —
  κανένα G2/G3 δεν τρέχει. Leakage-sensitive core rule: fix → poisoning ξανά → control run.
- **Σημειώσεις agent**: —

## G2 — Rebuild `hourly_load.parquet` με στήλη `load_fc` [status: DONE 2026-07-09 — compare v2 PASS (0 proxy rows σε 4/4 eval windows), fairness load rec+dir PASS, preflight --poison PASS· logs: g2_rebuild_load / g2_compare_load_v2 / g2_fairness_load_{rec,dir} / g2_preflight_poison _20260709.log· backup: data/processed/_backup_loadfc_20260709/]

- **Τι**: το per-task parquet του load ΔΕΝ έχει καθόλου day-ahead load forecast (αιτία του
  VOID `-loadfc` arm, Δ=0.000 bit-for-bit σε 8/8 — ABLATION_PLAN §5.12γ). Merge με το
  `data/processed/load_forecast_hourly.parquet`.
- **Πώς**: `data/processed/` = ΠΡΟΣΤΑΤΕΥΜΕΝΟ (hook deny) — ΜΟΝΟ μέσω script:
  νέο `scripts/rebuild_hourly_load.py` με (α) backup του παλιού σε
  `data/processed/_backup_loadfc_<ημερομηνία>/`, (β) merge, (γ) σύγκριση παλιού/νέου
  (ίδιες γραμμές/στήλες εκτός της νέας `load_fc`, ίδιο index/TZ frame CET/CEST-naive),
  (δ) αναφορά διαφορών. Το script περνά από `check_run_config.py` + epf-code-reviewer
  (PRE-RUN) πριν εκτελεστεί.
- **Κριτήριο επιτυχίας**: νέο parquet με `load_fc`, μηδενική αλλαγή σε υπάρχουσες στήλες,
  TZ guards PASS στο preflight.
- **⚠️ Σημείωση εγκυρότητας**: μετά το rebuild, ΟΛΑ τα παλιά load runs (Block C) είναι άλλο
  data snapshot — ΚΑΜΙΑ σύγκριση νέων vs παλιών load αριθμών. Το G3 χτίζει δικά του baselines.
- **Σημειώσεις agent (2026-07-09)**:
  - Υλοποίηση: δομικό fix στον builder (`src/data.py` — load_fc block για both tasks +
    άνευ όρων φόρτωση forecast στο main + docstring provenance), ΟΧΙ standalone merge —
    αλλιώς κάθε μελλοντικό `src.data --task load` rebuild θα ξαναχάνε τη στήλη.
  - epf-code-reviewer (CORE-DIFF+PRE-RUN): **APPROVE-WITH-FIXES** — fixes εφαρμόστηκαν:
    (1) window-scoped proxy check στο `scripts/compare_load_parquet.py` (κριτήριο 5 —
    proxy rows load_fc==load_lag24 ΠΡΕΠΕΙ 0 σε q1/summer/march/octnov, ΥΠΟΧΡΕΩΤΙΚΟ πριν G3)·
    (2) ατομικό backup→rebuild block· (3) snapshot attribution: base `c1d95c0`, dirty src:
    data.py (αυτό το fix) + feature_availability.py/master_forecast.py (ramp workstream,
    ήδη poison-tested) — το G2 validation evidence καλύπτει το συνδυασμένο tree.
  - Σημείωση reviewer: PASS στο `check_crosslag_fairness --task load` ΔΕΝ πιστοποιεί το
    load_fc (στατική parquet στήλη, εκτός AEL crosslag family) — το πιστοποιεί το κριτήριο 5.
  - Εκκρεμότητα core-change κανόνα: πρώτο post-change **price** run να πιάσει το anchor
    Q1 17.035±0.05 (LGBM default,dense weekly rec seed 42) → μπαίνει ως Block 0 στο G3 batch.
  - QA linter: 0 findings (compare script, 2 περάσματα).

## G3 — Πλήρες load re-ablation σε καθαρή βάση [status: ΑΝΤΙΚΑΤΑΣΤΑΘΗΚΕ από Phase 1 redesign 2026-07-10]

**2026-07-10 πρωί — v3 killed από χρήστη (~08:59, 9 xgb-rec OK πριν το kill) + πλήρες redesign
με οδηγίες χρήστη:** (1) explicit specs παντού, ΟΧΙ «default» — παρακάμπτονται loadlags duplicates
+ load_fc-in-default ΧΩΡΙΣ core change· (2) baseline = `calendar,lags,roll` ΧΩΡΙΣ meteo·
(3) forecasts (loadfc/resfc) = ξεχωριστή Phase 2 (το gen_fc κρύβει μέσα του το load forecast του TSO)·
(4) +pricelags arm (τιμή→φορτίο) εγκρίθηκε εννοιολογικά, θέλει ΠΡΩΤΑ ingest τιμής στο
hourly_load.parquet (καμία στήλη τιμής σήμερα) — επόμενο batch/snapshot.
**Phase 1 RUNNING 10:06** (goal χρήστη): lgbm recursive, specs base/+meteo/+dense/+genlags ×
Q1+summer = 8 runs → `runs/load_phase1/`, log `logs/load_phase1_lgbm_rec.log`,
script `scripts/load_phase1_lgbm_rec.sh` (linter 0 findings, inline PRE-RUN APPROVE — subagents
πλέον ΜΟΝΟ κατόπιν εντολής χρήστη). ΚΑΜΙΑ σύγκριση v2/v3 ↔ Phase 1 (άλλο feature contract).
⚠️ **(2026-07-10 μεσημέρι) Το +meteo arm του Phase 1 χρησιμοποιεί oracle weather** (observed,
Archive API) — τα meteo νούμερα δηλώνονται oracle, ΚΑΜΙΑ νίκη-ΑΔΜΗΕ ΔΕΚΤΗ πριν το G5 vintage.
Βλ. last.md §2 Α6. Το νέο ολοκληρωμένο load ablation χτίζεται πάνω σε αυτόν τον κανόνα.

**Ιστορικό εκτέλεσης:**
- v2 launch 04:53 (μετά από 1ο abort λόγω OneDrive down — σωστό gate). Block 0 4/4 PASS,
  anchor έτρεξε, 27 έγκυρα JSONs (1 anchor + 14 lgbm_rec + 12 lgbm_dir).
- **Kill v2 ~08:55 (απόφαση χρήστη)** — αιτία: (1) λάθος εκτίμηση διάρκειας (direct×static
  = 90 fits/run → 15-35'/run, όχι 2-3'), (2) redesign από 2 ευρήματα του χρήστη:
  **(α) loadlags = byte-identical διπλότυπα των y_lag* για task=load** (data.py:712-714) —
  το -loadlags arm μετράει θόρυβο, void-by-construction·
  **(β) load_fc (ΑΔΜΗΕ forecast) ΕΚΤΟΣ default για task=load** — δεν γίνεται το STLF headline
  να στηρίζεται στο forecast του TSO· νέο baseline verdicts: `default,-loadfc`, το «default»
  γίνεται +loadfc arm, και το load_fc μπαίνει ως ΞΕΧΩΡΙΣΤΟ baseline στους πίνακες (MAE ΑΔΜΗΕ).
- v3 continuation: `scripts/g3_load_ablation_v3.sh` — 24 runs (xgb_rec 7×2 → lgbm_dir dense ×2
  → xgb_dir trimmed 4 specs ×2), idempotent (SKIP σε υπάρχον JSON), ίδιο snapshot.
- **Follow-up core fix ΜΕΤΑ το harvest** (όχι μέσα στο batch): για task=load μη δημιουργία
  διπλότυπων load_lag* + per-task DEFAULT_GROUPS χωρίς loadfc — με πλήρες CORE-DIFF πρωτόκολλο.

**Harvest preconditions (αμετάβλητα):** anchor 17.035±0.05 · Block 0 PASS lines στο v1 log ·
synthesize-ablation με baseline `default,-loadfc` + validity-reviewer πριν από κάθε ΔΕΚΤΟ ·
-loadlags/-genlags,-loadlags arms = void/επανερμηνεία (βλ. πάνω) · ΚΑΜΙΑ σύγκριση με παλιό Block C.

- **Τι**: το νέο Block C — πρώτο έγκυρο feature ablation για task=load (μέχρι σήμερα μόνο το
  E2/meteo υπάρχει ως ΔΕΚΤΟ). Grid: 7 specs (default, -loadfc, -genlags/-loadlags,
  -loadlags, -meteo, +dense, lean) × LGBM+XGB × recursive+direct × Q1+καλοκαίρι,
  static retrain (απομόνωση feature αξίας), seed 42, `--gate strict`, πάντα `--out_json`.
- **Πώς**: preflight (`preflight_check.py --poison`) πρώτα → batch script σε
  `scripts/` (πέρασμα από PRE-RUN review) → detached εκτέλεση (Start-Process pattern) →
  outputs `runs/load_ablation_v2/` → `synthesize-ablation` (ΔMAE πίνακες + §2 pre-gate)
  → validity-reviewer πριν γραφτεί ΟΤΙΔΗΠΟΤΕ ως ΔΕΚΤΟ στο ABLATION_PLAN.
- **Κριτήριο επιτυχίας**: 0 FAILED runs, verdicts ανά ομάδα (ΔΕΚΤΟ/PENDING/MIXED) με trace
  (JSON paths + εντολές αναπαραγωγής), ενημέρωση ABLATION_PLAN + last.md.
- **Σημειώσεις agent**: —

## G4 — SS×weekly + SS σε XGB [status: PROPOSED από agent, ΟΧΙ queued — χρειάζεται OK χρήστη]

- **Τι**: challenger του headline baseline (16.96/13.74) — SS ΔΕΝ έχει δοκιμαστεί ΠΟΤΕ με
  weekly retrain (last.md §5 blocker #3, ABLATION §7 σημείο 6, §5.12ε). Κώδικας: SS
  καλωδιωμένο ΜΟΝΟ για recursive (όχι direct — code constraint, όχι επιλογή πειράματος).
  Στο price τα direct/recursive ablations έγιναν ιστορικά ΞΕΧΩΡΙΣΤΑ scripts (όχι combined
  loop όπως το load G3) — ακολούθα το ίδιο pattern εδώ.
- **Grid**: LGBM (ήδη static-only δοκιμασμένο, §5.12ε) + XGB, `--ss --ss_decay linear`,
  weekly retrain, recursive only, specs `default` + `default,dense`, Q1+summer.
- **Πριν launch**: PRE-RUN review (epf-code-reviewer) + linter, ίδιο πρωτόκολλο με G2/G3.
- **Μετά**: synthesize-ablation + validity-reviewer πριν γραφτεί οτιδήποτε ΔΕΚΤΟ.
- **Σημειώσεις agent**: —

## G5 — Vintage weather (D-1 forecast) → έντιμος διαγωνισμός load vs ΑΔΜΗΕ [status: DONE 2026-07-10 απόγευμα — wiring `meteo_vintage` ΟΛΟΚΛΗΡΩΘΗΚΕ (audit→TDD→poison→control→smoke PASS)· batch = G7 Batch 3]

**Απόφαση χρήστη (2026-07-10):** από εδώ και πέρα το LOAD είναι **γνήσιος day-ahead
διαγωνισμός με τον ΑΔΜΗΕ**, με **forecast καιρού — ΟΧΙ τέλεια/oracle πρόγνωση**.

- **Πρόβλημα (βλ. last.md §2 Α6)**: τα τρέχοντα `w_*` = observed (Open-Meteo Archive) = oracle
  covariate (μελλοντική πληροφορία στο gate). Κάθε meteo-based «νίκη ΑΔΜΗΕ» (π.χ. Q1 meteo
  ~160) είναι **χάρτινη** μέχρι να μπει vintage. Δεν το πιάνει το poisoning (exogenous).
- **Fix**: ingest Open-Meteo **Historical Forecast API** (αρχειοθετημένες προγνώσεις ~2022→),
  τραβώντας το **D-1 vintage forecast** για τα windows μας → ξανα-τρέξιμο των meteo arms με
  πραγματική πρόγνωση. Νέα πηγή → skill `feature-eng`/`ingest-audit` (gate timing · lagscan ·
  hour-profile), staging-merge, backup+σύγκριση parquet (data/processed/ προστατευμένο).
- **Ορίζοντας (επιβεβαιωμένο, Καν. 543/2013 άρ. 6(2))**: gate D-1 12:00, τελευταίο actual
  ~D-1 11:00 → στόχος D 00:00 = **13-14h** μπροστά, D 23:00 = **36-37h**. Γνήσιο multi-step
  day-ahead — ΟΧΙ 1-step. Ο ΑΔΜΗΕ εκδίδει ~10:00 CET D-1 → ορίζοντας 14-38h.
- **ΑΔΜΗΕ benchmark**: MAE 171-176 MW (δικά μας 2025-26 windows) / 133 MW (Shiblee-Koukaras
  2024). Caveat προς δήλωση: μέθοδος ΑΔΜΗΕ άγνωστη + η δημοσιευμένη σειρά load_fc μπορεί να
  κουβαλά post-gate ≥10% αναθεωρήσεις (κρατείται η τελευταία εκδοχή). Koukaras 69 MW =
  1h-ahead nowcasting (lag_1/roll3) — ΜΗ συγκρίσιμο (§ ABLATION load).
- **Κριτήριο επιτυχίας**: meteo arms με vintage weather → σύγκριση με ΑΔΜΗΕ baseline στο ίδιο
  window, validity-reviewer πριν γραφτεί ΔΕΚΤΟ. Το no-meteo baseline παραμένει το καθαρό anchor.
- **Σημειώσεις agent (2026-07-10)**:
  - **ΔΙΟΡΘΩΣΗ σχεδίου**: το «Historical Forecast API» ΔΕΝ δίνει vintage/lead-time δεδομένα
    (stitches τις πρώτες ώρες κάθε run — ουσιαστικά near-real-time, όχι D-1 forecast).
    Το σωστό API είναι το **Previous Runs API** (`previous-runs-api.open-meteo.com`) με
    σταθερά lead-time buckets: `<var>_previous_day1` = τιμή που προβλέφθηκε 24h πριν το
    valid time, `_previous_day2` = 48h πριν. ΔΕΝ υποστηρίζει επιλογή ώρας issue (μόνο
    24ωρα buckets) — άρα ανά ώρα-στόχο D επιλέγεται το ΜΙΚΡΟΤΕΡΟ bucket που εγγυάται
    issue-time ≤ gate cutoff (π.χ. gate=12h: h≤11 →day1, h≥12 →day2· gate=14h: h≤9 →day1,
    h≥10 →day2) — ΑΥΤΗ η ανά-ώρα επιλογή είναι μελλοντικό βήμα (feature_availability.py),
    ΔΕΝ έχει μπει ακόμα.
  - **Coverage** (επιβεβαιωμένο εμπειρικά): πλήρες 7/7 μεταβλητών μόνο από **~2024-02-01**
    (θερμοκρασία μόνο πίσω ως 2021, οι άλλες 6 μηδέν πριν 2024). Καλύπτει άνετα τα
    τρέχοντα windows (Q1/summer/octnov 2025-26).
  - **TZ quirk** (κρίσιμο, θα προκαλούσε σιωπηλό 1-2h misalignment): το API έχει το ΙΔΙΟ
    fixed-UTC+1 quirk με το Archive API — επιβεβαιώθηκε με cross-correlation έναντι της
    ήδη έμπιστης `w_gr_mean_*` στήλης (DJF corr=0.969, JJA corr=0.988, ΑΚΡΙΒΩΣ στο lag
    k=0 με το ίδιο fix `_fixed_utc1_to_cet_naive_index`). Ο fetcher αποθηκεύει RAW
    timestamps (όχι μετατροπή στο fetch — κανόνας "TZ conversions ONLY στους loaders").
  - **Υλοποίηση**: `src/fetch_open_meteo_vintage.py` (νέο, chunked+retry) →
    `data/processed/weather_vintage_hourly.parquet` (5 πόλεις × 7 vars × {day1,day2} +
    gr_mean, 2024-02-01→2026-07-08, 0 NaN). Δομικό join στο `src/data.py`
    (`load_weather_vintage_hourly()` + join **ΜΟΝΟ για task=load**, καθόλου αλλαγή στο
    price pipeline) — νέες στήλες `wv_*` πλήρως αδρανείς (καμία αναφορά στο
    `feature_availability.py` ακόμα, μηδενικός κίνδυνος leak σε υπάρχον run).
  - **Backup+compare**: `data/processed/_backup_vintageweather_20260710/` +
    `scripts/compare_vintage_weather_load.py` — 4/4 κριτήρια PASS (ίδιο index, 168 νέες
    στήλες + 0 unexpected, 135 υπάρχουσες στήλες bit-identical, 0 NaN στα 3 windows).
  - **Leakage checks**: `preflight_check.py --poison` PASS (exit 0)· explicit
    `check_crosslag_fairness --task load` PASS· **reproducibility control run**
    (`calendar,lags,roll` Q1 lgbm rec static) → MAE=254.9893 ΤΑΥΤΟΣΗΜΟ με το pre-change
    Phase 1 νούμερο (#features=17 αμετάβλητο) — μηδενική επίδραση σε υπάρχοντα configs.
    `lagscan.py` σε temperature/shortwave: χαμηλό |corr|≤0.18, ομαλό/συμμετρικό γύρω
    στο k=0 (αναμενόμενο για αργά-μεταβαλλόμενο εξωγενές covariate, ΟΧΙ leak-signature
    όπως το xborder 0.91-spike)· hour-profile peak 11:00/13:00 (φυσιολογικό). Επιπλέον:
    MAE(wv_day1, actual)=0.86°C / 13.1 W/m² με corr 0.986-0.993 — πραγματικό forecast
    error, ΟΧΙ σιωπηλό oracle-relabel.
  - ~~Εκκρεμεί: wiring `meteo_vintage`~~ ✅ **ΕΓΙΝΕ 2026-07-10 απόγευμα** (ingest-audit
    formal record: `docs/features/meteo_vintage/design.md` + ABLATION_PLAN §7.14):
    (α) ομάδα `meteo_vintage` ΕΚΤΟΣ default· features = ΜΟΝΟ gate-aware blend
    `wveff_* = day1 αν h ≤ 23−gap αλλιώς day2` (`add_meteo_vintage_features`, κληρονομεί
    `--delay`)· (β) **δομικός αποκλεισμός ωμών `wv_*` από το classify_columns** — ΝΕΟ
    μάθημα: unclassified στήλες πέφτουν στο `other` ⊂ default, άρα τα ωμά day1 ήταν
    σιωπηλά επιλέξιμα σε μελλοντικό `default` run (το G7 γλίτωσε επειδή τρέχει explicit
    specs)· (γ) forward market → ρητό ValueError (χρειάζεται day3+ buckets)· (δ) TDD:
    +8 unit tests (boundaries g12 h=11/12, g14 h=9/10) → pytest 94 PASS· (ε) poison PASS
    (`logs/mv_wiring_preflight_poison.log`)· (στ) control base Q1 static = 254.9893
    ΤΑΥΤΟΣΗΜΟ, #features=17 αμετάβλητο· (ζ) smoke static Q1 mv: **#features 17→101,
    MAE=221.28** (Δ=−33.7 vs base, ~36% του oracle effect −94.9 — ΚΑΤΩ από το
    pre-registered 50-100%, honest miss· T8 OK: χειρότερο από oracle 160.11 ✓).
    Runs: `runs/feat_meteo_vintage/`. Weekly batch → G7 Batch 3 (launched).

## G6 — Price ingest στο hourly_load.parquet (ενεργοποίηση pricelags) [status: QUEUED — goal χρήστη 2026-07-10]

- **Τι**: το `hourly_load.parquet` ΔΕΝ έχει καμία στήλη τιμής σήμερα → το `+pricelags` arm
  (τιμή→φορτίο) είναι αδύνατο. Ingest price lags ως ομάδα στο per-task load parquet.
- **Availability rule (κρίσιμο)**: η τιμή DAM της D-1 βγαίνει στο auction της D-2 → στο gate
  D-1 12:00 είναι γνωστή ΟΛΗ η D-1 (μέχρι 23:00). Άρα `price_lag1..N` ΝΟΜΙΜΑ διαθέσιμα (gap=0,
  όπως στο price task). Ρητό availability rule στο `src/feature_availability.py`, νέα ομάδα.
- **Πώς**: 3-βήμα pre-flight (gate timing · lagscan · hour-profile) → script rebuild με
  backup+σύγκριση (data/processed/ προστατευμένο) → poison/fairness → epf-code-reviewer CORE-DIFF.
- **Κριτήριο επιτυχίας**: νέα στήλη(ες) price στο load parquet, TZ/poison PASS, μηδενική αλλαγή
  υπαρχουσών στηλών. Ξεκλειδώνει το `+pricelags` arm του G7.
- **Σημειώσεις agent**: —

## G7 — ΝΕΟ ΟΛΟΚΛΗΡΩΜΕΝΟ load ablation = διαγωνισμός vs ΑΔΜΗΕ [status: RUNNING-PLAN 2026-07-10 — goal χρήστη]

**Στόχος χρήστη**: το LOAD = γνήσιος day-ahead διαγωνισμός με ΑΔΜΗΕ, weekly retrain, με σωστό
καιρό (μετά G5). Σταθερά: `task=load · dam · gate=strict · --out_json` πάντα · `runs/load_contest/`.

- **Retrain**: **weekly ΜΟΝΟ** (απόφαση χρήστη — όχι static). Feature-isolation φέρει retrain
  variance → ισχύει |ΔMAE|>0.15 + ίδιο πρόσημο σε ≥2 ανεξάρτητα windows.
- **Windows**: Q1 (2025-12-01→2026-02-28) + Καλοκαίρι + 3ο (Μάρτιος/Οκτ-Νοε) για independence.
- **Σειρά εκτέλεσης (απόφαση χρήστη)**: **LGBM → XGB → SS(recursive) → direct**.
- **Benchmark στήλες πινάκων**: `loadfc`-alone = MAE ΑΔΜΗΕ (171-176) · naive y_lag168/24 = floor ·
  δικό μας καθαρό baseline `calendar,lags,roll`.
- **GATE dimension (100% align vs ΑΔΜΗΕ — goal χρήστη 2026-07-10)**: κάθε arm τρέχει σε **2 gates**:
  **g12** = δικό μας deployable (gap=12, cutoff 11:00 CET, απόφαση ~12:00 CET) · **g14** =
  ΑΔΜΗΕ-aligned (`--delay 14`, gap=14, cutoff **09:00 CET** = έκδοση ~10:00 CET, ορίζοντας 14-38h).
  Το g14 αφαιρεί το 2h πλεονέκτημά μας → ο head-to-head vs ΑΔΜΗΕ γίνεται στο g14. Απλό CLI flag
  (`master_forecast.py:526/558` → `GateSpec.delay_override`)· το AEL crosslag gap κληρονομεί το ίδιο
  override (`feature_availability.py:273`) → πλήρως συνεπές, ΟΧΙ core change, ΟΧΙ leak.
- **Arms — καθαρά (τρέχουν ΤΩΡΑ, δεν εξαρτώνται από G5/G6)**:
  (1) `calendar,lags,roll` baseline · (2) `+dense` · (3) `+genlags` (AEL-frozen) ·
  (4) **`calendar,lags` (−roll)** ← ρητό «αφαίρεση y-roll» (goal χρήστη) · (5) `+loadfc`
  («με βοήθεια αντιπάλου», ξεχωριστό). **Χωρίς lean/core arm** (ο χρήστης το απέρριψε 2026-07-10).
- **Arms — gated**: `+meteo` **μόνο μετά G5 vintage** (σήμερα oracle, §2 Α6) · `+pricelags`
  **μόνο μετά G6**.
- **Batch 1 (ΤΩΡΑ)**: `scripts/load_contest_lgbm_weekly.sh` — LGBM weekly recursive × 5 arms ×
  3 windows × 2 gates = 30 runs, idempotent → `runs/load_contest/`. Πρώτη εικόνα.
- **Ροή ανά batch**: preflight --poison → PRE-RUN review (epf-code-reviewer) + linter → detached
  (Start-Process, ASCII args) → synthesize-ablation (§2 pre-gate) → validity-reviewer πριν ΔΕΚΤΟ.
- **Σημειώσεις agent (2026-07-10)**:
  - **Batch 1 (lgbm weekly rec) DONE — 30/30, 0 FAILED**, harvested. Νέα εργαλεία (system/py311,
    ΧΩΡΙΣ conda): `scripts/load_contest_report.py --algo <a> --mode recw` (ΔMAE ανά gate/window
    έναντι `base`) + `scripts/load_contest_benchmarks.py` (ΑΔΜΗΕ+naive MAE ανά window,
    read-only στο parquet). ΑΔΜΗΕ benchmark: q1=175.56 · summer=170.76 · octnov=146.81 MW
    (naive_lag168 πολύ χειρότερο, 238-604 — ΑΔΜΗΕ σαφώς όχι naive).
  - **Candidate (PENDING — 1 algo μόνο, θέλει XGB confirm)**: `dense` ΒΟΗΘΑΕΙ σταθερά — Δ<0 σε
    6/6 συνθήκες (3 windows × 2 gates), π.χ. g12 ΔMAE q1=-8.5/summer=-15.4/octnov=-10.7.
    `genlags` MIXED (βοηθάει q1/octnov, βλάπτει έντονα summer +17.8/+8.6). `noroll`
    (calendar,lags −roll) MIXED αλλά ασύμμετρο: βοηθάει πολύ summer (-31/-29), βλάπτει λίγο
    q1/octnov (+1 έως +3.3). `loadfc` arm κλείνει σημαντικά το χάσμα με ΑΔΜΗΕ αλλά ΔΕΝ τον
    ξεπερνά σε q1/summer (χειρότερο +5 έως +52 MW)· ΣΤΟ octnov όμως το ξεπερνά (139.7-141.2
    έναντι 146.8) — ΜΟΝΟ arm/window που πλησιάζει/ξεπερνά ΑΔΜΗΕ.
  - **Καθαρό εύρημα (χωρίς ΑΔΜΗΕ-βοήθεια)**: `dense` (χωρίς loadfc, χωρίς meteo) ΞΕΠΕΡΝΑΕΙ ΑΔΜΗΕ
    στο octnov και στα 2 gates (g12 142.97 vs 146.81 · g14 145.77 vs 146.81) — 1 window μόνο,
    ΔΕΝ πιάνει §2 independence ακόμα (χρειάζεται 2ο window). Q1/summer παραμένουν πολύ πίσω
    από ΑΔΜΗΕ με καθαρά features (247-386 vs 171-176).
  - **§2 threshold σημείωση**: το |ΔMAE|>0.15 είναι βαθμονομημένο για price (€/MWh, ~13-20 scale)·
    σε load (MW, ~150-400 scale) περνάει σχεδόν τετριμμένα — το ουσιαστικό τεστ εδώ είναι
    ίδιο πρόσημο σε ≥2 ανεξάρτητα windows, όχι το raw 0.15. Καταγράφεται εδώ, ΔΕΝ άλλαξε ο κανόνας.
  - **Batch 2 (xgb weekly rec) ΕΤΟΙΜΟ**: `scripts/load_contest_xgb_weekly.sh` (ίδιο grid,
    --algo xgb) — linter 0 findings, inline PRE-RUN APPROVE (mirrors το ήδη εγκεκριμένο Batch 1
    script 1:1, μόνο --algo swap + git-diff capture προστέθηκε για traceability parity).
    ΔΕΝ launched ακόμα — G8 (netload weekly, προτεραιότητα χρήστη) κατέχει το ΕΝΑ conda slot·
    launch αυτόματα μόλις τελειώσει (background wait armed).
  - **Batch 3 (vintage lgbm weekly) DONE 2026-07-10 23:02 — 12/12, 0 FAILED, harvested**:
    meteo_vintage T7 pre-gate PASS 6/6 (βλ. ABLATION §7.14 + πλήρης πίνακας στο
    `reports/feature_lifecycle_meteo_vintage_20260710.md`). **Πρώτη έντιμη νίκη vs ΑΔΜΗΕ:
    octnov 4/4 κελιά** (mv/densemv × g12/g14, 128.5-133.1 vs 146.81) — window-specific,
    1 algo/1 seed· q1/summer ΑΔΜΗΕ προηγείται. Κρίση αποδοχής με τον χρήστη.
  - **Batch 2 (xgb weekly rec) DONE 2026-07-11 πρωί — 30/30, 0 FAILED, harvested**
    (`runs/load_contest/*_xgb_recw_*.json`): **dense ΕΠΙΒΕΒΑΙΩΘΗΚΕ σε XGB — Δ<0 σε 6/6**
    (g12 q1 −10.4/summer −1.7/octnov −8.6 · g14 −5.6/−12.7/−7.2) → συνολικά **12/12 σε
    2 αλγορίθμους**, πιάνει §2 άνετα — κρίση αποδοχής με τον χρήστη. Επιπλέον: XGB dense
    κερδίζει ΑΔΜΗΕ στο octnov ΧΩΡΙΣ καιρό (137.46 g12 / 143.81 g14 vs 146.81) — το
    octnov claim πλέον 2 algos × 2 gates. genlags MIXED και σε XGB (συνεπές)· noroll
    ίδιο ασύμμετρο pattern (βοηθά μόνο summer)· loadfc κλείνει το χάσμα (178.7-221.4).
    XGB base καλύτερο από LGBM στο summer (337.8 vs 358.7). Deploy checklist meteo_vintage:
    `docs/features/meteo_vintage/deploy.md` (validator PASS)· FI evidence:
    `reports/fi_meteo_vintage_q1.txt`. Επόμενα G7 stages κατά τη σειρά χρήστη: SS → direct.
  - **Overnight 2026-07-11 (goal χρήστη «run overnight, ΟΧΙ 3ωρα runs»)**: detached chain
    (pid 22324, `scripts/_tmp_launch_overnight_20260711.ps1`), μικρά idempotent κομμάτια:
    (1) `load_contest_octnov_seeds.sh` — octnov seeds 7+123 × {dense,mv,densemv} × 2 gates
    = 12 runs (~3-5'/run) → `runs/load_contest_seeds/`, κλείδωμα του octnov-vs-ΑΔΜΗΕ σε 3
    seeds· (2) `load_contest_ss_lgbm_weekly.sh` — **SS×weekly probe** (η επόμενη στάση):
    LGBM rec weekly `--ss --ss_decay linear`, specs {base,dense} × windows {octnov,summer}
    × 2 gates = 8 runs → `runs/load_contest_ss/`. q1-dense + mv/densemv SS ΕΞΩ σκόπιμα
    (ss_rounds=3 → πολύωρα runs)· **direct ΕΞΩ** (direct×weekly load αδοκίμαστο/αργό —
    ξεχωριστό κομμάτι με OK χρήστη). Linter 0 findings + inline PRE-RUN APPROVE (mirrors
    Batch 1). Marker: `logs/overnight_20260711.done`. Harvest+κρίση με χρήστη το πρωί.
  - **(2026-07-10 απόγευμα) Το armed wait του προηγούμενου session ΧΑΘΗΚΕ** (session end) —
    το G8 τελείωσε (results/netload_weekly_lgbm.csv πλήρες 6/6) χωρίς να αυτο-εκκινήσει
    το Batch 2. **Νέο launch (pid 6716, detached chain, `scripts/_tmp_launch_vintage_chain.ps1`):
    Batch 3 vintage (12 runs, `scripts/load_contest_vintage_lgbm.sh`, linter 0 findings,
    inline PRE-RUN APPROVE — mirror του Batch 1) → μετά Batch 2 XGB (30 runs)**, σειριακά,
    logs: `logs/load_contest_vintage_lgbm.log` / `logs/load_contest_xgb_weekly.log`,
    marker: `logs/vintage_chain.done`. Harvest και των δύο μετά το marker.

## G8 — Netload ramp weekly confirm, LGBM, 2025 windows [status: RUNNING 2026-07-10 — goal χρήστη «να μπει σε προτεραιότητα», launched pid 23036]

- **Τι**: το static screen (results/netload_seeds.csv) έδειξε το `netload_ramp1h`
  (= diff(load_fc − solar_fc − wind_fc), day-ahead-known) ως τον καλύτερο ramp variant —
  βοηθάει σε 3/4 windows, countersignal μόνο w2025-XGB. Πριν από κάθε σκέψη για headline
  spec: weekly confirm (το headline είναι weekly). ΔΕΝ είναι promotion — screen ακόμα.
- **Grid (απόφαση χρήστη — «πιο πρόσφατα windows, LGBM, να κρίνουμε μαζί»)**: LGBM recursive
  weekly × (s2025 Ιουν-Αυγ, w2025 Ιαν-Φεβ) × seeds 42/7/123 × (default,dense vs +netload_ramp1h)
  = 12 runs. Script: `scripts/netload_weekly_lgbm_seeds.py` (linter: 1 αποδεκτό WARNING
  no_out_json — API screen, trace=CSV+log όπως το static screen· inline PRE-RUN APPROVE).
- **Anchor ελέγχου**: s2025/seed42 baseline αναμένεται ≈13.81±0.05 (Block A weekly
  default,dense summer) — dirty-src control. Απόκλιση = όλο το batch ύποπτο.
- **Outputs**: `results/netload_weekly_lgbm.csv` · `logs/netload_weekly.{out,err}.log`.
- **Μετά**: πίνακας Δ ανά window×seed → κρίση ΜΑΖΙ με τον χρήστη (ρητή οδηγία) — ΚΑΜΙΑ
  εγγραφή ΔΕΚΤΟ χωρίς αυτόν. Preflight PASS 2026-07-10 πριν το launch.
- **Σημειώσεις agent**: —

---

## Επόμενα υποψήφια (ΔΕΝ είναι στην ουρά — τα προάγει ο χρήστης όταν θέλει)

- Κλείσιμο renewable_ramp (ύποπτο −2.83, ανεξήγητος μηχανισμός — SHAP / ρητό «μήκος
  μηδενικής περιόδου» feature ή αναστολή)
- Conformal Β4: ≥2 μοντέλα × Q1+Μάρτιος — ΜΟΝΟ αφού σταθεροποιηθεί το point config
- Β6 προϊόν: daily runner → settle loop → delivery
