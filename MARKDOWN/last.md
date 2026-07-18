# SESSION HANDOFF — energy trading agent (GR EPF/STLF)

> **Ενημερώθηκε 2026-07-11** (Β3 leak-free re-ablation σε εξέλιξη· 2026-07-10: +§2 Α6
> oracle-weather κανόνας· **2026-07-11: G5 vintage DONE end-to-end + G7 load contest
> Batches 1-3 DONE — πρώτη έντιμη νίκη vs ΑΔΜΗΕ στο octnov, βλ. §1 LOAD**). ΜΟΝΟ τρέχουσα αλήθεια.
> Πλήρες ιστορικό: `OLD/docs/` (last_history_20260704, ABLATION_PLAN_full_20260704,
> VALIDITY_CHECKLIST_20260705 — το validity checklist ΣΥΓΧΩΝΕΥΤΗΚΕ εδώ, §2).
> Διαβάζεται ΜΑΖΙ με: `ABLATION_PLAN.md` (ευρήματα/πειράματα — το 2ο master αρχείο) ·
> skills: `.claude/skills/energy-forecast/SKILL.md` (κανόνες/εντολές) + `energy-runner`
> (operating loop) · design specs: `MASTER_PIPELINE_DESIGN.md`, `SYSTEM_DESIGN_TRADING_AGENT.md`.

## 1. Πού είμαστε (μία ματιά)

Θεμέλια ΟΛΑ σταθερά πλέον: **TZFIX ✅** (4 ρολόγια ευθυγραμμισμένα, guard στο preflight) →
**AEL ✅** (crosslag freeze-at-cutoff σε recursive/direct/training/conformal, poisoning PASS,
`SYSTEM_DESIGN §4.8`, `ABLATION §5.10`) → **Β1 ✅** (πρώτα leak-free νούμερα) →
**Β3 ✅ ΟΛΟΚΛΗΡΩΘΗΚΕ** → **OVERNIGHT batch (2026-07-05) ✅ 122/122 runs** → **follow-up batch
(2026-07-06) ✅ 12/12 runs, 0 FAILED** → 🏆 **ΝΕΟ HEADLINE ΚΛΕΙΔΩΣΕ (2026-07-06,
validity-reviewer ACCEPT)**:

### 🏆 LGBM `default,dense`, weekly retrain, recursive, DAM/price
**Q1 ≈ 16.96 €/MWh (std 0.060, 3 seeds) · Summer ≈ 13.74 €/MWh (std 0.054, 3 seeds)**
Σφιχτότερο από το μεθοδολογικό πρότυπο P6 (std≈0.11). Leak-free, cadence-σωστό, poisoning
PASS φρέσκο. Πλήρης τεκμηρίωση: `ABLATION_PLAN §5.12δ`. Παλιό 15.17 ΣΕ ΑΝΑΣΤΟΛΗ (leaked).

Υπόλοιπα ευρήματα:

- **Β3 recursive+direct** (LGBM+XGB × Q1+summer × 7 specs — `ABLATION §5.11`): meteo ΒΟΗΘΑΕΙ
  (ACCEPTED 8/8, rec+dir)· dense ΒΟΗΘΑΕΙ recursive (ACCEPTED 4/4)· resfc εποχιακό flip·
  genlags οριακό/mixed· lean core = winter-only, χάνει στο direct.
- **Block A ✅ 24/24 — ΔΕΚΤΟ**: weekly retrain > monthly (12/12)· `default,dense` καλύτερο
  spec (4/4) → **candidate** headline Q1=17.035/Summer=13.812 LGBM-weekly (ΔΕΝ κλειδωμένο, §5.12δ).
- **Block B ✅ 28/28 — ΔΕΚΤΟ**: dense ACCEPTED 4/4 Μάρτιος (2ο window)· 2 ανοιχτές συγκρούσεις
  (meteo, lean-core-direct) ΠΑΡΑΜΕΝΟΥΝ PENDING — χρειάζονται 3ο ανεξάρτητο window.
- **Block C (LOAD) ✅ 56/56 πλήρες αλλά 🔴 BLOCKED**: `-loadfc` VOID επιβεβαιωμένο (8/8 Δ=0.000,
  γνωστός infra bug). Όλα τα άλλα arms (genlags/loadlags, meteo, dense) αλλάζουν πρόσημο σε
  ΚΑΘΕ από τα 8 κελιά — καμία ομάδα δεν πιάνει §2. Effect sizes έως 30% baseline απαιτούν
  `check_crosslag_fairness.py --task load` (ΠΟΤΕ δεν έτρεξε — μόνο --task price έχει
  δοκιμαστεί) πριν γραφτεί οτιδήποτε ως εύρημα.
- **Block D (seeds) 🟢 ΔΙΟΡΘΩΘΗΚΕ + ΔΕΚΤΟ (follow-up Block G, 2026-07-06)**: σωστό re-run
  με `--retrain weekly`, seeds 7+123 → Q1={17.035,16.940,16.891} std=0.060, Summer=
  {13.812,13.691,13.708} std=0.054. Headline κλειδωμένο, βλ. §1 πάνω.
- **Block E (SS) 🟡 PENDING**: `default` (χωρίς dense) — SS ΒΛΑΠΤΕΙ, 2/2 windows ίδιο πρόσημο
  |Δ|>0.15 (πιάνει §2, αλλά μόνο 1 algorithm)· `default,dense` — μέσα στο noise floor, καμία
  ετυμηγορία. Το προ-AEL «SS ΔΕΚΤΟ −0.266» (παλιό §5.4) είναι πλέον ΣΕ ΑΝΑΣΤΟΛΗ.
- **Block F (conformal) 🔵 δεδομένα υπάρχουν, ΟΧΙ αρκετά για headline claim**: quantile-LGBM
  Q1 avg_pinball=6.075/MAE(p50)=16.4454, αλλά **coverage 43.19% έναντι 80% nominal**
  (σοβαρό miscalibration, πρέπει να αναφέρεται πάντα). split-conformal coverage=72.38%
  (καλύτερο). Μόνο 1 window (Μάρτιος n=0), 1 μοντέλο — χρειάζεται 2ο window + 2ο μοντέλο.
  Σημ.: ο summarizer έγραφε λάθος "NO-MAE" (schema mismatch, ΟΧΙ αποτυχία run) — αριθμοί
  υπάρχουν, βλ. `ABLATION_PLAN §5.12στ`.
- Παλιό headline **15.17 ΣΕ ΑΝΑΣΤΟΛΗ** (leaked). Νέο headline candidate: `default,dense`
  weekly retrain recursive Q1=17.035/Summer=13.812 — **ΔΕΝ κλειδώνει** πριν: (1) σωστό
  seed re-run (weekly, όχι static), (2) LOAD poisoning check, (3) SS 2ο algorithm/window.
- **LOAD contest (G7) — κατάσταση 2026-07-18 (πλήρης χάρτης: ABLATION §9, entries §7.19-23)**:
  Batches 1-3 (72 runs) + MLP/LEAR/LSTM 3 windows × 2 gates + direct LGBM 3 windows +
  seeds ΠΛΗΡΗ (LGBM {dense,mv,densemv} + XGB dense × 3 windows × 2 gates × 3 seeds) —
  όλα 0 FAILED. **`meteo_vintage` (G5) live end-to-end**. Ευρήματα (PENDING —
  validity-reviewer σε εξέλιξη, αποδοχή=χρήστης):
  (α) **octnov: κερδίζουμε ΑΔΜΗΕ seed-robust** — XGB dense ΧΩΡΙΣ καιρό g12
  137.5/139.6/138.8 · g14 143.8/142.7/142.9 vs 146.81, 3 seeds × 2 gates (§7.23)·
  window-specific (q1/summer: ΑΔΜΗΕ μπροστά). (β) **dense: βοηθάει σε ΟΛΟΥΣ τους
  αλγορίθμους** (LGBM/XGB/MLP/LEAR × 3 windows × 2 gates· XGB 18/18 κελιά-seeds).
  (γ) vintage κρατά 36-78% του oracle οφέλους (Α6 δικαιωμένο). (δ) **recursive
  παραμένει default και για load**: direct κερδίζει ΜΟΝΟ summer (3/4), recursive
  octnov 5/5 + q1 5/5 (§7.23)· direct dense 3/3 / genlags 3/3 εσωτερικά συνεπή.
  (ε) **LSTM**: root-cause fix (exposure bias) → βέλτιστο config `calendar+loadfc`
  3/3 windows × 2 gates. Τρέχει τώρα: direct XGB 3 windows (cross-algo confirm).
  **VALIDITY REVIEW 2026-07-18** (`reports/qa/validity_review_load_20260718.md`):
  Ε1 dense ACCEPT · Ε2 LSTM-loadfc ACCEPT (static-only, non-tradeable eval flags) ·
  Ε3 direct dense/genlags ACCEPT (scoped) · Ε4 recursive-default PENDING (summer
  αναστροφή — μόνο interaction ή loadfc-scoped μορφή) · Ε5 XGB<ΑΔΜΗΕ octnov ACCEPT
  window-specific (headline ΑΠΟΡΡΙΦΘΗΚΕ). **Οριστικό ΔΕΚΤΟ = απόφαση χρήστη.**

## 2. VALIDITY GATE (συγχωνευμένο checklist — έλεγχος πριν από κάθε claim)

**Α-invariants (όλα ✅ εκτός αν σημειώνεται):**
- **Α1 leakage**: AEL ενεργό (crosslag_mode=freeze default) ✅ · poisoning
  `src/check_crosslag_fairness.py` PASS σε rec-dam/dir-dam/rec-forward, αυτόματο με
  `preflight_check.py --poison` ✅ · same-day xborder δομικά εκτός parquet ✅ · `tf` μόνο
  δηλωμένο oracle ✅ · νέα πηγή → 3-βήμα pre-flight (πότε δημοσιεύεται/lagscan/hour-profile) ✅.
  **Dense y-path (2026-07-05)** ✅: νέο Test D στο fairness script (`--poison_y`) — poison y +
  26 y_lag cols (μαζί dense 4..23) μετά το cutoff → diff=0.000000 σε rec ΚΑΙ dir, controls>0.
  Σημ.: το script πλέον κάνει `add_dense_lags` όταν το spec έχει dense (πριν, spec `dense`
  σιωπηλά δεν τεστάριζε τίποτα) — εντολή: `python -X utf8 -m src.check_crosslag_fairness
  --features default,dense --poison_y [--strategy direct]`.
  ⚠️ Ανοιχτό: LSTM eval encoder δεν φιλτράρεται (non-tradeable ούτως ή άλλως).
- **Α2 timezone**: CET/CEST-naive frame, μετατροπές ΜΟΝΟ στο data.py, guards στο preflight ✅.
  ⚠️ Ανοιχτό: `fetch_weather_2026.py` ζητά UTC ενώ το parquet είναι UTC+1 (πριν το επόμενο append).
- **Α3 μεθοδολογία**: αποδοχή ⟺ |ΔMAE|>0.15 ΚΑΙ ίδιο πρόσημο σε ≥2 ανεξάρτητες συνθήκες ·
  ζεύγος (χειμώνας, καλοκαίρι) ανά ομάδα · headline μόνο με ≥3 seeds + 2ο window ·
  control-run σε κάθε δομική αλλαγή · reproducibility anchor ±0.05 · ensemble βάρη μόνο
  out-of-sample · συγκρίσεις πάντα ίδιο window/gate/data — ΠΟΤΕ παλιά suspended δίπλα σε νέα.
- **Α4 probabilistic**: conformal αυστηρά αιτιατό (trailing 4-8 εβδ., per-hour) — κώδικας
  υπάρχει ✅ · coverage ΠΑΝΤΑ μαζί με sharpness+pinball · ≥2 μοντέλα × ≥2 windows.
- **Α5 traceability**: κάθε νούμερο → JSON/CSV path σε runs/-results/ + εντολή αναπαραγωγής ·
  MD = μόνο τρέχουσα αλήθεια (παλιά → OLD/docs) · commit+push μετά από κάθε session
  (⚠️ ΠΟΤΕ push το `FEB272026_localhistory` — 4.3GB) · ⚠️ εκκρεμεί environment.yml.
- **Α6 exogenous-oracle (2026-07-10) ⚠️ ΑΝΟΙΧΤΟ**: τα `w_*` (weather) είναι **observed** τιμές
  (Open-Meteo **Archive** API, `src/fetch_weather_2026.py`) = ο πραγματοποιημένος καιρός της
  ώρας-στόχου → στο gate D-1 12:00 είναι **μελλοντική πληροφορία** («oracle covariate» /
  perfect-weather-foresight). ΔΕΝ είναι target leakage AEL (exogenous covariate — τα
  poisoning/crosslag ΔΕΝ το πιάνουν), γι' αυτό πέρασε απαρατήρητο· ισχύει για meteo σε **load
  ΚΑΙ price**. Κανόνες: (1) κάθε meteo-based αποτέλεσμα δηλώνεται ρητά **«oracle weather»**
  μέχρι να μπει vintage· (2) **ΚΑΜΙΑ σύγκριση/νίκη έναντι ΑΔΜΗΕ για load δεν γράφεται ΔΕΚΤΗ με
  oracle meteo** — μόνο το no-meteo baseline είναι καθαρό. Τα lags/rolls/calendar 100% καθαρά.
  **✅ Fix ΥΛΟΠΟΙΗΘΗΚΕ (2026-07-10/11, G5 DONE)**: ομάδα **`meteo_vintage`** (Open-Meteo
  **Previous Runs API** — ΟΧΙ το Historical Forecast, δεν δίνει vintage) με gate-aware blend
  `wveff_* = day1 αν h ≤ 23−gap αλλιώς day2`, ωμά `wv_*` δομικά εκτός όλων των ομάδων,
  poison+control+unit tests PASS — για **task=load μόνο**· το price meteo παραμένει oracle
  (κανόνας (1) σε ισχύ εκεί). Docs: `docs/features/meteo_vintage/{design,deploy}.md`.

**Β-σειρά εργασιών:** Β1 ✅ → **Β2** σχεδόν ✅ (μένουν: worktree prune, environment.yml,
fetch_weather TZ) → **Β3 ΤΩΡΑ** (re-ablation ✅ recursive/⏳ direct → cadence+3 seeds+Μάρτιος
→ **ΝΕΟ HEADLINE** → SS×weekly, weekly LGBM+XGB confirm) → **Β4** conformal (≥2 μοντέλα ×
Q1+Μάρτιος → pinball/coverage/sharpness) → **Β5** πληρότητα (task=load ablation ·
henex_premarket · xb_lag1_h0 · weather forecast archive · Chronos/TimesFM · LSTM calibration)
→ **Β6** προϊόν (daily runner, settle loop, delivery).

## 3. Κλειδωμένα συμπεράσματα (πλήρης τεκμηρίωση: `ABLATION_PLAN §5`)

1. **xborder ΕΚΤΟΣ default** — same-day = leakage (14.43/15.02 ΑΚΥΡΑ)· lagged βλάπτει χειμώνα.
2. **recursive > direct για DAM στο weekly (deployable) cadence — ΟΡΙΣΤΙΚΟ (2026-07-06)**:
   4/4 συνθήκες (Q1+Summer × LGBM+XGB, weekly retrain), Δ=−1.1 έως −2.9, ACCEPTED. Στο
   static cadence παραμένει strategy×cadence interaction (3/4 windows recursive, μόνο
   Q1-static ευνοεί direct) — καταγεγραμμένο ξεχωριστά, ΔΕΝ επηρεάζει το production
   config αφού το headline χρησιμοποιεί weekly. Πλήρης ανάλυση: `ABLATION_PLAN §5.12ζ`.
3. **meteo ΒΟΗΘΑΕΙ recursive** (Β3 4/4, ΑΝΑΤΡΟΠΗ του παλιού strategy-effect που μετρήθηκε
   σε leaked configs)· στο direct βοηθούσε πάντα και προ-AEL. ⚠️ **(2026-07-10) με oracle
   weather** — το «βοηθάει» ισχύει ως feature-value, αλλά ΚΑΜΙΑ νίκη-ΑΔΜΗΕ δεν στηρίζεται σε
   αυτό πριν vintage (§2 Α6 · GOALS G5).
4. **dense ΒΟΗΘΑΕΙ** (Β3 4/4, νέο).
5. **resfc/genlags**: leak-dependent — μετά το AEL, resfc=εποχιακό flip, genlags=οριακό.
6. **SS-linear ΔΕΝ είναι redundant με retrain** (−0.24 και σε monthly)· SS×weekly αδοκίμαστο.
7. **Ensembles: κανένα δεν κερδίζει στιβαρά το LGBM** — μόνο weekly LGBM+XGB (−0.148) PENDING.
8. **LEAR = fallback μόνο** (χειρότερο με πλήρη δεδομένα)· MLP bare-core > default· LSTM bug.

## 4. Λειτουργικοί κανόνες (αμετάβλητοι)

- `conda run -n epf --no-capture-output python -X utf8` · ΕΝΑ conda process ·
  multi-line python ΜΟΝΟ σε αρχείο (scripts/) · OneDrive πρέπει να τρέχει.
- Outputs: `runs/<study>/` + `results/*.csv` + `logs/` + `reports/` · πάντα `--out_json`.
- `--train_end` ΜΟΝΟ με static (monthly/weekly = expanding, το αγνοούν).
- Rebuild parquet: πάντα backup+σύγκριση πριν σβηστεί το παλιό.
- ΟΧΙ Optuna · strict gate default · tf μόνο διαγνωστικό.
- Online χρήστης → δώσε copy-paste block να τρέξει εκείνος + τι να επικολλήσει πίσω.
  Autonomous (/goal) → τρέχω εγώ, background, με progress ανά κατηγορία.
- Τέλος κάθε απάντησης με ανοιχτή δουλειά: PENDING + στάδιο + επόμενο prompt.

## 5. Επόμενο βήμα (ενημερώθηκε 2026-07-05 απόγευμα — OVERNIGHT BATCH ΤΕΛΕΙΩΣΕ, headline ΔΕΝ κλειδώνει)

`scripts/overnight_20260705.sh` **ΟΛΟΚΛΗΡΩΘΗΚΕ 2026-07-05 13:34:56** (122/122 runs) +
`scripts/followup_20260705.sh` **ΟΛΟΚΛΗΡΩΘΗΚΕ 2026-07-06 09:24:33** (12/12 runs, 0 FAILED) →
🏆 **headline ΚΛΕΙΔΩΣΕ** (βλ. §1). Verdicts πλήρη στο `ABLATION_PLAN §5.12` (πολλαπλά περάσματα
validity-reviewer). Απομένουν 2 από τα 3 αρχικά blockers (ο #1, seeds, λύθηκε):

1. ~~Ξανατρέξε Block D seeds με weekly~~ ✅ **ΕΓΙΝΕ 2026-07-06** — headline κλειδωμένο.
2. **`check_crosslag_fairness.py --task load`** (recursive+direct) — ΠΟΤΕ δεν έτρεξε για load.
   Χρειάζεται πριν γραφτεί οτιδήποτε από τα Block C arms (genlags/loadlags/meteo/dense).
3. **Block E SS σε 2ο algorithm (XGB) ή 3ο window** πριν το `default` (χωρίς dense) verdict
   γίνει οριστικό ΔΕΚΤΟ αντί για PENDING.

Δευτερεύοντα (όχι μπλοκάρουν headline): conformal 2ο window (Μάρτιος) + 2ο μοντέλο (XGB
weekly)· fix το `overnight_summarize.py` conformal-schema parsing bug (§5.12στ)· rebuild
`hourly_load.parquet` με `load_fc` (data/processed/ προστατευμένο, backup+σύγκριση πρώτα).
Οι 2 παλιές ανοιχτές συγκρούσεις Μαρτίου (meteo, lean-core-direct, §5.12β) ΠΑΡΑΜΕΝΟΥΝ PENDING
— χρειάζονται 3ο ανεξάρτητο window, δεν άλλαξαν σε αυτό το batch.

**✅ Follow-up batch ΟΛΟΚΛΗΡΩΘΗΚΕ 2026-07-06 09:24:33** (`scripts/followup_20260705.sh`,
12/12 runs, 0 FAILED) — Block G έλυσε το seed-cadence bug → **headline κλειδωμένο** (§1)·
Block H (weekly direct+dense) + Block I (4ο window, Οκτ-Νοε 2025) έλυσαν οριστικά το
recursive-vs-direct ερώτημα (§5.12ζ/§3 σημείο 2, validity-reviewer 2 ξεχωριστά ACCEPT).

**Σημείωση εγκυρότητας (2026-07-05):** η ιδέα «published y actuals όσο επιτρέπεται» (E1)
αποδείχθηκε ΗΔΗ υλοποιημένη για price/DAM — gap=0, cutoff=23:00 D-1
(feature_availability.py:236) και poisoning-proven (Test D2: poison D-1 → preds άλλαξαν).
Κανένα engine change δεν έγινε. Το dense y-path είναι πλέον poison-tested (βλ. §2/Α1).

**Έτοιμο prompt συνέχειας (τα 3 blockers, με τη σειρά):**
```
Διάβασε last.md §5 + ABLATION_PLAN §5.12δ/γ/ε (τι μπλοκάρει το headline). Κάνε τα 3
βήματα με τη σειρά: (1) re-run Block D seeds με --retrain weekly, (2) task=load
crosslag poisoning check, (3) Block E SS σε XGB. Πέρασέ τα από validity-reviewer πριν
γράψεις οτιδήποτε ACCEPTED. Αν το seed re-run πιάνει το ~0.05-0.15 anchor γύρω από
17.035/13.812 → κλείδωσε το headline και ενημέρωσε last.md §1-§3 + §5.8.
```

- **`henex_premarket` design doc (2026-07-06)**: `docs/features/henex_premarket/design.md`
  μέσω `feature-eng` skill, Στάδιο 1 μόνο (χωρίς conda, ενώ έτρεχε το followup batch).
  **BLOCKED πριν το lagscan** — 3 πραγματικά ευρήματα: data gap (parquet σταματάει
  2026-01-01, μόνο 1 πλήρες window διαθέσιμο), `data_future.py` ορφανό+λάθος path,
  T1 gate-timing ανεπιβεβαίωτο. Πλήρες σε `ABLATION_PLAN §7.9`.

## 6. Ανοιχτά υποδομής (μικρά)

- ✅ **Claude Code setup (2026-07-05, βάσει deep-research-report.md)**: `CLAUDE.md` (guidance,
  δείχνει εδώ §2 + ABLATION §2) · `.claude/settings.json` hooks → `scripts/claude_hooks/guard_edits.py`
  (deny: data/raw, data/processed, OLD/ · ask: leakage-sensitive src πυρήνας) ·
  `.claude/agents/validity-reviewer.md` (subagent ελέγχου claims) ·
  `scripts/build_run_ledger.py` → `results/run_ledger.csv` (auditability index, system python,
  χωρίς conda — στήλη crosslag_mode κενή = προ-AEL run). ✅ **Skills synthesize-ablation +
  ingest-audit ΦΤΙΑΧΤΗΚΑΝ (2026-07-05)** (`.claude/skills/`) μαζί με το εργαλείο τους
  `scripts/synthesize_ablation.py` (system python, ΔMAE πίνακες + αυτόματο §2 pre-gate με
  κανόνα ανεξαρτησίας windows — δοκιμασμένο σε b_march/a_cadence, αναπαράγει τα verdicts
  του validity-reviewer). ✅ GitHub Actions: Claude Code Action live σε epf_greece_starter
  (main) + DIPLOMATIKI (2026-07-05, test issues πέρασαν). Εκκρεμή από το report
  (προαιρετικά): per-run manifests. ✅ **Skill `feature-eng` ΦΤΙΑΧΤΗΚΕ (2026-07-06)** —
  end-to-end FeatureENG orchestrator για data_in extension: TDD pre-registration T1-T8
  (design-template) → ingest-audit → υλοποίηση → batch → synthesize-ablation → deploy
  checklist με validator (`.claude/skills/feature-eng/scripts/validate_deploy_checklist.py`,
  smoke-tested FAIL-σε-template/PASS-σε-γεμάτο)· mirrored στο `plugins/epf-ops/skills/`.
  Artifacts ανά feature: `docs/features/<name>/{design,deploy}.md`· το deploy.md
  προορίζεται ως πηγή εγγραφής για το μελλοντικό Feature Registry (compounding spec Rung 2).
  **v1.1 (2026-07-06 απόγευμα)**: T0 data-coverage check στο Στάδιο 0 · T1 απαιτεί πρωτογενή
  πηγή (όχι οικονομική λογική) · staging-merge σειρά για ολοκαίνουρια πηγή (lagscan μόνο μετά)
  · read-only inspections με Python311+fastparquet (όχι conda) · structural-break awareness
  (SDAC 15-min MTU 2025-10-01, lignite exit 2026) στα templates — μαθήματα από το henex
  dry-run. Handoff report για master agent: `docs/features/FEATURE_ENG_AGENT_REPORT.md`.

- **`data/processed/hourly_load.parquet` λείπει η στήλη `load_fc`** (2026-07-05, βρέθηκε
  στο Block C LOAD ablation) — το αρχείο που φορτώνεται για task=load δεν έχει καθόλου
  day-ahead load forecast, σε αντίθεση με το `hourly.parquet` του price task. Χρειάζεται
  rebuild (merge με `load_forecast_hourly.parquet`) πριν ξανατρέξει το `-loadfc` ablation
  arm· backup+σύγκριση πριν σβηστεί το παλιό, όπως πάντα με parquet rebuilds. ΟΧΙ τώρα
  εν μέσω του overnight batch.
- `git worktree prune` (ορφανά — προσπάθεια 2026-07-05: permission denied σε πολλά, Windows
  file lock· χρειάζεται χειροκίνητο cleanup/restart) · `conda env export -n epf > environment.yml` ·
  fetch_weather_2026.py TZ align (A2) · root untracked: deep-research-report.md,
  Energy Forecasting Session.pdf, energy-runner_SKILL.md (εγκαταστάθηκε ως skill 2026-07-05 —
  το root αντίγραφο μπορεί να μπει στο repo ή OLD/).
- Ξεχωριστό `.venv/` στο root (σύγχυση με conda — δεν αγγίχτηκε).
- solar_fc ύποπτο 2h shift στο 2024-25 κομμάτι → `solar_shift_check.py` (ABLATION §7.8).
- **`scripts/overnight_summarize.py` δεν διαβάζει το conformal JSON schema** (`results.<window>.
  mae_p50`/`avg_pinball`/`coverage`) — τυπώνει λάθος "NO-MAE" για Block F ενώ τα δεδομένα
  υπάρχουν (βρέθηκε 2026-07-05 στο validity audit του overnight batch, βλ. `ABLATION_PLAN
  §5.12στ`). Μικρό, όχι leakage-sensitive — μπορεί να διορθωθεί όποτε βολεύει.
- ✅ **`last.md`/`ABLATION_PLAN.md`/design docs μετακινήθηκαν στο `MARKDOWN/`** (2026-07-05,
  commit `bb12537`, οριστικό) — μελλοντικές αναφορές/edits σε ΑΥΤΑ τα paths (`MARKDOWN/last.md`,
  `MARKDOWN/ABLATION_PLAN.md`), ΟΧΙ στο root (επιβεβαιώθηκε 2026-07-06: `git status` καθαρό,
  καμία εκκρεμότητα commit).
- **Figures παράχθηκαν 2026-07-05 15:50** (`reports/overnight_20260705_figures/`: MAE-by-block,
  conformal calibration Q1, run-ledger MAE-over-time) — καλύπτουν το overnight batch, όχι ακόμα
  τα thesis-ready figures (`thesis/content/*.tex` παραμένει στο προ-TZFIX/προ-AEL Δεκ-2025
  campaign, βλ. νέο ανοιχτό θέμα παρακάτω).
- **`thesis/content/experiments.tex` + `conclusions.tex` είναι ΞΕΠΕΡΑΣΜΕΝΑ** (τελευταία
  τροποποίηση 2026-07-02, πριν το TZFIX/AEL) — δείχνουν το παλιό static Δεκ-2025 campaign
  (Naive-1=13.709, Ensemble-Best3=14.512 κ.λπ.), άσχετο πλέον με το τρέχον leak-free Q1/
  summer campaign. ΔΕΝ πρέπει να ξαναγραφτούν πριν κλειδώσει ο νέος headline (βλ. §5) —
  αλλιώς θα ξαναγραφτούν δύο φορές.
