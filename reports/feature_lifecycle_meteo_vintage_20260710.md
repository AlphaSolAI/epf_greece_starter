# Feature lifecycle report — meteo_vintage (έντιμος καιρός για το load contest)

> **Σκοπός εγγράφου**: παρουσιάσιμη, αυτοτελής αφήγηση της διαδικασίας (audit → design →
> TDD wiring → validation → κλιμακούμενη εκτέλεση → verdict) και των ευρημάτων, με πλήρη
> traces. Πηγές αλήθειας: `docs/features/meteo_vintage/design.md` (pre-registration),
> `MARKDOWN/ABLATION_PLAN.md §7.14`, `MARKDOWN/GOALS.md` G5/G7.
> Γράφτηκε 2026-07-10 (ενημερώνεται στο harvest του G7 Batch 3).

## 1. Το πρόβλημα — «oracle weather» (γιατί έπρεπε να γίνει)

Τα υπάρχοντα weather features (`w_*`, Open-Meteo **Archive** API) είναι **observed**
τιμές: ο πραγματοποιημένος καιρός της ώρας-στόχου. Στο gate D-1 12:00 CET αυτό είναι
**μελλοντική πληροφορία** («perfect weather foresight»). Δεν το πιάνει ΚΑΝΕΝΑ poisoning
test — είναι exogenous covariate, εκτός των AEL crosslag οικογενειών — γι' αυτό πέρασε
απαρατήρητο σε όλα τα προηγούμενα meteo ευρήματα (last.md §2 Α6, 2026-07-10).

Συνέπεια για το **load contest vs ΑΔΜΗΕ** (G7): κάθε meteo-based «νίκη» ήταν χάρτινη.
Το oracle meteo έδινε Q1 static Δ=−94.9 MW — αλλά με πληροφορία που δεν υπάρχει στο gate.

## 2. Η λύση — vintage forecasts (τι μπήκε)

**Open-Meteo Previous Runs API**: αρχειοθετημένες προγνώσεις με σταθερά lead-time buckets
— `previous_day1` = η τιμή που προβλέφθηκε 24h πριν το valid time, `previous_day2` = 48h.
(Το αρχικά προτεινόμενο «Historical Forecast API» απορρίφθηκε στο audit: ΔΕΝ δίνει vintage
— stitches τις πρώτες ώρες κάθε run = σχεδόν real-time.)

**Availability rule** (η καρδιά της εντιμότητας): για στόχο την ώρα h της ημέρας D,
το day1 bucket εκδόθηκε D-1 h:00. Νόμιμο στο gate ⟺ h ≤ 23−gap:
- **g12** (δικό μας gate, cutoff 11:00 D-1): ώρες 00-11 → day1 (φρέσκο), ώρες 12-23 → day2
- **g14** (ΑΔΜΗΕ-aligned, cutoff 09:00 D-1): ώρες 00-09 → day1, ώρες 10-23 → day2

Δηλαδή: ανά ώρα-στόχο επιλέγεται το **μικρότερο νόμιμο lead time** — ακριβώς ό,τι θα
έκανε ένας πραγματικός day-ahead forecaster με πρόσβαση στο ίδιο αρχείο προγνώσεων.

## 3. Διαδικασία (αναπαράξιμο playbook)

| Στάδιο | Τι έγινε | Trace |
|---|---|---|
| 0. Data ingest (G5, πρωί) | Fetcher `src/fetch_open_meteo_vintage.py` (5 πόλεις × 7 vars × day1/day2 + gr_mean) → staging parquet → δομικό join ΜΟΝΟ για task=load στο `src/data.py`, backup+bit-identical σύγκριση 135 υπαρχουσών στηλών | `data/processed/_backup_vintageweather_20260710/`, `scripts/compare_vintage_weather_load.py` 4/4 PASS |
| 0β. TZ audit | Ίδιο fixed-UTC+1 quirk με Archive API — επιβεβαίωση με cross-correlation vs έμπιστη `w_gr_mean_*` (DJF 0.969/JJA 0.988 ΑΚΡΙΒΩΣ σε k=0) → μετατροπή ΜΟΝΟ στον loader | `src/data.py::load_weather_vintage_hourly` |
| 1. Design/pre-registration (TDD) | Μηχανισμός, αναμενόμενο πρόσημο/μέγεθος, T1-T8 acceptance tests ΚΛΕΙΔΩΣΑΝ πριν το wiring | `docs/features/meteo_vintage/design.md` |
| 2. Ingest-audit (διαβατήριο) | T1 gate-timing γραπτά · T2 lagscan (\|corr\|≤0.099 temp / ≤0.18 swr, καμία spike) · T3 hour-profile (swr peak 11:00, temp 13:00) · βήμα-4 πληρότητα ανά task parquet (168 wv στο load, 0 στο price, 0 NaN στα 3 windows) | `logs/g5_lagscan_wv_{temp,swr}.log`, ABLATION §7.14 |
| 3. Wiring (CORE-DIFF) | Tests ΠΡΩΤΑ (+8, όρια g12 h=11/12 · g14 h=9/10) → ομάδα `meteo_vintage` + blend `wveff_*` + **δομικός αποκλεισμός ωμών `wv_*`** → inline CORE-DIFF review (1 fix: market guard μετά το pair discovery) | `reports/qa/20260710_coredriff_meteo_vintage_wiring.md` |
| 4. Validation | pytest 94 PASS · poison PASS rec+dir · control base Q1 static = 254.9893 **ΤΑΥΤΟΣΗΜΟ** (#features=17) | `logs/mv_wiring_preflight_poison.log`, `runs/feat_meteo_vintage/_control_postwiring_q1_lgbm_rec_base.json` |
| 5. Smoke (κλίμακα 1) | static Q1 mv: #features 17→**101** (T4 non-VOID), MAE=**221.28** | `runs/feat_meteo_vintage/q1_lgbm_rec_static_mv.json` |
| 6. Batch (κλίμακα 2-3) | Detached chain: 12 weekly LGBM vintage runs (2 arms × 3 windows × 2 gates) → XGB Batch 2 (30 runs) | `scripts/load_contest_vintage_lgbm.sh`, logs: `load_contest_vintage_lgbm.log` / `load_contest_xgb_weekly.log` |
| 7. Verdict | synthesize-ablation + §2 pre-gate — **εκκρεμεί το harvest** | (θα συμπληρωθεί) |

## 4. Ευρήματα μέχρι τώρα

1. **Smoke (static Q1, seed 42):** base 254.99 → +meteo_vintage **221.28** (Δ=−33.7)
   → +meteo_oracle 160.11 (Δ=−94.9). Το vintage κρατά **~36%** του oracle οφέλους —
   ΚΑΤΩ από το pre-registered 50-100% (honest miss του pre-registration, καταγεγραμμένο).
   Ερμηνεία: οι μισές ώρες (12-23 στο g12) σερβίρονται με day2 = 48ωρη πρόγνωση, και το
   NWP σφάλμα μεγαλώνει με το lead time. **Ο «καθρέφτης» oracle→vintage είναι το ίδιο
   το μέγεθος του oracle bias**: ό,τι αναφερόταν ως meteo όφελος ήταν ~2.8× υπερεκτιμημένο.
2. **T8 sanity ✓**: το vintage είναι ΧΕΙΡΟΤΕΡΟ από το oracle (όπως πρέπει — αν το κέρδιζε,
   θα ήταν red flag leak). Ταυτόχρονα σαφώς καλύτερο από το τίποτα → το meteo έχει
   πραγματική, νόμιμη αξία στο load, απλώς μικρότερη από την oracle ψευδαίσθηση.
3. **Δομικό μάθημα leak-prevention (νέο):** unclassified στήλες parquet πέφτουν στην
   ομάδα `other` που είναι ΜΕΣΑ στο default set — κάθε νέα ωμή στήλη είναι σιωπηλά
   επιλέξιμη αν δεν ταξινομηθεί ΤΑΥΤΟΧΡΟΝΑ με το merge. Τα ωμά `wv_*` αποκλείστηκαν
   δομικά στο `classify_columns` (δεν ανήκουν σε ΚΑΜΙΑ ομάδα — μόνο το gate-aware
   blend `wveff_*` είναι feature).
4. **Λειτουργικό μάθημα:** το PS 5.1 `Start-Process` ενώνει ArgumentList ΧΩΡΙΣ quoting
   → πολυ-λεκτικό `-c "cmd1; cmd2"` χρειάζεται ενσωματωμένα διπλά quotes, αλλιώς
   σιωπηλό no-op launch (διορθώθηκε το pattern στο energy-forecast SKILL.md).

## 5. Αποτελέσματα weekly contest (harvest 2026-07-10 23:02 — 12/12, 0 FAILED)

LGBM weekly recursive, seed 42. MAE σε MW· ΑΔΜΗΕ benchmark: q1 175.56 / summer 170.76 /
octnov 146.81.

| gate | arm | q1 | summer | octnov | Δ vs base (q1/su/on) |
|---|---|---|---|---|---|
| g12 | base | 256.01 | 358.65 | 153.64 | — |
| g12 | mv | 214.81 | 264.26 | **133.09** | −41.2 / −94.4 / −20.6 |
| g12 | densemv | 216.53 | 250.45 | **128.46** | −39.5 / −108.2 / −25.2 |
| g14 | base | 253.41 | 385.52 | 156.94 | — |
| g14 | mv | 220.09 | 277.74 | **132.38** | −33.3 / −107.8 / −24.6 |
| g14 | densemv | 224.70 | 259.75 | **132.87** | −28.7 / −125.8 / −24.1 |

- **T7 §2 pre-gate: PASS 6/6** — βελτίωση με ίδιο πρόσημο σε 3 ανεξάρτητα windows × 2 gates.
- **T8: PASS** — vintage ≤ oracle παντού (q1 214.8 vs oracle-weekly 157.4· summer 264.3 vs
  237.2). Κρατά ~42% (q1) / ~78% (summer) του oracle οφέλους στο weekly.
- **vs ΑΔΜΗΕ: νίκη ΜΟΝΟ στο octnov, αλλά 4/4 κελιά** (και στα δύο arms, ΚΑΙ στο
  ΑΔΜΗΕ-aligned g14) — έντιμη, window-specific. Q1/summer: ο ΑΔΜΗΕ προηγείται καθαρά.
- **Κατάσταση claim**: pre-gate PASS αλλά 1 algo / 1 seed — αποδοχή = ανθρώπινη απόφαση·
  headline-level θέλει ≥3 seeds + validity review.

**Συμπλήρωμα 2026-07-11 (XGB Batch 2, 30/30 0 FAILED):** το octnov εύρημα ενισχύθηκε —
XGB dense κερδίζει ΑΔΜΗΕ στο octnov ΚΑΙ χωρίς καιρό (137.46 g12 / 143.81 g14 vs 146.81),
άρα «καθαρό μοντέλο > ΑΔΜΗΕ στο octnov» ισχύει πλέον σε 2 αλγορίθμους × 2 gates (τα
vintage arms παραμένουν LGBM-only μέχρι XGB+mv run). Το dense πιάνει §2 με 12/12 κελιά
σε 2 αλγορίθμους. FI σημείωση (`reports/fi_meteo_vintage_q1.txt`): in-sample gain της
ομάδας 0.05% (top wveff rank 18/101) ενώ το serve-time όφελος είναι −20 έως −108 MW —
τα exogenous covariates υποτιμώνται συστηματικά από in-sample FI σε recursive setup
(στο serve τα y-lags είναι προβλέψεις, ο καιρός παραμένει αλήθεια)· διδακτικό σημείο
για τη διπλωματική.

## 6. Σχετικά αρχεία (πλήρης δείκτης)

- Design/pre-registration: `docs/features/meteo_vintage/design.md`
- Audit record: `MARKDOWN/ABLATION_PLAN.md §7.14` · lagscan logs `logs/g5_lagscan_wv_*.log`
- Wiring diff review: `reports/qa/20260710_coredriff_meteo_vintage_wiring.md`
- Engine: `src/feature_availability.py` (ομάδα+rule+blend) · `src/master_forecast.py` (callsite)
- Tests: `tests/test_feature_availability.py` (8 νέα)
- Runs: `runs/feat_meteo_vintage/` (smoke+control) · `runs/load_contest/` (weekly batch)
- Fetcher/loader: `src/fetch_open_meteo_vintage.py` · `src/data.py`
