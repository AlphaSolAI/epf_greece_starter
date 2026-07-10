# Deploy Checklist — meteo_vintage

> Συμπληρώθηκε 2026-07-11 (feature-eng Στάδιο 6), μετά το harvest του G7 Batch 3
> (vintage LGBM weekly, 12/12, 0 FAILED) και του Batch 2 (XGB, 30/30, 0 FAILED).

## 1. LEAKAGE-FREE PROOF

- Gate timing (T1): Open-Meteo Previous Runs buckets — day1 εκδίδεται 24h πριν το valid
  time (D-1 h:00 για στόχο D h:00)· blend ανά ώρα h ≤ 23−gap → day1, αλλιώς day2 (48h),
  άρα ΚΑΘΕ σερβιρισμένη τιμή έχει issue ≤ cutoff (11:00 CET D-1 στο g12 · 09:00 στο g14),
  δηλ. ΠΡΙΝ το gate 12:00 CET D-1.
- Lagscan (T2): `logs/g5_lagscan_wv_temp.log` (|corr|≤0.099, επίπεδο) ·
  `logs/g5_lagscan_wv_swr.log` (|corr|≤0.18, συμμετρικό γύρω από k=0, καμία spike).
- Hour-profile (T3): shortwave peak 11:00, temperature peak 13:00 / min 05:00 — φυσιολογικά,
  κανένα shift.
- Poisoning μετά το feature_availability.py edit (T5): `logs/mv_wiring_preflight_poison.log`
  — PASS (recursive+direct, leak_free=True) · επανάληψη στο Block 0 του batch:
  `logs/load_contest_vintage_lgbm.log` PASS.
- crosslag_mode: freeze (default) — επιβεβαιωμένο στα run JSONs: ναι
  (`runs/load_contest/*_mv.json` → crosslag_mode=freeze, crosslag_gap=12/14).
- Πληρότητα ανά task parquet (T4): price: ν.α. (0 wv στήλες by design — pipeline άθικτο) ·
  load: ναι (168 wv στο hourly_load.parquet, 0 NaN/0 missing στα q1/summer/octnov·
  #features 17→101 στο smoke log = non-VOID arm).

## 2. FEATURE IMPORTANCE

- FI run: `conda run -n epf --no-capture-output python -X utf8 scripts/fi_meteo_vintage_q1.py`
  → `reports/fi_meteo_vintage_q1.txt` (LGBM Q1-static fit, spec calendar,lags,roll,meteo_vintage).
- Rank νέας ομάδας: κάτω από lags/calendar (ομάδα 0.05% in-sample gain, top wveff rank
  18/101) vs pre-registered «κάτω από lags/calendar, τάξη του dense». Ποιοτικά συνεπές
  (κάτω από lags/calendar ✓)· το in-sample gain υποτιμά συστηματικά τα exogenous σε
  recursive setup: στο fit κυριαρχεί το y_lag1 (92%), στο serve τα y-lags γίνονται
  προβλέψεις ενώ το wveff μένει αληθινό — το όφελος εμφανίζεται στο rollout (βλ. §3).
- Too-good flag (T8): όχι — vintage ΧΕΙΡΟΤΕΡΟ από oracle σε ΟΛΑ τα συγκρίσιμα κελιά
  (q1 weekly 214.8 vs oracle 157.4 · summer 264.3 vs 237.2 · static Q1 221.3 vs 160.1)·
  κανένα targeted poisoning δεν χρειάστηκε.

## 3. EXPECTED vs ACTUAL

| Pre-registered (design.md §2) | Μετρημένο |
|---|---|
| Πρόσημο: βελτίωση σε όλα τα windows | Βελτίωση 6/6 (3 windows × 2 gates, LGBM weekly) ✓ |
| Μέγεθος: 50-100% του oracle οφέλους | static Q1: 36% (−33.7 vs −94.9) — ΚΑΤΩ από το εύρος· weekly: q1 42% / summer 78% — εντός/κοντά |
| Πού δρα: task=load, ισχυρότερα χειμώνα | Δρα παντού· ισχυρότερα ΚΑΛΟΚΑΙΡΙ σε weekly (−94/−108) — μερική απόκλιση |

- Πίνακας ΔMAE: `reports/feature_lifecycle_meteo_vintage_20260710.md §5` (πλήρης, ανά gate/window)·
  runs: `runs/load_contest/*_{mv,densemv}.json` + smoke `runs/feat_meteo_vintage/`.
- Αποκλίσεις & ερμηνεία: (α) το static μέγεθος βγήκε κάτω από το pre-registered εύρος —
  τίμιο miss, οι μισές ώρες σερβίρονται με 48ωρο day2 forecast· (β) το «ισχυρότερα
  χειμώνα» ίσχυε στο static, στο weekly το καλοκαίρι ωφελείται περισσότερο (cooling load
  = το πιο weather-εξαρτημένο κομμάτι· βλ. διαγνωστικό: το σφάλμα μας δεν συσχετίζεται
  με Tmax αλλά με regime μεταβάσεις, και ο καιρός αγκυρώνει το recursive rollout —
  σφάλμα 03:00-05:00 ≈ 135 MW vs 14:00-18:00 ≈ 315-325 MW χωρίs τέτοια άγκυρα).

## 4. KPI & VERDICT

- §2 pre-gate (T7): PASS — ΔMAE vs base: g12 q1 −41.2 / summer −94.4 / octnov −20.6 ·
  g14 −33.3 / −107.8 / −24.6 (ίδιο πρόσημο σε 3 ανεξάρτητα windows, |Δ| ≫ 0.15).
- validity-reviewer verdict: ΔΕΝ έτρεξε (subagents μόνο κατόπιν εντολής χρήστη,
  κανόνας 2026-07-10) — inline pre-gate 2026-07-11, τελική αποδοχή στον χρήστη.
- Απόφαση default set: ΔΕΝ μπαίνει στο default — μένει flag-μόνο (`meteo_vintage`),
  συνεπές με τον κανόνα «task=load contest arms με explicit specs». Το oracle `meteo`
  παραμένει ΞΕΧΩΡΙΣΤΗ ομάδα, δηλωμένη oracle (last.md §2 Α6) — ΑΝΘΡΩΠΙΝΗ έγκριση: εκκρεμεί.
- Headline impact: υποψήφιο ΜΟΝΟ για το octnov window-specific claim «καθαρό μοντέλο
  κερδίζει ΑΔΜΗΕ» (mv/densemv 128.5-133.1 vs 146.8, 4/4 κελιά· + XGB dense 137.5/143.8
  χωρίς καιρό) — για claim επιπέδου headline: ≥3 seeds + validity review: ΕΚΚΡΕΜΕΙ.
- KPI αναφοράς: MAE baseline 256.01 → με feature 214.81 (MW, window: Q1 g12 LGBM weekly)·
  153.64 → 133.09 (octnov g12, όπου κερδίζεται και ο ΑΔΜΗΕ 146.81).

## 5. DANGERS & ROLLBACK

- Κίνδυνοι που παραμένουν: (α) υπόθεση issue=valid−24h του API — αν ο πραγματικός
  χρόνος run είναι αργότερα, οριακό h στο g12 μπορεί να αγγίζει το cutoff (mitigation:
  g14 head-to-head με 2h μαξιλάρι)· (β) coverage από 2024-02 — training tail 0-filled
  με flags, νέο fetch χρειάζεται για νέα windows· (γ) πηγή τρίτου (Open-Meteo) μπορεί
  να αλλάξει schema/διαθεσιμότητα· (δ) forward market ΔΕΝ υποστηρίζεται (day3+ buckets
  δεν υπάρχουν — ρητό ValueError by design).
- Rollback: η ομάδα είναι opt-in flag — αφαίρεση = μη συμπερίληψη του `meteo_vintage`
  στο `--features` spec· καμία αλλαγή πυρήνα δεν χρειάζεται (control run απέδειξε
  μηδενική επίδραση σε specs χωρίς την ομάδα).
- Monitoring: rolling MAE(wv_day1 vs actual) ανά μήνα (τώρα 0.86°C/13.1 W/m²) — αύξηση
  = χάλασε το feed/TZ· ΔMAE(mv vs base) αλλάζει πρόσημο σε νέο window = investigate
  πριν από κάθε νέο claim· `_missing` flags > 0 σε eval window = κενό fetch.

## 6. TRACE

- Design doc: `docs/features/meteo_vintage/design.md`
- Runs: `runs/feat_meteo_vintage/` (smoke+control) · `runs/load_contest/*_{mv,densemv}.json`
  (weekly batch) · logs: `logs/load_contest_vintage_lgbm.log`, `logs/mv_wiring_preflight_poison.log`
- Εντολή αναπαραγωγής (πλήρης): `conda run -n epf --no-capture-output python -X utf8 -m src.master_forecast --algo lgbm --task load --market dam --strategy recursive --gate strict --retrain weekly --test_start "2025-10-01 00:00" --test_end "2025-11-30 23:00" --features "calendar,lags,roll,meteo_vintage" --seed 42 --out_json runs/load_contest/octnov_lgbm_recw_g12_mv.json`
- ABLATION_PLAN εγγραφή: §7.14 · last.md ενημερώθηκε: ναι (2026-07-11)
