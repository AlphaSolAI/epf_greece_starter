# CORE-DIFF review — meteo_vintage wiring (2026-07-10, inline*)

*Inline review (όχι subagent) — ρητή οδηγία χρήστη 2026-07-10: subagents ΜΟΝΟ κατόπιν εντολής.

## Scope

- `src/feature_availability.py`: νέα ομάδα `meteo_vintage` (ΕΚΤΟΣ default) ·
  δομικός αποκλεισμός ωμών `wv_*` από classify_columns · `meteo_vintage_day1_ok`
  (h ≤ 23−gap) · `add_meteo_vintage_features` (blend day1/day2 + _missing flags,
  fail-loud: unpaired sibling / market=forward με ζεύγη παρόντα).
- `src/master_forecast.py`: import + callsite `if "meteo_vintage" in groups:` μετά το
  ramp, πριν το dropna — το GateSpec (με --delay override) είναι ήδη χτισμένο.
- `tests/test_feature_availability.py`: +8 tests (classification/spec/boundaries/blend/fail-loud).

## Ευρήματα

1. **FIXED κατά το review**: το market guard (forward → ValueError) έτρεχε ΠΡΙΝ τον
   έλεγχο ζευγών — price run `--features all --market forward` (0 wv στήλες) θα
   έσκαγε άδικα. Μετακινήθηκε ΜΕΤΑ το pair discovery· no-op χωρίς ζεύγη ανεξαρτήτως
   market. Κλειδώθηκε με test (out_fw).
2. Boundary semantics επαληθευμένα έναντι `GateSpec.cutoff_for_block`
   (cutoff = D-1 (23−gap):00): issue(day1)=D-1 h:00 ≤ cutoff ⟺ h ≤ 23−gap.
   g12→h≤11, g14→h≤9 — ταυτίζονται με το G5 σχέδιο. Unit tests στα όρια (11/12, 9/10).
3. Ωμά `wv_*` έπεφταν στο `other` ⊂ DEFAULT_GROUPS (unclassified fallthrough) —
   ο δομικός αποκλεισμός κλείνει πραγματικό μελλοντικό leak path (μέχρι σήμερα
   αθέατο μόνο επειδή το G7 τρέχει explicit specs). Νέο γενικό μάθημα καταγεγραμμένο
   σε design.md §5 + ABLATION_PLAN §7.14.
4. AEL: wveff εκτός crosslag families (exogenous forecast) — freeze paths άθικτα.
   Idempotent (`if eff in df.columns: continue`), copy semantics, DST-safe (naive index.hour).
5. Κανένα υπάρχον config δεν αλλάζει: ομάδα opt-in, ωμά wv αδρανή.

## Verdict: **APPROVE** (μετά το fix #1)

## Evidence

- pytest: 94 passed, 1 skipped (2×).
- `preflight_check.py --poison`: PASS rec+dir (`logs/mv_wiring_preflight_poison.log`).
- Control (T6): base Q1 static MAE=254.9893 ΤΑΥΤΟΣΗΜΟ, #features=17
  (`runs/feat_meteo_vintage/_control_postwiring_q1_lgbm_rec_base.json`).
- Smoke (T4): mv Q1 static #features=101, MAE=221.2821
  (`runs/feat_meteo_vintage/q1_lgbm_rec_static_mv.json`) — Δ=−33.7, εντός oracle
  φράγματος (T8 ✓, oracle=160.11).
