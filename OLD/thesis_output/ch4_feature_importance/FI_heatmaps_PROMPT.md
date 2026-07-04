# Prompt — Heatmaps Σημαντικότητας Χαρακτηριστικών (Feature Importance)

Χρησιμοποίησε αυτό το prompt με τον Claude για να (ανα)παράγεις ή να βελτιώσεις
τα FI heatmaps της διπλωματικής (Κεφ. 8). Δύο είδη: (Α) σύνολο ανά στρατηγική,
(Β) ανά στρατηγική με κάθε μοντέλο.

---

## Δεδομένα (μοναδική πηγή αλήθειας)
Αρχείο: `thesis_output/ch4_feature_importance/appendix_fi_data.csv`
Στήλες: `task, strategy, model, importance_type, rank, feature, importance, share_pct, source_file`
- `task`: `price` | `load`
- `strategy`: `Teacher-Forcing` | `Recursive` | `MIMO` | `Direct`
- `model`: `lgbm` | `xgb` | `rf` | `mlp` | `svr`
- `importance_type`: `gain` (lgbm/xgb) · `MDI (impurity)` (rf) · `permutation` (mlp/svr)
- `share_pct`: ποσοστό σημαντικότητας (το νούμερο που μπαίνει στο κελί)
- `rank`: 1 = σημαντικότερο (ανά task×strategy×model)

Διαθεσιμότητα μοντέλων ανά στρατηγική: TF & Recursive = 5 (lgbm,xgb,rf,mlp,svr) ·
MIMO = 4 (xgb,rf,mlp,svr — όχι lgbm) · Direct = 2 (lgbm,xgb).

## Κανόνας επιλογής 10 χαρακτηριστικών (rank-based, ΟΧΙ μέσος όρος %)
Αιτία: οι κατανομές διαφέρουν δραστικά σε συγκέντρωση (TF: lag1≈88%, MIMO: επίπεδη
~2–4%). Ο μέσος όρος raw % θα έκρυβε τα χαρακτηριστικά που μετράνε μόνο στο MIMO.
Ο κανόνας με κατάταξη είναι ανεξάρτητος κλίμακας και εκπροσωπεί δίκαια κάθε στήλη:
1. Από κάθε στήλη (στρατηγική ή μοντέλο) κράτα τα **top-3 κατά rank**.
2. Ένωσέ τα. Αν >10, κράτα τα 10 με την καλύτερη **μέση κατάταξη**.
3. Αν <10, συμπλήρωσε κατά μέση κατάταξη μέχρι τα 10.
4. Σειρά γραμμών: αύξουσα μέση κατάταξη (πιο διαχρονικά σημαντικά πάνω).
Χαρακτηριστικό που λείπει από μια στήλη → κελί κενό/γκρι (όχι 0).

## Είδος Α — Σύνολο ανά στρατηγική (`fig_fi_C_heatmap_{price,load}.png`)
- 2 σχήματα: ένα τιμή, ένα φορτίο.
- Γραμμές = top-10 χαρακτηριστικά. Στήλες = 4 στρατηγικές, με **champion μοντέλο gain**:
  TF→lgbm, Recursive→lgbm, Direct→lgbm, MIMO→xgb (όλα gain → χρώμα συγκρίσιμο).
- Χρώμα: τιμή `share_pct` με `PowerNorm(gamma≈0.45)` (το lag1=88% συμπιέζεται ώστε
  να φαίνονται και οι μικρές τιμές). Ένα colorbar.

## Είδος Β — Ανά στρατηγική, κάθε μοντέλο (`fig_fi_D_{tf,rec,mimo,direct}_{price,load}.png`)
- 8 σχήματα (4 στρατηγικές × 2 tasks). Γραμμές = top-10. Στήλες = τα διαθέσιμα μοντέλα.
- Στήλες με μέθοδο: `LGBM (gain)`, `XGB (gain)`, `RF (MDI)`, `MLP (perm)`, `SVR (perm)`.
- Χρώμα: **κανονικοποίηση ανά στήλη** (κάθε στήλη 0..max), γιατί gain/MDI/permutation
  ΔΕΝ συγκρίνονται σε απόλυτη κλίμακα. Σημείωση αυτού κάτω από το σχήμα.
- Annotation: πάντα το πραγματικό `share_pct`.

## Στυλ (κλειδωμένο — Κανούσης-inspired, light)
- Λευκό φόντο παντού· `cmap = magma_r`· κενά κελιά γκρι `#f2f2f2`.
- Αριθμοί κελιών: 1 δεκαδικό· **κενό αν <0.5%**· χρώμα κειμένου λευκό/σκούρο ανάλογα
  με τη φωτεινότητα του κελιού.
- Λευκοί διαχωριστές κελιών (minor grid). Ετικέτες χαρακτηριστικών σε Latin
  (Price_lag1, Load_lag24, wind_lag48, gas_price, hour, dow...). Τίτλος + colorbar Ελληνικά.
- Μέγεθος ~7×5.5 in (Α), πλάτος ανάλογο πλήθους στηλών (Β). dpi=200, `bbox_inches='tight'`.

## Ονοματολογία χαρακτηριστικών (raw → label)
`y_lagK`→`Price_lagK`/`Load_lagK` (ανά task)· `y_rollK`→`Price_roll`/`Load_roll`·
`gen_wind_lagK`→`wind_lagK`· `gen_solar_lagK`→`solar_lagK`· `gas_lagK`→`gas_lagK`·
`gas_price`,`co2_price`,`load_fc`,`hour`,`hour_sin/cos`,`dow`,`dow_sin/cos`,`is_holiday` ως έχουν.

## Παραδείγματα εντολών προς τον Claude
- «Ξαναφτιάξε το Είδος Α για την τιμή με top-12 αντί 10, ίδιος κανόνας επιλογής.»
- «Στο Είδος Β (MIMO, φορτίο) βάλε και στήλη lgbm αν υπάρχει· αλλιώς εξήγησε γιατί λείπει.»
- «Άλλαξε cmap σε `cividis` και κράτα το PowerNorm· σύγκρινέ το με magma_r.»
- «Πρόσθεσε δεύτερη γραμμή annotation με το rank σε παρένθεση κάτω από το %.»

## Πού ζουν τα scripts (αναπαραγωγή)
- Είδος Α: λογική στο `/tmp/gen_fi_heatmap.py` (champions × στρατηγικές).
- Είδος Β: λογική στο `/tmp/gen_fi_heatmap_bystrat.py` (μοντέλα × στρατηγική).
Και τα δύο διαβάζουν `appendix_fi_data.csv` και γράφουν PNG στο
`thesis_output/ch4_feature_importance/`.
