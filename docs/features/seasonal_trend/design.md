# Feature Design Doc — seasonal_trend (`seas`)

> TDD pre-registration (feature-eng Στάδιο 1) — 2026-07-12, ΠΡΙΝ από κώδικα.
> Κίνητρο: goal χρήστη «γιατί τόσο μεγάλο σφάλμα το καλοκαίρι; δεν είναι λογικό — ίδια
> δεδομένα». Το /analyze απέδειξε ότι τα δεδομένα ΔΕΝ είναι «ίδια»: το ενδοημερήσιο σχήμα
> άλλαξε δομικά το 2025 (duck-curve deepening) και το μοντέλο δεν έχει κανένα feature
> εποχής/έτους για να το δει.

## 1. Ταυτότητα

- **Feature/πηγή:** ΚΑΜΙΑ εξωτερική πηγή — deterministic συναρτήσεις του index:
  `doy_sin = sin(2π·doy/365.25)` · `doy_cos = cos(2π·doy/365.25)` ·
  `t_trend = (t − 2015-01-01) σε ημέρες` (γραμμικός χρονικός δείκτης «εποχής»).
  Χτίζονται on-the-fly (pattern `engfc`/`ramp`/`meteo_vintage`) — ΚΑΜΙΑ αλλαγή parquet.
- **Ομάδα (flag):** `seas` — ΝΕΑ, ΕΚΤΟΣ default.
- **Στήλες:** 3 (`doy_sin`, `doy_cos`, `t_trend`) — και στα δύο tasks διαθέσιμες by
  construction (index-only)· probes εδώ: task=load.
- **Availability rule:** τετριμμένα νόμιμο — γνωστά ΑΙΩΝΕΣ πριν το gate (ημερολόγιο).
  Κανένα publication timing, κανένα AEL ζήτημα (όχι actuals).

## 2. Μηχανισμός & pre-registered προσδοκίες

- **Γιατί να βοηθάει (evidence-driven, 2026-07-12 deep-dive):**
  (α) **H1 era shift**: midday/evening ratio Ιουν-Ιουλ: 2023=1.159, 2024=1.153, **2025=1.086**
  (χάσμα mid−eve 1131→623 MW) — υπογραφή BTM PV growth. Το summer bias προφίλ μας:
  −207 MW @ 10:00 (υπερεκτίμηση) / +196 @ 18:00 (υποεκτίμηση) = το μοντέλο σερβίρει το
  ΜΕΣΟ ιστορικό σχήμα. Το `t_trend` δίνει στα δέντρα era-splits (και native interaction
  με hour/swr) ώστε να προσαρμόσουν το σχήμα ανά εποχή δεδομένων.
  (β) **H3 annual seasonality**: calendar ομάδα = hour/dow/holiday ΜΟΝΟ — καμία ετήσια
  εποχικότητα· doy_sin/cos κωδικοποιούν «πού στη χρονιά είμαστε» (τουρισμός, κλιματισμός).
- **Αναμενόμενο πρόσημο ΔMAE (summer static):** βελτίωση. **Μέγεθος:** 5-40 MW
  (κυρίως από t_trend×hour era correction — το bias-driven κομμάτι του MAE είναι ~60-150
  MW στις μεσημβρινές/βραδινές ώρες, αλλά static extrapolation πέρα από το train_end
  περιορίζει το όφελος). Υπερκαλύπτει το 0.15 noise floor.
- **Πού αναμένεται να δρα:** ισχυρότερα summer (εκεί ζει το era shift)· q1/octnov μικρό ή
  ουδέτερο· ΔΕΝ πρέπει να ΒΛΑΨΕΙ κανένα window >0.15 (αλλιώς red flag overfit).
- **Lagscan (T2):** Ν/Α — δεν υπάρχει εξωτερική χρονοσειρά· τα features είναι συναρτήσεις
  index (η «συσχέτιση με y» είναι η εποχικότητα, εξ ορισμού νόμιμη).
- **Hour-profile (T3):** Ν/Α (σταθερά ανά ημέρα/γραμμικά).
- **Αναμενόμενο FI rank:** t_trend/doy κάτω από lags, τάξης calendar features.
- **T8 κόκκινη γραμμή:** αν base+seas βελτιώσει το summer static >60 MW → ύποπτο
  (calendar-μόνο features δεν δικαιολογούν τέτοιο άλμα) → triaging-suspicious-results.

## 3. Acceptance tests

| Test | Κριτήριο PASS | Στάδιο |
|---|---|---|
| T1 gate timing | Τετριμμένο (ημερολόγιο — γραπτώς εδώ) ✅ | 2 |
| T2/T3 | Ν/Α (index-only, αιτιολόγηση §2) ✅ | 2 |
| T4 non-VOID | #features 17→20 (base→base+seas) στο πρώτο log | 4 |
| T5 poisoning | preflight --poison PASS μετά το wiring | 3 |
| T6 control | base spec runs bit-identical (ομάδα opt-in): summer static = 372.0053 | 3 |
| T7 §2 | Δ<0 ίδιο πρόσημο ≥2 ανεξάρτητα windows (probe: summer πρώτα, μετά q1/octnov αν ζει) | 5 |
| T8 | Δ όχι >> pre-registered (>60 MW = ύποπτο)· FI rank εύλογο | 5-6 |

Unit tests (TDD πριν το wiring): classify {doy_sin,doy_cos,t_trend} → seas· seas ∉ default,
∈ all· τιμές: doy_sin(1 Ιαν)≈0/doy_cos≈1, doy_sin(~1 Απρ)≈1· t_trend γνησίως αύξον,
t_trend(2015-01-01)=0· idempotent (υπάρχουσες στήλες δεν ξαναγράφονται).

## 4. Batch matrix (μικρά probes — οδηγία χρήστη «μικρά runs»)

Static, lgbm rec, seed 42, g12, task=load, `runs/feat_seas/`:
1. control: summer base (αναμένεται 372.0053 ΤΑΥΤΟΣΗΜΟ) ~2'
2. summer `base+seas` (T4: #features 17→20· πρώτο Δ) ~2'
3. summer `dense,meteo_vintage` (νέο static baseline για το ρεαλιστικό arm) ~3'
4. summer `dense,meteo_vintage,seas` (προσθέτει πάνω στο καλύτερο;) ~3'
Σύνολο ~10-12'. Αν Δ<−0.15 και στα δύο ζεύγη → επόμενο στάδιο (με OK χρήστη):
q1+octnov spot-checks και μετά weekly confirm.

## 5. Γνωστοί κίνδυνοι

- **t_trend extrapolation**: στο eval πέρα από train_end τα δέντρα δίνουν το leaf της
  νεότερης εποχής — επιθυμητό εδώ, αλλά σε ΜΑΚΡΙΝΑ μελλοντικά windows παγώνει (όχι
  συνεχής προσαρμογή) — δηλώνεται ως όριο, όχι bug.
- **Overfit σε 1 window**: γι' αυτό T7 απαιτεί ≥2 windows πριν από οτιδήποτε ΔΕΚΤΟ.
- **Συνύπαρξη με retrain cadence**: το weekly retrain ήδη «βλέπει» φρέσκα δεδομένα —
  το seas όφελος μπορεί να είναι μικρότερο στο weekly (θα φανεί στο confirm στάδιο).

## 6. Audit evidence

- T1: γραπτά εδώ (§1-§2) — index-only, κανένα εξωτερικό timing. Βήμα-4 πληρότητας: Ν/Α
  (δεν υπάρχουν στήλες parquet — χτίζονται on-the-fly και στα δύο tasks).
- Supporting analysis: scratchpad deep-dive 2026-07-12 (bias/ώρα, duck ratio ανά έτος,
  corr πίνακες) — αποτυπωμένο σε ABLATION §9-συνοδευτικά και στο chat log.
- Verdict σταδίου 2: ΠΡΟΧΩΡΑ (καμία εξωτερική πηγή — μηδενικό ρίσκο leakage by design).

## 7. Probe αποτελέσματα (Στάδιο 4, 2026-07-12 — static, lgbm rec, seed 42, g12)

| window | base | base+seas | Δ | σχόλιο |
|---|---|---|---|---|
| q1 | 254.99 | 219.11 | **−35.9** | ⚠️ ισχυρότερο από pre-registered «μικρό/ουδέτερο» — βλ. T8 |
| summer | 372.01 | 336.08 | **−35.9** | εντός pre-registered 5-40 |
| octnov | 151.04 | 133.97 | **−17.1** | ≈ επίπεδο weekly densemv με ΜΟΝΟ static! |
| summer (densemv± seas) | 286.78 | 271.10 | **−15.7** | προσθέτει και στο ισχυρότερο arm |

- T4 ✓ (#features 17→20) · T5 poison PASS (`logs/seas_wiring_preflight_poison.log`) ·
  T6 control 372.0053 ΤΑΥΤΟΣΗΜΟ · pytest 97 PASS (+3 seas unit tests).
- **T8 honest note:** το q1 −35.9 ξεπερνά το pre-registered — καμία δυνατότητα leak
  (index-only)· ερμηνεία: το era shift (t_trend) δρα ΟΛΟ τον χρόνο (BTM PV αυξάνεται και
  χειμώνα), το pre-registration ήταν στενά «καλοκαιρινό». Κάτω από κόκκινη γραμμή (60).
- **Εκκρεμούν πριν από ΔΕΚΤΟ (Στάδιο 5-6):** weekly confirm (mirror G7 arms) · XGB ·
  FI rank · ανθρώπινη αποδοχή. ΤΙΠΟΤΑ δεν γράφεται ΔΕΚΤΟ από αυτό το probe.
