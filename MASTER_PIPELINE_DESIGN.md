# MASTER PIPELINE — Σχεδιασμός AI Agent για Energy Trading (Ελλάδα)

> Τιμή (EPF) & Φορτίο (STLF) · 3 αγορές: Day-Ahead (DAM) / Intraday (IDM) / Forward (έως 1 εβδομάδα)
> Στόχος: σωστή διαθεσιμότητα πληροφορίας (χωρίς leakage), με όσο περισσότερη νόμιμη πληροφορία γίνεται, ανά αγορά & ανά στρατηγική.
>
> **Status update 2026-07-04**: το θεωρητικό πλαίσιο εδώ (§1-4) παραμένει ισχύον αυτούσιο.
> Πρακτικές εξελίξεις μετά τη συγγραφή: (α) η ομάδα `crosslags` (§2) έσπασε σε `genlags`/
> `loadlags`/`other` λεπτόκοκκα (παλιό όνομα λειτουργεί ως umbrella, καμία αλλαγή στη λογική
> gate) — βλ. `src/feature_availability.py`· (β) το LEAR (§6) ενσωματώθηκε πλήρως στο
> masterscript ως `--algo lear` (leakage-free, GateSpec-aware), όχι πια ξεχωριστό oracle script·
> (γ) **νέο μόνιμο pre-flight πρωτόκολλο** για κάθε νέα πηγή (πότε δημοσιεύεται vs gate +
> cross-correlation lag-scan + hour-profile) μετά το xborder same-day leakage incident —
> case study που επιβεβαίωσε στην πράξη όλη τη φιλοσοφία αυτού του doc: οι BG/IT-SUD same-day
> τιμές βγαίνουν από το ΙΔΙΟ SDAC auction με το target (δημοσίευση ~13:00 D-1 > gate 12:00) και
> έδιναν ψεύτικο −0.72 «όφελος». Βλ. `ABLATION_PLAN.md §1, §5.7`.
> Πλήρη εμπειρικά ευρήματα (ποια ομάδα αξίζει, ποιο μοντέλο/στρατηγική κερδίζει):
> `ABLATION_PLAN.md §5` — headline: LGBM recursive weekly `default` = 15.17 €/MWh (Q1 2026).

---

## 0. Ετυμηγορία εγκυρότητας — τηρούσαν τα παλιά μοντέλα το gate closure;

Απάντηση στην ανησυχία: **εν μέρει.** Ανάλυση ανά στρατηγική (τι πληροφορία χρησιμοποιεί κάθε μία στην πράξη, βάσει `data.py` + `recursive_openloop.py` + `eval_*`):

| Στρατηγική | Τι lags βλέπει | DAM-valid; | Σχόλιο |
|---|---|---|---|
| **CL (teacher-forced, `eval.py`)** | `y_lag1=actual(t-1)` για **ΚΑΘΕ** ώρα t | ❌ **ΟΧΙ** | Χρησιμοποιεί actual τιμή της ίδιας ημέρας παράδοσης (π.χ. actual 11:00 για να προβλέψει 12:00 της D). Αδύνατο σε πραγματικό DAM. Είναι **oracle / άνω φράγμα** — τα νούμερα (price 9.9€, load 73MW Dec) **δεν είναι tradeable**. |
| **OL recursive 24h-chunk (`eval_openloop_monthly.py`)** | anchor=actual μέχρι 23:00 D-1, μετά **recursive** εντός D | ✅ **ΝΑΙ για PRICE** / ⚠️ οριακά αισιόδοξο για LOAD | Για την τιμή: σωστό (όλες οι τιμές DAM της D-1 γνωστές). Για το φορτίο: το anchor στο 23:00 D-1 υποθέτει γνωστό βραδινό φορτίο D-1 (~12h leak). Νούμερα (price ~14€, load ~80MW Dec) = **ρεαλιστικά για την τιμή**. |
| **Direct / "MIMO" H=24** | origin=t0, προβλέπει y(t0+1..t0+24) απευθείας | ✅ **ΝΑΙ** | Ποτέ δεν αγγίζει actual εντός ορίζοντα. Καθαρό. (Αλλά: είναι **Direct**, όχι true MIMO — βλ. §4.) |
| **Walk-forward monthly-retrain TF** | teacher-forced (actual lags) | ❌ **ΟΧΙ** | Το εντυπωσιακό 10.5€ είναι oracle. Η **Rec** εκδοχή (18.5€) είναι η tradeable. |
| **Walk-forward Rec-SS-DO** | recursive + scheduled sampling | ✅ **ΝΑΙ** | Ρεαλιστικό. |

**Συμπέρασμα:** Τα «headline» νούμερα CL/TF είναι oracle upper bounds, όχι πραγματικές αποδόσεις αγοράς. Τα OL/Direct/Rec είναι τα σωστά για trading. Το masterscript κάνει αυτόν τον διαχωρισμό **ρητό** και **αδύνατο να παραβιαστεί κατά λάθος**.

---

## 1. Το ρολόι — τι είναι γνωστό στο gate closure

Κάθε πρόβλεψη ορίζεται από ένα **decision time `t0`** (στιγμή απόφασης / information cutoff). Ό,τι έχει *δημοσιευτεί* μέχρι `t0` επιτρέπεται· ό,τι υλοποιείται μετά, όχι.

Η λεπτότητα: **τι** ξέρεις στο `t0` διαφέρει ανά τύπο μεταβλητής.

| Μεταβλητή | Πότε γίνεται γνωστή | Στο DAM gate (D-1 12:00) ξέρεις… |
|---|---|---|
| Τιμή DAM | Auction αποτέλεσμα, δημοσίευση **προηγούμενη** ημέρα | **όλη** την D-1 (τελευταία γνωστή: 23:00 D-1) |
| Actual φορτίο | Real-time, δημοσίευση ~ωριαία | μέχρι ~**11:00 της D-1** |
| Actual παραγωγή (solar/wind/…) | Real-time | μέχρι ~**11:00 της D-1** |
| Day-ahead RES/gen forecast (ΑΔΜΗΕ/ENTSO-E) | Δημοσίευση D-1 πρωί για όλη την D | **όλο τον ορίζοντα** ✓ |
| Day-ahead load forecast | Δημοσίευση D-1 για όλη την D | **όλο τον ορίζοντα** ✓ |
| Weather forecast | Διαθέσιμο συνεχώς για μέλλον | **όλο τον ορίζοντα** ✓ (proxy: reanalysis) |
| Gas / CO2 futures | Ημερήσιο settlement | D-1 settlement (front-month) ✓ |

**Ασυμμετρία PRICE vs LOAD** (το κλειδί):
- **PRICE**: anchor τιμής = 23:00 D-1 → σωστό.
- **LOAD**: anchor φορτίου = 11:00 D-1 → το κενό 12:00 D-1 … 23:00 D καλύπτεται με recursive predictions ή/και με το επίσημο day-ahead load forecast.

Αυτό ελέγχεται από το `--gate`:
- `strict` (default): PRICE anchor 23:00 D-1 · LOAD anchor 11:00 D-1 (ρεαλιστικό, tradeable).
- `academic` (Lago 2021): και τα δύο anchor στο τέλος της D-1 (συγκρίσιμο με open-access EPF benchmarks).

---

## 2. Μητρώο διαθεσιμότητας feature (availability matrix)

Δεδομένου decision time `t0`, ώρας-στόχου `h`, και lag `k`: η τιμή `y(h−k)` είναι **καθαρό actual** ⟺ `h − k ≤ t0_cutoff(task,gate)`. Αλλιώς προέρχεται από recursion (predicted) ή απαγορεύεται.

| Ομάδα feature | Στήλες | Κανόνας διαθεσιμότητας | Ablation flag |
|---|---|---|---|
| Calendar | hour, dow, is_holiday, sin/cos | Πάντα (a priori) | (πάντα ON) |
| Target lags | `y_lag_k` | καθαρό αν `h−k ≤ t0_cutoff`· αλλιώς recursive | `lags` (core) |
| Dense intraday lags | `y_lag4..y_lag23` | ως άνω | `dense` |
| Rolling | `y_roll24`, `y_roll168` | recompute από running series | `roll` |
| Day-ahead RES/gen fc | `solar_fc_dayahead`, `wind_onshore_fc_dayahead`, `gen_fc_dayahead` | όλος ο ορίζοντας ✓ | `forecast` |
| Day-ahead load fc | `load_fc` (price task) | όλος ο ορίζοντας ✓ | `forecast` |
| Weather | `w_*` (42 στήλες, 6 πόλεις + GR mean) | όλος ο ορίζοντας ✓ (proxy) | `meteo` |
| Fuel/carbon | `gas_price/co2_price` (+lags) | D-1 settlement (lag≥1 strict) | `fuel` |
| Cross lags | `load_lag*`, `residual_load_lag*`, `gen_*_lag*` | καθαρό αν `h−k ≤ t0_cutoff` | `crosslags` |

**Καθαρή απαρίθμηση**: το masterscript, στο `strict` gate, για κάθε (h,k) που παραβιάζει τον κανόνα είτε (α) αντικαθιστά με recursive prediction, είτε (β) στο Direct αγνοεί εντελώς within-horizon actuals. Ποτέ actual εντός ορίζοντα.

---

## 3. Αγορά → (delay, horizon, stride)

Παραμετροποίηση με τρεις αριθμούς:
- **delay `d`**: ώρες από `t0` έως την 1η προβλεπόμενη ώρα (0 = επόμενη ώρα · 12 = DAM).
- **horizon `H`**: πόσες ώρες προβλέπονται **χωρίς** νέο actual.
- **stride `s`**: κάθε πόσες ώρες εκδίδεται νέα πρόβλεψη (re-anchoring).

| Αγορά | preset | delay `d` | horizon `H` | stride `s` | Λογική |
|---|---|---|---|---|---|
| **Day-Ahead (DAM)** | `--market dam` | 12 | 24 | 24 | Στις D-1 12:00 προβλέπω τις 24h της D. Re-anchor κάθε μέρα. |
| **Intraday (IDM)** | `--market idm` | 1 (ή 3) | 4–8 | 1 (ή 3) | Κοντινός ορίζοντας, συχνό re-anchoring με φρέσκα actuals. |
| **Forward (≤1 εβδ.)** | `--market forward` | 12 | 168 | 24 (ή 168) | Προβλέπω 7 ημέρες μπροστά· re-anchor ημερησίως ή εβδομαδιαίως. |

`--market custom --delay D --horizon H --stride S` για ό,τι άλλο.

**Παρατήρηση για DAM**: με `d=12, H=24, s=24` και anchor στην D-1 12:00, ο πρακτικός ορίζοντας φτάνει τις 35h (12:00 D-1 → 23:00 D). Βαθμολογούμε **μόνο** τις 24h της D.

---

## 4. Στρατηγικές πρόβλεψης (όλες leakage-free)

| Στρατηγική | `--strategy` | Πώς | Μοντέλα |
|---|---|---|---|
| **Recursive (delay-aware)** | `recursive` | anchor στο `t0`, iterate· τα `y_lag*`/`y_roll*` εντός ορίζοντα από running predictions· καλύπτει το κενό `delay` | xgb, lgbm, mlp, lstm(1-step) |
| **Direct multi-horizon** | `direct` | 1 μοντέλο ανά offset `o=1..H` από features@`t0` | xgb, lgbm, mlp |
| **True MIMO (seq2seq)** | `seq2seq` | **1** μοντέλο → διάνυσμα H εξόδων ταυτόχρονα (encoder-decoder LSTM) | lstm |
| **Recursive + Scheduled Sampling** | `recursive --ss` | recursive train με μίξη actual/predicted lags (robustness) | xgb, lgbm, mlp |
| Teacher-forced (διαγνωστικό) | `tf` | 1-step με actual lags — **ΜΟΝΟ** ως oracle upper bound, με προειδοποίηση | όλα |

> **«Ψεύτικο MIMO»**: το υπάρχον `MultiOutputRegressor` = H ανεξάρτητα μοντέλα = **Direct**, όχι true MIMO. Το πραγματικό MIMO (ένα μοντέλο, διανυσματική έξοδος, κοινή αναπαράσταση) υλοποιείται ως **seq2seq LSTM** (`--strategy seq2seq`).

---

## 5. Προδιαγραφή masterscript (`src/master_forecast.py`)

Ενιαία μηχανή — ίδια λογική διαθεσιμότητας για ΟΛΟΥΣ τους αλγόριθμους (εγγυάται fair comparison, καμία per-algo διαρροή).

```
python -m src.master_forecast \
  --algo {lgbm,xgb,mlp,lstm} \
  --task {price,load} \
  --market {dam,idm,forward,custom} [--delay D --horizon H --stride S] \
  --strategy {recursive,direct,seq2seq,tf} [--ss] \
  --gate {strict,academic} \
  --retrain {static,monthly,weekly} \
  --train_start ... --train_end ... --test_start ... --test_end ... \
  --features "lags,calendar,forecast,meteo,fuel,dense,roll,crosslags"  # ablation: ό,τι ΣΥΜΠΕΡΙΛΑΜΒΑΝΕΤΑΙ \
  --out_json <path> [--quiet]
```

Παράμετροι που ζήτησες:
- **(Α) Ορίζοντας** → `--horizon` (+ `--market` presets).
- **(Β) Delay** → `--delay` (0=επόμενη, 12=DAM).
- **(Γ) Retrain** → `--retrain {static,monthly,weekly}`.
- **(Δ) Ablation** → `--features` (include-list) + `--ss`.
- **(Ε) Χρονικό διάστημα** → `--train_*/--test_*`.

Έξοδος: JSON συμβατό με το dashboard schema (`strategy/task/dates/actual/series/metrics`) + `market/delay/horizon/gate/retrain/features` metadata για πλήρη αναπαραγωγιμότητα.

---

## 6. Αλγόριθμοι

- **LightGBM, XGBoost** — τα ισχυρότερα tree baselines (ήδη κορυφαία εδώ).
- **MLP (PyTorch)** — feedforward NN εκπρόσωπος.
- **LSTM** — δύο εκδοχές:
  - `recursive`: 1-step LSTM με autoregressive rollout.
  - `seq2seq`: encoder-decoder → **true MIMO** (το «σωστό» που έλειπε).
- **Βιβλιογραφία / ήδη διαθέσιμα**: `LEAR` (Lasso-AR, Lago 2021 — φθηνό & πολύ ισχυρό benchmark για DAM τιμή, υπάρχει `eval_lear.py`). Μελλοντικά: N-BEATS / Temporal Fusion Transformer.

---

## 7. Εκτίμηση χρόνου για Q1 2026 (Dec–Feb, ~2160h / 90 ημέρες)

Βάσεις (από προηγούμενα sessions, Windows/conda, sequential):
- LGBM/XGB train (70–98k rows): ~0.5–3 min · Direct H=24 (24 sub-models, threaded): ~2–6 min.
- MLP (PyTorch, σωστό early-stopping): ~3–10 min/train.
- LSTM (CPU, seq2seq, ~30 epochs): ~15–40 min/train.
- Eval rollout 90 ημερών: DAM=90 anchors, γρήγορο (<1–2 min tree, ~5 min LSTM).

**Ανά αλγόριθμο × task × market, χρόνος για 1 πλήρη Q1 τρέξιμο:**

| Retrain | # trainings | LGBM/XGB | MLP | LSTM |
|---|---|---|---|---|
| **(a) static** | 1 | ~3–6 min | ~5–10 min | ~20–45 min |
| **(b) monthly** | 3 | ~10–20 min | ~15–30 min | ~1–2 h |
| **(c) weekly** | ~13 | ~40–90 min | ~1–2 h | ~4–8 h |

**Συνολικό «όλα μαζί»** (4 algos × 2 tasks × 3 markets = 24 configs):
- **static**: ~4–8 ώρες (κυρίως ο LSTM).
- **monthly**: ~1.5–2.5 ημέρες μηχανοχρόνου.
- **weekly**: ~6–12 ημέρες μηχανοχρόνου (ο LSTM weekly κυριαρχεί).

**Πρακτική σύσταση**: full grid σε **static + monthly** για όλα (2–3 ημέρες), και **weekly μόνο για τους 2 νικητές** ανά task (π.χ. LGBM+XGB, ή τον καλύτερο DL) — γλιτώνει ~80% του χρόνου με ελάχιστη απώλεια πληροφορίας. Optuna: cached params (tune μία φορά, εφαρμογή σε κάθε retrain).

---

## 8. Σειρά υλοποίησης

1. `src/feature_availability.py` — καθαρή λογική availability (gate/delay/horizon) + feature-group selection (ablation). **Η καρδιά κατά του leakage.**
2. `src/master_forecast.py` — engine: rolling anchors, recursive/direct/seq2seq, retrain modes, JSON out. Tree path (lgbm/xgb) πρώτα + tests.
3. MLP + LSTM builders (PyTorch), seq2seq.
4. Leakage self-test (poisoning, όπως `check_openloop_fairness.py`) ενσωματωμένο ως `--selftest`.
5. Q1 runs: static → monthly → (weekly επιλεκτικά).
