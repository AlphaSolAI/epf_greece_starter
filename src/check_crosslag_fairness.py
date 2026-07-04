"""
check_crosslag_fairness.py — Poisoning self-test για τα CROSSLAG actuals
(gen_solar/gen_wind/residual_load/load lags), SYSTEM_DESIGN §4.10.2 / VALIDITY_CHECKLIST Β1.

Συμπληρώνει το check_openloop_fairness.py (που ελέγχει μόνο y-lags/y-rolls):
εδώ ελέγχουμε ότι το AEL (feature_availability.freeze_crosslags_for_gate) όντως
εμποδίζει τη διαρροή actual παραγωγής/φορτίου που δεν θα ήταν ακόμα δημοσιευμένη
στο cutoff_F της οικογένειάς τους (§4.8) — σε recursive ΚΑΙ direct.

⚠️ Σχεδιαστικό detail (ΣΗΜΑΝΤΙΚΟ): τα RAW crosslag lag columns εξυπηρετούν ΔΙΠΛΟ
ρόλο — (α) ως feature για τη γραμμή τους, (β) ως ΠΗΓΗ για το frozen-lookup ΚΑΠΟΙΑΣ
ΑΛΛΗΣ (συνήθως προηγούμενης) ημέρας/anchor (βλ. build_frozen_lookup: ανακατασκευάζει
gen_solar[t] από gen_solar_lag1[t+1]). Αν δηλητηριάσουμε «ό,τι είναι unsafe για τη
ΔΙΚΗ ΤΟΥ ημέρα» σε ΟΛΟΚΛΗΡΟ το ιστορικό (κάθε ημέρα ως ανεξάρτητο anchor), καταστρέφουμε
ΚΑΙ τιμές που χρειάζεται νόμιμα η ΕΠΟΜΕΝΗ ημέρα ως cutoff boundary — false positive,
όχι πραγματικό leak. Γι' αυτό εδώ δηλητηριάζουμε σε σχέση με ΕΝΑ σταθερό cutoff_F
(αυτό του single eval block) και κρατάμε το training strictly πριν από αυτό
(train_end < cutoff_F) — καμία διφορούμενη κελί.

Test A (leak test, recursive/direct eval): δηλητηριάζει ΚΑΘΕ crosslag κελί με
source timestamp (t-lag) > cutoff_F του ΕΝΟΣ eval block, μετά τρέχει το ΙΔΙΟ
config στο καθαρό vs το δηλητηριασμένο df. Αν το AEL δουλεύει, οι προβλέψεις
είναι ΑΜΕΤΑΒΛΗΤΕΣ.

Test B (control): δηλητηριάζει ΝΟΜΙΜΑ (source timestamp ≤ cutoff_F) crosslag
κελιά, ΜΟΝΟ μέσα στο eval window — οι προβλέψεις ΠΡΕΠΕΙ να αλλάξουν αισθητά
(αλλιώς το Test A δεν αποδεικνύει τίποτα, π.χ. genlags εκτός feature spec).

Test C (training-row freeze, direct assertion): επαληθεύει ότι
freeze_crosslags_for_gate πάνω σε πραγματικά training rows αντικαθιστά τα
unsafe κελιά με τη ΣΩΣΤΗ τιμή (ίδια με ανεξάρτητο υπολογισμό base_actual@cutoff)
και αφήνει τα safe κελιά ανέγγιχτα — χωρίς poisoning/model training (γρήγορο,
αποφεύγει το παραπάνω dual-duty πρόβλημα).

Χρήση:
  conda run -n epf --no-capture-output python -X utf8 -m src.check_crosslag_fairness \
      --task price --algo lgbm --strategy recursive --market dam --gate strict
Exit code: 0 = PASS (A leak-free, B αισθητό, C σωστό), 1 = FAIL.
"""
from __future__ import annotations

import argparse
import sys
import traceback

import numpy as np
import pandas as pd

from .feature_availability import (
    GateSpec,
    build_frozen_lookup,
    detect_crosslag_cols,
    freeze_crosslags_for_gate,
    parse_feature_spec,
    select_features,
)
from .master_forecast import run_forecast
from .split_utils import load_processed

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# forward: stride=168 (αντί του production 24) ώστε το test window να είναι ΕΝΑ
# block με ΕΝΑ anchor → ένα μοναδικό cutoff_F για όλο το poisoning (αλλιώς κάθε
# επικαλυπτόμενο anchor έχει δικό του cutoff και το single-threshold test δεν
# ορίζεται). Η per-block enforcement λογική που ελέγχεται είναι η ίδια.
MARKET_HS = {"dam": (24, 24), "idm": (6, 3), "forward": (168, 168)}


def _poison_crosslags(df: pd.DataFrame, feature_cols, wild: float, cutoff_F: pd.Timestamp,
                      side: str, row_scope=None):
    """
    side='after'  → δηλητηριάζει κελιά με source timestamp (t-lag) > cutoff_F
                    (έπρεπε να αγνοούνται — leak test).
    side='before' → δηλητηριάζει κελιά με source timestamp (t-lag) <= cutoff_F
                    (νόμιμα διαθέσιμα — control test).
    row_scope: προαιρετική boolean μάσκα (ίδιου μήκους με df) που περιορίζει ΠΟΙΕΣ
    γραμμές επιτρέπεται να δηλητηριαστούν (π.χ. μόνο το eval window, για να μην
    πειραχτεί το training όταν κάνουμε 'before' control).
    """
    out = df.copy()
    crosslag_cols = detect_crosslag_cols(feature_cols)
    scope = np.ones(len(out), dtype=bool) if row_scope is None else np.asarray(row_scope)
    cutoff_np = np.datetime64(cutoff_F)
    n_poisoned = 0
    for base, colmap in crosslag_cols.items():
        for col, lag in colmap.items():
            if col not in out.columns:
                continue
            src_time = (out.index - pd.Timedelta(hours=int(lag))).values
            cond = (src_time > cutoff_np) if side == "after" else (src_time <= cutoff_np)
            mask = cond & scope
            if mask.any():
                out.loc[mask, col] = wild
                n_poisoned += int(mask.sum())
    return out, n_poisoned


def _check_training_freeze(df: pd.DataFrame, feature_cols, gate: GateSpec,
                           train_start: pd.Timestamp, train_end: pd.Timestamp) -> bool:
    """
    Test C: επαληθεύει ΤΟ ΙΔΙΟ transform που καλεί το master_forecast._fit_at πριν
    το make_xy — χωρίς poisoning/training, μόνο σύγκριση τιμών.
    """
    dtr = df.loc[train_start:train_end]
    crosslag_cols = detect_crosslag_cols(feature_cols)
    frozen_lookup = build_frozen_lookup(df, crosslag_cols)
    cutoffs = gate.crosslag_cutoff_index(dtr.index)

    dtr_frozen = freeze_crosslags_for_gate(dtr, feature_cols, gate, df_full=df, mode="freeze")

    n_checked_unsafe = 0
    n_checked_safe = 0
    bad = []
    for base, colmap in crosslag_cols.items():
        lut = frozen_lookup.get(base)
        for col, lag in colmap.items():
            if col not in dtr.columns:
                continue
            src_time = (dtr.index - pd.Timedelta(hours=int(lag))).values
            unsafe = src_time > cutoffs.values
            # δείγμα ελέγχου: τα πρώτα 3 unsafe + πρώτα 3 safe (ανά στήλη, αρκετό
            # για assertion — δεν χρειάζεται να ελεγχθούν ΟΛΑ).
            unsafe_pos = np.flatnonzero(unsafe)[:3]
            safe_pos = np.flatnonzero(~unsafe)[:3]
            for p in unsafe_pos:
                n_checked_unsafe += 1
                got = dtr_frozen[col].iloc[p]
                exp = lut.reindex([cutoffs[p]]).iloc[0] if lut is not None else np.nan
                if not (pd.isna(got) and pd.isna(exp)) and abs(float(got) - float(exp)) > 1e-9:
                    bad.append((col, dtr.index[p], "unsafe", got, exp))
            for p in safe_pos:
                n_checked_safe += 1
                got = dtr_frozen[col].iloc[p]
                exp = dtr[col].iloc[p]
                if not (pd.isna(got) and pd.isna(exp)) and abs(float(got) - float(exp)) > 1e-9:
                    bad.append((col, dtr.index[p], "safe", got, exp))

    print(f"[RESULT] C_TRAINING_FREEZE: unsafe_checked={n_checked_unsafe} "
          f"safe_checked={n_checked_safe} mismatches={len(bad)}", flush=True)
    for b in bad[:10]:
        print(f"    MISMATCH col={b[0]} t={b[1]} kind={b[2]} got={b[3]} expected={b[4]}", flush=True)
    return len(bad) == 0 and n_checked_unsafe > 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["price", "load"], default="price")
    p.add_argument("--algo", choices=["lgbm", "xgb"], default="lgbm")
    p.add_argument("--strategy", choices=["recursive", "direct"], default="recursive")
    p.add_argument("--market", choices=["dam", "idm", "forward"], default="dam")
    p.add_argument("--gate", choices=["strict", "academic"], default="strict")
    p.add_argument("--features", default="default")
    p.add_argument("--train_start", default="2025-09-01 00:00")
    p.add_argument("--train_end", default="2025-11-29 23:00")
    p.add_argument("--test_start", default="2025-12-01 00:00")
    p.add_argument("--test_end", default="2025-12-01 23:00")
    p.add_argument("--n_estimators", type=int, default=200)
    p.add_argument("--wild", type=float, default=999999.0)
    p.add_argument("--tol", type=float, default=1e-6)
    args = p.parse_args()

    print("=== check_crosslag_fairness START ===", flush=True)
    print(f"[INFO] task={args.task} algo={args.algo} strategy={args.strategy} "
          f"market={args.market} gate={args.gate} features={args.features}", flush=True)

    gate = GateSpec(task=args.task, gate=args.gate, market=args.market)
    df = load_processed("hourly", task=args.task).dropna(subset=["y"]).sort_index()
    groups = parse_feature_spec(args.features)
    all_cols = [c for c in df.columns if c != "y"]
    feature_cols = select_features(all_cols, groups)
    feature_cols = [c for c in feature_cols if np.issubdtype(df[c].dtype, np.number)]

    crosslag_cols = detect_crosslag_cols(feature_cols)
    if not crosslag_cols:
        print(f"[ERROR] Καμία crosslag στήλη στα επιλεγμένα features={groups} — "
              f"το test δεν ελέγχει τίποτα. Χρησιμοποίησε --features με genlags/loadlags "
              f"(π.χ. 'default').", flush=True)
        sys.exit(1)
    n_cols = sum(len(m) for m in crosslag_cols.values())
    print(f"[INFO] crosslag cols υπό έλεγχο: {n_cols} από families={list(crosslag_cols.keys())}",
          flush=True)

    train_start = pd.Timestamp(args.train_start)
    train_end = pd.Timestamp(args.train_end)
    test_start = pd.Timestamp(args.test_start)
    test_end = pd.Timestamp(args.test_end)

    # ΕΝΑ σταθερό cutoff_F για το single eval block — anchor-based, ακριβώς όπως
    # το χρησιμοποιεί το production loop (run_forecast → crosslag_cutoff_for_anchor).
    cutoff_F = gate.crosslag_cutoff_for_anchor(test_start)
    print(f"[INFO] eval block=[{test_start} .. {test_end}]  crosslag cutoff_F={cutoff_F}", flush=True)
    if train_end >= cutoff_F:
        print(f"[ERROR] train_end={train_end} >= cutoff_F={cutoff_F} — το poisoning θα άγγιζε "
              f"training δεδομένα (dual-duty ασάφεια). Δώσε νωρίτερο --train_end.", flush=True)
        sys.exit(1)

    horizon, stride = MARKET_HS[args.market]
    common = dict(
        algo=args.algo, task=args.task, strategy=args.strategy, gate=gate,
        horizon=horizon, stride=stride, retrain="static", feature_cols=feature_cols,
        train_start=train_start, train_end=train_end,
        test_start=test_start, test_end=test_end,
        verbose=False, n_estimators=args.n_estimators, crosslag_mode="freeze",
    )

    print("[INFO] Reference run (καθαρό df)...", flush=True)
    idx_ref, y_true_ref, y_pred_ref = run_forecast(df=df, **common)
    print(f"[INFO] scored n={len(idx_ref)}  window=[{idx_ref.min()} .. {idx_ref.max()}]", flush=True)

    print("[INFO] Test A: δηλητηρίαση UNSAFE crosslag κελιών (source > cutoff_F)...", flush=True)
    df_a, n_a = _poison_crosslags(df, feature_cols, wild=args.wild, cutoff_F=cutoff_F, side="after")
    idx_a, _, y_pred_a = run_forecast(df=df_a, **common)
    if len(idx_a) != len(idx_ref) or not idx_a.equals(idx_ref):
        print("[ERROR] scored index άλλαξε μεταξύ clean/poisoned run — δεν συγκρίνεται.", flush=True)
        sys.exit(1)
    diff_a = np.abs(y_pred_ref - y_pred_a)
    mean_a, max_a = float(np.nanmean(diff_a)), float(np.nanmax(diff_a))
    print(f"[RESULT] A_POISON_UNSAFE_CROSSLAGS: n_poisoned_cells={n_a} "
          f"mean_abs_diff={mean_a:.6f} max_abs_diff={max_a:.6f}", flush=True)

    print("[INFO] Test B (control): δηλητηρίαση SAFE crosslag κελιών, ΜΟΝΟ στο eval window...",
          flush=True)
    # ΠΡΟΣΟΧΗ: το direct διαβάζει ΜΙΑ γραμμή ΣΤΟ cutoff (πριν το test_start) — όχι
    # μόνο στις scored ώρες. Το scope πρέπει να καλύπτει [cutoff, test_end] ώστε να
    # αγγίζει ό,τι ΠΡΑΓΜΑΤΙΚΑ διαβάζει η κάθε στρατηγική (recursive: roll_idx μέσα
    # στο [test_start,test_end]· direct: η ίδια η γραμμή cutoff).
    y_cutoff = gate.cutoff_for_block(test_start)
    eval_scope = (df.index >= y_cutoff) & (df.index <= test_end)
    df_b, n_b = _poison_crosslags(df, feature_cols, wild=args.wild, cutoff_F=cutoff_F,
                                  side="before", row_scope=eval_scope)
    idx_b, _, y_pred_b = run_forecast(df=df_b, **common)
    diff_b = np.abs(y_pred_ref - y_pred_b) if idx_b.equals(idx_ref) else None
    if diff_b is not None:
        mean_b, max_b = float(np.nanmean(diff_b)), float(np.nanmax(diff_b))
        print(f"[RESULT] B_POISON_SAFE_CROSSLAGS (control): n_poisoned_cells={n_b} "
              f"mean_abs_diff={mean_b:.6f} max_abs_diff={max_b:.6f}", flush=True)
    else:
        print("[WARN] scored index άλλαξε στο Test B — παραλείπεται η σύγκριση.", flush=True)
        mean_b = 0.0

    print("[INFO] Test C: direct έλεγχος training-row freeze (χωρίς poisoning/training)...",
          flush=True)
    c_ok = _check_training_freeze(df, feature_cols, gate, train_start, train_end)

    print("\n=== INTERPRETATION ===", flush=True)
    print("A (unsafe): ~0 αναμενόμενο => AEL σωστά αγνοεί ό,τι δεν ήταν ακόμα δημοσιευμένο (PASS).",
          flush=True)
    print("            μεγάλο       => LEAK: crosslag actuals μετά το cutoff_F διαρρέουν (FAIL).",
          flush=True)
    print("B (safe, control): μεγάλο αναμενόμενο => το μοντέλο όντως χρησιμοποιεί αυτά τα "
          "features (νομιμοποιεί το Test A). Αν ~0 ΚΙ ΕΔΩ, έλεγξε feature spec.", flush=True)
    print("C (training freeze): PASS => το ίδιο transform που τρέχει πριν το training είναι "
          "μαθηματικά σωστό στα πραγματικά δεδομένα.", flush=True)

    leak_free = max_a < args.tol
    control_meaningful = mean_b >= args.tol
    ok = leak_free and control_meaningful and c_ok
    print(f"\n=== {'PASS' if ok else 'FAIL'} (leak_free={leak_free}, "
          f"control_meaningful={control_meaningful}, training_freeze_ok={c_ok}) ===", flush=True)
    print("=== check_crosslag_fairness END ===", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        print("❌ Exception occurred:\n", flush=True)
        traceback.print_exc()
        raise
