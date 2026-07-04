"""
conformal.py — Probabilistic layer πάνω στο κλειδωμένο headline config
(LGBM recursive weekly, --features default → 15.17 €/MWh, Q1 2026, strict).

Δύο ανεξάρτητοι τρόποι (subcommands):

1) split-conformal
   Διαβάζει ένα ΕΚΤΕΤΑΜΕΝΟ per-hour point-forecast JSON (πρέπει να καλύπτει
   τουλάχιστον ~8 εβδομάδες ΠΡΙΝ το πρώτο eval window — βλ. εντολή παρακάτω).
   residual(t) = actual(t) − point_forecast(t). Για κάθε timestamp t υπολογίζει
   p10/p50/p90 = point_forecast(t) + quantile_a(residuals ίδιου hour-of-day,
   ΑΥΣΤΗΡΑ πριν το t, trailing παράθυρο 4-8 εβδομάδων). Ποτέ residuals από το
   ίδιο ή μελλοντικό test σημείο — καθαρά αιτιατό (causal) rolling calibration.

2) quantile-lgbm
   Εκπαιδεύει το ΙΔΙΟ config (lgbm, recursive, weekly retrain, --features
   default) με 3 ξεχωριστά LGBMRegressor(objective='quantile', alpha=
   0.1/0.5/0.9). Το p50 μοντέλο οδηγεί το open-loop recursive rollout (γράφει
   στο running y)· τα p10/p90 προβλέπουν στην ΙΔΙΑ γραμμή χαρακτηριστικών
   (recursive_predict_openloop(aux_models=...), src/recursive_openloop.py).

Αξιολόγηση (και τα δύο): pinball loss (μέσος όρος 3 quantiles) + empirical
coverage, σε δύο windows:
  Q1 2026    : 2025-12-01 00:00 .. 2026-02-28 23:00
  Μάρτιος 26 : 2026-03-01 00:00 .. 2026-03-19 23:00

Βλ. last.md §5 / ABLATION_PLAN.md §8.2 §Conformal-RESULTS.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .split_utils import load_processed, make_xy
from .recursive_openloop import OpenLoopConfig, recursive_predict_openloop
from .feature_availability import (
    GateSpec, describe_gate, freeze_crosslags_for_gate, parse_feature_spec, select_features,
)
from .master_forecast import (
    MARKET_PRESETS, add_dense_lags, add_engineered_features, build_and_fit, make_blocks,
)

try:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ALPHAS: Tuple[float, float, float] = (0.1, 0.5, 0.9)

EVAL_WINDOWS: Dict[str, Tuple[str, str]] = {
    "q1_2026": ("2025-12-01 00:00", "2026-02-28 23:00"),
    "march_2026": ("2026-03-01 00:00", "2026-03-19 23:00"),
}


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def evaluate_window(dates, actual, p10, p50, p90, window: Tuple[str, str]) -> dict:
    ts0, ts1 = pd.to_datetime(window[0]), pd.to_datetime(window[1])
    idx = pd.DatetimeIndex(dates)
    mask = (idx >= ts0) & (idx <= ts1)
    a = np.asarray(actual, dtype=float)[mask]
    q10 = np.asarray(p10, dtype=float)[mask]
    q50 = np.asarray(p50, dtype=float)[mask]
    q90 = np.asarray(p90, dtype=float)[mask]
    good = np.isfinite(a) & np.isfinite(q10) & np.isfinite(q50) & np.isfinite(q90)
    a, q10, q50, q90 = a[good], q10[good], q50[good], q90[good]
    n = int(len(a))
    if n == 0:
        return dict(n=0)
    pb10 = pinball_loss(a, q10, 0.1)
    pb50 = pinball_loss(a, q50, 0.5)
    pb90 = pinball_loss(a, q90, 0.9)
    return dict(
        n=n,
        pinball_p10=round(pb10, 4), pinball_p50=round(pb50, 4), pinball_p90=round(pb90, 4),
        avg_pinball=round((pb10 + pb50 + pb90) / 3.0, 4),
        coverage_80pct_nominal=round(float(np.mean((a >= q10) & (a <= q90)) * 100.0), 2),
        p10_empirical_pct=round(float(np.mean(a <= q10) * 100.0), 2),   # στόχος 10%
        p90_empirical_pct=round(float(np.mean(a <= q90) * 100.0), 2),   # στόχος 90%
        mae_p50=round(float(np.mean(np.abs(a - q50))), 4),
    )


def print_eval_table(title: str, results: Dict[str, dict]) -> None:
    print(f"\n=== {title} ===")
    hdr = f"{'window':<12} {'n':>5} {'avg_pinball':>12} {'MAE(p50)':>10} {'cov80%':>8} {'P10emp%':>9} {'P90emp%':>9}"
    print(hdr)
    for wname, r in results.items():
        if r.get("n", 0) == 0:
            print(f"{wname:<12} {'—':>5}  (καμία επικαλυπτόμενη ημερομηνία)")
            continue
        print(f"{wname:<12} {r['n']:>5} {r['avg_pinball']:>12} {r['mae_p50']:>10} "
              f"{r['coverage_80pct_nominal']:>8} {r['p10_empirical_pct']:>9} {r['p90_empirical_pct']:>9}")


# ----------------------------------------------------------------------------
# 1) split-conformal: rolling causal residual quantiles πάνω σε point forecast
# ----------------------------------------------------------------------------
def rolling_residual_quantiles(
    dates: pd.DatetimeIndex, resid: np.ndarray,
    min_days: int = 28, max_days: int = 56,
    alphas: Tuple[float, ...] = ALPHAS, min_samples: int = 10,
) -> Tuple[Dict[float, np.ndarray], np.ndarray]:
    """
    Για κάθε timestamp t: quantiles(alphas) των residuals ΙΔΙΟΥ hour-of-day,
    ΑΥΣΤΗΡΑ πριν το t, μέσα σε trailing παράθυρο έως max_days (8 εβδ.).
    Απαιτεί τουλάχιστον min_days (4 εβδ.) ιστορικού πριν αρχίσει να βγάζει
    τιμή (αλλιώς NaN — ανεπαρκές calibration set, ΠΟΤΕ από το ίδιο/μελλοντικό
    test σημείο).
    """
    idx = pd.DatetimeIndex(dates)
    resid = np.asarray(resid, dtype=float)
    n = len(idx)
    out = {a: np.full(n, np.nan) for a in alphas}
    n_calib = np.zeros(n, dtype=int)
    t0 = idx.min()

    for h in range(24):
        h_pos = np.where(idx.hour == h)[0]
        if len(h_pos) == 0:
            continue
        order = np.argsort(idx[h_pos].values)
        h_pos = h_pos[order]
        h_times = idx[h_pos]
        h_vals = resid[h_pos]
        for k in range(len(h_pos)):
            t = h_times[k]
            orig_i = h_pos[k]
            if (t - t0).days < min_days:
                continue  # ανεπαρκές ιστορικό — άφησέ το NaN
            window_start = t - pd.Timedelta(days=max_days)
            cand = h_vals[:k][(h_times[:k] >= window_start)]
            cand = cand[np.isfinite(cand)]
            n_calib[orig_i] = len(cand)
            if len(cand) >= min_samples:
                qs = np.quantile(cand, list(alphas))
                for a, q in zip(alphas, qs):
                    out[a][orig_i] = q
    return out, n_calib


def run_split_conformal(forecast_json: str, out_json: Optional[str],
                         min_weeks: int = 4, max_weeks: int = 8) -> dict:
    d = json.loads(Path(forecast_json).read_text(encoding="utf-8"))
    dates = pd.DatetimeIndex(pd.to_datetime(d["dates"]))
    actual = np.asarray(d["actual"], dtype=float)
    series = d["series"]
    if len(series) != 1:
        raise SystemExit(f"Περιμένω ΑΚΡΙΒΩΣ 1 μοντέλο στο forecast JSON (point forecast), βρέθηκαν: {list(series)}")
    model_name = next(iter(series))
    forecast = np.asarray(series[model_name], dtype=float)

    resid = actual - forecast
    qres, n_calib = rolling_residual_quantiles(
        dates, resid, min_days=min_weeks * 7, max_days=max_weeks * 7, alphas=ALPHAS,
    )
    p10 = forecast + qres[0.1]
    p50 = forecast + qres[0.5]
    p90 = forecast + qres[0.9]

    results = {w: evaluate_window(dates, actual, p10, p50, p90, win) for w, win in EVAL_WINDOWS.items()}
    print_eval_table(f"SPLIT-CONFORMAL ({model_name}, calib {min_weeks}-{max_weeks} εβδ., causal, hour-of-day)", results)

    out = dict(
        method="split_conformal", base_model=model_name,
        calib_min_weeks=min_weeks, calib_max_weeks=max_weeks,
        source_json=str(forecast_json), results=results,
        dates=[t.isoformat() for t in dates],
        actual=[None if not np.isfinite(v) else round(float(v), 4) for v in actual],
        p10=[None if not np.isfinite(v) else round(float(v), 4) for v in p10],
        p50=[None if not np.isfinite(v) else round(float(v), 4) for v in p50],
        p90=[None if not np.isfinite(v) else round(float(v), 4) for v in p90],
        n_calib=[int(v) for v in n_calib],
    )
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        print(f"\n💾 saved: {out_json}")
    return out


# ----------------------------------------------------------------------------
# 2) quantile-lgbm: ίδιο config, objective=quantile ανά alpha
# ----------------------------------------------------------------------------
def run_quantile_lgbm(
    *, test_start: str, test_end: str, retrain: str = "weekly", features: str = "default",
    gate: str = "strict", market: str = "dam", task: str = "price",
    seed: int = 42, n_estimators: Optional[int] = None,
    out_json: Optional[str] = None, verbose: bool = True,
) -> dict:
    gs = GateSpec(task=task, gate=gate, market=market)
    horizon = MARKET_PRESETS[market]["horizon"]
    stride = MARKET_PRESETS[market]["stride"]

    df = load_processed("hourly", task=task)
    groups = parse_feature_spec(features)
    if "dense" in groups:
        df = add_dense_lags(df)
    if "engfc" in groups:
        df = add_engineered_features(df)
    df = df.dropna(subset=["y"]).sort_index()
    all_cols = [c for c in df.columns if c != "y"]
    feature_cols = select_features(all_cols, groups)
    feature_cols = [c for c in feature_cols if np.issubdtype(df[c].dtype, np.number)]

    ts0, ts1 = pd.to_datetime(test_start), pd.to_datetime(test_end)
    if verbose:
        print("=" * 78)
        print(f"QUANTILE-LGBM | task={task} market={market} retrain={retrain}")
        print(f"  {describe_gate(gs)} | horizon={horizon} stride={stride} | #features={len(feature_cols)}")
        print(f"  test=[{ts0} .. {ts1}]")
        print("=" * 78)

    blocks = make_blocks(ts0, ts1, horizon, stride)
    y_full = df["y"].astype(float)

    pred_p10: Dict[pd.Timestamp, float] = {}
    pred_p50: Dict[pd.Timestamp, float] = {}
    pred_p90: Dict[pd.Timestamp, float] = {}

    models: Optional[Tuple[object, object, object]] = None
    last_key: Optional[str] = None

    def _train_key(cutoff: pd.Timestamp) -> str:
        if retrain == "static":
            return "static"
        if retrain == "monthly":
            return f"{cutoff.year}-{cutoff.month:02d}"
        if retrain == "weekly":
            iso = cutoff.isocalendar()
            return f"{iso[0]}-W{int(iso[1]):02d}"
        return "static"

    t_start = time.time()
    for bi, (b0, b1) in enumerate(blocks):
        cutoff = gs.cutoff_for_block(b0)
        scored_idx = set(pd.date_range(b0, b1, freq="H"))
        roll_idx = pd.date_range(cutoff + pd.Timedelta(hours=1), b1, freq="H").intersection(df.index)
        if len(roll_idx) == 0:
            continue

        key = _train_key(cutoff)
        if models is None or key != last_key:
            dtr = df.loc[:cutoff]
            # AEL (§4.8): ίδιο training-row freeze με το master engine (recursive
            # σχήμα: cutoff ανά ημέρα-της-γραμμής) — train/serve συνέπεια.
            dtr = freeze_crosslags_for_gate(dtr, feature_cols, gs, df_full=df, mode="freeze")
            Xtr, ytr = make_xy(dtr)
            Xtr = Xtr[feature_cols]
            q10 = build_and_fit("lgbm", Xtr, ytr, seed=seed, n_estimators=n_estimators,
                                 objective="quantile", quantile_alpha=0.1)
            q50 = build_and_fit("lgbm", Xtr, ytr, seed=seed, n_estimators=n_estimators,
                                 objective="quantile", quantile_alpha=0.5)
            q90 = build_and_fit("lgbm", Xtr, ytr, seed=seed, n_estimators=n_estimators,
                                 objective="quantile", quantile_alpha=0.9)
            models = (q10, q50, q90)
            last_key = key
            if verbose:
                print(f"   [fit] key={key} cutoff={cutoff}  (block {bi+1}/{len(blocks)})", flush=True)

        q10, q50, q90 = models
        preds50, aux = recursive_predict_openloop(
            model=q50, df_full=df, test_index=roll_idx, feature_cols=feature_cols,
            config=OpenLoopConfig(y_floor=None), aux_models={"p10": q10, "p90": q90},
            gate=gs, crosslag_mode="freeze",
            crosslag_cutoff=gs.crosslag_cutoff_for_anchor(b0),
        )
        for t, v50, v10, v90 in zip(roll_idx, preds50, aux["p10"], aux["p90"]):
            if t in scored_idx:
                pred_p50[t] = float(v50)
                pred_p10[t] = float(v10)
                pred_p90[t] = float(v90)

    if verbose:
        print(f"   [done] {len(blocks)} blocks σε {time.time()-t_start:.1f}s", flush=True)

    scored_all = pd.DatetimeIndex(sorted(pred_p50.keys())).intersection(df.index)
    actual = y_full.reindex(scored_all).to_numpy(dtype=float)
    p10 = np.array([pred_p10[t] for t in scored_all], dtype=float)
    p50 = np.array([pred_p50[t] for t in scored_all], dtype=float)
    p90 = np.array([pred_p90[t] for t in scored_all], dtype=float)

    results = {w: evaluate_window(scored_all, actual, p10, p50, p90, win) for w, win in EVAL_WINDOWS.items()}
    print_eval_table("QUANTILE-LGBM (objective=quantile, α=0.1/0.5/0.9)", results)

    out = dict(
        method="quantile_lgbm", retrain=retrain, features=groups, gate=gate, market=market, task=task,
        test_start=test_start, test_end=test_end, results=results,
        dates=[t.isoformat() for t in scored_all],
        actual=[None if not np.isfinite(v) else round(float(v), 4) for v in actual],
        p10=[None if not np.isfinite(v) else round(float(v), 4) for v in p10],
        p50=[None if not np.isfinite(v) else round(float(v), 4) for v in p50],
        p90=[None if not np.isfinite(v) else round(float(v), 4) for v in p90],
    )
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        print(f"\n💾 saved: {out_json}")
    return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Conformal / probabilistic layer πάνω στο headline LGBM config")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp1 = sub.add_parser("split-conformal", help="rolling causal residual-quantile intervals πάνω σε point forecast JSON")
    sp1.add_argument("--forecast_json", required=True, help="extended point-forecast JSON (master_forecast --out_json)")
    sp1.add_argument("--out_json", default=None)
    sp1.add_argument("--min_weeks", type=int, default=4)
    sp1.add_argument("--max_weeks", type=int, default=8)

    sp2 = sub.add_parser("quantile-lgbm", help="LGBM objective=quantile α=0.1/0.5/0.9, ίδιο recursive/weekly/default config")
    sp2.add_argument("--test_start", required=True)
    sp2.add_argument("--test_end", required=True)
    sp2.add_argument("--retrain", default="weekly", choices=["static", "monthly", "weekly"])
    sp2.add_argument("--features", default="default")
    sp2.add_argument("--gate", default="strict", choices=["strict", "academic"])
    sp2.add_argument("--market", default="dam", choices=["dam", "idm", "forward"])
    sp2.add_argument("--task", default="price", choices=["price", "load"])
    sp2.add_argument("--seed", type=int, default=42)
    sp2.add_argument("--n_estimators", type=int, default=None)
    sp2.add_argument("--out_json", default=None)
    sp2.add_argument("--quiet", action="store_true")

    args = p.parse_args()
    if args.cmd == "split-conformal":
        run_split_conformal(args.forecast_json, args.out_json, args.min_weeks, args.max_weeks)
    elif args.cmd == "quantile-lgbm":
        run_quantile_lgbm(
            test_start=args.test_start, test_end=args.test_end, retrain=args.retrain,
            features=args.features, gate=args.gate, market=args.market, task=args.task,
            seed=args.seed, n_estimators=args.n_estimators, out_json=args.out_json,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
