import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .split_utils import load_processed, split_time_series

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


def _parse_tuple(s: Optional[str], k: int) -> Optional[Tuple[int, ...]]:
    if s is None:
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != k:
        raise ValueError(f"Expected {k} comma-separated ints, got: {s}")
    return tuple(int(p) for p in parts)


def _seasonal_naive_1step(history: np.ndarray, season_len: int) -> float:
    if history.size >= season_len:
        return float(history[-season_len])
    if history.size > 0:
        return float(history[-1])
    return float("nan")


def _fit_sarimax(hist: np.ndarray, order: Tuple[int, int, int], seas: Tuple[int, int, int, int], maxiter: int):
    model = SARIMAX(
        hist,
        order=order,
        seasonal_order=seas,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False, maxiter=maxiter, method="lbfgs")

    converged = True
    if hasattr(res, "mle_retvals") and isinstance(res.mle_retvals, dict):
        converged = bool(res.mle_retvals.get("converged", True))
    if not converged:
        try:
            res = model.fit(disp=False, maxiter=maxiter, method="powell")
        except Exception:
            pass
    return res


def _fit_with_ladder(
    hist: np.ndarray,
    order: Tuple[int, int, int],
    seas: Tuple[int, int, int, int],
    maxiter: int,
    verbose: bool,
) -> Tuple[Optional[object], Dict[str, object]]:
    s = int(seas[3])
    candidates: List[Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]] = [
        (order, seas),
        ((1, order[1], 1), (seas[0], seas[1], 0, s)),
        ((1, 1, 1), (0, 1, 1, s)),
        ((1, 1, 1), (0, 0, 0, s)),
    ]

    info: Dict[str, object] = {"used": None, "attempts": []}

    for (o, so) in candidates:
        try:
            res = _fit_sarimax(hist, o, so, maxiter=maxiter)
            info["used"] = {"order": list(o), "seasonal_order": list(so)}
            info["attempts"].append({"order": list(o), "seasonal_order": list(so), "ok": True})
            return res, info
        except Exception as e:
            info["attempts"].append(
                {"order": list(o), "seasonal_order": list(so), "ok": False, "err": f"{type(e).__name__}: {e}"}
            )
            if verbose:
                print(f"[SARIMA] fit failed for order={o}, seasonal={so}: {e}")

    return None, info


def sarima_rolling_one_step(
    y_train: pd.Series,
    y_test: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    fit_window: Optional[int],
    refit_every: int,
    maxiter: int,
    verbose: bool,
) -> Tuple[np.ndarray, Dict[str, object]]:
    season_len = int(seasonal_order[3])

    train_vals = y_train.to_numpy(dtype=float)
    test_vals = y_test.to_numpy(dtype=float)
    preds = np.full(len(test_vals), np.nan, dtype=float)

    history = train_vals.copy()
    if fit_window is not None and history.size > int(fit_window):
        history = history[-int(fit_window) :]

    stats: Dict[str, object] = {
        "order_requested": list(order),
        "seasonal_requested": list(seasonal_order),
        "fit_window": int(fit_window) if fit_window is not None else None,
        "refit_every": int(refit_every),
        "maxiter": int(maxiter),
        "initial_fit": None,
        "refits_ok": 0,
        "fit_failures": 0,
        "forecast_failures": 0,
        "append_failures": 0,
        "fallback_steps": 0,
        "time_sec": None,
    }

    min_hist = max(80, 3 * season_len + 20)

    t0 = time.time()

    res = None
    if history.size >= min_hist:
        res, fit_info = _fit_with_ladder(history, order, seasonal_order, maxiter=maxiter, verbose=verbose)
        stats["initial_fit"] = fit_info
        if res is None:
            stats["fit_failures"] += 1
        else:
            stats["refits_ok"] += 1
    else:
        stats["fit_failures"] += 1
        stats["initial_fit"] = {"used": None, "reason": f"Not enough history: {history.size} < {min_hist}"}

    for i in range(len(test_vals)):
        if i > 0 and refit_every > 0 and (i % refit_every == 0):
            try:
                hist_fit = history
                if fit_window is not None and hist_fit.size > int(fit_window):
                    hist_fit = hist_fit[-int(fit_window) :]
                if hist_fit.size < min_hist:
                    raise ValueError("Not enough history for refit.")
                res2, _ = _fit_with_ladder(hist_fit, order, seasonal_order, maxiter=maxiter, verbose=verbose)
                if res2 is not None:
                    res = res2
                    stats["refits_ok"] += 1
                else:
                    stats["fit_failures"] += 1
                    res = None
            except Exception:
                stats["fit_failures"] += 1
                res = None

        yhat = np.nan
        if res is not None:
            try:
                fc = res.forecast(steps=1)
                yhat = float(np.asarray(fc, dtype=float).reshape(-1)[0])
                if not np.isfinite(yhat):
                    raise ValueError("Non-finite forecast")
            except Exception:
                stats["forecast_failures"] += 1
                yhat = np.nan

        if not np.isfinite(yhat):
            stats["fallback_steps"] += 1
            yhat = _seasonal_naive_1step(history, season_len)

        preds[i] = yhat

        y_true = float(test_vals[i])
        history = np.append(history, y_true)
        if fit_window is not None and history.size > int(fit_window):
            history = history[-int(fit_window) :]

        if res is not None and hasattr(res, "append"):
            try:
                try:
                    res = res.append([y_true], refit=False, copy_initialization=True)
                except TypeError:
                    res = res.append([y_true], refit=False)
            except Exception:
                stats["append_failures"] += 1
                res = None

        if res is None and history.size >= min_hist:
            res3, _ = _fit_with_ladder(history, order, seasonal_order, maxiter=maxiter, verbose=verbose)
            if res3 is not None:
                res = res3
                stats["refits_ok"] += 1

    stats["time_sec"] = round(time.time() - t0, 2)
    return preds, stats


def run(
    mode: str,
    test_size: Optional[int],
    train_start: Optional[str],
    train_end: Optional[str],
    test_start: Optional[str],
    test_end: Optional[str],
    order: Optional[Tuple[int, int, int]],
    seasonal: Optional[Tuple[int, int, int, int]],
    fit_window: Optional[int],
    refit_every: Optional[int],
    maxiter: Optional[int],
    verbose: bool,
) -> None:
    if mode != "hourly":
        raise ValueError("Only mode='hourly' is supported (daily removed).")

    df = load_processed(mode)
    df_train, df_test = split_time_series(
        df,
        mode=mode,
        test_size=test_size,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )

    y_train = df_train["y"].astype(float)
    y_test = df_test["y"].astype(float)

    default_order = (2, 1, 2)
    default_seasonal = (1, 1, 0, 24)
    default_fit_window = 24 * 90
    default_refit_every = 24
    default_maxiter = 120

    use_order = order if order is not None else default_order
    use_seasonal = seasonal if seasonal is not None else default_seasonal
    use_fit_window = int(fit_window) if fit_window is not None else default_fit_window
    use_refit_every = int(refit_every) if refit_every is not None else default_refit_every
    use_maxiter = int(maxiter) if maxiter is not None else default_maxiter

    print("🧠 SARIMA (HOURLY) [ROLLING 1-STEP | NO-LEAK | ROBUST]")
    print(f"[INFO] train={len(y_train)}, test={len(y_test)} | order={use_order} seasonal={use_seasonal}")
    if any(v is not None for v in [train_start, train_end, test_start, test_end]):
        print(f"[INFO] bounds: train_start={train_start} train_end={train_end} test_start={test_start} test_end={test_end}")

    preds, stats = sarima_rolling_one_step(
        y_train=y_train,
        y_test=y_test,
        order=use_order,
        seasonal_order=use_seasonal,
        fit_window=use_fit_window,
        refit_every=use_refit_every,
        maxiter=use_maxiter,
        verbose=verbose,
    )

    print(
        f"[SARIMA] done | refits_ok={stats['refits_ok']} | fit_failures={stats['fit_failures']} "
        f"| forecast_failures={stats['forecast_failures']} | append_failures={stats['append_failures']} "
        f"| fallback_steps={stats['fallback_steps']}/{len(preds)} | time={stats['time_sec']}s"
    )

    MODELS_DIR.mkdir(exist_ok=True)

    horizon = int(len(preds))
    cache_path = MODELS_DIR / f"sarima_{mode}_{horizon}.pkl"
    cache_obj = {
        "mode": mode,
        "horizon": horizon,
        "bounds": {
            "test_size": test_size,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        },
        "stats": stats,
        "dates": [d.isoformat() for d in y_test.index],
        "pred": [float(v) if np.isfinite(v) else None for v in preds],
    }
    joblib.dump(cache_obj, cache_path)
    print(f"✅ Saved cache: {cache_path}")

    legacy_path = MODELS_DIR / f"sarima_{mode}.json"
    with open(legacy_path, "w", encoding="utf-8") as f:
        json.dump(cache_obj, f, ensure_ascii=False, indent=2)
    print(f"✅ Updated legacy: {legacy_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", nargs="?", default="hourly", choices=["hourly"])
    p.add_argument("--test_size", type=int, default=None)
    p.add_argument("--train_start", type=str, default=None)
    p.add_argument("--train_end", type=str, default=None)
    p.add_argument("--test_start", type=str, default=None)
    p.add_argument("--test_end", type=str, default=None)

    p.add_argument("--order", type=str, default=None, help="e.g. 2,1,2")
    p.add_argument("--seasonal", type=str, default=None, help="e.g. 1,1,0,24")
    p.add_argument("--fit_window", type=int, default=None)
    p.add_argument("--refit_every", type=int, default=None)
    p.add_argument("--maxiter", type=int, default=None)
    p.add_argument("--verbose", action="store_true")

    args = p.parse_args()

    run(
        mode=args.mode,
        test_size=args.test_size,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        order=_parse_tuple(args.order, 3),
        seasonal=_parse_tuple(args.seasonal, 4),
        fit_window=args.fit_window,
        refit_every=args.refit_every,
        maxiter=args.maxiter,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
