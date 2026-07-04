"""
feature_strategy_compare1.py (hourly only)

Compares feature strategies for MIMO forecasting:
- all: use all available features (except target column)
- topk: use top-k ranked features (k in --topk)
- hybrid: forced + top-k ranked features

Supports single-week (--test_start + --week_label) and multi-week (--test_starts + --week_labels).

Run:
  python -m src.feature_strategy_compare1 hourly --horizon 24 --test_size 168 --train_end ... --test_start ...
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TARGET_COL = "y"


@dataclass
class WeekSpec:
    label: str
    test_start: pd.Timestamp


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _json_default(o):
    # json.dump helper for numpy/pandas types
    try:
        import numpy as _np
        import pandas as _pd
        if isinstance(o, (_np.integer,)):
            return int(o)
        if isinstance(o, (_np.floating,)):
            return float(o)
        if isinstance(o, (_np.ndarray,)):
            return o.tolist()
        if isinstance(o, (_pd.Timestamp,)):
            return o.isoformat(sep=" ")
    except Exception:
        pass
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def _df_records_jsonable(df: pd.DataFrame) -> List[Dict]:
    if df is None or df.empty:
        return []
    df2 = df.copy()
    # Convert datetime-like columns to strings (safe even if first values are NaN)
    for c in df2.columns:
        if pd.api.types.is_datetime64_any_dtype(df2[c]) or pd.api.types.is_datetime64tz_dtype(df2[c]):
            df2[c] = df2[c].astype(str)
    recs: List[Dict] = []
    for r in df2.to_dict(orient="records"):
        rr: Dict = {}
        for k, v in r.items():
            if isinstance(v, (np.integer,)):
                rr[k] = int(v)
            elif isinstance(v, (np.floating,)):
                rr[k] = float(v)
            else:
                rr[k] = v
        recs.append(rr)
    return recs


def _available_features(df: pd.DataFrame, feats: Sequence[str]) -> List[str]:
    cols = set(df.columns)
    return [f for f in feats if f in cols and f != TARGET_COL]


def _parse_csv_ints(s: str) -> List[int]:
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_csv_strs(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def load_processed_hourly() -> pd.DataFrame:
    """
    Project-specific loader:
    expects processed hourly dataframe with TARGET_COL='y' and engineered features.
    Adjust path/loader ONLY if your project differs.
    """
    # Try common locations
    candidates = [
        "data/processed/hourly.parquet",
        "data/processed/processed_hourly.parquet",
        "data/processed/entsoe_extra_hourly.parquet",
        "data/processed/hourly_processed.parquet",
    ]
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_parquet(p)
            break
    else:
        raise FileNotFoundError(
            "Δεν βρήκα processed hourly parquet. "
            "Βάλε το path σου μέσα στη load_processed_hourly()."
        )

    # Ensure datetime index
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.set_index("ds")
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Το hourly dataframe πρέπει να έχει DatetimeIndex ή στήλη ds.")

    df = df.sort_index()
    return df


def split_time_series(
    df: pd.DataFrame,
    mode: str,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _ = mode  # hourly only
    df = df.sort_index()

    df_train = df.loc[:train_end].copy()
    df_test = df.loc[test_start:].copy()

    if len(df_test) < int(test_size):
        raise ValueError(
            f"Not enough rows for test_size={test_size}. Got {len(df_test)} from test_start={test_start}."
        )

    df_test = df_test.iloc[: int(test_size)].copy()
    return df_train, df_test


def build_xy_mimo(df_train: pd.DataFrame, feature_cols: Sequence[str], horizon: int, y_col: str) -> Tuple[pd.DataFrame, np.ndarray]:
    df_train = df_train.copy()
    X = df_train.loc[:, list(feature_cols)].copy()

    y = df_train[y_col].values.astype(float)
    n = len(y)

    # create multioutput target [t+1 .. t+h]
    Y = []
    max_t = n - horizon
    for t in range(max_t):
        Y.append(y[t + 1 : t + 1 + horizon])
    Y = np.asarray(Y, dtype=float)

    X = X.iloc[:max_t].copy()
    return X, Y


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) + eps
    return float(np.mean(200.0 * np.abs(y_pred - y_true) / denom))


def eval_multi_origin_mimo(
    model,
    df_full: pd.DataFrame,
    feature_cols: Sequence[str],
    test_start: pd.Timestamp,
    test_size: int,
    horizon: int,
    origin_stride: int,
    y_col: str,
) -> Dict[str, float]:
    df_test = df_full.loc[test_start:].iloc[: int(test_size)].copy()
    y_true_full = df_test[y_col].values.astype(float)

    origins = list(range(0, int(test_size) - horizon + 1, int(origin_stride)))
    if not origins:
        origins = [0]

    maes, rmses, smapes = [], [], []
    for o in origins:
        x_row = df_test.iloc[o : o + 1][list(feature_cols)]
        y_true = y_true_full[o : o + horizon]
        y_pred = np.asarray(model.predict(x_row)).reshape(-1)
        y_pred = y_pred[:horizon]

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        s = smape(y_true, y_pred)

        maes.append(mae)
        rmses.append(rmse)
        smapes.append(s)

    return {
        "MAE": float(np.mean(maes)),
        "RMSE": float(np.mean(rmses)),
        "sMAPE": float(np.mean(smapes)),
        "MAE_std": float(np.std(maes)),
        "RMSE_std": float(np.std(rmses)),
        "sMAPE_std": float(np.std(smapes)),
        "n_origins": int(len(origins)),
    }


def rank_features(
    df_train: pd.DataFrame,
    feature_cols: Sequence[str],
    y_col: str,
    rank_steps: Sequence[int],
    n_estimators: int,
    max_rows: Optional[int],
    seed: int,
) -> pd.DataFrame:
    from lightgbm import LGBMRegressor

    X = df_train.loc[:, list(feature_cols)].copy()
    y = df_train[y_col].values.astype(float)

    # Create supervised samples using multiple steps (like t+24, t+168) to rank useful features
    rows_X = []
    rows_y = []
    n = len(y)

    for step in rank_steps:
        max_t = n - int(step)
        if max_t <= 0:
            continue
        Xi = X.iloc[:max_t].copy()
        yi = y[int(step) : int(step) + max_t]
        rows_X.append(Xi)
        rows_y.append(yi)

    if not rows_X:
        raise ValueError("rank_steps οδηγούν σε 0 samples. Μείωσε rank_steps ή αύξησε train window.")

    Xr = pd.concat(rows_X, axis=0, ignore_index=True)
    yr = np.concatenate(rows_y, axis=0)

    if max_rows and len(Xr) > int(max_rows):
        idx = np.linspace(0, len(Xr) - 1, int(max_rows)).astype(int)
        Xr = Xr.iloc[idx].copy()
        yr = yr[idx]

    m = LGBMRegressor(
        n_estimators=int(n_estimators),
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=int(seed),
        n_jobs=-1,
    )
    m.fit(Xr, yr)

    imp = m.feature_importances_.astype(float)
    out = pd.DataFrame({"feature": list(feature_cols), "importance": imp})
    out = out.sort_values("importance", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out[["rank", "feature", "importance"]]


def _select_topk(ranked: pd.DataFrame, k: int, forced_ok: Sequence[str]) -> List[str]:
    forced_set = set(forced_ok)
    top = [f for f in ranked["feature"].tolist() if f not in forced_set]
    top = top[: int(k)]
    return list(top)


def _make_model(name: str, seed: int = 13, use_gpu_xgb: bool = False):
    name = name.lower().strip()
    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=600,
            random_state=seed,
            n_jobs=-1,
            max_depth=None,
            min_samples_leaf=2,
        )
    if name == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "xgb":
        from xgboost import XGBRegressor
        params = dict(
            n_estimators=1400,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
        # XGBoost >=2 uses device='cuda' for GPU; keep CPU as default.
        if use_gpu_xgb:
            try:
                params["device"] = "cuda"
            except Exception:
                pass
        try:
            return XGBRegressor(**params)
        except TypeError:
            params.pop("device", None)
            return XGBRegressor(**params)
    if name == "mlp":
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("mlp", MLPRegressor(
                    hidden_layer_sizes=(256, 128),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=400,
                    random_state=seed,
                    early_stopping=True,
                    n_iter_no_change=20,
                )),
            ]
        )

    raise ValueError(f"Unknown model: {name}")


def _wrap_multioutput_if_needed(base_model, model_name: str):
    # All our estimators support multioutput via MultiOutputRegressor except those that natively handle it.
    # Safer: always wrap.
    from sklearn.multioutput import MultiOutputRegressor
    return MultiOutputRegressor(base_model, n_jobs=-1)


def run_week(
    df_full: pd.DataFrame,
    mode: str,
    week: WeekSpec,
    train_end: pd.Timestamp,
    test_size: int,
    horizon: int,
    eval_mode: str,
    origin_stride: int,
    models: Sequence[str],
    strategies: Sequence[str],
    topk_list: Sequence[int],
    forced_features: Sequence[str],
    rank_steps: Sequence[int],
    rank_n_estimators: int,
    rank_max_rows: Optional[int],
    train_max_rows: Optional[int],
    use_gpu_xgb: bool,
    print_features: bool,
    seed: int = 13,
) -> Tuple[pd.DataFrame, Dict]:
    df_train, _df_test = split_time_series(
        df_full, mode=mode,
        train_end=train_end,
        test_start=week.test_start,
        test_size=int(test_size),
    )

    feature_cols_all = [c for c in df_full.columns if c != TARGET_COL]
    forced_ok = _available_features(df_full, forced_features)

    ranked = rank_features(
        df_train=df_train,
        feature_cols=feature_cols_all,
        y_col=TARGET_COL,
        rank_steps=rank_steps,
        n_estimators=int(rank_n_estimators),
        max_rows=rank_max_rows,
        seed=seed,
    )

    feature_sets: Dict[Tuple[str, Optional[int]], List[str]] = {}

    if "all" in strategies:
        feature_sets[("all", None)] = feature_cols_all

    for k in topk_list:
        if "topk" in strategies:
            feature_sets[("topk", int(k))] = _select_topk(ranked, int(k), forced_ok)
        if "hybrid" in strategies:
            topk_feats = _select_topk(ranked, int(k), forced_ok)
            hybrid = list(dict.fromkeys(list(forced_ok) + list(topk_feats)))
            feature_sets[("hybrid", int(k))] = hybrid

    if print_features:
        print(f"\n[WEEK {week.label}] Forced features used ({len(forced_ok)}): {forced_ok}")
        print(f"[WEEK {week.label}] Ranked features (top 30):")
        for _, row in ranked.head(30).iterrows():
            print(f"  {int(row['rank']):>3d}. {row['feature']:<35s}  imp={row['importance']:.6f}")
        for k in topk_list:
            if ("topk", int(k)) in feature_sets:
                feats = feature_sets[("topk", int(k))]
                print(f"[WEEK {week.label}] topk k={k} ({len(feats)} feats): {feats}")
            if ("hybrid", int(k)) in feature_sets:
                feats = feature_sets[("hybrid", int(k))]
                print(f"[WEEK {week.label}] hybrid k={k} ({len(feats)} feats): {feats}")

    rows = []
    meta_week = {
        "label": week.label,
        "test_start": str(week.test_start),
        "train_end": str(train_end),
        "test_size": int(test_size),
        "horizon": int(horizon),
        "origin_stride": int(origin_stride),
        "rank_steps": [int(x) for x in rank_steps],
        "rank_n_estimators": int(rank_n_estimators),
        "rank_max_rows": int(rank_max_rows) if rank_max_rows else None,
        "train_max_rows": int(train_max_rows) if train_max_rows else None,
        "forced_used": forced_ok,
        "ranked_features": ranked.to_dict(orient="records"),
        "feature_sets": {f"{s}" + (f"_{k}" if k is not None else ""): v for (s, k), v in feature_sets.items()},
    }

    for model_name in models:
        base_model = _make_model(model_name, seed=seed, use_gpu_xgb=use_gpu_xgb)
        model = _wrap_multioutput_if_needed(base_model, model_name)

        for (strategy, k), feats in feature_sets.items():
            X_train, Y_train = build_xy_mimo(df_train, feats, horizon=horizon, y_col=TARGET_COL)
            if train_max_rows and len(X_train) > int(train_max_rows):
                idx = np.linspace(0, len(X_train) - 1, int(train_max_rows)).astype(int)
                X_train = X_train.iloc[idx]
                Y_train = Y_train[idx]

            model.fit(X_train, Y_train)

            if eval_mode == "multi_origin":
                metrics = eval_multi_origin_mimo(
                    model=model,
                    df_full=df_full,
                    feature_cols=feats,
                    test_start=week.test_start,
                    test_size=int(test_size),
                    horizon=int(horizon),
                    origin_stride=int(origin_stride),
                    y_col=TARGET_COL,
                )
            else:
                metrics = eval_multi_origin_mimo(
                    model=model,
                    df_full=df_full,
                    feature_cols=feats,
                    test_start=week.test_start,
                    test_size=int(test_size),
                    horizon=int(horizon),
                    origin_stride=int(test_size),
                    y_col=TARGET_COL,
                )

            rows.append({
                "week": week.label,
                "test_start": week.test_start,
                "model": model_name.upper(),
                "strategy": strategy,
                "k": k if k is not None else "",
                **metrics,
                "n_features": len(feats),
            })

    df_res = pd.DataFrame(rows)
    df_res = df_res.sort_values(["week", "MAE", "RMSE"], ascending=[True, True, True])

    return df_res, meta_week


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="feature_strategy_compare1.py", add_help=True)
    sub = p.add_subparsers(dest="mode", required=True)

    ph = sub.add_parser("hourly", help="Hourly processed dataset (only supported mode)")
    ph.add_argument("--horizon", type=int, required=True)
    ph.add_argument("--test_size", type=int, default=168)
    ph.add_argument("--train_end", type=str, required=True)
    ph.add_argument("--test_start", type=str, default=None)
    ph.add_argument("--test_starts", type=str, default=None)
    ph.add_argument("--week_label", type=str, default=None)
    ph.add_argument("--week_labels", type=str, default=None)

    ph.add_argument("--eval_mode", type=str, default="multi_origin", choices=["multi_origin", "single_origin"])
    ph.add_argument("--origin_stride", type=int, default=24)

    ph.add_argument("--models", type=str, default="mlp,rf,lgbm,xgb")
    ph.add_argument("--use_gpu_xgb", action="store_true", help="Try to run XGB on CUDA (xgboost>=2). If unsupported, falls back to CPU.")
    ph.add_argument("--strategies", type=str, default="all,topk,hybrid")
    ph.add_argument("--topk", type=str, default="10,20,30")

    ph.add_argument("--rank_steps", type=str, default="24,168")
    ph.add_argument("--rank_n_estimators", type=int, default=1000)
    ph.add_argument("--rank_max_rows", type=int, default=30000)
    ph.add_argument("--train_max_rows", type=int, default=30000)

    ph.add_argument("--print_features", action="store_true")
    ph.add_argument("--seed", type=int, default=13)
    ph.add_argument("--save_json", type=str, default=None)

    return p


def main():
    args = build_arg_parser().parse_args()

    if args.mode != "hourly":
        raise ValueError("Only hourly mode is supported here.")

    horizon = int(args.horizon)
    test_size = int(args.test_size)
    train_end = pd.to_datetime(args.train_end)

    eval_mode = str(args.eval_mode)
    origin_stride = int(args.origin_stride)

    models = _parse_csv_strs(args.models)
    strategies = _parse_csv_strs(args.strategies)
    topk_list = _parse_csv_ints(args.topk)

    rank_steps = _parse_csv_ints(args.rank_steps)
    rank_n_estimators = int(args.rank_n_estimators)
    rank_max_rows = int(args.rank_max_rows) if args.rank_max_rows is not None else None
    train_max_rows = int(args.train_max_rows) if args.train_max_rows is not None else None

    # Forced features (stable calendar + essential lags)
    forced_features = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_holiday",
        "y_lag1", "y_lag24", "y_lag168"
    ]

    df_full = load_processed_hourly()

    # Build week specs
    weeks: List[WeekSpec] = []
    if args.test_starts:
        starts = _parse_csv_strs(args.test_starts)
        labels = _parse_csv_strs(args.week_labels) if args.week_labels else [f"week{i+1}" for i in range(len(starts))]
        if len(labels) != len(starts):
            raise ValueError("week_labels must match number of test_starts.")
        for s, lab in zip(starts, labels):
            weeks.append(WeekSpec(label=lab, test_start=pd.to_datetime(s)))
    else:
        if not args.test_start or not args.week_label:
            raise ValueError("For single-week run, set --test_start AND --week_label.")
        weeks = [WeekSpec(label=str(args.week_label), test_start=pd.to_datetime(args.test_start))]

    print(f"\n[INFO] eval_mode={eval_mode} | origin_stride={origin_stride}h")
    print(f"[INFO] forecast_horizon={horizon} | test_size={test_size}")
    print(f"[INFO] train_end={train_end}")
    print(f"[INFO] models={models} | strategies={strategies} | topk={topk_list}")
    print(f"[INFO] rank_steps={rank_steps} | rank_n_estimators={rank_n_estimators} | rank_max_rows={rank_max_rows} | train_max_rows={train_max_rows}")

    all_results = []
    meta = {
        "mode": "hourly",
        "horizon": horizon,
        "test_size": test_size,
        "train_end": str(train_end),
        "eval_mode": eval_mode,
        "origin_stride": origin_stride,
        "models": models,
        "strategies": strategies,
        "topk": topk_list,
        "rank_steps": rank_steps,
        "rank_n_estimators": rank_n_estimators,
        "rank_max_rows": rank_max_rows,
        "train_max_rows": train_max_rows,
        "forced": forced_features,
        "weeks": {},
    }

    for week in weeks:
        print(f"\n[INFO] test_start={week.test_start} | label={week.label}")
        df_res, meta_week = run_week(
            df_full=df_full,
            mode="hourly",
            week=week,
            train_end=train_end,
            test_size=test_size,
            horizon=horizon,
            eval_mode=eval_mode,
            origin_stride=origin_stride,
            models=models,
            strategies=strategies,
            topk_list=topk_list,
            forced_features=forced_features,
            rank_steps=rank_steps,
            rank_n_estimators=rank_n_estimators,
            rank_max_rows=rank_max_rows,
            train_max_rows=train_max_rows,
            use_gpu_xgb=bool(args.use_gpu_xgb),
            print_features=bool(args.print_features),
            seed=int(args.seed),
        )

        print(f"\n=== RESULTS: {week.label} | test_start={week.test_start} ===")
        cols_show = ["model", "strategy", "k", "n_features", "MAE", "RMSE", "sMAPE", "MAE_std", "RMSE_std", "sMAPE_std", "n_origins"]
        cols_show = [c for c in cols_show if c in df_res.columns]
        print(df_res[cols_show].to_string(index=False))

        meta_week["rows"] = _df_records_jsonable(df_res)
        meta["weeks"][week.label] = meta_week
        all_results.append(df_res)

    df_all = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    if args.save_json:
        # Save both meta + a flat table of metrics rows so downstream scripts can aggregate objectively.
        df_all_safe = df_all.copy()
        meta["rows"] = _df_records_jsonable(df_all_safe)

        _ensure_dir(args.save_json)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=_json_default)
        print(f"\n✅ Saved JSON: {args.save_json}")


if __name__ == "__main__":
    main()
