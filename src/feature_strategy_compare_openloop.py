import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.split_utils import load_processed, split_time_series, make_xy

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import xgboost as xgb
except Exception:
    xgb = None


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def _as_list_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _as_int_list_csv(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _safe_mkdir(p: Optional[str]) -> Optional[Path]:
    if not p:
        return None
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_model(model_name: str, seed: int, n_jobs: int):
    name = model_name.lower()
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=400,
            random_state=seed,
            n_jobs=n_jobs,
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=2,
            max_features=1.0,
        )
    if name == "lgbm":
        if lgb is None:
            raise ImportError("lightgbm is not installed but model lgbm was requested.")
        return lgb.LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=seed,
            n_jobs=n_jobs,
        )
    if name == "xgb":
        if xgb is None:
            raise ImportError("xgboost is not installed but model xgb was requested.")
        return xgb.XGBRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=n_jobs,
        )
    if name == "mlp":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(256, 256),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        batch_size=512,
                        learning_rate="adaptive",
                        learning_rate_init=1e-3,
                        max_iter=350,
                        random_state=seed,
                        early_stopping=True,
                        n_iter_no_change=20,
                    ),
                ),
            ]
        )
    if name == "svr":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("svr", SVR(C=30.0, epsilon=0.1, gamma="scale")),
            ]
        )
    raise ValueError(f"Unknown model: {model_name}")


def rank_features(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    y_col: str,
    rank_steps: List[int],
    n_estimators: int,
    max_rows: int,
    seed: int,
    n_jobs: int,
) -> List[Tuple[str, float]]:
    rng = np.random.default_rng(seed)
    imp_acc = np.zeros(len(feature_cols), dtype=float)

    for step in rank_steps:
        y = df_train[y_col].shift(-step)
        X = df_train[feature_cols].copy()
        valid = y.notna()
        X = X.loc[valid]
        yv = y.loc[valid]

        if len(X) > max_rows:
            idx = rng.choice(len(X), size=max_rows, replace=False)
            X = X.iloc[idx]
            yv = yv.iloc[idx]

        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            random_state=seed + step,
            n_jobs=n_jobs,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
        )
        model.fit(X, yv)
        imp = getattr(model, "feature_importances_", None)
        if imp is None:
            continue
        imp_acc += np.asarray(imp, dtype=float)

    if imp_acc.sum() <= 0:
        return [(c, 0.0) for c in feature_cols]

    imp_acc = imp_acc / imp_acc.sum()
    order = np.argsort(-imp_acc)
    return [(feature_cols[i], float(imp_acc[i])) for i in order]


_Y_LAG_RE = re.compile(r"^y_lag(\d+)$")
_Y_ROLL_RE = re.compile(r"^y_roll(\d+)$")


@dataclass
class YFeatureSpec:
    lag_feats: Dict[str, int]
    roll_feats: Dict[str, int]
    max_window: int


def _infer_y_feature_spec(feature_cols: Sequence[str]) -> YFeatureSpec:
    lag_feats: Dict[str, int] = {}
    roll_feats: Dict[str, int] = {}
    max_window = 0

    for c in feature_cols:
        m = _Y_LAG_RE.match(c)
        if m:
            k = int(m.group(1))
            lag_feats[c] = k
            max_window = max(max_window, k)
            continue
        m = _Y_ROLL_RE.match(c)
        if m:
            k = int(m.group(1))
            roll_feats[c] = k
            max_window = max(max_window, k)

    return YFeatureSpec(lag_feats=lag_feats, roll_feats=roll_feats, max_window=max_window)


def _get_history_y(df_full: pd.DataFrame, t0: pd.Timestamp, y_col: str, window: int) -> np.ndarray:
    y_hist = df_full.loc[:t0, y_col].iloc[:-1].to_numpy(dtype=float)
    if len(y_hist) < window:
        raise ValueError(f"Not enough history before {t0} to fill window={window}. Have {len(y_hist)}.")
    return y_hist[-window:]


def eval_multi_origin_openloop(
    model,
    df_full: pd.DataFrame,
    test_index: pd.DatetimeIndex,
    feature_cols: List[str],
    horizon: int,
    origin_stride: int,
    y_col: str,
) -> Tuple[Dict[str, float], Dict[str, float], int]:
    spec = _infer_y_feature_spec(feature_cols)
    if spec.max_window <= 0:
        spec.max_window = 1

    maes: List[float] = []
    rmses: List[float] = []
    smapes: List[float] = []

    n_origins = 0
    last_start = len(test_index) - horizon
    if last_start < 0:
        raise ValueError(f"test_index too short ({len(test_index)}) for horizon={horizon}.")

    for start in range(0, last_start + 1, origin_stride):
        t0 = test_index[start]
        y_hist = _get_history_y(df_full, t0, y_col=y_col, window=spec.max_window).tolist()

        preds: List[float] = []
        ts_h = test_index[start : start + horizon]
        y_true = df_full.loc[ts_h, y_col].to_numpy(dtype=float)

        for t in ts_h:
            row = df_full.loc[t, feature_cols].copy()

            for fname, k in spec.lag_feats.items():
                row[fname] = float(y_hist[-k])
            for fname, k in spec.roll_feats.items():
                row[fname] = float(np.mean(y_hist[-k:]))

            X_row = pd.DataFrame([row.values], columns=feature_cols)
            yhat = float(model.predict(X_row)[0])
            preds.append(yhat)

            y_hist.append(yhat)
            if len(y_hist) > spec.max_window:
                y_hist = y_hist[-spec.max_window:]

        y_pred = np.asarray(preds, dtype=float)
        maes.append(mean_absolute_error(y_true, y_pred))
        rmses.append(math.sqrt(mean_squared_error(y_true, y_pred)))
        smapes.append(smape(y_true, y_pred))
        n_origins += 1

    mean_metrics = {"MAE": float(np.mean(maes)), "RMSE": float(np.mean(rmses)), "sMAPE": float(np.mean(smapes))}
    std_metrics = {
        "MAE_std": float(np.std(maes, ddof=0)),
        "RMSE_std": float(np.std(rmses, ddof=0)),
        "sMAPE_std": float(np.std(smapes, ddof=0)),
    }
    return mean_metrics, std_metrics, n_origins


def _build_feature_sets(
    all_features: List[str],
    ranked: List[Tuple[str, float]],
    forced: List[str],
    topk_list: List[int],
    strategies: List[str],
) -> Dict[Tuple[str, Optional[int]], List[str]]:
    ranked_names = [n for n, _ in ranked]
    forced_set = [f for f in forced if f in all_features]

    out: Dict[Tuple[str, Optional[int]], List[str]] = {}

    if "all" in strategies:
        out[("all", None)] = list(all_features)

    if "topk" in strategies:
        for k in topk_list:
            top = [f for f in ranked_names if f in all_features][:k]
            out[("topk", k)] = top

    if "hybrid" in strategies:
        for k in topk_list:
            top = [f for f in ranked_names if f in all_features][:k]
            seen = set()
            feats: List[str] = []
            for f in forced_set + top:
                if f not in seen:
                    feats.append(f)
                    seen.add(f)
            out[("hybrid", k)] = feats

    return out


def _to_jsonable(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    return obj


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    ph = sub.add_parser("hourly")
    ph.add_argument("--horizon", type=int, required=True)
    ph.add_argument("--origin_stride", type=int, default=24)
    ph.add_argument("--eval_mode", type=str, default="multi_origin", choices=["multi_origin"])

    ph.add_argument("--split_mode", type=str, default="tail", choices=["tail", "time_range"])
    ph.add_argument("--train_start", type=str, default=None)
    ph.add_argument("--train_end", type=str, required=True)
    ph.add_argument("--test_start", type=str, required=True)
    ph.add_argument("--test_end", type=str, default=None)
    ph.add_argument("--test_size", type=int, default=None)

    ph.add_argument("--models", type=str, default="lgbm,xgb,rf,mlp,svr")
    ph.add_argument("--strategies", type=str, default="all,topk,hybrid")
    ph.add_argument("--topk", type=str, default="10,20,30,40,60")
    ph.add_argument("--rank_steps", type=str, default="24,168")
    ph.add_argument("--rank_n_estimators", type=int, default=1000)
    ph.add_argument("--rank_max_rows", type=int, default=30000)
    ph.add_argument("--train_max_rows", type=int, default=30000)

    ph.add_argument("--seed", type=int, default=123)
    ph.add_argument("--n_jobs", type=int, default=-1)

    ph.add_argument("--print_features", action="store_true")
    ph.add_argument("--save_json", type=str, default=None)
    ph.add_argument("--save_pkl", type=str, default=None)

    ph.add_argument("--plot_dir", type=str, default=None)
    ph.add_argument("--plot_metric", type=str, default="sMAPE", choices=["MAE", "RMSE", "sMAPE"])

    args = ap.parse_args()
    mode = args.mode

    df = load_processed(mode=mode)
    df_train, df_test = split_time_series(
        df=df,
        mode=mode,
        split_mode=args.split_mode,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        test_size=args.test_size,
    )

    y_col = "y"
    all_features = [c for c in df_train.columns if c != y_col]
    forced = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_holiday", "y_lag1", "y_lag24", "y_lag168"]

    models = _as_list_csv(args.models)
    strategies = _as_list_csv(args.strategies)
    topk_list = _as_int_list_csv(args.topk)
    rank_steps = _as_int_list_csv(args.rank_steps)

    ranked = rank_features(
        df_train=df_train,
        feature_cols=all_features,
        y_col=y_col,
        rank_steps=rank_steps,
        n_estimators=args.rank_n_estimators,
        max_rows=args.rank_max_rows,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )

    if args.print_features:
        f_forced = [f for f in forced if f in all_features]
        print(f"\n[INFO] Forced features used ({len(f_forced)}): {f_forced}")
        print(f"[INFO] Ranked features (top {min(30, len(ranked))}):")
        for i, (name, imp) in enumerate(ranked[:30], start=1):
            print(f"  {i:>2}. {name:<35s} imp={imp:.6f}")

    feature_sets = _build_feature_sets(
        all_features=all_features,
        ranked=ranked,
        forced=forced,
        topk_list=topk_list,
        strategies=strategies,
    )

    rng = np.random.default_rng(args.seed)
    if len(df_train) > args.train_max_rows:
        idx = rng.choice(len(df_train), size=args.train_max_rows, replace=False)
        df_train_used = df_train.iloc[idx].sort_index()
    else:
        df_train_used = df_train

    results: List[Dict[str, object]] = []

    for (strategy, k), feats in feature_sets.items():
        X_train, y_train = make_xy(df_train_used, feats, y_col=y_col)

        for mname in models:
            rec: Dict[str, object] = {
                "model": mname.upper(),
                "strategy": strategy,
                "k": (int(k) if k is not None else None),
                "n_features": int(len(feats)),
                "n_train": int(len(df_train_used)),
                "n_origins": 0,
                "MAE": float("nan"),
                "RMSE": float("nan"),
                "sMAPE": float("nan"),
                "MAE_std": float("nan"),
                "RMSE_std": float("nan"),
                "sMAPE_std": float("nan"),
            }
            try:
                model = _get_model(mname, seed=args.seed, n_jobs=args.n_jobs)
                model.fit(X_train, y_train)

                mean_m, std_m, n_orig = eval_multi_origin_openloop(
                    model=model,
                    df_full=df,
                    test_index=df_test.index,
                    feature_cols=feats,
                    horizon=int(args.horizon),
                    origin_stride=int(args.origin_stride),
                    y_col=y_col,
                )
                rec["n_origins"] = int(n_orig)
                rec.update(mean_m)
                rec.update(std_m)
            except Exception as e:
                print(f"[WARN] {mname.upper()} {strategy}{'' if k is None else k} failed: {e}")

            results.append(rec)

    df_res = pd.DataFrame(results)
    metric = args.plot_metric
    df_res = df_res.sort_values(metric, ascending=True).reset_index(drop=True)

    print("\n📌 FEATURE STRATEGY COMPARE (OPENLOOP) [hourly]")
    print(f"[INFO] eval_mode=multi_origin | origin_stride={args.origin_stride}h | horizon={args.horizon}")
    print(f"[INFO] split_mode={args.split_mode} | train_end={args.train_end} | test_start={args.test_start}")
    print(f"[INFO] models={models} | strategies={strategies} | topk={topk_list}")
    print(f"[INFO] rank_steps={rank_steps} | rank_n_estimators={args.rank_n_estimators}")

    cols = ["model", "strategy", "k", "n_features", "n_train", "n_origins", "MAE", "RMSE", "sMAPE", "MAE_std", "RMSE_std", "sMAPE_std"]
    print("\n=== RESULTS (sorted by {}) ===".format(metric))
    print(df_res[cols].to_string(index=False))

    df_best = df_res.sort_values([metric]).groupby("model", as_index=False).first()
    print("\nBEST per model (by {}):".format(metric))
    print(df_best[cols].to_string(index=False))

    if args.save_pkl:
        Path(args.save_pkl).parent.mkdir(parents=True, exist_ok=True)
        df_res.to_pickle(args.save_pkl)

    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "mode": mode,
            "loop": "openloop",
            "split": {
                "split_mode": args.split_mode,
                "train_start": args.train_start,
                "train_end": args.train_end,
                "test_start": args.test_start,
                "test_end": args.test_end,
                "test_size": args.test_size,
                "n_train": int(len(df_train)),
                "n_test": int(len(df_test)),
                "data_min": str(df.index.min()),
                "data_max": str(df.index.max()),
            },
            "params": {
                "horizon": int(args.horizon),
                "origin_stride": int(args.origin_stride),
                "models": models,
                "strategies": strategies,
                "topk": topk_list,
                "rank_steps": rank_steps,
                "rank_n_estimators": int(args.rank_n_estimators),
                "rank_max_rows": int(args.rank_max_rows),
                "train_max_rows": int(args.train_max_rows),
                "seed": int(args.seed),
            },
            "forced_features": [f for f in forced if f in all_features],
            "ranked_features": [{"name": n, "importance": imp} for n, imp in ranked],
            "feature_sets": {
                ("all" if kk is None else f"{s}{kk}"): feats
                for (s, kk), feats in feature_sets.items()
            },
            "results": [{k: _to_jsonable(v) for k, v in r.items()} for r in results],
            "best_per_model": [{k: _to_jsonable(v) for k, v in r.items()} for r in df_best.to_dict(orient="records")],
        }
        Path(args.save_json).write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_dir = _safe_mkdir(args.plot_dir)
    if plot_dir is not None:
        import matplotlib.pyplot as plt

        topn = min(3, len(df_res))
        df_top = df_res.head(topn).copy()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.barh(range(len(df_top)), df_top[metric].to_numpy(dtype=float))
        ax.set_yticks(range(len(df_top)))
        ax.set_yticklabels(
            [f"{r.model}-{r.strategy}{'' if pd.isna(r.k) else int(r.k)}" for r in df_top.itertuples(index=False)]
        )
        ax.invert_yaxis()
        ax.set_xlabel(metric)
        ax.set_title(f"Top {topn} configs (lower is better)")
        fig.tight_layout()
        fig.savefig(plot_dir / f"top{topn}_{metric}.png", dpi=160)
        plt.close(fig)

        for model_name, g in df_res.groupby("model"):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for strat, gg in g.groupby("strategy"):
                ax.plot(
                    gg["n_features"].to_numpy(dtype=int),
                    gg[metric].to_numpy(dtype=float),
                    marker="o",
                    linestyle="-",
                    label=strat,
                )
            ax.set_xlabel("n_features")
            ax.set_ylabel(metric)
            ax.set_title(f"{model_name}: {metric} vs n_features")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / f"{model_name}_{metric}_by_n_features.png", dpi=160)
            plt.close(fig)


if __name__ == "__main__":
    main()
