"""
master_forecast.py — Ενιαία μηχανή πρόβλεψης για τον energy-trading agent.

Μία μηχανή, ίδια λογική διαθεσιμότητας για ΟΛΟΥΣ τους αλγόριθμους → fair
comparison, καμία per-algo διαρροή. Παραμετροποιεί ρητά:

  (Α) Ορίζοντα        --horizon H            (+ presets --market)
  (Β) Delay/gate      --delay / --gate       (0=επόμενη, 12=DAM κενό φορτίου)
  (Γ) Retrain         --retrain {static,monthly,weekly}
  (Δ) Ablation        --features "lags,calendar,forecast,meteo,fuel,dense,roll,crosslags"  --ss
  (Ε) Χρονικό διάστ.  --train_* / --test_*

Στρατηγικές (όλες leakage-free):
  recursive : anchor στο cutoff, autoregressive rollout (καλύπτει το κενό delay)
  direct    : 1 μοντέλο ανά offset από features@cutoff (ποτέ within-horizon actual)
  tf        : teacher-forced 1-step — ΔΙΑΓΝΩΣΤΙΚΟ oracle (με προειδοποίηση)
  seq2seq   : true MIMO (μόνο lstm — βλ. επόμενο build)

Χρήση:
  python -m src.master_forecast --algo lgbm --task price --market dam \
      --retrain static --train_end "2025-11-30 23:00" \
      --test_start "2025-12-01 00:00" --test_end "2025-12-31 23:00" \
      --features default --out_json out.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .split_utils import load_processed, make_xy
from .recursive_openloop import OpenLoopConfig, recursive_predict_openloop
from .feature_availability import (
    GateSpec,
    describe_gate,
    parse_feature_spec,
    select_features,
    classify_columns,
    detect_crosslag_cols,
    freeze_crosslags_for_gate,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

# ----------------------------------------------------------------------------
# Market presets → (delay-gap semantics via gate, horizon, stride)
# ----------------------------------------------------------------------------
MARKET_PRESETS = {
    #        horizon, stride  (delay/gap υπολογίζεται από GateSpec ανά task)
    "dam":     dict(horizon=24,  stride=24),
    "idm":     dict(horizon=6,   stride=3),
    "forward": dict(horizon=168, stride=24),
}


# ----------------------------------------------------------------------------
# Dense intraday lags (on-the-fly) — προστίθενται στο df ΠΡΙΝ το feature select
# ----------------------------------------------------------------------------
def add_dense_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Προσθέτει y_lag4..y_lag23 όπου λείπουν (shift στην πλήρη y series)."""
    df = df.copy()
    y = df["y"].astype(float)
    for lag in range(1, 24):
        col = f"y_lag{lag}"
        if col not in df.columns:
            df[col] = y.shift(lag)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineered day-ahead features (ομάδα 'engfc'), υπολογίσιμα από υπάρχουσες στήλες:
      resload_fc = load_fc − solar_fc_dayahead − wind_onshore_fc_dayahead
    (forecast residual load — όλα τα μέλη είναι day-ahead-known ⇒ leakage-free)
    """
    df = df.copy()
    need = ("load_fc", "solar_fc_dayahead", "wind_onshore_fc_dayahead")
    if all(c in df.columns for c in need) and "resload_fc" not in df.columns:
        df["resload_fc"] = (
            df["load_fc"].astype(float)
            - df["solar_fc_dayahead"].astype(float).fillna(0.0)
            - df["wind_onshore_fc_dayahead"].astype(float).fillna(0.0)
        )
    return df


# ----------------------------------------------------------------------------
# Model builders
# ----------------------------------------------------------------------------
def build_and_fit(algo: str, X: pd.DataFrame, y: np.ndarray, *, seed: int = 42,
                  n_estimators: Optional[int] = None,
                  objective: Optional[str] = None,
                  quantile_alpha: Optional[float] = None):
    """Επιστρέφει fitted μοντέλο με .predict(DataFrame).

    objective/quantile_alpha: μόνο για lgbm (conformal quantile-LGBM comparison,
    βλ. src/conformal.py) — αγνοούνται (None) διατηρούν την παλιά συμπεριφορά.
    """
    algo = algo.lower()
    if algo == "lgbm":
        import lightgbm as lgb
        kwargs = dict(
            n_estimators=n_estimators or 800, learning_rate=0.03, num_leaves=64,
            subsample=0.9, colsample_bytree=0.8, min_child_samples=40,
            reg_alpha=0.0, reg_lambda=0.0, random_state=seed, n_jobs=-1,
        )
        if objective == "quantile":
            kwargs["objective"] = "quantile"
            kwargs["alpha"] = float(quantile_alpha)
        m = lgb.LGBMRegressor(**kwargs)
        m.fit(X, y)
        return m
    if algo == "xgb":
        import xgboost as xgb
        m = xgb.XGBRegressor(
            n_estimators=n_estimators or 800, learning_rate=0.03, max_depth=8,
            subsample=0.9, colsample_bytree=0.8, min_child_weight=5,
            reg_alpha=0.0, reg_lambda=1.0, random_state=seed, n_jobs=-1,
            tree_method="hist",
        )
        m.fit(X, y)
        return m
    if algo == "mlp":
        from .train_mlp import TorchMLPRegressor
        m = TorchMLPRegressor(
            hidden=(256, 128), dropout=0.10, lr=1e-3, weight_decay=1e-4,
            batch_size=1024, max_epochs=200, patience=20, seed=seed, device="auto",
        )
        # μικρό validation tail για early stopping
        n = len(X)
        n_val = max(1, min(720, n // 10))
        Xtr, ytr = X.iloc[:-n_val], y[:-n_val]
        Xval, yval = X.iloc[-n_val:], y[-n_val:]
        m.fit(Xtr, ytr, X_val=Xval, y_val=yval, verbose=False)
        return m
    if algo == "lear":
        # LEAR (Lago et al. 2021): standardised features + LASSO, CV-selected penalty.
        from sklearn.linear_model import LassoCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(
            StandardScaler(),
            LassoCV(n_alphas=100, cv=5, max_iter=20000, n_jobs=-1, random_state=seed),
        )
        m.fit(X, y)
        return m
    if algo == "lstm":
        raise NotImplementedError("lstm builder προστίθεται στο επόμενο build (seq2seq/recursive).")
    raise ValueError(f"Unknown algo: {algo}")


# ----------------------------------------------------------------------------
# Direct multi-horizon (1 sub-model ανά offset) — leakage-free (origin=cutoff)
# ----------------------------------------------------------------------------
def _make_direct_xy(df_fit: pd.DataFrame, feature_cols: List[str], max_offset: int,
                    crosslag_mode: str = "freeze", crosslag_protect_cols: Optional[set] = None):
    """
    X = features@t ; Y = [y(t+1..t+max_offset)].
    Ο caller έχει ήδη περάσει `df_fit` από freeze_crosslags_for_gate (§4.8), οπότε
    οι crosslag στήλες που ήταν εκτός cutoff είναι ήδη frozen/NaN. Εδώ μόνο
    προσέχουμε να ΜΗΝ γεμίσουμε με 0.0 τα crosslag NaN στο NaN-sensitivity mode
    (τα δέντρα χρειάζονται πραγματικό NaN για native missing-value handling).
    """
    X = df_fit[feature_cols].select_dtypes(include=[np.number]).copy()
    y = df_fit["y"].astype(float)
    Ys = [y.shift(-o).rename(f"y_t+{o}") for o in range(1, max_offset + 1)]
    Y = pd.concat(Ys, axis=1)
    joined = X.join(Y, how="inner").dropna(subset=list(Y.columns))
    Xa = joined[X.columns].copy()
    if crosslag_mode == "nan" and crosslag_protect_cols:
        protect = [c for c in crosslag_protect_cols if c in Xa.columns]
        fill_cols = [c for c in Xa.columns if c not in protect]
        Xa[fill_cols] = Xa[fill_cols].fillna(0.0)
    else:
        Xa = Xa.fillna(0.0)
    Ya = joined[list(Y.columns)].to_numpy(dtype=float)
    return Xa, Ya


def fit_direct(algo: str, df_fit: pd.DataFrame, feature_cols: List[str],
               max_offset: int, *, seed: int = 42,
               crosslag_mode: str = "freeze", crosslag_protect_cols: Optional[set] = None):
    from sklearn.multioutput import MultiOutputRegressor
    Xa, Ya = _make_direct_xy(df_fit, feature_cols, max_offset,
                             crosslag_mode=crosslag_mode, crosslag_protect_cols=crosslag_protect_cols)
    algo = algo.lower()
    if algo == "lgbm":
        import lightgbm as lgb
        base = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=48,
                                 subsample=0.9, colsample_bytree=0.8, random_state=seed, n_jobs=1)
    elif algo == "xgb":
        import xgboost as xgb
        base = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=7,
                                subsample=0.9, colsample_bytree=0.8, random_state=seed,
                                n_jobs=1, tree_method="hist")
    elif algo == "lear":
        from sklearn.linear_model import LassoCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        # λιγότερα alphas/folds εδώ: τρέχει 1 φορά ανά offset (π.χ. 24x για DAM)
        base = make_pipeline(
            StandardScaler(),
            LassoCV(n_alphas=50, cv=3, max_iter=20000, n_jobs=1, random_state=seed),
        )
    else:
        raise NotImplementedError(f"direct για algo={algo} προστίθεται αργότερα")
    import joblib
    with joblib.parallel_backend("threading"):
        model = MultiOutputRegressor(base, n_jobs=-1).fit(Xa, Ya)
    model._feature_cols = feature_cols  # type: ignore
    return model


# ----------------------------------------------------------------------------
# Block generation
# ----------------------------------------------------------------------------
def make_blocks(test_start: pd.Timestamp, test_end: pd.Timestamp,
                horizon: int, stride: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Λίστα (block_start, block_end_inclusive) καλύπτοντας το [test_start,test_end]."""
    blocks = []
    cur = test_start
    while cur <= test_end:
        b_end = min(cur + pd.Timedelta(hours=horizon - 1), test_end)
        blocks.append((cur, b_end))
        cur = cur + pd.Timedelta(hours=stride)
    return blocks


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[m], y_pred[m]
    if len(yt) == 0:
        return dict(MAE=float("nan"), RMSE=float("nan"), sMAPE=float("nan"), n=0)
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    denom = (np.abs(yt) + np.abs(yp))
    smape = float(np.mean(np.where(denom > 1e-9, 2.0 * np.abs(yt - yp) / denom, 0.0)) * 100.0)
    return dict(MAE=round(mae, 4), RMSE=round(rmse, 4), sMAPE=round(smape, 4), n=int(len(yt)))


def _make_xy_crosslag_aware(dtr: pd.DataFrame, crosslag_mode: str, crosslag_protect_cols: set):
    """
    Σαν split_utils.make_xy, αλλά στο crosslag_mode='nan' ΔΕΝ κάνει ffill ούτε
    απορρίπτει γραμμές λόγω NaN στις crosslag στήλες (§4.8 NaN-sensitivity
    variant) — τα δέντρα βλέπουν πραγματικό NaN εκεί και το χειρίζονται native.
    Χωρίς NaN mode (ή χωρίς crosslag στήλες) ταυτίζεται 100% με το make_xy.
    """
    if crosslag_mode != "nan" or not crosslag_protect_cols:
        return make_xy(dtr)

    y = pd.to_numeric(dtr["y"], errors="coerce")
    X = dtr.drop(columns=["y"], errors="ignore").select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    protect = [c for c in crosslag_protect_cols if c in X.columns]
    fillable = [c for c in X.columns if c not in protect]
    if fillable:
        X[fillable] = X[fillable].ffill()

    good = np.isfinite(y.to_numpy())
    if fillable:
        good = good & np.all(np.isfinite(X[fillable].to_numpy()), axis=1)

    X = X.loc[good]
    y = y.loc[good].to_numpy(dtype=float)
    return X, y


# ----------------------------------------------------------------------------
# Core rollout
# ----------------------------------------------------------------------------
def run_forecast(
    *,
    algo: str,
    task: str,
    strategy: str,
    gate: GateSpec,
    horizon: int,
    stride: int,
    retrain: str,
    df: pd.DataFrame,
    feature_cols: List[str],
    train_start: Optional[pd.Timestamp],
    train_end: Optional[pd.Timestamp],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    seed: int = 42,
    verbose: bool = True,
    ss: bool = False,
    ss_decay: str = "linear",
    ss_rounds: int = 3,
    n_estimators: Optional[int] = None,
    crosslag_mode: str = "freeze",
) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """
    Επιστρέφει (scored_index, y_true, y_pred).
    Ο retrain scheduler ξαναχτίζει το μοντέλο πριν από blocks όταν χρειάζεται,
    ΠΑΝΤΑ με train_end ≤ cutoff (καμία διαρροή στο μέλλον).

    crosslag_mode ('freeze'|'nan'): AEL (§4.8) — πώς αντιμετωπίζονται τα
    gen_solar/gen_wind/residual_load/load lags που πέφτουν ΜΕΤΑ το cutoff_F της
    δικής τους οικογένειας (πριν ΔΕΝ ίσχυε κανένα enforcement εκτός των y-lags).
    'freeze' (default, deployable) = τελευταία γνωστή τιμή· 'nan' = sensitivity
    variant μόνο για δέντρα (native missing-value handling).
    """
    blocks = make_blocks(test_start, test_end, horizon, stride)
    y_full = df["y"].astype(float)

    # AEL (§4.8): σύνολο crosslag στηλών (gen/load actuals) μέσα στα επιλεγμένα
    # feature_cols — χρησιμοποιείται ΚΑΙ στο training row build ΚΑΙ στο direct eval.
    crosslag_protect_cols: set = set()
    for colmap in detect_crosslag_cols(feature_cols).values():
        crosslag_protect_cols.update(colmap.keys())

    # per-scored-hour: κρατάμε την ΠΙΟ πρόσφατη πρόβλεψη (freshest anchor)
    pred_map: dict = {}

    model = None
    model_offset_max = 0
    last_train_key = None

    def _train_key(cutoff: pd.Timestamp) -> str:
        if retrain == "static":
            return "static"
        if retrain == "monthly":
            return f"{cutoff.year}-{cutoff.month:02d}"
        if retrain == "weekly":
            iso = cutoff.isocalendar()
            return f"{iso[0]}-W{int(iso[1]):02d}"
        return "static"

    # max offset needed: από cutoff έως τελευταία ώρα block = gap + horizon
    _last_off = gate.gap_hours() + horizon

    # future-known columns για το LSTM (calendar/forecast/meteo/fuel)
    _fut_cols = None
    if algo == "lstm":
        g = classify_columns([c for c in df.columns if c != "y"])
        futset = set(g["calendar"] + g["resfc"] + g["loadfc"] + g["meteo"] + g["fuel"])
        _fut_cols = [c for c in feature_cols if c in futset] or \
                    [c for c in feature_cols if c in set(g["calendar"])]

    def _fit_at(cutoff: pd.Timestamp):
        # static: σεβόμαστε το --train_end cap.
        # rolling retrain (monthly/weekly): expanding window έως το cutoff —
        # αλλιώς κάθε refit ξανατρέχει σε παγωμένα δεδομένα και είναι no-op.
        if retrain == "static" and train_end is not None:
            te = min(train_end, cutoff)
        else:
            te = cutoff
        dtr = df.loc[:te]
        if train_start is not None:
            dtr = dtr.loc[train_start:]
        # AEL (§4.8): ΚΑΘΕ training row παγώνει τα crosslag lags της με το ΙΔΙΟ
        # σχήμα που θα ισχύσει στο serve (train/serve συνέπεια, VALIDITY Β1):
        #  - recursive/tf/lstm: η γραμμή t σερβίρεται μέσα στο block της ημέρας
        #    της → cutoff = day(t)-anchor (default του freeze_crosslags_for_gate).
        #  - direct: η γραμμή t είναι ORIGIN (row@cutoff) → στο serve το origin
        #    23:00 D-1 έχει cutoff 11:00 D-1 = t − crosslag_gap → ίδιος τύπος
        #    για κάθε training origin.
        # tf ΔΕΝ εξαιρείται: ο oracle-χαρακτήρας του αφορά μόνο το eval-time
        # teacher forcing· το μοντέλο του εκπαιδεύεται ΚΙ αυτό leak-free.
        if strategy == "direct":
            _cl_cut = dtr.index - pd.Timedelta(hours=gate.crosslag_gap_hours())
            dtr = freeze_crosslags_for_gate(dtr, feature_cols, gate, df_full=df,
                                            mode=crosslag_mode, cutoffs=_cl_cut)
        else:
            dtr = freeze_crosslags_for_gate(dtr, feature_cols, gate, df_full=df,
                                            mode=crosslag_mode)
        if algo == "lstm":
            from .lstm_models import Seq2SeqLSTM
            m = Seq2SeqLSTM(future_cols=_fut_cols, L=168, H=int(_last_off),
                            hidden=64, layers=1, epochs=30, batch_size=256,
                            seed=seed, device="cpu",
                            recursive_1step=(strategy == "recursive"))
            m.fit(dtr)
            return m, int(_last_off)
        if strategy == "direct":
            max_off = int(_last_off)
            return fit_direct(algo, dtr, feature_cols, max_off, seed=seed,
                              crosslag_mode=crosslag_mode,
                              crosslag_protect_cols=crosslag_protect_cols), max_off
        # recursive / tf: single-output
        Xtr, ytr = _make_xy_crosslag_aware(dtr, crosslag_mode, crosslag_protect_cols)
        Xtr = Xtr[feature_cols]
        if ss and strategy == "recursive":
            from .scheduled_sampling import fit_with_scheduled_sampling
            m = fit_with_scheduled_sampling(
                Xtr, ytr,
                lambda Xc, yc: build_and_fit(algo, Xc, yc, seed=seed, n_estimators=n_estimators),
                rounds=ss_rounds, decay=ss_decay, seed=seed, verbose=verbose,
            )
            return m, 0
        return build_and_fit(algo, Xtr, ytr, seed=seed, n_estimators=n_estimators), 0

    t_start = time.time()
    for bi, (b0, b1) in enumerate(blocks):
        cutoff = gate.cutoff_for_block(b0)
        # AEL (§4.8): ΕΝΑ crosslag cutoff για όλο το block (anchor-based) —
        # σε multi-day blocks (forward) τίποτα μετά το issue time δεν είναι γνωστό.
        cl_cutoff = gate.crosslag_cutoff_for_anchor(b0)
        # scored hours αυτού του block
        scored_idx = pd.date_range(b0, b1, freq="H")
        # πλήρες rollout: από cutoff+1 έως b1 (καλύπτει κενό + block)
        roll_idx = pd.date_range(cutoff + pd.Timedelta(hours=1), b1, freq="H")
        roll_idx = roll_idx.intersection(df.index)
        if len(roll_idx) == 0:
            continue

        # retrain αν άλλαξε το key
        key = _train_key(cutoff)
        if model is None or key != last_train_key:
            model, model_offset_max = _fit_at(cutoff)
            last_train_key = key
            if verbose:
                print(f"   [fit] key={key} cutoff={cutoff}  (block {bi+1}/{len(blocks)})", flush=True)

        if algo == "lstm":
            if cutoff not in df.index:
                continue
            vec = model.predict_block(df, cutoff, int(model_offset_max))
            for t in scored_idx:
                off = int((t - cutoff) / pd.Timedelta(hours=1))
                if 1 <= off <= len(vec):
                    pred_map[t] = float(vec[off - 1])
        elif strategy in ("recursive", "tf"):
            if strategy == "tf":
                # ΔΙΑΓΝΩΣΤΙΚΟ: teacher-forced — actual lags παντού (oracle).
                Xb = df.loc[scored_idx, feature_cols]
                yb = np.asarray(model.predict(Xb), dtype=float).reshape(-1)
                for t, v in zip(scored_idx, yb):
                    pred_map[t] = v
            else:
                preds = recursive_predict_openloop(
                    model=model, df_full=df, test_index=roll_idx,
                    feature_cols=feature_cols, config=OpenLoopConfig(y_floor=None),
                    gate=gate, crosslag_mode=crosslag_mode, crosslag_cutoff=cl_cutoff,
                )
                for t, v in zip(roll_idx, preds):
                    if t in set(scored_idx):
                        pred_map[t] = float(v)
        elif strategy == "direct":
            # origin = cutoff· row@cutoff → διάνυσμα offsets 1..model_offset_max
            if cutoff not in df.index:
                continue
            row = df.loc[[cutoff], feature_cols].select_dtypes(include=[np.number])
            # AEL (§4.8): το row@cutoff «βλέπει» crosslag actuals ΜΕΤΑ το δικό
            # τους cutoff_F (π.χ. price DAM: cutoff_y=23:00 D-1 αλλά τα gen/load
            # actuals κόβονται στις 11:00 D-1) — freeze/NaN πριν το predict,
            # με το anchor cutoff του block (ίδιο με το training origin σχήμα).
            row = freeze_crosslags_for_gate(row, feature_cols, gate, df_full=df,
                                            mode=crosslag_mode, fillna_other=0.0,
                                            cutoffs=pd.DatetimeIndex([cl_cutoff]))
            vec = np.asarray(model.predict(row), dtype=float).reshape(-1)
            for t in scored_idx:
                off = int((t - cutoff) / pd.Timedelta(hours=1))
                if 1 <= off <= len(vec):
                    pred_map[t] = float(vec[off - 1])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    if verbose:
        print(f"   [done] {len(blocks)} blocks σε {time.time()-t_start:.1f}s", flush=True)

    scored_all = pd.DatetimeIndex(sorted(pred_map.keys()))
    scored_all = scored_all.intersection(df.index)
    y_true = y_full.reindex(scored_all).to_numpy(dtype=float)
    y_pred = np.array([pred_map[t] for t in scored_all], dtype=float)
    return scored_all, y_true, y_pred


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Master forecast engine (energy trading agent)")
    p.add_argument("--algo", required=True, choices=["lgbm", "xgb", "mlp", "lstm", "lear"])
    p.add_argument("--task", required=True, choices=["price", "load"])
    p.add_argument("--mode", default="hourly", choices=["hourly"])
    p.add_argument("--strategy", default="recursive", choices=["recursive", "direct", "tf", "seq2seq"])
    p.add_argument("--ss", action="store_true", help="scheduled sampling (μόνο recursive)")
    p.add_argument("--ss_decay", default="linear", choices=["linear", "exp", "step"],
                   help="σχήμα μείωσης ε για SS")
    p.add_argument("--ss_rounds", type=int, default=3, help="rounds self-generated retraining")
    p.add_argument("--n_estimators", type=int, default=None,
                   help="override n_estimators για lgbm/xgb (E1: capacity×features)")
    p.add_argument("--seed", type=int, default=42, help="random seed (robustness checks)")
    p.add_argument("--market", default="dam", choices=["dam", "idm", "forward", "custom"])
    p.add_argument("--delay", type=int, default=None, help="uniform gap override (0=next,12=DAM-load)")
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--gate", default="strict", choices=["strict", "academic"])
    p.add_argument("--retrain", default="static", choices=["static", "monthly", "weekly"])
    p.add_argument("--train_start", type=str, default=None)
    p.add_argument("--train_end", type=str, default=None)
    p.add_argument("--test_start", type=str, required=True)
    p.add_argument("--test_end", type=str, required=True)
    p.add_argument("--features", type=str, default="default")
    p.add_argument("--crosslag_mode", default="freeze", choices=["freeze", "nan"],
                   help="AEL §4.8: μεταχείριση gen/load actual lags εκτός cutoff_F — "
                        "'freeze' (default, deployable) ή 'nan' (sensitivity, μόνο lgbm/xgb)")
    p.add_argument("--out_json", type=str, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    verbose = not args.quiet

    if args.crosslag_mode == "nan" and args.algo not in ("lgbm", "xgb"):
        raise SystemExit(f"--crosslag_mode nan υποστηρίζεται μόνο με lgbm/xgb (native NaN "
                          f"handling) — όχι με algo={args.algo}.")

    # market → horizon/stride
    if args.market in MARKET_PRESETS:
        horizon = args.horizon or MARKET_PRESETS[args.market]["horizon"]
        stride = args.stride or MARKET_PRESETS[args.market]["stride"]
    else:
        if args.horizon is None or args.stride is None:
            raise SystemExit("Για --market custom δώσε --horizon και --stride.")
        horizon, stride = args.horizon, args.stride

    gate = GateSpec(task=args.task, gate=args.gate, market=args.market, delay_override=args.delay)

    # data + dense lags / engineered features (αν ζητηθούν)
    df = load_processed(args.mode, task=args.task)
    groups = parse_feature_spec(args.features)
    if "dense" in groups:
        df = add_dense_lags(df)
    if "engfc" in groups:
        df = add_engineered_features(df)
    df = df.dropna(subset=["y"]).sort_index()

    all_cols = [c for c in df.columns if c != "y"]
    feature_cols = select_features(all_cols, groups)
    # μόνο αριθμητικά
    feature_cols = [c for c in feature_cols if np.issubdtype(df[c].dtype, np.number)]

    ts0 = pd.to_datetime(args.test_start)
    ts1 = pd.to_datetime(args.test_end)
    tr0 = pd.to_datetime(args.train_start) if args.train_start else None
    tr1 = pd.to_datetime(args.train_end) if args.train_end else None

    if verbose:
        print("=" * 78)
        print(f"MASTER FORECAST | algo={args.algo} task={args.task} strategy={args.strategy}")
        print(f"  {describe_gate(gate)} | horizon={horizon} stride={stride} retrain={args.retrain}")
        print(f"  groups={groups}  (#features={len(feature_cols)})")
        print(f"  test=[{ts0} .. {ts1}]  train_end={tr1}")
        print("=" * 78)
        if args.strategy == "tf":
            print("  ⚠️  strategy=tf είναι ΔΙΑΓΝΩΣΤΙΚΟ oracle (actual lags) — ΟΧΙ tradeable!")
        if args.retrain != "static" and tr1 is not None:
            print(f"  ℹ️  retrain={args.retrain}: το --train_end αγνοείται — "
                  "expanding window έως κάθε retrain cutoff (φρέσκα δεδομένα).")
        print(f"  AEL crosslag_mode={args.crosslag_mode} "
              f"(crosslag_gap={gate.crosslag_gap_hours()}h)")

    # seq2seq/recursive για LSTM → και τα δύο υλοποιούνται μέσω Seq2SeqLSTM
    if args.strategy == "seq2seq" and args.algo != "lstm":
        raise SystemExit("--strategy seq2seq υποστηρίζεται μόνο με --algo lstm.")

    scored_idx, y_true, y_pred = run_forecast(
        algo=args.algo, task=args.task, strategy=args.strategy, gate=gate,
        horizon=horizon, stride=stride, retrain=args.retrain, df=df,
        feature_cols=feature_cols, train_start=tr0, train_end=tr1,
        test_start=ts0, test_end=ts1, verbose=verbose, seed=args.seed,
        ss=args.ss, ss_decay=args.ss_decay, ss_rounds=args.ss_rounds,
        n_estimators=args.n_estimators, crosslag_mode=args.crosslag_mode,
    )

    met = _metrics(y_true, y_pred)
    unit = "€/MWh" if args.task == "price" else "MW"
    print(f"\n📊 RESULT | MAE={met['MAE']} {unit} | RMSE={met['RMSE']} | sMAPE={met['sMAPE']}% | n={met['n']}")

    if args.out_json:
        label = f"{args.algo.upper()}-{args.strategy}-{args.market}"
        out = {
            "strategy": args.strategy, "task": args.task, "market": args.market,
            "gate": args.gate, "delay_gap": gate.gap_hours(), "horizon": horizon,
            "stride": stride, "retrain": args.retrain, "features": groups,
            "crosslag_mode": args.crosslag_mode, "crosslag_gap": gate.crosslag_gap_hours(),
            "unit": unit,
            "dates": [t.isoformat() for t in scored_idx],
            "actual": [None if not np.isfinite(v) else round(float(v), 4) for v in y_true],
            "series": {label: [None if not np.isfinite(v) else round(float(v), 4) for v in y_pred]},
            "metrics": [dict(Model=label, Type="ml", **{k: met[k] for k in ("MAE", "RMSE", "sMAPE")})],
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        print(f"💾 saved: {args.out_json}")


if __name__ == "__main__":
    main()
