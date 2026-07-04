import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _is_list_of_dicts(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(i, dict) for i in x)


def _safe_float(x: Any) -> float:
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if not s:
        return np.nan
    s = s.replace("%", "").strip()
    try:
        return float(s)
    except Exception:
        return np.nan


def _pick(d: Dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def _pick_by_contains(d: Dict[str, Any], needle: str) -> Any:
    needle = needle.lower()
    for k, v in d.items():
        if needle in str(k).lower():
            return v
    return None


def _extract_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if _is_list_of_dicts(payload.get("rows")):
        return payload["rows"]

    weeks = payload.get("weeks")
    if isinstance(weeks, dict):
        out: List[Dict[str, Any]] = []
        for _, wk_data in weeks.items():
            if isinstance(wk_data, dict) and _is_list_of_dicts(wk_data.get("rows")):
                out.extend(wk_data["rows"])
        if out:
            return out

    def looks_like_rows(lst: List[Dict[str, Any]]) -> bool:
        if not lst:
            return False
        score = 0
        for r in lst[: min(len(lst), 50)]:
            mk = _pick(r, ["model", "Model", "MODEL"])
            sk = _pick(r, ["sMAPE", "smape", "SMAPE"]) or _pick_by_contains(r, "smape")
            if mk is not None and sk is not None:
                score += 1
        return score >= max(1, int(0.3 * min(len(lst), 50)))

    def find(obj: Any) -> Optional[List[Dict[str, Any]]]:
        if _is_list_of_dicts(obj) and looks_like_rows(obj):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                got = find(v)
                if got is not None:
                    return got
        if isinstance(obj, list):
            for v in obj:
                got = find(v)
                if got is not None:
                    return got
        return None

    got = find(payload)
    return got or []


def _normalize_rows(rows: List[Dict[str, Any]], week_fallback: Optional[str]) -> pd.DataFrame:
    out = []
    for r in rows:
        model = _pick(r, ["model", "Model", "MODEL"])
        strategy = _pick(r, ["strategy", "Strategy", "STRATEGY"])
        week = _pick(r, ["week", "Week", "WEEK", "label", "week_label"]) or week_fallback

        smape = _pick(r, ["sMAPE", "smape", "SMAPE"]) or _pick_by_contains(r, "smape")
        k = _pick(r, ["k", "K", "topk", "top_k"])
        nfeat = _pick(r, ["n_features", "nfeat", "nFeatures", "features"])

        if model is None or strategy is None or week is None or smape is None:
            continue

        model_s = str(model).strip().upper()
        strat_s = str(strategy).strip().lower()

        if strat_s not in {"all", "topk", "hybrid"}:
            if strat_s in {"lags_roll"}:
                strat_s = "topk"
            elif strat_s in {"hybrid_topk"}:
                strat_s = "hybrid"
            else:
                continue

        smape_v = _safe_float(smape)

        k_v: Optional[int] = None
        if k is not None and str(k).strip() != "":
            try:
                k_v = int(float(k))
            except Exception:
                k_v = None

        nfeat_v: Optional[int] = None
        if nfeat is not None and str(nfeat).strip() != "":
            try:
                nfeat_v = int(float(nfeat))
            except Exception:
                nfeat_v = None

        out.append(
            {
                "week": str(week),
                "model": model_s,
                "strategy": strat_s,
                "k": k_v,
                "sMAPE": smape_v,
                "n_features": nfeat_v,
            }
        )

    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["sMAPE"] = pd.to_numeric(df["sMAPE"], errors="coerce")
    df = df.dropna(subset=["sMAPE"])
    return df


def _winner_per_week(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    g = df.sort_values(["model", "week", "strategy", "sMAPE"], ascending=[True, True, True, True])
    best_within_strategy = g.groupby(["model", "week", "strategy"], as_index=False).head(1)

    w = best_within_strategy.sort_values(["model", "week", "sMAPE"], ascending=[True, True, True])
    winner = w.groupby(["model", "week"], as_index=False).head(1).reset_index(drop=True)
    winner = winner.rename(
        columns={"strategy": "winner_strategy", "k": "winner_k", "sMAPE": "winner_sMAPE"}
    )
    return winner, best_within_strategy


def _dominant_per_model(best_within_strategy: pd.DataFrame, winner: pd.DataFrame) -> pd.DataFrame:
    strat_avg = (
        best_within_strategy.groupby(["model", "strategy"], as_index=False)
        .agg(avg_sMAPE=("sMAPE", "mean"), n_weeks=("week", "nunique"), avg_n_features=("n_features", "mean"))
    )

    wins = (
        winner.groupby(["model", "winner_strategy"], as_index=False)
        .size()
        .rename(columns={"winner_strategy": "strategy", "size": "wins"})
    )
    strat_avg = strat_avg.merge(wins, on=["model", "strategy"], how="left").fillna({"wins": 0})
    strat_avg["wins"] = strat_avg["wins"].astype(int)

    k_stats = (
        winner.dropna(subset=["winner_k"])
        .groupby(["model", "winner_strategy"], as_index=False)
        .agg(rec_k=("winner_k", lambda x: int(np.median(list(x))) if len(x) else np.nan))
        .rename(columns={"winner_strategy": "strategy"})
    )
    strat_avg = strat_avg.merge(k_stats, on=["model", "strategy"], how="left")

    strat_avg = strat_avg.sort_values(
        ["model", "avg_sMAPE", "wins", "avg_n_features"],
        ascending=[True, True, False, True],
    )
    dominant = strat_avg.groupby("model", as_index=False).head(1).reset_index(drop=True)
    return dominant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=r"models\feature_sets")
    ap.add_argument("--expected_weeks", type=str, default="weekA,weekB,weekC,weekD")
    ap.add_argument("--core_prefix", type=str, default="fs_hourly_H24_rank24_168_")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    expected_weeks = [w.strip() for w in args.expected_weeks.split(",") if w.strip()]
    core_re = re.compile(rf"^{re.escape(args.core_prefix)}week[A-Za-z0-9]+\.json$", re.IGNORECASE)

    all_files = sorted(root.rglob("*.json"))
    core_files = [p for p in all_files if core_re.match(p.name)]

    week_to_file: Dict[str, Path] = {}
    for p in core_files:
        m = re.search(r"(week[A-Za-z0-9]+)\.json$", p.name, re.IGNORECASE)
        if not m:
            continue
        wk = m.group(1)
        if wk not in week_to_file or p.stat().st_mtime > week_to_file[wk].stat().st_mtime:
            week_to_file[wk] = p

    found_weeks_files = sorted(week_to_file.keys())
    missing_files = [w for w in expected_weeks if w not in week_to_file]

    print("\n===== USING CORE FILES ONLY =====")
    print(f"[DIR] {root}")
    print(f"[CORE_PREFIX] {args.core_prefix}")
    print(f"[EXPECTED_WEEKS] {expected_weeks}")
    print(f"[FOUND_CORE_WEEKS] {found_weeks_files}")
    if missing_files:
        print(f"[MISSING_CORE_FILES] {missing_files}")
        print("ΔΕΝ ΜΠΟΡΩ ΝΑ ΒΓΑΛΩ 4-week ranking πριν φτιαχτούν αυτά τα core files.")
        return
    else:
        print("[MISSING_CORE_FILES] none")

    all_dfs = []
    weeks_missing_rows: List[str] = []

    print("\n===== CORE FILES =====")
    for wk in expected_weeks:
        p = week_to_file[wk]
        print(f"{wk}: {p.as_posix()}")
        payload = json.loads(p.read_text(encoding="utf-8"))
        rows = _extract_rows(payload)
        if not rows:
            weeks_missing_rows.append(wk)
            continue
        df_part = _normalize_rows(rows, week_fallback=wk)
        if df_part.empty:
            weeks_missing_rows.append(wk)
            continue
        df_part["source_file"] = p.name
        all_dfs.append(df_part)

    if weeks_missing_rows:
        print("\n===== ERROR =====")
        print("ΒΡΗΚΑ core files αλλά ΔΕΝ βρήκα metrics rows (model/strategy/sMAPE) για:")
        for w in weeks_missing_rows:
            print(f" - {w}: {week_to_file[w].name}")
        print("\nΛΥΣΗ: ξανατρέξε feature_strategy_compare1 για αυτές τις εβδομάδες ΜΕ patched save (να γράφει rows).")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["week", "model", "strategy", "k", "sMAPE", "n_features", "source_file"])

    print("\n===== DATA SUMMARY =====")
    print("Weeks in data:", sorted(df["week"].unique().tolist()))
    print("Models in data:", sorted(df["model"].unique().tolist()))
    print("Strategies in data:", sorted(df["strategy"].unique().tolist()))
    print("Rows:", len(df))

    winner, best_within_strategy = _winner_per_week(df)
    dominant = _dominant_per_model(best_within_strategy, winner)

    print("\n===== WINNER PER WEEK (per model) =====")
    show_w = winner.sort_values(["model", "week"])[["model", "week", "winner_strategy", "winner_k", "winner_sMAPE"]]
    print(show_w.to_string(index=False))

    print("\n===== FINAL DOMINANT (4-week objective) =====")
    dom = dominant.copy()
    if "rec_k" in dom.columns:
        dom["rec_k"] = dom["rec_k"].astype("Int64")
    cols = ["model", "strategy", "avg_sMAPE", "wins", "n_weeks", "rec_k", "avg_n_features"]
    for c in cols:
        if c not in dom.columns:
            dom[c] = np.nan
    print(dom[cols].sort_values(["avg_sMAPE", "wins"], ascending=[True, False]).to_string(index=False))

    print("\n===== TRAINING CHOICE (use this for MIMO) =====")
    for _, r in dom.iterrows():
        model = r["model"]
        strat = r["strategy"]
        rec_k = r.get("rec_k", np.nan)
        if strat == "all":
            print(f"- {model}: strategy=all (use ALL features)")
        elif strat in {"topk", "hybrid"}:
            if pd.isna(rec_k):
                print(f"- {model}: strategy={strat} (k unknown -> take k from per-week winners)")
            else:
                print(f"- {model}: strategy={strat}, k={int(rec_k)}")
        else:
            print(f"- {model}: strategy={strat}")


if __name__ == "__main__":
    main()
