import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def _read_parquet_any(path: Path) -> pd.DataFrame:
    # Prefer fastparquet if present (pyarrow is not installed in your env)
    try:
        return pd.read_parquet(path, engine="fastparquet")
    except Exception:
        return pd.read_parquet(path)


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    for c in ["ds", "timestamp", "date", "datetime", "time"]:
        if c in df.columns:
            out = df.copy()
            out[c] = pd.to_datetime(out[c], errors="coerce")
            out = out.dropna(subset=[c]).set_index(c)
            return out.sort_index()
    raise ValueError("Processed df has no DatetimeIndex and no ds/timestamp/date/datetime column.")


def _pick_processed_path(mode: str, task: Optional[str]) -> Path:
    mode = str(mode).lower().strip()
    task = str(task).lower().strip() if task is not None else None

    candidates = []
    if mode == "hourly":
        if task == "load":
            candidates += ["hourly_load.parquet", "load_hourly.parquet", "hourly_loads.parquet"]
        elif task == "price":
            candidates += ["hourly_price.parquet", "price_hourly.parquet"]
        candidates += ["hourly.parquet"]
    elif mode == "daily":
        if task == "load":
            candidates += ["daily_load.parquet", "load_daily.parquet", "daily_loads.parquet"]
        elif task == "price":
            candidates += ["daily_price.parquet", "price_daily.parquet"]
        candidates += ["daily.parquet"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    for fn in candidates:
        p = PROCESSED_DIR / fn
        if p.exists():
            return p
    raise FileNotFoundError(f"No processed parquet found for mode={mode}, task={task}. Tried: {candidates}")


def load_processed(mode: str, task: Optional[str] = None) -> pd.DataFrame:
    """
    Load processed dataset.
    - Expects column 'y' already present (your data_future creates it).
    - task is used only to choose the file (hourly.parquet vs hourly_load.parquet).
    """
    path = _pick_processed_path(mode, task)
    df = _read_parquet_any(path)
    df = _ensure_datetime_index(df)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Basic cleanup
    df = df.replace([np.inf, -np.inf], np.nan)

    if "y" not in df.columns:
        raise ValueError(f"Processed parquet {path.name} must contain column 'y'.")

    return df


def split_time_series(
    df: pd.DataFrame,
    mode: str,
    test_size: Optional[int] = None,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    test_start: Optional[str] = None,
    test_end: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split. Supports:
      A) explicit test window: test_start/test_end (recommended)
      B) tail test_size
    """
    df = df.sort_index()

    if test_start is not None and test_end is not None:
        ts0 = pd.to_datetime(test_start)
        ts1 = pd.to_datetime(test_end)
        df_test = df.loc[ts0:ts1]

        # train: everything strictly before test unless train_end provided
        if train_end is not None:
            te = pd.to_datetime(train_end)
            df_train = df.loc[:te]
        else:
            # strict before first test point
            df_train = df.loc[df.index < ts0]

        if train_start is not None:
            tr0 = pd.to_datetime(train_start)
            df_train = df_train.loc[tr0:]

        # final safety: no overlap
        if len(df_train) and len(df_test):
            df_train = df_train.loc[df_train.index < df_test.index.min()]

        return df_train, df_test

    if test_size is None:
        raise ValueError("Provide either (test_start,test_end) or test_size.")

    test_size = int(test_size)
    if test_size <= 0 or test_size >= len(df):
        raise ValueError(f"Bad test_size={test_size} for n={len(df)}")

    df_test = df.iloc[-test_size:]
    df_train = df.iloc[:-test_size]

    if train_start is not None:
        df_train = df_train.loc[pd.to_datetime(train_start):]
    if train_end is not None:
        df_train = df_train.loc[:pd.to_datetime(train_end)]

    return df_train, df_test


def make_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Returns:
      X: numeric DataFrame (ffill only, no bfill)
      y: float ndarray aligned to X index
    """
    if "y" not in df.columns:
        raise ValueError("make_xy expects column 'y'.")

    y = pd.to_numeric(df["y"], errors="coerce")
    X = df.drop(columns=["y"], errors="ignore")

    # numeric only (as in your pipeline)
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan).ffill()

    # drop rows where y or any X is NaN
    good = np.isfinite(y.to_numpy())
    if len(X.columns) > 0:
        good = good & np.all(np.isfinite(X.to_numpy()), axis=1)

    X = X.loc[good]
    y = y.loc[good].to_numpy(dtype=float)
    return X, y
