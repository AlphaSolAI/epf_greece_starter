"""Fast, pure-logic unit tests for src/split_utils.py (no conda/training needed).

    conda run -n epf --no-capture-output python -X utf8 -m pytest tests/ -q

Includes a regression test for a silent-fallback bug found during a
2026-07-06 code review: _pick_processed_path(task="load") used to fall
through to the generic "hourly.parquet" (the PRICE file) whenever none of
the load-specific filenames existed, instead of failing loudly. That would
silently train/evaluate a "load" run against price targets. Fixed in
src/split_utils.py by only allowing the generic fallback for task in
{None, "price"}; this test locks the fix in.
"""
import numpy as np
import pandas as pd
import pytest

from src import split_utils
from src.split_utils import make_xy, split_time_series


def test_pick_processed_path_load_task_does_not_fall_back_to_generic(tmp_path, monkeypatch):
    monkeypatch.setattr(split_utils, "PROCESSED_DIR", tmp_path)
    # only the generic (price) file exists — no hourly_load*/load_hourly* variant
    (tmp_path / "hourly.parquet").write_text("not a real parquet, path existence only")

    with pytest.raises(FileNotFoundError):
        split_utils._pick_processed_path("hourly", "load")


def test_pick_processed_path_price_task_falls_back_to_generic(tmp_path, monkeypatch):
    monkeypatch.setattr(split_utils, "PROCESSED_DIR", tmp_path)
    (tmp_path / "hourly.parquet").write_text("placeholder")

    # price (and None) SHOULD legitimately fall back to hourly.parquet
    assert split_utils._pick_processed_path("hourly", "price") == tmp_path / "hourly.parquet"
    assert split_utils._pick_processed_path("hourly", None) == tmp_path / "hourly.parquet"


def test_pick_processed_path_load_task_prefers_specific_file(tmp_path, monkeypatch):
    monkeypatch.setattr(split_utils, "PROCESSED_DIR", tmp_path)
    (tmp_path / "hourly.parquet").write_text("price data")
    (tmp_path / "hourly_load.parquet").write_text("load data")

    assert split_utils._pick_processed_path("hourly", "load") == tmp_path / "hourly_load.parquet"


def _toy_df(n=48, start="2026-01-01 00:00", freq="h"):
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame({"y": np.arange(n, dtype=float), "x1": np.arange(n, dtype=float) * 2}, index=idx)


def test_split_time_series_explicit_window_no_overlap():
    df = _toy_df(n=72)  # 3 days hourly
    train, test = split_time_series(
        df, mode="hourly",
        test_start="2026-01-03 00:00", test_end="2026-01-03 23:00",
    )
    assert train.index.max() < test.index.min()
    assert test.index.min() == pd.Timestamp("2026-01-03 00:00")
    assert test.index.max() == pd.Timestamp("2026-01-03 23:00")


def test_split_time_series_train_end_still_excludes_test_overlap():
    """Even if train_end is given past the test start (misconfiguration),
    the function must not hand back overlapping train/test rows."""
    df = _toy_df(n=72)
    train, test = split_time_series(
        df, mode="hourly",
        train_end="2026-01-03 12:00",  # deliberately overlaps the test window below
        test_start="2026-01-03 00:00", test_end="2026-01-03 23:00",
    )
    assert train.index.max() < test.index.min()


def test_split_time_series_tail_test_size():
    df = _toy_df(n=72)
    train, test = split_time_series(df, mode="hourly", test_size=24)
    assert len(test) == 24
    assert len(train) == 48
    assert train.index.max() < test.index.min()


def test_split_time_series_requires_window_or_size():
    df = _toy_df(n=10)
    with pytest.raises(ValueError):
        split_time_series(df, mode="hourly")


def test_make_xy_drops_nan_rows_and_non_numeric_cols():
    df = _toy_df(n=5)
    df["y"] = [1.0, np.nan, 3.0, 4.0, 5.0]
    df["label"] = ["a", "b", "c", "d", "e"]  # non-numeric, must be dropped from X
    X, y = make_xy(df)
    assert "label" not in X.columns
    assert len(y) == 4  # the NaN-y row is dropped
    assert np.isfinite(y).all()


def test_make_xy_requires_y_column():
    df = pd.DataFrame({"x1": [1.0, 2.0]})
    with pytest.raises(ValueError):
        make_xy(df)
