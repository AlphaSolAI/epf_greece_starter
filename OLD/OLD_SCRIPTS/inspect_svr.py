import sys
sys.stdout.reconfigure(encoding="utf-8")

from src.split_utils import load_processed, make_xy, split_time_series

for task in ["price", "load"]:
    df = load_processed("hourly", task=task)
    train, test = split_time_series(
        df, mode="hourly",
        train_start="2022-01-01",
        train_end="2025-11-30 23:00",
        test_start="2025-12-01 00:00",
        test_end="2025-12-07 23:00"
    )
    train_all, _ = split_time_series(
        df, mode="hourly",
        train_end="2025-11-30 23:00",
        test_start="2025-12-01 00:00",
        test_end="2025-12-07 23:00"
    )
    X, y = make_xy(train)
    print(f"\n=== {task.upper()} ===")
    print(f"SVR train rows (from 2022): {len(train)}")
    print(f"LGBM train rows (all data): {len(train_all)}")
    print(f"Features: {X.shape[1]}")
    print(f"y range: {y.min():.2f} .. {y.max():.2f}")
    print(f"Feature names:\n  {list(X.columns)}")
