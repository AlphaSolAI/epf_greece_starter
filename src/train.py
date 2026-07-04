import argparse
import inspect
from typing import List, Optional

from .train_lgbm import train as train_lgbm
from .train_mlp import train as train_mlp
from .train_rf import train as train_rf
from .train_svr import train as train_svr
from .train_xgb import train as train_xgb


def _parse_models(s: str) -> List[str]:
    s = (s or "all").strip().lower()
    if s == "all":
        return ["svr", "mlp", "xgb", "rf", "lgbm"]
    return [m.strip() for m in s.split(",") if m.strip()]


def _call_trainer(
    fn,
    *,
    mode: str,
    task: str,
    test_size: Optional[int],
    train_start: Optional[str],
    train_end: Optional[str],
    test_start: Optional[str],
    test_end: Optional[str],
) -> None:
    """
    Call a trainer in a backwards-compatible way:
    - If trainer supports 'task', pass it.
    - Else call without it.
    """
    sig = inspect.signature(fn)
    kwargs = dict(
        mode=mode,
        test_size=test_size,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )
    if "task" in sig.parameters:
        kwargs["task"] = task

    fn(**kwargs)


def train_all(
    *,
    mode: str,
    task: str,
    models: List[str],
    test_size: Optional[int],
    train_start: Optional[str],
    train_end: Optional[str],
    test_start: Optional[str],
    test_end: Optional[str],
) -> None:
    trainers = {
        "svr": train_svr,
        "mlp": train_mlp,
        "xgb": train_xgb,
        "rf": train_rf,
        "lgbm": train_lgbm,
    }

    modes = ["daily", "hourly"] if mode == "both" else [mode]

    for md in modes:
        for m in models:
            if m not in trainers:
                raise ValueError(f"Unknown model id: {m}. Use one of {list(trainers.keys())} or 'all'.")

            _call_trainer(
                trainers[m],
                mode=md,
                task=task,
                test_size=test_size,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["daily", "hourly", "both"])
    parser.add_argument("--task", choices=["price", "load"], default="price")
    parser.add_argument("--models", type=str, default="all", help="Comma list: svr,mlp,xgb,rf,lgbm or all")
    parser.add_argument("--test_size", type=int, default=None, help="Rows in test (daily=days, hourly=hours)")
    parser.add_argument("--train_start", type=str, default=None)
    parser.add_argument("--train_end", type=str, default=None)
    parser.add_argument("--test_start", type=str, default=None)
    parser.add_argument("--test_end", type=str, default=None)
    args = parser.parse_args()

    models = _parse_models(args.models)
    train_all(
        mode=args.mode,
        task=args.task,
        models=models,
        test_size=args.test_size,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )


if __name__ == "__main__":
    main()
