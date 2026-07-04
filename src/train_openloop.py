import argparse
from typing import List, Optional

from .train_lgbm_openloop import train as train_lgbm_openloop
from .train_mlp_openloop import train as train_mlp_openloop
from .train_rf_openloop import train as train_rf_openloop
from .train_svr_openloop import train as train_svr_openloop
from .train_xgb_openloop import train as train_xgb_openloop


def _parse_models(s: str) -> List[str]:
    s = (s or "all").strip().lower()
    if s == "all":
        return ["svr", "mlp", "xgb", "rf", "lgbm"]
    return [m.strip() for m in s.split(",") if m.strip()]


def train_all_openloop(
    mode: str,
    task: str,
    models: List[str],
    train_start: Optional[str],
    train_end: Optional[str],
    device: str,
) -> None:
    trainers = {
        "svr": train_svr_openloop,
        "mlp": train_mlp_openloop,
        "xgb": train_xgb_openloop,
        "rf": train_rf_openloop,
        "lgbm": train_lgbm_openloop,
    }

    if mode == "both":
        modes = ["daily", "hourly"]
    else:
        modes = [mode]

    for md in modes:
        for m in models:
            if m not in trainers:
                raise ValueError(f"Unknown model id: {m}. Use one of {list(trainers.keys())} or 'all'.")
            if m == "xgb":
                trainers[m](md, task=task, train_start=train_start, train_end=train_end, device=("cuda" if device == "auto" else device))
            elif m == "mlp":
                trainers[m](md, task=task, train_start=train_start, train_end=train_end, device=device)
            else:
                trainers[m](md, task=task, train_start=train_start, train_end=train_end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["daily", "hourly", "both"])
    parser.add_argument("--task", choices=["price", "load"], default="price")
    parser.add_argument("--models", type=str, default="all", help="Comma list: svr,mlp,xgb,rf,lgbm or all")
    parser.add_argument("--train_start", type=str, default=None)
    parser.add_argument("--train_end", type=str, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Used by MLP(torch) and XGB (cuda/cpu)")
    args = parser.parse_args()

    models = _parse_models(args.models)
    train_all_openloop(
        mode=args.mode,
        task=args.task,
        models=models,
        train_start=args.train_start,
        train_end=args.train_end,
        device=args.device,
    )


if __name__ == "__main__":
    main()
