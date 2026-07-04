"""
make_ensemble.py — Ensemble (mean/median) πάνω σε forecast JSONs του master pipeline.

Δέχεται 2+ JSON αρχεία (dashboard schema: dates/actual/series/metrics) που αφορούν το
ΙΔΙΟ task/test window — π.χ. lgbm/xgb/mlp από run_master_grid ή master_forecast — και:
  1. ευθυγραμμίζει τα timestamps όλων των μελών,
  2. ξανα-υπολογίζει MAE κάθε μέλους στο ΚΟΙΝΟ παράθυρο (complete-case: μόνο ώρες
     όπου υπάρχουν ΟΛΑ τα μέλη + actual) ώστε η σύγκριση να είναι δίκαιη,
  3. βγάζει ensemble_mean / ensemble_median + metrics,
  4. (προαιρετικά) γράφει ενιαίο dashboard-συμβατό JSON με όλα τα series.

Βιβλιογραφία (Lago et al. 2021): ο απλός μέσος όρος ανόμοιων μοντέλων κερδίζει
συστηματικά κάθε μεμονωμένο μοντέλο — χωρίς κανένα tuning.

Παράδειγμα:
  python -m src.make_ensemble \
      master_grid_out/lgbm_price_dam_recursive_static.json \
      master_grid_out/xgb_price_dam_recursive_static.json \
      master_grid_out/mlp_price_dam_recursive_static.json \
      --out_json master_grid_out/ensemble_price_dam.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .master_forecast import _metrics

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]


def _load(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    dates = pd.to_datetime(d["dates"])
    actual = pd.Series([np.nan if v is None else float(v) for v in d.get("actual", [])],
                       index=dates, dtype=float)
    members = {}
    for name, vals in d.get("series", {}).items():
        members[name] = pd.Series([np.nan if v is None else float(v) for v in vals],
                                  index=dates, dtype=float)
    return d, actual, members


def _ser2json(s: pd.Series):
    return [None if not np.isfinite(v) else round(float(v), 4) for v in s.to_numpy()]


def main():
    p = argparse.ArgumentParser(description="Ensemble mean/median από forecast JSONs")
    p.add_argument("jsons", nargs="+", help="2+ forecast JSON (dashboard schema)")
    p.add_argument("--out_json", default=None, help="πού να γραφτεί το ενιαίο JSON")
    args = p.parse_args()

    paths = [Path(x) if Path(x).is_absolute() else BASE_DIR / x for x in args.jsons]
    for pth in paths:
        if not pth.exists():
            sys.exit(f"❌ Δεν βρέθηκε: {pth}")

    meta0 = None
    acts, all_members = [], {}
    for pth in paths:
        d, actual, members = _load(pth)
        if meta0 is None:
            meta0 = d
        acts.append(actual)
        for k, v in members.items():
            if k in all_members:  # ίδια ονόματα από διαφορετικά αρχεία
                k = f"{k}__{pth.stem}"
            all_members[k] = v

    if len(all_members) < 2:
        sys.exit("❌ Χρειάζονται τουλάχιστον 2 series για ensemble.")

    # actual: συνένωση + έλεγχος συνέπειας μεταξύ αρχείων
    A = pd.concat(acts, axis=1)
    span = (A.max(axis=1) - A.min(axis=1)).abs()
    n_bad = int((span > 1e-6).sum())
    if n_bad:
        print(f"⚠️ actual mismatch σε {n_bad} timestamps μεταξύ αρχείων — κρατώ το πρώτο μη-κενό")
    actual = A.bfill(axis=1).iloc[:, 0]

    M = pd.DataFrame(all_members).sort_index()
    actual = actual.reindex(M.index)

    # complete-case: ώρες με ΟΛΑ τα μέλη + actual → δίκαιη σύγκριση
    full = M.notna().all(axis=1) & actual.notna()
    n_full = int(full.sum())
    if n_full == 0:
        sys.exit("❌ Κανένα κοινό timestamp με όλα τα μέλη + actual — είναι ίδιο test window;")
    Mc, ac = M.loc[full], actual.loc[full]

    ens_mean = Mc.mean(axis=1)
    ens_median = Mc.median(axis=1)

    unit = meta0.get("unit", "")
    print(f"🎯 ENSEMBLE | {len(all_members)} μέλη | κοινό παράθυρο: {n_full} ώρες "
          f"({Mc.index[0]} → {Mc.index[-1]})")
    results = []
    for name, s in list(Mc.items()) + [("ensemble_mean", ens_mean), ("ensemble_median", ens_median)]:
        met = _metrics(ac.to_numpy(), s.to_numpy())
        results.append((name, met))
    results.sort(key=lambda r: r[1]["MAE"])

    print("\n" + "=" * 72)
    print(f"{'model':<44} {'MAE':>10} {'RMSE':>10} {'sMAPE%':>8}")
    print("-" * 72)
    for name, met in results:
        star = " ★" if name.startswith("ensemble") else ""
        print(f"{name:<44} {met['MAE']:>10.3f} {met['RMSE']:>10.3f} {met['sMAPE']:>8.2f}{star}")
    print("=" * 72)
    best_member = min((r for r in results if not r[0].startswith("ensemble")),
                      key=lambda r: r[1]["MAE"])
    for ens_name in ("ensemble_mean", "ensemble_median"):
        met = dict(results)[ens_name]
        d = met["MAE"] - best_member[1]["MAE"]
        verdict = "κερδίζει" if d < 0 else "χάνει από"
        print(f"   {ens_name}: {verdict} το καλύτερο μέλος ({best_member[0]}) κατά {abs(d):.3f} {unit}")

    if args.out_json:
        outp = Path(args.out_json) if Path(args.out_json).is_absolute() else BASE_DIR / args.out_json
        series = {name: _ser2json(M[name].loc[full]) for name in M.columns}
        series["ensemble_mean"] = _ser2json(ens_mean)
        series["ensemble_median"] = _ser2json(ens_median)
        out = {
            "strategy": "ensemble", "task": meta0.get("task"), "market": meta0.get("market"),
            "gate": meta0.get("gate"), "horizon": meta0.get("horizon"),
            "stride": meta0.get("stride"), "retrain": meta0.get("retrain"),
            "members": list(M.columns), "unit": unit,
            "dates": [t.isoformat() for t in Mc.index],
            "actual": _ser2json(ac),
            "series": series,
            "metrics": [dict(Model=name, Type="ml", MAE=met["MAE"],
                             RMSE=met["RMSE"], sMAPE=met["sMAPE"]) for name, met in results],
        }
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        print(f"\n💾 {outp}")


if __name__ == "__main__":
    main()
