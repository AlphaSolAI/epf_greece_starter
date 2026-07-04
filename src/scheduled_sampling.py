"""
scheduled_sampling.py — Scheduled Sampling για tree/tabular recursive μοντέλα.

Πρόβλημα (exposure bias): στο training τα y_lag* είναι actuals, στο recursive
inference είναι προβλέψεις → το μοντέλο δεν έχει μάθει να δουλεύει με τον δικό
του θόρυβο και τα λάθη συσσωρεύονται στα μακρινά offsets.

Λύση (Bengio et al. 2015, προσαρμοσμένο σε trees μέσω iterative self-generated
retraining, αφού τα δέντρα δεν εκπαιδεύονται per-epoch):

  round 0: fit σε καθαρά actuals (κλασικό).
  round r=1..R: πάρε τις one-step προβλέψεις ŷ(t) του μοντέλου του round r−1
    πάνω στο training set· φτιάξε corrupted αντίγραφο των features όπου κάθε
    y_lag_k(t) αντικαθίσταται από ŷ(t−k) με πιθανότητα 1−ε(r)· refit.
    Καθώς το ε μειώνεται, το μοντέλο βλέπει ολοένα πιο «ρεαλιστικά» (θορυβώδη)
    lags και μαθαίνει να στηρίζεται περισσότερο στα exogenous όπου πρέπει.

Σχήματα μείωσης ε (ε = πιθανότητα να ΚΡΑΤΗΘΕΙ το actual):
  linear : ε(r) = max(ε_min, 1 − r/R)
  exp    : ε(r) = max(ε_min, 0.5^r)
  step   : ε ∈ {0.75, 0.5, 0.25, ...} (γραμμικά κβαντισμένο στο [ε_min, 0.75])

Κόστος: (R+1) fits + R predicts ανά training (π.χ. R=3 ⇒ ~4× χρόνος fit).
"""
from __future__ import annotations

import re
from typing import Callable, List

import numpy as np
import pandas as pd

_LAG_RE = re.compile(r"y_lag(\d+)")


def epsilon_schedule(decay: str, r: int, rounds: int, eps_min: float = 0.25) -> float:
    """ε του round r (1-indexed). ε = πιθανότητα διατήρησης actual lag."""
    decay = decay.lower()
    if decay == "linear":
        return max(eps_min, 1.0 - r / rounds)
    if decay == "exp":
        return max(eps_min, 0.5 ** r)
    if decay == "step":
        steps = np.linspace(0.75, eps_min, rounds)
        return float(steps[min(r, rounds) - 1])
    raise ValueError(f"Άγνωστο ss_decay: {decay} (linear|exp|step)")


def fit_with_scheduled_sampling(
    X: pd.DataFrame,
    y: np.ndarray,
    build_fn: Callable[[pd.DataFrame, np.ndarray], object],
    *,
    rounds: int = 3,
    decay: str = "linear",
    eps_min: float = 0.25,
    seed: int = 42,
    verbose: bool = False,
):
    """
    X: training features (DatetimeIndex, περιέχει y_lag* στήλες), y: targets.
    build_fn(X, y) -> fitted μοντέλο με .predict(DataFrame).
    Επιστρέφει το μοντέλο του τελευταίου round.
    """
    lag_cols: List[str] = [c for c in X.columns if _LAG_RE.fullmatch(c)]
    model = build_fn(X, y)  # round 0: καθαρό
    if not lag_cols or rounds <= 0:
        return model

    idx = X.index
    for r in range(1, rounds + 1):
        eps = epsilon_schedule(decay, r, rounds, eps_min)
        # one-step self-predictions πάνω στο (καθαρό) training set
        yhat = pd.Series(np.asarray(model.predict(X), dtype=float).reshape(-1), index=idx)

        Xc = X.copy()
        rng = np.random.default_rng(seed * 1000 + r)
        for c in lag_cols:
            k = int(_LAG_RE.fullmatch(c).group(1))
            # ŷ(t−k): ευθυγράμμιση με shift στο timestamp
            pred_at_lag = yhat.reindex(idx - pd.Timedelta(hours=k))
            pred_vals = pred_at_lag.to_numpy()
            ok = np.isfinite(pred_vals)
            replace = (rng.random(len(idx)) > eps) & ok
            col_vals = Xc[c].to_numpy(dtype=float, copy=True)
            col_vals[replace] = pred_vals[replace]
            Xc[c] = col_vals

        model = build_fn(Xc, y)
        if verbose:
            frac = 1.0 - eps
            print(f"      [SS r{r}/{rounds}] ε={eps:.2f} (αντικατάσταση ~{frac:.0%} των lags)", flush=True)

    return model
