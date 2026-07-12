"""
lstm_models.py — LSTM για το master pipeline (το DL κομμάτι).

Δύο εκδοχές, και οι δύο leakage-free:

  Seq2SeqLSTM  (true MIMO)
    Encoder LSTM διαβάζει τις τελευταίες L ώρες [y + future-known exogenous].
    Decoder LSTM βγάζει ΟΛΟ το διάνυσμα των H μελλοντικών ωρών ταυτόχρονα,
    τροφοδοτούμενος με (α) τα future-known exogenous κάθε βήματος (calendar,
    day-ahead forecasts, meteo, fuel) και (β) autoregressively το προηγούμενο y_hat.
    → ΕΝΑ μοντέλο, διανυσματική έξοδος, κοινή αναπαράσταση = πραγματικό MIMO.

  RecursiveLSTM (1-step)
    Ίδιος encoder, έξοδος 1 βήμα· rollout autoregressive για οποιονδήποτε ορίζοντα.

Και οι δύο ΔΕΝ χρησιμοποιούν engineered lag columns — χτίζουν τη δική τους
ακολουθία από την ωμή σειρά y + τα future-known exogenous. Το decoder βλέπει
ΜΟΝΟ exogenous που είναι διαθέσιμα day-ahead (calendar/forecast/meteo/fuel),
άρα καμία διαρροή actual εντός ορίζοντα.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:
    _TORCH = False


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    if _TORCH:
        torch.manual_seed(seed)


class _Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)

    def forward(self, x):
        _, (h, c) = self.lstm(x)
        return h, c


class _Decoder(nn.Module):
    def __init__(self, exo_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        # decoder input ανά βήμα = [prev_y (1)] + [future exogenous (exo_dim)]
        self.lstm = nn.LSTM(1 + exo_dim, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward_step(self, prev_y, exo_step, h, c):
        # prev_y:(B,1) exo_step:(B,exo_dim)
        inp = torch.cat([prev_y, exo_step], dim=1).unsqueeze(1)  # (B,1,1+exo)
        out, (h, c) = self.lstm(inp, (h, c))
        y = self.head(out.squeeze(1))  # (B,1)
        return y, h, c


class Seq2SeqLSTM:
    """
    sklearn-ish wrapper. Χρήση:
        m = Seq2SeqLSTM(future_cols=[...], L=168, H=36, ...)
        m.fit(df_train)                       # df με 'y' + future_cols
        vec = m.predict_block(df_full, cutoff, n_steps)   # np.array μήκους n_steps
    """
    def __init__(self, future_cols: List[str], *, L: int = 168, H: int = 36,
                 hidden: int = 64, layers: int = 1, dropout: float = 0.1,
                 lr: float = 1e-3, epochs: int = 30, batch_size: int = 256,
                 patience: int = 6, seed: int = 42, device: str = "cpu",
                 recursive_1step: bool = False):
        if not _TORCH:
            raise RuntimeError("PyTorch δεν είναι διαθέσιμο.")
        self.future_cols = list(future_cols)
        self.L, self.H = int(L), int(H)
        self.hidden, self.layers, self.dropout = hidden, layers, dropout
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.patience, self.seed = patience, seed
        self.device = device
        self.recursive_1step = recursive_1step
        self.y_mean_ = self.y_std_ = None
        self.x_mean_ = self.x_std_ = None
        self.enc_ = self.dec_ = None

    # ---- scaling ----
    def _scale_y(self, y):
        return (y - self.y_mean_) / (self.y_std_ + 1e-8)

    def _unscale_y(self, y):
        return y * (self.y_std_ + 1e-8) + self.y_mean_

    def _scale_x(self, X):
        return (X - self.x_mean_) / (self.x_std_ + 1e-8)

    # ---- build training windows ----
    def _make_windows(self, y: np.ndarray, Xf: np.ndarray):
        """
        Encoder input: [y, Xf] για ώρες [t-L .. t-1]  (L βήματα)
        Decoder targets: y[t .. t+H-1]     ·  Decoder exo: Xf[t .. t+H-1]
        """
        L, H = self.L, self.H
        n = len(y)
        enc_list, dec_exo_list, dec_y_list = [], [], []
        for t in range(L, n - H + 1):
            enc = np.concatenate([y[t - L:t, None], Xf[t - L:t]], axis=1)  # (L,1+F)
            enc_list.append(enc)
            dec_exo_list.append(Xf[t:t + H])       # (H,F)
            dec_y_list.append(y[t:t + H])          # (H,)
        if not enc_list:
            raise ValueError("Λίγα δεδομένα για τα windows (μείωσε L/H).")
        return (np.asarray(enc_list, dtype=np.float32),
                np.asarray(dec_exo_list, dtype=np.float32),
                np.asarray(dec_y_list, dtype=np.float32))

    def fit(self, df_train: pd.DataFrame):
        _set_seed(self.seed)
        dev = torch.device(self.device)
        y = df_train["y"].astype(float).to_numpy()
        Xf = df_train[self.future_cols].astype(float).fillna(0.0).to_numpy()

        self.y_mean_, self.y_std_ = float(np.nanmean(y)), float(np.nanstd(y))
        self.x_mean_ = np.nanmean(Xf, axis=0)
        self.x_std_ = np.nanstd(Xf, axis=0)

        ys = self._scale_y(y)
        Xs = self._scale_x(Xf)

        enc, dec_exo, dec_y = self._make_windows(ys, Xs)
        enc_t = torch.tensor(enc, device=dev)
        dexo_t = torch.tensor(dec_exo, device=dev)
        dy_t = torch.tensor(dec_y, device=dev)

        F = Xs.shape[1]
        self.enc_ = _Encoder(1 + F, self.hidden, self.layers, self.dropout).to(dev)
        self.dec_ = _Decoder(F, self.hidden, self.layers, self.dropout).to(dev)
        opt = torch.optim.Adam(list(self.enc_.parameters()) + list(self.dec_.parameters()),
                               lr=self.lr, weight_decay=1e-5)
        lossf = nn.L1Loss()

        N = enc_t.shape[0]
        n_val = max(1, N // 10)
        idx = np.arange(N)
        tr_idx, va_idx = idx[:-n_val], idx[-n_val:]
        best_val, best_state, bad = np.inf, None, 0

        # FIX (2026-07-12, ABLATION §7.7/§7.17 debug): predict_block ΠΑΝΤΑ free-runs
        # πολυβηματικά (autoregressive) ασχέτως strategy (master_forecast.py:457 ελέγχει
        # algo=="lstm" ΠΡΙΝ το strategy branch) — άρα Hused=1 (recursive_1step) εκπαίδευε
        # σε task που ΔΕΝ αντιστοιχεί στο deployment rollout (exposure bias, bias +41 στο
        # §5.5). Hused ΠΑΝΤΑ = self.H ώστε train να ταιριάζει με το πραγματικό rollout·
        # recursive_1step διατηρείται στο API (backward compat) αλλά δεν επηρεάζει πια Hused.
        Hused = self.H

        for ep in range(self.epochs):
            self.enc_.train(); self.dec_.train()
            # scheduled sampling (γραμμική decay): epoch 0 = 100% teacher forcing,
            # τελευταίο epoch = 0% — κλείνει το train/inference χάσμα σταδιακά
            # (ίδιο πνεύμα με scheduled_sampling.py για LGBM/XGB).
            tf_prob = 1.0 - ep / max(1, self.epochs - 1)
            perm = np.random.permutation(tr_idx)
            for s in range(0, len(perm), self.batch_size):
                b = perm[s:s + self.batch_size]
                eb, xb, yb = enc_t[b], dexo_t[b], dy_t[b]
                h, c = self.enc_(eb)
                prev = eb[:, -1, 0:1]  # τελευταίο y του encoder
                loss = 0.0
                for k in range(Hused):
                    yhat, h, c = self.dec_.forward_step(prev, xb[:, k, :], h, c)
                    loss = loss + lossf(yhat.squeeze(1), yb[:, k])
                    use_tf = np.random.rand() < tf_prob
                    prev = yb[:, k:k + 1] if use_tf else yhat.detach()
                loss = loss / Hused
                opt.zero_grad(); loss.backward(); opt.step()

            # validation
            self.enc_.eval(); self.dec_.eval()
            with torch.no_grad():
                eb, xb, yb = enc_t[va_idx], dexo_t[va_idx], dy_t[va_idx]
                h, c = self.enc_(eb)
                prev = eb[:, -1, 0:1]
                vloss = 0.0
                for k in range(Hused):
                    yhat, h, c = self.dec_.forward_step(prev, xb[:, k, :], h, c)
                    vloss = vloss + lossf(yhat.squeeze(1), yb[:, k]).item()
                    prev = yhat  # autoregressive στο val
                vloss /= Hused
            if vloss < best_val - 1e-5:
                best_val, bad = vloss, 0
                best_state = ({k: v.detach().clone() for k, v in self.enc_.state_dict().items()},
                              {k: v.detach().clone() for k, v in self.dec_.state_dict().items()})
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.enc_.load_state_dict(best_state[0])
            self.dec_.load_state_dict(best_state[1])
        return self

    def predict_block(self, df_full: pd.DataFrame, cutoff: pd.Timestamp, n_steps: int) -> np.ndarray:
        """Προβλέπει n_steps ώρες μετά το cutoff (autoregressive). Leakage-free."""
        dev = torch.device(self.device)
        self.enc_.eval(); self.dec_.eval()

        enc_idx = pd.date_range(cutoff - pd.Timedelta(hours=self.L - 1), cutoff, freq="H")
        enc_idx = enc_idx.intersection(df_full.index)
        if len(enc_idx) < 2:
            return np.full(n_steps, np.nan)
        y_enc = self._scale_y(df_full.loc[enc_idx, "y"].astype(float).to_numpy())
        Xf_enc = self._scale_x(df_full.loc[enc_idx, self.future_cols].astype(float).fillna(0.0).to_numpy())
        enc = np.concatenate([y_enc[:, None], Xf_enc], axis=1)[None, :, :].astype(np.float32)
        enc_t = torch.tensor(enc, device=dev)

        fut_idx = pd.date_range(cutoff + pd.Timedelta(hours=1), periods=n_steps, freq="H")
        fut_idx_in = fut_idx.intersection(df_full.index)
        Xf_fut = np.zeros((n_steps, len(self.future_cols)), dtype=np.float32)
        if len(fut_idx_in) > 0:
            vals = self._scale_x(df_full.loc[fut_idx_in, self.future_cols].astype(float).fillna(0.0).to_numpy())
            pos = [fut_idx.get_loc(t) for t in fut_idx_in]
            Xf_fut[pos] = vals.astype(np.float32)
        xb = torch.tensor(Xf_fut[None, :, :], device=dev)

        with torch.no_grad():
            h, c = self.enc_(enc_t)
            prev = enc_t[:, -1, 0:1]
            outs = []
            for k in range(n_steps):
                yhat, h, c = self.dec_.forward_step(prev, xb[:, k, :], h, c)
                outs.append(float(yhat.squeeze().cpu().numpy()))
                prev = yhat
        return self._unscale_y(np.asarray(outs, dtype=float))
