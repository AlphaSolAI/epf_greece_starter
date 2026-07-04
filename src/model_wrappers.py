from typing import Union

import numpy as np
import pandas as pd


class ResidualAddBaselineWrapper:
    """
    Wrap a regressor trained to predict residuals (e.g., Δy),
    and return level predictions:
        y_hat = baseline(X) + residual_hat

    baseline is taken from a feature column (default: y_lag1).
    Stored here so joblib can unpickle consistently.
    """

    def __init__(self, model, baseline_col: str, feature_names: list):
        self.model = model
        self.baseline_col = baseline_col
        self.feature_names = list(feature_names)
        if baseline_col not in self.feature_names:
            raise ValueError(
                f"baseline_col='{baseline_col}' not found in feature_names. "
                f"Available columns: {self.feature_names[:20]}..."
            )
        self.baseline_idx = self.feature_names.index(baseline_col)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]):
        residual_hat = np.asarray(self.model.predict(X), dtype=float).reshape(-1)

        if isinstance(X, pd.DataFrame):
            base = X[self.baseline_col].to_numpy(dtype=float).reshape(-1)
        else:
            X = np.asarray(X)
            base = X[:, self.baseline_idx].astype(float).reshape(-1)

        return base + residual_hat
