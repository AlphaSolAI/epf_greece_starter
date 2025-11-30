import pandas as pd

def seasonal_naive(train: pd.DataFrame, test: pd.DataFrame, period: int = 7) -> pd.Series:
    """
    Seasonal naive: use value from N days ago.
    """
    preds = []
    for i in range(len(test)):
        preds.append(train["y"].iloc[-period + i])
    return pd.Series(preds, index=test.index)
