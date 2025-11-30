import pandas as pd

def naive_forecast(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """
    Naive forecast: predict tomorrow = yesterday's value.
    """
    last_value = train["y"].iloc[-1]
    return pd.Series([last_value] * len(test), index=test.index)
