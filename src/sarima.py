import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarima_forecast(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """
    SARIMA baseline model.
    """
    model = SARIMAX(train["y"],
                    order=(1,1,1),
                    seasonal_order=(1,0,1,7),
                    enforce_stationarity=False,
                    enforce_invertibility=False)

    res = model.fit(disp=False)

    pred = res.predict(start=test.index[0], end=test.index[-1])
    return pred
