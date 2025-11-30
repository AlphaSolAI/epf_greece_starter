import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io
import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# UTF-8 για να μη σακατεύει τα ελληνικά
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

# Προσπαθούμε να φορτώσουμε pmdarima για auto_arima
try:
    import pmdarima as pm
    HAVE_PMDARIMA = True
    print("✅ pmdarima βρέθηκε – θα γίνει αυτόματο auto_arima.")
except ImportError:
    HAVE_PMDARIMA = False
    print("⚠️ pmdarima ΔΕΝ βρέθηκε – θα χρησιμοποιηθούν σταθερές παράμετροι ARIMA.")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "daily.parquet"

TEST_DAYS = 60

# Default παράμετροι (fallback αν δεν έχουμε pmdarima)
DEFAULT_ARIMA_ORDER = (3, 0, 2)
DEFAULT_SEASONAL_ORDER = (1, 1, 1, 7)


def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return (
        np.mean(
            2.0 * np.abs(y_pred - y_true)
            / (np.abs(y_pred) + np.abs(y_true) + 1e-9)
        )
        * 100.0
    )


def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    # Χρησιμοποιούμε ** χωρίς squared=False για να παίζει με παλιά sklearn
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    s = smape(y_true, y_pred)
    return {"model": name, "MAE": mae, "RMSE": rmse, "sMAPE": s}


def choose_arima_params(y_train: pd.Series):
    """
    Επιλέγει αυτόματα ARIMA/SARIMA παραμέτρους με auto_arima.
    Αν δεν υπάρχει pmdarima, επιστρέφει τις default.
    """
    if not HAVE_PMDARIMA:
        print("\n🔧 Χρήση default ARIMA παραμέτρων "
              f"{DEFAULT_ARIMA_ORDER}, {DEFAULT_SEASONAL_ORDER}")
        return DEFAULT_ARIMA_ORDER, DEFAULT_SEASONAL_ORDER

    print("\n🔍 Τρέχει auto_arima στο TRAIN set...")
    print(f"   Δείγματα train: {len(y_train)}, seasonality m=7 (εβδομάδα)")
    print("   [Μπορεί να πάρει λίγα δευτερόλεπτα]\n")

    model = pm.auto_arima(
        y_train,
        start_p=1,
        start_q=1,
        max_p=3,
        max_q=3,
        m=7,                 # εβδομαδιαία εποχικότητα
        start_P=0,
        seasonal=True,
        d=None,
        D=1,
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )

    print("\n✅ auto_arima ολοκληρώθηκε.")
    print("   Βέλτιστο order:", model.order)
    print("   Βέλτιστο seasonal_order:", model.seasonal_order)

    return model.order, model.seasonal_order


def main():
    if not DATA_PATH.exists():
        print(f"❌ Δεν βρέθηκε το {DATA_PATH}")
        return

    df = pd.read_parquet(DATA_PATH)

    # Βεβαιωνόμαστε ότι το index είναι datetime με ημερήσια συχνότητα
    if not isinstance(df.index, pd.DatetimeIndex):
        if "ds" in df.columns:
            df["ds"] = pd.to_datetime(df["ds"])
            df = df.set_index("ds")
        else:
            raise RuntimeError("Το daily.parquet δεν έχει DatetimeIndex ή στήλη 'ds'.")

    df = df.sort_index()
    df = df.asfreq("D")  # για να σταματήσουν τα frequency warnings

    y = df["y"].astype(float)
    load = df.get("load", None)

    print("\n📘 Φόρτωση daily.parquet...")
    print(f"Συνολικές μέρες: {len(df)}, Train: {len(df) - TEST_DAYS}, Test: {TEST_DAYS}\n")

    train = df.iloc[:-TEST_DAYS]
    test = df.iloc[-TEST_DAYS:]
    y_train = train["y"]
    y_test = test["y"].values

    # 🔎 Επιλογή ARIMA παραμέτρων πάνω στο TRAIN
    arima_order, seasonal_order = choose_arima_params(y_train)

    results = []

    # 1. Naive (χθεσινή τιμή)
    yhat_naive = df["y"].shift(1).iloc[-TEST_DAYS:].values
    results.append(evaluate(y_test, yhat_naive, "Naive (yesterday)"))

    # 2. Seasonal Naive (7 μέρες πριν)
    yhat_seasonal = df["y"].shift(7).iloc[-TEST_DAYS:].values
    results.append(evaluate(y_test, yhat_seasonal, "Seasonal Naive (7)"))

    # 3. SARIMA μόνο με τιμές
    try:
        sarima_model = SARIMAX(
            train["y"],
            order=arima_order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        sarima_fit = sarima_model.fit(disp=False)
        sarima_forecast = sarima_fit.forecast(steps=TEST_DAYS)
        results.append(evaluate(y_test, sarima_forecast, "SARIMA"))
    except Exception as e:
        print(f"⚠️ SARIMA failed: {e}")

    # 4. SARIMAX με exogenous = load (αν υπάρχει)
    if load is not None:
        try:
            sarimax_model = SARIMAX(
                train["y"],
                exog=train[["load"]],
                order=arima_order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            sarimax_fit = sarimax_model.fit(disp=False)
            sarimax_forecast = sarimax_fit.forecast(
                steps=TEST_DAYS, exog=test[["load"]]
            )
            results.append(evaluate(y_test, sarimax_forecast, "SARIMAX (load)"))
        except Exception as e:
            print(f"⚠️ SARIMAX failed: {e}")

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("sMAPE").reset_index(drop=True)

    print("\n" + "=" * 52)
    print("  DAILY BASELINES RESULTS (auto_arima)")
    print("=" * 52)
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()