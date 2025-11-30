import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
import sys
import io
import warnings
import matplotlib.pyplot as plt

# Statsmodels imports
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Ρυθμίσεις κονσόλας
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "daily.parquet"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORT_DIR.mkdir(exist_ok=True)

def calculate_metrics(y_true, y_pred, model_name):
    """Υπολογισμός μετρικών (MAE, RMSE, MAPE)"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    
    print(f"{model_name:30s} | MAE: {mae:6.3f} | RMSE: {rmse:6.3f} | MAPE: {mape:6.2f}%")
    return {"Model": model_name, "MAE": mae, "RMSE": rmse, "MAPE": mape}

def plot_results(y_test, preds_dict):
    """Δημιουργία γραφήματος σύγκρισης"""
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test.values, label='Actual Price', color='black', linewidth=2.5)
    
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    for i, (name, preds) in enumerate(preds_dict.items()):
        c = colors[i % len(colors)]
        plt.plot(y_test.index, preds, label=name, linestyle='--', linewidth=1.5, color=c)
        
    plt.title("Forecast Comparison: Actual vs Baselines (Test Set)", fontsize=14)
    plt.ylabel("Price (€/MWh)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = REPORT_DIR / "baseline_comparison_final.png"
    plt.savefig(save_path)
    print(f"\n[Graph] Το γράφημα αποθηκεύτηκε στο: {save_path}")

def main():
    if not DATA_PATH.exists():
        print("❌ Error: Δεν βρέθηκε το daily.parquet")
        return

    df = pd.read_parquet(DATA_PATH)
    
    # Test set: Τελευταίες 60 μέρες
    TEST_DAYS = 60
    train = df.iloc[:-TEST_DAYS]
    test = df.iloc[-TEST_DAYS:]
    y_train = train['y']
    y_test = test['y']
    
    print(f"Set Info: Train={len(train)}, Test={len(test)}")
    print("="*85)
    print(f"{'Model':30s} | {'MAE':8s} | {'RMSE':8s} | {'MAPE':8s}")
    print("-" * 85)

    predictions = {}

    # --- 1. Naive (Yesterday) ---
    pred_naive = df['y'].shift(1).iloc[-TEST_DAYS:]
    calculate_metrics(y_test, pred_naive, "Naive (Yesterday)")
    predictions["Naive"] = pred_naive

    # --- 2. Seasonal Naive (Last Week) ---
    pred_snaive = df['y'].shift(7).iloc[-TEST_DAYS:]
    calculate_metrics(y_test, pred_snaive, "Seasonal Naive (7d)")
    predictions["S.Naive"] = pred_snaive

    # --- 3. Holt-Winters (Exponential Smoothing) ---
    # Καλό για εποχικότητα, γρήγορο.
    # Seasonal period = 7 (weekly)
    try:
        # Εκπαίδευση σε όλο το train
        hw_model = ExponentialSmoothing(y_train, seasonal_periods=7, trend='add', seasonal='add').fit()
        # Εδώ κάνουμε rolling forecast "μπακαλίστικα" για ταχύτητα (static fit, dynamic predict)
        # ή one-step ahead loop αν θέλουμε ακρίβεια. Θα κάνουμε one-step loop.
        hw_preds = []
        history = list(y_train)
        
        for t in range(len(y_test)):
            model = ExponentialSmoothing(history, seasonal_periods=7, trend='add', seasonal='add').fit()
            pred = model.forecast(1)[0]
            hw_preds.append(pred)
            history.append(y_test.iloc[t]) # Update history
            
        calculate_metrics(y_test, hw_preds, "Holt-Winters (Rolling)")
        predictions["Holt-Winters"] = hw_preds
    except Exception as e:
        print(f"HW Error: {e}")

    # --- 4. SARIMA Optimized (Rolling Forecast) ---
    # Οι παράμετροι που βρήκε το pmdarima: (3, 0, 2) x (1, 1, 1, 7)
    print("..." * 10)
    print("Εκπαίδευση Optimized SARIMA (3,0,2)x(1,1,1,7)")
    
    sarima_preds = []
    
    # Αρχικό fit
    model = SARIMAX(y_train, order=(3,0,2), seasonal_order=(1,1,1,7), 
                    enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    
    # Rolling forecast loop (χρήση filter για ταχύτητα)
    for t in range(len(y_test)):
        # One-step forecast
        yhat = model_fit.forecast()[0]
        sarima_preds.append(yhat)
        
        # Update with true value
        true_val = y_test.iloc[t]
        model_fit = model_fit.append([true_val], refit=False)

    calculate_metrics(y_test, sarima_preds, "SARIMA Opt (3,0,2)(1,1,1)[7]")
    predictions["SARIMA Opt"] = sarima_preds

    print("="*85)
    
    try:
        plot_results(y_test, predictions)
    except:
        pass

if __name__ == "__main__":
    main()