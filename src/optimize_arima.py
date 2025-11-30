import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io
import warnings

# UTF-8 για να μη σακατεύει τα ελληνικά
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

try:
    import pmdarima as pm
    print("✅ Η βιβλιοθήκη pmdarima βρέθηκε.")
except ImportError:
    print("❌ ΣΦΑΛΜΑ: Η βιβλιοθήκη pmdarima δεν είναι εγκατεστημένη.")
    print("   Τρέξε στο τερματικό: pip install pmdarima")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
DAILY_PATH = ROOT / "data" / "processed" / "daily.parquet"
HOURLY_PATH = ROOT / "data" / "processed" / "hourly.parquet"


def run_daily():
    if not DAILY_PATH.exists():
        print(f"❌ Δεν βρέθηκε το {DAILY_PATH}")
        return

    df = pd.read_parquet(DAILY_PATH)
    y = df["y"].astype(float)

    # δουλεύουμε μόνο με train (τελευταίες 60 μέρες = test)
    test_size = 60
    y_train = y.iloc[:-test_size]

    print(f"\n🔍 DAILY Auto-ARIMA σε {len(y_train)} σημεία (ημερήσια).")
    print("   m=7 (εβδομαδιαία εποχικότητα)")
    print("-" * 60)

    model = pm.auto_arima(
        y_train,
        start_p=1,
        start_q=1,
        max_p=3,
        max_q=3,
        m=7,
        start_P=0,
        seasonal=True,
        d=None,
        D=1,
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )

    print("\n" + "=" * 60)
    print("🏆 Βέλτιστο DAILY μοντέλο")
    print("=" * 60)
    print(model.summary())

    print("\n👉 Copy-paste για το baselines_daily.py:\n")
    print(f"ARIMA_ORDER = {model.order}")
    print(f"SEASONAL_ORDER = {model.seasonal_order}")
    print("\n(βάλε αυτά τα δύο στις σταθερές του SARIMA / SARIMAX.)")


def run_hourly():
    if not HOURLY_PATH.exists():
        print(f"❌ Δεν βρέθηκε το {HOURLY_PATH}")
        return

    df = pd.read_parquet(HOURLY_PATH)
    y = df["price"].astype(float)

    # train μόνο πριν από τις τελευταίες 7 μέρες (168 ώρες)
    test_hours = 24 * 7
    y_train = y.iloc[:-test_hours]

    print(f"\n🔍 HOURLY Auto-ARIMA σε {len(y_train)} σημεία (ωριαία).")
    print("   m=24 (ημερήσια εποχικότητα – 24 ώρες)")
    print("-" * 60)

    model = pm.auto_arima(
        y_train,
        start_p=1,
        start_q=1,
        max_p=3,
        max_q=3,
        m=24,              # ημερήσια seasonality
        start_P=0,
        seasonal=True,
        d=None,
        D=1,
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )

    print("\n" + "=" * 60)
    print("🏆 Βέλτιστο HOURLY μοντέλο")
    print("=" * 60)
    print(model.summary())

    print("\n👉 Copy-paste για τα hourly baselines:\n")
    print(f"ARIMA_ORDER = {model.order}")
    print(f"SEASONAL_ORDER = {model.seasonal_order}")


def main():
    mode = "daily"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    if mode == "daily":
        run_daily()
    elif mode == "hourly":
        run_hourly()
    else:
        print("❌ Άγνωστος τρόπος. Χρησιμοποίησε:")
        print("   python -m src.optimize_arima daily")
        print("   ή")
        print("   python -m src.optimize_arima hourly")


if __name__ == "__main__":
    main()
