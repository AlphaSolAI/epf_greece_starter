"""
Restore historical Dutch TTF gas prices using Open-Meteo or yfinance.
Fetches from Nov 2024 through Mar 2026, formats as Investing.com CSV,
then merges with the current partial gas CSV.
"""
import pandas as pd
import requests
import json
import sys
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[1]
GAS_CSV  = BASE_DIR / "data" / "raw" / "Dutch TTF Natural Gas Futures Historical Data.csv"

def fetch_ttf_from_stooq():
    """Fetch Dutch TTF gas from Stooq (free, no API key)."""
    url = "https://stooq.com/q/d/l/?s=ttf.f&d1=20140101&d2=20260320&i=d"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and "Date" in r.text:
            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
            print(f"[Stooq] Downloaded {len(df)} rows, cols: {list(df.columns)}")
            return df
    except Exception as e:
        print(f"[Stooq] Failed: {e}")
    return None

def fetch_ttf_from_yfinance():
    """Fetch TTF from yfinance."""
    try:
        import yfinance as yf
        # Try different tickers for TTF gas
        for ticker in ["TTF=F", "NG=F", "TTFM.AS"]:
            try:
                t = yf.Ticker(ticker)
                hist = t.history(start="2014-01-01", end="2026-03-21")
                if len(hist) > 100:
                    print(f"[yfinance] {ticker}: {len(hist)} rows, {hist.index.min()} -> {hist.index.max()}")
                    df = hist[["Close"]].reset_index()
                    df.columns = ["Date", "Price"]
                    df["Date"] = df["Date"].dt.strftime("%m/%d/%Y")
                    return df[["Date","Price"]]
            except Exception as e:
                print(f"[yfinance] {ticker}: {e}")
    except ImportError:
        print("[yfinance] not installed")
    return None

def main():
    # Load current gas CSV (only has Feb 20 - Mar 20, 2026)
    current_df = pd.read_csv(GAS_CSV)
    current_df["date_parsed"] = pd.to_datetime(current_df["Date"], format="%m/%d/%Y", errors="coerce")
    current_df = current_df.dropna(subset=["date_parsed"]).sort_values("date_parsed")
    print(f"Current gas CSV: {len(current_df)} rows | {current_df['date_parsed'].min().date()} -> {current_df['date_parsed'].max().date()}")

    if len(current_df) > 200:
        print("[INFO] Gas CSV already has sufficient history. No restoration needed.")
        return

    # Try yfinance first
    print("\n[1] Trying yfinance...")
    hist_df = fetch_ttf_from_yfinance()

    if hist_df is None:
        print("\n[2] Trying Stooq...")
        stooq = fetch_ttf_from_stooq()
        if stooq is not None:
            # Stooq format: Date, Open, High, Low, Close, Volume
            stooq["date_parsed"] = pd.to_datetime(stooq["Date"], errors="coerce")
            stooq = stooq.sort_values("date_parsed")
            hist_df = stooq[["Date","Close"]].rename(columns={"Close":"Price"})
            hist_df["Date"] = hist_df["date_parsed"].dt.strftime("%m/%d/%Y")
            hist_df = hist_df[["Date","Price"]]

    if hist_df is None:
        print("[ERROR] Could not fetch historical gas prices. Please upload full Dutch TTF CSV.")
        sys.exit(1)

    # Parse the historical data
    hist_df["date_parsed"] = pd.to_datetime(hist_df["Date"], format="%m/%d/%Y", errors="coerce")
    hist_df = hist_df.dropna(subset=["date_parsed"]).sort_values("date_parsed")
    print(f"\n[INFO] Historical data: {len(hist_df)} rows | {hist_df['date_parsed'].min().date()} -> {hist_df['date_parsed'].max().date()}")

    # Merge: use historical for pre-Feb-20, new CSV for Feb-20+
    cutoff = pd.Timestamp("2026-02-20")
    old_part = hist_df[hist_df["date_parsed"] < cutoff].copy()
    new_part = current_df[current_df["date_parsed"] >= cutoff].copy()

    # Rebuild merged CSV
    old_part = old_part[["Date","Price"]].copy()
    old_part["Open"] = old_part["Price"]
    old_part["High"] = old_part["Price"]
    old_part["Low"]  = old_part["Price"]
    old_part["Vol."] = "—"
    old_part["Change %"] = "—"

    new_part_out = new_part[["Date","Price","Open","High","Low","Vol.","Change %"]].copy()

    merged = pd.concat([old_part[["Date","Price","Open","High","Low","Vol.","Change %"]], new_part_out])
    merged["date_parsed"] = pd.to_datetime(merged["Date"], format="%m/%d/%Y", errors="coerce")
    merged = merged.sort_values("date_parsed", ascending=False).drop_duplicates("date_parsed")

    # Save
    merged = merged.drop(columns=["date_parsed"])
    merged.to_csv(GAS_CSV, index=False)
    print(f"\n✅ Saved restored gas CSV: {len(merged)} rows | {pd.to_datetime(merged['Date']).min().date()} -> {pd.to_datetime(merged['Date']).max().date()}")

if __name__ == "__main__":
    main()
