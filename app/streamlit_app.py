
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

st.set_page_config(page_title="EPF Greece – Day-Ahead (Daily)", layout="centered")

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "daily.parquet"
MODEL = Path(__file__).resolve().parents[1] / "models" / "xgb_day.pkl"

st.title("⚡ Electricity Price Forecast – Greece (Daily)")

if not DATA.exists():
    st.warning("Processed data not found. Run `python -m src.data` first.")
else:
    df = pd.read_parquet(DATA)
    st.write("Recent data preview:", df.tail(10))

if MODEL.exists():
    model = joblib.load(MODEL)
    st.success("Model loaded.")
else:
    st.info("XGB model not found. Train it with `python -m src.train_xgb`.")
    model = None

st.header("Predict next day")
with st.form("predict_form"):
    if df is not None and len(df) > 30:
        latest = df.iloc[[-1]].drop(columns=["y"])
        st.write("Using latest feature vector:")
        st.write(latest)
    submitted = st.form_submit_button("Predict")
    if submitted:
        if model is None:
            st.error("Train a model first.")
        else:
            yhat = float(model.predict(latest)[0])
            st.metric("Predicted next-day average price (€/MWh)", f"{yhat:.2f}")
