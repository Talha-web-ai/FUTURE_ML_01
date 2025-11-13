import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("FUTURE_ML_01 — Sales Forecasting")

uploaded = st.file_uploader("Upload sales CSV (two columns: ds, y)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("Preview:", df.head())
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    st.write("Forecast preview:", forecast[['ds','yhat','yhat_lower','yhat_upper']].tail())
    fig = model.plot(forecast)
    st.pyplot(fig)
else:
    st.info("Upload a CSV or put sales.csv into the /data folder")
