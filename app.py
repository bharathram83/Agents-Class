# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import os
from groq import Groq
from dotenv import load_dotenv
import json

# Load API Key
load_dotenv()
GROQ_API_KEY = os.getenv("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")
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="FP&A Forecast & Commentary", page_icon="📈", layout="wide")
st.title("📊 SaaS Revenue Forecasting & FP&A Commentary")

uploaded_file = st.file_uploader("Upload Excel file with 'Date' and 'Revenue' columns", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file, parse_dates=["Date"])
    df = df[["Date", "Revenue"]].rename(columns={"Date": "ds", "Revenue": "y"})

    # Forecast with Prophet
    st.subheader("🔮 Prophet Forecast")
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=30)  # Forecast next 30 days
    forecast = model.predict(future)

    # Plot forecast
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # Plot forecast components
    st.subheader("📈 Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    # FP&A Commentary Generation
    st.subheader("🤖 AI-Generated FP&A Commentary")
    client = Groq(api_key=GROQ_API_KEY)

    # Sample prompt
    data_for_ai = df.tail(60).to_dict(orient="records")  # Use recent data for commentary
    prompt = f"""You are the Head of FP&A at a SaaS company. Your task is to analyze the full budget variance table and provide:
    - Key insights from the data.
    - Areas of concern and key drivers for variance.
    - A CFO-ready summary using the Pyramid Principle.
    - Actionable recommendations to improve financial performance.

Here is the full dataset in JSON format:
{json.dumps(data_for_ai, indent=2)}"""

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a financial planning and analysis (FP&A) expert, specializing in SaaS companies."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-8b-8192",
    )

    ai_commentary = response.choices[0].message.content

    st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
    st.subheader("📖 AI-Generated FP&A Commentary")
    st.write(ai_commentary)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("📁 Please upload a valid Excel file to begin.")
