import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

st.set_page_config(page_title="Revenue Forecasting Agent", page_icon="📈", layout="wide")
st.title("📈 AI Agent: Forecast Revenue with Prophet")
st.markdown("Upload your Excel file with `Date`, `Revenue`, and optionally a `Category` column to forecast multiple lines.")

# Upload input file
uploaded_file = st.file_uploader("📤 Upload Excel File", type=["xlsx"])

# Forecast period input
forecast_period = st.slider("🔮 Forecast Horizon (months)", min_value=1, max_value=36, value=12)

if uploaded_file:
    # Read Excel file
    df = pd.read_excel(uploaded_file)

    # Check required columns
    if 'Date' not in df.columns or 'Revenue' not in df.columns:
        st.error("❌ File must contain 'Date' and 'Revenue' columns.")
        st.stop()

    # Optional multi-line support
    has_category = 'Category' in df.columns
    categories = df['Category'].unique() if has_category else ['All Data']

    # Convert Date
    df['Date'] = pd.to_datetime(df['Date'])

    # Loop through each category or full data
    for category in categories:
        st.divider()
        st.subheader(f"📦 Forecast for: {category}")

        df_subset = df[df['Category'] == category] if has_category else df.copy()
        prophet_df = df_subset.rename(columns={"Date": "ds", "Revenue": "y"})[['ds', 'y']]

        # Fit Prophet model
        model = Prophet()
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=forecast_period, freq='M')
        forecast = model.predict(future)

        # Plot forecast
        fig = model.plot(forecast)
        st.pyplot(fig)

        # Merge actual and forecast for AI summary
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        merged = pd.merge(prophet_df, forecast_df, on='ds', how='outer')
        data_for_ai = merged.tail(24).to_json(orient="records", date_format="iso")

        # AI Commentary
        client = Groq(api_key=GROQ_API_KEY)

        prompt = f"""
        You are the Head of FP&A at a SaaS company. Based on the revenue forecast and recent trends for the category '{category}', provide:
        - Key trends, seasonality, and growth/decline patterns.
        - Risks or concerns around revenue.
        - A concise CFO-level summary using the Pyramid Principle.
        - Strategic recommendations to improve financial performance.

        Dataset (last 24 records) in JSON: {data_for_ai}
        """

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a senior FP&A expert specializing in SaaS forecasting."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192"
        )

        ai_commentary = response.choices[0].message.content
        st.markdown("### 🧠 AI Commentary")
        st.write(ai_commentary)

else:
    st.info("Please upload an Excel file with 'Date', 'Revenue', and optional 'Category' columns.")

