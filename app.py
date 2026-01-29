import streamlit as st
import pandas as pd
import requests
import joblib
import time
import os
from datetime import datetime
from dotenv import load_dotenv

# =================================================
# LOAD ENV
# =================================================
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="🌾 Kissan Saathi | Real-Time Rain Advisory",
    layout="centered"
)

# =================================================
# API KEY CHECK
# =================================================
if not API_KEY:
    st.error("❌ OpenWeather API key not found. Check your .env file.")
    st.stop()

# =================================================
# LOAD MODEL
# =================================================
model = joblib.load("rain_prediction_model.pkl")

# =================================================
# FUNCTIONS
# =================================================
def type_writer(text, speed=0.04):
    placeholder = st.empty()
    typed = ""
    for ch in text:
        typed += ch
        placeholder.markdown(
            f"<h1 style='text-align:center; color:#2E7D32;'>{typed}</h1>",
            unsafe_allow_html=True
        )
        time.sleep(speed)

def get_realtime_weather(city):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if response.status_code != 200:
            st.error(f"❌ API Error: {data.get('message', 'Unknown error')}")
            return None

        return {
            "temp_max": data["main"]["temp_max"],
            "temp_min": data["main"]["temp_min"],
            "wind": data["wind"]["speed"],
            "precipitation": data.get("rain", {}).get("1h", 0.0)
        }

    except Exception as e:
        st.error(f"❌ Request failed: {e}")
        return None

def create_features(weather):
    return pd.DataFrame([{
        "precipitation": weather["precipitation"],
        "temp_max": weather["temp_max"],
        "temp_min": weather["temp_min"],
        "temp_range": weather["temp_max"] - weather["temp_min"],
        "wind": weather["wind"],
        "precip_rolling_3": weather["precipitation"],
        "precip_rolling_7": weather["precipitation"],
        "wind_rolling_3": weather["wind"],
        "month": datetime.now().month,
        "dayofweek": datetime.now().weekday()
    }])

# =================================================
# UI HEADER
# =================================================
type_writer("🌾 Kissan Saathi")

st.markdown(
    "<p style='text-align:center; font-size:18px;'>"
    "AI-powered Real-Time Rain Advisory for Farmers (2026 Ready)"
    "</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# =================================================
# USER INPUT
# =================================================
st.subheader("🌍 Enter Location")

city = st.text_input(
    "City Name (use format: City,CountryCode)",
    value="Delhi,IN"
)

st.caption(f"📅 Advisory generated on: {datetime.now().strftime('%d %B %Y, %I:%M %p')}")

# =================================================
# PREDICTION
# =================================================
if st.button("🌧️ Get Real-Time Rain Advisory"):
    with st.spinner("🔄 Fetching live weather data..."):
        weather = get_realtime_weather(city)
        time.sleep(1)

    if weather:
        X_live = create_features(weather)

        prediction = model.predict(X_live)[0]
        probability = model.predict_proba(X_live)[0][1] * 100

        st.markdown("### 📢 Advisory Result")

        if prediction == 1:
            st.error("🌧️ **Rain Expected Tomorrow**")
            st.metric("🌧️ Rain Probability", f"{probability:.2f}%")
            st.info(
                "🚜 **Farmer Advisory:**\n\n"
                "- Avoid irrigation today\n"
                "- Delay pesticide spraying\n"
                "- Protect harvested crops\n"
                "- Ensure proper drainage"
            )
        else:
            st.success("☀️ **No Rain Expected Tomorrow**")
            st.metric("☀️ Clear Weather Probability", f"{100 - probability:.2f}%")
            st.info(
                "🚜 **Farmer Advisory:**\n\n"
                "- Safe for irrigation\n"
                "- Fertilizer spraying recommended\n"
                "- Field operations can continue"
            )

        # -----------------------------------------
        # WEATHER SNAPSHOT
        # -----------------------------------------
        st.markdown("### 🌡️ Current Weather Snapshot")

        col1, col2, col3 = st.columns(3)
        col1.metric("🌡️ Max Temp (°C)", weather["temp_max"])
        col2.metric("🌡️ Min Temp (°C)", weather["temp_min"])
        col3.metric("💨 Wind Speed (m/s)", weather["wind"])

        st.metric("🌧️ Rain (last 1 hour)", f'{weather["precipitation"]} mm')

# =================================================
# FOOTER
# =================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; font-size:14px;'>"
    "Developed for farmers | Real-Time AI Decision Support System"
    "</p>",
    unsafe_allow_html=True
)
