"""
app.py
─────────────────────────────────────────────────────────
AirCast — Real-Time Air Quality Forecasting Dashboard

Loads trained LSTM model, fetches latest pollution data,
and displays live readings + 24-hour PM2.5 forecast.

Run locally: streamlit run app.py
─────────────────────────────────────────────────────────
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
import requests
import sys
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add src/ to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from model import AirCastLSTM

# ── CONFIG ────────────────────────────────────────────────────────────────────
CITIES = {
    "Delhi":     {"lat": 28.6139, "lon": 77.2090},
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
    "Chennai":   {"lat": 13.0827, "lon": 80.2707},
    "Kolkata":   {"lat": 22.5726, "lon": 88.3639},
}

AQI_CATEGORIES = [
    (0,   30,  "Good",              "#00C853"),
    (30,  60,  "Satisfactory",      "#64DD17"),
    (60,  90,  "Moderately Polluted","#FFD600"),
    (90,  120, "Poor",              "#FF6D00"),
    (120, 250, "Very Poor",         "#DD2C00"),
    (250, 999, "Severe",            "#8B0000"),
]

MODEL_PATH  = os.path.join("models", "aircast_lstm.pt")
SCALER_PATH = os.path.join("models", "scaler.pkl")
BASE_URL    = "https://air-quality-api.open-meteo.com/v1/air-quality"
# ──────────────────────────────────────────────────────────────────────────────


def get_aqi_category(pm25: float) -> tuple:
    for low, high, label, color in AQI_CATEGORIES:
        if low <= pm25 < high:
            return label, color
    return "Severe", "#8B0000"


@st.cache_resource
def load_model():
    """Load trained LSTM model. Cached so it only loads once."""
    model = AirCastLSTM()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


@st.cache_resource
def load_scaler():
    """Load saved MinMaxScaler. Cached so it only loads once."""
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data(ttl=3600)  # refresh every hour
def fetch_live_data(city: str) -> pd.DataFrame:
    """
    Fetch last 48 hours of pollution data for one city.
    Cached for 1 hour — matches our GitHub Actions cron schedule.
    """
    coords = CITIES[city]
    params = {
        "latitude":  coords["lat"],
        "longitude": coords["lon"],
        "hourly":    "pm10,pm2_5,nitrogen_dioxide",
        "timezone":  "Asia/Kolkata",
        "past_days": 2,
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        data   = response.json()
        hourly = data["hourly"]

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly["time"]),
            "pm25":      hourly["pm2_5"],
            "pm10":      hourly["pm10"],
            "no2":       hourly["nitrogen_dioxide"],
        })
        df = df.dropna()
        return df

    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return pd.DataFrame()


def make_forecast(df: pd.DataFrame, model, scaler) -> np.ndarray:
    """
    Run LSTM inference on the last 24 hours of data.
    Returns 24-hour PM2.5 forecast in real µg/m³ units.
    """
    features = ["pm25", "pm10", "no2"]

    # Take last 24 hours
    recent = df[features].tail(24).values

    if len(recent) < 24:
        return None

    # Normalize using saved scaler
    recent_scaled = scaler.transform(recent)

    # Convert to tensor: (1, 24, 3)
    x = torch.tensor(recent_scaled, dtype=torch.float32).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        pred_scaled = model(x).squeeze(0).numpy()  # (24,)

    # Inverse transform PM2.5 column only
    pad = np.zeros((24, 3))
    pad[:, 0] = pred_scaled
    pred_real = scaler.inverse_transform(pad)[:, 0]

    return pred_real


def plot_history(df: pd.DataFrame, city: str):
    """Plot last 48 hours of PM2.5 readings."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    ax.plot(df["timestamp"], df["pm25"],
            color="#4FC3F7", linewidth=1.8, label="PM2.5")
    ax.fill_between(df["timestamp"], df["pm25"],
                    alpha=0.15, color="#4FC3F7")

    ax.set_title(f"{city} — Last 48 Hours PM2.5",
                 color="white", fontsize=13, pad=10)
    ax.set_ylabel("µg/m³", color="white")
    ax.tick_params(colors="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    plt.xticks(rotation=30, ha="right")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.grid(alpha=0.15, color="white")
    plt.tight_layout()
    return fig


def plot_forecast(forecast: np.ndarray, last_timestamp):
    """Plot 24-hour PM2.5 forecast."""
    future_times = [last_timestamp + timedelta(hours=i+1) for i in range(24)]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    ax.plot(future_times, forecast,
            color="#FF7043", linewidth=2, linestyle="--",
            marker="o", markersize=3, label="Forecast")
    ax.fill_between(future_times, forecast,
                    alpha=0.15, color="#FF7043")

    ax.set_title("24-Hour PM2.5 Forecast",
                 color="white", fontsize=13, pad=10)
    ax.set_ylabel("µg/m³", color="white")
    ax.tick_params(colors="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    plt.xticks(rotation=30, ha="right")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.grid(alpha=0.15, color="white")
    plt.tight_layout()
    return fig


# ── MAIN APP ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AirCast",
    page_icon="🌬️",
    layout="wide",
)

# Header
st.title("🌬️ AirCast")
st.caption("Real-time air quality monitoring + 24-hour PM2.5 forecasting for Indian cities")

st.divider()

# Sidebar
with st.sidebar:
    st.header("Settings")
    selected_city = st.selectbox("Select City", list(CITIES.keys()))
    st.caption(f"Data updates every hour via GitHub Actions.")
    st.caption(f"Model: 2-layer PyTorch LSTM")
    st.caption(f"MAE: 4.63 µg/m³ on test set")
    st.divider()
    st.caption("Built by Ashok Chaturvedi")

# Load model and scaler
model  = load_model()
scaler = load_scaler()

# Fetch live data
with st.spinner(f"Fetching live data for {selected_city}..."):
    df = fetch_live_data(selected_city)

if df.empty:
    st.error("Could not fetch data. Please try again later.")
    st.stop()

# Current readings
latest = df.iloc[-1]
label, color = get_aqi_category(latest["pm25"])

st.subheader(f"📍 {selected_city} — Current Readings")

col1, col2, col3, col4 = st.columns(4)

col1.metric("PM2.5", f"{latest['pm25']:.1f} µg/m³")
col2.metric("PM10",  f"{latest['pm10']:.1f} µg/m³")
col3.metric("NO2",   f"{latest['no2']:.1f} µg/m³")
col4.markdown(f"""
**AQI Category**
<div style='background:{color};padding:6px 12px;border-radius:8px;
color:white;font-weight:bold;text-align:center;margin-top:4px'>
{label}
</div>
""", unsafe_allow_html=True)

st.caption(f"Last updated: {latest['timestamp'].strftime('%d %b %Y, %H:%M IST')}")

st.divider()

# Historical chart
st.subheader("📈 Historical PM2.5 (Last 48 Hours)")
fig_history = plot_history(df, selected_city)
st.pyplot(fig_history)

st.divider()

# Forecast
st.subheader("🔮 24-Hour PM2.5 Forecast (LSTM Model)")

forecast = make_forecast(df, model, scaler)

if forecast is not None:
    fig_forecast = plot_forecast(forecast, latest["timestamp"])
    st.pyplot(fig_forecast)

    # Forecast summary
    avg_forecast = np.mean(forecast)
    max_forecast = np.max(forecast)
    flabel, fcolor = get_aqi_category(avg_forecast)

    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("Avg Forecast PM2.5", f"{avg_forecast:.1f} µg/m³")
    fc2.metric("Peak Forecast PM2.5", f"{max_forecast:.1f} µg/m³")
    fc3.markdown(f"""
**Expected Category**
<div style='background:{fcolor};padding:6px 12px;border-radius:8px;
color:white;font-weight:bold;text-align:center;margin-top:4px'>
{flabel}
</div>
""", unsafe_allow_html=True)

else:
    st.warning("Not enough data to generate forecast.")

st.divider()
st.caption("AirCast | Data: Open-Meteo API | Model: PyTorch LSTM | MAE 4.63 µg/m³")