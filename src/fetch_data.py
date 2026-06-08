"""
src/fetch_data.py
─────────────────────────────────────────────────────────
Fetches hourly air quality data from Open-Meteo API.
No API key needed. Completely free.

Pollutants: PM2.5, PM10, NO2
Cities: Delhi, Mumbai, Bengaluru, Chennai, Kolkata

Run manually:   python src/fetch_data.py
Run automated:  GitHub Actions calls this every hour.
─────────────────────────────────────────────────────────
"""

import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
import os
import time


# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

CITIES = {
    "Delhi":     {"lat": 28.6139, "lon": 77.2090},
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
    "Chennai":   {"lat": 13.0827, "lon": 80.2707},
    "Kolkata":   {"lat": 22.5726, "lon": 88.3639},
}

OUTPUT_DIR = os.path.join("data", "raw")
# ──────────────────────────────────────────────────────────────────────────────


def fetch_city(city_name: str, lat: float, lon: float) -> list:
    """
    Fetch last 48 hours of PM2.5, PM10, NO2 for one city.
    Returns a list of row dicts, one per hour.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm10,pm2_5,nitrogen_dioxide",
        "timezone": "Asia/Kolkata",
        "past_days": 2,
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        hourly = data.get("hourly", {})
        times  = hourly.get("time", [])
        pm25   = hourly.get("pm2_5", [])
        pm10   = hourly.get("pm10", [])
        no2    = hourly.get("nitrogen_dioxide", [])

        rows = []
        for i, t in enumerate(times):
            rows.append({
                "timestamp": t,
                "city":      city_name,
                "pm25":      pm25[i] if i < len(pm25) else None,
                "pm10":      pm10[i] if i < len(pm10) else None,
                "no2":       no2[i]  if i < len(no2)  else None,
            })

        return rows

    except requests.exceptions.Timeout:
        print(f"  TIMEOUT: {city_name}")
        return []
    except requests.exceptions.HTTPError as e:
        print(f"  HTTP ERROR: {city_name} — {e}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"  NETWORK ERROR: {city_name} — {e}")
        return []


def fetch_all() -> pd.DataFrame:
    """Fetch data for all 5 cities. Returns combined DataFrame."""
    all_rows = []

    for city_name, coords in CITIES.items():
        print(f"  Fetching: {city_name}...", end=" ")
        rows = fetch_city(city_name, coords["lat"], coords["lon"])
        all_rows.extend(rows)
        print(f"{len(rows)} hourly readings")
        time.sleep(0.3)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["pm25", "pm10", "no2"], how="all")
    return df


def backfill_historical(days: int = 90) -> pd.DataFrame:
    """
    Fetch historical data for LSTM training.
    Call ONCE with: python src/fetch_data.py --backfill
    """
    print(f"\nBackfilling {days} days of historical data...")
    all_rows = []

    for city_name, coords in CITIES.items():
        print(f"  Fetching historical: {city_name}...", end=" ")

        end_date   = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        params = {
            "latitude":   coords["lat"],
            "longitude":  coords["lon"],
            "hourly":     "pm10,pm2_5,nitrogen_dioxide",
            "timezone":   "Asia/Kolkata",
            "start_date": start_date,
            "end_date":   end_date,
        }

        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data   = response.json()
            hourly = data.get("hourly", {})
            times  = hourly.get("time", [])
            pm25   = hourly.get("pm2_5", [])
            pm10   = hourly.get("pm10", [])
            no2    = hourly.get("nitrogen_dioxide", [])

            for i, t in enumerate(times):
                all_rows.append({
                    "timestamp": t,
                    "city":      city_name,
                    "pm25":      pm25[i] if i < len(pm25) else None,
                    "pm10":      pm10[i] if i < len(pm10) else None,
                    "no2":       no2[i]  if i < len(no2)  else None,
                })

            print(f"{len(times)} readings ({start_date} to {end_date})")

        except Exception as e:
            print(f"ERROR: {e}")

        time.sleep(0.5)

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["pm25", "pm10", "no2"], how="all")
    return df


def save(df: pd.DataFrame, filename: str = None) -> str:
    """Save to data/raw/YYYY-MM-DD.csv — appends and deduplicates."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if filename is None:
        today    = datetime.now().strftime("%Y-%m-%d")
        filepath = os.path.join(OUTPUT_DIR, f"{today}.csv")
    else:
        filepath = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(filepath):
        existing = pd.read_csv(filepath, parse_dates=["timestamp"])
        df       = pd.concat([existing, df], ignore_index=True)
        df       = df.drop_duplicates(subset=["timestamp", "city"])
        print(f"  Appended + deduplicated → {filepath}")
    else:
        print(f"  Created new file → {filepath}")

    df.to_csv(filepath, index=False)
    print(f"  Total rows saved: {len(df)}")
    return filepath


def main():
    import sys

    print("=" * 55)
    print(" AirCast — Data Fetch (Open-Meteo)")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M')} IST")
    print("=" * 55)

    if "--backfill" in sys.argv:
        df = backfill_historical(days=90)
        if not df.empty:
            filepath = save(df, filename="historical_90days.csv")
            print(f"\nBackfill complete: {len(df)} rows → {filepath}")
        return

    print("\nFetching latest readings:")
    df = fetch_all()

    if df.empty:
        print("\n[WARNING] No data fetched. Check internet connection.")
        return

    filepath = save(df)

    print("\nSample data:")
    print(df[["timestamp", "city", "pm25", "pm10", "no2"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()