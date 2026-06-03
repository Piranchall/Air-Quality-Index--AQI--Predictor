"""
fetch_data.py — Fetches raw AQI and weather data from external APIs.

Sources:
  - AQICN     : pollutant readings (AQI, PM2.5, PM10, O3, NO2, SO2, CO)
  - OpenWeather: weather readings  (temp, humidity, pressure, wind, rain)

Usage:
  # Fetch current data (single snapshot)
  python feature_pipeline/fetch_data.py

  # Backfill past N days of weather history (OpenWeather only supports 5-day history on free tier)
  python feature_pipeline/fetch_data.py --backfill --days 5
"""

import argparse
import sys
import os
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
from loguru import logger

# Allow running from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    AQICN_API_KEY,
    OPENWEATHER_API_KEY,
    CITY_NAME,
    CITY_LAT,
    CITY_LON,
)

# ── Constants ─────────────────────────────────────────────
url = "https://api.waqi.info/feed/A401143/" # University of Karachi, NED UET — active station
OPENWEATHER_CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHER_HISTORY_URL = "https://history.openweathermap.org/data/2.5/history/city"


# ── AQICN ─────────────────────────────────────────────────

def fetch_aqicn(city: str = CITY_NAME) -> dict:
    """
    Fetch current AQI and pollutant data from AQICN for a given city.
    Returns a flat dict ready for DataFrame construction.
    """
    url = "https://api.waqi.info/feed/A401143/"
    params = {"token": AQICN_API_KEY}

    logger.info(f"Fetching AQICN data for '{city}'...")
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    if data.get("status") != "ok":
        raise ValueError(f"AQICN API error: {data.get('data', 'Unknown error')}")

    station = data["data"]
    iaqi = station.get("iaqi", {})  # individual AQI readings per pollutant

    record = {
        "timestamp": datetime.now(timezone.utc).replace(
            minute=0, second=0, microsecond=0
        ).isoformat(),
        "city": city,
        "aqi": float(station.get("aqi", 0)) if station.get("aqi", "-") != "-" else None,
        "pm25": float(iaqi.get("pm25", {}).get("v", None)) if "pm25" in iaqi else None,
        "pm10": float(iaqi.get("pm10", {}).get("v", None)) if "pm10" in iaqi else None,
        "o3":   float(iaqi.get("o3",   {}).get("v", None)) if "o3"   in iaqi else None,
        "no2":  float(iaqi.get("no2",  {}).get("v", None)) if "no2"  in iaqi else None,
        "so2":  float(iaqi.get("so2",  {}).get("v", None)) if "so2"  in iaqi else None,
        "co":   float(iaqi.get("co",   {}).get("v", None)) if "co"   in iaqi else None,
    }

    logger.success(f"AQICN fetched — AQI: {record['aqi']}, PM2.5: {record['pm25']}")
    return record


# ── OpenWeather ───────────────────────────────────────────

def fetch_openweather_current(lat: float = CITY_LAT, lon: float = CITY_LON) -> dict:
    """
    Fetch current weather conditions from OpenWeatherMap.
    Returns a flat dict ready for DataFrame construction.
    """
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",  # Celsius
    }

    logger.info(f"Fetching OpenWeather current data for ({lat}, {lon})...")
    response = requests.get(OPENWEATHER_CURRENT_URL, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    record = {
        "timestamp": datetime.now(timezone.utc).replace(
            minute=0, second=0, microsecond=0
        ).isoformat(),
        "temp_c":        data["main"]["temp"],
        "feels_like_c":  data["main"]["feels_like"],
        "humidity_pct":  data["main"]["humidity"],
        "pressure_hpa":  data["main"]["pressure"],
        "wind_speed_ms": data["wind"]["speed"],
        "wind_deg":      data["wind"].get("deg", 0),
        "clouds_pct":    data["clouds"]["all"],
        "rain_1h_mm":    data.get("rain", {}).get("1h", 0.0),
        "weather_main":  data["weather"][0]["main"],        # e.g. "Clear"
        "weather_desc":  data["weather"][0]["description"], # e.g. "clear sky"
        "visibility_m":  data.get("visibility", None),
    }

    logger.success(
        f"OpenWeather fetched — Temp: {record['temp_c']}°C, "
        f"Humidity: {record['humidity_pct']}%, "
        f"Wind: {record['wind_speed_ms']} m/s"
    )
    return record


def fetch_openweather_history(
    lat: float = CITY_LAT,
    lon: float = CITY_LON,
    days: int = 5,
) -> list[dict]:
    """
    Fetch hourly historical weather from OpenWeatherMap.
    Free tier supports up to 5 days of history.
    Returns a list of dicts (one per hour).
    """
    records = []
    now = datetime.now(timezone.utc)

    for day_offset in range(1, days + 1):
        target_dt = now - timedelta(days=day_offset)
        unix_ts = int(target_dt.timestamp())

        params = {
            "lat": lat,
            "lon": lon,
            "type": "hour",
            "start": unix_ts,
            "cnt": 24,  # 24 hourly readings
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
        }

        logger.info(f"Fetching weather history for {target_dt.date()} ...")
        response = requests.get(OPENWEATHER_HISTORY_URL, params=params, timeout=10)

        if response.status_code == 401:
            logger.warning(
                "OpenWeather history requires a paid plan. "
                "Skipping historical weather — only current data will be used."
            )
            break

        response.raise_for_status()
        data = response.json()

        for entry in data.get("list", []):
            records.append({
                "timestamp":     datetime.fromtimestamp(
                    entry["dt"], tz=timezone.utc
                ).replace(minute=0, second=0, microsecond=0).isoformat(),
                "temp_c":        entry["main"]["temp"],
                "feels_like_c":  entry["main"]["feels_like"],
                "humidity_pct":  entry["main"]["humidity"],
                "pressure_hpa":  entry["main"]["pressure"],
                "wind_speed_ms": entry["wind"]["speed"],
                "wind_deg":      entry["wind"].get("deg", 0),
                "clouds_pct":    entry["clouds"]["all"],
                "rain_1h_mm":    entry.get("rain", {}).get("1h", 0.0),
                "weather_main":  entry["weather"][0]["main"],
                "weather_desc":  entry["weather"][0]["description"],
                "visibility_m":  entry.get("visibility", None),
            })

    logger.success(f"OpenWeather history fetched — {len(records)} hourly records")
    return records


# ── Combined fetch ─────────────────────────────────────────

def fetch_current_snapshot() -> pd.DataFrame:
    """
    Fetch one combined row of AQI + weather data for the current hour.
    This is what the hourly GitHub Actions workflow calls.
    """
    aqi_record     = fetch_aqicn()
    weather_record = fetch_openweather_current()

    # Merge on timestamp (both are floored to current hour)
    combined = {**aqi_record, **weather_record}
    # Remove duplicate timestamp key from weather (aqi_record timestamp wins)
    combined.pop("timestamp", None)
    combined["timestamp"] = aqi_record["timestamp"]

    df = pd.DataFrame([combined])
    logger.success(f"Combined snapshot ready — shape: {df.shape}")
    return df


def fetch_backfill(days: int = 5) -> pd.DataFrame:
    """
    Fetch historical weather data for backfill.
    AQI history is not available on AQICN free tier,
    so we fetch weather history and use current AQI as a placeholder.
    """
    logger.info(f"Starting backfill for {days} days...")

    weather_records = fetch_openweather_history(days=days)

    if not weather_records:
        logger.warning("No historical weather records returned. Using current snapshot only.")
        return fetch_current_snapshot()

    df = pd.DataFrame(weather_records)

    # AQICN doesn't provide history on free tier — add null pollutant columns
    # These will be filled during feature engineering with forward-fill
    for col in ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co"]:
        df[col] = None

    df["city"] = CITY_NAME
    logger.success(f"Backfill data ready — shape: {df.shape}")
    return df


# ── CLI ────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch AQI and weather data")
    parser.add_argument(
        "--backfill", action="store_true",
        help="Fetch historical data instead of current snapshot"
    )
    parser.add_argument(
        "--days", type=int, default=5,
        help="Number of days to backfill (default: 5, max free tier: 5)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save raw data to data/raw/ as CSV"
    )
    args = parser.parse_args()

    if args.backfill:
        df = fetch_backfill(days=args.days)
    else:
        df = fetch_current_snapshot()

    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    if args.save:
        os.makedirs("data/raw", exist_ok=True)
        fname = f"data/raw/raw_{'backfill' if args.backfill else 'snapshot'}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(fname, index=False)
        logger.success(f"Saved to {fname}")
