"""
backfill_historical.py — Fetches months of historical AQI + weather data
using Open-Meteo API (completely free, no API key required).

Open-Meteo provides:
  - Historical weather: temperature, humidity, wind, pressure, rain (years back)
  - Historical air quality: PM2.5, PM10, O3, NO2, dust (90+ days back)

Usage:
  # Backfill last 90 days
  python feature_pipeline/backfill_historical.py --days 90

  # Backfill a specific date range
  python feature_pipeline/backfill_historical.py --start 2025-01-01 --end 2025-05-01
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import numpy as np
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CITY_LAT, CITY_LON, CITY_NAME
from feature_pipeline.engineer_features import engineer_features
from feature_pipeline.upload_to_store import upload_features

# ── Open-Meteo endpoints (no API key needed) ───────────────
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def fetch_historical_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch hourly historical weather from Open-Meteo archive.
    Free, no API key, goes back years.
    """
    logger.info(f"Fetching weather history {start_date} → {end_date}...")

    params = {
        "latitude":   CITY_LAT,
        "longitude":  CITY_LON,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "rain",
            "surface_pressure",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m",
        ]),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }

    response = requests.get(WEATHER_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp":     hourly["time"],
        "temp_c":        hourly["temperature_2m"],
        "feels_like_c":  hourly["apparent_temperature"],
        "humidity_pct":  hourly["relative_humidity_2m"],
        "pressure_hpa":  hourly["surface_pressure"],
        "wind_speed_ms": hourly["wind_speed_10m"],
        "wind_deg":      hourly["wind_direction_10m"],
        "clouds_pct":    hourly["cloud_cover"],
        "rain_1h_mm":    hourly["rain"],
        "visibility_m":  None,
        "weather_main":  "Historical",
        "weather_desc":  "historical data",
    })

    # Fill nulls with forward fill
    df = df.ffill().bfill()
    logger.success(f"Weather history fetched — {len(df)} hourly rows")
    return df


def fetch_historical_air_quality(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch hourly historical air quality from Open-Meteo.
    Free, no API key, goes back 90+ days.
    """
    logger.info(f"Fetching air quality history {start_date} → {end_date}...")

    params = {
        "latitude":   CITY_LAT,
        "longitude":  CITY_LON,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly": ",".join([
            "pm10",
            "pm2_5",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "us_aqi",
        ]),
        "timezone": "UTC",
    }

    response = requests.get(AIR_QUALITY_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": hourly["time"],
        "aqi":       hourly["us_aqi"],
        "pm25":      hourly["pm2_5"],
        "pm10":      hourly["pm10"],
        "o3":        hourly["ozone"],
        "no2":       hourly["nitrogen_dioxide"],
        "so2":       hourly["sulphur_dioxide"],
        "co":        hourly["carbon_monoxide"],
    })

    df = df.ffill().bfill()
    logger.success(f"Air quality history fetched — {len(df)} hourly rows")
    return df


def fetch_and_upload_historical(
    start_date: str,
    end_date: str,
    upload: bool = True,
) -> pd.DataFrame:
    """
    Fetch weather + air quality, engineer features, and upload to Hopsworks.

    Args:
        start_date: 'YYYY-MM-DD'
        end_date  : 'YYYY-MM-DD'
        upload    : If True, push to Hopsworks Feature Store

    Returns:
        Engineered feature DataFrame
    """
    logger.info("=" * 55)
    logger.info(f"Historical Backfill: {start_date} → {end_date}")
    logger.info("=" * 55)

    # Fetch both sources
    df_weather = fetch_historical_weather(start_date, end_date)
    df_aq      = fetch_historical_air_quality(start_date, end_date)

    # Merge on timestamp
    df = pd.merge(df_weather, df_aq, on="timestamp", how="left")
    df["city"] = CITY_NAME

    logger.info(f"Merged dataset: {len(df)} rows, {len(df.columns)} columns")

    # Engineer features
    df_features = engineer_features(df)

    # Save locally
    os.makedirs("data/processed", exist_ok=True)
    out_path = f"data/processed/backfill_{start_date}_{end_date}.csv"
    df_features.to_csv(out_path, index=False)
    logger.success(f"Saved locally: {out_path}")

    # Upload to Hopsworks
    if upload:
        upload_features(df_features)

    logger.info(
        f"Backfill complete — {len(df_features)} rows ready for training"
    )
    return df_features


# ── CLI ────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill historical AQI + weather data from Open-Meteo"
    )
    parser.add_argument(
        "--days", type=int, default=90,
        help="Number of past days to backfill (default: 90)"
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Start date YYYY-MM-DD (overrides --days)"
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--no-upload", action="store_true",
        help="Skip uploading to Hopsworks (just save locally)"
    )
    args = parser.parse_args()

    today = datetime.now(timezone.utc).date()

    if args.start:
        start_date = args.start
        end_date   = args.end or str(today)
    else:
        start_date = str(today - timedelta(days=args.days))
        end_date   = str(today)

    fetch_and_upload_historical(
        start_date=start_date,
        end_date=end_date,
        upload=not args.no_upload,
    )
