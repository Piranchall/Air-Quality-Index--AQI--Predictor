"""
config.py — Central configuration for the AQI Predictor project.
All pipeline scripts import from here so settings are changed in one place.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ───────────────────────────────────────────────
AQICN_API_KEY = os.getenv("AQICN_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ── Hopsworks ──────────────────────────────────────────────
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_predictor")

# Feature group names and versions
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
MODEL_NAME = "aqi_forecaster"

# ── Target City ────────────────────────────────────────────
CITY_NAME = os.getenv("CITY_NAME", "London")
CITY_LAT = float(os.getenv("CITY_LAT", 51.5074))
CITY_LON = float(os.getenv("CITY_LON", -0.1278))

# ── Feature Engineering ────────────────────────────────────
# How many hours of history to use as input features for the model
LOOKBACK_HOURS = 24

# How many days ahead to forecast
FORECAST_DAYS = 3

# ── Training ───────────────────────────────────────────────
TEST_SIZE = 0.2          # 20% of data held out for evaluation
RANDOM_STATE = 42

# ── AQI Alert Thresholds ───────────────────────────────────
AQI_LEVELS = {
    "Good":                          (0,   50),
    "Moderate":                      (51,  100),
    "Unhealthy for Sensitive Groups": (101, 150),
    "Unhealthy":                     (151, 200),
    "Very Unhealthy":                (201, 300),
    "Hazardous":                     (301, float("inf")),
}

def get_aqi_category(aqi: float) -> str:
    """Return the AQI category label for a given AQI value."""
    for label, (low, high) in AQI_LEVELS.items():
        if low <= aqi <= high:
            return label
    return "Unknown"

# ── Raw feature columns expected from APIs ─────────────────
RAW_WEATHER_COLS = [
    "timestamp", "temp_c", "humidity_pct", "pressure_hpa",
    "wind_speed_ms", "wind_deg", "clouds_pct", "rain_1h_mm",
]

RAW_POLLUTANT_COLS = [
    "timestamp", "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
]
