"""
engineer_features.py — Computes ML-ready features from raw API data.

Features engineered:
  Time-based  : hour, day_of_week, month, is_weekend, is_rush_hour
  Lag features: aqi_lag_1h, aqi_lag_3h, aqi_lag_6h, aqi_lag_24h
  Rolling stats: aqi_rolling_mean_3h, aqi_rolling_mean_6h, aqi_rolling_std_3h
  Change rate : aqi_change_rate (delta AQI per hour)
  Wind        : wind_u, wind_v (decomposed into x/y components)
  Interaction : temp_humidity_index, pressure_change

Usage:
  from feature_pipeline.engineer_features import engineer_features
  df_features = engineer_features(df_raw)
"""

import sys
import os
from datetime import datetime

import pandas as pd
import numpy as np
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Main function ──────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw DataFrame (output of fetch_data.py) and returns
    a feature-engineered DataFrame ready for the Feature Store.

    Args:
        df: Raw DataFrame with columns from AQICN + OpenWeather

    Returns:
        df_feat: Engineered DataFrame with all ML features
    """
    logger.info(f"Engineering features on {len(df)} rows...")

    df = df.copy()

    # ── 1. Parse timestamp ────────────────────────────────
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── 2. Time-based features ────────────────────────────
    df["hour"]         = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.dayofweek   # 0=Monday, 6=Sunday
    df["month"]        = df["timestamp"].dt.month
    df["day_of_year"]  = df["timestamp"].dt.dayofyear
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    # Cyclical encoding of hour and month (captures periodicity)
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]    = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # ── 3. AQI lag features ───────────────────────────────
    # These tell the model what AQI was N hours ago
    if "aqi" in df.columns:
        df["aqi_lag_1h"]  = df["aqi"].shift(1)
        df["aqi_lag_3h"]  = df["aqi"].shift(3)
        df["aqi_lag_6h"]  = df["aqi"].shift(6)
        df["aqi_lag_24h"] = df["aqi"].shift(24)

        # ── 4. Rolling statistics ─────────────────────────
        df["aqi_rolling_mean_3h"]  = df["aqi"].rolling(window=3,  min_periods=1).mean()
        df["aqi_rolling_mean_6h"]  = df["aqi"].rolling(window=6,  min_periods=1).mean()
        df["aqi_rolling_mean_24h"] = df["aqi"].rolling(window=24, min_periods=1).mean()
        df["aqi_rolling_std_3h"]   = df["aqi"].rolling(window=3,  min_periods=1).std().fillna(0)
        df["aqi_rolling_max_6h"]   = df["aqi"].rolling(window=6,  min_periods=1).max()

        # ── 5. AQI change rate ────────────────────────────
        # How fast is AQI rising or falling?
        df["aqi_change_rate"] = df["aqi"].diff(1).fillna(0)
        df["aqi_change_rate_3h"] = df["aqi"].diff(3).fillna(0)

    # ── 6. Wind decomposition ─────────────────────────────
    # Wind direction + speed → U (east-west) and V (north-south) components
    # Better for ML than raw degrees
    if "wind_speed_ms" in df.columns and "wind_deg" in df.columns:
        wind_rad = np.deg2rad(df["wind_deg"])
        df["wind_u"] = -df["wind_speed_ms"] * np.sin(wind_rad)  # eastward
        df["wind_v"] = -df["wind_speed_ms"] * np.cos(wind_rad)  # northward

    # ── 7. Interaction features ───────────────────────────
    if "temp_c" in df.columns and "humidity_pct" in df.columns:
        # Heat index approximation — hot + humid = worse AQI dispersion
        df["temp_humidity_index"] = df["temp_c"] * df["humidity_pct"] / 100

    if "pressure_hpa" in df.columns:
        # Pressure drop often precedes worse air quality
        df["pressure_change"] = df["pressure_hpa"].diff(1).fillna(0)

    if "rain_1h_mm" in df.columns:
        # Rain washes out particulates — binary flag is useful
        df["is_raining"] = (df["rain_1h_mm"] > 0).astype(int)

    # ── 8. Pollutant ratios ───────────────────────────────
    if "pm25" in df.columns and "pm10" in df.columns:
        # PM2.5/PM10 ratio indicates particle source type
        df["pm_ratio"] = df["pm25"] / (df["pm10"] + 1e-6)
        df["pm_ratio"] = df["pm_ratio"].clip(0, 1)  # cap at 1

    # ── 9. Forward-fill missing pollutant data ────────────
    # AQICN may not always return all pollutants
    pollutant_cols = ["pm25", "pm10", "o3", "no2", "so2", "co", "aqi"]
    for col in pollutant_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # ── 10. Drop non-numeric/non-feature columns ──────────
    drop_cols = ["weather_main", "weather_desc", "city"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ── 11. Final null check ──────────────────────────────
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning(f"Nulls remaining after engineering:\n{null_counts[null_counts > 0]}")

    logger.success(
        f"Feature engineering complete — "
        f"{len(df)} rows, {len(df.columns)} features"
    )
    return df


def get_feature_names() -> list[str]:
    """
    Returns the list of all feature column names (excluding timestamp and target).
    Useful for the training pipeline to know what to feed the model.
    """
    return [
        # Time
        "hour", "day_of_week", "month", "day_of_year",
        "is_weekend", "is_rush_hour",
        "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
        # AQI lags
        "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_24h",
        # Rolling stats
        "aqi_rolling_mean_3h", "aqi_rolling_mean_6h", "aqi_rolling_mean_24h",
        "aqi_rolling_std_3h", "aqi_rolling_max_6h",
        # Change rate
        "aqi_change_rate", "aqi_change_rate_3h",
        # Weather
        "temp_c", "feels_like_c", "humidity_pct", "pressure_hpa",
        "wind_speed_ms", "wind_deg", "clouds_pct", "rain_1h_mm",
        "visibility_m",
        # Engineered weather
        "wind_u", "wind_v", "temp_humidity_index", "pressure_change",
        "is_raining",
        # Pollutants
        "pm25", "pm10", "o3", "no2", "so2", "co",
        # Ratios
        "pm_ratio",
    ]


# ── CLI / Quick test ───────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke test with synthetic data
    logger.info("Running smoke test with synthetic data...")

    now = pd.Timestamp.now(tz="UTC").floor("h")
    timestamps = pd.date_range(end=now, periods=48, freq="h", tz="UTC")

    synthetic = pd.DataFrame({
        "timestamp":     timestamps.astype(str),
        "city":          "Karachi",
        "aqi":           np.random.uniform(50, 180, 48),
        "pm25":          np.random.uniform(20, 80, 48),
        "pm10":          np.random.uniform(30, 120, 48),
        "o3":            np.random.uniform(10, 60, 48),
        "no2":           np.random.uniform(5,  40, 48),
        "so2":           np.random.uniform(1,  20, 48),
        "co":            np.random.uniform(0.1, 2, 48),
        "temp_c":        np.random.uniform(25, 42, 48),
        "feels_like_c":  np.random.uniform(28, 46, 48),
        "humidity_pct":  np.random.uniform(30, 90, 48),
        "pressure_hpa":  np.random.uniform(1005, 1020, 48),
        "wind_speed_ms": np.random.uniform(0, 8, 48),
        "wind_deg":      np.random.uniform(0, 360, 48),
        "clouds_pct":    np.random.uniform(0, 100, 48),
        "rain_1h_mm":    np.random.uniform(0, 2, 48),
        "visibility_m":  np.random.uniform(5000, 10000, 48),
        "weather_main":  "Clear",
        "weather_desc":  "clear sky",
    })

    df_feat = engineer_features(synthetic)
    print(df_feat.tail(3).T)
    print(f"\nTotal features: {len(df_feat.columns)}")
    print(f"Feature names:\n{get_feature_names()}")
