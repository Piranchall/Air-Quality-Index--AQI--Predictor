"""
upload_to_store.py — Pushes engineered features to the Hopsworks Feature Store.

This is the final step of the feature pipeline:
  fetch_data.py → engineer_features.py → upload_to_store.py

The Feature Store acts as the central hub — both the training pipeline
and inference pipeline read from here.

Usage:
  # Upload current snapshot (called every hour by GitHub Actions)
  python feature_pipeline/upload_to_store.py

  # Upload a backfill CSV
  python feature_pipeline/upload_to_store.py --file data/processed/backfill.csv
"""

import argparse
import sys
import os
from datetime import datetime

import pandas as pd
import hopsworks
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    HOPSWORKS_HOST,
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    CITY_NAME,
)
from feature_pipeline.fetch_data import fetch_current_snapshot, fetch_backfill
from feature_pipeline.engineer_features import engineer_features


# ── Hopsworks connection ───────────────────────────────────

def get_feature_store():
    """Connect to Hopsworks and return the Feature Store object."""
    logger.info(f"Connecting to Hopsworks project: '{HOPSWORKS_PROJECT_NAME}'...")
    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME,
    )
    fs = project.get_feature_store()
    logger.success("Connected to Hopsworks Feature Store")
    return fs


def get_or_create_feature_group(fs):
    """
    Get existing feature group or create it if it doesn't exist yet.
    Feature groups are like versioned tables in the Feature Store.
    """
    logger.info(f"Getting/creating feature group: '{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION}...")

    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description=(
            f"Hourly AQI and weather features for {CITY_NAME}. "
            "Includes pollutant readings, weather conditions, "
            "lag features, rolling stats, and time-based features."
        ),
        primary_key=["timestamp", "city"],
        event_time="timestamp",
        online_enabled=True,   # enables real-time reads for inference
    )

    logger.success(f"Feature group ready: '{fg.name}' v{fg.version}")
    return fg


# ── Upload ─────────────────────────────────────────────────

def upload_features(df: pd.DataFrame) -> None:
    """
    Upload a feature DataFrame to the Hopsworks Feature Store.

    Args:
        df: Engineered feature DataFrame (output of engineer_features())
    """
    if df.empty:
        logger.warning("Empty DataFrame — nothing to upload.")
        return

    # Hopsworks 4.7+ requires timestamp as datetime64, not string
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Add city column if missing (needed for primary key)
    if "city" not in df.columns:
        df["city"] = CITY_NAME

    # Drop any fully-null columns — Hopsworks doesn't accept them on first insert
    null_cols = [c for c in df.columns if df[c].isnull().all()]
    if null_cols:
        logger.warning(f"Dropping fully-null columns: {null_cols}")
        df = df.drop(columns=null_cols)

    logger.info(f"Uploading {len(df)} rows to Feature Store...")

    fs = get_feature_store()
    fg = get_or_create_feature_group(fs)

    # Ensure correct dtypes before inserting
    float_cols = ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
                "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_24h",
                "pm_ratio", "pressure_hpa", "temp_c", "feels_like_c",
                "humidity_pct", "wind_speed_ms", "wind_deg", "clouds_pct",
                "rain_1h_mm", "visibility_m"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    fg.insert(df, write_options={"wait_for_job": True})

    logger.success(
        f"Upload complete — {len(df)} rows inserted into "
        f"'{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION}"
    )


# ── Full pipeline run ──────────────────────────────────────

def run_pipeline(backfill: bool = False, days: int = 5, filepath: str = None):
    """
    Run the complete feature pipeline end to end:
      1. Fetch raw data  (or load from file)
      2. Engineer features
      3. Upload to Feature Store

    Args:
        backfill : If True, fetch historical data instead of current snapshot
        days     : Number of days to backfill (used if backfill=True)
        filepath : Path to a CSV file to upload directly (skips fetch step)
    """
    logger.info("=" * 50)
    logger.info("Feature Pipeline Starting")
    logger.info(f"Mode: {'backfill' if backfill else 'snapshot'} | City: {CITY_NAME}")
    logger.info("=" * 50)

    # Step 1: Get raw data
    if filepath:
        logger.info(f"Loading raw data from file: {filepath}")
        df_raw = pd.read_csv(filepath)
    elif backfill:
        df_raw = fetch_backfill(days=days)
    else:
        df_raw = fetch_current_snapshot()

    # Step 2: Engineer features
    df_features = engineer_features(df_raw)

    # Step 3: Save processed data locally (optional, for debugging)
    os.makedirs("data/processed", exist_ok=True)
    out_path = (
        f"data/processed/features_"
        f"{'backfill' if backfill else 'snapshot'}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    )
    df_features.to_csv(out_path, index=False)
    logger.info(f"Features saved locally to {out_path}")

    # Step 4: Upload to Hopsworks
    upload_features(df_features)

    logger.info("Feature Pipeline Complete ✓")


# ── CLI ────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AQI feature pipeline")
    parser.add_argument(
        "--backfill", action="store_true",
        help="Run in backfill mode (fetch historical data)"
    )
    parser.add_argument(
        "--days", type=int, default=5,
        help="Number of days to backfill (default: 5)"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Path to a raw CSV file to process and upload instead of fetching from API"
    )
    args = parser.parse_args()

    run_pipeline(
        backfill=args.backfill,
        days=args.days,
        filepath=args.file,
    )
