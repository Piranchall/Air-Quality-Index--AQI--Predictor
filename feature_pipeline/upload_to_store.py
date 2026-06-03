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
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    HOPSWORKS_HOST,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    CITY_NAME,
)
from feature_pipeline.fetch_data import fetch_current_snapshot, fetch_backfill
from feature_pipeline.engineer_features import engineer_features


# ── Expected schema ────────────────────────────────────────
# All columns the feature group expects — must always be present

FLOAT_COLS = [
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_24h",
    "aqi_rolling_mean_3h", "aqi_rolling_mean_6h", "aqi_rolling_mean_24h",
    "aqi_rolling_std_3h", "aqi_rolling_max_6h",
    "aqi_change_rate", "aqi_change_rate_3h",
    "temp_c", "feels_like_c", "humidity_pct", "pressure_hpa",
    "wind_speed_ms", "wind_deg", "clouds_pct", "rain_1h_mm",
    "wind_u", "wind_v", "temp_humidity_index", "pressure_change",
    "pm_ratio",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
]

INT_COLS = [
    "hour", "day_of_week", "month", "day_of_year",
    "is_weekend", "is_rush_hour", "is_raining",
]


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
        online_enabled=True,
    )

    logger.success(f"Feature group ready: '{fg.name}' v{fg.version}")
    return fg


# ── Schema enforcement ─────────────────────────────────────

def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame matches the feature group schema exactly:
    - All expected columns are present (add as NaN if missing)
    - Correct dtypes for every column
    - Timestamp is datetime64[us, UTC]
    - City column is always present
    """
    df = df.copy()

    # ── Timestamp ──────────────────────────────────────────
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # ── City ───────────────────────────────────────────────
    if "city" not in df.columns:
        df["city"] = CITY_NAME

    # ── Float columns — add as NaN if missing ──────────────
    for col in FLOAT_COLS:
        if col not in df.columns:
            logger.warning(f"Adding missing float column: {col}")
            df[col] = float("nan")
        else:
            df[col] = df[col].astype("float64")

    # ── Int columns — add as 0 if missing ──────────────────
    for col in INT_COLS:
        if col not in df.columns:
            logger.warning(f"Adding missing int column: {col}")
            df[col] = 0
        else:
            df[col] = df[col].astype("int32")

    # visibility_m not in feature group schema — drop it
    if "visibility_m" in df.columns:
        df = df.drop(columns=["visibility_m"])
        
    # ── Drop unexpected extra columns ──────────────────────
    expected_cols = FLOAT_COLS + INT_COLS + ["timestamp", "city"]
    extra_cols = [c for c in df.columns if c not in expected_cols]
    if extra_cols:
        logger.warning(f"Dropping unexpected columns: {extra_cols}")
        df = df.drop(columns=extra_cols)

    return df


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

    # Enforce schema before uploading
    df = enforce_schema(df)

    logger.info(f"Uploading {len(df)} rows to Feature Store...")

    fs = get_feature_store()
    fg = get_or_create_feature_group(fs)

    fg.insert(df, write_options={"wait_for_job": True})

    logger.success(
        f"Upload complete — {len(df)} rows inserted into "
        f"'{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION}"
    )


# ── Full pipeline run ──────────────────────────────────────

# def run_pipeline(backfill: bool = False, days: int = 5, filepath: str = None):
#     """
#     Run the complete feature pipeline end to end:
#       1. Fetch raw data  (or load from file)
#       2. Engineer features
#       3. Enforce schema
#       4. Upload to Feature Store

#     Args:
#         backfill : If True, fetch historical data instead of current snapshot
#         days     : Number of days to backfill (used if backfill=True)
#         filepath : Path to a CSV file to upload directly (skips fetch step)
#     """
#     logger.info("=" * 50)
#     logger.info("Feature Pipeline Starting")
#     logger.info(f"Mode: {'backfill' if backfill else 'snapshot'} | City: {CITY_NAME}")
#     logger.info("=" * 50)

#     # Step 1: Get raw data
#     if filepath:
#         logger.info(f"Loading raw data from file: {filepath}")
#         df_raw = pd.read_csv(filepath)
#     elif backfill:
#         df_raw = fetch_backfill(days=days)
#     else:
#         df_raw = fetch_current_snapshot()

#     # Step 2: Engineer features
#     df_features = engineer_features(df_raw)

#     # Step 3: Save processed data locally (optional, for debugging)
#     os.makedirs("data/processed", exist_ok=True)
#     out_path = (
#         f"data/processed/features_"
#         f"{'backfill' if backfill else 'snapshot'}_"
#         f"{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
#     )
#     df_features.to_csv(out_path, index=False)
#     logger.info(f"Features saved locally to {out_path}")

#     # Step 4: Upload to Hopsworks
#     upload_features(df_features)

#     logger.info("Feature Pipeline Complete ✓")

# ── Full pipeline run ──────────────────────────────────────
 
def run_pipeline(backfill: bool = False, days: int = 5, filepath: str = None):
    """
    Run the complete feature pipeline end to end:
      1. Fetch raw data  (or load from file)
      2. Engineer features
      3. Enforce schema
      4. Upload to Feature Store
 
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
 
    # Step 2: For live snapshots, fetch last 24 rows for lag context
    if not backfill and not filepath:
        try:
            import hopsworks as hw
            logger.info("Fetching last 24 rows for lag context...")
            _project = hw.login(
                host=HOPSWORKS_HOST,
                api_key_value=HOPSWORKS_API_KEY,
                project=HOPSWORKS_PROJECT_NAME,
            )
            _fs = _project.get_feature_store()
            _fg = _fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
            df_history = _fg.read()
            df_history["timestamp"] = pd.to_datetime(df_history["timestamp"], utc=True)
            df_history = df_history.sort_values("timestamp").tail(24)
            raw_cols = [c for c in df_raw.columns if c in df_history.columns]
            df_history = df_history[raw_cols]
            df_raw = pd.concat([df_history, df_raw], ignore_index=True)
            df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)
            logger.success(f"Combined {len(df_raw)} rows for feature engineering")
        except Exception as e:
            logger.warning(f"Could not fetch history: {e} — proceeding without lag context")
 
    # Step 3: Engineer features on combined data
    df_features = engineer_features(df_raw)
 
    # Keep only the new row for upload
    if not backfill and not filepath and len(df_features) > 1:
        df_features = df_features.tail(1).reset_index(drop=True)
        logger.info("Keeping only new row for upload")
 
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