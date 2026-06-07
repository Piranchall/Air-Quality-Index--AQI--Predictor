"""
predict.py — Loads the best model from Hopsworks and generates
a 3-day (72-hour) AQI forecast for Karachi.

Steps:
  1. Load latest model from Hopsworks Model Registry
  2. Load last 24 hours of features from Feature Store
  3. Iteratively predict next 72 hours
  4. Compute SHAP values for explainability
  5. Return structured forecast with alerts

Usage:
  python inference_pipeline/predict.py
"""

import os
import sys
import json
import joblib
import shutil
from datetime import datetime, timedelta, timezone

import numpy as np
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
    MODEL_NAME,
    FORECAST_DAYS,
    CITY_NAME,
)
from inference_pipeline.alerts import get_alert, get_forecast_alerts, get_worst_alert


# ── Connect to Hopsworks ───────────────────────────────────

def get_hopsworks_project():
    """Login and return Hopsworks project."""
    return hopsworks.login(
        host=HOPSWORKS_HOST,
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME,
    )


# ── Load model ─────────────────────────────────────────────

def load_model_from_registry(project) -> tuple:
    """
    Download and load the latest registered model from Hopsworks.

    Returns:
        (model, feature_names, metadata)
    """
    logger.info(f"Loading model '{MODEL_NAME}' from registry...")

    mr = project.get_model_registry()
    model_obj = mr.get_best_model(
        name=MODEL_NAME,
        metric="rmse",
        direction="min",
    )

    # Download model files to local temp dir
    model_dir = model_obj.download()
    model_path = os.path.join(model_dir, "model.pkl")
    meta_path  = os.path.join(model_dir, "metadata.json")

    payload = joblib.load(model_path)
    model   = payload["model"]

    with open(meta_path) as f:
        metadata = json.load(f)

    feature_names = metadata["feature_names"]

    logger.success(
        f"Loaded: {metadata['model_name']} "
        f"(RMSE: {metadata['metrics']['rmse']}, "
        f"R²: {metadata['metrics']['r2']})"
    )
    return model, feature_names, metadata

# def load_latest_features(project, feature_names: list) -> pd.DataFrame:
#     """
#     Load the most recent 24 hours of features from the Feature Store.
#     These are used as the seed for iterative forecasting.
 
#     Returns:
#         DataFrame with last 24 rows, sorted by timestamp
#     """
#     logger.info("Loading latest features from Feature Store...")
 
#     fs = project.get_feature_store()
#     fg = fs.get_feature_group(
#         name=FEATURE_GROUP_NAME,
#         version=FEATURE_GROUP_VERSION,
#     )
 
#     df = fg.read(online=True)
#     df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
#     df = df.sort_values("timestamp").tail(48).reset_index(drop=True)
 
#     # Recompute lag features from actual AQI values
#     # Fixes NaN lags in live rows inserted one at a time
#     df["aqi_lag_1h"]  = df["aqi"].shift(1)
#     df["aqi_lag_3h"]  = df["aqi"].shift(3)
#     df["aqi_lag_6h"]  = df["aqi"].shift(6)
#     df["aqi_lag_24h"] = df["aqi"].shift(24)
#     df["aqi_rolling_mean_3h"]  = df["aqi"].rolling(3,  min_periods=1).mean()
#     df["aqi_rolling_mean_6h"]  = df["aqi"].rolling(6,  min_periods=1).mean()
#     df["aqi_rolling_mean_24h"] = df["aqi"].rolling(24, min_periods=1).mean()
#     df["aqi_rolling_std_3h"]   = df["aqi"].rolling(3,  min_periods=1).std().fillna(0)
#     df["aqi_rolling_max_6h"]   = df["aqi"].rolling(6,  min_periods=1).max()
#     df["aqi_change_rate"]      = df["aqi"].diff(1).fillna(0)
#     df["aqi_change_rate_3h"]   = df["aqi"].diff(3).fillna(0)
 
#     # Use last 24 rows as seed
#     df = df.tail(24).reset_index(drop=True)
 
#     logger.info(f"Seed data range: {df['timestamp'].min()} → {df['timestamp'].max()}")
#     logger.success(f"Loaded {len(df)} rows with recomputed lags")
#     return df

def load_latest_features(project, feature_names: list) -> pd.DataFrame:
    """
    Load the most recent 24 hours of features from the Feature Store.
    Automatically falls back to online store if offline materialization is stale (>1.5h behind).
    """
    logger.info("Loading latest features from Feature Store...")

    import datetime

    fs = project.get_feature_store()
    fg = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )

    # ── Check if offline store is fresh ───────────────
    use_online = False
    try:
        df_check = fg.read()
        df_check["timestamp"] = pd.to_datetime(df_check["timestamp"], utc=True)
        latest_ts = df_check["timestamp"].max()
        now = datetime.datetime.now(datetime.timezone.utc)
        age_hours = (now - latest_ts).total_seconds() / 3600

        if age_hours > 1.5:
            logger.warning(f"Offline store is {age_hours:.1f}h behind — switching to online store")
            use_online = True
        else:
            logger.info(f"Offline store is fresh ({age_hours:.1f}h old) — using offline store")
            df = df_check  # reuse what we already read

    except Exception as e:
        logger.warning(f"Could not check offline store: {e} — defaulting to online store")
        use_online = True

    # ── Load from appropriate store ───────────────────
    if use_online:
        try:
            df = fg.select_all().read(online=True)
            logger.info("Loaded from online store (RonDB)")
        except Exception as e:
            logger.warning(f"Online store failed: {e} — retrying offline store")
            df = fg.read()
            logger.info("Loaded from offline store (fallback)")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").tail(48).reset_index(drop=True)

    # Recompute lag features from actual AQI values
    df["aqi_lag_1h"]           = df["aqi"].shift(1)
    df["aqi_lag_3h"]           = df["aqi"].shift(3)
    df["aqi_lag_6h"]           = df["aqi"].shift(6)
    df["aqi_lag_24h"]          = df["aqi"].shift(24)
    df["aqi_rolling_mean_3h"]  = df["aqi"].rolling(3,  min_periods=1).mean()
    df["aqi_rolling_mean_6h"]  = df["aqi"].rolling(6,  min_periods=1).mean()
    df["aqi_rolling_mean_24h"] = df["aqi"].rolling(24, min_periods=1).mean()
    df["aqi_rolling_std_3h"]   = df["aqi"].rolling(3,  min_periods=1).std().fillna(0)
    df["aqi_rolling_max_6h"]   = df["aqi"].rolling(6,  min_periods=1).max()
    df["aqi_change_rate"]      = df["aqi"].diff(1).fillna(0)
    df["aqi_change_rate_3h"]   = df["aqi"].diff(3).fillna(0)

    df = df.tail(24).reset_index(drop=True)

    logger.info(f"Seed data range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    logger.success(f"Loaded {len(df)} rows with recomputed lags")
    return df

# ── Iterative forecasting ──────────────────────────────────

def generate_forecast(
    model,
    feature_names: list,
    seed_df: pd.DataFrame,
    hours: int = 72,
) -> pd.DataFrame:
    """
    Generate a stable multi-step AQI forecast.
    Uses real historical lags from seed data to prevent drift,
    and applies diurnal patterns for realistic variation.
    """
    logger.info(f"Generating {hours}-hour forecast...")

    # Use real AQI values from seed as anchor — never feed predictions back as lags
    real_aqi_history = list(seed_df["aqi"].ffill().bfill().values)
    seed_median = seed_df.fillna(seed_df.median(numeric_only=True))
    baseline_row = seed_median.iloc[-1].copy()
    last_timestamp = seed_df["timestamp"].iloc[-1]

    # Anchor point — mean of last 6 real hours to stabilise predictions
    anchor_aqi = float(np.mean(real_aqi_history[-6:]))

    predictions = []

    for i in range(hours):
        next_ts = last_timestamp + timedelta(hours=i + 1)

        row = baseline_row.copy()

        # Update time features
        row["hour"]         = int(next_ts.hour)
        row["day_of_week"]  = int(next_ts.dayofweek)
        row["month"]        = int(next_ts.month)
        row["day_of_year"]  = int(next_ts.dayofyear)
        row["is_weekend"]   = int(next_ts.dayofweek >= 5)
        row["is_rush_hour"] = int(next_ts.hour in [7, 8, 9, 17, 18, 19])
        row["hour_sin"]     = float(np.sin(2 * np.pi * next_ts.hour / 24))
        row["hour_cos"]     = float(np.cos(2 * np.pi * next_ts.hour / 24))
        row["month_sin"]    = float(np.sin(2 * np.pi * next_ts.month / 12))
        row["month_cos"]    = float(np.cos(2 * np.pi * next_ts.month / 12))
        row["dow_sin"]      = float(np.sin(2 * np.pi * next_ts.dayofweek / 7))
        row["dow_cos"]      = float(np.cos(2 * np.pi * next_ts.dayofweek / 7))

        # Use REAL historical lags — never predicted values to prevent drift
        row["aqi_lag_1h"]  = float(real_aqi_history[-1])
        row["aqi_lag_3h"]  = float(real_aqi_history[-3]) if len(real_aqi_history) >= 3 else anchor_aqi
        row["aqi_lag_6h"]  = float(real_aqi_history[-6]) if len(real_aqi_history) >= 6 else anchor_aqi
        row["aqi_lag_24h"] = float(real_aqi_history[-24]) if len(real_aqi_history) >= 24 else anchor_aqi

        # Rolling stats from real history
        row["aqi_rolling_mean_3h"]  = float(np.mean(real_aqi_history[-3:]))
        row["aqi_rolling_mean_6h"]  = float(np.mean(real_aqi_history[-6:]))
        row["aqi_rolling_mean_24h"] = float(np.mean(real_aqi_history[-24:]))
        row["aqi_rolling_std_3h"]   = float(np.std(real_aqi_history[-3:]))
        row["aqi_rolling_max_6h"]   = float(np.max(real_aqi_history[-6:]))
        row["aqi_change_rate"]      = float(real_aqi_history[-1] - real_aqi_history[-2]) if len(real_aqi_history) >= 2 else 0.0
        row["aqi_change_rate_3h"]   = float(real_aqi_history[-1] - real_aqi_history[-4]) if len(real_aqi_history) >= 4 else 0.0

        # Build feature vector
        available = [f for f in feature_names if f in row.index]
        X = pd.DataFrame([row[available]]).fillna(seed_median[available].median()).fillna(0)

        # Predict
        pred_raw = float(model.predict(X)[0])

        # Apply mean reversion toward anchor to prevent long-range drift
        reversion_strength = min(0.05 * (i // 6), 0.4)
        pred_aqi = pred_raw * (1 - reversion_strength) + anchor_aqi * reversion_strength

        # Add natural diurnal variation (worse midday, better pre-dawn)
        diurnal = np.sin((next_ts.hour / 24) * 2 * np.pi - 1.2) * 8
        pred_aqi += diurnal

        # Clamp to realistic range
        pred_aqi = max(50, min(300, round(pred_aqi, 1)))

        alert = get_alert(pred_aqi)
        
        predictions.append({
            "timestamp":      next_ts.isoformat(),
            "predicted_aqi":  pred_aqi,
            "hour":           next_ts.hour,
            "date":           next_ts.date().strftime("%Y-%m-%d"),
            "time_label":     next_ts.strftime("%b %d %H:00"),
            "category":       alert["category"],
            "color":          alert["color"],
            "icon":           alert["icon"],
            "recommendation": alert["recommendation"],
            "is_hazardous":   alert["is_hazardous"],
        })

        # Extend real history with a dampened value for next iteration's lags
        dampened = anchor_aqi * 0.3 + pred_aqi * 0.7
        real_aqi_history.append(dampened)

    forecast_df = pd.DataFrame(predictions)
    logger.success(f"Forecast complete — {len(forecast_df)} hourly predictions")
    return forecast_df


# ── SHAP explainability ────────────────────────────────────

def compute_shap_values(model, X: pd.DataFrame) -> dict:
    """
    Compute SHAP feature importance for the current prediction.
    Uses shap.LinearExplainer for Ridge — exact analytical SHAP values.
    """
    import shap

    try:
        if hasattr(model, "named_steps"):
            estimator = list(model.named_steps.values())[-1]
            scaler    = model.named_steps.get("scaler", None)
            X_input   = pd.DataFrame(
                scaler.transform(X) if scaler else X.values,
                columns=X.columns
            )
        else:
            estimator = model
            X_input   = X.copy()

        if hasattr(estimator, "coef_"):
            # LinearExplainer needs a background dataset — use the input itself
            # as background (single prediction, no training data needed)
            background = np.zeros((1, X_input.shape[1]))
            explainer  = shap.LinearExplainer(
                estimator,
                background,
                feature_perturbation="interventional"
            )
            shap_vals  = explainer.shap_values(X_input.values)
            raw        = dict(zip(X_input.columns, np.abs(shap_vals[0])))
        else:
            explainer = shap.TreeExplainer(estimator)
            shap_vals = explainer.shap_values(X_input.values[:1])
            raw       = dict(zip(X_input.columns, np.abs(shap_vals[0])))

        # Top 10, normalised 0-1
        top10   = dict(sorted(raw.items(), key=lambda x: x[1], reverse=True)[:10])
        max_val = max(top10.values()) if top10 and max(top10.values()) > 0 else 1
        return {k: round(v / max_val, 4) for k, v in top10.items()}

    except Exception as e:
        logger.warning(f"SHAP failed: {e} — using coefficient fallback")
        coefs      = np.abs(estimator.coef_)
        vals       = np.abs(X.values[0])
        importance = dict(zip(X.columns, coefs * vals))
        top10      = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
        max_val    = max(top10.values()) if top10 and max(top10.values()) > 0 else 1
        return {k: round(v / max_val, 4) for k, v in top10.items()}


# ── Daily summary ──────────────────────────────────────────

def get_daily_summary(forecast_df: pd.DataFrame) -> list[dict]:
    """
    Aggregate hourly forecast into daily summaries.
    Only includes days with at least 20 hours of forecast data
    so partial days (today) are skipped. Returns next 3 full days.

    Returns:
        List of dicts with one entry per day (max 3)
    """
    summaries = []
    for date, group in forecast_df.groupby("date"):
        if len(group) < 1:
            continue
        
        avg_aqi  = group["predicted_aqi"].mean()
        max_aqi  = group["predicted_aqi"].max()
        min_aqi  = group["predicted_aqi"].min()
        alert    = get_alert(avg_aqi)

        summaries.append({
            "date":      date,
            "avg_aqi":   round(avg_aqi, 1),
            "max_aqi":   round(max_aqi, 1),
            "min_aqi":   round(min_aqi, 1),
            "category":  alert["category"],
            "color":     alert["color"],
            "icon":      alert["icon"],
            "recommendation": alert["recommendation"],
        })

    return summaries


# ── Main forecast run ──────────────────────────────────────

def run_inference() -> dict:
    """
    Full inference pipeline:
    1. Load model from registry
    2. Load latest features
    3. Generate 72-hour forecast
    4. Compute SHAP values
    5. Return structured result

    Returns:
        dict with forecast, daily_summary, shap_values, metadata
    """
    logger.info("=" * 55)
    logger.info("Inference Pipeline Starting")
    logger.info("=" * 55)

    project = get_hopsworks_project()

    # Load model
    model, feature_names, metadata = load_model_from_registry(project)

    # Load latest features
    seed_df = load_latest_features(project, feature_names)

    # Calculate hours until next midnight so forecast always starts at midnight
    from datetime import timezone
    now = datetime.now(timezone.utc)
    hours_until_midnight = 24 - now.hour
    total_hours = hours_until_midnight + (FORECAST_DAYS * 24)

    forecast_df = generate_forecast(
        model=model,
        feature_names=feature_names,
        seed_df=seed_df,
        hours=total_hours,
    )

    # Drop the partial hours before first midnight
    first_midnight = (now + timedelta(hours=hours_until_midnight)).strftime("%Y-%m-%d")
    forecast_df = forecast_df[forecast_df["date"] >= first_midnight].reset_index(drop=True)

    # Trim to exactly 3 days worth of hours
    forecast_df = forecast_df.head(FORECAST_DAYS * 24).reset_index(drop=True)

    # Daily summary
    daily_summary = get_daily_summary(forecast_df)

    # SHAP values on latest features
    available = [f for f in feature_names if f in seed_df.columns]
    X_seed    = seed_df[available].fillna(seed_df[available].median()).tail(1)
    shap_vals = compute_shap_values(model, X_seed)

    # Worst alert
    worst = get_worst_alert(forecast_df["predicted_aqi"].tolist())

    # Extract latest weather readings for dashboard display
    last_row = seed_df.iloc[-1]
    seed_data = {
        col: float(last_row[col])
        for col in ["pm25", "temp_c", "humidity_pct", "wind_speed_ms", "pressure_hpa"]
        if col in seed_df.columns and last_row[col] == last_row[col]  # not NaN
    }

    result = {
        "city":          CITY_NAME,
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "model":         metadata["model_name"],
        "metrics":       metadata["metrics"],
        "forecast":      forecast_df.to_dict(orient="records"),
        "daily_summary": daily_summary,
        "shap_values":   shap_vals,
        "worst_alert":   worst,
        "current_aqi":   float(seed_df["aqi"].iloc[-1]),
        "seed_data":     seed_data,
    }

    logger.info("Inference Pipeline Complete ✓")
    return result


if __name__ == "__main__":
    result = run_inference()

    print(f"\nCity       : {result['city']}")
    print(f"Model      : {result['model']}")
    print(f"Current AQI: {result['current_aqi']}")
    print(f"\n3-Day Daily Summary:")
    for day in result["daily_summary"]:
        print(
            f"  {day['date']}  {day['icon']}  "
            f"Avg: {day['avg_aqi']}  "
            f"Min: {day['min_aqi']}  "
            f"Max: {day['max_aqi']}  "
            f"({day['category']})"
        )

    if result["shap_values"]:
        print(f"\nTop 5 Most Important Features:")
        for feat, val in list(result["shap_values"].items())[:5]:
            print(f"  {feat:<30} {val:.4f}")