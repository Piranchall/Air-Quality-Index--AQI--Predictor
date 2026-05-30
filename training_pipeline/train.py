"""
train.py — Training pipeline for AQI forecasting.

Steps:
  1. Fetch historical features from Hopsworks Feature Store
  2. Prepare features and target (next-hour AQI)
  3. Train three models: Random Forest, Ridge Regression, TensorFlow LSTM
  4. Evaluate all models with RMSE, MAE, R²
  5. Save the best model locally (register_model.py pushes it to Hopsworks)

Usage:
  python training_pipeline/train.py
  python training_pipeline/train.py --model rf        # train only Random Forest
  python training_pipeline/train.py --model ridge     # train only Ridge
  python training_pipeline/train.py --model lstm      # train only LSTM
  python training_pipeline/train.py --model xgb       # train only XGBoost
"""

import argparse
import os
import sys
import joblib
from datetime import datetime

import numpy as np
import pandas as pd
import hopsworks
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    HOPSWORKS_HOST,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    TEST_SIZE,
    RANDOM_STATE,
)
from feature_pipeline.engineer_features import get_feature_names
from training_pipeline.evaluate import evaluate_model, compare_models, print_evaluation_table


# ── Constants ─────────────────────────────────────────────
TARGET_COL = "aqi"
MODELS_DIR = "models"


# ── Data loading ──────────────────────────────────────────

def load_features_from_store() -> pd.DataFrame:
    """
    Connect to Hopsworks and load all features from the Feature Store.
    Returns a DataFrame sorted by timestamp.
    """
    logger.info("Connecting to Hopsworks Feature Store...")

    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME,
    )
    fs = project.get_feature_store()

    fg = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )

    logger.info("Reading all features from feature group...")
    df = fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.success(f"Loaded {len(df)} rows from Feature Store")
    return df



def load_features_from_csv(filepath: str = None) -> pd.DataFrame:
    """
    Load features from a local CSV file instead of Hopsworks.
    Useful when Hopsworks is unreachable due to network issues.
    """
    import glob
    if filepath is None:
        files = glob.glob("data/processed/backfill_*.csv")
        if not files:
            files = glob.glob("data/processed/*.csv")
        if not files:
            raise FileNotFoundError(
                "No local CSV files found in data/processed/. "
                "Run backfill_historical.py with --no-upload flag first."
            )
        filepath = max(files, key=os.path.getmtime)

    logger.info(f"Loading features from local CSV: {filepath}")
    df = pd.read_csv(filepath)
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.success(f"Loaded {len(df)} rows from local CSV")
    return df


def prepare_data(df: pd.DataFrame) -> tuple:
    """
    Prepare features and target from the loaded DataFrame.

    Target: next-hour AQI (shift AQI back by 1 so model learns to predict ahead)
    Features: all engineered columns from get_feature_names()

    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info("Preparing features and target...")

    # Target is the NEXT hour's AQI — shift by -1
    df["target_aqi"] = df[TARGET_COL].shift(-1)

    # Drop last row (no target for it) and any rows missing the target
    df = df.dropna(subset=["target_aqi"])

    # Get feature columns that actually exist in the DataFrame
    available_features = [
        col for col in get_feature_names()
        if col in df.columns and col != TARGET_COL
    ]

    logger.info(f"Using {len(available_features)} features")

    X = df[available_features].copy()
    y = df["target_aqi"].copy()

    # Fill any remaining nulls with column median
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=False,  # keep time order — don't shuffle time series data
    )

    logger.success(
        f"Data split — Train: {len(X_train)} rows | Test: {len(X_test)} rows"
    )
    return X_train, X_test, y_train, y_test, available_features


# ── Model training ────────────────────────────────────────

def train_random_forest(X_train, y_train) -> Pipeline:
    """Train a Random Forest model inside a sklearn Pipeline."""
    logger.info("Training Random Forest...")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ))
    ])

    model.fit(X_train, y_train)
    logger.success("Random Forest trained")
    return model


def train_ridge(X_train, y_train) -> Pipeline:
    """Train a Ridge Regression model inside a sklearn Pipeline."""
    logger.info("Training Ridge Regression...")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])

    model.fit(X_train, y_train)
    logger.success("Ridge Regression trained")
    return model



def train_xgboost(X_train, y_train) -> Pipeline:
    """Train an XGBoost model inside a sklearn Pipeline."""
    logger.info("Training XGBoost...")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ))
    ])

    model.fit(X_train, y_train)
    logger.success("XGBoost trained")
    return model


def train_lstm(X_train, y_train, X_test, y_test) -> object:
    """
    Train a simple LSTM model using TensorFlow/Keras.
    Reshapes input to (samples, timesteps=1, features) for LSTM.
    """
    logger.info("Training LSTM (TensorFlow)...")

    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.preprocessing import StandardScaler

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # Reshape for LSTM: (samples, timesteps, features)
        X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
        X_test_lstm  = X_test_scaled.reshape(X_test_scaled.shape[0],  1, X_test_scaled.shape[1])

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(1, X_train_scaled.shape[1])),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )

        model.fit(
            X_train_lstm, y_train,
            validation_data=(X_test_lstm, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0,
        )

        # Wrap in a helper class so it has a .predict(X) interface like sklearn
        class LSTMWrapper:
            def __init__(self, keras_model, scaler):
                self.model  = keras_model
                self.scaler = scaler

            def predict(self, X):
                X_scaled = self.scaler.transform(X)
                X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
                return self.model.predict(X_reshaped, verbose=0).flatten()

        logger.success("LSTM trained")
        return LSTMWrapper(model, scaler)

    except Exception as e:
        logger.warning(f"LSTM training failed: {e} — skipping LSTM")
        return None


# ── Save model ────────────────────────────────────────────

def save_model(model, model_name: str, metrics: dict, feature_names: list) -> str:
    """
    Save the trained model and its metadata to the models/ directory.

    Args:
        model        : Trained model object
        model_name   : Short name like 'random_forest'
        metrics      : Dict with rmse, mae, r2
        feature_names: List of feature column names the model was trained on

    Returns:
        Path to saved model file
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = os.path.join(MODELS_DIR, f"{model_name}_{timestamp}.pkl")

    payload = {
        "model":         model,
        "model_name":    model_name,
        "metrics":       metrics,
        "feature_names": feature_names,
        "trained_at":    timestamp,
    }

    joblib.dump(payload, model_path)
    logger.success(f"Model saved to {model_path}")
    return model_path


# ── Main training run ─────────────────────────────────────

def run_training(model_filter: str = "all", use_local: bool = False, local_file: str = None) -> str:
    """
    Run the full training pipeline.

    Args:
        model_filter: 'all', 'rf', 'ridge', 'lstm', or 'xgb'

    Returns:
        Path to the best saved model file
    """
    logger.info("=" * 55)
    logger.info("Training Pipeline Starting")
    logger.info("=" * 55)

    # Load data
    if use_local:
        df = load_features_from_csv(local_file)
    else:
        df = load_features_from_store()

    if len(df) < 10:
        raise ValueError(
            f"Not enough data to train: only {len(df)} rows in Feature Store. "
            "Run the feature pipeline with --backfill to collect more data."
        )

    X_train, X_test, y_train, y_test, feature_names = prepare_data(df)

    results  = []
    models   = {}

    # ── Random Forest ──────────────────────────────────────
    if model_filter in ("all", "rf"):
        rf = train_random_forest(X_train, y_train)
        rf_metrics = evaluate_model(rf, X_test, y_test, "Random Forest")
        results.append(rf_metrics)
        models["random_forest"] = (rf, rf_metrics)

    # ── Ridge Regression ───────────────────────────────────
    if model_filter in ("all", "ridge"):
        ridge = train_ridge(X_train, y_train)
        ridge_metrics = evaluate_model(ridge, X_test, y_test, "Ridge Regression")
        results.append(ridge_metrics)
        models["ridge"] = (ridge, ridge_metrics)

    # ── XGBoost ───────────────────────────────────────────
    if model_filter in ("all", "xgb"):
        xgb = train_xgboost(X_train, y_train)
        xgb_metrics = evaluate_model(xgb, X_test, y_test, "XGBoost")
        results.append(xgb_metrics)
        models["xgboost"] = (xgb, xgb_metrics)

    # ── LSTM ───────────────────────────────────────────────
    if model_filter in ("all", "lstm"):
        lstm = train_lstm(X_train, y_train, X_test, y_test)
        if lstm is not None:
            lstm_metrics = evaluate_model(lstm, X_test, y_test, "LSTM")
            results.append(lstm_metrics)
            models["lstm"] = (lstm, lstm_metrics)

    # ── Compare and pick best ──────────────────────────────
    print_evaluation_table(results)
    best_result = compare_models(results)
    best_name   = best_result["model_name"].lower().replace(" ", "_")

    # Find matching key in models dict
    best_key = next(k for k in models if best_name.startswith(k) or k in best_name)
    best_model, best_metrics = models[best_key]

    # Save best model
    model_path = save_model(best_model, best_key, best_metrics, feature_names)

    logger.info("Training Pipeline Complete ✓")
    logger.info(f"Best model saved to: {model_path}")
    return model_path


# ── CLI ────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AQI forecasting models")
    parser.add_argument(
        "--model",
        choices=["all", "rf", "ridge", "lstm", "xgb"],
        default="all",
        help="Which model to train (default: all)"
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Load from local CSV instead of Hopsworks (use when network is unavailable)"
    )
    parser.add_argument(
        "--local-file", type=str, default=None,
        help="Path to specific local CSV file (default: latest in data/processed/)"
    )
    args = parser.parse_args()

    run_training(model_filter=args.model, use_local=args.local, local_file=args.local_file)