"""
register_model.py — Registers the best trained model in the Hopsworks Model Registry.

The Model Registry stores versioned models so:
  - the inference pipeline always loads the latest best model
  - you can roll back to any previous version
  - metrics are tracked alongside each version

Usage:
  # Register the latest model in models/ directory automatically
  python training_pipeline/register_model.py

  # Register a specific model file
  python training_pipeline/register_model.py --model-path models/random_forest_20260528_1900.pkl
"""

import argparse
import os
import sys
import glob
import json
import shutil
import joblib
from datetime import datetime

import hopsworks
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    HOPSWORKS_HOST,
    MODEL_NAME,
)

MODELS_DIR = "models"


# ── Helpers ───────────────────────────────────────────────

def get_latest_model_path() -> str:
    """Find the most recently saved model file in models/ directory."""
    pattern = os.path.join(MODELS_DIR, "*.pkl")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"No model files found in '{MODELS_DIR}/'. "
            "Run training_pipeline/train.py first."
        )

    latest = max(files, key=os.path.getmtime)
    logger.info(f"Latest model file: {latest}")
    return latest


def load_model_payload(model_path: str) -> dict:
    """Load the model payload dict saved by train.py."""
    payload = joblib.load(model_path)
    logger.info(
        f"Loaded model: {payload['model_name']} | "
        f"RMSE: {payload['metrics']['rmse']} | "
        f"R²: {payload['metrics']['r2']}"
    )
    return payload


# ── Registration ──────────────────────────────────────────

def register_model(model_path: str = None) -> None:
    """
    Upload and register a trained model to the Hopsworks Model Registry.

    Args:
        model_path: Path to the .pkl file. If None, uses the latest file in models/
    """
    if model_path is None:
        model_path = get_latest_model_path()

    payload = load_model_payload(model_path)
    metrics = payload["metrics"]

    # ── Connect to Hopsworks ───────────────────────────────
    logger.info("Connecting to Hopsworks...")
    project = hopsworks.login(
        host=HOPSWORKS_HOST,
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME,
    )

    mr = project.get_model_registry()
    logger.success("Connected to Model Registry")

    # ── Create a temp directory with model artifacts ───────
    # Hopsworks uploads a directory, not a single file
    model_dir = os.path.join(MODELS_DIR, "registry_upload")
    os.makedirs(model_dir, exist_ok=True)

    # Copy model file into upload dir
    shutil.copy(model_path, os.path.join(model_dir, "model.pkl"))

    # Save feature names and metrics as JSON alongside the model
    metadata = {
        "model_name":    payload["model_name"],
        "trained_at":    payload["trained_at"],
        "feature_names": payload["feature_names"],
        "metrics":       metrics,
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Uploading model to registry as '{MODEL_NAME}'...")

    # ── Register in Hopsworks ──────────────────────────────
    model = mr.python.create_model(
        name=MODEL_NAME,
        metrics={
            "rmse": metrics["rmse"],
            "mae":  metrics["mae"],
            "r2":   metrics["r2"],
        },
        description=(
            f"AQI forecaster — {payload['model_name']} — "
            f"trained {payload['trained_at']} — "
            f"RMSE: {metrics['rmse']}"
        ),
    )

    model.save(model_dir)

    logger.success(
        f"Model registered successfully!\n"
        f"  Name   : {MODEL_NAME}\n"
        f"  Version: {model.version}\n"
        f"  RMSE   : {metrics['rmse']}\n"
        f"  MAE    : {metrics['mae']}\n"
        f"  R²     : {metrics['r2']}"
    )

    # Clean up temp upload dir
    shutil.rmtree(model_dir)


# ── CLI ────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Register the best trained model to Hopsworks Model Registry"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a specific .pkl model file (default: latest in models/)"
    )
    args = parser.parse_args()

    register_model(model_path=args.model_path)
