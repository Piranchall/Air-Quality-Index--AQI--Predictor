"""
evaluate.py — Model evaluation utilities.

Computes RMSE, MAE, and R² for any trained model.
Used by train.py to compare models and pick the best one.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from loguru import logger


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> dict:
    """
    Evaluate a trained model on test data.

    Args:
        model     : Any sklearn-compatible model (has .predict())
        X_test    : Test features
        y_test    : True AQI values
        model_name: Label for logging

    Returns:
        dict with rmse, mae, r2 scores
    """
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    logger.info(
        f"{model_name} — RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}"
    )

    return {
        "model_name": model_name,
        "rmse": round(rmse, 4),
        "mae":  round(mae,  4),
        "r2":   round(r2,   4),
        "predictions": y_pred.tolist(),
    }


def compare_models(results: list[dict]) -> dict:
    """
    Given a list of evaluation result dicts, return the best one by RMSE.

    Args:
        results: List of dicts returned by evaluate_model()

    Returns:
        The result dict of the best model
    """
    best = min(results, key=lambda r: r["rmse"])
    logger.success(
        f"Best model: {best['model_name']} "
        f"(RMSE: {best['rmse']}, MAE: {best['mae']}, R²: {best['r2']})"
    )
    return best


def print_evaluation_table(results: list[dict]) -> None:
    """Print a formatted comparison table of all model results."""
    print("\n" + "=" * 55)
    print(f"{'Model':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print("-" * 55)
    for r in sorted(results, key=lambda x: x["rmse"]):
        print(
            f"{r['model_name']:<25} "
            f"{r['rmse']:>8.2f} "
            f"{r['mae']:>8.2f} "
            f"{r['r2']:>8.4f}"
        )
    print("=" * 55 + "\n")
