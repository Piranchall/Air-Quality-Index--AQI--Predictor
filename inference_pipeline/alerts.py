"""
alerts.py — AQI alert levels and health recommendations.

Used by both the inference pipeline and dashboard to
classify AQI predictions and generate health advisories.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AQI_LEVELS


# ── Alert colours (for dashboard UI) ──────────────────────
AQI_COLORS = {
    "Good":                           "#00e400",
    "Moderate":                       "#ffff00",
    "Unhealthy for Sensitive Groups": "#ff7e00",
    "Unhealthy":                      "#ff0000",
    "Very Unhealthy":                 "#8f3f97",
    "Hazardous":                      "#7e0023",
    "Unknown":                        "#cccccc",
}

# ── Health recommendations per level ──────────────────────
AQI_RECOMMENDATIONS = {
    "Good": (
        "Air quality is satisfactory. "
        "Enjoy outdoor activities freely."
    ),
    "Moderate": (
        "Air quality is acceptable. "
        "Unusually sensitive people should consider limiting prolonged outdoor exertion."
    ),
    "Unhealthy for Sensitive Groups": (
        "Members of sensitive groups may experience health effects. "
        "Children, elderly, and people with heart/lung disease should reduce outdoor activity."
    ),
    "Unhealthy": (
        "Everyone may begin to experience health effects. "
        "Sensitive groups should avoid outdoor exertion. "
        "Wear an N95 mask if going outside."
    ),
    "Very Unhealthy": (
        "Health alert — everyone may experience serious health effects. "
        "Avoid all outdoor activities. Keep windows closed."
    ),
    "Hazardous": (
        "Emergency conditions. "
        "Everyone should avoid all outdoor exertion. "
        "Stay indoors with air purifier if available."
    ),
    "Unknown": "AQI level unknown.",
}

# ── Emoji icons per level ──────────────────────────────────
AQI_ICONS = {
    "Good":                           "🟢",
    "Moderate":                       "🟡",
    "Unhealthy for Sensitive Groups": "🟠",
    "Unhealthy":                      "🔴",
    "Very Unhealthy":                 "🟣",
    "Hazardous":                      "⚫",
    "Unknown":                        "⚪",
}


def get_alert(aqi: float) -> dict:
    """
    Given an AQI value, return a full alert dict with:
    - category: string label
    - color: hex color for UI
    - icon: emoji
    - recommendation: health advice string
    - aqi: original value
    - is_hazardous: bool (True if Unhealthy or worse)

    Args:
        aqi: AQI value (float)

    Returns:
        dict with all alert info
    """
    if aqi is None or (isinstance(aqi, float) and aqi != aqi):  # NaN check
        category = "Unknown"
    else:
        category = "Unknown"
        for label, (low, high) in AQI_LEVELS.items():
            if low <= aqi <= high:
                category = label
                break

    return {
        "aqi":            round(float(aqi), 1) if aqi is not None else None,
        "category":       category,
        "color":          AQI_COLORS.get(category, "#cccccc"),
        "icon":           AQI_ICONS.get(category, "⚪"),
        "recommendation": AQI_RECOMMENDATIONS.get(category, ""),
        "is_hazardous":   category not in ("Good", "Moderate", "Unknown"),
    }


def get_forecast_alerts(predictions: list[float]) -> list[dict]:
    """
    Generate alerts for a list of AQI predictions.

    Args:
        predictions: List of AQI float values

    Returns:
        List of alert dicts
    """
    return [get_alert(aqi) for aqi in predictions]


def get_worst_alert(predictions: list[float]) -> dict:
    """
    Return the alert for the worst predicted AQI in the forecast.
    Useful for showing a headline warning on the dashboard.
    """
    if not predictions:
        return get_alert(None)
    return get_alert(max(predictions))


if __name__ == "__main__":
    # Quick test
    test_values = [45, 85, 125, 165, 210, 320]
    for aqi in test_values:
        alert = get_alert(aqi)
        print(f"AQI {aqi:3d} → {alert['icon']} {alert['category']:<35} {alert['color']}")
