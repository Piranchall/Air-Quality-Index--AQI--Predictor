"""
api/main.py — FastAPI REST interface for the Karachi AQI Predictor.

Exposes the inference pipeline as a JSON API so external systems
(mobile apps, monitoring tools, other dashboards) can consume
predictions without needing the Streamlit interface.

Run locally:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET /              → API info
    GET /health        → model + feature store status
    GET /current       → current AQI + weather snapshot
    GET /predict       → full 72h forecast + daily summary + SHAP
    GET /forecast      → hourly forecast only (lightweight)
    GET /daily         → 3-day daily summary only
    GET /shap          → feature importance only
    GET /alert         → current health advisory
    GET /docs          → Swagger UI (auto-generated)
"""

import sys
import os
import time
from datetime import datetime, timezone
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import (
    CITY_NAME, CITY_LAT, CITY_LON,
    FORECAST_DAYS, MODEL_NAME,
)

# ── App setup ─────────────────────────────────────────────
app = FastAPI(
    title="Karachi AQI Predictor API",
    description=(
        "REST API for the Karachi Air Quality Index Predictor. "
        "Provides real-time AQI readings, 72-hour forecasts, daily summaries, "
        "and SHAP-based feature importance explanations. "
        "Powered by a Ridge Regression model trained on Open-Meteo historical data."
    ),
    version="1.0.0",
    contact={
        "name": "Piranchal Mukesh",
        "url": "https://github.com/Piranchal",
    },
    license_info={
        "name": "MIT",
    },
)

# ── CORS — allow dashboard and any frontend to call the API ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── In-memory cache — avoid re-running inference on every request ──
_cache: dict = {
    "result":     None,
    "fetched_at": None,
    "ttl":        3600,   # refresh every 60 minutes
}


def _get_inference_result() -> dict:
    """
    Return cached inference result, refreshing if stale or missing.
    TTL = 60 minutes so predictions stay fresh without hammering Hopsworks.
    """
    now = time.time()
    if (
        _cache["result"] is None
        or _cache["fetched_at"] is None
        or (now - _cache["fetched_at"]) > _cache["ttl"]
    ):
        try:
            from inference_pipeline.predict import run_inference
            result = run_inference()
            _cache["result"]     = result
            _cache["fetched_at"] = now
        except Exception as e:
            if _cache["result"] is not None:
                # Return stale cache rather than error if inference fails
                return _cache["result"]
            raise HTTPException(
                status_code=503,
                detail=f"Inference pipeline unavailable: {str(e)}"
            )
    return _cache["result"]


# ── Response models ───────────────────────────────────────
class HourlyForecast(BaseModel):
    timestamp:     str
    predicted_aqi: float
    hour:          int
    date:          str
    time_label:    str
    category:      str
    color:         str
    icon:          str
    recommendation: str
    is_hazardous:  bool


class DailySummary(BaseModel):
    date:        str
    avg_aqi:     float
    min_aqi:     float
    max_aqi:     float
    category:    str
    color:       str
    icon:        str
    recommendation: str


class ShapValue(BaseModel):
    feature:    str
    importance: float


class CurrentSnapshot(BaseModel):
    city:        str
    latitude:    float
    longitude:   float
    current_aqi: float
    category:    str
    pm25:        Optional[float]
    temp_c:      Optional[float]
    humidity_pct: Optional[float]
    wind_speed_ms: Optional[float]
    pressure_hpa: Optional[float]
    generated_at: str


class HealthStatus(BaseModel):
    status:      str
    model:       str
    rmse:        Optional[float]
    r2:          Optional[float]
    city:        str
    cache_age_s: Optional[float]
    generated_at: str


# ── Helpers ───────────────────────────────────────────────
def _category(aqi: float) -> str:
    if aqi <= 50:  return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very Unhealthy"
    return "Hazardous"


# ══════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════

from fastapi.responses import HTMLResponse

@app.get("/", tags=["Info"], response_class=HTMLResponse)
def root():
    """API landing page."""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
  <title>Karachi AQI Predictor API</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: #0f111a; color: #eef1ff; font-family: 'Segoe UI', sans-serif;
           display: flex; align-items: center; justify-content: center; min-height: 100vh; }
    .card { background: rgba(28,34,68,.6); border: 1px solid rgba(150,170,255,.14);
            border-radius: 24px; padding: 48px; max-width: 620px; width: 90%; backdrop-filter: blur(16px); }
    h1 { font-size: 28px; font-weight: 800; margin-bottom: 8px;
         background: linear-gradient(135deg, #7c5cff, #22d3ee); -webkit-background-clip: text;
         -webkit-text-fill-color: transparent; }
    .sub { color: #a6aede; font-size: 14px; margin-bottom: 32px; }
    .badge { display: inline-block; background: rgba(34,211,238,.15);
             border: 1px solid rgba(34,211,238,.3); color: #22d3ee;
             padding: 4px 12px; border-radius: 100px; font-size: 11px;
             letter-spacing: .1em; margin-bottom: 32px; }
    .endpoints { display: flex; flex-direction: column; gap: 10px; }
    .ep:hover { background: rgba(124,92,255,.12); border-color: rgba(124,92,255,.4); }
    .method { background: rgba(124,92,255,.25); color: #a78bfa;
              padding: 3px 8px; border-radius: 6px; font-size: 10px;
              font-weight: 700; letter-spacing: .1em; flex-shrink: 0; }
    .path { font-family: monospace; font-size: 13px; color: #22d3ee; flex-shrink: 0; }
    .desc { font-size: 12px; color: #6f78ad; margin-left: auto; text-align: right; }
    .docs-btn { display: block; margin-top: 24px; text-align: center; padding: 14px;
                background: linear-gradient(135deg, #7c5cff, #22d3ee); border-radius: 12px;
                font-weight: 700; font-size: 14px; text-decoration: none; color: white;
                letter-spacing: .05em; }
    .docs-btn:hover { opacity: .9; }
    .city { color: #6f78ad; font-size: 12px; margin-top: 24px; text-align: center; }
  </style>
</head>
<body>
  <div class="card">
    <div class="badge">🌫️ LIVE · KARACHI AQI PREDICTOR</div>
    <h1>AQI Forecast API</h1>
    <p class="sub">72-hour air quality predictions for Karachi, Pakistan · Ridge Regression · R² = 0.985</p>
    <a class="docs-btn" href="/docs">📖 Open Interactive API Docs →</a>
    <p class="city">📍 Karachi · 24.8607°N 67.0011°E · Updated every hour</p>
  </div>
</body>
</html>
""")


@app.get("/health", response_model=HealthStatus, tags=["Status"])
def health():
    """
    Check model and pipeline health.
    Returns model metrics and cache freshness.
    Used by monitoring systems to verify the API is operational.
    """
    cache_age = None
    model_name = MODEL_NAME
    rmse = r2 = None

    if _cache["fetched_at"] is not None:
        cache_age = round(time.time() - _cache["fetched_at"], 1)

    if _cache["result"] is not None:
        r = _cache["result"]
        model_name = r.get("model", MODEL_NAME)
        metrics = r.get("metrics", {})
        rmse = metrics.get("rmse")
        r2   = metrics.get("r2")

    return HealthStatus(
        status       = "ok",
        model        = model_name,
        rmse         = rmse,
        r2           = r2,
        city         = CITY_NAME,
        cache_age_s  = cache_age,
        generated_at = datetime.now(timezone.utc).isoformat(),
    )


@app.get("/current", response_model=CurrentSnapshot, tags=["AQI"])
def current():
    """
    Current AQI reading and weather snapshot for Karachi.

    Returns the most recent observed AQI along with weather conditions
    (temperature, humidity, wind speed, pressure, PM2.5).
    Data is sourced from AQICN and OpenWeatherMap APIs.
    """
    result = _get_inference_result()

    seed  = result.get("seed_data", {})
    aqi   = result.get("current_aqi", 0)

    return CurrentSnapshot(
        city          = result.get("city", CITY_NAME),
        latitude      = CITY_LAT,
        longitude     = CITY_LON,
        current_aqi   = aqi,
        category      = _category(aqi),
        pm25          = seed.get("pm25"),
        temp_c        = seed.get("temp_c"),
        humidity_pct  = seed.get("humidity_pct"),
        wind_speed_ms = seed.get("wind_speed_ms"),
        pressure_hpa  = seed.get("pressure_hpa"),
        generated_at  = result.get("generated_at", datetime.now(timezone.utc).isoformat()),
    )


@app.get("/predict", tags=["Forecast"])
def predict():
    """
    Full inference result — 72h hourly forecast, 3-day daily summary, and SHAP values.

    This is the primary endpoint. Returns everything the dashboard displays:
    - Current AQI and weather
    - 72 hourly predictions with category and health advice
    - 3-day daily averages with min/max envelope
    - Top feature importances from Ridge model coefficients
    - Worst alert level in the forecast window
    """
    result = _get_inference_result()
    return JSONResponse(content=result)


@app.get("/forecast", tags=["Forecast"])
def forecast(
    hours: int = Query(default=72, ge=1, le=72, description="Number of forecast hours to return (1–72)")
):
    """
    Hourly AQI forecast for the next N hours (default 72).

    Each record contains: timestamp, predicted_aqi, category, health recommendation.
    Use the `hours` query parameter to request a subset, e.g. `/forecast?hours=24`.
    """
    result  = _get_inference_result()
    records = result.get("forecast", [])[:hours]

    return {
        "city":         result.get("city", CITY_NAME),
        "model":        result.get("model", MODEL_NAME),
        "hours":        len(records),
        "generated_at": result.get("generated_at"),
        "forecast":     records,
    }


@app.get("/daily", tags=["Forecast"])
def daily():
    """
    3-day daily AQI summary.

    Returns one record per day with average, minimum, and maximum predicted AQI,
    along with the dominant health category and recommendation for that day.
    Useful for mobile apps or notification systems that only need daily granularity.
    """
    result = _get_inference_result()

    return {
        "city":          result.get("city", CITY_NAME),
        "model":         result.get("model", MODEL_NAME),
        "generated_at":  result.get("generated_at"),
        "daily_summary": result.get("daily_summary", []),
    }


@app.get("/shap", tags=["Explainability"])
def shap(
    top_n: int = Query(default=10, ge=1, le=50, description="Number of top features to return")
):
    """
    SHAP feature importance for the current prediction.

    Returns the top N features ranked by their contribution to today's AQI prediction.
    Uses Ridge model coefficients as a proxy for SHAP values — each value represents
    the relative weight the model places on that feature.

    Useful for understanding *why* the model is predicting a particular AQI level.
    """
    result = _get_inference_result()
    shap_dict = result.get("shap_values", {})

    # Sort by importance descending, take top N
    sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]

    return {
        "city":         result.get("city", CITY_NAME),
        "model":        result.get("model", MODEL_NAME),
        "generated_at": result.get("generated_at"),
        "top_n":        top_n,
        "shap_values": [
            {"feature": feat, "importance": round(val, 4)}
            for feat, val in sorted_shap
        ],
    }


@app.get("/alert", tags=["AQI"])
def alert():
    """
    Current and worst forecast health advisory.

    Returns:
    - Current AQI category and health recommendation
    - Worst AQI category expected in the next 72 hours
    - Whether any hazardous conditions are forecast
    - Hours until the worst conditions are expected

    Use this endpoint to power push notifications or alert banners.
    """
    result = _get_inference_result()

    current_aqi  = result.get("current_aqi", 0)
    worst        = result.get("worst_alert", {})
    forecast     = result.get("forecast", [])

    # Find hour index of worst AQI
    worst_hour = None
    if forecast:
        max_aqi  = max(f["predicted_aqi"] for f in forecast)
        worst_hour = next(
            (i + 1 for i, f in enumerate(forecast) if f["predicted_aqi"] == max_aqi),
            None
        )

    from inference_pipeline.alerts import get_alert
    current_alert = get_alert(current_aqi)

    return {
        "city":           result.get("city", CITY_NAME),
        "generated_at":   result.get("generated_at"),
        "current": {
            "aqi":            current_aqi,
            "category":       current_alert["category"],
            "recommendation": current_alert["recommendation"],
            "color":          current_alert["color"],
            "is_hazardous":   current_alert["is_hazardous"],
        },
        "forecast_worst": {
            "category":       worst.get("category"),
            "aqi":            worst.get("aqi"),
            "is_hazardous":   worst.get("is_hazardous", False),
            "hours_until":    worst_hour,
        },
        "action_required": worst.get("is_hazardous", False),
    }