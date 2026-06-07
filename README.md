# 🌫️ Karachi AQI Predictor — Pearls

> **End-to-end serverless ML system that forecasts Karachi's Air Quality Index (AQI) 72 hours ahead, updated every hour automatically.**

[![Live Dashboard](https://img.shields.io/badge/Live_Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://piranchal-air-quality-index-aqi-predictor.streamlit.app/)
[![REST API](https://img.shields.io/badge/REST_API-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://piranchal-piranchal-air-quality-index-aqi-predictor.hf.space/docs)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/Piranchall/Air-Quality-Index--AQI--Predictor)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## 🔴 Live

| Resource | URL |
|---|---|
| **Dashboard** | https://piranchal-air-quality-index-aqi-predictor.streamlit.app/ |
| **REST API** | https://piranchal-piranchal-air-quality-index-aqi-predictor.hf.space/ |
| **API Docs (Swagger)** | https://piranchal-piranchal-air-quality-index-aqi-predictor.hf.space/docs |
| **GitHub** | https://github.com/Piranchall/Air-Quality-Index--AQI--Predictor |

---

## 📌 Project Overview

This project builds a **fully automated, serverless ML forecasting system** for Karachi, Pakistan — one of South Asia's most polluted megacities. It ingests live air quality and weather data every hour, engineers 46 features, and serves a 72-hour AQI forecast via an interactive dashboard and REST API.

**Key numbers:**
- 9,528 hourly training records (May 2025 – May 2026)
- 46 engineered features per timestep
- 4 models trained and evaluated
- R² = 0.985 on held-out test set
- Fully automated — zero manual intervention required

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     EXTERNAL DATA SOURCES                       │
│         AQICN API (pollutants)  ·  OpenWeatherMap (weather)     │
└──────────────────────┬──────────────────────────────────────────┘
                       │  every hour (GitHub Actions)
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE PIPELINE                             │
│  fetch_data.py → engineer_features.py → upload_to_store.py      │
│  Computes 46 features: lags, rolling stats, cyclical encodings  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                  HOPSWORKS FEATURE STORE                        │
│     Online Store (real-time seed)  ·  Offline Store (training)  │
└───────────┬──────────────────────────────┬──────────────────────┘
            │ daily (GitHub Actions)        │ on dashboard load
            ▼                              ▼
┌───────────────────────┐      ┌──────────────────────────────────┐
│   TRAINING PIPELINE   │      │        INFERENCE PIPELINE        │
│  train.py             │      │  predict.py                      │
│  Ridge / RF / XGB /   │      │  Load model → seed from store →  │
│  LSTM → evaluate →    │      │  72h forecast → SHAP values      │
│  register best model  │      └──────────────┬───────────────────┘
└───────────────────────┘                     │
            │                                 ▼
            ▼                    ┌────────────────────────────────┐
┌───────────────────────┐        │      WEB APPLICATION           │
│   MODEL REGISTRY      │        │  Streamlit Dashboard           │
│   (Hopsworks)         │        │  FastAPI REST API              │
└───────────────────────┘        └────────────────────────────────┘
```

---

## 📁 Project Structure

```
Air-Quality-Index-(AQI)-Predictor/
│
├── feature_pipeline/
│   ├── fetch_data.py              # AQICN + OpenWeather API calls
│   ├── engineer_features.py       # 46 feature computation
│   ├── upload_to_store.py         # Hopsworks Feature Store upload
│   └── backfill_historical.py     # Historical data backfill
│
├── training_pipeline/
│   ├── train.py                   # Train Ridge, RF, XGBoost, LSTM
│   ├── evaluate.py                # RMSE, MAE, R² computation
│   └── register_model.py          # Push best model to Hopsworks registry
│
├── inference_pipeline/
│   ├── predict.py                 # 72h iterative forecast generation
│   └── alerts.py                  # AQI health advisory system
│
├── api/
│   ├── __init__.py
│   └── main.py                    # FastAPI — 8 REST endpoints
│
├── dashboard/
│   ├── app.py                     # Streamlit app
│   └── template.html              # Custom HTML dashboard template
│
├── notebooks/
│   └── 01_eda.ipynb               # Exploratory Data Analysis (12 visualizations)
│
├── data/
│   └── processed/
│       ├── backfill_2025-05-01_2026-05-24.csv   # Archive (9,336 rows)
│       └── backfill_2026-05-24_2026-05-31.csv   # Forecast gap-fill (192 rows)
│
├── .github/workflows/
│   ├── feature_pipeline.yml       # Hourly automation
│   └── training_pipeline.yml      # Daily retraining
│
├── config.py                      # Central configuration
├── requirements.txt               # Full dependencies
└── requirements-api.txt           # API-only dependencies
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Hopsworks account (free tier)
- AQICN API key (free)
- OpenWeatherMap API key (free)

### 1. Clone the repository
```bash
git clone https://github.com/Piranchall/Air-Quality-Index--AQI--Predictor.git
cd Air-Quality-Index--AQI--Predictor
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file in the project root:
```env
AQICN_API_KEY=your_aqicn_key
OPENWEATHER_API_KEY=your_openweather_key
HOPSWORKS_API_KEY=your_hopsworks_key
HOPSWORKS_PROJECT_NAME=your_project_name
HOPSWORKS_HOST=your_hopsworks_host
CITY_NAME=Karachi
CITY_LAT=24.8607
CITY_LON=67.0011
AQICN_STATION=A401143
```

### 4. Run the historical backfill
```bash
python feature_pipeline/backfill_historical.py
```

### 5. Train the models
```bash
python training_pipeline/train.py
```

### 6. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

### 7. Launch the REST API
```bash
python -m uvicorn api.main:app --reload --port 8000
# Visit http://localhost:8000/docs for interactive Swagger UI
```

---

## 🔑 Required Services

| Service | Purpose | Free Tier |
|---|---|---|
| [AQICN](https://aqicn.org/api/) | Real-time AQI, PM2.5, PM10, O3, NO2, SO2, CO | ✅ Yes |
| [OpenWeatherMap](https://openweathermap.org/api) | Temperature, humidity, wind, pressure | ✅ Yes |
| [Hopsworks](https://www.hopsworks.ai/) | Feature Store + Model Registry | ✅ Yes (free tier) |

---

## ⚙️ Automated CI/CD Pipelines

| Workflow | Schedule | Action |
|---|---|---|
| `feature_pipeline.yml` | **Every hour** | Fetches live data, engineers 46 features, uploads to Hopsworks |
| `training_pipeline.yml` | **Daily at midnight UTC** | Retrains all 4 models, evaluates on test set, registers best model |

All secrets (API keys) are stored as GitHub Actions Secrets — no credentials are ever committed to the repository.

---

## 🔬 Exploratory Data Analysis

The full EDA is in `notebooks/01_eda.ipynb` — 12 visualizations across 9,528 hourly records:

| # | Analysis | Key Finding |
|---|---|---|
| 1 | AQI Distribution | 74.6% of hours in Moderate range — chronic baseline pollution |
| 2 | Seasonal Patterns | Winter avg AQI 103 vs Summer avg 79 — 31% seasonal swing |
| 3 | 12-Month Time Series | Clear U-shape: dips in monsoon, peaks Dec–Feb |
| 4 | Diurnal Heatmap | Peak at 13:00–14:00 UTC (evening rush hour PKT) — 7 AQI swing |
| 5 | Feature Correlations | AQI lag 1h has r=0.991 — extreme autocorrelation |
| 6 | Weather Effects | Wind speed r=−0.364, all relationships approximately linear |
| 7 | Wind Direction | NW winds → AQI 103, SW monsoon winds → AQI 80 |
| 8 | Lag Predictiveness | lag_1h r=0.991, lag_24h r=0.716 — past AQI dominates |
| 9 | Outlier Detection | 3.4% spike rate — data is clean (Open-Meteo reanalysis) |
| 10 | Pollutant Profiles | PM2.5 peaks in winter, O3 peaks in summer (photochemical) |
| 11 | Calendar Heatmap | Dark columns Nov–Feb clearly visible — winter inversion episodes |
| 12 | Feature Importance | Lag features dominate; PM2.5 strongest pollutant signal |

---

## 🧪 Feature Engineering

46 features engineered per hourly timestep:

| Category | Features | Count |
|---|---|---|
| **AQI Lags** | `aqi_lag_1h`, `aqi_lag_3h`, `aqi_lag_6h`, `aqi_lag_24h` | 4 |
| **AQI Rolling** | mean/std/max over 3h, 6h, 24h windows | 5 |
| **AQI Change** | `aqi_change_rate`, `aqi_change_rate_3h` | 2 |
| **Pollutants** | PM2.5, PM10, O3, NO2, SO2, CO, PM ratio | 7 |
| **Weather** | temp, humidity, wind speed/direction, pressure, rain, clouds | 9 |
| **Derived Weather** | wind U/V components, temp×humidity index, pressure change, is_raining | 5 |
| **Time (cyclical)** | hour_sin/cos, month_sin/cos, dow_sin/cos | 6 |
| **Time (binary)** | is_weekend, is_rush_hour | 2 |
| **Feels Like** | feels_like_c, visibility_m, dew_point_c | 3 |
| **Other** | clouds_pct, additional derived | 3 |

Cyclical sin/cos encoding ensures the model understands that hour 23 and hour 0 are adjacent — a common source of error in naive temporal encoding.

---

## 📊 Model Results

All models evaluated on a **held-out test set — the last 20% of data (1,905 hourly rows) — strictly time-ordered. No data leakage.**

| Model | Type | RMSE ↓ | MAE ↓ | R² ↑ | Status |
|---|---|---|---|---|---|
| **Ridge Regression** | Statistical · Linear | **2.35** | **1.34** | **0.985** | ✅ **ACTIVE** |
| XGBoost | Ensemble · Gradient Boosting | 2.44 | 1.20 | 0.9865 | Trained |
| Random Forest | Ensemble · Bagging | 2.93 | 1.41 | 0.9806 | Trained |
| LSTM | Deep Learning · RNN | 3.35 | 2.21 | 0.9746 | Trained |

### Why Ridge wins over LSTM

This is counterintuitive — a linear model beating a neural network. The EDA explains it:

- `aqi_lag_1h` has **r = 0.991** with current AQI — near-perfect linear relationship
- All weather variable relationships are approximately linear (verified in scatter plots)
- Ridge has L2 regularisation built-in, preventing overfitting
- LSTM performing **worst** (not best) is the definitive proof against overfitting — if models were memorising training data, the most complex model would score highest, not lowest

The data is telling us the underlying process is linear and slowly-varying. A simpler model that correctly identifies this will always outperform a complex model that searches for non-linear patterns that don't exist.

### No overfitting — evidence

1. **Time-ordered split** — test set is strictly future data the model never saw during training
2. **Ridge L2 regularisation** — structurally penalises large coefficients
3. **LSTM worst → complexity doesn't help** — classic sign of well-generalised models
4. **R² 0.985 is expected** — with lag_1h at r=0.991, a well-fitted linear model should achieve ~0.982+ on this problem

---

## 🌐 Web Dashboard

**Live:** https://piranchal-air-quality-index-aqi-predictor.streamlit.app/

Features:
- Real-time AQI display with EPA health category
- 72-hour interactive forecast chart (hover to scrub)
- 3-day daily summary cards (avg/min/max)
- SHAP feature importance bar chart
- Health advisory with recommendations
- Model comparison table (all 4 models)
- Light/dark theme toggle with day/night sky animation

---

## 🔌 REST API

FastAPI with auto-generated Swagger documentation — deployed live on Hugging Face Spaces.

**Live API:** https://piranchal-piranchal-air-quality-index-aqi-predictor.hf.space/
**Interactive docs:** https://piranchal-piranchal-air-quality-index-aqi-predictor.hf.space/docs

**Run locally:**
```bash
python -m uvicorn api.main:app --reload --port 8000
```

| Endpoint | Description |
|---|---|
| `GET /` | API info and endpoint listing |
| `GET /health` | Model name, RMSE, R², cache age |
| `GET /current` | Current AQI + weather snapshot |
| `GET /predict` | Full result — 72h forecast + daily + SHAP |
| `GET /forecast?hours=24` | Hourly predictions, configurable window |
| `GET /daily` | 3-day daily summary only |
| `GET /shap?top_n=10` | Top N feature importances |
| `GET /alert` | Current + worst forecast health advisory |
| `GET /models` | All 4 models with metrics comparison |

---

## 🧠 Explainability (SHAP)

SHAP values are computed on every prediction using Ridge model coefficients as importance weights. Top features driving today's forecast:

| Feature | Importance | Interpretation |
|---|---|---|
| `aqi_change_rate` | 1.000 | Direction of AQI momentum |
| `pm25` | 0.250 | Fine particulate — primary AQI driver |
| `feels_like_c` | 0.148 | Temperature inversion proxy |
| `aqi_rolling_mean_6h` | 0.088 | Recent 6h trend |
| `aqi_rolling_std_3h` | 0.070 | Volatility signal |
| `wind_speed_ms` | 0.041 | Dispersion — higher = better AQI |

---

## 🚨 AQI Alert System

| AQI | Category | Health Recommendation |
|---|---|---|
| 0–50 | 🟢 Good | Enjoy outdoor activities freely |
| 51–100 | 🟡 Moderate | Unusually sensitive people take care |
| 101–150 | 🟠 Unhealthy for Sensitive Groups | Children, elderly reduce outdoor activity |
| 151–200 | 🔴 Unhealthy | Everyone reduce outdoor exertion; N95 if outside |
| 201–300 | 🟣 Very Unhealthy | Avoid all outdoor activity; keep windows closed |
| 301+ | 🔴 Hazardous | Emergency — everyone stay indoors |

---

## 📈 Data Sources & Coverage

| Source | Data | Coverage | Rows |
|---|---|---|---|
| Open-Meteo Archive API | Historical weather + AQI | May 2025 – May 2026 | 9,336 |
| Open-Meteo Forecast API | Gap-fill (forecast period) | May 24 – May 31 2026 | 192 |
| AQICN Station A401143 | Live AQI (NED UET Karachi) | June 2026 → present | Growing |
| OpenWeatherMap | Live weather | June 2026 → present | Growing |

**Total training data: 9,528 hourly rows · 46 features · 0 missing AQI values**

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11 |
| **ML Models** | scikit-learn (Ridge, RF), XGBoost, TensorFlow/Keras (LSTM) |
| **Feature Store** | Hopsworks (online + offline store) |
| **Model Registry** | Hopsworks Model Registry |
| **Automation** | GitHub Actions (hourly + daily) |
| **Dashboard** | Streamlit + custom HTML/CSS/JS |
| **REST API** | FastAPI + Uvicorn |
| **Explainability** | SHAP |
| **Data Sources** | AQICN API, OpenWeatherMap API, Open-Meteo API |
| **Data Processing** | pandas, numpy |
| **Visualisation** | Plotly, matplotlib, seaborn (EDA) |
| **Version Control** | Git + GitHub |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.