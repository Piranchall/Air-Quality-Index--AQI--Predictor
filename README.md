# 🌫️ Pearls AQI Predictor

An end-to-end machine learning pipeline that forecasts the **Air Quality Index (AQI) for the next 3 days** using a fully serverless architecture.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Hopsworks](https://img.shields.io/badge/Feature_Store-Hopsworks-purple)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

This project builds a fully automated ML system that:
- Fetches real-time weather and pollutant data from external APIs every hour
- Engineers features and stores them in a Hopsworks Feature Store
- Trains and evaluates multiple forecasting models daily
- Serves a 3-day AQI forecast via an interactive web dashboard with SHAP explainability

---

## 🏗️ Architecture

```
External APIs (AQICN / OpenWeather)
        │
        ▼
┌─────────────────────┐
│   Feature Pipeline  │  ← runs every hour via GitHub Actions
│  (fetch + engineer) │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Hopsworks         │
│   Feature Store     │  ← central data hub
└──────┬──────┬───────┘
       │      │
       ▼      ▼
┌──────────┐  ┌──────────────────┐
│ Training │  │    Inference     │
│ Pipeline │  │    Pipeline      │
│(daily)   │  │ (real-time pred) │
└────┬─────┘  └────────┬─────────┘
     │                 │
     ▼                 ▼
┌──────────────┐  ┌──────────────────────┐
│   Model      │  │   Web Dashboard      │
│   Registry   │  │   Streamlit + Flask  │
└──────────────┘  └──────────────────────┘
```

---

## 📁 Project Structure

```
aqi-predictor/
├── feature_pipeline/        # Data fetching + feature engineering
│   ├── fetch_data.py        # API calls to AQICN & OpenWeather
│   ├── engineer_features.py # Compute derived features
│   └── upload_to_store.py   # Push features to Hopsworks
│
├── training_pipeline/       # Model training & evaluation
│   ├── train.py             # Train all models, log metrics
│   ├── evaluate.py          # RMSE, MAE, R² evaluation
│   └── register_model.py    # Save best model to registry
│
├── inference_pipeline/      # Real-time predictions
│   ├── predict.py           # Load model + latest features, run forecast
│   └── alerts.py            # AQI hazard level alerts
│
├── dashboard/               # Web application
│   ├── app.py               # Streamlit dashboard
│   ├── api.py               # Flask/FastAPI backend
│   └── components/          # Dashboard UI components
│
├── notebooks/               # EDA and experiments
│   ├── 01_eda.ipynb
│   └── 02_model_experiments.ipynb
│
├── tests/                   # Unit tests
│
├── data/
│   ├── raw/                 # Raw API responses (gitignored)
│   └── processed/           # Processed feature CSVs (gitignored)
│
├── .github/workflows/       # CI/CD automation
│   ├── feature_pipeline.yml # Runs every hour
│   └── training_pipeline.yml# Runs daily
│
├── .env.example             # Environment variable template
├── requirements.txt         # Python dependencies
├── config.py                # Central config (city, thresholds, etc.)
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/aqi-predictor.git
cd aqi-predictor
```

### 2. Set up a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 3. Configure environment variables
```bash
cp .env.example .env
# Fill in your API keys and Hopsworks credentials
```

### 4. Run the feature pipeline manually (first run)
```bash
python feature_pipeline/fetch_data.py --backfill --days 90
```

### 5. Train the models
```bash
python training_pipeline/train.py
```

### 6. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

---

## 🔑 Required API Keys & Accounts

| Service | Purpose | Free Tier |
|---|---|---|
| [AQICN](https://aqicn.org/api/) | Pollutant data (PM2.5, PM10, O3…) | ✅ Yes |
| [OpenWeatherMap](https://openweathermap.org/api) | Weather data (temp, humidity, wind) | ✅ Yes |
| [Hopsworks](https://www.hopsworks.ai/) | Feature Store + Model Registry | ✅ Yes |

---

## ⚙️ Automated Pipelines (GitHub Actions)

| Workflow | Schedule | What it does |
|---|---|---|
| `feature_pipeline.yml` | Every hour | Fetches + engineers + stores features |
| `training_pipeline.yml` | Daily at midnight | Retrains models, updates registry |

---

## 📊 Models

| Model | Library | Notes |
|---|---|---|
| Random Forest | scikit-learn | Strong baseline, handles non-linearity |
| Ridge Regression | scikit-learn | Fast, good for linear trends |
| LSTM / Neural Net | TensorFlow | Captures temporal patterns |

---

## 📈 Evaluation Metrics

- **RMSE** — Root Mean Square Error (penalises large errors)
- **MAE** — Mean Absolute Error (average error in AQI units)
- **R²** — Coefficient of determination (how much variance explained)

---

## 🧠 Explainability

SHAP values are computed for every prediction, showing which features (humidity, PM2.5, wind speed, etc.) drove the forecast up or down.

---

## 🚨 AQI Alert Levels

| AQI Range | Category | Action |
|---|---|---|
| 0–50 | Good | None |
| 51–100 | Moderate | Sensitive groups take care |
| 101–150 | Unhealthy for sensitive groups | Reduce outdoor activity |
| 151–200 | Unhealthy | Avoid prolonged outdoor exposure |
| 201–300 | Very Unhealthy | Stay indoors |
| 301+ | Hazardous | Emergency conditions |

---

## 🛠️ Tech Stack

`Python` · `scikit-learn` · `TensorFlow` · `Hopsworks` · `GitHub Actions` · `Streamlit` · `Flask` · `AQICN API` · `OpenWeather API` · `SHAP` · `pandas` · `Git`

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
