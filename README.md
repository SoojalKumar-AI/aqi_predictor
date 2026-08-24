# Lahore AQI Prediction System

An end-to-end air-quality forecasting system for Lahore. The project collects recent hourly pollution and weather data from [Open-Meteo](https://open-meteo.com/), engineers time-series features, trains machine-learning models, and serves PM2.5 and PM10 forecasts through a FastAPI backend with a Streamlit dashboard.

The system supports forecasts for two horizons:

- PM2.5 three hours ahead: `pm2_5_t+3h`
- PM2.5 six hours ahead: `pm2_5_t+6h`
- PM10 three hours ahead: `pm10_t+3h`
- PM10 six hours ahead: `pm10_t+6h`

## Features

- Hourly air-quality and weather data collection with a configurable rolling history window.
- Feature engineering with calendar features, rolling statistics, changes, interactions, and lags up to 168 hours.
- Regression models from scikit-learn, CatBoost, LightGBM, and XGBoost.
- Model evaluation with cross-validation and regression metrics.
- SHAP feature-importance summaries for model explanations.
- FastAPI endpoints for forecasts, health checks, metadata, time series, EDA data, and explanations.
- Streamlit dashboard pages for forecasts, explanations, and exploratory data analysis.
- Scheduled daily data refresh and retraining through GitHub Actions.

## Project Structure

```text
aqi_predictor/
├── backend/
│   ├── __init__.py
│   ├── eda_schema.py
│   ├── main.py                 # FastAPI application
│   └── utils.py                # Data loading, feature building, and model helpers
├── data/
│   ├── lahore_features_no_targets.csv
│   └── raw_lahore_hourly.csv
├── datasets_per_target/        # Feature-engineered training data by target
├── models/
│   ├── shap/                   # SHAP feature-importance CSV files
│   ├── features_used.json
│   ├── metrics.json
│   └── *.joblib                # Trained model bundles
├── scripts/
│   ├── data_collect_update.py  # Fetch data and build datasets
│   ├── model_pipeline.py       # Training and model-selection pipeline
│   └── train_daily.py          # Train all four targets
├── streamlit_app/
│   ├── Home.py
│   └── pages/
│       ├── 01_Forecast.py
│       ├── 02_Explanations.py
│       └── 03_EDA.py
├── .env.example
├── classification_models.json
├── ensemble_strategies.json
├── regression_models.json
├── render.yaml
└── requirements.txt
```

## Requirements

- Python 3.11 or newer
- Internet access for Open-Meteo data collection
- Enough memory and disk space for model training and SHAP calculations

## Installation

Clone the repository and enter its directory:

```bash
git clone https://github.com/SoojalKumar-AI/aqi_predictor.git
cd aqi_predictor
```

Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration

Copy the example environment file before running the services or pipeline.

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS or Linux:

```bash
cp .env.example .env
```

The scripts read environment variables from the process environment. The `.env` file is a reference for local configuration; load it in your shell or with your preferred environment-variable loader before running commands.

Important settings include:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LAT` | `31.5497` | Collection latitude for Lahore |
| `LON` | `74.3436` | Collection longitude for Lahore |
| `MONTHS_BACK` | `24` | Months of historical data to request |
| `API_URL` | `http://localhost:8080` | Backend URL used by Streamlit |
| `CORS_ORIGINS` | `*` | Comma-separated allowed frontend origins |
| `PROJECT_ROOT` | automatic | Optional project-root override for training |
| `DATASETS_DIR` | `datasets_per_target` | Optional training-dataset directory |
| `MODELS_DIR` | `models` | Optional model-output directory |
| `SHAP_DIR` | `models/shap` | Optional SHAP-output directory |
| `N_ITER_SEARCH` | `20` | Randomized-search iterations |
| `CV_FOLDS` | `5` | Cross-validation folds |
| `K_TOP` | `50` | Maximum selected features |
| `VAR_THR` | `0.0` | Variance threshold |
| `SEARCH_CAP` | `5000` | Maximum rows used during search |
| `BIG_ROWS` | `15000` | Large-dataset threshold |
| `ENSEMBLE_MAX` | `3` | Maximum ensemble size supported by the pipeline |
| `EXPLAIN` | `1` | GitHub Actions training flag for SHAP output |

The data collector uses Open-Meteo and does not require an API key. Do not commit `.env` or place credentials in `.env.example`.

## Run the Data and Training Pipeline

Run these commands from the repository root. Collect data before training so the generated files are current.

Fetch the latest hourly data and generate the target datasets:

```bash
python scripts/data_collect_update.py
```

Train the four configured models and write model bundles, metrics, and SHAP summaries:

```bash
python scripts/train_daily.py
```

The pipeline writes the following outputs:

- `data/raw_lahore_hourly.csv`: merged raw pollution and weather data.
- `data/lahore_features_no_targets.csv`: feature-engineered data without forecast targets.
- `datasets_per_target/*.csv`: one training dataset per forecast target.
- `models/*.joblib`: trained model bundles.
- `models/features_used.json`: ordered feature list used by inference.
- `models/metrics.json`: training and evaluation metrics.
- `models/shap/*.csv`: feature-importance summaries.

## Run the FastAPI Backend

Start the API on port `8080`:

```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8080
```

Open the interactive API documentation at `http://localhost:8080/docs` and verify the service with:

```bash
curl http://localhost:8080/health
```

Useful endpoints:

| Endpoint | Description |
| --- | --- |
| `GET /health` | Service status and loaded-model information |
| `GET /predict?horizon=3` | PM2.5 and PM10 forecasts three hours ahead |
| `GET /predict?horizon=6` | PM2.5 and PM10 forecasts six hours ahead |
| `GET /timeseries?hours=168` | Recent raw time-series values |
| `GET /models` | Stored model metrics |
| `GET /features` | Ordered inference feature list |
| `GET /explanations?target=pm2_5_t+3h&top_k=20` | SHAP feature explanations |
| `GET /eda/raw?hours=168` | Recent raw data for EDA |
| `GET /eda/training?target=pm2_5_t+3h` | Feature-engineered target data |
| `GET /reload` | Reload model files without restarting the API |

## Run the Streamlit Dashboard

Start the API first, then launch the dashboard in a second terminal:

```bash
streamlit run streamlit_app/Home.py --server.port 8501
```

The dashboard opens at `http://localhost:8501`. It uses `API_URL` to find the FastAPI backend. For the local setup shown above, the default value `http://localhost:8080` is correct.

## Automated Daily Retraining

The workflow at `.github/workflows/daily_pipeline.yml` runs every day at `03:00 UTC`. It:

1. Installs Python and the project dependencies.
2. Collects the latest rolling data window.
3. Trains all four targets.
4. Uploads generated data and models as workflow artifacts.
5. Commits updated datasets, models, and metrics back to the repository.

It can also be started manually from the repository's **Actions** tab by selecting **Daily AQI Data & Retrain** and choosing **Run workflow**.

## Deployment with Render

`render.yaml` defines two services:

- `aqi-api`: FastAPI served by Uvicorn on Render's `PORT`.
- `aqi-ui`: Streamlit dashboard configured to call `https://aqi-api.onrender.com`.

After deploying the API, set the dashboard's `API_URL` to the deployed API URL if it differs from the value in `render.yaml`. Set `CORS_ORIGINS` on the API to the dashboard URL when restricting cross-origin access.

