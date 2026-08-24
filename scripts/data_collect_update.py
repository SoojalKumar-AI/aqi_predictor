# scripts/data_collect_update.py
import os
import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

# ------------------ CONFIG ------------------
LAT, LON = float(os.getenv("LAT", 31.5497)), float(os.getenv("LON", 74.3436))  # Lahore
MONTHS_BACK = int(os.getenv("MONTHS_BACK", "24"))  # sliding window
HORIZONS = [3, 6]  # hours ahead

DATA_DIR = "data"
DST_DIR = "datasets_per_target"
MODELS_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DST_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def month_range_utc(months_back: int):
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    end = now - timedelta(hours=1)  # last complete hour
    first_this_month = datetime(end.year, end.month, 1)
    m = first_this_month.month - months_back
    y = first_this_month.year + (m - 1)//12
    m = ((m - 1) % 12) + 1
    start = datetime(y, m, 1)
    return start.date().isoformat(), end.date().isoformat()

def fetch_data():
    start_date, end_date = month_range_utc(MONTHS_BACK)
    print(f"[collect] Fetching {start_date} → {end_date}")

    aq_url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,sulphur_dioxide"
    )
    wx_url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
    )

    aq_json = requests.get(aq_url, timeout=120).json()
    wx_json = requests.get(wx_url, timeout=120).json()

    df_aq = pd.DataFrame(aq_json.get("hourly", {}))
    df_wx = pd.DataFrame(wx_json.get("hourly", {}))

    if "time" not in df_aq or "time" not in df_wx:
        raise RuntimeError("Open-Meteo response missing 'hourly.time'.")

    df = pd.merge(df_aq, df_wx, on="time", how="inner")
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df

def add_time_features(dfin: pd.DataFrame) -> pd.DataFrame:
    d = dfin.copy()
    d["hour"]    = d["time"].dt.hour
    d["weekday"] = d["time"].dt.dayofweek
    d["month"]   = d["time"].dt.month
    d["is_weekend"] = (d["weekday"] >= 5).astype(int)

    d["hour_sin"] = np.sin(2*np.pi*d["hour"]/24)
    d["hour_cos"] = np.cos(2*np.pi*d["hour"]/24)
    d["wday_sin"] = np.sin(2*np.pi*d["weekday"]/7)
    d["wday_cos"] = np.cos(2*np.pi*d["weekday"]/7)
    d["month_sin"] = np.sin(2*np.pi*d["month"]/12)
    d["month_cos"] = np.cos(2*np.pi*d["month"]/12)

    if "wind_direction_10m" in d.columns:
        ang = np.deg2rad(d["wind_direction_10m"])
        d["wind_dir_sin"] = np.sin(ang)
        d["wind_dir_cos"] = np.cos(ang)
    else:
        d["wind_dir_sin"] = 0.0
        d["wind_dir_cos"] = 0.0
    return d

def add_rolls_and_lags(dfin: pd.DataFrame) -> pd.DataFrame:
    d = dfin.copy()
    # short rolling
    d["pm2_5_avg_3h"]       = d["pm2_5"].rolling(3, min_periods=1).mean()
    d["temperature_avg_3h"] = d["temperature_2m"].rolling(3, min_periods=1).mean()

    for col in ["pm2_5", "pm10"]:
        for w in [6, 12]:
            d[f"{col}_mean_{w}h"] = d[col].rolling(w, min_periods=1).mean()
            d[f"{col}_std_{w}h"]  = d[col].rolling(w, min_periods=1).std()
            d[f"{col}_min_{w}h"]  = d[col].rolling(w, min_periods=1).min()
            d[f"{col}_max_{w}h"]  = d[col].rolling(w, min_periods=1).max()

    d["pm2_5_change_rate"] = d["pm2_5"].diff()
    d["pm10_change_rate"]  = d["pm10"].diff()
    d["ozone_change_rate"] = d["ozone"].diff()

    d["pm_ratio"]       = d["pm2_5"] / (d["pm10"] + 1e-6)
    d["temp_pm2_5"]     = d["temperature_2m"] * d["pm2_5"]
    d["humidity_pm2_5"] = d["relative_humidity_2m"] * d["pm2_5"]

    for col in ["pm2_5","pm10","temperature_2m","relative_humidity_2m","ozone","wind_speed_10m"]:
        for lag in [1, 3, 24, 168]:
            d[f"{col}_lag_{lag}h"] = d[col].shift(lag)
    return d

def export_per_target(df: pd.DataFrame):
    # Save raw and FE-no-targets
    raw_path = os.path.join(DATA_DIR, "raw_lahore_hourly.csv")
    df.sort_values("time").to_csv(raw_path, index=False)

    fe_path = os.path.join(DATA_DIR, "lahore_features_no_targets.csv")
    df_fe_no_targets.to_csv(fe_path, index=False)

    # Export per target CSVs
    for target in target_cols:
        cols_to_save = feature_cols + [target]
        out_path = os.path.join(DST_DIR, f"{target}.csv")
        df_model[cols_to_save].to_csv(out_path, index=False)
        print(f"[collect] Saved {out_path} ({df_model.shape[0]} rows)")

# === Run ===
df_raw = fetch_data()

df_fe = add_time_features(df_raw)
df_fe = add_rolls_and_lags(df_fe)

# Targets
for h in HORIZONS:
    df_fe[f"pm2_5_t+{h}h"] = df_fe["pm2_5"].shift(-h)
    df_fe[f"pm10_t+{h}h"]  = df_fe["pm10"].shift(-h)

base_cols = [
    "carbon_monoxide","nitrogen_dioxide","ozone","sulphur_dioxide",
    "temperature_2m","relative_humidity_2m","wind_speed_10m",
    "hour","weekday","month","is_weekend",
    "pm2_5_avg_3h","temperature_avg_3h",
    "pm2_5_change_rate","pm10_change_rate","ozone_change_rate",
    "pm_ratio","temp_pm2_5","humidity_pm2_5",
    "hour_sin","hour_cos","wday_sin","wday_cos","month_sin","month_cos",
    "wind_dir_sin","wind_dir_cos",
    "pm2_5_mean_6h","pm2_5_std_6h","pm2_5_min_6h","pm2_5_max_6h",
    "pm2_5_mean_12h","pm2_5_std_12h","pm2_5_min_12h","pm2_5_max_12h",
    "pm10_mean_6h","pm10_std_6h","pm10_min_6h","pm10_max_6h",
    "pm10_mean_12h","pm10_std_12h","pm10_min_12h","pm10_max_12h",
]
lag_cols = [c for c in df_fe.columns if c.endswith(("_lag_1h","_lag_3h","_lag_24h","_lag_168h"))]
feature_cols = base_cols + lag_cols

target_cols = [f"pm2_5_t+{h}h" for h in HORIZONS] + [f"pm10_t+{h}h" for h in HORIZONS]
df_model = df_fe.dropna(subset=feature_cols + target_cols).reset_index(drop=True)

# This FE copy (without targets) for EDA
df_fe_no_targets = df_fe.drop(columns=target_cols, errors="ignore")

export_per_target(df_raw)

# Save features list for API
with open(os.path.join(MODELS_DIR, "features_used.json"), "w") as f:
    json.dump({"features": feature_cols}, f, indent=2)

print("[collect] Done.")
