# utils.py
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional
import json
import math
import joblib
from fastapi import HTTPException

# ---------- paths ----------
BASE_DIR   = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "data"
DST_DIR  = BASE_DIR / "datasets_per_target"

TARGETS: set[str] = {"pm2_5_t+3h","pm2_5_t+6h","pm10_t+3h","pm10_t+6h"}

# ---------------- Globals ----------------
MODELS: Dict[str, Any] = {}          # tgt -> loaded estimator or bundle
FEATURES: List[str] = []             # ordered feature list used by training
RAW_DF: Optional[pd.DataFrame] = None


# --------- Paths (resolve relative to repo root when not provided) ----------
def _default_data_dir() -> Path:
    here = Path(__file__).resolve().parents[1]  # project root (…/aqi-prediction-system)
    return here / "data"

# --------- EPA helpers you already had (kept for imports in main.py) --------
def epa_cat_pm25(x: float) -> str:
    if x <= 12: return "Good"
    elif x <= 35.4: return "Moderate"
    elif x <= 55.4: return "Unhealthy for Sensitive Groups"
    elif x <= 150.4: return "Unhealthy"
    elif x <= 250.4: return "Very Unhealthy"
    else: return "Hazardous"

def epa_cat_pm10(x: float) -> str:
    if x <= 54: return "Good"
    elif x <= 154: return "Moderate"
    elif x <= 254: return "Unhealthy for Sensitive Groups"
    elif x <= 354: return "Unhealthy"
    elif x <= 424: return "Very Unhealthy"
    else: return "Hazardous"

def worst_category(*cats: str) -> str:
    order = ["Good","Moderate","Unhealthy for Sensitive Groups","Unhealthy","Very Unhealthy","Hazardous"]
    return max(cats, key=lambda c: order.index(c))

# --------- Core loaders -----------------------------------------------------
def load_raw(data_dir: Path | str | None = None) -> pd.DataFrame:
    """
    Load the latest merged hourly raw file produced by your data collection.
    Prefers data/raw_lahore_hourly.csv; else tries the newest CSV in data/.
    Returns a DataFrame with a parsed 'time' column.
    """
    data_dir = Path(data_dir) if data_dir else _default_data_dir()
    raw_path = data_dir / "raw_lahore_hourly.csv"
    
    if raw_path.exists():
        df = pd.read_csv(raw_path)
    else:
        # fallback: pick most recent CSV in data/
        csvs = list(data_dir.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        raw_path = max(csvs, key=os.path.getmtime)
        df = pd.read_csv(raw_path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df

# --------- Feature engineering mirrors training (minimal) -------------------
def _add_time_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["hour"] = d["time"].dt.hour
    d["weekday"] = d["time"].dt.dayofweek
    d["month"] = d["time"].dt.month
    d["is_weekend"] = (d["weekday"] >= 5).astype(int)

    d["hour_sin"] = np.sin(2*np.pi*d["hour"]/24)
    d["hour_cos"] = np.cos(2*np.pi*d["hour"]/24)
    d["wday_sin"] = np.sin(2*np.pi*d["weekday"]/7)
    d["wday_cos"] = np.cos(2*np.pi*d["weekday"]/7)
    d["month_sin"] = np.sin(2*np.pi*d["month"]/12)
    d["month_cos"] = np.cos(2*np.pi*d["month"]/12)
    return d

def _add_rolls_and_lags(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()

    # basic rolls used in training
    d["pm2_5_avg_3h"]       = d["pm2_5"].rolling(3, min_periods=1).mean()
    d["temperature_avg_3h"] = d["temperature_2m"].rolling(3, min_periods=1).mean()

    # extended stats used in dataset_per_target_export.py
    for col in ["pm2_5","pm10"]:
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

def _rebuild_features_from_raw(data_dir: Path | None, needed_features: list[str]) -> pd.DataFrame:
    """
    Build feature table from raw data, then return the **last row** with at least the
    requested columns present (fills remaining with zeros if still missing).
    """
    df_raw = load_raw(data_dir)
    d = _add_time_features(df_raw)
    d = _add_rolls_and_lags(d)

    # last row should have all lags/rolls (if you have >=168 hours of data)
    last = d.sort_values("time").tail(1).copy()

    # sanitize NaNs for inference
    last = last.replace([np.inf, -np.inf], np.nan)
    # keep 'time' if present, but we only return needed feature columns
    # fill missing required features
    for col in needed_features:
        if col not in last.columns:
            last[col] = 0.0
    last = last[needed_features].fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    return last

# --------- Public: build latest feature row -------------------------------
def build_latest_feature_row(features: list[str]) -> pd.DataFrame:
    """
    Return a **single-row DataFrame** with exactly the `features` columns, NaN-free,
    to be fed into your trained models.
    Strategy:
      1) Prefer the precomputed table `data/lahore_features_no_targets.csv` (fast).
      2) If missing, rebuild features from raw and return the last row.
    """
    data_dir = _default_data_dir()
    precomp = data_dir / "lahore_features_no_targets.csv"
    if precomp.exists():
        df = pd.read_csv(precomp)
        # Best effort: if 'time' exists, use last chronologically
        if "time" in df.columns:
            try:
                df["time"] = pd.to_datetime(df["time"])
                df = df.sort_values("time")
            except Exception:
                pass
        last = df.tail(1).copy()
        # ensure all needed columns exist
        for c in features:
            if c not in last.columns:
                last[c] = 0.0
        # at the end of the function, right before return
        last = last.reindex(columns=features, fill_value=0.0)
        last = last.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return last

    # Fallback: rebuild from raw
    return _rebuild_features_from_raw(data_dir, features)

# ---------------- JSON safety helpers ----------------
def _to_json_safe_scalar(x):
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if x is None:
        return None
    if isinstance(x, (str, bool, np.bool_)):
        return x
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_json_safe(v) for v in obj)
    if isinstance(obj, (pd.Series,)):
        return [to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (pd.DataFrame,)):
        df = obj.replace([np.inf, -np.inf], np.nan)
        df = df.where(pd.notnull(df), None)
        return to_json_safe(df.to_dict("records"))
    return _to_json_safe_scalar(obj)

# ---------------- helpers ----------------
def safe_name(target: str) -> str:
    return target.replace("+", "plus").replace(" ", "_")

def ensure_target(t: str) -> str:
    if t not in TARGETS:
        raise HTTPException(status_code=400, detail=f"Invalid target: {t}")
    return t

def _features_paths() -> list[Path]:
    # try models/features_used.json then project-root/features_used.json
    return [MODELS_DIR / "features_used.json", BASE_DIR / "features_used.json"]

def features_path() -> Path:
    for p in _features_paths():
        if p.exists():
            return p
    # last resort: raise; caller will handle and attempt fallback
    raise FileNotFoundError("features_used.json not found in models/ or project root")

def model_path_for_target(tgt: str) -> Path:
    return MODELS_DIR / f"{safe_name(tgt)}_best.joblib"

def _try_load_json(p: Path) -> Optional[dict]:
    try:
        return json.load(open(p, "r"))
    except Exception:
        return None

# ---------------- Loaders ----------------
def load_features_list() -> List[str]:
    # Prefer explicit file(s)
    for p in _features_paths():
        d = _try_load_json(p)
        if d and "features" in d and isinstance(d["features"], list) and len(d["features"]) > 0:
            return d["features"]

    # Fallback: infer from any per-target dataset (intersection of features across targets)
    ds_dir = BASE_DIR / "datasets_per_target"
    if ds_dir.exists():
        feats_sets = []
        for tgt in TARGETS:
            fp = ds_dir / f"{tgt}.csv"
            if fp.exists():
                df = pd.read_csv(fp, nrows=1)
                cols = [c for c in df.columns if c != tgt and c != "time"]
                feats_sets.append(set(cols))
        if feats_sets:
            feats = sorted(list(set.intersection(*feats_sets))) if len(feats_sets) > 1 else sorted(list(feats_sets[0]))
            if feats:
                return feats

    raise FileNotFoundError("Unable to determine feature list. Provide models/features_used.json or datasets_per_target/*.csv")

def load_models_from_disk() -> Tuple[Dict[str, Any], List[str]]:
    loaded: Dict[str, Any] = {}
    missing: List[str] = []
    for tgt in TARGETS:
        mp = model_path_for_target(tgt)
        if mp.exists():
            try:
                loaded[tgt] = joblib.load(mp)
            except Exception as e:
                print(f"[API] Failed to load {mp}: {e}")
                missing.append(str(mp))
        else:
            missing.append(str(mp))
    return loaded, missing

def init_or_reload():
    global MODELS, FEATURES, RAW_DF
    # features
    try:
        FEATURES[:] = load_features_list()
    except Exception as e:
        print(f"[API] Warning: load_features_list failed: {e}")
        FEATURES.clear()
    # models
    MODELS, missing = load_models_from_disk()
    # raw
    try:
        RAW_DF = load_raw(DATA_DIR)
    except TypeError:
        RAW_DF = load_raw()
    except Exception as e:
        print(f"[API] Warning: load_raw() failed: {e}")
        RAW_DF = None
    return missing
