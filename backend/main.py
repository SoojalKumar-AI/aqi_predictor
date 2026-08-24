# app/main.py
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .utils import (
    epa_cat_pm25, epa_cat_pm10, load_raw,
    build_latest_feature_row, to_json_safe,
    safe_name, ensure_target, features_path,
    model_path_for_target, _try_load_json, init_or_reload,
    FEATURES, MODELS, RAW_DF
)
from backend.eda_schema import EDAFrame

# ---------- paths ----------
BASE_DIR   = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "data"
DST_DIR  = BASE_DIR / "datasets_per_target"

TARGETS: set[str] = {"pm2_5_t+3h","pm2_5_t+6h","pm10_t+3h","pm10_t+6h"}

app = FastAPI(title="Lahore AQI Forecasting API", version="1.0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Startup ----------------
@app.on_event("startup")
def _startup():
    missing = init_or_reload()
    if missing:
        print("[API] Warning: missing model files:\n  " + "\n  ".join(missing))

@app.get("/")
def root():
    return {"ok": True, "message": "AQI API ready"}

# ---------------- Admin ----------------
@app.get("/reload")
def reload_models():
    missing = init_or_reload()
    return JSONResponse(content=to_json_safe({"reloaded": True, "missing_models": missing, "features_len": len(FEATURES)}))

@app.get("/health")
def health():
    missing_targets = [t for t in TARGETS if t not in MODELS]
    return JSONResponse(content=to_json_safe({
        "status": "ok" if not missing_targets else "degraded",
        "loaded_models": list(MODELS.keys()),
        "missing_models": missing_targets,
        "models_dir": str(MODELS_DIR),
        "features_file": str(features_path()) if FEATURES else "N/A",
        "features_count": len(FEATURES),
        "raw_loaded": bool(RAW_DF is not None and not RAW_DF.empty),
    }))

# ---------------- Metadata ----------------
@app.get("/features")
def features():
    return JSONResponse(content=to_json_safe({"features": FEATURES}))

@app.get("/models")
def models_info():
    mp = MODELS_DIR / "metrics.json"
    if mp.exists():
        return JSONResponse(content=to_json_safe(_try_load_json(mp) or {}))
    return JSONResponse(content={})

# ---------------- Timeseries ----------------
@app.get("/timeseries")
def timeseries(hours: int = 168):
    global RAW_DF
    if RAW_DF is None or RAW_DF.empty:
        try:
            RAW_DF = load_raw(DATA_DIR)
        except TypeError:
            RAW_DF = load_raw()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Raw data not available: {e}")

    df = RAW_DF.copy()
    if "time" not in df.columns:
        raise HTTPException(status_code=500, detail="Raw data missing 'time' column.")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").tail(hours)

    # sanitize
    df = df.replace([np.inf, -np.inf], np.nan)
    cols = [c for c in df.columns if c != "time"]
    if cols:
        df[cols] = df[cols].fillna(method="ffill").fillna(method="bfill")
        df[cols] = df[cols].fillna(df[cols].mean()).fillna(0.0)

    payload = {
        "time": df["time"].astype(str).tolist(),
        "pm2_5": df.get("pm2_5", pd.Series([None]*len(df))).tolist(),
        "pm10": df.get("pm10", pd.Series([None]*len(df))).tolist(),
        "ozone": df.get("ozone", pd.Series([None]*len(df))).tolist(),
        "nitrogen_dioxide": df.get("nitrogen_dioxide", pd.Series([None]*len(df))).tolist(),
        "carbon_monoxide": df.get("carbon_monoxide", pd.Series([None]*len(df))).tolist(),
        "sulphur_dioxide": df.get("sulphur_dioxide", pd.Series([None]*len(df))).tolist(),
    }
    return JSONResponse(content=to_json_safe(payload))

# ---------------- SHAP explanations ----------------
@app.get("/explanations")
def explanations(target: str = Query(..., description="e.g. pm2_5_t+3h"), top_k: int = 20):
    t = ensure_target(target)
    p1 = MODELS_DIR / "shap" / f"{t}_shap.csv"
    p2 = MODELS_DIR / "shap" / f"{safe_name(t)}_shap.csv"
    shap_csv = p1 if p1.exists() else (p2 if p2.exists() else None)
    if shap_csv is None:
        raise HTTPException(status_code=404, detail="SHAP summary not found. Re-train with SHAP enabled.")
    try:
        df = pd.read_csv(shap_csv)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read SHAP file: {e}")
    df = df.head(max(1, top_k))
    return JSONResponse(content=to_json_safe({
        "target": t,
        "top_k": top_k,
        "features": df.to_dict("records")
    }))

# ---------------- Prediction ----------------
@app.get("/predict")
def predict(horizon: int = Query(3, enum=[3, 6])):
    # map horizon -> targets
    targets = ["pm2_5_t+3h","pm10_t+3h"] if horizon == 3 else ["pm2_5_t+6h","pm10_t+6h"]

    # Build the latest feature row (must match training FEATURES)
    if not FEATURES:
        raise HTTPException(status_code=503, detail="Feature list not loaded.")
    try:
        row = build_latest_feature_row(FEATURES)  # 1-row DataFrame, NaN-free
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build feature row: {e}")

    out = {}
    for t in targets:
        # 1) prefer preloaded model bundle; 2) fallback to disk
        bundle = MODELS.get(t)
        if bundle is None:
            mp = model_path_for_target(t)
            if not mp.exists():
                out[t] = {"error": "model not available"}
                continue
            try:
                bundle = joblib.load(mp)
            except Exception as e:
                out[t] = {"error": f"failed to load model: {e}"}
                continue

        try:
            # ---------- Case A: stacked ensemble bundle we built for pm10_t+6h ----------
            if isinstance(bundle, dict) and bundle.get("type") == "stacked":
                feats = bundle.get("features", FEATURES)
                X = row[feats] if feats else row
                p1 = bundle["lgbm"].predict(X)
                p2 = bundle["xgb"].predict(X)
                w0, w1 = bundle["weights"]
                pred_val = float(w0 * float(np.asarray(p1)[0]) + w1 * float(np.asarray(p2)[0]))

            # ---------- Case B: plain estimator saved as {"model": est, "features": [...]} ----------
            elif isinstance(bundle, dict) and "model" in bundle:
                est = bundle["model"]
                feats = bundle.get("features")
                X = row[feats] if feats else row
                pred_val = float(np.asarray(est.predict(X))[0])

            # ---------- Case C: AutoMLPipeline rich bundle saved with {"pipeline": pipe, ...} ----------
            elif isinstance(bundle, dict) and "pipeline" in bundle:
                pipe = bundle["pipeline"]
                # drop target col if present; pipeline handles its own preprocessing/selection
                X = row.drop(columns=[t], errors="ignore")
                pred_val = float(np.asarray(pipe.predict(X))[0])

            # ---------- Case D: already a fitted estimator/pipeline ----------
            else:
                # Best effort: treat as sklearn estimator/pipeline
                X = row.drop(columns=[t], errors="ignore")
                pred_val = float(np.asarray(bundle.predict(X))[0])

        except Exception as e:
            out[t] = {"error": f"inference failed: {e}"}
            continue

        # EPA category helper
        cat = epa_cat_pm25(pred_val) if t.startswith("pm2_5") else epa_cat_pm10(pred_val)
        out[t] = {"prediction": pred_val, "category": cat}

    return JSONResponse(content=to_json_safe(out))

# ---------- EDA endpoints (raw + training) ----------
def _df_to_records(df: pd.DataFrame, limit: int | None = None) -> tuple[list[str], list[dict]]:
    d = df.copy()
    d = d.replace([np.inf, -np.inf], np.nan).where(pd.notnull(d), None)
    if limit is not None and limit > 0:
        d = d.tail(limit)
    return d.columns.tolist(), d.to_dict(orient="records")

@app.get("/eda/raw", response_model=EDAFrame)
def eda_raw(hours: int = Query(168, ge=1, le=24*90, description="How many hours (max 90 days)")):
    """Return the merged RAW file produced by your collector."""
    fp = DATA_DIR / "raw_lahore_hourly.csv"
    if not fp.exists():
        raise HTTPException(404, f"Missing {fp}")
    try:
        df = pd.read_csv(fp)
    except Exception as e:
        raise HTTPException(500, f"Failed reading raw file: {e}")

    if "time" not in df.columns:
        raise HTTPException(500, "raw_lahore_hourly.csv missing 'time' column")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").tail(hours)
    start = str(df["time"].min()) if not df.empty else None
    end   = str(df["time"].max()) if not df.empty else None
    df["time"] = df["time"].astype(str)

    cols, recs = _df_to_records(df)
    meta = {
        "source": "Open-Meteo (Air-Quality + Weather), merged by data_collect_update.py",
        "rows": len(recs),
        "start": start, "end": end,
        "note": "Raw snapshot used by the pipeline (no feature engineering)."
    }
    return {"meta": meta, "columns": cols, "records": recs}

@app.get("/eda/training", response_model=EDAFrame)
def eda_training(
    target: str = Query(..., description="pm2_5_t+3h | pm2_5_t+6h | pm10_t+3h | pm10_t+6h"),
    limit: int = Query(20000, ge=100, le=200000, description="Tail rows to return"),
):
    """Return the feature-engineered training table for a target from datasets_per_target/<target>.csv"""
    ensure_target(target)  # you already have this helper
    fp = DST_DIR / f"{target}.csv"
    if not fp.exists():
        raise HTTPException(404, f"Missing training dataset: {fp}")
    try:
        df = pd.read_csv(fp)
    except Exception as e:
        raise HTTPException(500, f"Failed reading dataset: {e}")

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time")
        start = str(df["time"].min()); end = str(df["time"].max())
        df["time"] = df["time"].astype(str)
    else:
        start = end = None

    if len(df) > limit:
        df = df.tail(limit)

    cols, recs = _df_to_records(df)
    meta = {
        "source": "datasets_per_target",
        "target": target,
        "rows": len(recs),
        "start": start, "end": end,
        "note": "Feature-engineered training table for this target (lags/rolls/etc.)."
    }
    return {"meta": meta, "columns": cols, "records": recs}
