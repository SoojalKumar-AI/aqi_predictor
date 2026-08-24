# main_all.py
# Train all four AQI targets, and write models/metrics.json in the requested format.

import os
import json
from pathlib import Path
from datetime import datetime, timezone
from model_pipeline import AutoMLPipeline

# ---------- Resolve project root robustly ----------
def find_project_root() -> Path:
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        if (p / "datasets_per_target").exists():
            return p
    here = Path(__file__).resolve().parent
    cwd  = Path.cwd()
    for base in [here, here.parent, here.parent.parent, cwd, cwd.parent, cwd.parent.parent]:
        if (base / "datasets_per_target").exists():
            return base.resolve()
    return here.resolve()

ROOT = find_project_root()

# ---------- Directories (env overrides allowed) ----------
DATASETS_DIR = Path(os.getenv("DATASETS_DIR", str(ROOT / "datasets_per_target"))).resolve()
MODELS_DIR   = Path(os.getenv("MODELS_DIR",   str(ROOT / "models"))).resolve()
SHAP_DIR     = Path(os.getenv("SHAP_DIR",     str(MODELS_DIR / "shap"))).resolve()
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SHAP_DIR.mkdir(parents=True, exist_ok=True)

METRICS_JSON = MODELS_DIR / "metrics.json"

# ---------- Targets & their single best model ----------
TARGETS = ["pm2_5_t+3h", "pm2_5_t+6h", "pm10_t+3h", "pm10_t+6h"]
BEST_MODEL_PER_TARGET = {
    "pm2_5_t+3h": "catboost_regressor",
    "pm2_5_t+6h": "lightgbm_regressor",
    "pm10_t+3h":  "lightgbm_regressor",
    "pm10_t+6h":  "xgboost_regressor",  # single best here
}

def safe_name(t: str) -> str:
    return t.replace("+", "plus").replace(" ", "_")

# ---------- Tuning / CV knobs (env overridable) ----------
N_ITER_SEARCH       = int(os.getenv("N_ITER_SEARCH", "20"))
CV_FOLDS            = int(os.getenv("CV_FOLDS", "5"))
K_TOP               = int(os.getenv("K_TOP", "50"))
VARIANCE_THRESHOLD  = float(os.getenv("VAR_THR", "0.0"))
SEARCH_SAMPLE_CAP   = int(os.getenv("SEARCH_CAP", "5000"))
BIG_ROWS_THRESHOLD  = int(os.getenv("BIG_ROWS", "15000"))
ENSEMBLE_MAX        = int(os.getenv("ENSEMBLE_MAX", "3"))  # unused in single-model mode

def load_existing_metrics() -> dict:
    if METRICS_JSON.exists():
        try:
            return json.load(open(METRICS_JSON, "r", encoding="utf-8"))
        except Exception:
            pass
    return {}

def write_metrics_blob(blob: dict):
    json.dump(blob, open(METRICS_JSON, "w", encoding="utf-8"), indent=2)
    print(f"[OK] wrote metrics → {METRICS_JSON}")

def train_target(target: str, metrics_accum: dict):
    """Train one target, save model/shap, and update metrics_accum in-place."""
    model_key = BEST_MODEL_PER_TARGET[target]
    data_path = (DATASETS_DIR / f"{target}.csv").resolve()
    save_path = (MODELS_DIR / f"{safe_name(target)}_best.joblib").resolve()
    shap_csv  = (SHAP_DIR   / f"{safe_name(target)}_shap.csv").resolve()
    best_json = (MODELS_DIR / f"{safe_name(target)}_best_params.json").resolve()

    print("\n==================================================")
    print(f"[target]       {target}")
    print(f"[model_key]    {model_key}")
    print(f"[data_path]    {data_path}")
    print(f"[save_model]   {save_path}")
    print(f"[shap_csv]     {shap_csv}")
    print(f"[best_params]  {best_json}")
    print("--------------------------------------------------")

    if not data_path.exists():
        existing = sorted(p.name for p in DATASETS_DIR.glob("*.csv"))
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            f"Available in {DATASETS_DIR}:\n - " + "\n - ".join(existing)
        )

    pipe = AutoMLPipeline(
        data_path=str(data_path),
        target_column=target,
        k_top=K_TOP,
        variance_threshold=VARIANCE_THRESHOLD,
        n_iter_search=N_ITER_SEARCH,
        cv_folds=CV_FOLDS,
        ensemble_max=ENSEMBLE_MAX,           # ignored in single-model mode
        search_sample_cap=SEARCH_SAMPLE_CAP,
        big_rows_threshold=BIG_ROWS_THRESHOLD,
    )
    pipe.best_params_path = str(best_json)

    # IMPORTANT: run() is expected to RETURN a list of dicts like:
    # [{"model": tm.key, "cv_score": tm.cv_score, **tm.test_metrics}]
    print("[run] starting pipeline…")
    results = pipe.run(
        single_model_key=model_key,
        save_model_path=str(save_path),  # rich joblib bundle
        prefer_ensemble=False,
        shap_csv_path=str(shap_csv),
    )
    print("[run] finished.")

    if not results or not isinstance(results, list):
        raise RuntimeError("Pipeline did not return metrics list. Ensure AutoMLPipeline.run() returns results.")

    # We trained only ONE model per target → take the first record
    rec = results[0]
    # rec contains: model, cv_score, and test_metrics (for regression: r2, mse, rmse, mae)
    model_name = rec.get("model", model_key)
    cv_score   = float(rec.get("cv_score", 0.0))
    # unify keys for regression
    r2   = float(rec.get("r2", rec.get("R2", 0.0)))
    mse  = float(rec.get("mse", 0.0))
    rmse = float(rec.get("rmse", (mse ** 0.5 if mse else 0.0)))
    mae  = float(rec.get("mae", 0.0))

    metrics_accum[target] = {
        "model": model_name,
        "cv_score": cv_score,
        "rmse": rmse,
        "mse": mse,
        "r2": r2,
        "mae": mae,
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model_path": str(save_path),
        "shap_summary": str(shap_csv),
    }

    print(f"[metrics] {target} → {metrics_accum[target]}")

def main():
    print("========== AutoML Trainer (ALL TARGETS) ==========")
    print(f"[root]         {ROOT}")
    print(f"[datasets_dir] {DATASETS_DIR} (exists={DATASETS_DIR.exists()})")
    print(f"[models_dir]   {MODELS_DIR}   (exists={MODELS_DIR.exists()})")
    print(f"[shap_dir]     {SHAP_DIR}     (exists={SHAP_DIR.exists()})")
    print(f"[knobs] n_iter={N_ITER_SEARCH}  cv_folds={CV_FOLDS}  k_top={K_TOP}  "
          f"var_thr={VARIANCE_THRESHOLD}  search_cap={SEARCH_SAMPLE_CAP}  big_rows={BIG_ROWS_THRESHOLD}")
    print("==================================================")

    metrics_blob = load_existing_metrics()
    failures = []

    for tgt in TARGETS:
        try:
            train_target(tgt, metrics_blob)
            # write after each target so partial progress is saved
            write_metrics_blob(metrics_blob)
        except Exception as e:
            failures.append((tgt, str(e)))
            print(f"[ERROR] Training failed for {tgt}: {e}")

    print("\n==================== SUMMARY =====================")
    if metrics_blob:
        for k, v in metrics_blob.items():
            print(f"✔ {k}: {v['model']} | r2={v['r2']:.6f} rmse={v['rmse']:.6f}")
    else:
        print("No metrics written.")
    if failures:
        print("\nFailures:")
        for tgt, err in failures:
            print(f"✘ {tgt}: {err}")
    print("==================================================")

if __name__ == "__main__":
    main()
