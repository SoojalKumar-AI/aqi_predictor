import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn import __version__ as SKL_VERSION
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Core estimators
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from joblib import dump, load
from sklearn.calibration import CalibratedClassifierCV
import json as _json  # for metrics export
import numbers
from datetime import datetime

warnings.filterwarnings("ignore")

RNG = 42
np.random.seed(RNG)


def _to_py(o):
    """Make objects JSON-serializable (numpy -> python scalars/lists)."""
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (set,)):
        return list(o)
    return o

# ---------------------- Version helpers ----------------------
def _skl_version_ge(major: int, minor: int = 0) -> bool:
    try:
        parts = SKL_VERSION.split(".")
        M = int(parts[0]); m = int(parts[1])
        return (M > major) or (M == major and m >= minor)
    except Exception:
        return True


# ---------------------- Config loading -----------------------
with open("classification_models.json", "r") as f:
    CLASSIFICATION_MODELS = json.load(f)["classification"]
with open("regression_models.json", "r") as f:
    REGRESSION_MODELS = json.load(f)["regression"]
with open("ensemble_strategies.json", "r") as f:
    ENSEMBLE_STRATEGIES = json.load(f)["ensemble_strategies"]


# ---------------------- Task inference -----------------------
def is_classification_target(y: pd.Series, max_unique=20, max_ratio=0.2) -> bool:
    if y.dtype.name in ("object", "category", "bool"):
        return True
    if pd.api.types.is_integer_dtype(y) or pd.api.types.is_bool_dtype(y):
        nunique = y.nunique(dropna=True)
        if nunique <= max_unique and (nunique / max(1, len(y))) <= max_ratio:
            return True
    return False


# ---------------------- Preprocess builder -------------------
def build_preprocess_pipeline(
    task_type: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
    variance_threshold: float = 0.0,
    k_top: int = 50,
) -> Pipeline:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    # version-safe OHE
    if _skl_version_ge(1, 2):
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    else:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe),
        ]
    )

    ct = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    selector = SelectKBest(
        score_func=f_classif if task_type == "classification" else f_regression,
        k=k_top,
    )

    preprocess = Pipeline(
        steps=[
            ("ct", ct),
            ("var", VarianceThreshold(threshold=variance_threshold)),
            ("sel", selector),
        ]
    )
    return preprocess


# ---------------------- Model catalog & search spaces ----------
def make_estimator(model_key: str, task_type: str):
    # Classification
    if task_type == "classification":
        if model_key.startswith("logistic_regression"):
            return LogisticRegression(max_iter=1000, class_weight=None, random_state=RNG)
        if model_key.startswith("naive_bayes"):
            from sklearn.naive_bayes import GaussianNB
            return GaussianNB()
        if model_key.startswith("decision_tree"):
            return DecisionTreeClassifier(random_state=RNG)
        if model_key.startswith("random_forest"):
            return RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1)
        if model_key.startswith("extra_trees"):
            return ExtraTreesClassifier(n_estimators=300, random_state=RNG, n_jobs=-1)
        if model_key.startswith("gradient_boosting"):
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(random_state=RNG)
        if model_key.startswith("svm"):
            return SVC(probability=True, random_state=RNG)  # disabled on big data below
        if model_key.startswith("mlp"):
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(max_iter=1000, random_state=RNG)

        if model_key.startswith("xgboost"):
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    random_state=RNG, eval_metric="logloss", n_estimators=300,
                    tree_method="hist", n_jobs=-1, verbosity=0
                )
            except Exception as e:
                print(f"[WARN] ({model_key}) xgboost not available; falling back to RandomForestClassifier. [{e}]")
                return RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1)
        if model_key.startswith("lightgbm"):
            try:
                from lightgbm import LGBMClassifier
                return LGBMClassifier(
                    random_state=RNG, n_estimators=300, n_jobs=-1,
                    verbosity=-1, force_col_wise=True
                )
            except Exception as e:
                print(f"[WARN] ({model_key}) lightgbm not available; falling back to RandomForestClassifier. [{e}]")
                return RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1)
        if model_key.startswith("catboost"):
            try:
                from catboost import CatBoostClassifier
                return CatBoostClassifier(random_state=RNG, verbose=False)
            except Exception as e:
                print(f"[WARN] ({model_key}) catboost not available; falling back to RandomForestClassifier. [{e}]")
                return RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1)

    # Regression
    else:
        if model_key.startswith("linear_regression"):
            return LinearRegression()
        if model_key.startswith("ridge"):
            return Ridge()
        if model_key.startswith("lasso"):
            return Lasso()
        if model_key.startswith("elastic_net"):
            return ElasticNet()
        if model_key.startswith("decision_tree"):
            return DecisionTreeRegressor(random_state=RNG)
        if model_key.startswith("random_forest"):
            return RandomForestRegressor(n_estimators=400, random_state=RNG, n_jobs=-1)
        if model_key.startswith("extra_trees"):
            return ExtraTreesRegressor(n_estimators=400, random_state=RNG, n_jobs=-1)
        if model_key.startswith("gradient_boosting"):
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(random_state=RNG)
        if model_key.startswith("svm"):
            return SVR()  # disabled on big data below
        if model_key.startswith("mlp"):
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor(max_iter=1000, random_state=RNG)

        if model_key.startswith("xgboost"):
            try:
                from xgboost import XGBRegressor
                return XGBRegressor(
                    random_state=RNG, n_estimators=400, n_jobs=-1,
                    tree_method="hist", verbosity=0
                )
            except Exception as e:
                print(f"[WARN] ({model_key}) xgboost not available; falling back to RandomForestRegressor. [{e}]")
                return RandomForestRegressor(n_estimators=400, random_state=RNG, n_jobs=-1)
        if model_key.startswith("lightgbm"):
            try:
                from lightgbm import LGBMRegressor
                return LGBMRegressor(
                    random_state=RNG, n_estimators=400, n_jobs=-1,
                    verbosity=-1, force_col_wise=True
                )
            except Exception as e:
                print(f"[WARN] ({model_key}) lightgbm not available; falling back to RandomForestRegressor. [{e}]")
                return RandomForestRegressor(n_estimators=400, random_state=RNG, n_jobs=-1)
        if model_key.startswith("catboost"):
            try:
                from catboost import CatBoostRegressor
                return CatBoostRegressor(random_state=RNG, verbose=False)
            except Exception as e:
                print(f"[WARN] ({model_key}) catboost not available; falling back to RandomForestRegressor. [{e}]")
                return RandomForestRegressor(n_estimators=400, random_state=RNG, n_jobs=-1)

    raise ValueError(f"Model {model_key} not implemented")


def param_distributions(model_key: str, task_type: str) -> Dict[str, List]:
    # estimator params are prefixed with 'est__' inside the pipeline
    if task_type == "classification":
        if model_key.startswith("logistic_regression"):
            return {"est__C": np.logspace(-3, 3, 12),
                    "est__penalty": ["l2", "none"],
                    "est__class_weight": [None, "balanced"]}
        if model_key.startswith("naive_bayes"):
            return {}
        if model_key.startswith("decision_tree"):
            return {"est__max_depth": [None, 3, 5, 8, 12, 16, 24],
                    "est__min_samples_split": [2, 5, 10, 20],
                    "est__min_samples_leaf": [1, 2, 4, 8]}
        if model_key.startswith("random_forest") or model_key.startswith("extra_trees"):
            return {"est__n_estimators": [200, 300, 500],
                    "est__max_depth": [None, 6, 10, 16, 24],
                    "est__min_samples_leaf": [1, 2, 4, 8],
                    "est__max_features": ["sqrt", "log2", 0.5, None]}
        if model_key.startswith("gradient_boosting"):
            return {"est__n_estimators": [150, 300, 450],
                    "est__learning_rate": [0.03, 0.05, 0.1, 0.2],
                    "est__max_depth": [2, 3, 4, 5],
                    "est__subsample": [0.7, 0.85, 1.0]}
        if model_key.startswith("svm"):
            return {"est__C": np.logspace(-2, 2, 8),
                    "est__gamma": np.logspace(-3, -1, 4),
                    "est__kernel": ["rbf", "linear"]}
        if model_key.startswith("mlp"):
            return {"est__hidden_layer_sizes": [(64,), (128,), (64, 32)],
                    "est__alpha": np.logspace(-5, -2, 4)}
    else:
        if model_key.startswith("linear_regression"):
            return {}
        if model_key.startswith("ridge"):
            return {"est__alpha": np.logspace(-4, 2, 12)}
        if model_key.startswith("lasso"):
            return {"est__alpha": np.logspace(-4, 1, 12)}
        if model_key.startswith("elastic_net"):
            return {"est__alpha": np.logspace(-4, 1, 8),
                    "est__l1_ratio": np.linspace(0.1, 0.9, 9)}
        if model_key.startswith("decision_tree"):
            return {"est__max_depth": [None, 4, 6, 10, 14, 20],
                    "est__min_samples_split": [2, 5, 10, 20],
                    "est__min_samples_leaf": [1, 2, 4, 8]}
        if model_key.startswith("random_forest") or model_key.startswith("extra_trees"):
            return {"est__n_estimators": [300, 500, 700],
                    "est__max_depth": [None, 6, 10, 14],
                    "est__min_samples_leaf": [1, 2, 4, 8],
                    "est__max_features": ["sqrt", "log2", 0.5, None]}
        if model_key.startswith("gradient_boosting"):
            return {"est__n_estimators": [200, 400, 600],
                    "est__learning_rate": [0.03, 0.05, 0.1, 0.2],
                    "est__max_depth": [2, 3, 4, 5],
                    "est__subsample": [0.7, 0.85, 1.0]}
        if model_key.startswith("svm"):
            return {"est__C": np.logspace(-2, 2, 8),
                    "est__gamma": np.logspace(-3, -1, 4),
                    "est__kernel": ["rbf", "linear"]}
        if model_key.startswith("mlp"):
            return {"est__hidden_layer_sizes": [(128,), (64, 32)],
                    "est__alpha": np.logspace(-5, -2, 4)}

    return {}


# ---------------------- Feature names through pipeline ----------
def get_final_feature_names(preprocess: Pipeline) -> List[str]:
    ct: ColumnTransformer = preprocess.named_steps["ct"]
    names: List[str] = []
    try:
        names = ct.get_feature_names_out().tolist()
    except Exception:
        for name, trans, cols in ct.transformers_:
            if name == "num":
                names += [f"num__{c}" for c in cols]
            elif name == "cat":
                try:
                    ohe = ct.named_transformers_["cat"].named_steps["ohe"]
                    for col, cats in zip(cols, ohe.categories_):
                        names += [f"{col}={cat}" for cat in cats]
                except Exception:
                    pass

    var_mask = preprocess.named_steps["var"].get_support(indices=True)
    if names:
        names = [names[i] for i in var_mask]
    else:
        names = [f"f_{i}" for i in var_mask]

    kb_mask = preprocess.named_steps["sel"].get_support(indices=True)
    names = [names[i] for i in kb_mask]
    return names


# ---------------------- Training & Evaluation -------------------------
@dataclass
class TrainedModel:
    key: str
    pipeline: Pipeline
    cv_score: float
    test_metrics: Dict[str, float]
    test_pred: np.ndarray
    test_proba: Optional[np.ndarray]
    feat_names: List[str]


class AutoMLPipeline:
    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        data_path: Optional[str] = None,
        target_column: Optional[str] = None,
        task_override: Optional[str] = None,
        metric_override: Optional[str] = None,
        k_top: int = 50,
        variance_threshold: float = 0.0,
        n_iter_search: int = 20,
        cv_folds: int = 5,
        ensemble_max: int = 3,
        search_sample_cap: int = 5000,
        big_rows_threshold: int = 15000,
        # NEW: train exactly one model if provided
        single_model_key: Optional[str] = None,
    ):
        self.single_model_key = single_model_key
        self.data_path = data_path
        self.data = data
        self.target_column = target_column
        self.task_type = task_override
        self.metric_override = metric_override
        self.k_top = k_top
        self.variance_threshold = variance_threshold
        self.n_iter_search = n_iter_search
        self.cv_folds = cv_folds
        self.ensemble_max = ensemble_max
        self.search_sample_cap = search_sample_cap
        self.big_rows_threshold = big_rows_threshold

        self.X_train_raw = self.X_test_raw = None
        self.y_train = self.y_test = None
        self.trained: List[TrainedModel] = []
        self.ensemble = None
        self.ensemble_result = None
        self.tuning_log = {}                 # model_key -> dict with best params & features
        self.best_params_path = "best_params.json"

    # ---------- Data I/O ----------
    def load_data(self):
        if self.data is not None:
            return
        if self.data_path is None:
            raise ValueError("No data provided")
        if self.data_path.endswith(".csv"):
            self.data = pd.read_csv(self.data_path)
        elif self.data_path.endswith(".xlsx"):
            self.data = pd.read_excel(self.data_path)
        else:
            raise ValueError("Unsupported format")

    def determine_task_type(self):
        if self.task_type in ("classification", "regression"):
            return
        y = self.data[self.target_column]
        self.task_type = "classification" if is_classification_target(y) else "regression"

    def split(self):
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        strat = y if (self.task_type == "classification" and pd.Series(y).nunique() > 1) else None
        self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=RNG, stratify=strat
        )

    # ---------- Helpers ----------
    def _is_big(self) -> bool:
        return len(self.X_train_raw) + len(self.X_test_raw) >= self.big_rows_threshold

    def _cv(self):
        # Reduce folds on big data to save time
        folds = 3 if self._is_big() else self.cv_folds
        return (
            StratifiedKFold(n_splits=folds, shuffle=True, random_state=RNG)
            if self.task_type == "classification"
            else KFold(n_splits=folds, shuffle=True, random_state=RNG)
        )

    def _scorer(self):
        if self.metric_override:
            return self.metric_override
        return "balanced_accuracy" if self.task_type == "classification" else "r2"

    def _auto_k_top(self, default_k: int = None) -> int:
        default_k = self.k_top if default_k is None else default_k
        if self.X_train_raw is None or self.y_train is None:
            return max(1, default_k)
        X = self.X_train_raw.copy(); y = self.y_train
        if X.shape[1] == 0:
            return 1
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        X_enc = X.copy()
        for c in cat_cols:
            X_enc[c] = X_enc[c].astype("category").cat.codes
        discrete_mask = [col in cat_cols for col in X_enc.columns]
        try:
            if self.task_type == "classification":
                mi = mutual_info_classif(X_enc, y, discrete_features=discrete_mask, random_state=RNG)
            else:
                mi = mutual_info_regression(X_enc, y, discrete_features=discrete_mask, random_state=RNG)
            mi = np.asarray(mi)
            k_pos = int((mi > 0).sum())
            k = max(1, min(default_k, k_pos if k_pos > 0 else 1, X.shape[1]))
            return k
        except Exception:
            k = int(max(1, round(0.6 * X.shape[1])))
            return min(default_k, k)

    def _record_tuning(self,
                    model_key: str,
                    best_params: dict,
                    feat_names: list,
                    cv,
                    search_obj,
                    searched_on_n: int,
                    used_subsample: bool,
                    pre_pipeline: Pipeline):
        """Store best params + feature names + meta in self.tuning_log (JSON-safe)."""
        # Pull k actually used in SelectKBest (after our auto heuristic)
        try:
            k_used = pre_pipeline.named_steps["sel"].k
        except Exception:
            k_used = None

        # Some searches may not have params (empty space)
        best_score = None
        try:
            best_score = float(search_obj.best_score_) if hasattr(search_obj, "best_score_") else None
        except Exception:
            best_score = None

        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "task_type": self.task_type,
            "scorer": self._scorer(),
            "cv_folds": int(cv.get_n_splits()) if hasattr(cv, "get_n_splits") else None,
            "search_n_iter": int(search_obj.n_iter) if hasattr(search_obj, "n_iter") else 1,
            "searched_on_rows": int(searched_on_n),
            "used_subsample_for_search": bool(used_subsample),
            "preprocess": {
                "variance_threshold": float(self.variance_threshold),
                "k_top_requested": int(self.k_top),
                "k_top_effective": int(k_used) if k_used is not None else None
            },
            "best_cv_score": best_score,
            "best_params": {k: _to_py(v) for k, v in (best_params or {}).items()},
            "selected_features_after_selection": list(feat_names),
        }

        # If the same model_key appears again, keep a list of runs
        if model_key in self.tuning_log:
            # normalize to list
            if isinstance(self.tuning_log[model_key], dict):
                self.tuning_log[model_key] = [self.tuning_log[model_key]]
            self.tuning_log[model_key].append(entry)
        else:
            self.tuning_log[model_key] = entry

    def _flush_tuning_log(self, path: str = None):
        path = path or self.best_params_path
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.tuning_log, f, indent=2, default=_to_py)
            print(f"[OK] Wrote tuned params & features to {path}")
        except Exception as e:
            print(f"[WARN] Could not write {path}: {e}")

    def _task_catalog(self) -> List[dict]:
        """Return raw JSON catalog for current task."""
        return CLASSIFICATION_MODELS if self.task_type == "classification" else REGRESSION_MODELS

    def _all_model_keys_for_task(self) -> List[str]:
        """All model keys for current task from JSON, in file order."""
        return [m["model_key"] for m in self._task_catalog()]

    def _validate_model_key(self, key: str) -> bool:
        """Check key exists in current task catalog."""
        return key in self._all_model_keys_for_task()

    def list_available_model_keys(self) -> List[str]:
        """Public: list all keys for the inferred/current task (useful for UIs/CLIs)."""
        return self._all_model_keys_for_task()

    def run_single(self, model_key: str):
        """Public: run pipeline training ONLY for this model key (keeps all other logic)."""
        self.single_model_key = model_key
        return self.run()
    
    def _save_bar_plot(self, df: pd.DataFrame, value_col: str, label_col: str, out_path: str, title: str):
        try:
            import matplotlib.pyplot as plt
            top = df.copy()
            plt.figure(figsize=(10, 6))
            plt.barh(top[label_col], top[value_col])
            plt.gca().invert_yaxis()
            plt.title(title)
            plt.xlabel(value_col)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"[OK] Saved plot: {out_path}")
        except Exception as e:
            print(f"[WARN] Could not save plot '{out_path}': {e}")

    # ---------- Candidate selection ----------
    def _candidates_from_json(self) -> List[str]:
        cfg = CLASSIFICATION_MODELS if self.task_type == "classification" else REGRESSION_MODELS
        n_rows = len(self.data)
        n_feat = self.data.shape[1] - 1
        keys = []
        for m in cfg:
            cond = m.get("suitable_dataset_conditions", {})
            if cond:
                min_r = cond.get("min_rows", 0)
                max_r = cond.get("max_rows", 10**12)
                max_f = cond.get("max_features", 10**6)
                if not (min_r <= n_rows <= max_r and n_feat <= max_f):
                    continue
            k = m["model_key"]
            # Drop very slow models on big data
            if self._is_big() and (k.startswith("svm") or k.startswith("mlp")):
                continue
            keys.append(k)
        return list(dict.fromkeys(keys))

    # ---------- Train + Tune a single candidate ----------
    def _train_one(self, model_key: str) -> Optional[TrainedModel]:
        num_cols = self.X_train_raw.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.X_train_raw.select_dtypes(exclude=[np.number]).columns.tolist()

        pre = build_preprocess_pipeline(
            self.task_type, num_cols, cat_cols,
            variance_threshold=self.variance_threshold,
            k_top=self._auto_k_top(default_k=min(self.k_top, max(1, len(num_cols) + len(cat_cols))))
        )

        est = make_estimator(model_key, self.task_type)
        pipe = Pipeline([("pre", pre), ("est", est)])

        params = param_distributions(model_key, self.task_type)

        # --------- subsample for search on big data ----------
        Xs, ys = self.X_train_raw, self.y_train
        used_subsample = False
        if self._is_big():
            rs = np.random.RandomState(RNG)
            n = min(len(Xs), self.search_sample_cap)
            idx = rs.choice(len(Xs), size=n, replace=False)
            Xs = Xs.iloc[idx]
            ys = ys.iloc[idx] if isinstance(ys, pd.Series) else ys[idx]
            used_subsample = True

        n_iter = min(self.n_iter_search, 10 if self._is_big() else self.n_iter_search)
        cv = self._cv()

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=params,
            n_iter=n_iter if params else 1,
            cv=cv,
            scoring=self._scorer(),
            n_jobs=-1,
            random_state=RNG,
            verbose=0,
            refit=False,  # we'll refit best params on full train below
        )
        try:
            search.fit(Xs, ys)
            best_params = search.best_params_ if params else {}
            # >>> NEW: record & flush right after tuning <<<
            self._record_tuning(
                model_key=model_key,
                best_params=best_params,
                feat_names=[],                 # fill after we compute them
                cv=cv,
                search_obj=search,
                searched_on_n=len(Xs),
                used_subsample=used_subsample,
                pre_pipeline=pre
            )
            self._flush_tuning_log()  # write incrementally so we don't lose info
            # <<< NEW
        except Exception as e:
            print(f"[ERROR] {model_key} failed during tuning: {e}")
            return None

        # Rebuild a fresh pipeline with best params and fit on FULL training data
        best_pipe: Pipeline = Pipeline([("pre", pre), ("est", est)])
        if best_params:
            best_pipe.set_params(**best_params)
        best_pipe.fit(self.X_train_raw, self.y_train)

        # Test metrics
        y_hat = best_pipe.predict(self.X_test_raw)
        proba = None
        if self.task_type == "classification" and hasattr(best_pipe.named_steps["est"], "predict_proba"):
            try:
                proba = best_pipe.predict_proba(self.X_test_raw)
            except Exception:
                proba = None

        metrics = self._compute_metrics(self.y_test, y_hat, proba)
        feat_names = get_final_feature_names(best_pipe.named_steps["pre"])

        # >>> NEW: update the just-written entry with the final feature names
        # If the key maps to a list (repeated trainings), update the last one.
        if model_key in self.tuning_log:
            if isinstance(self.tuning_log[model_key], list):
                self.tuning_log[model_key][-1]["selected_features_after_selection"] = list(feat_names)
            else:
                self.tuning_log[model_key]["selected_features_after_selection"] = list(feat_names)
            self._flush_tuning_log()
        # <<< NEW

        # Use CV score from search; if no params, make a simple score
        cv_score = None
        try:
            cv_score = float(search.best_score_) if params else None
        except Exception:
            cv_score = None
        if cv_score is None:
            cv_score = float(metrics["balanced_accuracy"] if self.task_type == "classification" else metrics["r2"])

        return TrainedModel(
            key=model_key,
            pipeline=best_pipe,
            cv_score=cv_score,
            test_metrics=metrics,
            test_pred=y_hat,
            test_proba=proba,
            feat_names=feat_names,
        )

    def _maybe_calibrate(self, tm: "TrainedModel") -> "TrainedModel":
        if self.task_type != "classification":
            return tm
        est = tm.pipeline.named_steps.get("est", None)
        if est is None or hasattr(est, "predict_proba"):
            return tm
        calibrated_est = CalibratedClassifierCV(est, method="sigmoid", cv=3)
        new_pipe = Pipeline([("pre", tm.pipeline.named_steps["pre"]), ("est", calibrated_est)])
        try:
            new_pipe.fit(self.X_train_raw, self.y_train)
            y_hat = new_pipe.predict(self.X_test_raw)
            proba = None
            try: proba = new_pipe.predict_proba(self.X_test_raw)
            except Exception: proba = None
            metrics = self._compute_metrics(self.y_test, y_hat, proba)
            return TrainedModel(
                key=tm.key + "_calibrated",
                pipeline=new_pipe,
                cv_score=tm.cv_score,
                test_metrics=metrics,
                test_pred=y_hat,
                test_proba=proba,
                feat_names=tm.feat_names,
            )
        except Exception:
            return tm

    def _compute_metrics(self, y_true, y_pred, proba=None) -> Dict[str, float]:
        if self.task_type == "classification":
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "precision_w": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
                "recall_w": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
                "f1_w": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            }
        else:
            mse = mean_squared_error(y_true, y_pred)
            return {
                "r2": float(r2_score(y_true, y_pred)),
                "mse": float(mse),
                "rmse": float(np.sqrt(mse)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
            }

    # ---------- Ensemble ----------
    def _classification_disagreement(self, preds_list: List[np.ndarray]) -> float:
        stack = np.vstack(preds_list)
        return float(np.mean(np.ptp(stack, axis=0) > 0))

    def _regression_correlation(self, preds_list: List[np.ndarray]) -> float:
        if len(preds_list) < 2:
            return 1.0
        arr = np.vstack(preds_list)
        corr = np.corrcoef(arr)
        iu = np.triu_indices_from(corr, k=1)
        return float(np.mean(np.abs(corr[iu])))

    def _maybe_build_ensemble(self, top_models: List[TrainedModel]):
        if len(top_models) < 2:
            return None
        top_models = sorted(top_models, key=lambda m: m.cv_score, reverse=True)[: self.ensemble_max]
        test_preds = [m.pipeline.predict(self.X_test_raw) for m in top_models]
        if self.task_type == "classification":
            disagree = self._classification_disagreement(test_preds)
            if disagree < 0.05:
                print("[INFO] Base classifiers very similar; skipping ensemble.")
                return None
            all_proba = all(hasattr(m.pipeline.named_steps["est"], "predict_proba") for m in top_models)
            voting = "soft" if all_proba else "hard"
            weights = [max(0.0, m.cv_score) for m in top_models]
            ens = VotingClassifier(
                estimators=[(m.key, m.pipeline) for m in top_models],
                voting=voting,
                weights=weights if voting == "soft" else None,
            )
        else:
            corr = self._regression_correlation(test_preds)
            if corr > 0.98:
                print("[INFO] Base regressors highly correlated; skipping ensemble.")
                return None
            weights = [max(1e-6, m.cv_score) for m in top_models]
            ens = VotingRegressor(
                estimators=[(m.key, m.pipeline) for m in top_models],
                weights=weights,
            )

        ens.fit(self.X_train_raw, self.y_train)
        y_hat = ens.predict(self.X_test_raw)
        proba = None
        if self.task_type == "classification" and hasattr(ens, "predict_proba"):
            try: proba = ens.predict_proba(self.X_test_raw)
            except Exception: proba = None
        result = self._compute_metrics(self.y_test, y_hat, proba)
        return ens, result

    # ---------- Explainability ----------

    # ---------- Persistence ----------
    def save_model(self, path: str, prefer_ensemble: bool = True):
        if prefer_ensemble and self.ensemble is not None:
            bundle = {"kind": "ensemble", "pipeline": self.ensemble,
                      "task_type": self.task_type, "target_column": self.target_column}
        else:
            if not self.trained:
                raise RuntimeError("No trained models to save.")
            best = sorted(self.trained, key=lambda m: m.cv_score, reverse=True)[0]
            bundle = {"kind": "single", "pipeline": best.pipeline,
                      "task_type": self.task_type, "target_column": self.target_column,
                      "feature_names": best.feat_names, "model_key": best.key,
                      "cv_score": best.cv_score, "test_metrics": best.test_metrics}
        dump(bundle, path)
        print(f"[OK] Saved to {path}")

    @staticmethod
    def load_model(path: str):
        bundle = load(path)
        class _Deployed:
            def __init__(self, bundle):
                self.bundle = bundle
                self.pipeline = bundle["pipeline"]
                self.task_type = bundle["task_type"]
                self.target_column = bundle.get("target_column")
            def predict(self, df: pd.DataFrame, return_proba: bool = False):
                X = df.copy()
                if self.target_column and self.target_column in X.columns:
                    X = X.drop(columns=[self.target_column])
                y = self.pipeline.predict(X)
                if return_proba and self.task_type == "classification" and hasattr(self.pipeline, "predict_proba"):
                    try: p = self.pipeline.predict_proba(X); return y, p
                    except Exception: return y, None
                return y
        return _Deployed(bundle)

    def export_results(self, df: pd.DataFrame, base_path: str = "leaderboard"):
        csv_path = f"{base_path}.csv"; json_path = f"{base_path}.json"
        df.to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            _json.dump(_json.loads(df.to_json(orient="records")), f, indent=2)
        print(f"[OK] Exported metrics to {csv_path} and {json_path}")

    # ---------- Main flow ----------
    # --- replace your explain(...) with this ---
    def explain(self, model_pipeline: Pipeline, top_n: int = 12, save_csv_path: Optional[str] = None):
        """
        Print top feature importance via SHAP if possible, else permutation importance.
        If save_csv_path is provided, also write the table to CSV and return the DataFrame.
        """
        print("\nGenerating feature importance (SHAP if available)…")
        pre: Pipeline = model_pipeline.named_steps["pre"]
        est = model_pipeline.named_steps["est"]

        n_bg = min(200, len(self.X_train_raw))
        rs = np.random.RandomState(RNG)
        idx = rs.choice(len(self.X_train_raw), size=n_bg, replace=False)

        Xt_bg = pre.transform(self.X_train_raw)
        Xt_bg = Xt_bg[idx] if hasattr(Xt_bg, "__getitem__") else Xt_bg
        Xt_eval = pre.transform(self.X_test_raw)

        to_dense = lambda A: A.toarray() if hasattr(A, "toarray") else np.asarray(A)
        Xt_bg_d = to_dense(Xt_bg)
        Xt_eval_d = to_dense(Xt_eval)

        feat_names = get_final_feature_names(pre)

        df = None
        try:
            import shap
            explainer = shap.Explainer(est, Xt_bg_d)
            sv = explainer(Xt_eval_d)
            mean_abs = np.mean(np.abs(sv.values), axis=0)
            order = np.argsort(-mean_abs)[:top_n]
            df = pd.DataFrame(
                {"feature": [feat_names[i] for i in order],
                "mean_abs_shap": mean_abs[order]}
            )
            print(df.to_string(index=False))
        except Exception as e:
            print(f"[WARN] SHAP failed ({e}). Falling back to permutation importance.")
            try:
                from sklearn.inspection import permutation_importance
                r = permutation_importance(est, Xt_eval_d, self.y_test, n_repeats=5, random_state=RNG, n_jobs=-1)
                importances = r.importances_mean
                order = np.argsort(-importances)[:top_n]
                df = pd.DataFrame(
                    {"feature": [feat_names[i] for i in order],
                    "mean_importance": importances[order]}
                )
                print(df.to_string(index=False))
            except Exception as e2:
                print(f"[ERROR] permutation importance failed: {e2}")

        if save_csv_path and df is not None:
            try:
                df.to_csv(save_csv_path, index=False)
                print(f"[OK] Saved explanation to {save_csv_path}")
            except Exception as e:
                print(f"[WARN] Could not write SHAP/permutation CSV: {e}")

        return df


    # --- replace your run(...) with this ---
    def run(
            self,
            single_model_key: Optional[str] = None,
            save_model_path: Optional[str] = None,
            prefer_ensemble: bool = True,
            shap_csv_path: Optional[str] = None,
            simple_joblib_path: Optional[str] = None,
            feature_names: Optional[List[str]] = None
        ):
        """
        Execute the full flow.

        Args:
            single_model_key: if provided (here or via constructor), train only that model and skip ensembling.
            save_model_path: optional — save the usual rich bundle (includes metrics, etc.).
            prefer_ensemble: used only when save_model_path is provided and multiple models were trained.
            shap_csv_path: optional — path to save SHAP/permutation summary CSV.
            simple_joblib_path: optional — if set, save a minimal joblib with keys
                {'pipeline', 'target_column', 'feature_names'}.
            feature_names: optional — explicit feature-name list to store in the minimal joblib;
                if None, uses the model’s selected features.
        """
        # ----- setup -----
        self.load_data()
        if self.target_column is None:
            raise ValueError("target_column is required")
        self.determine_task_type()
        self.split()

        # allow run() arg to override constructor
        if single_model_key is not None:
            self.single_model_key = single_model_key

        print(f"\nDataset loaded: {len(self.data)} samples, {self.data.shape[1]-1} raw features")
        print(f"Task type: {self.task_type}")
        print(f"Target column: {self.target_column}")

        # ----- candidate selection -----
        if self.single_model_key:
            if not self._validate_model_key(self.single_model_key):
                all_keys = ", ".join(self._all_model_keys_for_task())
                raise ValueError(
                    f"Unknown model_key '{self.single_model_key}' for task '{self.task_type}'. "
                    f"Available keys: {all_keys}"
                )
            candidates = [self.single_model_key]
            print(f"\n[Custom] Training ONLY this model: {self.single_model_key}")
        else:
            candidates = self._candidates_from_json()
            if not candidates:
                raise RuntimeError("No suitable models found from configuration.")
            print(f"\n[Auto] Training candidates: {', '.join(candidates)}")

        # ----- train & collect -----
        records = []
        for key in candidates:
            tm = self._train_one(key)
            if tm is None:
                continue
            tm = self._maybe_calibrate(tm)
            self.trained.append(tm)
            records.append({"model": tm.key, "cv_score": tm.cv_score, **tm.test_metrics})

        if not self.trained:
            raise RuntimeError("All candidate trainings failed.")

        # ----- leaderboard -----
        results_df = pd.DataFrame(records).sort_values("cv_score", ascending=False).reset_index(drop=True)
        print("\nModel Performance (sorted by CV score):")
        print(results_df.to_string(index=False))
        # self.export_results(results_df)

        # keep sorted
        self.trained = sorted(self.trained, key=lambda m: m.cv_score, reverse=True)

        # ----- explain best single -----
        best = self.trained[0]
        print(f"\nBest single model by CV: {best.key}")
        try:
            self.explain(best.pipeline, save_csv_path=shap_csv_path)
        except Exception as ex:
            print(f"[WARN] explain() failed for {best.key}: {ex}")

        # ----- minimal joblib save (your requested format) -----
        if simple_joblib_path:
            feat_list = feature_names if feature_names is not None else list(best.feat_names)
            save_obj = {
                "model": best.pipeline,              
                "features": feat_list,
                "target_column": self.target_column               
            }
            try:
                # we imported `dump` at top: from joblib import dump
                dump(save_obj, simple_joblib_path)
                print(f"[OK] Saved minimal joblib to {simple_joblib_path}")
            except Exception as e:
                print(f"[WARN] Could not write minimal joblib '{simple_joblib_path}': {e}")

        # ----- ensemble (only in multi-model mode) -----
        if not self.single_model_key and len(self.trained) >= 2:
            topN = self.trained[:max(2, min(self.ensemble_max, len(self.trained)))]
            ens = self._maybe_build_ensemble(topN)
            if ens:
                self.ensemble, self.ensemble_result = ens
                print("\nEnsemble Performance (test):")
                print(pd.DataFrame([self.ensemble_result]).to_string(index=False))
        elif self.single_model_key:
            print("[Info] Skipping ensemble because a single custom model was requested.")

        # ----- persist tuning log -----
        self._flush_tuning_log(self.best_params_path)

        # ----- optional rich bundle save (unchanged behavior) -----
        if save_model_path:
            try:
                # In single-model mode, prefer_ensemble is ignored
                self.save_model(save_model_path, prefer_ensemble=(prefer_ensemble and not self.single_model_key))
            except Exception as e:
                print(f"[WARN] Could not save model to {save_model_path}: {e}")

        print("\nDone.")

        return records
