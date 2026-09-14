"""
src/models/classical_ml.py
===========================
Classical ML baselines:
  Classification: SVC(rbf) and RandomForestClassifier
  Regression:     SVR and XGBRegressor
  Trivial floor:  majority class / train-fold mean angle (every model must beat these)

Results are saved to JSON — never only printed.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    warnings.warn("XGBoost not installed. XGBRegressor will be unavailable.")
    HAS_XGB = False

from src.config import CFG
from src.evaluation.metrics import compute_fixation_metrics


def build_svc_pipeline(use_grid_search: bool = False) -> Pipeline:
    """SVC with RBF kernel, StandardScaler preprocessing."""
    svc = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
              random_state=CFG.cv.random_seed)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", svc)])
    if use_grid_search:
        param_grid = {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", "auto"]}
        return GridSearchCV(pipe, param_grid, cv=3, scoring="f1_weighted", n_jobs=-1, refit=True)
    return pipe


def build_rf_classifier(use_grid_search: bool = False) -> RandomForestClassifier:
    """RandomForest classifier."""
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, class_weight="balanced",
        random_state=CFG.cv.random_seed, n_jobs=-1
    )
    if use_grid_search:
        param_grid = {"n_estimators": [100, 200], "max_depth": [None, 20]}
        return GridSearchCV(rf, param_grid, cv=3, scoring="f1_weighted", n_jobs=-1, refit=True)
    return rf


def build_svr_pipeline(use_grid_search: bool = False):
    """SVR (multioutput via MultiOutputRegressor)."""
    svr = SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.1)
    pipe = Pipeline([("scaler", StandardScaler()), ("reg", MultiOutputRegressor(svr, n_jobs=-1))])
    if use_grid_search:
        param_grid = {
            "reg__estimator__C": [0.1, 1.0, 10.0],
            "reg__estimator__epsilon": [0.01, 0.1],
        }
        return GridSearchCV(pipe, param_grid, cv=3, scoring="neg_mean_squared_error",
                            n_jobs=-1, refit=True)
    return pipe


def build_xgb_regressor(use_grid_search: bool = False):
    """XGBRegressor (multioutput via MultiOutputRegressor)."""
    if not HAS_XGB:
        raise ImportError("XGBoost not installed. pip install xgboost")
    xgb = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=CFG.cv.random_seed, n_jobs=-1, verbosity=0
    )
    reg = MultiOutputRegressor(xgb, n_jobs=1)
    if use_grid_search:
        param_grid = {
            "estimator__n_estimators": [100, 200],
            "estimator__max_depth": [4, 6],
        }
        return GridSearchCV(reg, param_grid, cv=3, scoring="neg_mean_squared_error",
                            n_jobs=-1, refit=True)
    return reg


def _subsample_train(X_tr: np.ndarray, y_tr: np.ndarray, model_label: str,
                     weights: Optional[np.ndarray] = None):
    """
    Subsample the TRAIN set for kernel-SVM models, whose fit cost is O(n²).
    The test set is never subsampled. Controlled by CFG.cv.svm_max_train_samples
    (0 disables). Prints a warning whenever it activates.

    Returns (X, y, weights); weights stays None when none were given.
    """
    cap = getattr(CFG.cv, "svm_max_train_samples", 0)
    if not cap or len(X_tr) <= cap:
        return X_tr, y_tr, weights
    print(
        f"  [INFO] {model_label}: subsampling train set {len(X_tr)} -> {cap} "
        f"(kernel-SVM O(n²) fit cost; test set unchanged). "
        "Tune via CFG.cv.svm_max_train_samples."
    )
    rng = np.random.RandomState(CFG.cv.random_seed)
    sub_idx = rng.choice(len(X_tr), cap, replace=False)
    return X_tr[sub_idx], y_tr[sub_idx], (None if weights is None else weights[sub_idx])


def run_classification_cv(
    X: np.ndarray,
    y: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    model_name: str = "svc",
    use_grid_search: bool = None,
) -> Dict:
    """
    Run cross-validated classification and return results dict.
    """
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    if use_grid_search is None:
        use_grid_search = getattr(CFG.cv, "use_grid_search", False)

    results = {"model": model_name, "folds": []}
    all_preds, all_true = [], []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if model_name == "svc":
            model = build_svc_pipeline(use_grid_search)
            X_fit, y_fit, _ = _subsample_train(X_tr, y_tr, "SVC")
            max_samples = CFG.cv.grid_search_max_samples
            if use_grid_search and max_samples and len(X_fit) > max_samples:
                sub_idx = np.random.RandomState(CFG.cv.random_seed).choice(
                    len(X_fit), max_samples, replace=False
                )
                X_fit, y_fit = X_fit[sub_idx], y_fit[sub_idx]
            model.fit(X_fit, y_fit)
        elif model_name == "rf":
            model = build_rf_classifier(use_grid_search)
            model.fit(X_tr, y_tr)
        elif model_name == "majority":
            model = DummyClassifier(strategy="most_frequent")
            model.fit(X_tr, y_tr)
        else:
            raise ValueError(f"Unknown classifier: {model_name}")

        y_pred = model.predict(X_te)

        fold_result = {
            "fold": fold_i,
            "f1_weighted": float(f1_score(y_te, y_pred, average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(y_te, y_pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        }
        results["folds"].append(fold_result)
        all_preds.extend(y_pred.tolist())
        all_true.extend(y_te.tolist())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    class_names = CFG.segmentation.classes
    results["pooled_f1_weighted"] = float(f1_score(all_true, all_preds, average="weighted", zero_division=0))
    results["pooled_f1_macro"] = float(f1_score(all_true, all_preds, average="macro", zero_division=0))
    results["pooled_confusion_matrix"] = confusion_matrix(all_true, all_preds).tolist()
    results["classification_report"] = classification_report(
        all_true, all_preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
    return results


def run_regression_cv(
    X: np.ndarray,
    y: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    model_name: str = "svr",
    use_grid_search: bool = None,
    metadata: Optional[List[dict]] = None,
) -> Dict:
    """
    Run cross-validated regression and return results dict.

    With per-window metadata the results also carry the fixation-window MAE
    (see compute_fixation_metrics). CFG.cv.regression_train_windows picks the
    training windows: "all", "fixation" (fixation windows only), or "weighted"
    (all windows, non-fixation ones weighted by CFG.cv.regression_nonfixation_weight).
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    if use_grid_search is None:
        use_grid_search = getattr(CFG.cv, "use_grid_search", False)
    train_windows = CFG.cv.regression_train_windows
    if train_windows not in ("all", "fixation", "weighted"):
        raise ValueError("CFG.cv.regression_train_windows must be 'all', 'fixation' or "
                         f"'weighted', got {train_windows!r}")
    if train_windows != "all" and metadata is None:
        raise ValueError(f"regression_train_windows={train_windows!r} needs per-window metadata")
    fixation = (np.array([bool(m.get("is_fixation", False)) for m in metadata])
                if metadata is not None else None)

    results = {"model": model_name, "train_windows": train_windows, "folds": []}
    if train_windows == "weighted":
        results["nonfixation_weight"] = CFG.cv.regression_nonfixation_weight
    all_preds, all_true, all_meta = [], [], []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        if train_windows == "fixation":
            train_idx = train_idx[fixation[train_idx]]
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        w_tr = (np.where(fixation[train_idx], 1.0, CFG.cv.regression_nonfixation_weight)
                if train_windows == "weighted" else None)

        valid_tr = ~np.isnan(y_tr).any(axis=1)
        if not valid_tr.all():
            X_tr, y_tr = X_tr[valid_tr], y_tr[valid_tr]
            w_tr = None if w_tr is None else w_tr[valid_tr]
        valid_te = ~np.isnan(y_te).any(axis=1)
        if not valid_te.all():
            X_te, y_te = X_te[valid_te], y_te[valid_te]
        if metadata is not None:
            all_meta.extend(metadata[i] for i in test_idx[valid_te])

        weight_param = "sample_weight"
        if model_name == "svr":
            model = build_svr_pipeline(use_grid_search)
            X_fit, y_fit, w_fit = _subsample_train(X_tr, y_tr, "SVR", w_tr)
            weight_param = "reg__sample_weight"
        elif model_name == "xgb":
            model = build_xgb_regressor(use_grid_search)
            X_fit, y_fit, w_fit = X_tr, y_tr, w_tr
        elif model_name == "mean":
            model = DummyRegressor(strategy="mean")
            X_fit, y_fit, w_fit = X_tr, y_tr, w_tr
        else:
            raise ValueError(f"Unknown regressor: {model_name}")

        model.fit(X_fit, y_fit, **({} if w_fit is None else {weight_param: w_fit}))
        y_pred = model.predict(X_te)

        rmse_h = float(np.sqrt(mean_squared_error(y_te[:, 0], y_pred[:, 0])))
        rmse_v = float(np.sqrt(mean_squared_error(y_te[:, 1], y_pred[:, 1])))
        mae_h = float(mean_absolute_error(y_te[:, 0], y_pred[:, 0]))
        mae_v = float(mean_absolute_error(y_te[:, 1], y_pred[:, 1]))
        r2_h = float(r2_score(y_te[:, 0], y_pred[:, 0]))
        r2_v = float(r2_score(y_te[:, 1], y_pred[:, 1]))

        fold_result = {
            "fold": fold_i,
            "rmse_h_deg": rmse_h, "rmse_v_deg": rmse_v,
            "mae_h_deg": mae_h, "mae_v_deg": mae_v,
            "r2_h": r2_h, "r2_v": r2_v,
        }
        results["folds"].append(fold_result)
        all_preds.append(y_pred)
        all_true.append(y_te)

    all_preds = np.vstack(all_preds)
    all_true = np.vstack(all_true)
    results["pooled_rmse_h_deg"] = float(np.sqrt(mean_squared_error(all_true[:, 0], all_preds[:, 0])))
    results["pooled_rmse_v_deg"] = float(np.sqrt(mean_squared_error(all_true[:, 1], all_preds[:, 1])))
    results["pooled_mae_h_deg"] = float(mean_absolute_error(all_true[:, 0], all_preds[:, 0]))
    results["pooled_mae_v_deg"] = float(mean_absolute_error(all_true[:, 1], all_preds[:, 1]))
    results["pooled_r2_h"] = float(r2_score(all_true[:, 0], all_preds[:, 0]))
    results["pooled_r2_v"] = float(r2_score(all_true[:, 1], all_preds[:, 1]))
    if metadata is not None:
        results.update(compute_fixation_metrics(all_true, all_preds, all_meta))
    return results


def save_results(results: Dict, name: str, results_dir: str = None) -> str:
    """Save results dict as JSON. Returns file path."""
    if results_dir is None:
        results_dir = CFG.paths.results
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {path}")
    return path
