"""
src/evaluation/metrics.py
==========================
Reusable metric computation from saved fold predictions.
Never re-runs the model — reads saved results JSONs only.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

from src.config import CFG


def load_results(name: str, results_dir: str = None) -> Dict:
    if results_dir is None:
        results_dir = CFG.paths.results
    path = os.path.join(results_dir, f"{name}.json")
    with open(path) as f:
        return json.load(f)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
) -> Dict:
    """Return F1, confusion matrix, and classification report dict."""
    if class_names is None:
        class_names = CFG.segmentation.classes
    return {
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        ),
    }


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict:
    """Return RMSE, MAE, R² for H and V channels separately and combined."""
    return {
        "rmse_h_deg": float(np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))),
        "rmse_v_deg": float(np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))),
        "mae_h_deg": float(mean_absolute_error(y_true[:, 0], y_pred[:, 0])),
        "mae_v_deg": float(mean_absolute_error(y_true[:, 1], y_pred[:, 1])),
        "r2_h": float(r2_score(y_true[:, 0], y_pred[:, 0])),
        "r2_v": float(r2_score(y_true[:, 1], y_pred[:, 1])),
        "rmse_combined_deg": float(
            np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel()))
        ),
    }


def compute_fixation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: List[dict],
) -> Dict:
    """
    Gaze error scored the way Barbara et al. (BSPC 2023) report it: mean absolute
    error over fixation windows only (metadata "is_fixation"), computed per subject
    and then averaged across subjects (± SD across subjects).

    Returns an empty dict when the metadata has no fixation flags or no fixations.
    """
    fixation = np.array([bool(m.get("is_fixation", False)) for m in metadata])
    if len(fixation) != len(y_true) or not fixation.any():
        return {}
    subjects = np.array([m["subject_id"] for m in metadata])
    abs_err = np.abs(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float))
    subject_ids = np.unique(subjects[fixation])
    per_subject = np.array([abs_err[fixation & (subjects == s)].mean(axis=0) for s in subject_ids])
    return {
        "fixation_mae_h_deg": float(per_subject[:, 0].mean()),
        "fixation_mae_v_deg": float(per_subject[:, 1].mean()),
        "fixation_mae_h_sd_deg": float(per_subject[:, 0].std()),
        "fixation_mae_v_sd_deg": float(per_subject[:, 1].std()),
        "n_fixation_windows": int(fixation.sum()),
        "fixation_mae_per_subject": {str(s): [float(h), float(v)] for s, (h, v) in zip(subject_ids, per_subject)},
    }


def print_summary_table(results_dir: str = None) -> None:
    """Print a formatted summary of all saved result files."""
    if results_dir is None:
        results_dir = CFG.paths.results
    files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    if not files:
        print("No result files found.")
        return

    print("\n" + "=" * 80)
    print(f"{'Model':<30} {'F1 Weighted':>12} {'RMSE H°':>10} {'RMSE V°':>10}")
    print("=" * 80)

    for fname in sorted(files):
        try:
            d = load_results(fname.replace(".json", ""), results_dir)
            f1 = d.get("pooled_f1_weighted", float("nan"))
            rmse_h = d.get("pooled_rmse_h_deg", float("nan"))
            rmse_v = d.get("pooled_rmse_v_deg", float("nan"))
            model = d.get("model", fname.replace(".json", ""))
            print(f"{model:<30} {f1:>12.4f} {rmse_h:>10.3f} {rmse_v:>10.3f}")
        except Exception as e:
            print(f"  Could not parse {fname}: {e}")
    print("=" * 80)
