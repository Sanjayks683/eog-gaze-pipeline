"""
scripts/run_full_evaluation.py
===============================
Full-scale evaluation script covering ALL parameters and architectures:
  1. Classical ML (SVC, RF, SVR, XGBoost) with hyperparameter GridSearch
  2. Deep Model 1: Conv1D (EOGMultiTaskNet)
  3. Deep Model 2: BiLSTM (EOGLSTMNet)
  4. Deep Model 3: Conv1D + BiLSTM (EOGConvLSTMNet - Best Proposed)
  5. Phase 9: Head-Pose Ablation
  6. Phase 10: Direction-Invariance Ablation
  7. Phase 8: Master Comparison Table Generation

Usage:
    python -u scripts/run_full_evaluation.py
"""

from __future__ import annotations

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import CFG
from src.data.datasets import load_processed
from src.training.cv_splits import load_folds
from src.models.classical_ml import run_classification_cv, run_regression_cv, save_results
from src.training.train import train_cv
from scripts.run_ablations import rebuild_angle_preds, run_head_pose_ablation, run_direction_ablation_full
from src.evaluation.compare_to_paper import run_comparison


def main():
    print("\n" + "=" * 80, flush=True)
    print("  EOG GAZE PIPELINE — FULL-SCALE MASTER EVALUATION", flush=True)
    print("=" * 80 + "\n", flush=True)

    t_start = time.time()

    print("[1/5] Loading preprocessed data and 5-fold CV splits...", flush=True)
    X_cls, y_cls, meta = load_processed("classification")
    X_reg, y_reg, _ = load_processed("regression")
    folds, _ = load_folds()
    print(f"  Dataset: {len(X_cls):,} windows across {len(folds)} folds.", flush=True)

    print("\n[2/5] Training Deep Multi-Task Models across all architectures on GPU...", flush=True)
    deep_models = ["conv1d", "lstm", "conv_bilstm"]

    CFG.model.loss_type = "uncertainty"
    CFG.model.max_epochs = 20
    CFG.model.early_stopping_patience = 6

    for m_type in deep_models:
        print(f"\n  --> Deep Architecture: {m_type.upper()}", flush=True)
        CFG.model.model_type = m_type
        r = train_cv(X_cls, y_cls, y_reg, folds, metadata=meta)
        save_results(r, f"deep_model_{m_type}")
        print(f"      {m_type.upper()} Pooled F1: {r['pooled_f1_weighted']:.4f} | RMSE H: {r['pooled_rmse_h_deg']:.3f}°, V: {r['pooled_rmse_v_deg']:.3f}°", flush=True)

    print("\n[3/5] Running Classical ML Baselines...", flush=True)
    from main import _extract_features_respecting_fs
    Xf_cls = _extract_features_respecting_fs(X_cls, meta)
    Xf_reg = _extract_features_respecting_fs(X_reg, meta)

    for clf in ["svc", "rf"]:
        print(f"\n  --> Classical Classifier: {clf.upper()}", flush=True)
        r = run_classification_cv(Xf_cls, y_cls, folds, model_name=clf, use_grid_search=False)
        save_results(r, f"classical_clf_{clf}")
        print(f"      Pooled F1 (weighted): {r['pooled_f1_weighted']:.4f}", flush=True)

    for reg in ["svr", "xgb"]:
        print(f"\n  --> Classical Regressor: {reg.upper()}", flush=True)
        r = run_regression_cv(Xf_reg, y_reg, folds, model_name=reg, use_grid_search=False)
        save_results(r, f"classical_reg_{reg}")
        print(f"      Pooled RMSE H: {r['pooled_rmse_h_deg']:.3f}°, V: {r['pooled_rmse_v_deg']:.3f}°", flush=True)

    print("\n[4/5] Running Phase 9 & Phase 10 Ablations...", flush=True)
    angle_pred, angle_true, test_meta = rebuild_angle_preds()
    run_head_pose_ablation(angle_pred, angle_true, test_meta)
    run_direction_ablation_full(angle_pred, angle_true, test_meta)

    print("\n[5/5] Generating Master Results Table & Comparison CSV...", flush=True)
    run_comparison()

    elapsed = (time.time() - t_start) / 60.0
    print("\n" + "=" * 80, flush=True)
    print(f"  FULL EVALUATION COMPLETE in {elapsed:.1f} minutes", flush=True)
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    main()
