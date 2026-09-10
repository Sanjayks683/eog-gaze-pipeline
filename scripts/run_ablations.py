"""
scripts/run_ablations.py
========================
Phase 9 + Phase 10: Re-run inference on test folds using saved checkpoints
to produce angle_pred.npy / angle_true.npy, then run:
  - Phase 9: head-pose ablation (Dataset 3 vs Datasets 1/2/4)
  - Phase 10: direction-invariance ablation (Dataset 4)

Usage:
    python scripts/run_ablations.py
"""

from __future__ import annotations

import os
import pickle
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import CFG, get_device
from src.data.datasets import load_processed
from src.models.deep_multitask import build_model, build_loss
from src.training.cv_splits import load_folds
from src.training.train import (
    make_dataloader, eval_epoch, checkpoint_path, load_checkpoint, config_fingerprint,
)


def _cached_predictions_are_fresh(preds_dir: str, model_type: str, n_folds: int) -> bool:
    """True if cached angle predictions are newer than every usable checkpoint."""
    pred_path = os.path.join(preds_dir, f"angle_pred_{model_type}.npy")
    true_path = os.path.join(preds_dir, f"angle_true_{model_type}.npy")
    if not (os.path.isfile(pred_path) and os.path.isfile(true_path)):
        return False
    ckpt_dir = os.path.join(CFG.paths.data_processed, "checkpoints")
    for fold_i in range(n_folds):
        ckpt = checkpoint_path(ckpt_dir, model_type, fold_i)
        if os.path.isfile(ckpt) and os.path.getmtime(ckpt) > os.path.getmtime(pred_path):
            return False
    return True


def rebuild_angle_preds(model_type: Optional[str] = None):
    """Re-run inference using saved best-fold checkpoints and save predictions."""
    preds_dir = os.path.join(CFG.paths.data_processed, "angle_preds")
    model_type = model_type or CFG.model.model_type

    folds, _ = load_folds()

    pred_path = os.path.join(preds_dir, f"angle_pred_{model_type}.npy")
    true_path = os.path.join(preds_dir, f"angle_true_{model_type}.npy")
    meta_path = os.path.join(preds_dir, f"test_meta_{model_type}.pkl")

    if _cached_predictions_are_fresh(preds_dir, model_type, len(folds)):
        print(f"Cached angle predictions for '{model_type}' are up to date — loading from disk.")
        angle_pred = np.load(pred_path)
        angle_true = np.load(true_path)
        with open(meta_path, "rb") as f:
            test_meta = pickle.load(f)
        return angle_pred, angle_true, test_meta

    print(f"Rebuilding angle predictions for '{model_type}' from saved checkpoints …")
    os.makedirs(preds_dir, exist_ok=True)

    device = get_device()
    X, y_cls, meta = load_processed("classification")
    _, y_reg, _    = load_processed("regression")

    if len(meta) != len(X):
        raise ValueError(
            f"metadata length ({len(meta)}) != number of windows ({len(X)}). "
            "Re-run preprocessing."
        )

    n_ch = int(X.shape[1])
    if getattr(CFG.model, "in_channels", None) != n_ch:
        print(f"Setting CFG.model.in_channels = {n_ch} (from data)")
        CFG.model.in_channels = n_ch
    fingerprint = config_fingerprint(X.shape)

    criterion = build_loss(device)
    all_pred, all_true, all_meta = [], [], []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        ckpt = checkpoint_path(os.path.join(CFG.paths.data_processed, "checkpoints"), model_type, fold_i)
        if not os.path.isfile(ckpt):
            print(f"  [WARN] Checkpoint not found: {ckpt} — skipping fold {fold_i}")
            continue

        X_te      = X[test_idx]
        y_cls_te  = y_cls[test_idx]
        y_ang_te  = y_reg[test_idx]
        meta_te   = [meta[i] for i in test_idx]

        valid = ~np.isnan(y_ang_te).any(axis=1)
        X_te, y_cls_te, y_ang_te = X_te[valid], y_cls_te[valid], y_ang_te[valid]
        meta_te = [meta_te[i] for i, v in enumerate(valid) if v]

        loader = make_dataloader(X_te, y_cls_te, y_ang_te, CFG.model.batch_size, shuffle=False)

        model = build_model(model_type=model_type, device=device)
        if not load_checkpoint(ckpt, model, model_type, device, fingerprint):
            continue

        metrics = eval_epoch(model, loader, criterion, device)
        all_pred.append(metrics["angle_pred"])
        all_true.append(metrics["angle_true"])
        all_meta.extend(meta_te)
        print(f"  Fold {fold_i}: {len(meta_te)} test windows  "
              f"rmse_H={metrics['rmse_h_deg']:.3f}°  rmse_V={metrics['rmse_v_deg']:.3f}°")

    if not all_pred:
        raise FileNotFoundError(
            f"No usable checkpoints found for model_type={model_type!r} "
            "(missing, or trained under a different config/data). "
            f"Run `python main.py --phase train_deep` first."
        )

    angle_pred = np.vstack(all_pred)
    angle_true = np.vstack(all_true)

    np.save(pred_path, angle_pred)
    np.save(true_path, angle_true)
    with open(meta_path, "wb") as f:
        pickle.dump(all_meta, f)
    print(f"\nSaved {len(angle_pred)} test windows -> {preds_dir}")
    return angle_pred, angle_true, all_meta


def run_head_pose_ablation(angle_pred, angle_true, metadata):
    """
    Compare regression error on Dataset 3 (non-stationary / moving head)
    vs all other datasets using src.ablations.head_pose_compensation.
    """
    has_d3 = any(m.get("dataset_source") == "dataset3" for m in metadata)
    if not has_d3:
        print("[SKIP] Phase 9 Head-Pose Ablation skipped (Dataset 3 not in current dataset scope).")
        return {}
    from src.ablations.head_pose_compensation import run_head_pose_ablation_by_metadata
    return run_head_pose_ablation_by_metadata(
        angle_pred=angle_pred,
        angle_true=angle_true,
        metadata=metadata,
    )


def run_direction_ablation_full(angle_pred, angle_true, metadata):
    """Run the direction-invariance ablation (Phase 10) using the
    already-imported function from src.ablations.direction_invariance."""
    mask = np.array([m.get("dataset_source") == "dataset4" for m in metadata])
    if not np.any(mask):
        print("[SKIP] Phase 10 Direction-Invariance Ablation skipped (Dataset 4 not in current dataset scope).")
        return {}

    from src.ablations.direction_invariance import run_direction_ablation
    meta_filt  = [metadata[i] for i in range(len(metadata)) if mask[i]]
    results = run_direction_ablation(
        angle_true=angle_true[mask],
        angle_pred=angle_pred[mask],
        metadata=meta_filt,
    )
    return results


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  EOG Pipeline — Phase 9 + 10 Ablation Runner")
    print("="*60)

    angle_pred, angle_true, test_meta = rebuild_angle_preds()

    print("\n--- Phase 9: Head-Pose Ablation ---")
    run_head_pose_ablation(angle_pred, angle_true, test_meta)

    print("\n--- Phase 10: Direction-Invariance Ablation ---")
    run_direction_ablation_full(angle_pred, angle_true, test_meta)

    print("\nAll ablations complete.")
