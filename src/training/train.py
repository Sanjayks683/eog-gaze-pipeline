"""
src/training/train.py
======================
Full training loop for EOGMultiTaskNet.
Includes: Adam optimizer, ReduceLROnPlateau, early stopping,
per-fold loss curve logging, and model checkpointing.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import CFG, get_device
from src.models.deep_multitask import MultiTaskLoss, build_model, build_loss
from src.training.cv_splits import FoldList


def checkpoint_path(checkpoint_dir: str, model_type: str, fold_i: int) -> str:
    """
    Per-architecture, per-fold checkpoint path.
    The model_type prefix is REQUIRED: different architectures must never share
    a checkpoint file (a state_dict from another architecture either crashes
    load_state_dict or silently corrupts evaluation).
    """
    return os.path.join(checkpoint_dir, f"best_{model_type}_fold{fold_i}.pt")


def config_fingerprint(input_shape: Tuple[int, ...]) -> str:
    """
    Hash of everything that determines a trained model's weights: model,
    preprocessing, segmentation, augmentation and data config, CV split
    settings, and the input data shape. Stored in every checkpoint so one
    trained under a different config is retrained instead of silently reused.
    """
    model_cfg = dataclasses.asdict(CFG.model)
    model_cfg.pop("calibration_prompt_sec", None)
    payload = {
        "model": model_cfg,
        "preprocessing": dataclasses.asdict(CFG.preprocessing),
        "segmentation": dataclasses.asdict(CFG.segmentation),
        "augmentation": dataclasses.asdict(CFG.augmentation),
        "data": dataclasses.asdict(CFG.data),
        "cv": {k: getattr(CFG.cv, k) for k in ("strategy", "k", "random_seed")},
        "input_shape": [int(s) for s in input_shape],
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def load_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    expected_model_type: str,
    device: torch.device,
    expected_fingerprint: Optional[str] = None,
) -> bool:
    """
    Load a checkpoint into `model`. Returns True on success, False if the
    checkpoint belongs to a different architecture, was trained under a
    different config (when `expected_fingerprint` is given), or fails to load.
    """
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception as e:
        warnings.warn(f"Could not read checkpoint {ckpt_path}: {e}")
        return False

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        if ckpt.get("model_type") != expected_model_type:
            warnings.warn(
                f"Checkpoint {ckpt_path} was saved for model_type="
                f"{ckpt.get('model_type')!r}, expected {expected_model_type!r}. Ignoring it."
            )
            return False
        state_dict = ckpt["state_dict"]
        found_fingerprint = ckpt.get("fingerprint")
    else:
        state_dict = ckpt
        found_fingerprint = None

    if expected_fingerprint is not None and found_fingerprint != expected_fingerprint:
        warnings.warn(
            f"Checkpoint {ckpt_path} was trained under a different config/data "
            f"(fingerprint {found_fingerprint!r} != {expected_fingerprint!r}). Ignoring it."
        )
        return False

    try:
        model.load_state_dict(state_dict)
        return True
    except RuntimeError as e:
        warnings.warn(f"Checkpoint {ckpt_path} does not match the model: {e}")
        return False


def make_dataloader(
    X: np.ndarray,
    y_class: np.ndarray,
    y_angle: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader from numpy arrays."""
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y_class).long(),
        torch.from_numpy(y_angle).float(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available())


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: MultiTaskLoss,
    device: torch.device,
) -> Dict[str, float]:
    from src.preprocessing.augmentation import augment_eog_batch_tensor, mixup_batch

    model.train()
    total_loss = ce_total = mse_total = 0.0
    correct = total = 0
    
    mixup_alpha = getattr(CFG.model, "mixup_alpha", 0.0)

    for X_batch, y_cls, y_ang in loader:
        X_batch = X_batch.to(device)
        y_cls = y_cls.to(device)
        y_ang = y_ang.to(device)

        if getattr(CFG.preprocessing, "use_augmentation", False):
            aug = CFG.augmentation
            X_batch = augment_eog_batch_tensor(
                X_batch,
                scale_range=tuple(aug.scale_range),
                noise_std=aug.noise_std,
                channel_dropout_prob=aug.channel_dropout_prob,
            )

        if mixup_alpha > 0:
            X_batch, y_cls_a, y_cls_b, y_ang_mixed, lam = mixup_batch(
                X_batch, y_cls, y_ang, alpha=mixup_alpha
            )
        else:
            y_cls_a = y_cls_b = y_cls
            y_ang_mixed = y_ang
            lam = 1.0

        optimizer.zero_grad()
        logits, angles = model(X_batch)
        
        if mixup_alpha > 0 and lam < 1.0:
            loss_a, ce_a, mse_a = criterion(logits, y_cls_a, angles, y_ang_mixed)
            loss_b, ce_b, _ = criterion(logits, y_cls_b, angles, y_ang_mixed)
            loss = lam * loss_a + (1 - lam) * loss_b
            ce = lam * ce_a + (1 - lam) * ce_b
            mse = mse_a
        else:
            loss, ce, mse = criterion(logits, y_cls, angles, y_ang)
        
        loss.backward()
        clip_norm = getattr(CFG.model, "grad_clip_norm", 0.0)
        if clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()

        total_loss += loss.item() * len(X_batch)
        ce_total += ce.item() * len(X_batch)
        mse_total += mse.item() * len(X_batch)
        correct += (logits.argmax(1) == y_cls_a).sum().item()
        total += len(X_batch)

    n = total or 1
    return {
        "loss": total_loss / n,
        "ce_loss": ce_total / n,
        "mse_loss": mse_total / n,
        "acc": correct / n,
    }


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = ce_total = mse_total = 0.0
    correct = total = 0
    all_preds, all_true, all_angle_pred, all_angle_true = [], [], [], []

    with torch.no_grad():
        for X_batch, y_cls, y_ang in loader:
            X_batch = X_batch.to(device)
            y_cls = y_cls.to(device)
            y_ang = y_ang.to(device)

            logits, angles = model(X_batch)
            loss, ce, mse = criterion(logits, y_cls, angles, y_ang)

            total_loss += loss.item() * len(X_batch)
            ce_total += ce.item() * len(X_batch)
            mse_total += mse.item() * len(X_batch)
            preds = logits.argmax(1)
            correct += (preds == y_cls).sum().item()
            total += len(X_batch)

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y_cls.cpu().numpy())
            all_angle_pred.append(angles.cpu().numpy())
            all_angle_true.append(y_ang.cpu().numpy())

    n = total or 1
    angle_pred = np.vstack(all_angle_pred)
    angle_true = np.vstack(all_angle_true)
    rmse_h = float(np.sqrt(np.mean((angle_pred[:, 0] - angle_true[:, 0]) ** 2)))
    rmse_v = float(np.sqrt(np.mean((angle_pred[:, 1] - angle_true[:, 1]) ** 2)))

    return {
        "loss": total_loss / n,
        "ce_loss": ce_total / n,
        "mse_loss": mse_total / n,
        "acc": correct / n,
        "rmse_h_deg": rmse_h,
        "rmse_v_deg": rmse_v,
        "preds": np.array(all_preds),
        "true": np.array(all_true),
        "angle_pred": angle_pred,
        "angle_true": angle_true,
    }


def compute_balanced_class_weights(
    y_train: np.ndarray,
    num_classes: int,
) -> Optional[torch.Tensor]:
    """
    "Balanced" class weights (sklearn convention): n_samples / (n_classes * count_c),
    computed from the TRAIN labels of this fold only — never from validation or
    test data. Classes absent from training get weight 0 (they cannot occur in
    a consistent label space anyway). Returns None when the loss should stay uniform.
    """
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    present = counts > 0
    if not present.any() or present.sum() <= 1:
        return None
    weights = np.zeros(num_classes, dtype=np.float64)
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32)


def validation_indices(
    train_idx: np.ndarray,
    metadata: Optional[List[dict]],
    seed: int,
) -> np.ndarray:
    """
    Positions (into train_idx) of this fold's validation windows.

    Holds out ~10% of the fold's TRAINING subjects (at least one) so early
    stopping measures generalization to unseen subjects — random windows would
    overlap (50% stride) with training windows and track train loss instead.
    Falls back to a random 10% of windows when subject ids are unknown or the
    fold has fewer than two training subjects.
    """
    rng = np.random.RandomState(seed)
    if metadata is not None:
        subjects = np.array([metadata[i]["subject_id"] for i in train_idx])
        unique = np.unique(subjects)
        if len(unique) >= 2:
            n_val = max(1, int(round(0.1 * len(unique))))
            val_subjects = rng.choice(unique, n_val, replace=False)
            return np.flatnonzero(np.isin(subjects, val_subjects))
    n_val = max(1, int(len(train_idx) * 0.1))
    return rng.choice(len(train_idx), n_val, replace=False)


def train_cv(
    X: np.ndarray,
    y_class: np.ndarray,
    y_angle: np.ndarray,
    folds: FoldList,
    metadata: Optional[List[dict]] = None,
    results_dir: str = None,
    checkpoint_dir: str = None,
    preds_dir: str = None,
) -> Dict:
    """
    Train EOGMultiTaskNet across all CV folds.

    Parameters
    ----------
    X        : (n_windows, 2, window_len) float32
    y_class  : (n_windows,) int64
    y_angle  : (n_windows, 2) float32
    folds    : list of (train_idx, test_idx)
    metadata : per-window metadata list; must match len(X). If None, loaded
               from the saved folds file.
    results_dir   : where to save loss curves and results JSON
    checkpoint_dir: where to save best model per fold
    preds_dir     : where to save angle predictions for ablation scripts

    Returns
    -------
    dict with per-fold and aggregate results
    """
    if results_dir is None:
        results_dir = CFG.paths.results
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(CFG.paths.data_processed, "checkpoints")
    if preds_dir is None:
        preds_dir = os.path.join(CFG.paths.data_processed, "angle_preds")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    device = get_device()
    print(f"\nTraining on device: {device}")

    batch_size = CFG.model.batch_size
    max_epochs = CFG.model.max_epochs
    patience = CFG.model.early_stopping_patience
    model_type = CFG.model.model_type

    if metadata is None:
        try:
            from src.training.cv_splits import load_folds
            _, metadata = load_folds()
        except FileNotFoundError:
            metadata = None
    if metadata is not None and len(metadata) != len(X):
        raise ValueError(
            f"metadata length ({len(metadata)}) does not match number of "
            f"windows ({len(X)}). Re-run preprocessing so both are regenerated "
            "from the same data."
        )

    fingerprint = config_fingerprint(X.shape)

    all_fold_results = []
    all_preds, all_true, all_angle_pred, all_angle_true = [], [], [], []
    all_test_meta = []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"  Fold {fold_i + 1} / {len(folds)}")
        print(f"{'='*60}")

        X_tr, X_te = X[train_idx], X[test_idx]
        y_cls_tr, y_cls_te = y_class[train_idx], y_class[test_idx]
        y_ang_tr, y_ang_te = y_angle[train_idx], y_angle[test_idx]
        meta_te = [metadata[i] for i in test_idx] if metadata is not None else []

        val_idx = validation_indices(train_idx, metadata, CFG.cv.random_seed + fold_i)
        if metadata is not None:
            val_subjects = sorted({metadata[train_idx[i]]["subject_id"] for i in val_idx})
            print(f"  Validation: {len(val_idx)} windows from subjects {val_subjects}")
        train_mask = np.ones(len(X_tr), dtype=bool)
        train_mask[val_idx] = False

        X_va, y_va_c, y_va_r = X_tr[val_idx], y_cls_tr[val_idx], y_ang_tr[val_idx]
        X_tr, y_tr_c, y_tr_r = X_tr[train_mask], y_cls_tr[train_mask], y_ang_tr[train_mask]

        valid_tr = ~np.isnan(y_tr_r).any(axis=1)
        X_tr, y_tr_c, y_tr_r = X_tr[valid_tr], y_tr_c[valid_tr], y_tr_r[valid_tr]
        
        valid_va = ~np.isnan(y_va_r).any(axis=1)
        X_va, y_va_c, y_va_r = X_va[valid_va], y_va_c[valid_va], y_va_r[valid_va]

        valid_te = ~np.isnan(y_ang_te).any(axis=1)
        X_te, y_cls_te, y_ang_te = X_te[valid_te], y_cls_te[valid_te], y_ang_te[valid_te]
        if meta_te:
            meta_te = [meta_te[i] for i, v in enumerate(valid_te) if v]

        train_loader = make_dataloader(X_tr, y_tr_c, y_tr_r, batch_size, shuffle=True)
        val_loader = make_dataloader(X_va, y_va_c, y_va_r, batch_size, shuffle=False)
        test_loader = make_dataloader(X_te, y_cls_te, y_ang_te, batch_size, shuffle=False)

        model = build_model(device=device)
        fold_class_weights = None
        if CFG.model.class_weights is not None:
            fold_class_weights = torch.tensor(
                CFG.model.class_weights, dtype=torch.float32, device=device
            )
        elif getattr(CFG.model, "auto_class_weights", False):
            fold_class_weights = compute_balanced_class_weights(
                y_tr_c, CFG.model.num_classes
            )
            if fold_class_weights is not None:
                fold_class_weights = fold_class_weights.to(device)
                print(f"  Auto class weights (from train fold): "
                      f"{[round(w, 3) for w in fold_class_weights.tolist()]}")
        criterion = build_loss(device=device, class_weights=fold_class_weights)
        params = list(model.parameters()) + list(criterion.parameters())
        optimizer = torch.optim.Adam(
            params,
            lr=CFG.model.lr,
            weight_decay=CFG.model.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=CFG.model.lr_scheduler_patience,
            factor=CFG.model.lr_scheduler_factor,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        train_history, val_history = [], []
        best_ckpt = checkpoint_path(checkpoint_dir, model_type, fold_i)

        if os.path.exists(best_ckpt) and load_checkpoint(
            best_ckpt, model, model_type, device, fingerprint
        ):
            print(f"  Checkpoint {best_ckpt} matches this architecture and config. "
                  "Skipping training and evaluating...")
        else:
            for epoch in range(max_epochs):
                t0 = time.time()
                tr_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
                val_metrics = eval_epoch(model, val_loader, criterion, device)
                scheduler.step(val_metrics["loss"])

                train_history.append(tr_metrics)
                val_history.append(val_metrics)

                elapsed = time.time() - t0
                print(
                    f"  Epoch {epoch+1:3d}/{max_epochs} | "
                    f"tr_loss={tr_metrics['loss']:.4f} acc={tr_metrics['acc']:.3f} | "
                    f"val_loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.3f} "
                    f"rmse_H={val_metrics['rmse_h_deg']:.2f}° "
                    f"rmse_V={val_metrics['rmse_v_deg']:.2f}° | "
                    f"{elapsed:.1f}s",
                    flush=True,
                )

                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    patience_counter = 0
                    torch.save({"model_type": model_type, "fingerprint": fingerprint,
                                "state_dict": model.state_dict()}, best_ckpt)
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

            if os.path.exists(best_ckpt):
                load_checkpoint(best_ckpt, model, model_type, device, fingerprint)

        test_metrics = eval_epoch(model, test_loader, criterion, device)

        print(
            f"\n  Test results (fold {fold_i}):\n"
            f"    acc={test_metrics['acc']:.3f} "
            f"rmse_H={test_metrics['rmse_h_deg']:.2f}° "
            f"rmse_V={test_metrics['rmse_v_deg']:.2f}°"
        )

        curves_path = os.path.join(results_dir, f"loss_curves_{model_type}_fold{fold_i}.json")
        with open(curves_path, "w") as f:
            json.dump({
                "train": [{k: v for k, v in m.items() if not isinstance(v, np.ndarray)}
                          for m in train_history],
                "val": [{k: v for k, v in m.items() if not isinstance(v, np.ndarray)}
                        for m in val_history],
            }, f, indent=2)

        fold_result = {
            "fold": fold_i,
            "test_acc": float(test_metrics["acc"]),
            "test_rmse_h_deg": float(test_metrics["rmse_h_deg"]),
            "test_rmse_v_deg": float(test_metrics["rmse_v_deg"]),
            "best_val_loss": float(best_val_loss) if best_val_loss != float("inf") else -1.0,
            "epochs_trained": len(train_history),
        }
        all_fold_results.append(fold_result)
        all_preds.extend(test_metrics["preds"].tolist())
        all_true.extend(test_metrics["true"].tolist())
        all_angle_pred.append(test_metrics["angle_pred"])
        all_angle_true.append(test_metrics["angle_true"])
        all_test_meta.extend(meta_te)

    from sklearn.metrics import f1_score, confusion_matrix
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    all_preds_arr = np.array(all_preds)
    all_true_arr = np.array(all_true)
    all_ang_pred = np.vstack(all_angle_pred)
    all_ang_true = np.vstack(all_angle_true)

    aggregate = {
        "model": model.__class__.__name__,
        "model_type": model_type,
        "folds": all_fold_results,
        "pooled_f1_weighted": float(f1_score(all_true_arr, all_preds_arr, average="weighted", zero_division=0)),
        "pooled_f1_macro": float(f1_score(all_true_arr, all_preds_arr, average="macro", zero_division=0)),
        "pooled_confusion_matrix": confusion_matrix(all_true_arr, all_preds_arr).tolist(),
        "pooled_rmse_h_deg": float(np.sqrt(mean_squared_error(all_ang_true[:, 0], all_ang_pred[:, 0]))),
        "pooled_rmse_v_deg": float(np.sqrt(mean_squared_error(all_ang_true[:, 1], all_ang_pred[:, 1]))),
        "pooled_mae_h_deg": float(mean_absolute_error(all_ang_true[:, 0], all_ang_pred[:, 0])),
        "pooled_mae_v_deg": float(mean_absolute_error(all_ang_true[:, 1], all_ang_pred[:, 1])),
        "pooled_r2_h": float(r2_score(all_ang_true[:, 0], all_ang_pred[:, 0])),
        "pooled_r2_v": float(r2_score(all_ang_true[:, 1], all_ang_pred[:, 1])),
        "test_metadata": all_test_meta,
    }

    for fname in (f"deep_model_{model_type}.json", "deep_model_results.json"):
        results_path = os.path.join(results_dir, fname)
        with open(results_path, "w") as f:
            json.dump(aggregate, f, indent=2)
    print(f"\nDeep model results saved: "
          f"{os.path.join(results_dir, f'deep_model_{model_type}.json')}")

    np.save(os.path.join(preds_dir, f"angle_pred_{model_type}.npy"), all_ang_pred)
    np.save(os.path.join(preds_dir, f"angle_true_{model_type}.npy"), all_ang_true)
    import pickle as _pkl
    with open(os.path.join(preds_dir, f"test_meta_{model_type}.pkl"), "wb") as _f:
        _pkl.dump(aggregate.get("test_metadata", []), _f)
    print(f"Angle predictions saved to: {preds_dir}")

    return aggregate
