"""
scripts/run_finetuning.py
==========================
Subject-specific fine-tuning script.

Takes a pre-trained generalized model and fine-tunes it on a small
calibration set from a specific subject, then evaluates on their test set.

Usage:
    python scripts/run_finetuning.py
    python scripts/run_finetuning.py --dataset dataset2 --subject S1
    python scripts/run_finetuning.py --dataset dataset3 --subject S5 --freeze-backbone
"""

import os
import sys
import argparse
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.config import CFG, get_device
from src.data.datasets import load_processed
from src.models.deep_multitask import build_model, build_loss
from src.training.train import (
    make_dataloader, eval_epoch, train_epoch, checkpoint_path, load_checkpoint,
)


def find_best_checkpoint(checkpoint_dir: str, model_type: str) -> str:
    """Find the first checkpoint for the given architecture (never another model's)."""
    pattern = os.path.join(checkpoint_dir, f"best_{model_type}_fold*.pt")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No {model_type!r} checkpoints found matching {pattern}. "
            "Run the training pipeline first."
        )
    return candidates[0]


def run_finetuning(args):
    device = get_device()
    print(f"Using device: {device}")
    
    print("Loading data...")
    X_reg, y_reg, _ = load_processed("regression")
    X_cls, y_cls, meta = load_processed("classification")
    
    target_ds = args.dataset
    target_subj = args.subject
    
    subject_mask = np.array([
        m.get("dataset_source") == target_ds and m.get("subject_id") == target_subj
        for m in meta
    ])
    if not subject_mask.any():
        print(f"Error: Could not find any data for {target_ds} - {target_subj}.")
        available_subjects = set(
            (m.get("dataset_source"), m.get("subject_id")) for m in meta
        )
        print(f"Available subjects: {sorted(available_subjects)}")
        return
        
    X_subj = X_reg[subject_mask]
    y_reg_subj = y_reg[subject_mask]
    y_cls_subj = y_cls[subject_mask]
    
    valid_mask = ~np.isnan(y_reg_subj).any(axis=1)
    X_subj = X_subj[valid_mask]
    y_reg_subj = y_reg_subj[valid_mask]
    y_cls_subj = y_cls_subj[valid_mask]
    
    print(f"Found {len(X_subj)} valid continuous windows for {target_ds} {target_subj}.")
    
    calib_fraction = args.calib_fraction
    calib_size = int(len(X_subj) * calib_fraction)
    
    X_calib = X_subj[:calib_size]
    y_reg_calib = y_reg_subj[:calib_size]
    y_cls_calib = y_cls_subj[:calib_size]
    
    X_test = X_subj[calib_size:]
    y_reg_test = y_reg_subj[calib_size:]
    y_cls_test = y_cls_subj[calib_size:]
    
    print(f"Split: {len(X_calib)} Calibration windows, {len(X_test)} Test windows.")
    
    batch_size = CFG.model.batch_size
    calib_loader = make_dataloader(X_calib, y_cls_calib, y_reg_calib, batch_size=batch_size, shuffle=True)
    test_loader = make_dataloader(X_test, y_cls_test, y_reg_test, batch_size=batch_size, shuffle=False)
    
    checkpoint_dir = os.path.join(CFG.paths.data_processed, "checkpoints")
    model_type = CFG.model.model_type
    ckpt = find_best_checkpoint(checkpoint_dir, model_type)
    print(f"Using checkpoint: {ckpt}")

    model_baseline = build_model(device=device)
    if not load_checkpoint(ckpt, model_baseline, model_type, device):
        raise RuntimeError(f"Failed to load checkpoint {ckpt}")
    criterion = build_loss(device=device)

    baseline_metrics = eval_epoch(model_baseline, test_loader, criterion, device)
    print(f"\n[BASELINE] Generalized Model on Test Set:")
    print(f"  RMSE_H = {baseline_metrics['rmse_h_deg']:.3f}°")
    print(f"  RMSE_V = {baseline_metrics['rmse_v_deg']:.3f}°")

    model_finetuned = build_model(device=device)
    if not load_checkpoint(ckpt, model_finetuned, model_type, device):
        raise RuntimeError(f"Failed to load checkpoint {ckpt}")
    
    if args.freeze_backbone:
        for name, param in model_finetuned.named_parameters():
            if "cls_head" not in name and "reg_head" not in name:
                param.requires_grad = False
        print("Fine-tuning mode: FROZEN backbone (heads only)")
    else:
        for param in model_finetuned.parameters():
            param.requires_grad = True
        print("Fine-tuning mode: UNFROZEN (all parameters)")
        
    ft_lr = args.lr if args.lr else CFG.model.lr * 0.5
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_finetuned.parameters()), 
        lr=ft_lr,
        weight_decay=CFG.model.weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    
    epochs = args.epochs if args.epochs else CFG.model.max_epochs
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    early_stop_patience = 15
    
    print(f"\nFine-Tuning on {len(X_calib)} calibration windows for up to {epochs} epochs...")
    for epoch in range(epochs):
        model_finetuned.train()
        train_metrics = train_epoch(model_finetuned, calib_loader, optimizer, criterion, device)
        
        current_loss = train_metrics['loss']
        scheduler.step(current_loss)
        
        if current_loss < best_val_loss:
            best_val_loss = current_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model_finetuned.state_dict().items()}
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Calib Loss: {current_loss:.4f} | MSE: {train_metrics['mse_loss']:.4f}")
        
        if patience_counter >= early_stop_patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    if best_state is not None:
        model_finetuned.load_state_dict(best_state)
            
    finetuned_metrics = eval_epoch(model_finetuned, test_loader, criterion, device)
    print(f"\n[FINETUNED] Subject-Specific Model on exact same Test Set:")
    print(f"  RMSE_H = {finetuned_metrics['rmse_h_deg']:.3f}°")
    print(f"  RMSE_V = {finetuned_metrics['rmse_v_deg']:.3f}°")
    
    delta_h = baseline_metrics['rmse_h_deg'] - finetuned_metrics['rmse_h_deg']
    delta_v = baseline_metrics['rmse_v_deg'] - finetuned_metrics['rmse_v_deg']
    print(f"\n  Improvement: ΔH = {delta_h:+.3f}°, ΔV = {delta_v:+.3f}°")
    
    plot_len = min(300, len(y_reg_test))
    
    movement = np.abs(y_reg_test[:, 0]) > 5
    valid_starts = np.where(movement)[0]
    start_idx = max(0, valid_starts[len(valid_starts)//2] - 100) if len(valid_starts) > 0 else 0
    end_idx = start_idx + plot_len
    
    time_axis = np.arange(plot_len) * (CFG.segmentation.stride_ms / 1000.0)
    
    true_H = y_reg_test[start_idx:end_idx, 0]
    base_H = baseline_metrics['angle_pred'][start_idx:end_idx, 0]
    fine_H = finetuned_metrics['angle_pred'][start_idx:end_idx, 0]
    
    true_V = y_reg_test[start_idx:end_idx, 1]
    base_V = baseline_metrics['angle_pred'][start_idx:end_idx, 1]
    fine_V = finetuned_metrics['angle_pred'][start_idx:end_idx, 1]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    ax1.plot(time_axis, true_H, label="True Gaze Angle", color="black", linewidth=2.5)
    ax1.plot(time_axis, base_H, label="Baseline (Generalized)", color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax1.plot(time_axis, fine_H, label="Fine-Tuned (Subject-Specific)", color="tab:blue", linestyle="-", linewidth=2.0)
    ax1.set_ylabel("Horizontal Angle (°)")
    ax1.set_title(f"Horizontal EOG Gaze - {target_ds} {target_subj} Test Set")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(time_axis, true_V, label="True Gaze Angle", color="black", linewidth=2.5)
    ax2.plot(time_axis, base_V, label="Baseline (Generalized)", color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax2.plot(time_axis, fine_V, label="Fine-Tuned (Subject-Specific)", color="tab:green", linestyle="-", linewidth=2.0)
    ax2.set_ylabel("Vertical Angle (°)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_title(f"Vertical EOG Gaze")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_dir = os.path.join("reports", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "finetuning_comparison.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\nComparison plot saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subject-specific fine-tuning")
    parser.add_argument("--dataset", type=str, default="dataset2",
                        help="Dataset source (default: dataset2)")
    parser.add_argument("--subject", type=str, default="S1",
                        help="Subject ID (default: S1)")
    parser.add_argument("--calib-fraction", type=float, default=0.15,
                        help="Fraction of data for calibration (default: 0.15)")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze backbone, only train heads (recommended for small calibration sets)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Max fine-tuning epochs (default: CFG.model.max_epochs)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Fine-tuning learning rate (default: CFG.model.lr * 0.5)")
    args = parser.parse_args()
    run_finetuning(args)
