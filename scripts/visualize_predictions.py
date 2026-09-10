"""
scripts/visualize_predictions.py
================================
Plot a continuous segment of true vs. predicted gaze angles from the saved
angle predictions of the currently configured deep model.

Usage:
    python scripts/visualize_predictions.py            # uses CFG.model.model_type
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from src.config import CFG


def plot_predictions():
    model_type = CFG.model.model_type
    preds_dir = os.path.join(CFG.paths.data_processed, "angle_preds")
    pred_path = os.path.join(preds_dir, f"angle_pred_{model_type}.npy")
    true_path = os.path.join(preds_dir, f"angle_true_{model_type}.npy")

    if not os.path.isfile(pred_path) or not os.path.isfile(true_path):
        print(f"Error: predictions for model_type={model_type!r} not found in {preds_dir}. "
              f"Run `python main.py --phase train_deep --model-type {model_type}` first.")
        return

    y_pred = np.load(pred_path)
    y_true = np.load(true_path)

    movement_mask = np.abs(y_true[:, 0]) > 5.0
    valid_indices = np.where(movement_mask)[0]

    if len(valid_indices) == 0:
        start_idx = 0
    else:
        start_idx = max(0, valid_indices[len(valid_indices)//2] - 50)

    num_windows = 200
    end_idx = min(len(y_true), start_idx + num_windows)

    slice_true = y_true[start_idx:end_idx]
    slice_pred = y_pred[start_idx:end_idx]
    stride_s = CFG.segmentation.stride_ms / 1000.0
    time_axis = np.arange(len(slice_true)) * stride_s

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(time_axis, slice_true[:, 0], label="True Horizontal Angle", color='black', linewidth=2)
    ax1.plot(time_axis, slice_pred[:, 0], label="Predicted Horizontal Angle", color='tab:blue', linestyle='--', linewidth=2)
    ax1.set_ylabel("Angle (degrees)")
    ax1.set_title(f"Horizontal EOG Gaze Regression — {model_type} (Windows {start_idx} to {end_idx})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(time_axis, slice_true[:, 1], label="True Vertical Angle", color='black', linewidth=2)
    ax2.plot(time_axis, slice_pred[:, 1], label="Predicted Vertical Angle", color='tab:red', linestyle='--', linewidth=2)
    ax2.set_ylabel("Angle (degrees)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_title("Vertical EOG Gaze Regression")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    out_dir = CFG.paths.reports_figures
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"inference_sample_{model_type}.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    plot_predictions()
