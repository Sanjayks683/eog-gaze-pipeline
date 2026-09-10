"""
src/ablations/head_pose_compensation.py
========================================
Phase 9: Dataset 3 head-pose compensation ablation.

Compares regression error on Dataset 3:
  - Against raw target_angle (no correction)
  - Against target_angle_corrected (head-pose subtracted)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from src.config import CFG


def run_head_pose_ablation_by_metadata(
    angle_pred: np.ndarray,
    angle_true: np.ndarray,
    metadata: List[dict],
    results_dir: str = None,
) -> Dict:
    """
    Compare regression error on Dataset 3 (moving head) vs stationary datasets
    using window metadata. Primary entry point for the Phase 9 ablation.
    """
    if results_dir is None:
        results_dir = CFG.paths.results
    os.makedirs(results_dir, exist_ok=True)

    has_d3 = any(m.get("dataset_source") == "dataset3" for m in metadata)
    if not has_d3:
        print("[SKIP] Head-pose ablation skipped: Dataset 3 (moving head) is not present in the loaded dataset scope.")
        return {}

    groups = {
        "dataset3 (moving head)": "dataset3",
        "others (stationary)": None,
    }

    records = {}
    for label, ds_filter in groups.items():
        if ds_filter is not None:
            mask = np.array([m.get("dataset_source") == ds_filter for m in metadata])
        else:
            mask = np.array([m.get("dataset_source") != "dataset3" for m in metadata])

        if not np.any(mask):
            records[label] = {"rmse_h": None, "rmse_v": None, "n": 0}
            continue

        h_err = float(np.sqrt(np.mean((angle_pred[mask, 0] - angle_true[mask, 0]) ** 2)))
        v_err = float(np.sqrt(np.mean((angle_pred[mask, 1] - angle_true[mask, 1]) ** 2)))
        records[label] = {"rmse_h": h_err, "rmse_v": v_err, "n": int(mask.sum())}

    path = os.path.join(results_dir, "head_pose_ablation.json")
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Head-pose ablation saved: {path}")

    return records


def run_head_pose_ablation(
    angle_true_raw: np.ndarray,
    angle_true_corrected: np.ndarray,
    angle_pred: np.ndarray,
    results_dir: str = None,
    figures_dir: str = None,
) -> Dict:
    """
    Compare regression error against raw vs. head-pose-corrected labels
    (both (n, 2) arrays) and plot pred-vs-true scatters for both variants.
    For the metadata-based comparison across datasets use
    run_head_pose_ablation_by_metadata().
    """
    if results_dir is None:
        results_dir = CFG.paths.results
    if figures_dir is None:
        figures_dir = CFG.paths.reports_figures

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    rmse_h_raw = float(np.sqrt(np.mean((angle_pred[:, 0] - angle_true_raw[:, 0]) ** 2)))
    rmse_v_raw = float(np.sqrt(np.mean((angle_pred[:, 1] - angle_true_raw[:, 1]) ** 2)))

    rmse_h_corr = float(np.sqrt(np.mean((angle_pred[:, 0] - angle_true_corrected[:, 0]) ** 2)))
    rmse_v_corr = float(np.sqrt(np.mean((angle_pred[:, 1] - angle_true_corrected[:, 1]) ** 2)))

    pct_h = 100.0 * (rmse_h_raw - rmse_h_corr) / (rmse_h_raw + 1e-9)
    pct_v = 100.0 * (rmse_v_raw - rmse_v_corr) / (rmse_v_raw + 1e-9)

    results = {
        "dataset": "dataset3",
        "raw_labels": {"rmse_h_deg": rmse_h_raw, "rmse_v_deg": rmse_v_raw},
        "corrected_labels": {"rmse_h_deg": rmse_h_corr, "rmse_v_deg": rmse_v_corr},
        "pct_reduction_rmse_h": pct_h,
        "pct_reduction_rmse_v": pct_v,
        "interpretation": (
            f"H: {pct_h:+.1f}% | V: {pct_v:+.1f}% error change after head-pose correction. "
            "Positive = improvement."
        ),
    }

    path = os.path.join(results_dir, "head_pose_ablation.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Head-pose ablation results saved: {path}")
    print(f"  Raw:       RMSE H={rmse_h_raw:.3f}°  V={rmse_v_raw:.3f}°")
    print(f"  Corrected: RMSE H={rmse_h_corr:.3f}°  V={rmse_v_corr:.3f}°")
    print(f"  Reduction: H={pct_h:+.1f}%  V={pct_v:+.1f}%")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Dataset 3 — Head-Pose Compensation Ablation\n"
                 "Predicted vs. True Gaze Angle", fontsize=14)

    for col, (true_arr, label) in enumerate(
        [(angle_true_raw, "Raw (no correction)"), (angle_true_corrected, "Corrected")]
    ):
        for row, ch_name in enumerate(["H (horizontal)", "V (vertical)"]):
            ax = axes[row, col]
            t = true_arr[:, row]
            p = angle_pred[:, row]
            lim = max(np.abs(t).max(), np.abs(p).max()) * 1.1
            ax.scatter(t, p, alpha=0.3, s=5, color="steelblue")
            ax.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="Perfect")
            rmse = float(np.sqrt(np.mean((p - t) ** 2)))
            ax.set_title(f"{ch_name} — {label}\nRMSE={rmse:.3f}°")
            ax.set_xlabel("True angle (°)")
            ax.set_ylabel("Predicted angle (°)")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "head_pose_ablation_scatter.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Scatter plot saved: {fig_path}")

    return results
