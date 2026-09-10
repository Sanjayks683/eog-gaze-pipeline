"""
src/ablations/direction_invariance.py
=======================================
Phase 10 (stretch goal): Dataset 4 direction-invariance ablation.

Buckets Dataset 4 trials by target direction and computes per-direction
regression error. Produces a bar chart: error vs. direction.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from src.config import CFG


def bucket_by_direction(
    angle_true: np.ndarray,
    angle_pred: np.ndarray,
    metadata: List[dict],
    n_bins: int = 8,
) -> Dict[str, Dict]:
    """
    Bucket windows by target direction (angle of the true gaze vector)
    and compute RMSE per bucket.

    Parameters
    ----------
    angle_true : (n, 2) ground truth [H, V] in degrees
    angle_pred : (n, 2) predictions
    metadata   : list of dicts with dataset_source per window
    n_bins     : number of direction buckets (default 8 = every 45°)

    Returns
    -------
    dict mapping direction label → {'rmse_h', 'rmse_v', 'n_windows'}
    """
    gaze_direction_rad = np.arctan2(angle_true[:, 1], angle_true[:, 0])
    gaze_direction_deg = np.degrees(gaze_direction_rad) % 360.0

    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_labels = [
        f"{int(bin_edges[i])}-{int(bin_edges[i+1])}°" for i in range(n_bins)
    ]

    results = {}
    for i, label in enumerate(bin_labels):
        in_bin = (gaze_direction_deg >= bin_edges[i]) & (gaze_direction_deg < bin_edges[i + 1])
        if not np.any(in_bin):
            results[label] = {"rmse_h_deg": None, "rmse_v_deg": None, "n_windows": 0}
            continue
        pred_b = angle_pred[in_bin]
        true_b = angle_true[in_bin]
        results[label] = {
            "rmse_h_deg": float(np.sqrt(np.mean((pred_b[:, 0] - true_b[:, 0]) ** 2))),
            "rmse_v_deg": float(np.sqrt(np.mean((pred_b[:, 1] - true_b[:, 1]) ** 2))),
            "n_windows": int(np.sum(in_bin)),
        }

    return results


def plot_direction_bar_chart(
    direction_results: Dict[str, Dict],
    figures_dir: str = None,
) -> str:
    """Bar chart: RMSE vs. direction bucket."""
    if figures_dir is None:
        figures_dir = CFG.paths.reports_figures
    os.makedirs(figures_dir, exist_ok=True)

    labels = list(direction_results.keys())
    rmse_h = [direction_results[l].get("rmse_h_deg") or 0 for l in labels]
    rmse_v = [direction_results[l].get("rmse_v_deg") or 0 for l in labels]
    counts = [direction_results[l]["n_windows"] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("Dataset 4 — Direction-Invariance Ablation\nRegression Error per Direction Bucket")

    ax1.bar(x - width / 2, rmse_h, width, label="RMSE H", color="steelblue")
    ax1.bar(x + width / 2, rmse_v, width, label="RMSE V", color="coral")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_ylabel("RMSE (degrees)")
    ax1.set_title("Regression Error by Gaze Direction")
    ax1.legend()

    ax2.bar(x, counts, color="gray", alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_ylabel("Window count")
    ax2.set_title("Number of Windows per Direction Bucket")

    plt.tight_layout()
    path = os.path.join(figures_dir, "direction_invariance_bar_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Direction bar chart saved: {path}")
    return path


def run_direction_ablation(
    angle_true: np.ndarray,
    angle_pred: np.ndarray,
    metadata: List[dict],
    results_dir: str = None,
    figures_dir: str = None,
    n_bins: int = 8,
) -> Dict:
    """Full direction ablation: bucket → save JSON → plot bar chart."""
    if results_dir is None:
        results_dir = CFG.paths.results
    os.makedirs(results_dir, exist_ok=True)

    direction_results = bucket_by_direction(angle_true, angle_pred, metadata, n_bins)

    path = os.path.join(results_dir, "direction_invariance_results.json")
    with open(path, "w") as f:
        json.dump(direction_results, f, indent=2)
    print(f"Direction ablation results saved: {path}")

    plot_direction_bar_chart(direction_results, figures_dir)
    return direction_results
