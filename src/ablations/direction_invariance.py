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
    min_radius_deg: float = 3.0,
) -> Dict[str, Dict]:
    """
    Bucket windows by target direction (angle of the true gaze vector)
    and compute RMSE per bucket.

    A target at the screen centre has no direction: arctan2(0, 0) = 0 would put it in
    the first bucket, and Dataset 4 trials return to the centre, so most windows are
    there. Targets within min_radius_deg of the centre form a separate "centre" entry.

    Parameters
    ----------
    angle_true : (n, 2) ground truth [H, V] in degrees
    angle_pred : (n, 2) predictions
    metadata   : list of dicts with dataset_source per window
    n_bins     : number of direction buckets (default 8 = every 45°)
    min_radius_deg : targets closer to the centre than this are not given a direction

    Returns
    -------
    dict mapping bucket label → {'rmse_h_deg', 'rmse_v_deg', 'rmse_total_deg', 'n_windows'};
    rmse_total_deg is the RMSE of the 2-D gaze error, comparable across directions.
    """
    gaze_direction_deg = np.degrees(np.arctan2(angle_true[:, 1], angle_true[:, 0])) % 360.0
    centre = np.hypot(angle_true[:, 0], angle_true[:, 1]) < min_radius_deg

    def stats(mask: np.ndarray) -> Dict:
        if not np.any(mask):
            return {"rmse_h_deg": None, "rmse_v_deg": None, "rmse_total_deg": None, "n_windows": 0}
        err = angle_pred[mask] - angle_true[mask]
        return {
            "rmse_h_deg": float(np.sqrt(np.mean(err[:, 0] ** 2))),
            "rmse_v_deg": float(np.sqrt(np.mean(err[:, 1] ** 2))),
            "rmse_total_deg": float(np.sqrt(np.mean(np.sum(err ** 2, axis=1)))),
            "n_windows": int(np.sum(mask)),
        }

    bin_edges = np.linspace(0, 360, n_bins + 1)
    results = {f"centre (< {min_radius_deg:g}°)": stats(centre)}
    for i in range(n_bins):
        in_bin = ~centre & (gaze_direction_deg >= bin_edges[i]) & (gaze_direction_deg < bin_edges[i + 1])
        results[f"{int(bin_edges[i])}-{int(bin_edges[i + 1])}°"] = stats(in_bin)
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
    rmse_total = [direction_results[l].get("rmse_total_deg") or 0 for l in labels]
    counts = [direction_results[l]["n_windows"] for l in labels]

    x = np.arange(len(labels))
    width = 0.27

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("Dataset 4 — Direction-Invariance Ablation\nRegression Error per Direction Bucket")

    ax1.bar(x - width, rmse_h, width, label="RMSE H", color="steelblue")
    ax1.bar(x, rmse_v, width, label="RMSE V", color="coral")
    ax1.bar(x + width, rmse_total, width, label="RMSE, 2-D error", color="gray")
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
