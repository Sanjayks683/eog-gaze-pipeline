"""
Zero-Leakage Few-Shot Subject Calibration Protocol for EOG Gaze Estimation.

In commercial eye-trackers (Tobii, EyeLink) and academic EOG literature
(e.g., Barbara et al., 2023, BSPC vol. 86), eye-tracking systems perform a brief
user calibration routine to account for inter-individual corneal-retinal dipole
voltage and electrode-skin impedance variations.

This module evaluates few-shot calibration under strict cross-subject GroupKFold:
  1. For each unseen test subject, the initial portion of their session
     (e.g. prompt_sec = 5.0 s) is allocated strictly as the calibration set.
  2. A 2-parameter affine transformation (gain and offset) is fit on the
     calibration set only.
  3. Performance is evaluated EXCLUSIVELY on the remaining unseen test windows.
  4. Both uncalibrated (zero-shot) and calibrated errors are computed on the
     exact same held-out evaluation windows for fair, leak-free comparison.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.config import CFG


def fit_affine_calibration(
    y_pred_cal: np.ndarray,
    y_true_cal: np.ndarray,
    alpha: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit affine parameters: y_true ≈ gain * y_pred + offset for H and V.

    Uses regularized least squares (Ridge) to guarantee numerical stability
    even when calibration windows have small dynamic range.

    Parameters
    ----------
    y_pred_cal : (N, 2) predicted angles on calibration split
    y_true_cal : (N, 2) true angles on calibration split
    alpha      : L2 regularization strength

    Returns
    -------
    gains   : (2,) array [gain_h, gain_v]
    offsets : (2,) array [offset_h, offset_v]
    """
    gains = np.ones(2, dtype=np.float32)
    offsets = np.zeros(2, dtype=np.float32)

    for dim in range(2):
        p = y_pred_cal[:, dim]
        t = y_true_cal[:, dim]

        var_t = np.var(t)
        var_p = np.var(p)

        if var_t < 5.0 or var_p < 1e-4:
            gains[dim] = 1.0
            offsets[dim] = float(np.mean(t) - np.mean(p))
            continue

        cov_pt = np.mean((p - p.mean()) * (t - t.mean()))
        gain = cov_pt / (var_p + alpha)
        
        gain = np.clip(gain, 0.2, 5.0)
        
        offset = np.mean(t) - gain * np.mean(p)
        
        gains[dim] = float(gain)
        offsets[dim] = float(offset)

    return gains, offsets


def apply_affine_calibration(
    y_pred: np.ndarray,
    gains: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Apply gain and offset calibration to predicted angles."""
    return y_pred * gains + offsets


def evaluate_subject_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: List[dict],
    prompt_sec: float = 5.0,
    min_cal_windows: int = 15,
) -> Dict:
    """
    Perform zero-leakage few-shot calibration per subject.

    Parameters
    ----------
    y_true          : (N, 2) ground truth angles
    y_pred          : (N, 2) model predictions
    metadata        : list of dicts with 'subject_id' and 'window_start' or 'fs'
    prompt_sec      : duration of initial calibration prompt in seconds
    min_cal_windows : minimum number of windows to use for calibration

    Returns
    -------
    Dictionary of overall and per-subject uncalibrated vs. calibrated metrics.
    """
    subjects = np.array([m.get("subject_id", "unknown") for m in metadata])
    unique_subs = np.unique(subjects)

    all_uncal_err_h, all_uncal_err_v = [], []
    all_cal_err_h, all_cal_err_v = [], []
    all_trues_h, all_trues_v = [], []
    all_preds_uncal_h, all_preds_cal_h = [], []

    sub_results = {}

    for sub in unique_subs:
        sub_indices = np.where(subjects == sub)[0]
        if len(sub_indices) < min_cal_windows + 5:
            continue

        starts = np.array([metadata[i].get("window_start", i) for i in sub_indices])
        order = np.argsort(starts)
        sorted_indices = sub_indices[order]

        fs = float(metadata[sorted_indices[0]].get("fs", 250.0))
        if len(sorted_indices) > 1 and starts[order[1]] > starts[order[0]]:
            stride_samples = starts[order[1]] - starts[order[0]]
            window_stride_sec = stride_samples / fs
        else:
            window_stride_sec = 0.150

        n_cal = max(min_cal_windows, int(np.ceil(prompt_sec / max(window_stride_sec, 0.05))))
        n_cal = min(n_cal, len(sorted_indices) // 4)

        cal_idx = sorted_indices[:n_cal]
        eval_idx = sorted_indices[n_cal:]

        if len(eval_idx) == 0:
            continue

        gains, offsets = fit_affine_calibration(
            y_pred_cal=y_pred[cal_idx],
            y_true_cal=y_true[cal_idx],
        )

        eval_true = y_true[eval_idx]
        eval_uncal = y_pred[eval_idx]
        eval_cal = apply_affine_calibration(eval_uncal, gains, offsets)

        err_uncal_h = eval_true[:, 0] - eval_uncal[:, 0]
        err_uncal_v = eval_true[:, 1] - eval_uncal[:, 1]
        err_cal_h = eval_true[:, 0] - eval_cal[:, 0]
        err_cal_v = eval_true[:, 1] - eval_cal[:, 1]

        all_uncal_err_h.extend(err_uncal_h)
        all_uncal_err_v.extend(err_uncal_v)
        all_cal_err_h.extend(err_cal_h)
        all_cal_err_v.extend(err_cal_v)

        all_trues_h.extend(eval_true[:, 0])
        all_trues_v.extend(eval_true[:, 1])
        all_preds_uncal_h.extend(eval_uncal[:, 0])
        all_preds_cal_h.extend(eval_cal[:, 0])

        sub_results[str(sub)] = {
            "n_calibration_windows": int(n_cal),
            "n_evaluation_windows": int(len(eval_idx)),
            "gains": [float(gains[0]), float(gains[1])],
            "offsets": [float(offsets[0]), float(offsets[1])],
            "uncal_rmse_h": float(np.sqrt(np.mean(err_uncal_h**2))),
            "uncal_rmse_v": float(np.sqrt(np.mean(err_uncal_v**2))),
            "cal_rmse_h": float(np.sqrt(np.mean(err_cal_h**2))),
            "cal_rmse_v": float(np.sqrt(np.mean(err_cal_v**2))),
        }

    uncal_err_h = np.array(all_uncal_err_h)
    uncal_err_v = np.array(all_uncal_err_v)
    cal_err_h = np.array(all_cal_err_h)
    cal_err_v = np.array(all_cal_err_v)
    trues_h = np.array(all_trues_h)

    var_true_h = np.var(trues_h) if len(trues_h) > 0 else 1.0
    r2_uncal_h = 1.0 - (np.mean(uncal_err_h**2) / (var_true_h + 1e-8))
    r2_cal_h = 1.0 - (np.mean(cal_err_h**2) / (var_true_h + 1e-8))

    overall = {
        "prompt_duration_sec": float(prompt_sec),
        "n_evaluated_windows": int(len(uncal_err_h)),
        "uncalibrated_rmse_h_deg": float(np.sqrt(np.mean(uncal_err_h**2))),
        "uncalibrated_rmse_v_deg": float(np.sqrt(np.mean(uncal_err_v**2))),
        "uncalibrated_mae_h_deg": float(np.mean(np.abs(uncal_err_h))),
        "uncalibrated_mae_v_deg": float(np.mean(np.abs(uncal_err_v))),
        "uncalibrated_r2_h": float(r2_uncal_h),
        "calibrated_rmse_h_deg": float(np.sqrt(np.mean(cal_err_h**2))),
        "calibrated_rmse_v_deg": float(np.sqrt(np.mean(cal_err_v**2))),
        "calibrated_mae_h_deg": float(np.mean(np.abs(cal_err_h))),
        "calibrated_mae_v_deg": float(np.mean(np.abs(cal_err_v))),
        "calibrated_r2_h": float(r2_cal_h),
        "per_subject": sub_results,
    }
    return overall


def run_calibration_pipeline(
    model_type: str = "conv_bilstm",
    results_dir: Optional[str] = None,
) -> Dict:
    """
    Load test angle predictions and run the zero-leakage calibration evaluation.
    Saves calibration_results.json in reports/.
    """
    if results_dir is None:
        results_dir = CFG.paths.results

    preds_dir = os.path.join(CFG.paths.data_processed, "angle_preds")
    pred_path = os.path.join(preds_dir, f"angle_pred_{model_type}.npy")
    true_path = os.path.join(preds_dir, f"angle_true_{model_type}.npy")
    meta_path = os.path.join(preds_dir, f"test_meta_{model_type}.pkl")

    if not (os.path.isfile(pred_path) and os.path.isfile(true_path) and os.path.isfile(meta_path)):
        from scripts.run_ablations import rebuild_angle_preds
        print(f"Rebuilding angle predictions for {model_type}...")
        rebuild_angle_preds(model_type=model_type)

    import pickle
    y_pred = np.load(pred_path)
    y_true = np.load(true_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    prompt_sec = getattr(CFG.model, "calibration_prompt_sec", 5.0)
    results = evaluate_subject_calibration(
        y_true=y_true,
        y_pred=y_pred,
        metadata=metadata,
        prompt_sec=prompt_sec,
    )

    out_path = os.path.join(results_dir, "calibration_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFew-Shot Calibration Results ({prompt_sec}s prompt) saved to: {out_path}")
    print(f"  Uncalibrated (Zero-Shot) RMSE: H = {results['uncalibrated_rmse_h_deg']:.3f}°, V = {results['uncalibrated_rmse_v_deg']:.3f}°")
    print(f"  Calibrated (Zero-Leakage) RMSE: H = {results['calibrated_rmse_h_deg']:.3f}°, V = {results['calibrated_rmse_v_deg']:.3f}°")
    print(f"  Calibrated (Zero-Leakage) MAE:  H = {results['calibrated_mae_h_deg']:.3f}°, V = {results['calibrated_mae_v_deg']:.3f}°")
    return results


if __name__ == "__main__":
    run_calibration_pipeline()
