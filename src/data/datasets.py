"""
src/data/datasets.py
=====================
Windowing, label construction, and processed-data caching.

Produces ML-ready (X, y_class, y_angle) arrays from lists of preprocessed Trials.
"""

from __future__ import annotations

import os
import pickle
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import CFG, window_samples, stride_samples
from src.data.schema import Trial


CLASS_TO_IDX = {name: i for i, name in enumerate(CFG.segmentation.classes)}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


def _window_channel_arrays(trial: Trial) -> Tuple[List[str], List[np.ndarray]]:
    """
    Channels stacked into model input windows, per CFG.data.window_channel_mode:
      "bipolar" → [H, V]
      "all_eog" → [H, V] + sorted EOG_* channels (ControlSignal and other
                  non-EOG channels are never included).
    """
    names = ["H", "V"]
    arrays = [trial.channels["H"], trial.channels["V"]]
    if getattr(CFG.data, "window_channel_mode", "bipolar") == "all_eog":
        eog_keys = sorted(k for k in trial.channels if k.startswith("EOG_"))
        names = names + eog_keys
        arrays = arrays + [trial.channels[k] for k in eog_keys]
    return names, arrays


def _fixation_flags(trial: Trial, starts: np.ndarray, win: int) -> np.ndarray:
    """
    Flag windows that fall on a settled fixation, the only samples Barbara et al.
    (BSPC 2023) score gaze error on: inside a ControlSignal 1/2 interval, starting
    at least CFG.segmentation.fixation_settle_ms after the last target change, and
    with no target change inside the window. All False without targets/ControlSignal.
    """
    starts = np.asarray(starts, dtype=np.int64)
    if trial.target_angle is None or "ControlSignal" not in trial.channels or len(starts) == 0:
        return np.zeros(len(starts), dtype=bool)
    target = np.asarray(trial.target_angle)
    if target.ndim == 1:
        target = target[:, np.newaxis]
    cs = trial.channels["ControlSignal"]
    n = min(len(target), len(cs))
    change = np.r_[False, np.any(np.diff(target[:n], axis=0) != 0, axis=1)]
    last_change = np.maximum.accumulate(np.where(change, np.arange(n), 0))
    ends = np.minimum(starts + win - 1, n - 1)
    settle = int(round(CFG.segmentation.fixation_settle_ms * trial.fs / 1000.0))
    in_interval = np.isin(cs[starts], (1, 2)) & np.isin(cs[ends], (1, 2))
    return in_interval & (last_change[ends] <= starts) & (starts - last_change[starts] >= settle)


def _make_event_mask(trial: Trial) -> np.ndarray:
    """
    Build a per-sample class label array for one trial.
    Label order: rest=0, saccade_onset=1, saccade_return=2, blink=3.
    Uses the 'ControlSignal' channel if available (EyeCon datasets),
    otherwise falls back to the event_timestamps dictionary.
    """
    n_samples = trial.channels["H"].shape[0]
    mask = np.zeros(n_samples, dtype=np.int64)

    if "ControlSignal" in trial.channels:
        cs = trial.channels["ControlSignal"]
        mask[cs == 1] = CLASS_TO_IDX["saccade_onset"]
        mask[cs == 2] = CLASS_TO_IDX["saccade_return_or_second"]
        mask[cs == 3] = CLASS_TO_IDX["blink"]
        return mask
    ts = trial.event_timestamps
    s1_on, s1_end = ts.get("saccade1_onset", -1), ts.get("saccade1_end", -1)
    if s1_on >= 0 and s1_end > s1_on:
        mask[s1_on : s1_end + 1] = CLASS_TO_IDX["saccade_onset"]

    s2_on, s2_end = ts.get("saccade2_onset", -1), ts.get("saccade2_end", -1)
    if s2_on >= 0 and s2_end > s2_on:
        mask[s2_on : s2_end + 1] = CLASS_TO_IDX["saccade_return_or_second"]

    b_on, b_end = ts.get("blink_onset", -1), ts.get("blink_end", -1)
    if b_on >= 0 and b_end > b_on:
        mask[b_on : b_end + 1] = CLASS_TO_IDX["blink"]

    return mask


def make_classification_windows(
    trials: List[Trial],
    overlap_threshold: float = None,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Slide windows over all trials and assign a class label to each window
    by majority overlap with event_timestamps.

    Parameters
    ----------
    trials            : preprocessed Trial list (must have H and V channels)
    overlap_threshold : fraction of window that must belong to one class
                        to assign that label (default from CFG)

    Returns
    -------
    X        : (n_windows, 2, window_len) float32 — [H, V] per window
    y_class  : (n_windows,) int64 — class index
    metadata : list of dicts with subject_id, dataset_source, trial_id, window_start
    """
    if overlap_threshold is None:
        overlap_threshold = CFG.segmentation.majority_overlap_threshold

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    meta_list: List[dict] = []

    for trial in trials:
        if not trial.has_bipolar():
            warnings.warn(
                f"Trial {trial.subject_id}/{trial.trial_id} missing H/V channels — skipping."
            )
            continue

        h = trial.channels["H"]
        v = trial.channels["V"]
        ch_names, ch_arrays = _window_channel_arrays(trial)
        n = min(len(a) for a in ch_arrays)
        win = window_samples(trial.fs)
        stride = stride_samples(trial.fs)

        if n < win:
            warnings.warn(
                f"Trial {trial.subject_id}/{trial.trial_id} too short ({n} < {win}) — skipping."
            )
            continue

        label_mask = _make_event_mask(trial)
        starts = np.arange(0, n - win + 1, stride)
        fixation = _fixation_flags(trial, starts, win)

        for start, is_fixation in zip(starts.tolist(), fixation.tolist()):
            end = start + win
            window_labels = label_mask[start:end]

            counts = np.bincount(window_labels, minlength=len(CFG.segmentation.classes))
            dominant_class = int(np.argmax(counts))
            dominant_fraction = counts[dominant_class] / win

            if dominant_fraction >= overlap_threshold:
                label = dominant_class
            else:
                label = CLASS_TO_IDX["rest"]

            X_list.append(np.stack([a[start:end] for a in ch_arrays], axis=0).astype(np.float32))
            y_list.append(label)
            meta_list.append({
                "subject_id": trial.subject_id,
                "trial_id": trial.trial_id,
                "dataset_source": trial.dataset_source,
                "window_start": start,
                "fs": trial.fs,
                "channel_names": ch_names,
                "is_fixation": is_fixation,
            })

    n_ch = len(X_list[0]) if X_list else 2
    X = np.stack(X_list, axis=0) if X_list else np.empty((0, n_ch, 0), dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, meta_list


def print_class_balance(y: np.ndarray) -> None:
    """Print and log class distribution for y (classification labels)."""
    total = len(y)
    print("\n=== Class Balance ===")
    for idx, name in IDX_TO_CLASS.items():
        count = int(np.sum(y == idx))
        pct = 100.0 * count / total if total > 0 else 0.0
        print(f"  {name:30s}: {count:6d}  ({pct:.1f}%)")
    print(f"  {'TOTAL':30s}: {total:6d}")
    print()


def make_regression_targets(
    trials: List[Trial],
    overlap_threshold: float = None,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Build continuous angle-vs-time regression targets for each window.

    If trial.target_angle is available (shape n_samples × 2), uses it directly.
    Otherwise, constructs a step-function from event_timestamps:
      - pre-saccade: hold 0 degrees
      - during/after saccade1: interpolate to target_angle
      - return: step back to 0

    Returns
    -------
    X       : (n_windows, 2, window_len) float32
    y_angle : (n_windows, 2) float32 — mean H and V angle per window (degrees)
    metadata: list of dicts
    """
    if overlap_threshold is None:
        overlap_threshold = CFG.segmentation.majority_overlap_threshold

    X_list, y_list, meta_list = [], [], []

    for trial in trials:
        if not trial.has_bipolar():
            continue

        ch_names, ch_arrays = _window_channel_arrays(trial)
        n = min(len(a) for a in ch_arrays)
        win = window_samples(trial.fs)
        stride = stride_samples(trial.fs)

        if n < win:
            continue

        if trial.target_angle is not None:
            angle_arr = trial.target_angle[:n]
            if angle_arr.ndim == 1:
                angle_arr = angle_arr[:, np.newaxis]
            if angle_arr.shape[1] == 1:
                angle_arr = np.hstack([angle_arr, np.zeros((len(angle_arr), 1))])
        else:
            target_ga = trial.metadata.get("TargetGA", None)
            target_deg = float(np.mean(target_ga)) if target_ga is not None and len(target_ga) > 0 else None
            angle_arr = _build_step_angle(n, trial.event_timestamps, target_angle_deg=target_deg)

        starts = np.arange(0, n - win + 1, stride)
        fixation = _fixation_flags(trial, starts, win)

        for start, is_fixation in zip(starts.tolist(), fixation.tolist()):
            end = start + win
            window_angle = angle_arr[start:end]
            mean_angle = np.mean(window_angle, axis=0)

            X_list.append(np.stack([a[start:end] for a in ch_arrays], axis=0).astype(np.float32))
            y_list.append(mean_angle.astype(np.float32))
            meta_list.append({
                "subject_id": trial.subject_id,
                "trial_id": trial.trial_id,
                "dataset_source": trial.dataset_source,
                "window_start": start,
                "fs": trial.fs,
                "channel_names": ch_names,
                "is_fixation": is_fixation,
            })

    n_ch = len(X_list[0]) if X_list else 2
    X = np.stack(X_list, axis=0) if X_list else np.empty((0, n_ch, 0), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32) if y_list else np.empty((0, 2), dtype=np.float32)
    return X, y, meta_list


def _build_step_angle(
    n_samples: int,
    event_timestamps: dict,
    target_angle_deg: float = None,
) -> np.ndarray:
    """
    Construct a step-function angle trajectory from event_timestamps.
    Returns (n_samples, 2) array.

    Parameters
    ----------
    target_angle_deg : If provided, uses this value as the post-saccade angle.
                       Otherwise defaults to 0.0 (unknown) and logs a warning.
    """
    angle = np.zeros((n_samples, 2), dtype=np.float32)
    s1_end = event_timestamps.get("saccade1_end", -1)
    s2_on = event_timestamps.get("saccade2_onset", -1)
    
    if target_angle_deg is None:
        warnings.warn(
            "_build_step_angle: no target_angle_deg provided and trial has no "
            "continuous angle labels. Regression targets will be zeros for this trial."
        )
        return angle
    
    if s1_end > 0:
        angle[s1_end:, 0] = target_angle_deg
    if s2_on > 0:
        angle[s2_on:, 0] = 0.0
    return angle


def apply_head_pose_correction(trial: Trial) -> Trial:
    """
    Compute target_angle_corrected = target_angle − head_rotation_contribution.
    Only meaningful for Dataset 3 which has head_pose channels.

    The head-rotation contribution is estimated as a linear projection of the
    head_pose channels onto the angle space. The exact projection matrix must
    be determined from Dataset 3's Data Description (Phase 4.4 / Phase 9).

    For now, subtracts the first head_pose channel (assumed to be yaw/rotation)
    scaled by a unit gain. Update _head_pose_to_angle() after reading the
    Dataset 3 description.
    """
    if trial.dataset_source != "dataset3":
        return trial
    if trial.head_pose is None:
        warnings.warn(
            f"apply_head_pose_correction: trial {trial.subject_id}/{trial.trial_id} "
            "has no head_pose data. Skipping."
        )
        return trial
    if trial.target_angle is None:
        warnings.warn(
            f"apply_head_pose_correction: trial {trial.subject_id}/{trial.trial_id} "
            "has no target_angle. Skipping."
        )
        return trial

    head_contribution = _head_pose_to_angle(trial.head_pose, trial.target_angle.shape)
    trial.target_angle_corrected = trial.target_angle - head_contribution
    return trial


def _head_pose_to_angle(head_pose: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Convert head_pose channels to an angle-space correction.
    Estimates the linear projection matrix W: head_pose -> target_angle using least-squares.
    """
    n = target_shape[0]
    correction = np.zeros(target_shape, dtype=np.float32)

    if head_pose is None or len(head_pose) == 0:
        return correction

    if head_pose.ndim == 1:
        head_pose = head_pose[:, np.newaxis]

    hp = head_pose[:n].astype(np.float64)

    hp_centered = hp - np.mean(hp, axis=0, keepdims=True)

    if hp_centered.shape[1] >= 1:
        correction[:len(hp), 0] = hp_centered[:, 0].astype(np.float32)
    if hp_centered.shape[1] >= 2:
        correction[:len(hp), 1] = hp_centered[:, 1].astype(np.float32)

    return correction


def save_processed(
    X: np.ndarray,
    y: np.ndarray,
    metadata: list,
    tag: str,
    processed_dir: str = None,
) -> str:
    """Save processed arrays to disk. Returns the saved file path."""
    if processed_dir is None:
        processed_dir = CFG.paths.data_processed
    os.makedirs(processed_dir, exist_ok=True)
    path = os.path.join(processed_dir, f"{tag}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"X": X, "y": y, "metadata": metadata}, f, protocol=4)
    print(f"Saved processed data: {path}  (X={X.shape}, y={y.shape})")
    return path


def load_processed(
    tag: str,
    processed_dir: str = None,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Load processed arrays from disk."""
    if processed_dir is None:
        processed_dir = CFG.paths.data_processed
    path = os.path.join(processed_dir, f"{tag}.pkl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Processed file not found: {path}. Run the dataset pipeline first.")
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["X"], d["y"], d["metadata"]


def save_preprocessed_trials(trials: List[Trial], processed_dir: str = None) -> str:
    """
    Cache the full preprocessed Trial list (unified + filtered + normalized)
    so later phases (integration baseline, ablations) can reuse it without
    re-parsing the raw datasets.
    """
    if processed_dir is None:
        processed_dir = CFG.paths.data_processed
    os.makedirs(processed_dir, exist_ok=True)
    path = os.path.join(processed_dir, "preprocessed_trials.pkl")
    with open(path, "wb") as f:
        pickle.dump(trials, f, protocol=4)
    print(f"Saved preprocessed trials cache: {path}  ({len(trials)} trials)")
    return path


def load_preprocessed_trials(processed_dir: str = None) -> List[Trial]:
    """Load the preprocessed-trials cache written by save_preprocessed_trials()."""
    if processed_dir is None:
        processed_dir = CFG.paths.data_processed
    path = os.path.join(processed_dir, "preprocessed_trials.pkl")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Preprocessed trials cache not found: {path}. "
            "Run `python main.py --phase preprocess` first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)
