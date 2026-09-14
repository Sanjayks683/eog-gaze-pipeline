"""
src/preprocessing/normalize.py
================================
Per-subject z-score normalization.

CRITICAL ORDER: This must be called AFTER drift removal and noise filtering,
and AFTER blink detection. Normalizing before drift removal lets drift dominate
the computed mean/std, corrupting all downstream features.

Correct pipeline order:
  1. remove_baseline_drift()
  2. remove_noise()
  3. get_blink_intervals()
  4. zscore_per_subject()   ← this module
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data.schema import Trial
from src.preprocessing.blink import get_rest_segment


def zscore_trial(
    sig: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """
    Z-score a signal given pre-computed mean and std.
    If std == 0, returns zero array with a warning.
    """
    if std == 0.0:
        warnings.warn("zscore_trial: std is 0 — returning zeros.")
        return np.zeros_like(sig)
    return (sig - mean) / std


def compute_rest_stats(
    trials: List[Trial],
    channel: str = "H",
) -> Dict[str, Tuple[float, float]]:
    """
    Compute per-subject mean and std from each subject's own rest segments.

    Parameters
    ----------
    trials  : list of Trial objects (must have channels['H'] and channels['V'])
    channel : which channel to compute stats for ('H' or 'V')

    Returns
    -------
    Dict mapping subject_id → (mean, std)
    """
    subject_segments: Dict[str, List[np.ndarray]] = {}

    for trial in trials:
        if channel not in trial.channels:
            continue
        sig = trial.channels[channel]
        rest = get_rest_segment(sig, trial.event_timestamps, trial.fs)
        if trial.subject_id not in subject_segments:
            subject_segments[trial.subject_id] = []
        subject_segments[trial.subject_id].append(rest)

    stats: Dict[str, Tuple[float, float]] = {}
    for subj, segs in subject_segments.items():
        combined = np.concatenate(segs)
        stats[subj] = (float(np.mean(combined)), float(np.std(combined)))

    return stats


def zscore_per_subject(
    trials: List[Trial],
    channels_to_normalize: Optional[List[str]] = None,
    inplace: bool = True,
) -> Tuple[List[Trial], Dict[str, Dict[str, Tuple[float, float]]]]:
    """
    Apply per-subject z-score normalization to all trials.

    Mean and std are computed from each subject's rest segments only
    (pre-saccade periods), not the full trial, to avoid drift or blink
    contamination.

    Parameters
    ----------
    trials                 : list of Trial objects (post-filtering)
    channels_to_normalize  : which channels to normalize (default: ['H', 'V'])
    inplace                : if True, modify channels in-place

    Returns
    -------
    (normalized_trials, stats_dict)
    stats_dict: channel → subject_id → (mean, std) — save these for
                applying the same normalization to test sets.
    """
    if channels_to_normalize is None:
        channels_to_normalize = ["H", "V"]

    stats_dict: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for ch in channels_to_normalize:
        stats_dict[ch] = compute_rest_stats(trials, channel=ch)

    normalized: List[Trial] = []
    for trial in trials:
        t = trial if inplace else _copy_trial(trial)
        for ch in channels_to_normalize:
            if ch not in t.channels:
                continue
            if trial.subject_id not in stats_dict[ch]:
                warnings.warn(
                    f"No normalization stats for subject {trial.subject_id}, channel {ch}. "
                    "Skipping normalization for this trial."
                )
                continue
            mean, std = stats_dict[ch][trial.subject_id]
            t.channels[ch] = zscore_trial(t.channels[ch], mean, std)
        normalized.append(t)

    return normalized, stats_dict


def apply_saved_stats(
    trials: List[Trial],
    stats_dict: Dict[str, Dict[str, Tuple[float, float]]],
    channels_to_normalize: Optional[List[str]] = None,
    inplace: bool = True,
) -> List[Trial]:
    """
    Apply pre-computed normalization stats (from training set) to new trials.
    Use this for test/validation sets to prevent leakage.

    Parameters
    ----------
    trials      : list of Trial objects
    stats_dict  : output of zscore_per_subject() on the training set
    channels_to_normalize : default ['H', 'V']
    inplace     : if True, modify channels in-place

    Returns
    -------
    List of normalized Trial objects.
    """
    if channels_to_normalize is None:
        channels_to_normalize = ["H", "V"]

    result: List[Trial] = []
    for trial in trials:
        t = trial if inplace else _copy_trial(trial)
        for ch in channels_to_normalize:
            if ch not in t.channels:
                continue
            if ch not in stats_dict or trial.subject_id not in stats_dict[ch]:
                warnings.warn(
                    f"No saved stats for subject={trial.subject_id}, channel={ch}. "
                    "Skipping normalization (test subject not in training set)."
                )
                continue
            mean, std = stats_dict[ch][trial.subject_id]
            t.channels[ch] = zscore_trial(t.channels[ch], mean, std)
        result.append(t)
    return result


def _signal_channel_names(trials: List[Trial]) -> List[str]:
    """
    All EOG signal channels present across trials, in canonical order:
    [H, V] first, then EOG_* sorted. ControlSignal and other non-EOG channels
    are excluded (ControlSignal is a label stream, not an input signal).
    """
    names = {ch for t in trials for ch in t.channels
             if ch in ("H", "V") or ch.startswith("EOG_")}
    ordered = [c for c in ("H", "V") if c in names]
    ordered += sorted(n for n in names if n not in ("H", "V"))
    return ordered


def preprocess_trials(trials: List[Trial]) -> List[Trial]:
    """
    Apply the full preprocessing pipeline to a list of trials in the correct order:
      1. Baseline drift removal (high-pass or polynomial detrend)
      2. Noise filtering (low-pass)
      3. Optional notch at 50 Hz (auto-detected)
      4. Per-subject z-score normalization

    Applies to every EOG signal channel (H, V and any EOG_* monopolar
    channels). Returns preprocessed Trial list (channels modified in-place).
    """
    from src.preprocessing.filtering import filter_trial_channels

    signal_channels = _signal_channel_names(trials)
    print(f"Preprocessing channels: {signal_channels}")

    for trial in trials:
        filtered = filter_trial_channels(
            {k: v for k, v in trial.channels.items() if k in signal_channels},
            trial.fs,
            apply_drift=True,
            apply_noise=True,
            auto_notch=True,
        )
        trial.channels.update(filtered)

    trials, stats = zscore_per_subject(trials, channels_to_normalize=signal_channels, inplace=True)
    for trial in trials:
        trial.metadata["zscore_std"] = {
            ch: stats[ch][trial.subject_id][1]
            for ch in signal_channels if trial.subject_id in stats[ch]
        }
    return trials


def _copy_trial(trial: Trial) -> Trial:
    """Shallow copy of trial with deep-copied channel arrays."""
    import copy
    t = copy.copy(trial)
    t.channels = {k: v.copy() for k, v in trial.channels.items()}
    return t
