"""
src/preprocessing/blink.py
===========================
Blink detection from the vertical EOG channel.

Priority order (use whichever is available):
  1. Dataset-provided blink timestamps (already in trial.event_timestamps)
  2. This detector (threshold on |V| + minimum duration constraint)

The detector is used AFTER drift removal and noise filtering, but BEFORE
normalization — it needs the signal in physical units, not z-scored.
"""

from __future__ import annotations

import warnings
from typing import List, Tuple

import numpy as np

from src.config import CFG


def detect_blinks(
    v_signal: np.ndarray,
    fs: float,
    rest_segment: np.ndarray = None,
    threshold_std_multiplier: float = None,
    min_duration_ms: float = None,
) -> List[Tuple[int, int]]:
    """
    Detect blink intervals in the vertical EOG channel.

    Parameters
    ----------
    v_signal              : 1-D array — vertical EOG (post drift-removal, post noise-filter)
    fs                    : sampling rate in Hz
    rest_segment          : Optional 1-D array to compute std from rest (recommended).
                            If None, uses the full signal std (less accurate).
    threshold_std_multiplier : |V| > N * std triggers blink candidate.
                               Default from CFG (4.5).
    min_duration_ms       : Minimum blink duration in ms to reject spike artefacts.
                            Default from CFG (40 ms).

    Returns
    -------
    List of (onset_sample, offset_sample) tuples (inclusive, 0-indexed).
    """
    if threshold_std_multiplier is None:
        threshold_std_multiplier = CFG.preprocessing.blink_threshold_std
    if min_duration_ms is None:
        min_duration_ms = CFG.preprocessing.blink_min_duration_ms

    min_samples = max(1, int(round(min_duration_ms * fs / 1000.0)))

    if rest_segment is not None and len(rest_segment) > 0:
        ref_std = np.std(rest_segment)
    else:
        ref_std = np.std(v_signal)
        warnings.warn(
            "blink detection: rest_segment not provided — using full signal std. "
            "This may underestimate std if blinks dominate."
        )

    if ref_std == 0:
        warnings.warn("blink detection: reference std is 0 — cannot threshold. Returning empty list.")
        return []

    threshold = threshold_std_multiplier * ref_std
    above = np.abs(v_signal) > threshold

    blinks: List[Tuple[int, int]] = []
    in_blink = False
    onset = 0

    for i, val in enumerate(above):
        if val and not in_blink:
            onset = i
            in_blink = True
        elif not val and in_blink:
            duration = i - onset
            if duration >= min_samples:
                blinks.append((onset, i - 1))
            in_blink = False

    if in_blink:
        duration = len(v_signal) - onset
        if duration >= min_samples:
            blinks.append((onset, len(v_signal) - 1))

    return blinks


def blinks_to_mask(
    n_samples: int,
    blink_intervals: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Convert list of (onset, offset) blink intervals to a boolean mask.

    Parameters
    ----------
    n_samples       : total signal length
    blink_intervals : list of (onset, offset) sample pairs

    Returns
    -------
    Boolean array of shape (n_samples,) — True where blink is detected.
    """
    mask = np.zeros(n_samples, dtype=bool)
    for onset, offset in blink_intervals:
        mask[onset : offset + 1] = True
    return mask


def use_provided_blink_timestamps(trial_event_timestamps: dict) -> List[Tuple[int, int]]:
    """
    Extract blink (onset, offset) from trial.event_timestamps if available.
    Returns list with one tuple if both keys are != -1, else empty list.
    """
    onset = trial_event_timestamps.get("blink_onset", -1)
    end = trial_event_timestamps.get("blink_end", -1)
    if onset != -1 and end != -1 and end > onset:
        return [(onset, end)]
    return []


def get_blink_intervals(
    v_signal: np.ndarray,
    fs: float,
    event_timestamps: dict,
    rest_segment: np.ndarray = None,
) -> List[Tuple[int, int]]:
    """
    Smart blink detection: use provided timestamps if available,
    otherwise fall back to threshold detector.

    Parameters
    ----------
    v_signal          : vertical EOG channel array
    fs                : sampling rate in Hz
    event_timestamps  : trial.event_timestamps dict
    rest_segment      : optional reference segment for std computation

    Returns
    -------
    List of (onset, offset) blink intervals.
    """
    provided = use_provided_blink_timestamps(event_timestamps)
    if provided:
        return provided
    return detect_blinks(v_signal, fs, rest_segment=rest_segment)


def get_rest_segment(
    signal: np.ndarray,
    event_timestamps: dict,
    fs: float,
    pre_saccade_ms: float = 200.0,
) -> np.ndarray:
    """
    Extract a representative rest segment from before the first saccade onset.
    Used to compute the reference std for blink detection and z-score normalization.

    Parameters
    ----------
    signal           : 1-D signal array
    event_timestamps : trial.event_timestamps dict
    fs               : sampling rate in Hz
    pre_saccade_ms   : how many ms before saccade1_onset to use as rest

    Returns
    -------
    1-D array of the rest segment. Falls back to first 10% of signal if
    saccade onset is unknown.
    """
    onset = event_timestamps.get("saccade1_onset", -1)

    if onset > 0:
        n_rest = max(1, int(round(pre_saccade_ms * fs / 1000.0)))
        start = max(0, onset - n_rest)
        return signal[start:onset]
    else:
        end = max(1, len(signal) // 10)
        return signal[:end]
