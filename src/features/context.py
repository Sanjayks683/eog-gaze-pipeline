"""
src/features/context.py
========================
Per-window context for gaze regression that a single 300 ms window cannot see.

A window's DC level is gaze plus the error of the drift-baseline estimate, and on
Dataset 2 that error depends on recent gaze: a past-only baseline has partly
absorbed the latest fixations, and the mean gaze inside any baseline window
wanders with the random targets. Without context, regressors hedge by pulling
predictions toward the centre of the screen. Three summaries let them correct it:

  * baseline disagreement — window mean of (x − other baseline) − (x − main
    baseline) for each CFG.preprocessing.context_baselines [method, window_sec];
  * lagged level — window mean of the main preprocessed signal
    CFG.preprocessing.context_lags_sec seconds earlier;
  * rolling range — for each CFG.preprocessing.context_range_windows_sec and
    [low, high] pair in context_range_quantiles, the window level minus the
    rolling mid-range (low + high quantile) / 2, the range (high − low), and their
    ratio. Cue positions are spread roughly uniformly over a bounded screen, and
    for such data the mid-range locates the centre far more precisely than the
    mean (error ~1/N rather than ~1/sqrt(N)); the range is a label-free estimate
    of the subject's EOG gain.

All use the main baseline's causal setting, so with a causal baseline no
feature uses samples after the window's end (apart from what the whole pipeline
already shares: the zero-phase 30 Hz low-pass and the per-subject z-score scale).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import CFG, window_samples
from src.data.schema import Trial
from src.preprocessing.filtering import remove_baseline_drift


def _window_means(sig: np.ndarray, starts: np.ndarray, win: int) -> np.ndarray:
    csum = np.r_[0.0, np.cumsum(sig, dtype=np.float64)]
    return (csum[starts + win] - csum[starts]) / win


def _main_baseline_causal() -> bool:
    pp = CFG.preprocessing
    return pp.median_baseline_causal if pp.drift_removal_method == "moving_median" else pp.baseline_causal


def _range_features(sig: np.ndarray, starts: np.ndarray, win: int, fs: float,
                    window_sec: float, low: float, high: float, causal: bool) -> List[np.ndarray]:
    """Window level relative to the rolling mid-range, the rolling range, and their ratio."""
    step = max(1, int(fs // 8))
    n_blocks = len(sig) // step
    blocks = pd.Series(sig[: n_blocks * step].reshape(n_blocks, step).mean(axis=1))
    roll = blocks.rolling(max(2, int(round(window_sec * fs / step))),
                          min_periods=int(round(10.0 * fs / step)), center=not causal)
    q_low, q_high = roll.quantile(low).to_numpy(), roll.quantile(high).to_numpy()
    # causal: the last block that ends inside the window; centred: the window's middle block
    block = (starts + win) // step - 1 if causal else (starts + win // 2) // step
    block = np.clip(block, 0, n_blocks - 1)
    level = _window_means(sig, starts, win)
    mid, rng = (q_low[block] + q_high[block]) / 2, q_high[block] - q_low[block]
    # The first 10 s have no range estimate yet; 0 keeps kernel models (SVR) usable.
    return [np.nan_to_num(f) for f in (level - mid, rng, (level - mid) / (rng + 1e-6))]


def context_feature_names(channel_names: List[str]) -> List[str]:
    pp = CFG.preprocessing
    names = [f"{ch}_base_{method}{float(sec):g}s_minus_main"
             for method, sec in pp.context_baselines for ch in channel_names]
    names += [f"{ch}_level_lag{float(lag):g}s" for lag in pp.context_lags_sec for ch in channel_names]
    names += [f"{ch}_{kind}_q{low:g}-{high:g}_{float(sec):g}s"
              for sec in pp.context_range_windows_sec for low, high in pp.context_range_quantiles
              for ch in channel_names for kind in ("level_minus_midrange", "range", "level_over_range")]
    return names


def make_context_features(
    trials: List[Trial],
    raw_signals: List[Dict[str, np.ndarray]],
    metadata: List[dict],
) -> np.ndarray:
    """
    Context features aligned with `metadata` (one row per window).

    Parameters
    ----------
    trials      : preprocessed trials; each needs metadata["zscore_std"] (set by
                  preprocess_trials) to put baseline differences in z-score units
    raw_signals : per trial, the channels before preprocessing (same order as trials)
    metadata    : window metadata from make_regression_targets

    Returns
    -------
    (n_windows, n_features) float32; (n_windows, 0) when no context is configured.
    """
    pp = CFG.preprocessing
    causal = _main_baseline_causal()
    by_trial: Dict[tuple, List[int]] = {}
    for i, m in enumerate(metadata):
        by_trial.setdefault((m["subject_id"], m["trial_id"]), []).append(i)

    out = None
    for trial, raw in zip(trials, raw_signals):
        idx = np.array(by_trial.get((trial.subject_id, trial.trial_id), []), dtype=np.int64)
        if len(idx) == 0:
            continue
        channel_names = metadata[idx[0]]["channel_names"]
        starts = np.array([metadata[i]["window_start"] for i in idx], dtype=np.int64)
        win = window_samples(trial.fs)
        feats = []
        main = {}
        for method, window_sec in pp.context_baselines:
            for ch in channel_names:
                sig = np.asarray(raw[ch], dtype=np.float64)
                if ch not in main:
                    main[ch] = remove_baseline_drift(sig, trial.fs)
                other = remove_baseline_drift(sig, trial.fs, method, float(window_sec), causal)
                std = trial.metadata["zscore_std"][ch] or 1.0
                feats.append(_window_means(other - main[ch], starts, win) / std)
        for lag in pp.context_lags_sec:
            lagged = np.maximum(starts - int(round(float(lag) * trial.fs)), 0)
            feats += [_window_means(trial.channels[ch], lagged, win) for ch in channel_names]
        for window_sec in pp.context_range_windows_sec:
            for low, high in pp.context_range_quantiles:
                for ch in channel_names:
                    feats += _range_features(trial.channels[ch], starts, win, trial.fs,
                                             float(window_sec), float(low), float(high), causal)
        if out is None:
            out = np.zeros((len(metadata), len(feats)), dtype=np.float32)
        if feats:
            out[idx] = np.column_stack(feats)
    return out if out is not None else np.zeros((len(metadata), 0), dtype=np.float32)
