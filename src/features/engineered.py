"""
src/features/engineered.py
===========================
Hand-crafted feature extraction for classical ML baselines.
All features computed on a single (2, window_len) EOG window.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal
from typing import List


def extract_features(window: np.ndarray, fs: float, channel_names: List[str] = None) -> np.ndarray:
    """
    Extract a 1-D feature vector from a single EOG window.

    Parameters
    ----------
    window : (C, window_len) float array — first two rows are [H, V];
             extra rows are monopolar EOG channels (window_channel_mode="all_eog")
    fs     : sampling rate in Hz
    channel_names : optional names for the C rows (feature naming only)

    Returns
    -------
    1-D numpy array of features (see get_feature_names() for order)
    """
    assert window.ndim == 2, f"Expected (C, window_len) input, got {window.shape}"
    n_ch = window.shape[0]

    features = []
    for ch_idx in range(n_ch):
        ch = window[ch_idx].astype(np.float64)
        features.extend(_channel_features(ch, fs))

    h, v = window[0].astype(np.float64), window[1].astype(np.float64)
    hv_corr = float(np.corrcoef(h, v)[0, 1])
    if not np.isfinite(hv_corr):
        hv_corr = 0.0
    features.append(hv_corr)
    features.append(float(np.sqrt(np.mean(window.astype(np.float64) ** 2))))

    return np.array(features, dtype=np.float32)


def _channel_features(ch: np.ndarray, fs: float) -> List[float]:
    """Features for a single channel array."""
    n = len(ch)
    feats = []

    feats.append(float(np.mean(ch)))
    feats.append(float(np.std(ch)))
    feats.append(float(np.max(np.abs(ch))))
    feats.append(float(np.max(ch)))
    feats.append(float(np.min(ch)))
    feats.append(float(np.max(ch) - np.min(ch)))

    t = np.arange(n)
    slope = np.polyfit(t, ch, 1)[0]
    feats.append(float(slope))

    if n > 1:
        diff = np.diff(ch) * fs
        feats.append(float(np.max(np.abs(diff))))
        feats.append(float(np.mean(np.abs(diff))))
    else:
        feats.extend([0.0, 0.0])

    peak = np.max(np.abs(ch))
    thresh = 0.5 * peak if peak > 0 else 1e-9
    above = np.sum(np.abs(ch) > thresh) / fs
    feats.append(float(above))

    demeaned = ch - np.mean(ch)
    zc = int(np.sum(np.diff(np.sign(demeaned)) != 0))
    feats.append(float(zc))

    feats.append(float(np.sqrt(np.mean(ch ** 2))))

    nperseg = min(n, max(16, int(fs)))
    freqs, psd = sp_signal.welch(ch, fs=fs, nperseg=nperseg)

    bands = [(0, 5), (5, 15), (15, 30)]
    for f_lo, f_hi in bands:
        mask = (freqs >= f_lo) & (freqs < f_hi)
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        bp = trapz_fn(psd[mask], freqs[mask]) if np.any(mask) else 0.0
        feats.append(float(bp))

    if len(psd) > 0:
        feats.append(float(freqs[np.argmax(psd)]))
    else:
        feats.append(0.0)

    psd_norm = psd / (np.sum(psd) + 1e-12)
    sp_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    feats.append(float(sp_entropy))

    return feats


_PER_CHANNEL_FEATURES = [
    "mean", "std", "peak_abs", "max", "min", "peak_to_peak",
    "slope", "max_velocity", "mean_velocity",
    "duration_above_half_peak", "zero_crossings", "rms",
    "band_0_5Hz", "band_5_15Hz", "band_15_30Hz",
    "dominant_freq", "spectral_entropy",
]


def get_feature_names(n_channels: int = 2, channel_names: List[str] = None) -> List[str]:
    """Feature names for a window with n_channels rows (default H, V)."""
    names = []
    ch_names = channel_names or ["H", "V"][:n_channels]
    for ch_name in ch_names[:n_channels]:
        for feat in _PER_CHANNEL_FEATURES:
            names.append(f"{ch_name}_{feat}")
    names += ["HV_correlation", "combined_rms"]
    return names


FEATURE_NAMES = get_feature_names()


def extract_features_batch(
    X: np.ndarray,
    fs: float,
    channel_names: List[str] = None,
) -> np.ndarray:
    """
    Extract features from a batch of windows.

    Parameters
    ----------
    X  : (n_windows, C, window_len) float array
    fs : sampling rate in Hz
    channel_names : optional names for the C channel rows

    Returns
    -------
    (n_windows, n_features) float32 array
    """
    return np.array(
        [extract_features(X[i], fs, channel_names=channel_names) for i in range(len(X))],
        dtype=np.float32,
    )
