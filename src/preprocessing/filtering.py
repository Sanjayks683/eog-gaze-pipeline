"""
src/preprocessing/filtering.py
================================
Baseline drift removal and noise filtering for EOG signals.

Order (ALWAYS apply in this sequence):
  1. remove_baseline_drift()
  2. remove_noise()
  3. apply_notch()  — only if 50 Hz spike visible in FFT
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import median_filter
from typing import Literal

from src.config import CFG


def remove_baseline_drift(
    sig: np.ndarray,
    fs: float,
    method: Literal["highpass", "polynomial_detrend", "moving_median",
                    "robust_mean", "robust_line"] = None,
    window_sec: float = None,
    causal: bool = None,
) -> np.ndarray:
    """
    Remove low-frequency baseline drift from an EOG signal.

    Parameters
    ----------
    sig        : 1-D numpy array (n_samples,)
    fs         : sampling rate in Hz
    method     : "highpass", "polynomial_detrend", "moving_median", "robust_mean"
                 or "robust_line" (default from CFG)
    window_sec : baseline window for the moving-window methods (default from CFG)
    causal     : past-only baseline window for those methods (default from CFG)

    Returns
    -------
    Filtered 1-D array same shape as input.

    Notes
    -----
    High-pass cutoff is deliberately conservative (0.1–0.3 Hz) to avoid
    blunting the actual saccade step amplitude — that flat-step distortion
    is the primary failure mode to watch for, not drift visibility.
    """
    if method is None:
        method = CFG.preprocessing.drift_removal_method

    if method == "highpass":
        return _butterworth_highpass(sig, fs)
    elif method == "polynomial_detrend":
        return _polynomial_detrend(sig, fs)
    elif method == "moving_median":
        return _moving_median_detrend(sig, fs, window_sec, causal)
    elif method in ("robust_mean", "robust_line"):
        return _robust_baseline_detrend(sig, fs, method == "robust_line", window_sec, causal)
    else:
        raise ValueError(
            f"Unknown drift removal method: {method!r}. Use 'highpass', "
            "'polynomial_detrend', 'moving_median', 'robust_mean' or 'robust_line'."
        )


def remove_noise(sig: np.ndarray, fs: float) -> np.ndarray:
    """
    Apply a low-pass Butterworth filter to remove high-frequency noise.
    Default cutoff is 30 Hz (from CFG); mains interference handled separately.

    Parameters
    ----------
    sig : 1-D numpy array
    fs  : sampling rate in Hz

    Returns
    -------
    Filtered 1-D array.
    """
    cutoff = CFG.preprocessing.lowpass_cutoff_hz
    order = CFG.preprocessing.filter_order
    nyq = 0.5 * fs
    if cutoff >= nyq:
        import warnings
        warnings.warn(
            f"Low-pass cutoff {cutoff} Hz >= Nyquist {nyq} Hz. "
            "Skipping low-pass filter."
        )
        return sig.copy()
    b, a = sp_signal.butter(order, cutoff / nyq, btype="low", analog=False)
    return sp_signal.filtfilt(b, a, sig)


def apply_notch(sig: np.ndarray, fs: float, notch_hz: float = None) -> np.ndarray:
    """
    Apply a notch filter at the specified frequency (default: 50 Hz mains).
    Should ONLY be applied if the FFT of a representative signal shows a
    visible spike at this frequency — do not apply blindly.

    Parameters
    ----------
    sig      : 1-D numpy array
    fs       : sampling rate in Hz
    notch_hz : frequency to notch out (default from CFG)

    Returns
    -------
    Filtered 1-D array.
    """
    if notch_hz is None:
        notch_hz = CFG.preprocessing.notch_hz
    Q = CFG.preprocessing.notch_quality_factor
    nyq = 0.5 * fs
    if notch_hz >= nyq:
        import warnings
        warnings.warn(f"Notch frequency {notch_hz} Hz >= Nyquist {nyq} Hz. Skipping notch.")
        return sig.copy()
    b, a = sp_signal.iirnotch(notch_hz / nyq, Q)
    return sp_signal.filtfilt(b, a, sig)


def has_mains_spike(
    sig: np.ndarray,
    fs: float,
    threshold_db: float = 10.0,
    target_hz: float = None,
) -> bool:
    """
    Return True if the PSD shows a spike at the mains frequency that is
    threshold_db above the surrounding band. Use this to decide whether to
    call apply_notch().

    Parameters
    ----------
    sig          : 1-D numpy array
    fs           : sampling rate in Hz
    threshold_db : dB above surrounding band to count as a spike
    target_hz    : frequency to inspect (default: CFG.preprocessing.notch_hz)

    Returns
    -------
    bool
    """
    if target_hz is None:
        target_hz = CFG.preprocessing.notch_hz
        if target_hz is None:
            return False
    freqs, psd = sp_signal.welch(sig, fs=fs, nperseg=min(len(sig), int(fs * 4)))
    nyq = 0.5 * fs
    if target_hz >= nyq:
        return False
    target_idx = np.argmin(np.abs(freqs - target_hz))
    band_mask = ((freqs >= target_hz - 10) & (freqs < target_hz - 2)) | \
                ((freqs > target_hz + 2) & (freqs <= target_hz + 10))
    if not np.any(band_mask):
        return False
    surrounding_mean = np.mean(psd[band_mask])
    if surrounding_mean == 0:
        return False
    spike_db = 10 * np.log10(psd[target_idx] / surrounding_mean)
    return spike_db > threshold_db


def _butterworth_highpass(sig: np.ndarray, fs: float) -> np.ndarray:
    """4th-order Butterworth high-pass at CFG.preprocessing.highpass_cutoff_hz."""
    cutoff = CFG.preprocessing.highpass_cutoff_hz
    order = CFG.preprocessing.filter_order
    nyq = 0.5 * fs
    if cutoff >= nyq:
        import warnings
        warnings.warn(
            f"High-pass cutoff {cutoff} Hz >= Nyquist {nyq} Hz. "
            "Skipping high-pass filter."
        )
        return sig.copy()
    b, a = sp_signal.butter(order, cutoff / nyq, btype="high", analog=False)
    return sp_signal.filtfilt(b, a, sig)


def _polynomial_detrend(sig: np.ndarray, fs: float) -> np.ndarray:
    """Fit and subtract a low-order polynomial to remove drift."""
    order = CFG.preprocessing.polynomial_detrend_order
    t = np.arange(len(sig))
    coeffs = np.polyfit(t, sig, order)
    trend = np.polyval(coeffs, t)
    return sig - trend


def _moving_median_detrend(sig: np.ndarray, fs: float, window_sec: float = None,
                           causal: bool = None) -> np.ndarray:
    """
    Subtract a slow moving-median baseline (CFG.preprocessing.median_baseline_sec).

    Unlike a high-pass filter this keeps the DC level of each fixation, which is
    what encodes absolute gaze angle, while removing slow electrode drift; the
    median is insensitive to blinks and saccade steps. The baseline is estimated
    on the signal decimated to ~8 Hz for speed. With median_baseline_causal=True
    each sample's baseline uses only past samples (real-time compatible).
    """
    if window_sec is None:
        window_sec = CFG.preprocessing.median_baseline_sec
    if causal is None:
        causal = CFG.preprocessing.median_baseline_causal
    step = max(1, int(fs // 8))
    decimated = sig[::step]
    w = max(1, int(round(window_sec * fs / step)))
    if causal:
        import pandas as pd
        base = pd.Series(decimated).rolling(w, min_periods=1).median().to_numpy()
        baseline = np.repeat(base, step)[: len(sig)]
    else:
        base = median_filter(decimated, size=w, mode="nearest")
        baseline = np.interp(np.arange(len(sig)), np.arange(len(decimated)) * step, base)
    return sig - baseline


def _robust_baseline_detrend(sig: np.ndarray, fs: float, fit_line: bool,
                             window_sec: float = None, causal: bool = None) -> np.ndarray:
    """
    Subtract a moving robust-mean (or robust-line) baseline.

    Gaze targets are spread roughly uniformly over the screen, and for such a
    spread the moving median wanders ~sqrt(3)x more than the moving mean. Samples
    further than baseline_clip_mad robust SDs from the window median are dropped
    before averaging, so blinks don't pull the mean (clipping them instead would
    still let each blink add a full baseline_clip_mad SDs). With fit_line=True a
    line is fit to the kept samples and evaluated at the samples being corrected;
    for a past-only (causal) window this removes the half-window lag of the mean.

    The signal is block-averaged to ~8 Hz first. In causal mode the baseline for
    each block uses only earlier blocks, so no sample ever sees the future.
    """
    cfg = CFG.preprocessing
    if window_sec is None:
        window_sec = cfg.baseline_window_sec
    if causal is None:
        causal = cfg.baseline_causal
    step = max(1, int(fs // 8))
    n_blocks = len(sig) // step
    if n_blocks < 2:
        return sig - np.mean(sig)
    blocks = sig[: n_blocks * step].reshape(n_blocks, step).mean(axis=1)
    w = max(2, int(round(window_sec * fs / step)))

    if causal:
        padded = np.r_[np.full(w - 1, blocks[0]), blocks]
    else:
        half = w // 2
        padded = np.r_[np.full(half, blocks[0]), blocks, np.full(w - 1 - half, blocks[-1])]
    windows = sliding_window_view(padded, w)
    med = np.median(windows, axis=1, keepdims=True)
    dev = np.abs(windows - med)
    robust_sd = 1.4826 * np.median(dev, axis=1, keepdims=True) + 1e-12
    # Floored at 0.68 robust SDs (= 1 MAD) so at least half of every window is kept.
    keep = dev <= max(cfg.baseline_clip_mad, 0.68) * robust_sd
    n_keep = keep.sum(axis=1)
    base = (windows * keep).sum(axis=1) / n_keep

    if causal:
        if fit_line:
            t = np.arange(w) - (w - 1) / 2
            t_mean = (keep * t).sum(axis=1) / n_keep
            dt = t - t_mean[:, None]
            slope = ((keep * dt * (windows - base[:, None])).sum(axis=1)
                     / ((keep * dt ** 2).sum(axis=1) + 1e-12))
            base = base + slope * ((w - 1) / 2 + 1 - t_mean)  # evaluated at the next block
        # block j's baseline corrects block j+1; the first block uses sig[0]
        return sig - np.repeat(np.r_[sig[0], base], step)[: len(sig)]

    centres = np.arange(n_blocks) * step + (step - 1) / 2
    return sig - np.interp(np.arange(len(sig)), centres, base)


def filter_trial_channels(
    channels: dict,
    fs: float,
    apply_drift: bool = True,
    apply_noise: bool = True,
    auto_notch: bool = True,
) -> dict:
    """
    Apply the full filter chain to all channels in a dict.

    Parameters
    ----------
    channels    : dict of channel_name → np.ndarray
    fs          : sampling rate in Hz
    apply_drift : whether to apply baseline drift removal
    apply_noise : whether to apply low-pass noise filter
    auto_notch  : if True, automatically apply notch only if spike detected

    Returns
    -------
    New dict with filtered arrays (originals not mutated).
    """
    filtered = {}
    for name, arr in channels.items():
        ch = arr.copy().astype(np.float64)

        if apply_drift:
            ch = remove_baseline_drift(ch, fs)

        if apply_noise:
            ch = remove_noise(ch, fs)

        if auto_notch and has_mains_spike(ch, fs):
            ch = apply_notch(ch, fs)

        filtered[name] = ch

    return filtered
