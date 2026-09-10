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
from typing import Literal

from src.config import CFG


def remove_baseline_drift(
    sig: np.ndarray,
    fs: float,
    method: Literal["highpass", "polynomial_detrend"] = None,
) -> np.ndarray:
    """
    Remove low-frequency baseline drift from an EOG signal.

    Parameters
    ----------
    sig    : 1-D numpy array (n_samples,)
    fs     : sampling rate in Hz
    method : "highpass" or "polynomial_detrend" (default from CFG)

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
    else:
        raise ValueError(f"Unknown drift removal method: {method!r}. Use 'highpass' or 'polynomial_detrend'.")


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
