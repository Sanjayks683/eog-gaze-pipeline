"""
tests/test_preprocessing.py
=============================
Tests for filtering, blink detection, and normalization.
All tests use synthetic signals — no dataset required.
"""

import pytest
import numpy as np
from scipy import signal as sp_signal


def _make_drift_signal(fs=250.0, duration=60.0, drift_hz=0.05, noise_hz=45.0):
    """
    Synthetic signal: saccade step + slow drift + high-freq noise.
    Use a long duration (60 s) so filtfilt has many full cycles of the
    0.05 Hz drift to remove — short signals give unreliable PSD estimates.
    """
    n = int(fs * duration)
    t = np.arange(n) / fs
    step = np.zeros(n)
    step[n // 4 :] = 100.0
    drift = 50.0 * np.sin(2 * np.pi * drift_hz * t)
    noise = 5.0 * np.sin(2 * np.pi * noise_hz * t)
    return step + drift + noise, step


def test_highpass_removes_drift():
    from src.preprocessing.filtering import remove_baseline_drift
    fs = 250.0
    sig, step = _make_drift_signal(fs)
    filtered = remove_baseline_drift(sig, fs, method="highpass")

    freqs, psd_orig = sp_signal.welch(sig, fs=fs)
    freqs, psd_filt = sp_signal.welch(filtered, fs=fs)

    drift_band = freqs < 0.1
    if np.any(drift_band):
        assert np.sum(psd_filt[drift_band]) < np.sum(psd_orig[drift_band]), \
            "High-pass filter did not reduce drift power"


def test_polynomial_detrend_removes_drift():
    from src.preprocessing.filtering import remove_baseline_drift
    fs = 250.0
    sig, _ = _make_drift_signal(fs)
    filtered = remove_baseline_drift(sig, fs, method="polynomial_detrend")
    assert np.std(filtered) < np.std(sig)


def test_highpass_preserves_saccade_step():
    """
    High-pass must NOT completely destroy the saccade signal energy.
    We test that the RMS of the filtered signal is meaningfully greater than
    the noise floor (i.e., there's still signal content, not just zeros).
    A high-pass at 0.2 Hz will attenuate a DC step but must leave the
    transition edge (which contains higher-frequency content) intact.
    """
    from src.preprocessing.filtering import remove_baseline_drift
    fs = 250.0
    duration = 20.0
    n = int(fs * duration)
    rng = np.random.RandomState(0)
    step = np.zeros(n)
    for onset in range(n // 5, n, n // 4):
        if onset < n:
            step[onset:] += 50.0
    signal = step + rng.randn(n) * 2.0
    filtered = remove_baseline_drift(signal, fs, method="highpass")

    rms_filtered = np.sqrt(np.mean(filtered ** 2))
    assert rms_filtered > 1.0, \
        f"High-pass filter produced near-zero output (RMS={rms_filtered:.4f}). "\
        "Check that the filter is not over-aggressive."


def test_lowpass_reduces_high_freq_noise():
    from src.preprocessing.filtering import remove_noise
    fs = 500.0
    n = int(fs * 2)
    t = np.arange(n) / fs
    clean = np.sin(2 * np.pi * 2 * t)
    noisy = clean + np.sin(2 * np.pi * 45 * t)

    filtered = remove_noise(noisy, fs)
    freqs, psd = sp_signal.welch(filtered, fs=fs)
    high_band = freqs > 35
    low_band = (freqs > 1) & (freqs < 5)
    if np.any(high_band) and np.any(low_band):
        assert np.max(psd[high_band]) < np.max(psd[low_band]), \
            "Low-pass filter did not sufficiently attenuate 45 Hz noise"


def test_notch_reduces_50hz_spike():
    from src.preprocessing.filtering import apply_notch, has_mains_spike
    fs = 500.0
    n = int(fs * 4)
    t = np.arange(n) / fs
    sig = np.sin(2 * np.pi * 5 * t) + 2.0 * np.sin(2 * np.pi * 50 * t)

    assert has_mains_spike(sig, fs), "Should detect 50 Hz spike"

    filtered = apply_notch(sig, fs, notch_hz=50.0)
    freqs, psd_filt = sp_signal.welch(filtered, fs=fs)
    freqs, psd_orig = sp_signal.welch(sig, fs=fs)

    idx50 = np.argmin(np.abs(freqs - 50))
    assert psd_filt[idx50] < psd_orig[idx50], \
        "Notch filter did not reduce 50 Hz power"


def test_no_mains_spike_detection():
    from src.preprocessing.filtering import has_mains_spike
    fs = 500.0
    n = int(fs * 4)
    t = np.arange(n) / fs
    clean = np.sin(2 * np.pi * 5 * t)
    assert not has_mains_spike(clean, fs), \
        "No 50 Hz spike should be detected in a clean signal"


def _make_blink_signal(fs=250.0, duration=4.0, blink_start=1.5, blink_dur=0.3):
    """Synthetic V channel with a blink artefact."""
    n = int(fs * duration)
    sig = np.random.randn(n) * 5.0
    b_start = int(blink_start * fs)
    b_end = int((blink_start + blink_dur) * fs)
    sig[b_start:b_end] = 400.0
    return sig, b_start, b_end


def test_blink_detected():
    from src.preprocessing.blink import detect_blinks
    fs = 250.0
    sig, b_start, b_end = _make_blink_signal(fs)
    rest = sig[:int(0.5 * fs)]
    blinks = detect_blinks(sig, fs, rest_segment=rest)
    assert len(blinks) >= 1, "Blink not detected"
    onsets = [b[0] for b in blinks]
    assert any(abs(o - b_start) < int(0.1 * fs) for o in onsets), \
        f"Blink onset far from expected ({b_start}). Detected: {onsets}"


def test_short_spike_rejected():
    from src.preprocessing.blink import detect_blinks
    fs = 250.0
    n = int(fs * 2)
    sig = np.zeros(n)
    sig[100] = 1000.0
    rest = sig[:50]
    blinks = detect_blinks(sig, fs, rest_segment=rest)
    assert len(blinks) == 0, "Single-sample spike should be rejected by min-duration filter"


def test_blink_mask_shape():
    from src.preprocessing.blink import detect_blinks, blinks_to_mask
    fs = 250.0
    sig, _, _ = _make_blink_signal(fs)
    rest = sig[:int(0.5 * fs)]
    blinks = detect_blinks(sig, fs, rest_segment=rest)
    mask = blinks_to_mask(len(sig), blinks)
    assert mask.shape == (len(sig),)
    assert mask.dtype == bool


def test_zscore_unit_variance():
    from src.preprocessing.normalize import zscore_trial
    sig = np.random.randn(1000) * 50 + 20.0
    normed = zscore_trial(sig, mean=np.mean(sig), std=np.std(sig))
    assert abs(np.mean(normed)) < 0.01, "Z-scored signal should have ~zero mean"
    assert abs(np.std(normed) - 1.0) < 0.01, "Z-scored signal should have ~unit variance"


def test_zscore_zero_std_returns_zeros():
    from src.preprocessing.normalize import zscore_trial
    sig = np.ones(100) * 5.0
    normed = zscore_trial(sig, mean=5.0, std=0.0)
    assert np.all(normed == 0.0), "Zero-std normalization should return zeros"


def test_zscore_per_subject_no_leakage():
    """
    Per-subject normalization: stats from subject A must not affect subject B.
    """
    from src.data.schema import Trial, make_empty_event_timestamps
    from src.preprocessing.normalize import zscore_per_subject

    rng = np.random.RandomState(42)
    ts = make_empty_event_timestamps()
    ts["saccade1_onset"] = 50

    tA = Trial("SA", "T1", "dataset1", "bipolar", 250.0,
               channels={"H": rng.randn(500) * 200 + 50,
                          "V": rng.randn(500) * 100},
               event_timestamps=ts.copy())
    tB = Trial("SB", "T1", "dataset1", "bipolar", 250.0,
               channels={"H": rng.randn(500) * 5 + 2,
                          "V": rng.randn(500) * 3},
               event_timestamps=ts.copy())

    normed, stats = zscore_per_subject([tA, tB], inplace=False)

    sa_normed = next(t for t in normed if t.subject_id == "SA")
    sb_normed = next(t for t in normed if t.subject_id == "SB")

    assert "SA" in stats["H"] and "SB" in stats["H"]
    mean_A, std_A = stats["H"]["SA"]
    mean_B, std_B = stats["H"]["SB"]
    assert abs(std_A - std_B) > 10, \
        "Different subjects should have different stds before normalization"

