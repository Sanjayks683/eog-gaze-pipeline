"""
Unit tests for the zero-leakage few-shot calibration module.
"""

import numpy as np
import pytest

from src.evaluation.calibration import (
    fit_affine_calibration,
    apply_affine_calibration,
    evaluate_subject_calibration,
)


def test_fit_affine_calibration_identity():
    """Verify that identical inputs yield identity affine parameters."""
    rng = np.random.RandomState(42)
    y_true = rng.randn(100, 2).astype(np.float32)
    y_pred = y_true.copy()

    gains, offsets = fit_affine_calibration(y_pred, y_true)
    assert np.allclose(gains, [1.0, 1.0], atol=1e-2)
    assert np.allclose(offsets, [0.0, 0.0], atol=1e-2)


def test_fit_affine_calibration_scaling_and_shift():
    """Verify recovery of known gain and offset."""
    rng = np.random.RandomState(42)
    y_pred = rng.randn(100, 2).astype(np.float32) * 5.0
    true_gains = np.array([2.5, 0.8], dtype=np.float32)
    true_offsets = np.array([3.0, -2.0], dtype=np.float32)
    y_true = y_pred * true_gains + true_offsets

    gains, offsets = fit_affine_calibration(y_pred, y_true)
    assert np.allclose(gains, true_gains, atol=1e-2)
    assert np.allclose(offsets, true_offsets, atol=1e-2)


def test_fit_affine_calibration_does_not_overstretch_noisy_predictions():
    """With noisy predictions the MSE-optimal gain is var_t / (var_t + var_noise)
    = 0.5 here; variance matching would give sqrt(2) and inflate test error."""
    rng = np.random.RandomState(0)
    y_true = rng.randn(5000, 2) * 10.0
    y_pred = y_true + rng.randn(5000, 2) * 10.0

    gains, _ = fit_affine_calibration(y_pred, y_true)
    assert np.allclose(gains, [0.5, 0.5], atol=0.05)


def test_evaluate_subject_calibration_zero_leakage():
    """Verify zero-leakage: evaluation windows are strictly distinct from calibration windows."""
    rng = np.random.RandomState(42)
    n_samples = 200
    y_true = rng.randn(n_samples, 2).astype(np.float32) * 10.0
    y_pred = (y_true - 4.0) / 1.8

    metadata = [
        {"subject_id": "S1", "window_start": i * 37, "fs": 250.0}
        for i in range(n_samples)
    ]

    res = evaluate_subject_calibration(
        y_true=y_true,
        y_pred=y_pred,
        metadata=metadata,
        prompt_sec=5.0,
        min_cal_windows=15,
    )

    sub_res = res["per_subject"]["S1"]
    n_cal = sub_res["n_calibration_windows"]
    n_eval = sub_res["n_evaluation_windows"]

    assert n_cal + n_eval == n_samples
    assert n_cal >= 15
    assert n_eval > 0

    assert sub_res["cal_rmse_h"] < sub_res["uncal_rmse_h"] * 0.5
    assert res["calibrated_rmse_h_deg"] < res["uncalibrated_rmse_h_deg"] * 0.5
