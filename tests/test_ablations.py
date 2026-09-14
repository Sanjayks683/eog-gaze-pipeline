"""
tests/test_ablations.py
=======================
Direction-invariance ablation bucketing (Phase 10).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ablations.direction_invariance import bucket_by_direction


def test_centre_targets_get_their_own_bucket_instead_of_direction_zero():
    true = np.array([[0.0, 0.0], [0.5, -0.5], [10.0, 0.0], [0.0, 10.0], [-10.0, 0.0], [0.0, -12.0]])
    pred = true + np.array([1.0, 1.0])
    results = bucket_by_direction(true, pred, [{}] * len(true))

    assert results["centre (< 3°)"]["n_windows"] == 2
    assert [results[k]["n_windows"] for k in ("0-45°", "90-135°", "180-225°", "270-315°")] == [1, 1, 1, 1]
    assert sum(r["n_windows"] for r in results.values()) == len(true)
    assert results["0-45°"]["rmse_total_deg"] == pytest.approx(np.sqrt(2.0))
    assert results["45-90°"] == {"rmse_h_deg": None, "rmse_v_deg": None, "rmse_total_deg": None, "n_windows": 0}
