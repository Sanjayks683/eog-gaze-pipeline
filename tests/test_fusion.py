"""
tests/test_fusion.py
====================
The fused known-start estimator: a complementary filter between the detected-saccade
estimate and a cross-subject absolute estimate.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.known_start import FUSION_TAUS_SEC, _choose_taus, _fused_predictions

FS = 256.0


def _recording(n: int, absolute: np.ndarray, target: np.ndarray) -> dict:
    """No detected movements, so the saccade estimate stays at the starting gaze."""
    return {"fs": FS, "hv": np.zeros((2, n)), "target": target, "absolute": absolute, "vor": None,
            "mv_onset": np.zeros(0, dtype=int), "mv_known": np.zeros(0, dtype=int),
            "mv_before": np.zeros((0, 2)), "mv_delta": np.zeros((0, 2))}


def test_fusion_moves_from_the_known_start_toward_the_absolute_estimate():
    n = 30 * int(FS)
    target = np.tile([4.0, -2.0], (n, 1))
    d = _recording(n, absolute=target.copy(), target=target)
    windows = [{"start": 10, "gaze0": np.zeros(2), "scored": np.arange(10, n)}]
    A = np.eye(2)

    pred, tgt = _fused_predictions(d, windows, A, np.ones(2), (np.inf, 1.0, 1000.0))
    assert np.allclose(pred[0], 0.0)  # tau = inf: the saccade estimate alone
    assert np.allclose(pred[1, int(10 * FS)], [4.0, -2.0], atol=1e-3)  # fast filter reaches the absolute level
    assert np.all(np.abs(pred[2, -1]) < np.abs(pred[1, -1]))  # slow filter lags behind
    # every variant starts at the known gaze (the first filter step moves it by alpha x difference)
    assert np.allclose(pred[:, 0], 0.0, atol=(1 - np.exp(-1 / FS)) * 4.0 + 1e-9)

    d["absolute"][: int(20 * FS)] = np.nan  # no absolute estimate yet: no correction
    pred, _ = _fused_predictions(d, windows, A, np.ones(2), (1.0,))
    assert np.allclose(pred[0, : int(19 * FS)], 0.0)

    # an absolute estimate that is right beats the drifting saccade estimate, so a short tau wins
    d["absolute"] = target.copy()
    taus = _choose_taus([(d, windows, np.ones(2))], A)
    assert taus.tolist() == [FUSION_TAUS_SEC[0], FUSION_TAUS_SEC[0]]
    # a useless absolute estimate is never trusted
    d["absolute"] = target + 30.0
    d["target"] = np.zeros((n, 2))
    assert np.isinf(_choose_taus([(d, windows, np.ones(2))], A)).all()
