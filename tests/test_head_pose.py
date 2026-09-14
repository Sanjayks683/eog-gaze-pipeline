"""
tests/test_head_pose.py
=======================
Dataset 3 head pose: the head-pose context features for regression and the
head-rotation term of the known-start estimator.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import CFG, _CONFIG_SECTIONS
from src.data.schema import Trial
from src.evaluation.known_start import _head_rotation_between_movements, _vor_displacement
from src.features.context import _head_pose_features, context_feature_names

FS = 256.0


@pytest.fixture()
def restore_cfg():
    snapshot = {section: dict(vars(getattr(CFG, section))) for section in _CONFIG_SECTIONS}
    yield CFG
    for section, values in snapshot.items():
        sub = getattr(CFG, section)
        for k, v in values.items():
            setattr(sub, k, v)


def _trial(pose: np.ndarray) -> Trial:
    n = pose.shape[1]
    return Trial(subject_id="S1", trial_id="S1_trial", dataset_source="dataset3", montage_type="monopolar",
                 fs=FS, channels={"H": np.zeros(n), "V": np.zeros(n)}, head_pose=pose)


def test_head_pose_features_are_window_level_and_deviation_from_the_past(restore_cfg):
    CFG.preprocessing.baseline_window_sec = 10.0
    n, win = 60 * int(FS), 77
    rng = np.random.RandomState(0)
    pose = np.cumsum(rng.randn(3, n) * 0.05, axis=1)
    starts = np.arange(20 * int(FS), n - win, 500)

    feats = _head_pose_features(_trial(pose), starts, win, causal=True)
    assert len(feats) == 6
    assert np.allclose(feats[0], [pose[0, s:s + win].mean() for s in starts])
    past = [pose[0, s + win - 10 * int(FS):s + win].mean() for s in starts]
    assert np.allclose(feats[1], feats[0] - past)

    changed = pose.copy()
    k = 40 * int(FS)
    changed[:, k:] += 20.0
    before = starts + win <= k
    again = _head_pose_features(_trial(changed), starts, win, causal=True)
    assert all(np.allclose(a[before], b[before]) for a, b in zip(feats, again))

    CFG.preprocessing.context_head_pose = True
    assert context_feature_names(["H", "V"])[-6:] == [
        "head_yaw_level", "head_yaw_minus_10s_mean", "head_pitch_level", "head_pitch_minus_10s_mean",
        "head_roll_level", "head_roll_minus_10s_mean"]
    with pytest.raises(ValueError, match="no head pose"):
        _head_pose_features(Trial("S1", "t", "dataset2", "monopolar", FS, {"H": np.zeros(10)}), starts, win, True)


def test_head_rotation_counts_only_between_detected_movements(restore_cfg):
    CFG.known_start.after_gap_ms, CFG.known_start.level_window_ms = 0.0, 0.0
    n = 20 * int(FS)
    yaw = np.linspace(0.0, 10.0, n)  # steady turn, 0.5 deg/s
    pose = np.vstack([yaw, -yaw, np.zeros(n)])
    events = {"onset": np.array([5 * int(FS)]), "offset": np.array([9 * int(FS)])}

    cum = _head_rotation_between_movements(_trial(pose), events, n, FS)
    masked = 4 * int(FS) * 10.0 / (n - 1)
    assert cum.shape == (2, n)
    assert cum[0, -1] == pytest.approx(10.0 - masked, abs=0.05)
    assert cum[1, -1] == pytest.approx(-(10.0 - masked), abs=0.05)

    d = {"vor": cum}
    rate = 10.0 * FS / (n - 1)  # deg/s
    shift = _vor_displacement(d, int(FS), np.array([4 * int(FS), 15 * int(FS)]))
    assert shift[0, 0] == pytest.approx(-3 * rate, abs=0.02)  # minus the head's 3 s turn
    assert shift[1, 1] == pytest.approx((14 - 4) * rate, abs=0.05)  # pitch = -yaw; 4 s of it inside the movement

    no_head = Trial("S1", "t", "dataset2", "monopolar", FS, {"H": np.zeros(n), "V": np.zeros(n)})
    assert _head_rotation_between_movements(no_head, events, n, FS) is None
