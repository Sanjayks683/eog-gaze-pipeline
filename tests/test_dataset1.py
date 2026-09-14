"""
tests/test_dataset1.py
======================
Dataset 1 stores one target row per cue; the loader spreads them over the samples.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.loaders import expand_cue_targets
from src.data.schema import Trial


def _trial(cs, target):
    n = len(cs)
    return Trial(subject_id="S1", trial_id="S1_trial", dataset_source="dataset1", montage_type="bipolar",
                 fs=256.0, channels={"H": np.zeros(n), "V": np.zeros(n), "ControlSignal": np.array(cs)},
                 target_angle=np.array(target, dtype=float))


def test_cue_targets_hold_from_each_cue_until_the_next():
    cs = [1, 1, 2, 2, 3, 3, 1, 1, 2, 3, 30]
    trial = _trial(cs, [[5, 1], [0, 0], [-3, 2], [0, 0]])
    expand_cue_targets(trial)

    expected = [[5, 1]] * 2 + [[0, 0]] * 4 + [[-3, 2]] * 2 + [[0, 0]] * 3
    assert trial.target_angle.tolist() == expected


def test_per_sample_targets_are_left_alone_and_mismatches_raise():
    trial = _trial([1, 2, 3], [[1, 1], [0, 0], [0, 0]])
    expand_cue_targets(trial)
    assert trial.target_angle.tolist() == [[1, 1], [0, 0], [0, 0]]

    with pytest.raises(ValueError, match="cues"):
        expand_cue_targets(_trial([1, 1, 2, 2, 3, 3, 1, 1], [[5, 1], [0, 0]]))
