"""
tests/test_multi_dataset.py
===========================
Behaviour Datasets 3 and 4 need: fixation windows from the ControlSignal when the
target moves with the head (Dataset 3), head pose kept apart from head position in
the loader, and published numbers listed only for the dataset they belong to.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import CFG, _CONFIG_SECTIONS, stride_samples, window_samples
from src.data.datasets import _fixation_flags
from src.data.loaders import _parse_mat_trial
from src.data.schema import Trial

FS = 256.0


@pytest.fixture()
def restore_cfg():
    snapshot = {section: dict(vars(getattr(CFG, section))) for section in _CONFIG_SECTIONS}
    yield CFG
    for section, values in snapshot.items():
        sub = getattr(CFG, section)
        for k, v in values.items():
            setattr(sub, k, v)


def _cued(n_trials: int = 20, seed: int = 0):
    rng = np.random.RandomState(seed)
    sec = int(FS)
    cs, target = [], []
    for _ in range(n_trials):
        p1, p2 = rng.uniform(-20, 20, 2), rng.uniform(-20, 20, 2)
        cs += [1] * sec + [2] * sec + [3] * 2 * sec
        target += [p1] * sec + [p2] * 3 * sec
    return np.array(cs), np.array(target)


def _trial(cs: np.ndarray, target: np.ndarray) -> Trial:
    n = len(cs)
    return Trial(subject_id="S1", trial_id="S1_trial", dataset_source="dataset3", montage_type="monopolar",
                 fs=FS, channels={"H": np.zeros(n), "V": np.zeros(n), "ControlSignal": cs}, target_angle=target)


def test_fixation_windows_follow_the_cues_when_the_target_moves_with_the_head(restore_cfg):
    cs, target = _cued()
    t = np.arange(len(cs)) / FS
    head = np.column_stack([0.8 * np.sin(2 * np.pi * t / 7.0), 0.4 * np.cos(2 * np.pi * t / 5.0)])
    win, stride = window_samples(FS), stride_samples(FS)
    starts = np.arange(0, len(cs) - win + 1, stride)

    still = _fixation_flags(_trial(cs, target), starts, win)
    moving = _fixation_flags(_trial(cs, target + head), starts, win)

    assert still.mean() > 0.1
    assert np.array_equal(still, moving)
    settle = int(round(CFG.segmentation.fixation_settle_ms * FS / 1000.0))
    onsets = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    for s in starts[moving]:
        onset = onsets[onsets <= s][-1]
        assert cs[s] in (1, 2) and s - onset >= settle
        assert not np.any((onsets > s) & (onsets < s + win))


@pytest.mark.parametrize("order", [("Head_Pose", "Head_Position"), ("Head_Position", "Head_Pose")])
def test_loader_keeps_head_pose_and_head_position_apart(order):
    n = 500
    rng = np.random.RandomState(0)
    arrays = {
        "EOG": rng.randn(4, n),
        "Target_GA_stream": rng.randn(2, n),
        "Head_Pose": rng.uniform(-10, 10, (3, n)),
        "Head_Position": rng.uniform(-0.03, 0.03, (3, n)),
    }
    mat = {k: arrays[k] for k in ("EOG", "Target_GA_stream", *order)}

    trial = _parse_mat_trial(mat, "S1", "S1_trial", "dataset3", "monopolar", FS, has_head_pose=True)

    assert np.array_equal(trial.head_pose, arrays["Head_Pose"])
    assert np.array_equal(trial.metadata["head_position"], arrays["Head_Position"])


def test_master_table_lists_published_numbers_only_for_the_loaded_dataset(tmp_path, restore_cfg):
    from src.evaluation.compare_to_paper import _detect_dataset_label, build_master_table

    published = lambda rows: [r for r in rows if r["method"].startswith("Published")]
    CFG.data.datasets_to_load = ["dataset4"]
    assert _detect_dataset_label() == "Dataset 4 only"
    assert published(build_master_table(results_dir=str(tmp_path))) == []

    CFG.data.datasets_to_load = ["dataset2"]
    assert _detect_dataset_label() == "Dataset 2 only"
    assert len(published(build_master_table(results_dir=str(tmp_path)))) == 4


def test_known_start_table_lists_published_numbers_only_for_dataset2():
    from src.evaluation.known_start import known_start_rows

    keys = ("mae_h_deg", "mae_v_deg", "mae_h_sd_deg", "mae_v_sd_deg",
            "excluded_mae_h_deg", "excluded_mae_v_deg", "excluded_fraction")
    fits = {k: {key: 1.0 for key in keys} for k in ("short_saccades", "short_level", "long_saccades")}
    base = {"same_subject": fits, "unseen_subject": fits}
    published = lambda rows: [r for r in rows if r["method"].startswith("Published")]

    assert len(published(known_start_rows({**base, "datasets": ["dataset2"]}))) == 4
    assert published(known_start_rows({**base, "datasets": ["dataset4"]})) == []
